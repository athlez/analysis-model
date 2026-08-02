"""Critical A/B evaluation of two ideas on 5 held-out (unseen-by-training,
labelled-so-scorable) clips:

  EXP1 Temporal Motion Trail : Kalman history + predict-through-gaps
       (PredictiveBallTracker) vs plain per-frame SAHI detection.
  EXP2 Biomechanics Release  : MediaPipe pose release vs the current
       detection-based release, both scored against the label-derived release.

Per clip metrics: release-frame error, detection recall, missed-recovered
(prediction correctly filled a detector miss), false recoveries, track
continuity, trajectory smoothness (mean |2nd-difference|, lower=smoother),
false positives, processing time.
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))
import scripts.gt_trajectories as G
import scripts.hawkeye_tube as ht
from pipeline.ingest import ingest_video
from pipeline.detection import YOLOBallDetector
from pipeline.tracking import PredictiveBallTracker
from pipeline.release_detector import ReleaseDetector

CLIPS = ["bowling3", "Test13", "bowling_5", "bowling_47", "bowling_99"]
WEIGHTS = "models/ball_detector_tiled.pt"   # v2
TOLF = 0.03
SLICE = 384


def sahi_centers(model, img):
    from sahi.predict import get_sliced_prediction
    r = get_sliced_prediction(img[:, :, ::-1], model, slice_height=SLICE, slice_width=SLICE,
                              overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                              perform_standard_pred=False, verbose=0)
    return [((o.bbox.minx + o.bbox.maxx) / 2, (o.bbox.miny + o.bbox.maxy) / 2) for o in r.object_prediction_list]


def smoothness(seq):
    """mean magnitude of the 2nd difference of a point sequence (px). lower=smoother."""
    p = np.array(seq, float)
    if len(p) < 3:
        return float("nan")
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.mean(np.hypot(d2[:, 0], d2[:, 1])))


def main():
    sahi = ht.load_sahi_model(WEIGHTS, 0.25)
    roi = YOLOBallDetector(WEIGHTS, conf_threshold=0.12, imgsz=640)
    rd = ReleaseDetector()
    rows = []

    for clip in CLIPS:
        boxes = G.parse(clip)
        fs = sorted(boxes)
        end_gt = G.trim_frozen_tail(fs, boxes)
        rel_gt = G.RELEASE_OVERRIDE.get(clip) or G.detect_release(fs, boxes, end_gt)
        flight = [f for f in fs if rel_gt <= f <= end_gt]
        video = ingest_video(G.find_video(clip), sample_rate=1)
        frames = list(video.frames)
        imgs = [f.image for f in frames]
        W, H = imgs[0].shape[1], imgs[0].shape[0]
        tol = TOLF * np.hypot(W, H)

        # ---- detection-only (per labelled flight frame) ----
        t0 = time.time()
        det_hit = {}; det_fp = 0; det_seq = []
        for f in flight:
            if f >= len(imgs):
                continue
            ds = sahi_centers(sahi, imgs[f])
            gx, gy = boxes[f]
            near = [d for d in ds if np.hypot(d[0] - gx, d[1] - gy) < tol]
            det_hit[f] = (min(near, key=lambda d: np.hypot(d[0] - gx, d[1] - gy)) if near else None)
            det_fp += len(ds) - len(near)
            if near:
                det_seq.append(det_hit[f])
        t_det = time.time() - t0
        gtN = len([f for f in flight if f < len(imgs)])
        recall_det = sum(v is not None for v in det_hit.values()) / max(gtN, 1)

        # ---- EXP1: temporal tracker ----
        t0 = time.time()
        tracker = PredictiveBallTracker(yolo=roi, sahi_model=sahi, sahi_slice=SLICE, accept_conf=0.12,
                                        max_misses=15, base_roi=110.0, gravity=0.0,
                                        acquire_budget=120, acquire_stride=2)
        out = tracker.detect_sequence(frames)
        t_trk = time.time() - t0
        m = tracker.metrics
        trk_hit = 0; recovered = 0; false_rec = 0; trk_fp = 0; trk_seq = []
        for f in flight:
            o = out[f] if f < len(out) else None
            on = o is not None and np.hypot(o.center[0] - boxes[f][0], o.center[1] - boxes[f][1]) < tol
            trk_hit += on
            if o is not None:
                trk_seq.append(o.center)
                if not on:
                    trk_fp += 1
                if det_hit.get(f) is None:           # detector missed this GT frame
                    if on:
                        recovered += 1                # prediction correctly filled the miss
                    else:
                        false_rec += 1                # prediction present but wrong
        recall_trk = trk_hit / max(gtN, 1)
        # continuity: longest run of non-None tracker output over the flight span
        span = [out[f] if f < len(out) else None for f in range(rel_gt, end_gt + 1)]
        best = cur = 0
        for o in span:
            cur = cur + 1 if o is not None else 0
            best = max(best, cur)
        continuity = best / max(len(span), 1)

        # ---- EXP2: release ----
        t0 = time.time()
        ev = rd.detect(frames)
        t_pose = time.time() - t0
        pose_rel = ev.frame_index if ev else None
        cur_rel = next((f for f in flight if det_hit.get(f) is not None), None)  # first reliable detection

        rows.append(dict(
            clip=clip, gtN=gtN, rel_gt=rel_gt,
            recall_det=recall_det, det_fp=det_fp, smooth_det=smoothness(det_seq), t_det=t_det,
            recall_trk=recall_trk, recovered=recovered, false_rec=false_rec, trk_fp=trk_fp,
            continuity=continuity, smooth_trk=smoothness(trk_seq), t_trk=t_trk,
            pose_rel=pose_rel, cur_rel=cur_rel, t_pose=t_pose,
        ))
        print(f"[done] {clip}", flush=True)

    # ---- report ----
    def rel_err(r, key):
        v = r[key]
        return "abstain" if v is None else f"{abs(v - r['rel_gt'])}"
    print("\n===== PER-CLIP =====")
    print(f"{'clip':<10}{'recallDet':>10}{'recallTrk':>10}{'recovered':>10}{'falseRec':>9}{'contin':>8}{'smoDet':>8}{'smoTrk':>8}{'FPdet':>7}{'FPtrk':>7}")
    for r in rows:
        print(f"{r['clip']:<10}{r['recall_det']*100:>9.0f}%{r['recall_trk']*100:>9.0f}%{r['recovered']:>10}{r['false_rec']:>9}"
              f"{r['continuity']*100:>7.0f}%{r['smooth_det']:>8.1f}{r['smooth_trk']:>8.1f}{r['det_fp']:>7}{r['trk_fp']:>7}")
    print("\n===== EXP2 RELEASE (frames off GT; lower=better) =====")
    print(f"{'clip':<10}{'GTrel':>7}{'current':>10}{'|err|':>7}{'pose':>10}{'|err|':>10}{'poseTime':>10}")
    for r in rows:
        cr = r['cur_rel']; pr = r['pose_rel']
        print(f"{r['clip']:<10}{r['rel_gt']:>7}{str(cr):>10}{('' if cr is None else abs(cr-r['rel_gt'])):>7}"
              f"{str(pr):>10}{(rel_err(r,'pose_rel')):>10}{r['t_pose']:>9.1f}s")
    print("\n===== TIME (s/clip) =====")
    for r in rows:
        print(f"  {r['clip']}: detect-only {r['t_det']:.0f}  tracker {r['t_trk']:.0f}  pose {r['t_pose']:.0f}")
    # aggregate
    import statistics as st
    print("\n===== AGGREGATE (mean over 5) =====")
    print(f"  recall  detect-only {np.mean([r['recall_det'] for r in rows])*100:.0f}%  "
          f"tracker {np.mean([r['recall_trk'] for r in rows])*100:.0f}%")
    print(f"  missed-recovered total {sum(r['recovered'] for r in rows)}  false-recoveries {sum(r['false_rec'] for r in rows)}")
    print(f"  continuity mean {np.mean([r['continuity'] for r in rows])*100:.0f}%")
    print(f"  smoothness detect {np.nanmean([r['smooth_det'] for r in rows]):.1f}  tracker {np.nanmean([r['smooth_trk'] for r in rows]):.1f}")
    print(f"  pose release: {sum(r['pose_rel'] is not None for r in rows)}/5 produced a release")


if __name__ == "__main__":
    main()
