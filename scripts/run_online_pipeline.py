"""ONLINE pipeline over EVERY video under data/ (recursive discovery).

For each video: ingest -> PredictiveBallTracker.detect_sequence (online state
machine: acquire -> lock -> predict-through-gaps -> re-acquire -> terminate) ->
render the trajectory tube from the continuous tracked path (detections +
predictions bridging gaps) -> collect honest metrics.

Reuses PredictiveBallTracker, KalmanBall2D, the SAHI/tiled detector, and the
existing tube renderer. Only the execution flow is online.

Writes output/online/<tag>_hawkeye.{mp4,jpg} for successful tracks and
output/online/_report.md + _metrics.json with honest aggregate metrics.
"""
import argparse, glob, json, os, time
import cv2
import numpy as np

from pipeline.ingest import ingest_video
from pipeline.detection import YOLOBallDetector
from pipeline.tracking import PredictiveBallTracker
import scripts.hawkeye_tube as ht

DATA = "data"
OUT = "output/online"
WEIGHTS = "models/ball_detector_tiled.pt"
MIN_DETECTED = 5          # need this many REAL detections (not predictions) to trust a track


def discover():
    vids = []
    for root, _, files in os.walk(DATA):
        if ".venv" in root:
            continue
        for fn in files:
            if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                vids.append(os.path.join(root, fn))
    return sorted(vids)


def tag_for(path):
    rel = os.path.relpath(path, DATA).replace(os.sep, "__")
    return os.path.splitext(rel)[0]


def render(frames, pts, W, H, fps, tag, radius=5, alpha=0.55):
    grid, xs, ys = ht.faithful_centerline(pts, win=3)
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].image.copy()
    ht.draw_tube(still, xs, ys, radius=radius, alpha=alpha)
    cv2.imwrite(os.path.join(OUT, f"{tag}_hawkeye.jpg"), still)
    vw = cv2.VideoWriter(os.path.join(OUT, f"{tag}_hawkeye.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.image.copy()
        upto = int(np.searchsorted(grid, i)) if i >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=radius, alpha=alpha)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--acquire-budget", type=int, default=120)
    ap.add_argument("--acquire-stride", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="process only first N (debug)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    roi_yolo = YOLOBallDetector(WEIGHTS, conf_threshold=args.conf, imgsz=640)
    sahi_model = ht.load_sahi_model(WEIGHTS, args.conf)

    vids = discover()
    if args.limit:
        vids = vids[:args.limit]
    rows = []
    t_start = time.time()
    for vi, path in enumerate(vids, 1):
        tag = tag_for(path)
        t0 = time.time()
        rec = {"video": os.path.relpath(path, DATA), "tag": tag}
        try:
            video = ingest_video(path, sample_rate=1)
            frames = list(video.frames)
            W, H, fps = video.metadata.width, video.metadata.height, video.metadata.fps
        except Exception as e:
            rec.update(status="error", reason=f"ingest: {str(e)[:50]}")
            rows.append(rec); print(f"[{vi}/{len(vids)}] {tag}: INGEST ERROR", flush=True); continue

        tracker = PredictiveBallTracker(
            yolo=roi_yolo, sahi_model=sahi_model, sahi_slice=384, accept_conf=args.conf,
            max_misses=15, base_roi=110.0,
            acquire_budget=args.acquire_budget, acquire_stride=args.acquire_stride)
        dets = tracker.detect_sequence(frames)
        m = tracker.metrics
        rec.update(m); rec["secs"] = round(time.time() - t0, 1)

        # trajectory = tracked path up to the LAST real detection (drop terminal coast)
        det_idx = [i for i, d in enumerate(dets)
                   if d is not None and d.source in ("yolo", "motion", "reacquired")]
        if not m.get("acquired"):
            rec.update(status="rejected", reason="never acquired ball")
        elif m["detected"] < MIN_DETECTED:
            rec.update(status="rejected", reason=f"too few detections ({m['detected']}) — mostly prediction")
        else:
            last_det = det_idx[-1]
            pts = [(i, dets[i].center[0], dets[i].center[1]) for i in range(len(dets))
                   if dets[i] is not None and i <= last_det]
            if len(pts) < 6:
                rec.update(status="rejected", reason="track too short")
            else:
                interior_pred = sum(1 for i in range(det_idx[0], last_det + 1)
                                    if dets[i] is not None and dets[i].source == "predicted")
                rec["bridged_gaps"] = interior_pred > 0
                render(frames, pts, W, H, fps, tag)
                rec.update(status="success")
        rows.append(rec)
        print(f"[{vi}/{len(vids)}] {tag}: {rec['status']} "
              f"det={m.get('detected',0)} pred={m.get('predicted',0)} rec={m.get('recoveries',0)} ({rec['secs']}s)",
              flush=True)

    # ---- aggregate metrics ---------------------------------------------- #
    ok = [r for r in rows if r.get("status") == "success"]
    rej = [r for r in rows if r.get("status") in ("rejected", "error")]

    def avg(key, subset=ok):
        vals = [r.get(key, 0) or 0 for r in subset]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    report = {
        "total_videos": len(rows),
        "successful_trajectories": len(ok),
        "failed_trajectories": len(rej),
        "avg_detected_points": avg("detected"),
        "avg_predicted_points": avg("predicted"),
        "avg_tracked_points": round(avg("detected") + avg("predicted"), 1),
        "avg_trajectory_length_frames": avg("tracked_span"),
        "tracker_recovery_count": sum(r.get("recoveries", 0) or 0 for r in rows),
        "videos_prediction_bridged_gaps": sum(1 for r in ok if r.get("bridged_gaps")),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
    }
    with open(os.path.join(OUT, "_metrics.json"), "w") as f:
        json.dump({"summary": report, "per_video": rows}, f, indent=2)

    with open(os.path.join(OUT, "_report.md"), "w") as f:
        f.write("# Online pipeline — results (honest metrics)\n\n")
        for k, v in report.items():
            f.write(f"- **{k.replace('_',' ')}**: {v}\n")
        f.write("\n## Rejected videos (reason)\n\n| Video | Reason | det | pred |\n|---|---|---:|---:|\n")
        for r in rej:
            f.write(f"| {r['video']} | {r.get('reason','')} | {r.get('detected',0)} | {r.get('predicted',0)} |\n")
        f.write("\n## Successful trajectories\n\n| Video | det | pred | recov | span | bridged |\n|---|---:|---:|---:|---:|:--:|\n")
        for r in ok:
            f.write(f"| {r['video']} | {r['detected']} | {r['predicted']} | {r.get('recoveries',0)} | "
                    f"{r.get('tracked_span',0)} | {'yes' if r.get('bridged_gaps') else 'no'} |\n")
    print("\n===== SUMMARY =====")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT}/_report.md + _metrics.json")


if __name__ == "__main__":
    main()
