"""Honest generalization test: SAHI recall on held-out clips (never trained on),
OLD detector vs NEW (v2). Recall = fraction of ground-truth ball frames where a
detection lands within 3% of the frame diagonal of the labeled centre.
"""
import glob, os, sys
import numpy as np
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
sys.path.insert(0, os.path.abspath("."))
import scripts.gt_trajectories as G

HOLDOUT = ["bowling3", "Test13", "bowling_5", "bowling_18", "bowling_47", "bowling_99"]
MODELS = {"OLD": "models/ball_detector_tiled.pt", "NEW_v2": "models/ball_detector_tiled_v2.pt"}
CONF, SLICE = 0.25, 384


def recall_for(model, stem, frames, gt, W, H):
    tol = 0.03 * np.hypot(W, H)
    fs = sorted(gt)
    sample = fs[::2]  # every 2nd labeled frame
    hit = tot = det = 0
    for fi in sample:
        if fi >= len(frames):
            continue
        res = get_sliced_prediction(frames[fi][:, :, ::-1], model, slice_height=SLICE, slice_width=SLICE,
                                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                    perform_standard_pred=False, verbose=0)
        ds = [((o.bbox.minx + o.bbox.maxx) / 2, (o.bbox.miny + o.bbox.maxy) / 2) for o in res.object_prediction_list]
        tot += 1; det += len(ds)
        gx, gy = gt[fi]
        if any(np.hypot(x - gx, y - gy) < tol for x, y in ds):
            hit += 1
    return hit, tot, det


def main():
    # preload frames + GT per clip
    data = {}
    for stem in HOLDOUT:
        vp = G.find_video(stem)
        cap = cv2.VideoCapture(vp); W = int(cap.get(3)); H = int(cap.get(4)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        data[stem] = (frames, G.parse(stem), W, H)

    print(f"{'clip':<12} {'OLD recall':>14} {'NEW_v2 recall':>16}")
    agg = {"OLD": [0, 0], "NEW_v2": [0, 0]}
    per = {stem: {} for stem in HOLDOUT}
    for name, path in MODELS.items():
        m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=path,
                                               confidence_threshold=CONF, device="cuda:0")
        for stem in HOLDOUT:
            frames, gt, W, H = data[stem]
            hit, tot, det = recall_for(m, stem, frames, gt, W, H)
            per[stem][name] = (hit, tot, det)
            agg[name][0] += hit; agg[name][1] += tot

    for stem in HOLDOUT:
        o = per[stem]["OLD"]; n = per[stem]["NEW_v2"]
        print(f"{stem:<12} {o[0]}/{o[1]}={o[0]/max(o[1],1)*100:>3.0f}%      {n[0]}/{n[1]}={n[0]/max(n[1],1)*100:>3.0f}%   (FP old {o[2]-o[0]}, new {n[2]-n[0]})")
    print("-" * 44)
    op = agg["OLD"][0] / max(agg["OLD"][1], 1) * 100
    np_ = agg["NEW_v2"][0] / max(agg["NEW_v2"][1], 1) * 100
    print(f"{'OVERALL':<12} {agg['OLD'][0]}/{agg['OLD'][1]}={op:.0f}%      {agg['NEW_v2'][0]}/{agg['NEW_v2'][1]}={np_:.0f}%")


if __name__ == "__main__":
    main()
