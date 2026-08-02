"""ONLINE ball tracking + trajectory in one pass (fixes 'trajectory loses the
tiny ball').

Instead of detecting every frame independently and fitting afterwards, this runs
detection and a Kalman predictor TOGETHER, frame by frame:

  * acquire the ball with SAHI tiled detection (sees the tiny ball),
  * seed velocity from the first hits,
  * each frame: predict -> search a small ROI (crop = zoom, so the tiny ball is
    detectable) with the tiled detector -> correct the state,
  * when a frame misses (ball too small / blurred), the filter PREDICTS the
    position and keeps going instead of dropping the point.

The result is a continuous trajectory (detections + physics predictions), which
is then drawn as the Hawk-Eye tube.

    python scripts/track_and_trajectory.py --video <clip> [--out output/trajectories]
"""
import argparse, os
import cv2
import numpy as np

from pipeline.ingest import ingest_video
from pipeline.detection import YOLOBallDetector
from pipeline.tracking import PredictiveBallTracker
import scripts.hawkeye_tube as ht


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="models/ball_detector_tiled.pt")
    ap.add_argument("--out", default="output/trajectories")
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--roi-imgsz", type=int, default=640)
    ap.add_argument("--slice", type=int, default=384)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    video = ingest_video(args.video, sample_rate=1)
    frames = list(video.frames)
    W, H, fps = video.metadata.width, video.metadata.height, video.metadata.fps

    # tiled detector for ROI search + SAHI (same weights) for acquisition
    roi_yolo = YOLOBallDetector(args.weights, conf_threshold=args.conf, imgsz=args.roi_imgsz)
    sahi_model = ht.load_sahi_model(args.weights, args.conf)
    tracker = PredictiveBallTracker(
        yolo=roi_yolo, sahi_model=sahi_model, sahi_slice=args.slice,
        accept_conf=args.conf, max_misses=args.max_misses, base_roi=110.0)

    dets = tracker.detect_sequence(frames)
    pts = [(i, d.center[0], d.center[1]) for i, d in enumerate(dets) if d is not None]
    n_det = sum(1 for d in dets if d is not None and d.source != "predicted")
    n_pred = sum(1 for d in dets if d is not None and d.source == "predicted")
    print(f"{stem}: {len(pts)} tracked points ({n_det} detected + {n_pred} predicted)")
    if len(pts) < 6:
        print(f"{stem}: too few tracked points — skipping"); return

    grid, xs, ys = ht.build_centerline(pts)
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].image.copy()
    ht.draw_tube(still, xs, ys, radius=args.radius, alpha=args.alpha)
    cv2.imwrite(os.path.join(args.out, f"{stem}_hawkeye.jpg"), still)

    vw = cv2.VideoWriter(os.path.join(args.out, f"{stem}_hawkeye.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.image.copy()
        upto = int(np.searchsorted(grid, i)) if i >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=args.radius, alpha=args.alpha)
        vw.write(img)
    for _ in range(int(fps * 1.0)):
        img = frames[-1].image.copy(); ht.draw_tube(img, xs, ys, radius=args.radius, alpha=args.alpha); vw.write(img)
    vw.release()
    print("wrote", os.path.join(args.out, f"{stem}_hawkeye.mp4"))


if __name__ == "__main__":
    main()
