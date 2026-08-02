"""Run the prediction-driven tracker on a clip and render the fused track.

    python scripts/track_predictive.py --video <clip> [--weights best.pt] [--no-pose]

Overlay per frame: the ball box coloured by observation source
(green=YOLO, orange=motion, grey=coasted prediction) + a fading trail. This is
a verification harness for pipeline.tracking.PredictiveBallTracker — the tracker
itself plugs into CricketPipeline(ball_detector=...) unchanged.
"""
import argparse, os
import cv2

from pipeline.ingest import ingest_video
from pipeline.tracking import build_predictive_tracker

SRC = {"yolo": (0, 220, 0), "motion": (0, 165, 255), "predicted": (170, 170, 170),
       "interpolated": (200, 120, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--out", default="output/demo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-pose", action="store_true", help="disable MediaPipe release detection")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    video = ingest_video(args.video, sample_rate=1)
    frames = list(video.frames)
    weights = args.weights if os.path.isfile(args.weights) else None
    tracker = build_predictive_tracker(weights=weights, use_pose=not args.no_pose, imgsz=args.imgsz)

    dets = tracker.detect_sequence(frames)
    counts = {}
    for d in dets:
        if d is not None:
            counts[d.source] = counts.get(d.source, 0) + 1
    n = sum(1 for d in dets if d is not None)
    print(f"{stem}: tracked {n}/{len(frames)} frames  {counts}")

    W, H = video.metadata.width, video.metadata.height
    vw = cv2.VideoWriter(os.path.join(args.out, f"{stem}_predictive.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), video.metadata.fps, (W, H))
    trail = []
    for f, d in zip(frames, dets):
        img = f.image.copy()
        if d is not None:
            trail.append((int(d.center[0]), int(d.center[1]), d.source))
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, a[:2], b[:2], SRC.get(b[2], (255, 255, 255)), 2, cv2.LINE_AA)
        if d is not None:
            col = SRC.get(d.source, (255, 255, 255))
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{d.source} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{stem} f{f.index}  predictive tracker (Kalman + ROI YOLO)",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print("wrote", os.path.join(args.out, f"{stem}_predictive.mp4"))


if __name__ == "__main__":
    main()
