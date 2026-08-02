"""Run the trained ball detector + ByteTrack tracker on a video and render an
overlay: bbox (green), confidence, track ID, ball center (red), and a trail of
previous centers. Also dumps per-frame detections to JSON for analysis.

Frozen downstream: this touches ONLY detection/tracking. Nothing about
calibration/projection/trajectory/physics is used here.

Usage:
  python scripts/detect_overlay.py --weights runs/ball/it0/weights/best.pt \
      --video data/raw/own_recordings/bowling2.mp4 --conf 0.15 --imgsz 960
"""
import argparse, json, os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="output/detect")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = YOLO(args.weights)
    out_path = os.path.join(args.out, f"{stem}_det.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trails = defaultdict(lambda: deque(maxlen=25))
    per_frame = []
    n_with_det = 0

    # stream=True yields one Results per frame; ByteTrack keeps IDs across frames
    results = model.track(source=args.video, tracker="bytetrack.yaml", persist=True,
                          conf=args.conf, imgsz=args.imgsz, device=args.device,
                          stream=True, verbose=False)
    for fi, r in enumerate(results):
        img = r.orig_img.copy()
        dets = []
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            n_with_det += 1
            for b in boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0])
                tid = int(b.id[0]) if b.id is not None else -1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dets.append({"bbox": [round(x1,1),round(y1,1),round(x2,1),round(y2,1)],
                             "conf": round(conf,3), "track_id": tid,
                             "center": [round(cx,1),round(cy,1)]})
                # draw
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,220,0), 2)
                cv2.circle(img, (int(cx),int(cy)), 4, (0,0,255), -1)
                cv2.putText(img, f"ball#{tid} {conf:.2f}", (int(x1), max(int(y1)-6,12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 2, cv2.LINE_AA)
                if tid >= 0:
                    trails[tid].append((int(cx),int(cy)))
        # draw trails
        for tid, pts in trails.items():
            for a, b in zip(pts, list(pts)[1:]):
                cv2.line(img, a, b, (0,165,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{stem}  f{fi}/{total-1}  dets={len(dets)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        vw.write(img)
        per_frame.append({"frame": fi, "n": len(dets), "dets": dets})

    vw.release()
    n_frames = len(per_frame)
    summary = {
        "video": stem, "weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
        "frames": n_frames, "frames_with_detection": n_with_det,
        "detection_rate": round(n_with_det / max(n_frames,1), 3),
        "unique_track_ids": len({d["track_id"] for f in per_frame for d in f["dets"]}),
    }
    with open(os.path.join(args.out, f"{stem}_det.json"), "w") as f:
        json.dump({"summary": summary, "frames": per_frame}, f, indent=1)
    print(json.dumps(summary, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
