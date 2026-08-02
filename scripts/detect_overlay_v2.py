"""Detector + tracker overlay with STATIC false-positive rejection (#4).

Insight from iteration 0: the detector fires low-confidence boxes on STATIC
background objects (distant stumps, net fixings) that recur at the SAME pixel
location every frame. The real ball MOVES fast. So:

  Pass 1: collect all detections across the whole clip.
  Static map: any pixel location where a detection recurs (within `radius`) in
              more than `static_frac` of frames is flagged BACKGROUND.
  Pass 2: suppress detections landing on static locations; render the rest
          (moving = ball candidates) with bbox, conf, track ID, center, trail.

This touches only detection/tracking. Nothing downstream is used.
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
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--radius", type=float, default=18.0, help="static clustering radius (px)")
    ap.add_argument("--static_frac", type=float, default=0.15,
                    help="fraction of frames a location must recur in to be background")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    model = YOLO(args.weights)

    # ---- Pass 1: collect raw detections ----
    raw = []  # list per frame of [ (cx,cy,x1,y1,x2,y2,conf) ]
    frames_imgs = []
    for r in model.predict(source=args.video, conf=args.conf, imgsz=args.imgsz,
                           device=args.device, stream=True, verbose=False):
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                dets.append((( x1+x2)/2, (y1+y2)/2, x1, y1, x2, y2, float(b.conf[0])))
        raw.append(dets)
        frames_imgs.append(r.orig_img)
    n = len(raw)

    # ---- Static-location map: greedy cluster all detection centres ----
    clusters = []  # [cx,cy,count]
    for dets in raw:
        for (cx, cy, *_ ) in dets:
            hit = None
            for c in clusters:
                if (cx - c[0])**2 + (cy - c[1])**2 <= args.radius**2:
                    hit = c; break
            if hit:
                hit[0] = (hit[0]*hit[2] + cx) / (hit[2]+1)
                hit[1] = (hit[1]*hit[2] + cy) / (hit[2]+1)
                hit[2] += 1
            else:
                clusters.append([cx, cy, 1])
    static = [(c[0], c[1]) for c in clusters if c[2] >= max(2, args.static_frac * n)]

    def is_static(cx, cy):
        return any((cx-sx)**2 + (cy-sy)**2 <= args.radius**2 for sx, sy in static)

    # ---- Pass 2: render with static rejection + simple nearest-neighbour track ----
    H, W = frames_imgs[0].shape[:2]
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0
    vw = cv2.VideoWriter(os.path.join(args.out, f"{stem}_v2.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trail = deque(maxlen=25)
    per_frame = []
    kept_frames = 0
    prev_center = None
    for fi, (img0, dets) in enumerate(zip(frames_imgs, raw)):
        img = img0.copy()
        moving = [d for d in dets if not is_static(d[0], d[1])]
        # choose the moving detection most consistent with motion (nearest to prev),
        # else highest confidence
        chosen = None
        if moving:
            if prev_center is not None:
                chosen = min(moving, key=lambda d: (d[0]-prev_center[0])**2 + (d[1]-prev_center[1])**2)
            else:
                chosen = max(moving, key=lambda d: d[6])
        # draw static-suppressed (grey, thin) for transparency
        for d in dets:
            if is_static(d[0], d[1]):
                cv2.rectangle(img, (int(d[2]),int(d[3])), (int(d[4]),int(d[5])), (120,120,120), 1)
        if chosen is not None:
            cx, cy, x1, y1, x2, y2, cf = chosen
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,220,0), 2)
            cv2.circle(img, (int(cx),int(cy)), 4, (0,0,255), -1)
            cv2.putText(img, f"ball {cf:.2f}", (int(x1), max(int(y1)-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 2, cv2.LINE_AA)
            trail.append((int(cx),int(cy)))
            prev_center = (cx, cy)
            kept_frames += 1
            per_frame.append({"frame": fi, "center":[round(cx,1),round(cy,1)], "conf":round(cf,3)})
        else:
            per_frame.append({"frame": fi, "center": None, "conf": None})
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(img, a, b, (0,165,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{stem} f{fi}/{n-1} moving={len(moving)} static_bg={len(static)}",
                    (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    summary = {"video": stem, "weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
               "frames": n, "static_bg_locations": len(static),
               "frames_with_moving_ball": kept_frames,
               "moving_detection_rate": round(kept_frames/max(n,1),3)}
    with open(os.path.join(args.out, f"{stem}_v2.json"), "w") as f:
        json.dump({"summary": summary, "static": static, "frames": per_frame}, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
