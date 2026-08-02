"""Detect + track the ball with best.pt, then fit a smooth IMAGE-SPACE trajectory
through the detections (robust polynomial + outlier rejection) and render it.

Image-space only (no camera calibration assumed) -> honest smooth ball path, not
a fabricated metric 3D curve.
"""
import argparse, glob, os
import cv2
import numpy as np
from ultralytics import YOLO


def robust_polyfit(t, v, deg=2, iters=3, keep=2.0):
    """Least-squares polyfit with iterative residual outlier rejection."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    mask = np.ones(len(t), bool)
    c = np.polyfit(t[mask], v[mask], deg)
    for _ in range(iters):
        res = v - np.polyval(c, t)
        s = np.std(res[mask]) or 1.0
        mask = np.abs(res) < keep * s
        if mask.sum() <= deg + 1:
            break
        c = np.polyfit(t[mask], v[mask], deg)
    return c, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--out", default="output/best_outputs")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--mode", choices=["yolo", "labels"], default="yolo",
                    help="'labels' uses ground-truth CVAT labels; 'yolo' runs best.pt")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    det = {}   # frame -> (cx, cy, conf, bbox)
    frames_bgr = []
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.mode == "labels":
        # read all frames + ground-truth ball centres from the dataset labels
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames_bgr.append(fr); idx += 1
        cap.release()
        for lp in glob.glob(f"data/ball_dataset/sources/{stem}/labels/*.txt"):
            fi = int(os.path.basename(lp).split("_f")[1][:4])
            c, cx, cy, w, h = map(float, open(lp).read().split()[:5])
            det[fi] = (cx * W, cy * H, 1.0,
                       ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    else:
        cap.release()
        model = YOLO(args.weights)
        for fi, r in enumerate(model.track(source=args.video, tracker="bytetrack.yaml", persist=True,
                                           conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)):
            frames_bgr.append(r.orig_img.copy())
            if r.boxes is not None and len(r.boxes) > 0:
                b = max(r.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                det[fi] = ((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0]), (x1, y1, x2, y2))

    if len(det) < 5:
        print(f"{stem}: only {len(det)} detections — too few for a trajectory fit")
        return

    fs = np.array(sorted(det))
    xs = np.array([det[f][0] for f in fs]); ys = np.array([det[f][1] for f in fs])
    cx_c, mx = robust_polyfit(fs, xs, args.deg)
    cy_c, my = robust_polyfit(fs, ys, args.deg)
    inlier = mx & my
    # smooth curve sampled across the inlier frame span
    f0, f1 = int(fs[inlier].min()), int(fs[inlier].max())
    grid = np.linspace(f0, f1, max(f1 - f0 + 1, 2))
    curve = list(zip(np.polyval(cx_c, grid), np.polyval(cy_c, grid), grid))
    n_in = int(inlier.sum())
    print(f"{stem}: {len(det)} detections, {n_in} inliers, trajectory frames {f0}-{f1}")

    # --- Pass 2: render ---
    out_path = os.path.join(args.out, f"{stem}_trajectory.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, img in enumerate(frames_bgr):
        # full fitted trajectory (faint), then the portion up to now (bright)
        full = [(int(x), int(y)) for x, y, g in curve]
        for a, b in zip(full, full[1:]):
            cv2.line(img, a, b, (200, 120, 0), 1, cv2.LINE_AA)
        grown = [(int(x), int(y)) for x, y, g in curve if g <= fi]
        for a, b in zip(grown, grown[1:]):
            cv2.line(img, a, b, (255, 220, 0), 3, cv2.LINE_AA)
        # raw detections up to now (small red dots)
        for f in fs[fs <= fi]:
            cv2.circle(img, (int(det[f][0]), int(det[f][1])), 3, (0, 0, 255), -1)
        # current ball box
        if fi in det:
            x1, y1, x2, y2 = (int(v) for v in det[fi][3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(img, f"ball {det[fi][2]:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
        # moving point on the fitted curve
        on = [(int(x), int(y)) for x, y, g in curve if abs(g - fi) < 1.0]
        if on:
            cv2.circle(img, on[0], 6, (0, 255, 255), 2)
        cv2.putText(img, f"{stem} f{fi}  ball trajectory (image-space fit, deg {args.deg})",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
