"""Run a trained TrackNet on a video -> per-frame ball position -> Hawk-Eye tube.

    python tracknet/infer.py --video <clip> --weights tracknet/tracknet_best.pt

Slides a 3-frame window over the video, reads the ball as the heatmap peak
(above --thresh), maps it back to full-res, then renders the trajectory with the
existing tube renderer. No SAHI, no tiling — one fast forward pass per frame.
"""
import argparse, os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracknet.model import TrackNetV2
import scripts.hawkeye_tube as ht

IN_W, IN_H = 288, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="tracknet/tracknet_best.pt")
    ap.add_argument("--out", default="output/tracknet")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = TrackNetV2(3).to(dev).eval()
    model.load_state_dict(torch.load(args.weights, map_location=dev))

    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    H, W = frames[0].shape[:2]
    small = [cv2.cvtColor(cv2.resize(f, (IN_W, IN_H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
             for f in frames]

    pts = []  # (frame_idx, cx_full, cy_full)
    with torch.no_grad():
        for i in range(2, len(frames)):
            x = np.concatenate([small[i - 2].transpose(2, 0, 1),
                                small[i - 1].transpose(2, 0, 1),
                                small[i].transpose(2, 0, 1)], 0)[None]
            p = model(torch.from_numpy(x).to(dev)).cpu().numpy()[0, 2]  # heatmap of the latest frame
            if p.max() >= args.thresh:
                py, px = np.unravel_index(p.argmax(), p.shape)
                pts.append((i, px / IN_W * W, py / IN_H * H))
    print(f"{stem}: ball found in {len(pts)}/{len(frames)} frames")
    if len(pts) < 6:
        print("too few -> skip"); return

    pts = ht._reject_outliers(pts)
    grid, xs, ys = ht.build_centerline(pts)
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].copy()
    ht.draw_tube(still, xs, ys, radius=args.radius, alpha=args.alpha)
    cv2.imwrite(f"{args.out}/{stem}_hawkeye.jpg", still)
    vw = cv2.VideoWriter(f"{args.out}/{stem}_hawkeye.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.copy()
        upto = int(np.searchsorted(grid, i)) if i >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=args.radius, alpha=args.alpha)
        vw.write(img)
    vw.release()
    print("wrote", f"{args.out}/{stem}_hawkeye.mp4")


if __name__ == "__main__":
    main()
