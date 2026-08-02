"""Arcade-style glowing ball-trail render ("video game" look).

Unlike the Hawk-Eye tube (draws the whole path after the delivery), this streams
a neon comet-trail BEHIND the ball in real time as it flies: a bright glowing
head + a rainbow tail that fades out behind it. Uses the ground-truth labels so
the trail rides the real ball.

    python scripts/ball_trail_game.py --stem bowling_16
"""
import argparse, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))
import scripts.gt_trajectories as G
import scripts.hawkeye_tube as ht

OUT = "output/ball_trail_game"
TRAIL = 22   # how many frames of tail follow the ball


def rainbow(t):
    """t in 0..1 (0=head, 1=tail) -> BGR. Red trail: bright red, warming toward
    orange near the head."""
    return (0, int(70 * (1 - t)), 255)   # BGR: pure red tail -> orange-red head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="bowling_16")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--trail", type=int, default=TRAIL)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    boxes = G.parse(args.stem)
    fs = sorted(boxes)
    if len(fs) < 5:
        print(f"{args.stem}: too few labels"); return
    end_all = G.trim_frozen_tail(fs, boxes)
    rel = G.RELEASE_OVERRIDE.get(args.stem) or G.detect_release(fs, boxes, end_all)
    end = G.END_OVERRIDE.get(args.stem) or G.flight_end(fs, boxes, rel)
    flight = [(f, boxes[f][0], boxes[f][1]) for f in fs if rel <= f <= end]
    grid, xs, ys = ht.faithful_centerline(flight, win=2)
    pos = {int(g): (float(x), float(y)) for g, x, y in zip(grid, xs, ys)}   # per-frame ball px

    frames, fps = G.load_frames(args.stem)
    H, W = frames[0].shape[:2]
    f0, f1 = int(grid[0]), int(grid[-1])

    vw = cv2.VideoWriter(os.path.join(args.out, f"{args.stem}_trail.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.copy()
        if f0 <= i <= f1:
            tail = [pos[g] for g in range(max(f0, i - args.trail), i + 1) if g in pos]
            if len(tail) >= 2:
                glow = np.zeros_like(img)
                n = len(tail)
                for k in range(1, n):
                    a, b = tail[k - 1], tail[k]
                    frac = k / n                      # 0=oldest .. 1=head
                    col = rainbow(1 - frac)
                    thick = max(1, int(2 + 14 * frac))
                    cv2.line(glow, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, thick, cv2.LINE_AA)
                glow = cv2.GaussianBlur(glow, (0, 0), 7)
                img = cv2.add(img, glow)              # additive neon glow
                # bright core over the glow
                for k in range(1, n):
                    a, b = tail[k - 1], tail[k]
                    frac = k / n
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             (255, 255, 255), max(1, int(1 + 4 * frac)), cv2.LINE_AA)
                # glowing ball head
                hx, hy = int(tail[-1][0]), int(tail[-1][1])
                halo = np.zeros_like(img)
                cv2.circle(halo, (hx, hy), 16, (255, 255, 255), -1, cv2.LINE_AA)
                img = cv2.add(img, cv2.GaussianBlur(halo, (0, 0), 9))
                cv2.circle(img, (hx, hy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (hx, hy), 6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print(f"{args.stem}: trail frames {f0}-{f1} of {len(frames)} -> {args.out}/{args.stem}_trail.mp4")


if __name__ == "__main__":
    main()
