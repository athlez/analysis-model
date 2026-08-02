"""Before/after comparison for the trajectory RENDERER only.
Same (xs,ys) trajectory feed both renderers -> proves data is unchanged.
Produces: before/after still, side-by-side video, FPS, coord-identity check.
"""
import argparse, hashlib, os, sys, time
import numpy as np
import cv2
sys.path.insert(0, os.path.abspath("."))
import scripts.gt_trajectories as G
import scripts.hawkeye_tube as ht

OUT = "output/render_compare"


def h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="bowling3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stem = args.stem

    boxes = G.parse(stem); fs = sorted(boxes)
    end_all = G.trim_frozen_tail(fs, boxes)
    rel = G.RELEASE_OVERRIDE.get(stem) or G.detect_release(fs, boxes, end_all)
    end = G.END_OVERRIDE.get(stem) or G.flight_end(fs, boxes, rel)
    flight = [(f, boxes[f][0], boxes[f][1]) for f in fs if rel <= f <= end]
    grid, xs, ys = ht.faithful_centerline(flight, win=3)      # THE trajectory data
    frames, fps = G.load_frames(stem)
    f0, f1 = int(grid[0]), int(grid[-1]); last = min(f1, len(frames) - 1)
    W, H = frames[0].shape[1], frames[0].shape[0]

    # ---- coordinate-identity check (hash before/after each render) ----
    xs0, ys0 = xs.copy(), ys.copy()
    before = (h(xs), h(ys))
    bg = frames[last]

    old = bg.copy(); ht.draw_tube(old, xs, ys, radius=5, alpha=0.55)
    after_old = (h(xs), h(ys))
    new = bg.copy(); ht.draw_tube_hq(new, xs, ys, radius=5, alpha=0.55)
    after_new = (h(xs), h(ys))
    identical = (before == after_old == after_new) and np.array_equal(xs, xs0) and np.array_equal(ys, ys0)

    # ---- before/after still ----
    def lab(im, t):
        im = im.copy(); cv2.putText(im, t, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(im, t, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2); return im
    cv2.imwrite(f"{OUT}/{stem}_before_after.jpg", np.hstack([lab(old, "BEFORE (segments)"), lab(new, "AFTER (spline HQ)")]))

    # ---- FPS impact (time each renderer over the flight) ----
    def timeit(fn):
        t = time.time()
        for i in range(f0, f1 + 1):
            im = bg.copy(); fn(im, i)
        return (time.time() - t) / max(f1 - f0 + 1, 1)
    t_old = timeit(lambda im, i: ht.draw_tube(im, xs, ys, upto_idx=int(np.searchsorted(grid, i))))
    t_new = timeit(lambda im, i: ht.draw_tube_hq(im, xs, ys, upto_frac=(i - f0) / max(f1 - f0, 1)))

    # ---- side-by-side growing video ----
    vw = cv2.VideoWriter(f"{OUT}/{stem}_sidebyside.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H))
    for i in range(len(frames)):
        L = frames[i].copy(); R = frames[i].copy()
        if f0 <= i <= f1:
            ht.draw_tube(L, xs, ys, upto_idx=int(np.searchsorted(grid, i)))
            ht.draw_tube_hq(R, xs, ys, upto_frac=(i - f0) / max(f1 - f0, 1))
        elif i > f1:
            ht.draw_tube(L, xs, ys); ht.draw_tube_hq(R, xs, ys)
        vw.write(np.hstack([lab(L, "BEFORE"), lab(R, "AFTER")]))
    for _ in range(int(fps)):
        L = frames[last].copy(); R = frames[last].copy()
        ht.draw_tube(L, xs, ys); ht.draw_tube_hq(R, xs, ys)
        vw.write(np.hstack([lab(L, "BEFORE"), lab(R, "AFTER")]))
    vw.release()

    print(f"stem={stem}  points={len(xs)}  flight {f0}-{f1}")
    print(f"COORD IDENTITY: xs/ys hash before={before} after_old={after_old} after_new={after_new} -> {'IDENTICAL' if identical else 'CHANGED!'}")
    print(f"FPS: old {1/t_old:.0f} fps ({t_old*1000:.1f} ms/frame) | new {1/t_new:.0f} fps ({t_new*1000:.1f} ms/frame)")
    print(f"wrote {OUT}/{stem}_before_after.jpg + {stem}_sidebyside.mp4")


if __name__ == "__main__":
    main()
