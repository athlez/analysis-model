"""Run the ball detector (tiled model + SAHI) + trajectory tube over EVERY video
in own_recordings, writing Hawk-Eye tubes to output/trajectories/.

- The 9 CVAT-labelled clips keep their exact ground-truth tubes (more accurate
  than detection) — skipped here unless --overwrite-labeled.
- Everything else is detected. A quality gate skips clips whose detections are
  too few / too scattered to form a real trajectory, so the output stays clean.
- Writes output/trajectories/_summary.md with per-clip results.

Model loaded ONCE for the whole batch.
"""
import argparse, glob, os, time
import cv2
import numpy as np

import scripts.hawkeye_tube as ht

SRC = "data/raw/own_recordings"
OUT = "output/trajectories"
LABELED = {"bowling1", "bowling2", "bowling3", "bowling25", "bowling26",
           "bowling27", "bowling29", "bowling32", "bowling41"}
MIN_PTS = 8        # need this many inlier ball points for a real trajectory
MIN_SPAN = 15      # ...spanning at least this many frames


def render(frames, pts, stem, fps, W, H, radius, alpha):
    grid, xs, ys = ht.build_centerline(pts)
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].copy()
    ht.draw_tube(still, xs, ys, radius=radius, alpha=alpha)
    cv2.imwrite(os.path.join(OUT, f"{stem}_hawkeye.jpg"), still)
    vw = cv2.VideoWriter(os.path.join(OUT, f"{stem}_hawkeye.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, fr in enumerate(frames):
        img = fr.copy()
        upto = int(np.searchsorted(grid, fi)) if fi >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=radius, alpha=alpha)
        vw.write(img)
    for _ in range(int(fps * 1.0)):
        img = frames[-1].copy(); ht.draw_tube(img, xs, ys, radius=radius, alpha=alpha); vw.write(img)
    vw.release()
    return f0, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/ball_detector_tiled.pt")
    ap.add_argument("--slice", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--overwrite-labeled", action="store_true")
    ap.add_argument("--stride", type=int, default=2, help="detect every Nth frame (fit interpolates)")
    ap.add_argument("--include", default="", help="only process clips whose name contains this substring")
    ap.add_argument("--min-pts", type=int, default=MIN_PTS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    model = ht.load_sahi_model(args.weights, args.conf)
    # dedupe: Windows is case-insensitive, so *.mp4 and *.MP4 match the same files
    seen, vids = set(), []
    for pat in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
        for p in glob.glob(os.path.join(SRC, pat)):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key); vids.append(p)
    vids.sort()
    if args.include:
        vids = [v for v in vids if args.include.lower() in os.path.basename(v).lower()]
    rows = []
    for i, vid in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(vid))[0]
        t0 = time.time()
        if stem in LABELED and not args.overwrite_labeled:
            rows.append((stem, "ground-truth (kept)", "")); print(f"[{i}/{len(vids)}] {stem}: GT kept", flush=True)
            continue
        cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 8:
            rows.append((stem, "skipped", "unreadable/too short")); print(f"[{i}/{len(vids)}] {stem}: unreadable", flush=True); continue
        try:
            pts = ht.detect_centers_sahi(frames, slice_size=args.slice, conf=args.conf,
                                         model=model, stride=args.stride)
        except Exception as e:
            rows.append((stem, "error", str(e)[:60])); print(f"[{i}/{len(vids)}] {stem}: ERROR {e}", flush=True); continue
        span = (max(p[0] for p in pts) - min(p[0] for p in pts)) if pts else 0
        if len(pts) < args.min_pts or span < MIN_SPAN:
            rows.append((stem, "skipped", f"only {len(pts)} pts / span {span}"))
            print(f"[{i}/{len(vids)}] {stem}: skip ({len(pts)} pts, span {span})", flush=True); continue
        f0, f1 = render(frames, pts, stem, fps, W, H, args.radius, args.alpha)
        rows.append((stem, "detected", f"{len(pts)} pts, frames {f0}-{f1}"))
        print(f"[{i}/{len(vids)}] {stem}: OK {len(pts)} pts ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "_summary.md"), "w") as f:
        f.write("# Trajectory outputs — all videos\n\n| Clip | Result | Detail |\n|---|---|---|\n")
        for s, r, d in rows:
            f.write(f"| {s} | {r} | {d} |\n")
        ok = sum(1 for _, r, _ in rows if r in ("detected", "ground-truth (kept)"))
        f.write(f"\n**{ok}/{len(rows)} videos produced a trajectory.**\n")
    n_det = sum(1 for _, r, _ in rows if r == "detected")
    print(f"\nDONE: {n_det} detected + {len(LABELED)} ground-truth tubes in {OUT}/")


if __name__ == "__main__":
    main()
