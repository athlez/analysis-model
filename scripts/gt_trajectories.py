"""Regenerate correct ground-truth trajectories for all labelled clips.

Reads raw CVAT XML, then for each clip:
  * auto-detect RELEASE (largest position discontinuity = ball leaving the hand;
    if the track is already clean flight, release = first frame),
  * auto-trim the FROZEN TAIL (ball stationary after it stops),
  * FAITHFUL fit (follows the labels, no rigid-V distortion),
  * render the flight-only Hawk-Eye tube.

Overrides for release can be supplied (per user correction).
Writes output/trajectories/<stem>_hawkeye.{mp4,jpg}, a frame-numbered diagnostic
in output/gt_verify/<stem>_FRAMES.jpg, and a _summary.md of release/end frames.
"""
import argparse, glob, os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

import scripts.hawkeye_tube as ht

SOURCES = "data/ball_dataset/sources"
# every labelled clip in the dataset (9 bowling + 8 Test after the new import)
LABELED = sorted(os.listdir(SOURCES)) if os.path.isdir(SOURCES) else []
OUT = "output/labelled_trajectories"
DIAG = "output/gt_verify"
# user-confirmed release/end frames (override auto-detection)
RELEASE_OVERRIDE = {"bowling32": 22, "bowling_100": 401}   # clips whose auto-release fails
END_OVERRIDE = {"bowling32": 79}   # exclude the hit-away tail (ball struck up & out)


def find_video(stem):
    for d in ("data/raw/own_recordings", "data/main_data"):
        for p in glob.glob(f"{d}/{stem}.*"):
            if p.lower().endswith((".mp4", ".mov")):
                return p
    return None


def parse(stem):
    """Ball centres (px) per frame, read from the dataset's YOLO labels
    (works for all clips: original CVAT bowling + the new Test imports)."""
    vp = find_video(stem)
    cap = cv2.VideoCapture(vp); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
    boxes = {}
    for lp in glob.glob(f"{SOURCES}/{stem}/labels/*.txt"):
        try:
            fi = int(os.path.basename(lp).split("_f")[1][:4])
        except (IndexError, ValueError):
            continue
        parts = open(lp).read().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        boxes[fi] = (cx * W, cy * H)
    return boxes


def trim_frozen_tail(fs, boxes):
    """Last frame before the ball goes stationary for good (walk back from end)."""
    k = len(fs) - 1
    while k > 0 and np.hypot(boxes[fs[k]][0] - boxes[fs[k - 1]][0],
                             boxes[fs[k]][1] - boxes[fs[k - 1]][1]) < 2.0:
        k -= 1
    return fs[k]


def detect_release(fs, boxes, end):
    """Release detection over the flight-trimmed track:
      1. if there's a large position discontinuity (ball label jumps hand->air),
         release = the frame right after it (handles 'arm swings down then flight');
      2. otherwise release = the apex (highest image point, min y) which is the
         top of the arm arc (handles 'arm rises to apex then flight')."""
    tr = [f for f in fs if f <= end]
    if len(tr) < 3:
        return fs[0]
    jumps = [(np.hypot(boxes[b][0] - boxes[a][0], boxes[b][1] - boxes[a][1]), b)
             for a, b in zip(tr, tr[1:])]
    med = np.median([j for j, _ in jumps]) or 1.0
    # a release discontinuity happens EARLY (first half of the flight); a large
    # late jump is the ball being hit, not release.
    cutoff = tr[0] + 0.5 * (end - tr[0])
    early = [(d, b) for d, b in jumps if b <= cutoff]
    if early:
        dmax, bmax = max(early)
        if dmax > max(4 * med, 80):
            return bmax
    return min(tr, key=lambda f: boxes[f][1])   # apex


def flight_end(fs, boxes, release):
    """First frame that starts a sustained (>=4) stationary run after release."""
    idx = [i for i, f in enumerate(fs) if f >= release]
    for k in idx[:-1]:
        run = 1
        while (k + run < len(fs) and
               np.hypot(boxes[fs[k + run]][0] - boxes[fs[k + run - 1]][0],
                        boxes[fs[k + run]][1] - boxes[fs[k + run - 1]][1]) < 2.0):
            run += 1
        if run >= 4:
            return fs[k]
    return fs[-1]


def load_frames(stem):
    cap = cv2.VideoCapture(find_video(stem))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def detect_bounce(xs, ys, min_rise=8.0):
    """Bounce = lowest point on screen (max y) that the ball then rises off of.
    Returns the centreline index, or None if the flight only descends (full toss
    / geometry with no visible bounce)."""
    ys = np.asarray(ys)
    if len(ys) < 6:
        return None
    bi = int(np.argmax(ys))
    if bi <= 1 or bi >= len(ys) - 2:
        return None
    if ys[bi] - ys[bi + 1:].min() < min_rise:      # needs a real rise after the low point
        return None
    return bi


def draw_bounce(img, x, y, tube_radius=5, ss=4):
    """Realistic red cricket ball at the pitch point, sized to sit INSIDE the
    tube's circumference. Supersampled sprite: spherical leather shading lit from
    top-left, glossy specular highlight, curved cream seam with stitches."""
    import math
    r = max(int(tube_radius * 0.9), 4)
    R = r * ss
    S = 2 * R + 4
    spr = np.zeros((S, S, 4), np.uint8)                        # BGRA
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = (xx - c) / R, (yy - c) / R
    d2 = dx * dx + dy * dy
    inside = d2 <= 1.0
    # sphere normal -> Lambert shade from top-left, plus ambient
    nz = np.sqrt(np.clip(1.0 - d2, 0, 1))
    lx, ly, lz = -0.5, -0.5, 0.7
    lam = np.clip(dx * lx + dy * ly + nz * lz, 0, 1)
    shade = 0.35 + 0.75 * lam                                  # ambient + diffuse
    base = np.array([38, 38, 150], np.float32)                 # red leather (BGR)
    body = np.clip(base[None, None, :] * shade[..., None], 0, 255)
    # specular glossy highlight (tight, upper-left)
    spec = np.exp(-(((dx + 0.42) ** 2 + (dy + 0.42) ** 2) / 0.05)) * 210
    body = np.clip(body + spec[..., None], 0, 255)
    spr[..., :3] = body.astype(np.uint8)
    spr[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    # dark rim (limb darkening)
    rim = (d2 > 0.72) & inside
    spr[rim, :3] = (spr[rim, :3].astype(np.float32) * 0.5).astype(np.uint8)
    # curved cream seam + stitches
    seam = (225, 235, 240)
    axis = math.radians(22)
    for t in np.linspace(-0.85, 0.85, 60):                     # seam arc across the face
        sx = t
        sy = 0.30 * math.sin(t * 1.4)                          # gentle curve
        px = sx * math.cos(axis) - sy * math.sin(axis)
        py = sx * math.sin(axis) + sy * math.cos(axis)
        ix, iy = int(c + px * R), int(c + py * R)
        if 0 <= ix < S and 0 <= iy < S and spr[iy, ix, 3]:
            cv2.circle(spr, (ix, iy), max(ss // 2, 1), (*seam, 255), -1, cv2.LINE_AA)
    for t in np.linspace(-0.8, 0.8, 9):                        # stitch ticks
        sx, sy = t, 0.30 * math.sin(t * 1.4)
        px = sx * math.cos(axis) - sy * math.sin(axis)
        py = sx * math.sin(axis) + sy * math.cos(axis)
        nx, ny = math.sin(axis), -math.cos(axis)               # normal to seam
        x1 = int(c + (px - nx * 0.14) * R); y1 = int(c + (py - ny * 0.14) * R)
        x2 = int(c + (px + nx * 0.14) * R); y2 = int(c + (py + ny * 0.14) * R)
        cv2.line(spr, (x1, y1), (x2, y2), (*seam, 255), max(ss // 2, 1), cv2.LINE_AA)
    small = cv2.resize(spr, (S // ss, S // ss), interpolation=cv2.INTER_AREA)
    sw = small.shape[0]
    x0, y0 = int(x - sw / 2), int(y - sw / 2)
    H, W = img.shape[:2]
    ax0, ay0 = max(x0, 0), max(y0, 0); ax1, ay1 = min(x0 + sw, W), min(y0 + sw, H)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    sub = small[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    img[ay0:ay1, ax0:ax1] = (img[ay0:ay1, ax0:ax1].astype(np.float32) * (1 - a)
                             + sub[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def diagnostic(stem, fs, boxes, release, end, frames):
    img = frames[min(end, len(frames) - 1)].copy()
    n = len(fs)
    for i, f in enumerate(fs):
        x, y = boxes[f]
        in_flight = release <= f <= end
        col = (0, 255, 0) if in_flight else (128, 128, 128)
        cv2.circle(img, (int(x), int(y)), 4, col, -1)
        if f % 3 == 1 or f in (release, end):
            cv2.putText(img, str(f), (int(x) + 4, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1)
    cv2.putText(img, f"{stem}  release f{release}  end f{end}  (green=flight, grey=trimmed)",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    os.makedirs(DIAG, exist_ok=True)
    cv2.imwrite(os.path.join(DIAG, f"{stem}_FRAMES.jpg"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=9)
    ap.add_argument("--alpha", type=float, default=0.60)   # semi-transparent
    ap.add_argument("--only", default="", help="comma-separated stems to process (default: all)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    todo = args.only.split(",") if args.only else LABELED

    for stem in todo:
        boxes = parse(stem)
        fs = sorted(boxes)
        end_all = trim_frozen_tail(fs, boxes)
        release = RELEASE_OVERRIDE.get(stem) or detect_release(fs, boxes, end_all)
        end = END_OVERRIDE.get(stem) or flight_end(fs, boxes, release)
        flight = [(f, boxes[f][0], boxes[f][1]) for f in fs if release <= f <= end]
        frames, fps = load_frames(stem)
        diagnostic(stem, fs, boxes, release, end, frames)
        note = "override" if stem in RELEASE_OVERRIDE else "auto"
        if len(flight) < 5:
            rows.append((stem, release, end, len(flight), "TOO FEW — check release"))
            print(f"{stem}: release {release} end {end} -> only {len(flight)} pts (skip)")
            continue

        grid, xs, ys = ht.faithful_centerline(flight, win=3)
        W, H = frames[0].shape[1], frames[0].shape[0]
        f0, f1 = int(grid[0]), int(grid[-1])
        last = min(f1, len(frames) - 1)
        bi = detect_bounce(xs, ys)

        # still: full HQ tube (spline + supersampled) + bounce marker
        still = frames[last].copy()
        ht.draw_tube_hq(still, xs, ys, radius=args.radius, alpha=args.alpha)
        if bi is not None:
            draw_bounce(still, xs[bi], ys[bi], args.radius)
        cv2.imwrite(os.path.join(OUT, f"{stem}_hawkeye.jpg"), still)

        # video: play the delivery CLEAN, then draw the trail in AFTER it finishes
        vw = cv2.VideoWriter(os.path.join(OUT, f"{stem}_hawkeye.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for i in range(0, last + 1):                      # 1) the delivery, no tube
            vw.write(frames[i])
        bg = frames[last]
        reveal = max(int(fps * 1.0), 10)
        for k in range(1, reveal + 1):                    # 2) trail draws in (HQ)
            im = bg.copy()
            ht.draw_tube_hq(im, xs, ys, upto_frac=k / reveal, radius=args.radius, alpha=args.alpha)
            vw.write(im)
        for _ in range(int(fps * 1.5)):                   # 3) hold full trail + bounce
            im = bg.copy()
            ht.draw_tube_hq(im, xs, ys, radius=args.radius, alpha=args.alpha)
            if bi is not None:
                draw_bounce(im, xs[bi], ys[bi], args.radius)
            vw.write(im)
        vw.release()
        rows.append((stem, release, end, len(flight), note + (f", bounce f~{int(grid[bi])}" if bi is not None else ", no bounce")))
        print(f"{stem}: release {release} ({note}) end {end}, {len(flight)} pts, "
              f"bounce={'f'+str(int(grid[bi])) if bi is not None else 'none'}")

    with open(os.path.join(OUT, "_gt_summary.md"), "w") as f:
        f.write("# Ground-truth trajectories (faithful fit, flight-only)\n\n")
        f.write("Check the release frame vs output/gt_verify/<stem>_FRAMES.jpg; "
                "tell me a corrected frame for any clip and I'll re-render.\n\n")
        f.write("| Clip | Release | End | Flight pts | Release source |\n|---|---:|---:|---:|---|\n")
        for s, r, e, n, note in rows:
            f.write(f"| {s} | {r} | {e} | {n} | {note} |\n")
    print("\nwrote", os.path.join(OUT, "_gt_summary.md"))


if __name__ == "__main__":
    main()
