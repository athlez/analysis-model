"""Hawk-Eye-style 'tube' trajectory render (image-space).

Takes the ball path (ground-truth CVAT labels), smooths it, and draws it as a
shaded cylindrical tube over the video — the classic broadcast look. Produces a
growing-tube video and a final still.

NOTE: image-space stylisation of the real 2D ball path (not a metric 3D
reconstruction — calibration is not reliable on this footage).
"""
import argparse, glob, os
import cv2
import numpy as np


def load_centers(stem, W, H):
    pts = []
    for lp in glob.glob(f"data/ball_dataset/sources/{stem}/labels/*.txt"):
        fi = int(os.path.basename(lp).split("_f")[1][:4])
        _, cx, cy, w, h = map(float, open(lp).read().split()[:5])
        pts.append((fi, cx * W, cy * H))
    pts.sort()
    return pts


def load_sahi_model(weights, conf=0.25, device="cuda:0"):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights,
        confidence_threshold=conf, device=device)


def detect_centers_sahi(frames, weights=None, slice_size=384, conf=0.25, model=None, stride=1):
    """Detect the ball on each full frame with SAHI tiling (tiny-ball recipe),
    keep the highest-confidence box per frame, then robustly reject the few
    false positives so only the real flight remains. Pass a preloaded ``model``
    to avoid reloading weights when processing many clips. ``stride`` subsamples
    frames for detection (the trajectory fit interpolates the gaps)."""
    from sahi.predict import get_sliced_prediction
    if model is None:
        model = load_sahi_model(weights, conf)
    all_dets = []  # (frame, x, y, conf) — every box, not just the best
    for fi in range(0, len(frames), max(1, stride)):
        res = get_sliced_prediction(
            frames[fi][:, :, ::-1], model, slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            perform_standard_pred=False, verbose=0)   # tiles only -> faster, tiny ball
        for o in res.object_prediction_list:
            b = o.bbox
            all_dets.append((fi, (b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2, o.score.value))

    # Reject STATIC detections (stumps, fixed marks): a real ball moves, so a
    # position that recurs across many distinct frames is a background object.
    R, static_frames = 16.0, 8
    kept = []
    for (fi, x, y, c) in all_dets:
        n = len({fj for (fj, xj, yj, cj) in all_dets if abs(xj - x) < R and abs(yj - y) < R})
        if n < static_frames:
            kept.append((fi, x, y, c))

    # highest-confidence moving detection per frame, then outlier rejection
    perf = {}
    for (fi, x, y, c) in kept:
        if fi not in perf or c > perf[fi][2]:
            perf[fi] = (x, y, c)
    pts = [(fi, xy[0], xy[1]) for fi, xy in sorted(perf.items())]
    return _reject_outliers(pts)


def _reject_outliers(pts, deg=2, iters=3, keep=2.5):
    """Iterative polynomial-residual outlier rejection on (frame, x, y)."""
    if len(pts) < 5:
        return pts
    t = np.array([p[0] for p in pts], float)
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    mask = np.ones(len(t), bool)
    for _ in range(iters):
        cx = np.polyfit(t[mask], x[mask], deg); cy = np.polyfit(t[mask], y[mask], deg)
        rx = x - np.polyval(cx, t); ry = y - np.polyval(cy, t)
        res = np.hypot(rx, ry)
        s = np.std(res[mask]) or 1.0
        nm = res < keep * s
        if nm.sum() <= deg + 2 or nm.sum() == mask.sum():
            mask = nm; break
        mask = nm
    return [pts[i] for i in range(len(pts)) if mask[i]]


def faithful_centerline(pts, win=3):
    """Trajectory that FOLLOWS the points (light smoothing only). Correct for
    clean ground-truth labels — unlike build_centerline's rigid descending/bounce/
    rising model, which over-smooths and cuts real corners (fine for noisy
    detections, wrong for exact labels)."""
    t = np.array([p[0] for p in pts], float)
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    grid = np.arange(int(t.min()), int(t.max()) + 1).astype(float)
    xi = np.interp(grid, t, x); yi = np.interp(grid, t, y)
    if win > 1 and len(grid) > 2 * win:
        k = np.ones(win) / win
        xs = np.convolve(xi, k, "same"); ys = np.convolve(yi, k, "same")
        xs[:win] = xi[:win]; xs[-win:] = xi[-win:]
        ys[:win] = yi[:win]; ys[-win:] = yi[-win:]
        return grid, xs, ys
    return grid, xi, yi


def build_centerline(pts, n=260):
    """Clean, non-wobbly centreline: fit the flight as a descending segment + a
    bounce + a rising segment (a low-order poly on each side of the pitch point),
    so the tube edges stay straight instead of tracing label jitter."""
    t = np.array([p[0] for p in pts], float)
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    # bounce = lowest point on screen (max y), found on a lightly smoothed y
    ys_s = np.convolve(y, np.ones(5) / 5, mode="same")
    tb = t[int(np.argmax(ys_s))]
    mA, mB = t <= tb, t >= tb

    def seg(mask, g):
        tt = t[mask]
        deg = 2 if len(tt) >= 4 else 1
        return np.polyval(np.polyfit(tt, x[mask], deg), g), np.polyval(np.polyfit(tt, y[mask], deg), g)

    if mA.sum() >= 2 and mB.sum() >= 2:
        sA, sB = tb - t.min(), t.max() - tb
        nA = max(int(n * sA / (sA + sB)), 2); nB = max(n - nA, 2)
        gA = np.linspace(t.min(), tb, nA); gB = np.linspace(tb, t.max(), nB)
        xA, yA = seg(mA, gA); xB, yB = seg(mB, gB)
        return (np.concatenate([gA, gB]),
                np.concatenate([xA, xB]), np.concatenate([yA, yB]))
    deg = 2 if len(t) >= 4 else 1
    grid = np.linspace(t.min(), t.max(), n)
    return grid, np.polyval(np.polyfit(t, x, deg), grid), np.polyval(np.polyfit(t, y, deg), grid)


# minimal palette (BGR): pure-white translucent fill + a neutral grey edge
FILL = (55, 55, 220)     # (used by legacy draw_tube) red
EDGE = (18, 18, 80)      # dark red edge
# Hawk-Eye-style shaded red tube: concentric strokes, dark-red rim -> bright core.
# (width_scale relative to full tube diameter, BGR)
TUBE_RAMP = [
    (1.00, (20, 20, 55)),     # dark red rim
    (0.74, (34, 34, 120)),    # red body
    (0.42, (44, 44, 165)),    # slightly lifted centre (subtle roundness, NOT a glow)
]


def draw_tube(img, xs, ys, upto_idx=None, radius=8, alpha=0.5):
    """Flat, see-through tube: one translucent fill + a thin edge, blended once."""
    end = len(xs) if upto_idx is None else min(upto_idx, len(xs))
    if end < 2:
        return
    pts = np.stack([xs[:end], ys[:end]], 1).astype(np.int32).reshape(-1, 1, 2)
    ov = img.copy()
    cv2.polylines(ov, [pts], False, EDGE, radius * 2 + 2, cv2.LINE_AA)   # subtle edge
    cv2.polylines(ov, [pts], False, FILL, radius * 2 - 2, cv2.LINE_AA)   # translucent fill
    # small ball marker at the leading end (also part of the overlay -> translucent)
    hx, hy = int(xs[end - 1]), int(ys[end - 1])
    cv2.circle(ov, (hx, hy), radius - 1, FILL, -1, cv2.LINE_AA)
    cv2.circle(ov, (hx, hy), radius - 1, EDGE, 1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


# --------------------------------------------------------------------------- #
# High-quality renderer (spline + arc-resample + supersampled continuous tube)
# Consumes the EXACT same (xs, ys) trajectory points — no data is altered.
# --------------------------------------------------------------------------- #
def _catmull_rom(P, n=18):
    """Centripetal-ish Catmull-Rom spline through every control point (C1
    continuous; the curve passes exactly through each trajectory point)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    pad = np.vstack([P[0], P, P[-1]])            # clamp ends
    seg = []
    for i in range(1, len(pad) - 2):
        p0, p1, p2, p3 = pad[i - 1], pad[i], pad[i + 1], pad[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        seg.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.vstack(seg + [pad[-2]])


def _arc_resample(dense, step=1.5):
    """Uniform spacing along arc length -> no clusters/gaps, adaptive density
    (long spans get more samples, short spans fewer)."""
    d = np.asarray(dense, float)
    if len(d) < 3:
        return d
    seglen = np.hypot(np.diff(d[:, 0]), np.diff(d[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    if total < step:
        return d
    s = np.linspace(0, total, max(int(total / step), 2))
    return np.stack([np.interp(s, cum, d[:, 0]), np.interp(s, cum, d[:, 1])], 1)


def draw_tube_hq(img, xs, ys, radius=5, alpha=0.55, upto_frac=1.0, ss=3, bounce_idx=None):
    """Hawk-Eye-grade pipe: Catmull-Rom spline through the SAME points, arc-length
    resampled, drawn as ONE continuous constant-radius stroke with rounded caps,
    rendered supersampled (ss x) and downsampled (INTER_AREA) for clean edges.
    Does not modify xs/ys.

    If bounce_idx is given, the bounce is treated as a DISCONTINUITY: the path is
    split into two independent splines (release->bounce and bounce->end). Both pass
    exactly through the bounce point but share no tangent, so the bounce renders as
    a crisp V — no smoothing across it."""
    P = list(zip(xs, ys))
    if len(P) < 2:
        return
    if bounce_idx is not None and 1 <= bounce_idx <= len(P) - 2:
        b = int(bounce_idx)
        d1 = _catmull_rom(P[:b + 1])            # release -> bounce (independent tangents)
        d2 = _catmull_rom(P[b:])                # bounce -> end
        dense = _arc_resample(np.vstack([d1, d2]), step=1.5)
    else:
        dense = _arc_resample(_catmull_rom(P), step=1.5)
    if upto_frac < 1.0:
        dense = dense[:max(2, int(len(dense) * upto_frac))]
    H, W = img.shape[:2]
    pad = radius + 3
    x0 = max(int(dense[:, 0].min() - pad), 0); y0 = max(int(dense[:, 1].min() - pad), 0)
    x1 = min(int(dense[:, 0].max() + pad) + 1, W); y1 = min(int(dense[:, 1].max() + pad) + 1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    bw, bh = x1 - x0, y1 - y0
    layer = np.zeros((bh * ss, bw * ss, 3), np.uint8)
    mask = np.zeros((bh * ss, bw * ss), np.uint8)
    poly = ((dense - [x0, y0]) * ss).astype(np.int32).reshape(-1, 1, 2)
    diam = max(int(2 * radius * ss), 3)                             # full tube diameter (SS space)
    ends = (poly[0, 0], poly[-1, 0])
    # gradient tube: dark-red rim outward -> bright core, drawn widest first
    for scale, col in TUBE_RAMP:
        w = max(int(scale * diam), 1)
        cv2.polylines(layer, [poly], False, col, w, cv2.LINE_AA)
        for e in ends:                                             # rounded caps, same gradient
            cv2.circle(layer, tuple(int(v) for v in e), max(w // 2, 1), col, -1, cv2.LINE_AA)
    cv2.polylines(mask, [poly], False, 255, diam, cv2.LINE_AA)     # opacity mask = full width
    for e in ends:
        cv2.circle(mask, tuple(int(v) for v in e), diam // 2, 255, -1, cv2.LINE_AA)
    layer = cv2.resize(layer, (bw, bh), interpolation=cv2.INTER_AREA)
    m = (cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * alpha)[..., None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - m) + layer.astype(np.float32) * m).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/own_recordings/bowling2.mp4")
    ap.add_argument("--out", default="output/demo")
    ap.add_argument("--alpha", type=float, default=0.5, help="tube opacity (0=invisible, 1=solid)")
    ap.add_argument("--radius", type=int, default=5, help="tube thickness")
    ap.add_argument("--source", choices=["labels", "detect"], default="labels",
                    help="'labels' uses ground-truth CVAT labels; 'detect' runs the tiled detector + SAHI")
    ap.add_argument("--weights", default="models/ball_detector_tiled.pt")
    ap.add_argument("--slice", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    if args.source == "detect":
        pts = detect_centers_sahi(frames, args.weights, args.slice, args.conf)
    else:
        pts = load_centers(stem, W, H)
    if len(pts) < 4:
        print(f"{stem}: too few ball points ({len(pts)}) — skipping"); return
    grid, xs, ys = build_centerline(pts)
    f_first, f_last = int(grid[0]), int(grid[-1])

    # --- still: full tube on the last flight frame ---
    still_bg = frames[min(f_last, len(frames) - 1)].copy()
    draw_tube(still_bg, xs, ys, radius=args.radius, alpha=args.alpha)
    still_path = os.path.join(args.out, f"{stem}_hawkeye.jpg")
    cv2.imwrite(still_path, still_bg)

    # --- video: tube grows with the ball ---
    vw = cv2.VideoWriter(os.path.join(args.out, f"{stem}_hawkeye.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, fr in enumerate(frames):
        img = fr.copy()
        # how much of the tube to show: proportional to current frame vs flight
        upto = int(np.searchsorted(grid, fi)) if fi >= f_first else 0
        if upto >= 2:
            draw_tube(img, xs, ys, upto_idx=upto, radius=args.radius, alpha=args.alpha)
        vw.write(img)
    # hold the completed tube for ~1.2 s at the end
    for _ in range(int(fps * 1.2)):
        img = frames[-1].copy()
        draw_tube(img, xs, ys, radius=args.radius, alpha=args.alpha)
        vw.write(img)
    vw.release()
    print(f"{stem}: {len(pts)} ball points, tube frames {f_first}-{f_last}")
    print("wrote", still_path, "and", os.path.join(args.out, f"{stem}_hawkeye.mp4"))


if __name__ == "__main__":
    main()
