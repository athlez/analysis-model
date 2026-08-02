"""Autonomous ball auto-labeler (trajectory-based).

For a fixed-camera bowling clip, isolate the cricket ball's flight and emit YOLO
labels — WITHOUT any manual clicking.

Pipeline:
  1. Median background over the clip (static camera) -> foreground = |gray-median|.
  2. Per frame: threshold foreground, find SMALL ball-sized blobs (candidates).
  3. Trajectory search: greedily build tracklets of candidates that move fast &
     smoothly (constant-velocity gate); score by length, straightness, speed.
  4. Accept the best tracklet only if it clears strict quality gates (else skip
     the clip — never fabricate a ball).
  5. Emit per-frame YOLO labels (class 0 = ball) + a QA overlay video + JSON.

Why this isolates the ball: the ball is the one small object that travels in a
fast, temporally-consistent, near-low-order path across several frames; body and
net motion jitter locally and fail the smoothness/speed gates.
"""
import argparse, glob, json, os
import cv2
import numpy as np


def median_background(frames_gray, max_samples=60):
    idx = np.linspace(0, len(frames_gray) - 1, min(max_samples, len(frames_gray))).astype(int)
    stack = np.stack([frames_gray[i] for i in idx], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def extract_candidates(frames_gray, bg, diff_thr, min_area, max_area, max_side, cap=25):
    """Return list per frame of candidate dicts, capped to `cap` most ball-like
    (compact, roundish) blobs to keep the trajectory search tractable."""
    cand = []
    for g in frames_gray:
        d = cv2.absdiff(g, bg)
        _, m = cv2.threshold(d, diff_thr, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = []
        for c in cnts:
            a = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if a < min_area or a > max_area or max(w, h) > max_side:
                continue
            fill = a / max(w * h, 1)                 # roundness/compactness proxy
            aspect = max(w, h) / max(min(w, h), 1)
            score = fill / (1.0 + 0.3 * (aspect - 1))  # prefer compact, low-aspect
            cs.append({"cx": x + w / 2.0, "cy": y + h / 2.0, "w": float(w), "h": float(h), "s": score})
        cs.sort(key=lambda c: c["s"], reverse=True)
        cand.append(cs[:cap])
    return cand


def ransac_track(cand, min_len, min_speed, max_speed, tol, iters=6000, seed=0):
    """Find the best fast, near-constant-velocity track through the candidates.

    RANSAC: hypothesize velocity from two candidates a few frames apart, count
    inliers (one closest candidate per frame within `tol` of the linear
    prediction). Robust to heavy clutter; fixed iteration budget => fast.
    Returns list of (frame,cx,cy,w,h) sorted by frame, or [].
    """
    rng = np.random.default_rng(seed)
    pts = []  # (f, x, y, w, h)
    for f, cs in enumerate(cand):
        for c in cs:
            pts.append((f, c["cx"], c["cy"], c["w"], c["h"]))
    if len(pts) < min_len:
        return []
    P = np.array(pts, float)
    F, X, Y = P[:, 0], P[:, 1], P[:, 2]
    nfr = len(cand)
    frame_start_idx = {}
    order = np.argsort(F)
    Fs, Xs, Ys = F[order], X[order], Y[order]
    Psorted = P[order]

    best_inl, best_track = 0, []
    n = len(pts)
    for _ in range(iters):
        i, j = rng.integers(0, n), rng.integers(0, n)
        fa, fb = F[i], F[j]
        if fb <= fa or (fb - fa) > 4:
            continue
        vx = (X[j] - X[i]) / (fb - fa)
        vy = (Y[j] - Y[i]) / (fb - fa)
        sp = np.hypot(vx, vy)
        if sp < min_speed or sp > max_speed:
            continue
        # predicted position for every candidate at its own frame
        predx = X[i] + vx * (Fs - fa)
        predy = Y[i] + vy * (Fs - fa)
        dist = np.hypot(Xs - predx, Ys - predy)
        inmask = dist < tol
        if inmask.sum() < min_len:
            continue
        # one (closest) candidate per frame
        chosen = {}
        for k in np.where(inmask)[0]:
            f = int(Fs[k])
            if f not in chosen or dist[k] < chosen[f][0]:
                chosen[f] = (dist[k], Psorted[k])
        frames_hit = sorted(chosen)
        if len(frames_hit) < min_len:
            continue
        # require the hit frames to be reasonably contiguous (a real flight)
        span = frames_hit[-1] - frames_hit[0] + 1
        if len(frames_hit) < 0.5 * span:
            continue
        if len(frames_hit) > best_inl:
            best_inl = len(frames_hit)
            best_track = [tuple(chosen[f][1]) for f in frames_hit]
    return [(int(t[0]), t[1], t[2], t[3], t[4]) for t in best_track]


def build_tracklets(cand, max_jump, min_len, max_accel):
    """Greedy constant-velocity tracklets over candidate points.

    Start a track from each candidate; extend to the next frame by picking the
    candidate closest to the constant-velocity prediction, within max_jump and
    with acceleration below max_accel. Returns list of tracklets (list of
    (frame, cx, cy, w, h)).
    """
    n = len(cand)
    used = [set() for _ in range(n)]
    tracks = []
    for f0 in range(n - min_len):
        for i0, c0 in enumerate(cand[f0]):
            if i0 in used[f0]:
                continue
            # seed with a second point in the next 1-2 frames
            for gap in (1, 2):
                if f0 + gap >= n:
                    continue
                for i1, c1 in enumerate(cand[f0 + gap]):
                    dx = (c1["cx"] - c0["cx"]) / gap
                    dy = (c1["cy"] - c0["cy"]) / gap
                    speed = np.hypot(dx, dy)
                    if speed < 4 or speed > max_jump:  # must actually move
                        continue
                    track = [(f0, c0), (f0 + gap, c1)]
                    px, py, vx, vy = c1["cx"], c1["cy"], dx, dy
                    fcur = f0 + gap
                    while fcur + 1 < n:
                        fnext = fcur + 1
                        predx, predy = px + vx, py + vy
                        best, bd = None, 1e9
                        for j, cj in enumerate(cand[fnext]):
                            dd = np.hypot(cj["cx"] - predx, cj["cy"] - predy)
                            if dd < bd:
                                bd, best = dd, (j, cj)
                        if best is None or bd > max_accel + max_jump * 0.0:
                            break
                        if bd > max(20.0, 0.6 * np.hypot(vx, vy)):  # gate: accel bound
                            break
                        j, cj = best
                        nvx, nvy = cj["cx"] - px, cj["cy"] - py
                        track.append((fnext, cj))
                        px, py, vx, vy = cj["cx"], cj["cy"], nvx, nvy
                        fcur = fnext
                    if len(track) >= min_len:
                        for (ff, cc) in track:
                            # mark approx used
                            pass
                        tracks.append([(ff, cc["cx"], cc["cy"], cc["w"], cc["h"]) for ff, cc in track])
    return tracks


def score_track(tr):
    """Higher = more ball-like: long, fast, smooth (low residual to quadratic)."""
    if len(tr) < 3:
        return 0.0
    t = np.array([p[0] for p in tr], float)
    x = np.array([p[1] for p in tr], float)
    y = np.array([p[2] for p in tr], float)
    # fit quadratic in t for x and y; residual = smoothness
    try:
        rx = np.polyfit(t, x, 2, full=True)[1]
        ry = np.polyfit(t, y, 2, full=True)[1]
        res = float((rx[0] if len(rx) else 0) + (ry[0] if len(ry) else 0)) / len(tr)
    except Exception:
        res = 1e9
    disp = np.hypot(x[-1] - x[0], y[-1] - y[0])
    speed = disp / max(t[-1] - t[0], 1)
    smooth = 1.0 / (1.0 + res / 50.0)
    length = len(tr)
    net = disp / (1e-6 + sum(np.hypot(np.diff(x), np.diff(y))))  # straightness 0..1
    return length * smooth * min(speed / 8.0, 3.0) * (0.5 + 0.5 * net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/ball_autolabel")
    ap.add_argument("--start", type=float, default=0.0, help="start time s (0=whole clip)")
    ap.add_argument("--end", type=float, default=0.0, help="end time s (0=whole clip)")
    ap.add_argument("--diff_thr", type=int, default=22)
    ap.add_argument("--min_area", type=float, default=3.0)
    ap.add_argument("--max_area", type=float, default=400.0)
    ap.add_argument("--max_side", type=float, default=45.0)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--min_speed", type=float, default=6.0, help="px/frame")
    ap.add_argument("--max_speed", type=float, default=200.0)
    ap.add_argument("--tol", type=float, default=14.0, help="inlier tol (px)")
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--qa", action="store_true", help="write QA overlay video")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, grays = [], []
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = fi / fps
        if (args.end == 0 or t <= args.end) and t >= args.start:
            frames.append(fr)
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        fi += 1
    cap.release()
    if len(frames) < args.min_len + 2:
        print(json.dumps({"video": stem, "accepted": False, "reason": "too few frames"}))
        return

    H, W = frames[0].shape[:2]
    bg = median_background(grays)
    cand = extract_candidates(grays, bg, args.diff_thr, args.min_area, args.max_area,
                              args.max_side, cap=args.cap)
    best = ransac_track(cand, args.min_len, args.min_speed, args.max_speed, args.tol)
    best_score = score_track(best) if best else 0.0
    accepted = len(best) >= args.min_len

    result = {"video": stem, "fps": round(fps, 2), "frames": len(frames),
              "n_candidates_total": int(sum(len(c) for c in cand)),
              "track_len": len(best), "best_score": round(float(best_score), 2),
              "accepted": bool(accepted)}

    if accepted:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
        box = 22  # fixed label box side (px) — ball is tiny; consistent size
        for (ff, cx, cy, w, h) in best:
            img = frames[ff]
            bw = max(w, 10) + 6
            bh = max(h, 10) + 6
            name = f"{stem}_f{ff:04d}"
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img)
            with open(os.path.join(args.out, "labels", name + ".txt"), "w") as fh:
                fh.write(f"0 {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}\n")
        result["labels_written"] = len(best)

    if args.qa or accepted:
        os.makedirs(os.path.join(args.out, "qa"), exist_ok=True)
        vw = cv2.VideoWriter(os.path.join(args.out, "qa", f"{stem}_autolabel.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        track_by_frame = {}
        if best:
            for (ff, cx, cy, w, h) in best:
                track_by_frame[ff] = (cx, cy, w, h)
        trail = []
        for i, fr in enumerate(frames):
            vis = fr.copy()
            for c in cand[i]:
                cv2.circle(vis, (int(c["cx"]), int(c["cy"])), 3, (120, 120, 120), 1)
            if i in track_by_frame:
                cx, cy, w, h = track_by_frame[i]
                trail.append((int(cx), int(cy)))
                cv2.rectangle(vis, (int(cx-w/2-4), int(cy-h/2-4)), (int(cx+w/2+4), int(cy+h/2+4)), (0, 255, 0), 2)
                cv2.putText(vis, "BALL", (int(cx)+8, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for a, b in zip(trail, trail[1:]):
                cv2.line(vis, a, b, (0, 165, 255), 2)
            cv2.putText(vis, f"{stem} f{i} score={best_score:.1f} acc={accepted}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(vis)
        vw.release()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
