"""Generate contrast-enhanced frame strips around each delivery so a human can
eyeball whether the cricket ball is visible/labelable.

Per clip -> output/frame_strips/<clip>/fNNNN.jpg (full-res, CLAHE-enhanced) for
~17 consecutive frames centred on the motion peak (release/flight), plus a
downscaled montage <clip>_overview.jpg.
"""
import glob, os, cv2, numpy as np

SRC = "data/raw/own_recordings"
OUT = "output/frame_strips"
PRE, POST = 5, 12  # frames before/after the motion peak


def enhance(fr):
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def process(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    frames, prev, energy = [], None, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        g = cv2.cvtColor(cv2.resize(fr, (fr.shape[1] // 4, fr.shape[0] // 4)), cv2.COLOR_BGR2GRAY)
        energy.append(0.0 if prev is None else float(np.abs(g.astype(int) - prev).mean()))
        prev = g
    cap.release()
    if not frames:
        return stem, 0, None
    peak = int(np.argmax(energy))
    lo, hi = max(0, peak - PRE), min(len(frames) - 1, peak + POST)
    cdir = os.path.join(OUT, stem)
    os.makedirs(cdir, exist_ok=True)
    tiles = []
    for fi in range(lo, hi + 1):
        enh = enhance(frames[fi])
        lbl = enh.copy()
        cv2.putText(lbl, f"{stem} f{fi}" + ("  <-peak" if fi == peak else ""),
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(cdir, f"f{fi:04d}.jpg"), lbl)
        th = 420
        tiles.append(cv2.resize(lbl, (int(lbl.shape[1] * th / lbl.shape[0]), th)))
    # overview montage (rows of 6)
    rows = [cv2.hconcat(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(os.path.join(OUT, f"{stem}_overview.jpg"), cv2.vconcat(rows))
    return stem, hi - lo + 1, peak


def main():
    os.makedirs(OUT, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(SRC, "*.mp4")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    for i, c in enumerate(clips, 1):
        stem, n, peak = process(c)
        print(f"[{i}/{len(clips)}] {stem}: {n} enhanced frames around peak f{peak}", flush=True)
    print(f"\nDone -> {OUT}/ (per-clip folders + *_overview.jpg)")


if __name__ == "__main__":
    main()
