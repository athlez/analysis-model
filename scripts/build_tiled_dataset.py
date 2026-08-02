"""Slice the ball dataset into TILES so the tiny (4-14px) ball is large enough
for YOLO to learn. Standard SAHI small-object recipe: train on tiles, infer on
tiles. Full-frame training shrank the ball below the detectable limit (that's
why the earlier fine-tune stalled at mAP~0); tiling fixes that.

Reads data/ball_dataset/images/{train,val} (+ labels) and writes an equivalent
tiled dataset at data/ball_tiles/ with the SAME train/val split preserved.
Keeps every tile containing a ball, plus a capped number of empty tiles as
negatives (to suppress false positives).
"""
import argparse, glob, os, random
import cv2
import numpy as np

SRC = "data/ball_dataset"
OUT = "data/ball_tiles"
HOLDOUT = set()   # clip stems reserved for held-out testing (set in main)


def tile_origins(total, size, stride):
    if total <= size:
        return [0]
    xs = list(range(0, total - size + 1, stride))
    if xs[-1] != total - size:
        xs.append(total - size)
    return xs


def process_split(split, size, overlap, neg_per_img, min_frac):
    img_dir = os.path.join(SRC, "images", split)
    out_img = os.path.join(OUT, "images", split)
    out_lbl = os.path.join(OUT, "labels", split)
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)
    stride = int(size * (1 - overlap))
    n_pos = n_neg = 0

    for ip in glob.glob(os.path.join(img_dir, "*.jpg")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        if stem.split("_f")[0] in HOLDOUT:      # clip-level holdout -> never tiled/trained
            continue
        lp = os.path.join(SRC, "labels", split, stem + ".txt")
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []  # abs (x1,y1,x2,y2)
        if os.path.isfile(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, p[:5])
                x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
                boxes.append((x1, y1, x2, y2))

        empties = []
        for oy in tile_origins(H, size, stride):
            for ox in tile_origins(W, size, stride):
                tb = []
                for (x1, y1, x2, y2) in boxes:
                    ix1, iy1 = max(x1, ox), max(y1, oy)
                    ix2, iy2 = min(x2, ox + size), min(y2, oy + size)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    barea = max((x2 - x1) * (y2 - y1), 1e-6)
                    if inter / barea < min_frac:      # ball mostly outside tile
                        continue
                    ncx = ((ix1 + ix2) / 2 - ox) / size
                    ncy = ((iy1 + iy2) / 2 - oy) / size
                    nw = (ix2 - ix1) / size; nh = (iy2 - iy1) / size
                    tb.append((ncx, ncy, nw, nh))
                name = f"{stem}_x{ox}_y{oy}"
                if tb:
                    cv2.imwrite(os.path.join(out_img, name + ".jpg"), img[oy:oy + size, ox:ox + size])
                    with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
                        for (cx, cy, w, h) in tb:
                            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    n_pos += 1
                else:
                    empties.append((ox, oy, name))
        # keep a few negatives per image
        random.shuffle(empties)
        for ox, oy, name in empties[:neg_per_img]:
            cv2.imwrite(os.path.join(out_img, name + ".jpg"), img[oy:oy + size, ox:ox + size])
            open(os.path.join(out_lbl, name + ".txt"), "w").close()
            n_neg += 1
    return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384, help="tile size (px)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--neg_per_img", type=int, default=2, help="empty tiles kept per frame")
    ap.add_argument("--min_frac", type=float, default=0.35, help="min fraction of ball inside a tile to label it")
    ap.add_argument("--holdout", default="", help="comma-separated clip stems to EXCLUDE (held out for testing)")
    args = ap.parse_args()
    random.seed(0)
    global HOLDOUT
    HOLDOUT = set(s for s in args.holdout.split(",") if s)
    if HOLDOUT:
        print(f"holding out (not trained): {sorted(HOLDOUT)}")

    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    totals = {}
    for split in ("train", "val"):
        totals[split] = process_split(split, args.size, args.overlap, args.neg_per_img, args.min_frac)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT).replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['ball']\n")
    for split, (p, n) in totals.items():
        print(f"{split}: {p} positive tiles + {n} negative tiles")
    print(f"tile size {args.size}px, overlap {args.overlap}; wrote {OUT}/data.yaml")


if __name__ == "__main__":
    main()
