"""Build a unified single-class ('ball') YOLO DETECTION dataset from the
Roboflow datasets + valid FullDataset labels.

- detection (v9), tracking (v1): bbox labels, copied.
- ball model (v1), segmentation (v5): polygon/seg labels -> converted to bbox
  (axis-aligned min/max of the polygon vertices).
- FullDataset: only images that have a matching (valid) label; unlabeled ignored.

All classes remapped to 0 = 'ball'. Each source's train/valid/test is preserved
(valid -> val). Filenames are prefixed per source to avoid collisions. Every
box is validated (0<=cx,cy<=1, 0<w,h<=1); degenerate boxes are dropped.
"""
import os, glob, shutil, sys

OUT = "data/yolo_ball"
SRC = {
    "det":  ("data/raw/Roboflow/cricket ball detection.v9i.yolov8", "splits"),
    "trk":  ("data/raw/Roboflow/Cricket Ball Tracking.v1i.yolov8", "splits"),
    "bm":   ("data/raw/Roboflow/Ball model.v1i.yolov8", "splits"),
    "seg":  ("data/raw/Roboflow/cricket ball segmentation.v5i.yolov8", "splits"),
    "full": ("data/raw/FullDataset/YOLO_CombinedDataSplit", "flat"),
}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def convert_line(line):
    """Return (cx,cy,w,h) in [0,1] for a bbox or polygon YOLO line, or None."""
    p = line.split()
    if len(p) < 5:
        return None
    vals = list(map(float, p[1:]))
    if len(vals) == 4:
        cx, cy, w, h = vals
    else:  # polygon: x1 y1 x2 y2 ...
        xs, ys = vals[0::2], vals[1::2]
        if len(xs) < 3:
            return None
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    # clamp centre, validate size
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    if not (1e-4 < w <= 1.0 and 1e-4 < h <= 1.0):
        return None
    return cx, cy, w, h


def convert_label_file(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b = convert_line(line)
            if b:
                out.append(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
    return out


def iter_pairs(root, layout):
    """Yield (split, image_path, label_path) for existing image+label pairs."""
    if layout == "splits":
        splits = [("train", "train"), ("valid", "valid"), ("test", "test")]
    else:  # flat: images/train, labels/train
        splits = [("train", "train")]
    for src_split, _ in splits:
        idir = os.path.join(root, src_split, "images") if layout == "splits" else os.path.join(root, "images", src_split)
        ldir = os.path.join(root, src_split, "labels") if layout == "splits" else os.path.join(root, "labels", src_split)
        if not os.path.isdir(idir):
            continue
        for img in glob.glob(os.path.join(idir, "*")):
            if os.path.splitext(img)[1].lower() not in IMG_EXT:
                continue
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(ldir, stem + ".txt")
            if os.path.isfile(lbl):
                yield SPLIT_MAP[src_split], img, lbl


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)

    stats = {}
    for pfx, (root, layout) in SRC.items():
        c = {"train": 0, "val": 0, "test": 0, "boxes": 0, "skipped_nolabel": 0}
        for split, img, lbl in iter_pairs(root, layout):
            boxes = convert_label_file(lbl)
            if not boxes:  # label present but all boxes invalid -> skip image
                c["skipped_nolabel"] += 1
                continue
            stem = f"{pfx}_{os.path.splitext(os.path.basename(img))[0]}"
            ext = os.path.splitext(img)[1].lower()
            shutil.copy(img, os.path.join(OUT, "images", split, stem + ext))
            with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(boxes) + "\n")
            c[split] += 1
            c["boxes"] += len(boxes)
        stats[pfx] = c

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: " + os.path.abspath(OUT).replace("\\", "/") + "\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames: ['ball']\n")

    print(f"{'source':6s} {'train':>6s} {'val':>5s} {'test':>5s} {'boxes':>7s} {'skipped':>8s}")
    tot = {"train": 0, "val": 0, "test": 0, "boxes": 0}
    for pfx, c in stats.items():
        print(f"{pfx:6s} {c['train']:6d} {c['val']:5d} {c['test']:5d} {c['boxes']:7d} {c['skipped_nolabel']:8d}")
        for k in tot:
            tot[k] += c[k]
    print(f"{'TOTAL':6s} {tot['train']:6d} {tot['val']:5d} {tot['test']:5d} {tot['boxes']:7d}")
    print(f"\ndata.yaml -> {os.path.join(OUT, 'data.yaml')}")


if __name__ == "__main__":
    main()
