"""Integrate the 8 confidently-matched new CVAT annotations into ball_dataset.

Adds each (annotation -> Test video) as a new source (frames + YOLO labels),
re-verifies alignment with a zoomed-crop montage, then rebuilds the train/val
split across ALL sources (old bowling + new Test). Does NOT touch the ambiguous
job zips.
"""
import glob, os, sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import scripts.convert_cvat_to_yolo as conv

ROOT = "data/ball_dataset"
NEWL = "C:/Users/VINOD/AppData/Local/Temp/claude/C--Users-VINOD/07743d38-f739-4678-ae4a-93579fd3b812/scratchpad/newlabels"
VERIFY = "output/gt_verify/new_labels"
# job dir -> source video stem (confident matches only)
MATCHES = {"job_2": "Test14", "job_8": "Test6", "job_9": "Test2", "job_10": "Test13",
           "job_15": "Test11", "job_6": "Test24", "job_4": "Test25", "job_5": "Test26"}


def boxes_of(jobdir):
    r = ET.parse(f"{NEWL}/{jobdir}/annotations.xml").getroot()
    W = int(r.findtext(".//original_size/width")); H = int(r.findtext(".//original_size/height"))
    d = {}
    for t in r.findall(".//track"):
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            d[int(b.get("frame"))] = (float(b.get("xtl")), float(b.get("ytl")),
                                      float(b.get("xbr")), float(b.get("ybr")))
    return d, W, H


def add_source(jobdir, stem):
    boxes, W, H = boxes_of(jobdir)
    vp = glob.glob(f"data/raw/own_recordings/{stem}.*")
    vp = [p for p in vp if p.lower().endswith((".mp4", ".mov"))][0]
    idir = f"{ROOT}/sources/{stem}/images"; ldir = f"{ROOT}/sources/{stem}/labels"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = 0
    for fi, (x1, y1, x2, y2) in sorted(boxes.items()):
        if fi >= len(frames):
            continue
        cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
        bw = abs(x2 - x1) / W; bh = abs(y2 - y1) / H
        if bw <= 0 or bh <= 0:
            continue
        cv2.imwrite(f"{idir}/{stem}_f{fi:04d}.jpg", frames[fi])
        with open(f"{ldir}/{stem}_f{fi:04d}.txt", "w") as fh:
            fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n += 1
    return n, boxes, frames, W, H


def verify_tile(stem, boxes, frames, W, H):
    fs = [f for f in sorted(boxes) if f < len(frames)]
    sel = [fs[int(k)] for k in np.linspace(0, len(fs) - 1, 6)]
    tiles = []
    for fi in sel:
        x1, y1, x2, y2 = boxes[fi]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        C = 90; crop = np.zeros((C, C, 3), np.uint8)
        sx, sy = max(cx - C // 2, 0), max(cy - C // 2, 0)
        p = frames[fi][sy:sy + C, sx:sx + C]; crop[:p.shape[0], :p.shape[1]] = p
        cv2.rectangle(crop, (int(x1 - sx), int(y1 - sy)), (int(x2 - sx), int(y2 - sy)), (0, 255, 0), 1)
        crop = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_NEAREST)
        tiles.append(crop)
    row = np.hstack(tiles)
    cv2.putText(row, stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return row


def main():
    os.makedirs(VERIFY, exist_ok=True)
    rows = []
    print("== adding new sources ==")
    for jobdir, stem in MATCHES.items():
        n, boxes, frames, W, H = add_source(jobdir, stem)
        rows.append(verify_tile(stem, boxes, frames, W, H))
        print(f"  {stem}: {n} labeled frames ({W}x{H})")
    cv2.imwrite(f"{VERIFY}/alignment.jpg", np.vstack(rows))

    # rebuild split across ALL sources
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = f"{ROOT}/{sub}/{split}"; os.makedirs(d, exist_ok=True)
            for f in glob.glob(f"{d}/*"):
                os.remove(f)
    counts = conv.assemble_split()
    with open(f"{ROOT}/data.yaml", "w") as f:
        f.write("path: " + os.path.abspath(ROOT).replace(os.sep, "/") + "\n")
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['ball']\n")
    nsrc = len(glob.glob(f"{ROOT}/sources/*"))
    print(f"\nsplit rebuilt: {counts}  | total sources now: {nsrc}")
    print(f"alignment montage: {VERIFY}/alignment.jpg")


if __name__ == "__main__":
    main()
