"""Integrate the batch-2 labelled zips (named bowlingN[labelled].zip, where N
refers to data/main_data/bowling_N.mp4) into ball_dataset, with verification.

For each zip: extract CVAT XML -> sanity-check resolution/frames against the
main_data video -> write sources/bowling_N/{images,labels} -> alignment tile.
Then rebuild the train/val split across ALL sources.
"""
import glob, os, re, sys, zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import scripts.convert_cvat_to_yolo as conv

ROOT = "data/ball_dataset"
ZIPS = "data/raw/labelled_videos"
MAIN = "data/main_data"
VERIFY = "output/gt_verify/new_labels"
# already-integrated numeric ids: any bowling_N source, plus the original 9
_srcnums = {int(m.group(1)) for s in (os.listdir(f"{ROOT}/sources") if os.path.isdir(f"{ROOT}/sources") else [])
            for m in [re.match(r"bowling_(\d+)$", s)] if m}
EXISTING = _srcnums | {1, 2, 3, 25, 26, 27, 29, 32, 41}


def parse_xml(xml_path):
    r = ET.parse(xml_path).getroot()
    W = int(r.findtext(".//original_size/width")); H = int(r.findtext(".//original_size/height"))
    size = int(r.findtext(".//job/size") or 0)
    boxes = {}
    for t in r.findall(".//track"):
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = (float(b.get("xtl")), float(b.get("ytl")),
                                          float(b.get("xbr")), float(b.get("ybr")))
    return boxes, W, H, size


def main():
    os.makedirs(VERIFY, exist_ok=True)
    # Accept any bowling/ball/labelled zip with a number (skip CVAT job_/test exports).
    # For clips with several zips, keep the one with the MOST ball boxes.
    cand = {}   # n -> (nboxes, zip)
    for z in glob.glob(os.path.join(ZIPS, "*.zip")):
        low = os.path.basename(z).lower()
        if low.startswith(("job_", "test")) or "cvat" in low:
            continue
        m = re.search(r"(\d+)", low)
        if not m:
            continue
        n = int(m.group(1))
        if n in EXISTING:
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                xn = [x for x in zf.namelist() if x.endswith("annotations.xml")][0]
                r = ET.fromstring(zf.read(xn))
            nb = sum(1 for t in r.findall(".//track") for b in t.findall("box") if b.get("outside") != "1")
        except Exception:
            nb = -1
        if n not in cand or nb > cand[n][0]:
            cand[n] = (nb, z)
    new = sorted((n, z) for n, (nb, z) in cand.items())
    print(f"candidate new clips: {[n for n,_ in new]}")

    tiles = []; ok = 0
    for n, z in new:
        stem = f"bowling_{n}"
        vids = glob.glob(f"{MAIN}/bowling_{n}.*")
        vids = [v for v in vids if v.lower().endswith((".mp4", ".mov"))]
        if not vids:
            print(f"  {stem}: NO VIDEO in main_data — skip"); continue
        vp = vids[0]
        exdir = f"{ZIPS}/_extracted/{stem}labelled"
        os.makedirs(exdir, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(exdir)
        xmls = glob.glob(f"{exdir}/**/annotations.xml", recursive=True)
        if not xmls:
            print(f"  {stem}: no annotations.xml in zip — skip"); continue
        boxes, W, H, size = parse_xml(xmls[0])

        cap = cv2.VideoCapture(vp)
        vw_, vh_ = int(cap.get(3)), int(cap.get(4))
        nprop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (vw_, vh_) != (W, H):
            cap.release(); print(f"  {stem}: RESOLUTION MISMATCH xml {W}x{H} vs video {vw_}x{vh_} — SKIP"); continue
        if abs(nprop - size) > max(6, int(0.08 * size)):
            cap.release(); print(f"  {stem}: FRAME-COUNT MISMATCH (xml {size} vs video {nprop}) — SKIP"); continue

        # STREAM frames (4K clips would blow up RAM if fully buffered). Downscale
        # oversized frames to <=1920 long side; labels are normalised so they align.
        scale = min(1.0, 1920.0 / max(vw_, vh_))
        idir = f"{ROOT}/sources/{stem}/images"; ldir = f"{ROOT}/sources/{stem}/labels"
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        fs_all = sorted(boxes)
        sel = set(fs_all[int(k)] for k in np.linspace(0, len(fs_all) - 1, 6)) if fs_all else set()
        tcache = {}
        nw = 0; idx = 0
        while True:
            okf, fr = cap.read()
            if not okf:
                break
            if idx in boxes:
                d = cv2.resize(fr, (int(vw_ * scale), int(vh_ * scale))) if scale < 1.0 else fr
                x1, y1, x2, y2 = boxes[idx]
                cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
                bw = abs(x2 - x1) / W; bh = abs(y2 - y1) / H
                if bw > 0 and bh > 0:
                    cv2.imwrite(f"{idir}/{stem}_f{idx:04d}.jpg", d)
                    with open(f"{ldir}/{stem}_f{idx:04d}.txt", "w") as fh:
                        fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    nw += 1
                if idx in sel:
                    tcache[idx] = d
            idx += 1
        cap.release()
        if nw < 5:
            print(f"  {stem}: only {nw} usable frames — SKIP"); continue
        ok += 1
        print(f"  {stem}: {nw} labeled frames ({W}x{H}{' ->downscaled' if scale<1 else ''})")

        # alignment tile (box coords scaled to the downscaled frame)
        row = []
        for fi in sorted(tcache):
            x1, y1, x2, y2 = (c * scale for c in boxes[fi])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            C = 90; crop = np.zeros((C, C, 3), np.uint8)
            sx, sy = max(cx - C // 2, 0), max(cy - C // 2, 0)
            p = tcache[fi][sy:sy + C, sx:sx + C]; crop[:p.shape[0], :p.shape[1]] = p
            cv2.rectangle(crop, (int(x1 - sx), int(y1 - sy)), (int(x2 - sx), int(y2 - sy)), (0, 255, 0), 1)
            row.append(cv2.resize(crop, (120, 120), interpolation=cv2.INTER_NEAREST))
        if row:
            img = np.hstack(row)
            cv2.putText(img, stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)

    if tiles:
        cv2.imwrite(f"{VERIFY}/alignment_batch2.jpg", np.vstack(tiles))

    # rebuild split across ALL sources
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = f"{ROOT}/{sub}/{split}"; os.makedirs(d, exist_ok=True)
            for f in glob.glob(f"{d}/*"):
                os.remove(f)
    counts = conv.assemble_split()
    nsrc = len(glob.glob(f"{ROOT}/sources/*"))
    print(f"\nintegrated {ok} new clips | split: {counts} | total sources: {nsrc}")
    print(f"alignment: {VERIFY}/alignment_batch2.jpg")


if __name__ == "__main__":
    main()
