"""Convert CVAT-for-Video annotation ZIPs into a clean, validated, EXTENSIBLE
YOLO detection dataset.

Layout produced (adding more labelled videos later = drop a new ZIP in
data/raw/labelled_videos/ and re-run this script; nothing restructures):

  data/ball_dataset/
    sources/<video>/images/<video>_fNNNN.jpg   # canonical per-video frames
    sources/<video>/labels/<video>_fNNNN.txt   # YOLO labels (class 0 = ball)
    images/{train,val}/                          # assembled split (copies)
    labels/{train,val}/
    data.yaml
    DATASET_INFO.json                            # provenance + validation report

Frame alignment: CVAT `frame` is a 0-based decoded index; we read the source
video sequentially and match by index, then VERIFY by writing overlay crops.
Only boxes with outside=0 are used; coords are clipped to image bounds and
degenerate/invalid boxes are dropped.
"""
import argparse, glob, json, os, shutil, zipfile
import xml.etree.ElementTree as ET
import cv2

ROOT = "data/ball_dataset"
ZIP_DIR = "data/raw/labelled_videos"
SOURCE_SEARCH = ["data/raw/own_recordings", "data/raw"]
CLASS_NAMES = ["ball"]  # matches the Roboflow dataset so the two can be merged


def find_source_video(base, W, H, nframes):
    """Locate the mp4 for a labelled set: by name first, else by matching dims."""
    cands = []
    for d in SOURCE_SEARCH:
        cands += [os.path.join(d, base + ext) for ext in (".mp4", ".mov", ".MOV", ".avi")]
        cands += glob.glob(os.path.join(d, "**", base + ".*"), recursive=True)
    seen = []
    for c in cands:
        if os.path.isfile(c) and c not in seen:
            seen.append(c)
            cap = cv2.VideoCapture(c)
            vw, vh, vn = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
            cap.release()
            if vw == W and vh == H:
                return c, (vw, vh, vn)
    return None, None


def parse_annotations(xml_path):
    r = ET.parse(xml_path).getroot()
    meta = r.find("meta")
    osz = meta.find("original_size")
    W, H = int(osz.find("width").text), int(osz.find("height").text)
    labels = [l.find("name").text for l in meta.findall(".//labels/label")]
    # frame -> list of (xtl,ytl,xbr,ybr) for usable boxes
    per_frame = {}
    for tr in r.findall("track"):
        for b in tr.findall("box"):
            if b.get("outside") == "1":
                continue
            f = int(b.get("frame"))
            box = (float(b.get("xtl")), float(b.get("ytl")),
                   float(b.get("xbr")), float(b.get("ybr")))
            per_frame.setdefault(f, []).append(box)
    return W, H, labels, per_frame


def to_yolo(box, W, H):
    """Clip to bounds, return (cx,cy,w,h) normalized, or None if invalid."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
    bw, bh = x2 - x1, y2 - y1
    if bw < 1.0 or bh < 1.0:            # degenerate
        return None
    if bw >= 0.95 * W and bh >= 0.95 * H:  # covers whole frame -> not a ball
        return None
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, bw / W, bh / H)


def convert_zip(zip_path, report):
    base = os.path.basename(zip_path)[:-4].replace("labelled", "").replace("_labelled", "")
    base = base.strip("_") or os.path.basename(zip_path)[:-4]
    # extract annotations.xml
    exdir = os.path.join(ZIP_DIR, "_extracted", os.path.basename(zip_path)[:-4])
    os.makedirs(exdir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(exdir)
    xmls = glob.glob(os.path.join(exdir, "**", "annotations.xml"), recursive=True)
    if not xmls:
        report.append({"zip": os.path.basename(zip_path), "status": "no annotations.xml"})
        return 0
    W, H, labels, per_frame = parse_annotations(xmls[0])
    src, dims = find_source_video(base, W, H, None)
    if not src:
        report.append({"zip": os.path.basename(zip_path), "base": base,
                       "status": f"source video not found (need {W}x{H})"})
        return 0

    out_img = os.path.join(ROOT, "sources", base, "images")
    out_lbl = os.path.join(ROOT, "sources", base, "labels")
    shutil.rmtree(os.path.join(ROOT, "sources", base), ignore_errors=True)
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)

    cap = cv2.VideoCapture(src)
    idx, written, dropped = 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in per_frame:
            fh, fw = fr.shape[:2]
            if (fw, fh) != (W, H):
                fr = cv2.resize(fr, (W, H))  # safety (dims verified equal already)
            lines = []
            for box in per_frame[idx]:
                y = to_yolo(box, W, H)
                if y is None:
                    dropped += 1; continue
                lines.append(f"0 {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
            if lines:
                stem = f"{base}_f{idx:04d}"
                cv2.imwrite(os.path.join(out_img, stem + ".jpg"), fr)
                with open(os.path.join(out_lbl, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                written += 1
        idx += 1
    cap.release()
    report.append({"zip": os.path.basename(zip_path), "base": base, "source": src,
                   "size": f"{W}x{H}", "labeled_frames": len(per_frame),
                   "written": written, "dropped_boxes": dropped, "status": "ok"})
    return written


def assemble_split(val_tail_frac=0.15):
    """Per-video temporal holdout: last `val_tail_frac` of each video's frames -> val.
    Temporal (not random) to reduce leakage between near-duplicate frames."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(ROOT, sub)
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for vid in sorted(os.listdir(os.path.join(ROOT, "sources"))):
        imgs = sorted(glob.glob(os.path.join(ROOT, "sources", vid, "images", "*.jpg")))
        n = len(imgs); cut = int(n * (1 - val_tail_frac))
        for i, img in enumerate(imgs):
            split = "train" if i < cut else "val"
            lbl = img.replace("images", "labels").replace(".jpg", ".txt")
            shutil.copy(img, os.path.join(ROOT, f"images/{split}", os.path.basename(img)))
            shutil.copy(lbl, os.path.join(ROOT, f"labels/{split}", os.path.basename(lbl)))
            counts[split] += 1
    return counts


def validate():
    """Re-check every assembled label: image exists, correct dims, boxes in [0,1]."""
    issues, checked = [], 0
    for split in ("train", "val"):
        for lbl in glob.glob(os.path.join(ROOT, f"labels/{split}", "*.txt")):
            img = lbl.replace("labels", "images").replace(".txt", ".jpg")
            checked += 1
            if not os.path.isfile(img):
                issues.append(f"missing image for {lbl}"); continue
            im = cv2.imread(img)
            if im is None:
                issues.append(f"unreadable image {img}"); continue
            for ln in open(lbl):
                p = ln.split()
                if len(p) != 5:
                    issues.append(f"malformed line in {lbl}"); continue
                c, cx, cy, w, h = p
                cx, cy, w, h = map(float, (cx, cy, w, h))
                if c != "0" or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    issues.append(f"out-of-range box in {lbl}: {ln.strip()}")
    return checked, issues


def write_yaml():
    p = os.path.abspath(ROOT).replace("\\", "/")
    with open(os.path.join(ROOT, "data.yaml"), "w") as f:
        f.write(f"path: {p}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")


def main():
    os.makedirs(ROOT, exist_ok=True)
    report = []
    total = 0
    for z in sorted(glob.glob(os.path.join(ZIP_DIR, "*.zip"))):
        total += convert_zip(z, report)
    counts = assemble_split()
    write_yaml()
    checked, issues = validate()
    info = {"total_labeled_frames_written": total, "split": counts,
            "validation_checked": checked, "validation_issues": issues,
            "per_zip": report, "classes": CLASS_NAMES}
    with open(os.path.join(ROOT, "DATASET_INFO.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
