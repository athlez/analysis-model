"""Ground-truth pipeline verification — NO YOLO, NO converted labels.

Reads the raw CVAT-for-Video XML directly and proves each stage in isolation:

  1. PARSE      : XML boxes -> per-frame (xtl,ytl,xbr,ybr)
  2. ALIGN      : draw each box on its frame; zoomed-crop montage shows whether
                  the ball actually sits inside the box, frame by frame
  3. FIT        : centres -> smooth image-space trajectory
  4. RENDER     : trajectory tube over the video

Emits diagnostics per stage so any error can be localised.

    python scripts/verify_gt_pipeline.py --stem bowling3
"""
import argparse, glob, os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

import scripts.hawkeye_tube as ht


def parse_cvat(xml_path):
    root = ET.parse(xml_path).getroot()
    meta_w = int(root.findtext(".//original_size/width"))
    meta_h = int(root.findtext(".//original_size/height"))
    size = int(root.findtext(".//size") or 0)
    boxes, outside = {}, 0
    for track in root.findall("track"):
        for b in track.findall("box"):
            fi = int(b.get("frame"))
            if b.get("outside") == "1":
                outside += 1
                continue
            boxes[fi] = (float(b.get("xtl")), float(b.get("ytl")),
                         float(b.get("xbr")), float(b.get("ybr")))
    return boxes, meta_w, meta_h, size, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="bowling3")
    ap.add_argument("--out", default="output/gt_verify")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    xml = f"data/raw/labelled_videos/_extracted/{args.stem}labelled/annotations.xml"
    vid = glob.glob(f"data/raw/own_recordings/{args.stem}.*")[0]

    # ---- STAGE 1: PARSE -------------------------------------------------- #
    boxes, mw, mh, size, n_outside = parse_cvat(xml)
    fs = sorted(boxes)
    print("== STAGE 1 PARSE ==")
    print(f"  boxes parsed: {len(boxes)} (+{n_outside} 'outside' skipped)")
    print(f"  frame range: {fs[0]}..{fs[-1]}   CVAT size: {size}")
    print(f"  CVAT resolution: {mw}x{mh}")

    # ---- load video, check alignment prerequisites ----------------------- #
    cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    print("== FRAME ALIGNMENT PREREQS ==")
    print(f"  video resolution: {W}x{H}   (CVAT {mw}x{mh}) -> {'OK' if (W,H)==(mw,mh) else 'MISMATCH!'}")
    print(f"  decoded frames: {len(frames)}   CVAT size: {size} -> {'OK' if len(frames)==size else 'MISMATCH!'}")
    print(f"  max label frame {fs[-1]} < decoded {len(frames)} -> {'OK' if fs[-1] < len(frames) else 'OUT OF RANGE!'}")

    # ---- STAGE 2: ALIGN — zoomed crop montage ---------------------------- #
    sel = [fs[int(k)] for k in np.linspace(0, len(fs) - 1, 15)]
    tiles = []
    C = 110
    for fi in sel:
        x1, y1, x2, y2 = boxes[fi]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cropped = np.zeros((C, C, 3), np.uint8)
        sx, sy = max(cx - C // 2, 0), max(cy - C // 2, 0)
        patch = frames[fi][sy:sy + C, sx:sx + C]
        cropped[:patch.shape[0], :patch.shape[1]] = patch
        # box drawn in crop coords
        cv2.rectangle(cropped, (int(x1 - sx), int(y1 - sy)), (int(x2 - sx), int(y2 - sy)), (0, 255, 0), 1)
        cropped = cv2.resize(cropped, (150, 150), interpolation=cv2.INTER_NEAREST)
        cv2.putText(cropped, f"f{fi}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(cropped)
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, 15, 5)]
    cv2.imwrite(os.path.join(args.out, f"{args.stem}_ALIGN.jpg"), np.vstack(rows))
    print("== STAGE 2 ALIGN ==")
    print(f"  wrote {args.stem}_ALIGN.jpg (zoomed crops; ball should sit inside each green box)")

    # full overlay video with the raw box on every labelled frame
    vw = cv2.VideoWriter(os.path.join(args.out, f"{args.stem}_boxes.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, fr in enumerate(frames):
        img = fr.copy()
        if fi in boxes:
            x1, y1, x2, y2 = boxes[fi]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(img, f"{args.stem} f{fi}  GT box", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    # ---- STAGE 3: FIT ---------------------------------------------------- #
    pts = [(fi, (boxes[fi][0] + boxes[fi][2]) / 2, (boxes[fi][1] + boxes[fi][3]) / 2) for fi in fs]
    grid, xs, ys = ht.faithful_centerline(pts, win=3)   # follow the GT labels
    # measure fit fidelity to the raw centres
    dev = [min(np.hypot(px - fx, py - fy) for fx, fy in zip(xs, ys)) for _, px, py in pts]
    print("== STAGE 3 FIT (faithful) ==")
    print(f"  centre points: {len(pts)}")
    print(f"  centreline frames {int(grid[0])}..{int(grid[-1])}")
    print(f"  fit-vs-raw deviation: mean {np.mean(dev):.1f}px  max {np.max(dev):.1f}px")

    # ---- STAGE 4: RENDER ------------------------------------------------- #
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].copy()
    ht.draw_tube(still, xs, ys, radius=5, alpha=0.5)
    cv2.imwrite(os.path.join(args.out, f"{args.stem}_TRAJ.jpg"), still)
    vw = cv2.VideoWriter(os.path.join(args.out, f"{args.stem}_trajectory.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.copy()
        upto = int(np.searchsorted(grid, i)) if i >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=5, alpha=0.5)
        vw.write(img)
    vw.release()
    print("== STAGE 4 RENDER ==")
    print(f"  wrote {args.stem}_TRAJ.jpg + {args.stem}_trajectory.mp4")


if __name__ == "__main__":
    main()
