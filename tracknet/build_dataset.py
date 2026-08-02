"""Package the labelled clips into a TrackNet training set you can upload to a
cloud GPU (Colab).

For every labelled clip it reads the SOURCE VIDEO (TrackNet needs the real
consecutive frames, not just the sparse labelled ones), resizes each frame in
the labelled flight range to the network input size, and writes:

    tracknet/dataset/frames/<clip>/<idx>.jpg   (resized frames, small)
    tracknet/dataset/manifest.csv              (clip, idx, x_norm, y_norm, visible)

Heatmap targets are generated on-the-fly during training from the manifest, so
the package stays small. Zip tracknet/dataset/ and upload it to Colab.
"""
import csv, glob, os, sys
import cv2

sys.path.insert(0, os.path.abspath("."))
import scripts.gt_trajectories as G

SOURCES = "data/ball_dataset/sources"
OUT = "tracknet/dataset"
IN_W, IN_H = 288, 512          # portrait network input (W x H), /8 friendly
CONTEXT = 2                    # extra frames before the first label (for triplets)


def labels_for(stem):
    """frame_idx -> (cx_norm, cy_norm) from the YOLO source labels."""
    d = {}
    for lp in glob.glob(f"{SOURCES}/{stem}/labels/*.txt"):
        try:
            fi = int(os.path.basename(lp).split("_f")[1][:4])
        except (IndexError, ValueError):
            continue
        p = open(lp).read().split()
        if len(p) >= 5:
            d[fi] = (float(p[1]), float(p[2]))   # already normalised 0..1
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    clips = sorted(os.listdir(SOURCES))
    for stem in clips:
        lab = labels_for(stem)
        if len(lab) < 5:
            continue
        vp = G.find_video(stem)
        if not vp:
            print(f"{stem}: no video, skip"); continue
        lo, hi = min(lab) - CONTEXT, max(lab)
        fdir = os.path.join(OUT, "frames", stem)
        os.makedirs(fdir, exist_ok=True)
        cap = cv2.VideoCapture(vp)
        idx = 0; n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if lo <= idx <= hi:
                fr = cv2.resize(fr, (IN_W, IN_H))
                cv2.imwrite(os.path.join(fdir, f"{idx:05d}.jpg"), fr)
                x, y = lab.get(idx, (-1.0, -1.0))
                rows.append((stem, idx, f"{x:.6f}", f"{y:.6f}", int(idx in lab)))
                n += 1
            idx += 1
            if idx > hi:
                break
        cap.release()
        print(f"{stem}: {n} frames ({sum(1 for f in lab)} labelled)")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "idx", "x_norm", "y_norm", "visible"])
        w.writerows(rows)
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"input_w={IN_W}\ninput_h={IN_H}\nn_frames=3\nclips={len(clips)}\nrows={len(rows)}\n")
    print(f"\n{len(rows)} frames from {len(clips)} clips -> {OUT}/  (zip and upload to Colab)")


if __name__ == "__main__":
    main()
