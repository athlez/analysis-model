"""Detector-based trajectories on chosen UNSEEN (unlabelled) clips using the v2
detector. Model loaded once; SAHI every Nth frame (fit interpolates). Renders
the Hawk-Eye tube to output/unseen_trajectories/.
"""
import sys, glob, os
import cv2
import numpy as np
import scripts.hawkeye_tube as ht

OUT = "output/unseen_trajectories"
WEIGHTS = "models/ball_detector_tiled.pt"   # = v2 now
STRIDE = 3
NUMS = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    ["113", "118", "125", "133", "88", "95", "105", "12", "34"]

os.makedirs(OUT, exist_ok=True)
model = ht.load_sahi_model(WEIGHTS, 0.25)

for n in NUMS:
    vids = glob.glob(f"data/main_data/bowling_{n}.*")
    vids = [v for v in vids if v.lower().endswith((".mp4", ".mov"))]
    if not vids:
        print(f"bowling_{n}: no video", flush=True); continue
    stem = f"bowling_{n}"
    cap = cv2.VideoCapture(vids[0]); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ow, oh = int(cap.get(3)), int(cap.get(4))
    # downscale 4K so the longer side <= 1920 (fast SAHI + sane RAM); ball stays detectable
    scale = min(1.0, 1920.0 / max(ow, oh))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale < 1.0:
            fr = cv2.resize(fr, (int(ow * scale), int(oh * scale)))
        frames.append(fr)
    cap.release()
    W, H = frames[0].shape[1], frames[0].shape[0]
    pts = ht.detect_centers_sahi(frames, slice_size=384, conf=0.25, model=model, stride=STRIDE)
    if len(pts) < 6:
        print(f"{stem}: only {len(pts)} detections -> skip", flush=True); continue
    grid, xs, ys = ht.build_centerline(pts)
    f0, f1 = int(grid[0]), int(grid[-1])
    still = frames[min(f1, len(frames) - 1)].copy()
    ht.draw_tube(still, xs, ys, radius=5, alpha=0.55)
    cv2.imwrite(f"{OUT}/{stem}_hawkeye.jpg", still)
    vw = cv2.VideoWriter(f"{OUT}/{stem}_hawkeye.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, fr in enumerate(frames):
        img = fr.copy()
        upto = int(np.searchsorted(grid, i)) if i >= f0 else 0
        if upto >= 2:
            ht.draw_tube(img, xs, ys, upto_idx=upto, radius=5, alpha=0.55)
        vw.write(img)
    vw.release()
    print(f"{stem}: {len(pts)} detections, tube frames {f0}-{f1} -> rendered", flush=True)

print("DONE", flush=True)
