# TrackNet ball tracker (cloud-trained)

A heatmap tracker for the tiny/fast/blurred cricket ball. Unlike YOLO (per-frame
appearance), TrackNet takes **3 consecutive frames** and predicts a ball-position
**heatmap**, using the ball's motion streak. This is the right tool for a 4–14px
ball and is expected to beat the YOLO+SAHI pipeline's ~40–60% recall.

Local GPU is 4 GB → **train in the cloud** (Colab free/T4 is plenty).

## Files
- `model.py` — TrackNetV2 (3 frames in → 3 heatmaps out), ~11M params
- `build_dataset.py` — packages the 47 labelled clips into an uploadable dataset
- `dataset.py` — triplet + Gaussian-heatmap Dataset
- `train.py` — training loop with clip-level holdout + honest peak-recall metric
- `infer.py` — video → ball positions → Hawk-Eye tube (reuses existing renderer)

## Workflow

**1. Build the dataset (local, once):**
```
python tracknet/build_dataset.py          # -> tracknet/dataset/  (frames + manifest.csv)
```
Zip and upload `tracknet/dataset/` to Colab (a few hundred MB of small JPGs).

**2. Train on Colab (GPU):**
```
!pip install torch torchvision opencv-python-headless
!python train.py --data dataset --epochs 40 \
    --holdout bowling3,Test13,bowling_5,bowling_18,bowling_47,bowling_99
```
It prints **held-out peak-recall** each epoch (recall on clips it never trained
on) and saves `tracknet_best.pt`. That number is the honest generalization score
to compare against the YOLO pipeline's ~39%.

**3. Download `tracknet_best.pt`, run inference locally:**
```
python tracknet/infer.py --video data/main_data/bowling_88.mov --weights tracknet/tracknet_best.pt
```

## Why this should work where YOLO didn't
- Uses **motion** (3-frame input) → the blurred ball's streak is a strong signal.
- Full-frame heatmap → no per-tile SAHI, one fast forward pass per frame.
- Dense pixel supervision (Gaussian target) → learns tiny balls better than box regression.

## Notes / knobs
- Input is portrait `288×512`; bump to `360×640` in `build_dataset.py`/`infer.py`
  for a bit more resolution on the ball (more GPU).
- More labelled clips still help — but TrackNet needs far fewer than a detector
  to reach usable recall, because motion does the heavy lifting.
