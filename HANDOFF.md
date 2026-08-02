# Stage 4 (ball detection) — Handoff

Autonomous session summary. Read `AUTONOMOUS_LOG.md` for the full chronological
log. This file is the "start here when you return" guide.

## TL;DR
- Built the full training + auto-labeling + evaluation + overlay tooling and a
  GPU environment. Trained a cricket-ball YOLO (benchmark **mAP50 ≈ 0.89**).
- **Genuine blocker reached:** the ball is **not observable** in the current
  own_recordings / Pune footage — verified across 30 fps, 60 fps and **4K**
  clips with 6 independent methods. No detector can be trained to track a ball
  that leaves no signal in the video. The cause is the **camera angle**
  (filmed directly behind the batsman → ball travels along the optical axis)
  plus frame-rate/blur. This needs **new footage**, which requires you.
- I did **not** fabricate labels (the auto-labeler locked onto the bowler's
  foot; those were rejected, not used).

## What works right now
- `.venv-gpu` (Python 3.12 + torch 2.6 cu124 + ultralytics 8.4.90 + albumentations),
  GPU (RTX 3050) training verified.
- `data/yolo_ball/` — unified Roboflow+FullDataset detection dataset
  (4569/507/249), single class `ball`, labels QA-verified.
- Trained weights: `runs/detect/runs/ball/it1_blur/weights/best.pt` (and
  `it1_full` when its run finishes) → finalized copy at `models/ball_detector.pt`.
- Trained detector is **wired into the pipeline** (`_default_ball_detector` in
  `pipeline/pipeline.py`): it auto-loads `models/ball_detector.pt` when
  ultralytics is present, else falls back to motion-only. Downstream
  (calibration/projection/trajectory/physics) untouched.

## The blocker in one paragraph
Every clip is filmed from **directly behind the batsman**, so the ball is
bowled **toward the lens**: tiny near the bowler at release, minimal lateral
image motion, occluded against the body, then it rushes at the camera and blurs
out. At 30–60 fps (and even 4K) the ball produces **no reliable motion or
appearance signal** over the pitch — motion is dominated by the bowler, the
wind-blown net, and camera shake. Proven with: trained YOLO (fires on static
background), frame-differencing, small-blob analysis, RANSAC trajectory search
(locked onto a foot), candidate accumulation (turf empty of ball), and full-res
4K flight diffs (black). See `AUTONOMOUS_LOG.md` for the evidence images.

## FIX: how to capture footage the ball IS visible in (do this first)
Film a handful (10–20) of deliveries with:
1. **Side-on camera** — square-leg or mid-off, ~10–15 m to the side of the
   pitch, so the ball travels **across the frame** (large lateral motion),
   not toward the lens. This single change is the biggest fix.
2. **High frame rate** — 120 fps (or 240 fps slow-mo) on the phone.
3. **High shutter / good light** — fast shutter to reduce motion blur; bright
   daylight. The ball should look like a crisp dot, not a smear.
4. Keep the camera **fixed** (tripod). Fill the frame with the pitch.
5. A **contrasting ball** vs background helps (red ball on green, or white ball
   in daylight).

## Continue in ONE workflow when you have good footage
Put new clips in `data/raw/new_side_on/`. Then (all with `.venv-gpu`):

```bash
# 1. Auto-label the ball in each clip (trajectory-based). ALWAYS verify the QA
#    video before trusting labels — reject clips where the box isn't on the ball.
for v in data/raw/new_side_on/*.mp4; do
  .venv-gpu/Scripts/python.exe scripts/autolabel_ball.py --video "$v" --qa \
      --min_speed 12 --min_len 6
done
#    -> writes data/ball_autolabel/{images,labels}, QA videos in .../qa/

# 2. VERIFY (open the qa/*.mp4). Delete any clip's labels that are wrong.

# 3. Merge verified in-domain labels into the training set (oversample x3),
#    then fine-tune from the current detector:
.venv-gpu/Scripts/python.exe scripts/train_ball_it1.py \
    --init models/ball_detector.pt --name it2_indomain --epochs 40

# 4. Verify on a real clip (overlay video):
.venv-gpu/Scripts/python.exe scripts/detect_overlay.py \
    --weights runs/detect/runs/ball/it2_indomain/weights/best.pt \
    --video data/raw/new_side_on/<clip>.mp4 --conf 0.2 --imgsz 1280
#    Success = the green box stays on the real ball for most of the delivery.

# 5. Run the full pipeline demo (uses the trained detector automatically):
.venv-gpu/Scripts/python.exe -m validation.demo data/raw/new_side_on/<clip>.mp4 \
    --weights models/ball_detector.pt
```
(Step 3's merge/oversample of `data/ball_autolabel` into `data/yolo_ball` is a
small addition to `scripts/build_ball_dataset.py` — a `--include-autolabel`
branch; noted as the only remaining glue to add once real labels exist.)

## Key scripts
- `scripts/build_ball_dataset.py` — build/merge the YOLO dataset.
- `scripts/train_ball.py` / `scripts/train_ball_it1.py` — train / fine-tune
  (it1 adds motion-blur augmentation).
- `scripts/autolabel_ball.py` — trajectory-based ball auto-labeler (+ QA video).
- `scripts/detect_overlay.py` / `detect_overlay_v2.py` — inference + tracking
  overlay (v2 adds static-FP rejection).
- `validation/demo.py` — full-pipeline overlay (`--weights` to plug the detector).

## Do NOT
- Do not trust auto-labels without watching the QA video (it can lock onto feet).
- Do not train on the current behind-the-batsman footage expecting ball
  tracking — it cannot work (no ball signal). New side-on footage is required.
