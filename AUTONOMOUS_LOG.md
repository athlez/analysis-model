# Autonomous work log — Stage 4 (ball detection + tracking)

Objective: make the detector reliably follow the REAL cricket ball on our own
bowling videos. User is offline; working autonomously. No fabrication; every
step verified with a real test.

## Environment (established earlier)
- Python 3.14 (system) has NO CUDA torch → built `.venv-gpu` (Python 3.12 +
  torch 2.6.0+cu124 + ultralytics 8.4.90 + albumentations). GPU = RTX 3050, 4 GB.
- All training/inference/labeling runs use `.venv-gpu/Scripts/python.exe`.

## Prior state
- it0 (yolo11n, Roboflow only): val mAP50 0.82 — does NOT track our ball
  (fires on static background).
- it1 (blur aug, warm-start): val mAP50 0.89, recall 0.83 — still does NOT
  track our ball on real footage.
- Proven: Roboflow-only training can't detect our tiny/blurred ball. Motion,
  small-blob, and YOLO all fail on bowling2. Root cause = no in-domain labels +
  behind-the-batsman angle is hard.

## Plan (this session)
1. Build a trajectory-based **auto-labeler**: median-background foreground →
   small ball-sized candidates → search for the fast, smooth, consistent
   tracklet (the ball). Strict quality gate; NEVER fabricate — skip clips with
   no clear ball track.
2. Verify auto-labels visually (render overlays; inspect montages).
3. Build in-domain dataset from verified auto-labels; merge with Roboflow
   (oversample in-domain).
4. Fine-tune YOLO (epochs); evaluate (mAP/P/R/FP/FN).
5. Run on bowling2 + others; overlays; verify ball tracking.
6. Iterate until the box stays on the real ball, or document a genuine blocker.

## Design decisions & rationale
- **Median background** (per fixed camera) over MOG2: the camera is static, so
  the per-pixel median over the clip is the clean background; |frame-median|
  isolates all moving objects (ball + bowler). Robust and simple.
- **Trajectory-consistency** is the ball discriminator: the ball is the small
  blob whose positions across consecutive frames fit a fast, smooth, roughly
  low-order-polynomial path. Body/net clutter jitters locally and fails this.
- Auto-labeling is offline with look-ahead + strict acceptance, which is far
  more tractable than real-time detection.

## Progress

### Dataset built (verified)
`data/yolo_ball/` — 4569 train / 507 val / 249 test, single class `ball`.
Merged: Roboflow detection(bbox) + tracking(bbox) + Ball Model(seg→bbox) +
segmentation v5(seg→bbox) + FullDataset(250 valid). Label QA passed (boxes on
balls; polygon→bbox conversion correct). ultralytics scan: 0 corrupt.

### Auto-labeler built + tested (scripts/autolabel_ball.py)
Approach: median background → small ball-sized motion candidates → RANSAC
constant-velocity trajectory search (robust to clutter, bounded runtime).
- Fixed a runtime blowup (greedy O(C^2) search on frames with 800+ candidates)
  by capping candidates/frame + switching to bounded RANSAC. Verified fast.
- Tested on bowling14 (60fps outdoor): produced a 15-frame "track", BUT zoomed
  crop verification showed it tracked the **bowler's foot/shin** (a smoothly
  moving body part), NOT the ball. Trajectory-smoothness alone is fooled by
  smoothly moving body parts. **Did NOT accept these as labels (would be
  fabrication).**

### Decisive diagnostics — is the ball observable at all?
- bowling2 (30fps): YOLO fires on static bg; motion-diff & small-blob show the
  ball buried in body-motion clutter (11–34 ball-sized blobs/frame on the
  bowler); no separable ball path.
- bowling14 (60fps) windowed candidate-accumulation (blue=early→red=late):
  motion candidates dominated by **wind-blown net/vegetation + bowler**; the
  **turf is empty of ball candidates** — the ball produces no trackable motion
  signature crossing the pitch.
- Tausif 4K whole-clip accumulation: noise-saturated (camera shake makes the
  whole frame register as motion) — not usable as-is; would need windowing +
  stabilization, but the systemic angle problem remains.

## BLOCKER (genuine — needs human input / new data)
**The cricket ball is not reliably observable in the own_recordings footage by
any available method** (trained YOLO mAP50 0.89, frame-diff, small-blob,
trajectory RANSAC, candidate accumulation — on 30fps and 60fps clips).

Root causes (systemic to how the clips were filmed):
1. **Camera directly behind the batsman** → the ball is bowled along the
   optical axis (toward the lens): tiny near the bowler at release, minimal
   lateral image motion, occluded by/against the body, then rushes at the
   camera and blurs out. Hardest possible angle for ball tracking.
2. **30–60 fps + phone rolling shutter** → the fast ball is a faint smear or
   skips between frames; below the motion-detection noise floor over the turf.
3. **Dominant competing motion** (bowler's body, wind-blown net/trees, camera
   shake) swamps the ball's weak signal.

No amount of training/augmentation fixes unobservable data. The
demo/pipeline target (bowling2) in particular contains no observable ball, so
no detector can succeed on it. Autonomous honest labeling is therefore
impossible on this footage; fabricating labels is disallowed.

## Everything tried (chronological)
1. Trained yolo11n on Roboflow (it0) → mAP50 0.82; fails on our footage.
2. Retrained with motion-blur aug (it1) → mAP50 0.89, recall 0.83; still fails
   on our footage (fires on static background at conf 0.15–0.35).
3. Static-FP rejection (#4) → removes background but leaves no ball.
4. Higher-res inference (imgsz 1280) → no ball detected.
5. Median-bg + small-blob motion analysis → ball buried in body clutter.
6. RANSAC trajectory auto-labeler → locks onto smoothly-moving foot, not ball.
7. Candidate-accumulation visualization → turf empty of ball signal.

## Constructive work delivered despite blocker (see HANDOFF.md)
- Finalized trained detector `models/ball_detector.pt` (yolo11n + blur aug;
  val P 0.944 / R 0.829 / mAP50 0.89 / mAP50-95 0.554).
- Wired the detector into the pipeline (`_default_ball_detector` in
  pipeline/pipeline.py): auto-loads the weights when ultralytics is present,
  else motion fallback. Verified it constructs + loads YOLO. Downstream frozen.
- Final end-to-end verification on bowling2 (`output/demo_final/`): with the
  trained detector wired in, YOLO still finds no ball in the delivery window →
  motion fallback → 4.6 km/h (unchanged). Confirms the blocker end-to-end.
- Ready-to-run labeling + fine-tuning workflow (scripts/autolabel_ball.py,
  train_ball_it1.py, build_ball_dataset.py) documented in HANDOFF.md.
- Filming spec (side-on + 120fps + fast shutter) in HANDOFF.md.

## Decision: stopped further Roboflow-only training
Reasoning (documented per instruction): more epochs on Roboflow-only data
cannot make the detector track a ball that is unobservable in our footage
(proven). The finalized detector already plateaued at mAP50 0.89. Burning ~1h
GPU on a redundant complete run had no path to improving the actual objective,
so it was stopped. Productive epoch-work resumes the moment in-domain labels
from suitable footage exist (workflow ready).

## Batch overlay run on ALL own_recordings (for review)
Ran the full pipeline (trained detector wired in) on all 15 clips ->
`output/demo_all/<clip>_output.mp4` (+ _report.json, _summary.json). Driver:
`scripts/run_all_demos.py` (run with PYTHONPATH=project root).

Honest result across all 15: `det_sources` are entirely **motion/interpolated**
— the trained YOLO detected NO ball in any clip's delivery window, so every clip
fell back to the motion detector. Consequences visible in the videos:
- boxes sit on body/background motion, not the ball;
- speeds are physically impossible (e.g. bowling7 691, bowling13 1202,
  bowling14 2838 km/h) because the "track" is body/artifact motion fed through
  the (separately degenerate) calibration.
This demonstrates the BLOCKER across the entire own_recordings set, not just
bowling2. "tracking=True" in the summary only means a track existed, NOT that it
is the ball. Videos are for visual review of the current (blocked) state.

## UPDATE — in-domain labels received (partial unblock)
User provided CVAT-for-Video label ZIPs for 3 clips (bowling1/2/3). Built
`scripts/convert_cvat_to_yolo.py` → clean, validated, extensible YOLO dataset at
`data/ball_dataset/` (**291 ball boxes**; 246 train / 45 val; 0 validation
issues; median box ~7 px = genuine ball). Frame alignment visually verified
(boxes land on the ball). Extensible: drop more `<name>labelled.zip` in
`data/raw/labelled_videos/` + re-run the converter (auto-discovers, no
restructuring). See `data/ball_dataset/README.md`.

NOTE: bowling2 IS labeled frames 30–108 — i.e. a human COULD see/track the ball
(with interpolation) even though motion/auto methods could not. This means
fine-tuning on these in-domain labels is now viable. Training NOT started
(per instruction). Next: merge with Roboflow (oversample in-domain) + fine-tune
from models/ball_detector.pt, then verify on held-out clips.

## STATUS: in-domain dataset prepared & validated; ready to fine-tune on request.


