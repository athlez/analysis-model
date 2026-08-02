# Design Proposal — MediaPipe Pose for Bowling Release Detection

**Status:** research / design only. No code written, nothing implemented.
**Question:** Can MediaPipe Pose estimate the release frame and gate the ball
detector to improve reliability?

---

## TL;DR / Recommendation

**Conditional YES — adopt it as a *scoped, separate module*, but not for every
clip and not as a fix for our core recall problem.**

- Pose-based release detection is **viable and valuable on footage where the
  bowler is large in frame** — i.e. the **behind/beside-the-bowler clips**
  (`Test*.mov`, `bowling16–23`, and any future side-on capture). There it can
  replace the unreliable motion-energy segmentation and **sharply cut false
  positives**.
- It is **not viable on the behind-the-batsman clips** (`bowling1/2/3`), because
  the bowler is a tiny distant figure — MediaPipe Pose needs the person to
  occupy a meaningful fraction of the frame.
- It **improves precision, not recall.** It tells the ball detector *when/where*
  to look; the ball must still be detectable there. So it should be sequenced
  **after** the yolo11m in-domain fine-tune, as a precision/efficiency layer.
- Build it as a dedicated `ReleaseDetector` module feeding the existing
  splitter/detector seam — **do not mix it into the detector**.

---

## 1. Grounding: our footage decides everything

MediaPipe Pose (BlazePose) tracks a single prominent person and needs that
person to be reasonably large and unoccluded. Our clips fall into two camps:

| Setup | Bowler in frame | Pose viable? |
|---|---|---|
| Behind batsman (`bowling1/2/3`) | far, tiny (<15% frame height) | **No** — too small, jittery/absent landmarks |
| Behind/beside bowler (`Test*.mov`, `bowling16–23`) | large, foreground | **Yes** — this is where pose works |

This is the single most important finding: **pose gating helps exactly the
setups we are moving toward (side/behind-bowler), and not the hardest legacy
setup.** Conveniently, the side-on clips are also the better angle for ball
tracking, so the two improvements compound.

---

## 2. Answers to the eight questions

### Q1. Can MediaPipe consistently detect the bowling arm throughout the action?
- **On foreground-bowler clips: mostly yes, with one critical gap.** Landmarks
  are stable through run-up, load-up, delivery stride and follow-through.
- **The gap is the release instant itself:** the bowling arm moves fastest
  exactly at arm-over, producing motion blur → the wrist/elbow can **jitter or
  drop out for 1–3 frames right where we need them.** Visibility scores fall.
  Mitigation: smooth + interpolate across the blurred frames; rely on the
  *arc/velocity profile* rather than a single frame.
- **On behind-batsman clips: no** — the bowler is too small/distant.
- **Multi-person caveat:** MediaPipe Pose is single-person by default. With a
  batsman (and sometimes others) in frame it may lock onto the wrong person.
  Requires a person-selection step (pick the bowler: the largest moving figure
  near the bowling crease, or a person detector + crop).

### Q2. Which landmarks are most useful?
Ranked by usefulness for *release*:
1. **Wrist (15/16)** — primary. Its trajectory peak and speed define release.
2. **Elbow (13/14)** — arm-extension / shoulder–elbow–wrist angle; near-straight
   at delivery (also the landmark used for the 15° elbow-flex legality rule).
3. **Shoulder (11/12)** — reference to measure arm-over rotation (the
   shoulder→wrist vector sweeping from behind the body to over the top).
4. **Hip (23/24)** — body orientation, delivery-stride/front-foot timing, and to
   **normalise scale** (shoulder–hip distance) so thresholds are camera-distance
   independent. Indirect for the exact instant, useful for the action window.

The **bowling arm is identified dynamically** as the arm whose wrist has the
largest vertical sweep / highest peak over the delivery window (handles
left/right-arm, over/round the wicket automatically).

### Q3. Best signal combination for the release frame?
Release ≈ the moment the ball leaves the hand, which is **at or 1–2 frames after
the top of the arm arc**, coinciding with **peak wrist speed** and a
**near-extended elbow**. Robust heuristic:

```
1. Select bowling arm (max vertical wrist sweep in the window).
2. Track & smooth wrist (x,y) [+ z if using world landmarks].
3. arm_over_angle = angle(shoulder→wrist) vs vertical; find the frame the arm
   crosses vertical moving forward  → "top of arc" (t_top).
4. wrist_speed = |Δwrist|/Δt (smoothed); find its local max near t_top.
5. elbow_extension = shoulder–elbow–wrist angle (gate: near-straight).
6. release_frame ≈ argmax(wrist_speed) in [t_top, t_top+3],
   gated by elbow_extension high and wrist near its height peak.
```
In practice **release ≈ frame of peak wrist speed just after top-of-arc.** For
slower actions (wrist-spin / "chinaman", as in `Test6`) the whip is weaker, so
fall back to `top-of-arc` (highest wrist) as the release proxy.

### Q4. How many frames before/after release should YOLO run?
The ball is at the hand at release and reaches the batsman in ~0.4–0.6 s.
- **Before:** pad for release uncertainty → **~0.15 s before** (≈5 frames @30fps,
  ≈9 @60fps).
- **After:** cover the full flight → **~0.6–0.7 s after** (≈18–21 frames @30fps,
  ≈36–42 @60fps).
- **Rule of thumb:** run YOLO on `[release − 0.15 s, release + 0.7 s]`. At 30 fps
  ≈ `[r−5, r+20]`; at 60 fps ≈ `[r−9, r+42]`.

### Q5. Would limiting YOLO to those frames reduce false positives?
**Yes — strongly. This is the biggest win.** Our documented false positives came
from static background (net, stumps, spectators, sightscreen) firing across the
*whole* clip. Restricting inference to the ~0.5–0.85 s release+flight window
removes FPs during run-up, follow-through, idle, and between deliveries, and cuts
compute ~5–10×. Precision improvement is the clearest, most reliable benefit of
this pipeline.

### Q6. Would a Region-of-Interest around the bowling arm after release help?
**Partially — only for the first few frames.**
- At release the ball is beside the wrist → a **ROI around the wrist for the
  first ~2–3 frames** shrinks the search area (fewer FPs) and lets us **upscale /
  tile (SAHI-style)** the ROI so the tiny ball gets more pixels → **recall
  boost** at the hardest moment.
- But the ball leaves the hand fast and separates from the arm within a few
  frames. After that the ROI must **follow the predicted ball path** (expanding
  Kalman/constant-velocity gate), **not** the arm. So: wrist-ROI at release →
  hand off to a motion-predicted ROI → full-frame fallback.
- Net: useful as a *release-instant* recall aid, cannot stay tied to the arm.

### Q7. Limitations
- **Angle/size dependent** — fails on behind-batsman footage (the hardest, and
  where the ball is also hardest). Only helps foreground-bowler clips.
- **Motion blur at release** → landmark jitter/dropout at the exact frame → release
  uncertainty ±1–3 frames (see Q8 accuracy).
- **Multi-person** — must select the bowler (extra logic / person detector).
- **Action variety** — pace vs spin vs chinaman, over/round the wicket, left/right
  arm all have different arm paths; heuristic must be robust (dynamic arm
  selection + fallbacks).
- **Frame rate** — 30 fps is coarse (release spans 1–2 frames) → inherent ±1–2
  frame floor; 60/120 fps materially better.
- **It gates, it doesn't detect** — pose tells us *when/where*; the ball detector
  must still have recall in-window. **Pose cannot fix low recall**, only remove
  out-of-window FPs.
- **Extra dependency & failure mode** — adds `mediapipe`, CPU cost, and another
  component that can mis-fire (e.g. no bowler found → no window → whole delivery
  missed). Needs a graceful fallback to the current motion segmenter.
- **Does not address calibration / metric accuracy** at all.

### Q8. How do commercial / academic systems combine pose + object detection?
- **Broadcast / Hawk-Eye:** rely on **multiple calibrated high-speed cameras +
  dedicated ball tracking + triangulation**, *not* pose-gated single-view
  detection. Pose isn't their release mechanism.
- **Biomechanics / legality analysis:** markerless pose (OpenPose / BlazePose /
  vendor systems) is used to measure the **elbow-extension (15°) rule** and to
  segment the **bowling action phases** (run-up → bound → delivery stride →
  release → follow-through) — i.e. pose-for-events is an established use.
- **Academic sports-ball tracking:** the dominant approach for *tiny fast balls*
  is **TrackNet-style temporal CNNs** (stack of 3 consecutive frames → heatmap of
  ball position), which materially outperform single-frame YOLO for this exact
  problem (tennis/badminton/cricket). Several works **use pose/event detection to
  trigger or constrain** the ball detector.
- **Takeaway:** the proposed pattern (pose → temporal gating → ball detector) is
  a recognised, sensible design. Two refinements worth noting: (a) **TrackNet is a
  stronger detector than YOLO** for our tiny ball and worth evaluating; (b) pose
  is best used for **windowing/phase-segmentation**, exactly as proposed.

---

## 3. Evaluation of the proposed pipeline

Proposed:
```
Video → MediaPipe Pose → Release Frame → YOLO (around release) → Tracking → Trajectory
```

**Verdict: sound, with refinements.** Issues to address:
- Pose only works on foreground-bowler clips → need a **fallback path** (current
  motion segmenter) when no reliable bowler pose is found, so the pipeline
  degrades gracefully instead of dropping the delivery.
- Add a **person-selection** step before pose (pick the bowler).
- The window should be **release-anchored but flight-length** (Q4), not a fixed
  symmetric window.
- ROI should be **wrist at release → motion-predicted after** (Q6), not arm-locked.

Refined pipeline:
```
                         ┌─────────────────────────────────────────┐
   Video ───────────────►│  Person select (pick bowler)            │
                         └───────────────┬─────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────┐
                         │  MediaPipe Pose (bowler crop)            │
                         │  wrist / elbow / shoulder / hip tracks   │
                         └───────────────┬─────────────────────────┘
                          pose reliable? │ yes            │ no / bowler too small
                                         ▼                ▼
                         ┌───────────────────────┐   ┌─────────────────────────┐
                         │ Release-frame estimate │   │ FALLBACK: existing      │
                         │ (arc + wrist-speed +   │   │ motion-energy segmenter │
                         │  elbow-extension)      │   └───────────┬─────────────┘
                         └───────────┬───────────┘               │
                                     ▼                            │
                         window = [r−0.15s, r+0.7s]               │
                                     ├────────────────────────────┘
                                     ▼
                         ┌─────────────────────────────────────────┐
                         │ YOLO ONLY in window                      │
                         │  release+2–3f: wrist-ROI (upscaled/tiled)│
                         │  after: motion-predicted expanding ROI   │
                         └───────────────┬─────────────────────────┘
                                         ▼
                              Tracking → Trajectory Reconstruction
```

---

## 4. Expected accuracy

| Condition | Release accuracy |
|---|---|
| Foreground bowler, 60 fps, low blur | **±1–2 frames (~±17–33 ms)** |
| Foreground bowler, 30 fps | **±1–3 frames (~±33–100 ms)** |
| Bowler <~15% frame height / heavy blur / occluded | **unusable** → use fallback |

Because we then run YOLO over a *padded* window (Q4), a ±3-frame release error is
harmless — it just widens the search slightly. The window tolerates the pose
error by design.

---

## 5. Fit in the MVP architecture

- Implement as a **standalone `ReleaseDetector`** (input: frames → output:
  `release_frame`, `confidence`, `window`, optional `wrist_track`), living beside
  the splitter. It should expose the **same "delivery window" contract** the
  pipeline already consumes, so it drops into the existing seam with **no changes
  to detection/calibration/trajectory**.
- Keep the current motion-energy `DeliverySplitter` as the **fallback** when pose
  is unreliable — mirrors the calibration projector's graceful-fallback pattern
  already in the codebase.
- This keeps pose **out of the detector** (per your intent) and cleanly optional.

---

## 6. Recommendation & sequencing

**Adopt — but scoped and sequenced:**

1. **First, get YOLO recall up on the target setup** (the yolo11m in-domain
   fine-tune on the side-on `Test*`/`bowling16–23` clips). Pose gating multiplies
   precision but *cannot create recall* — do the recall work first.
2. **Then add the `ReleaseDetector` module** for the foreground-bowler clips:
   biggest payoff is **FP reduction (Q5)** + **release-instant ROI recall (Q6)**,
   and it replaces the flaky motion segmentation for those clips.
3. **Do NOT** invest in it for behind-batsman footage — pose won't work there.
4. **Worth a parallel spike:** evaluate **TrackNet** as the in-window ball
   detector (Q8) — likely a bigger recall win for the tiny fast ball than YOLO.

**Priority: medium.** High-value, low-regret (isolated module, graceful
fallback), but it is an *amplifier* of a working detector, not a substitute for
one. Green-light implementation **once the fine-tuned detector shows usable
in-window recall on the side-on clips.**

---

## 7. Rough effort estimate (if approved later)
- `ReleaseDetector` (pose + person-select + heuristic + fallback): ~1–2 days.
- Window gating + wrist-ROI/tiling in the detection stage: ~0.5–1 day.
- Validation harness (release accuracy vs a few hand-marked release frames): ~0.5 day.
- (Optional) TrackNet spike: separate, larger effort.

*No code has been written. This document is a proposal for review only.*
