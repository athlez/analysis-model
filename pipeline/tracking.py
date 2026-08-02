"""Prediction-driven ball tracking (Kalman primary, YOLO/motion as sensors).

The default :class:`~pipeline.detection.BlurAwareBallDetector` treats every
frame independently: it runs YOLO over the *whole* frame and only lightly gates
the result. That produces false positives and jittery tracks.

:class:`PredictiveBallTracker` inverts the relationship. A constant-acceleration
Kalman filter is the **primary** tracker holding the ball state
(position/velocity/acceleration). Each frame it *predicts* first, derives a
small search region from that prediction, and runs the existing detectors only
inside that region. YOLO and motion are demoted to **observation sources** that
merely correct the predicted state; detections that violate simple physics are
rejected. On a miss the filter keeps predicting for several frames and tries to
reacquire near the prediction rather than terminating the track.

It implements the same ``detect_sequence(frames) -> List[Optional[BallDetection]]``
contract as ``BlurAwareBallDetector``, so it drops straight into
``CricketPipeline(ball_detector=...)`` with no other change. MediaPipe release
detection is optional — without it the tracker acquires the ball with a
full-frame YOLO pass and proceeds identically thereafter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pipeline.detection import BallDetection, MotionBallDetector, YOLOBallDetector
from pipeline.ingest import Frame

logger = logging.getLogger("cricket_ai.tracking")


# --------------------------------------------------------------------------- #
# Constant-acceleration Kalman filter (2-D image space)
# --------------------------------------------------------------------------- #
class KalmanBall2D:
    """6-state constant-acceleration filter: [x, y, vx, vy, ax, ay] (pixels, /frame).

    A plain NumPy implementation so no new dependency (filterpy etc.) is needed.
    ``dt`` is one frame; measurements are ball centres in pixels.
    """

    def __init__(self, accel_noise: float = 45.0):
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], float)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], float)
        # process noise driven by acceleration jerk
        q = accel_noise ** 2
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0, q, q])
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1e3

    def init_state(self, pos: Tuple[float, float], vel: Tuple[float, float] = (0.0, 0.0),
                   pos_var: float = 25.0, vel_var: float = 1e4) -> None:
        self.x = np.array([[pos[0]], [pos[1]], [vel[0]], [vel[1]], [0.0], [0.0]])
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var, 1e4, 1e4])

    def predict(self, gravity: float = 0.0) -> None:
        self.x = self.F @ self.x
        if gravity:
            # gravity-aware control input: nudge the y-motion downward so the
            # prediction (and thus the ROI) follows a parabolic flight path even
            # when detections are sparse. Updates still correct it when the ball
            # is seen; this dominates only while coasting through misses.
            self.x[3, 0] += gravity          # vy += g
            self.x[1, 0] += 0.5 * gravity    # y  += 0.5 g
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas: Tuple[float, float], meas_var: float) -> None:
        z = np.array([[meas[0]], [meas[1]]])
        R = np.eye(2) * meas_var
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    # -- accessors --------------------------------------------------------- #
    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def acc(self) -> Tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])

    @property
    def pos_std(self) -> Tuple[float, float]:
        return float(np.sqrt(max(self.P[0, 0], 0))), float(np.sqrt(max(self.P[1, 1], 0)))


# --------------------------------------------------------------------------- #
# Predictive tracker
# --------------------------------------------------------------------------- #
@dataclass
class _Cand:
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    source: str


class PredictiveBallTracker:
    """Kalman-primary ball tracker; YOLO + motion are ROI-limited sensors.

    Parameters
    ----------
    yolo:
        Existing :class:`YOLOBallDetector` (reused, not replaced). Optional —
        omit to run motion-only.
    motion:
        :class:`MotionBallDetector` used as a secondary observation source.
    release_detector:
        Optional :class:`~pipeline.release_detector.ReleaseDetector`. When it
        returns an event, the ball is initialised at the wrist and no detection
        runs before release. When ``None`` / unavailable, the tracker acquires
        the ball with a full-frame YOLO pass instead.
    accept_conf:
        YOLO confidence to accept an observation.
    max_misses:
        Frames the filter will coast (predict-only) before terminating.
    base_roi:
        Half-size (px) of the search box around the prediction on a clean frame.
    max_speed:
        Physical gate — largest plausible ball-centre speed (px/frame).
    """

    def __init__(
        self,
        yolo: Optional[YOLOBallDetector] = None,
        motion: Optional[MotionBallDetector] = None,
        release_detector=None,
        accept_conf: float = 0.15,
        max_misses: int = 8,
        base_roi: float = 90.0,
        max_speed: float = 220.0,
        accel_noise: float = 45.0,
        release_speed: float = 38.0,
        seed_frames: int = 6,
        seed_boost: float = 1.8,
        sahi_model=None,
        sahi_slice: int = 384,
        expand_after: int = 3,
        reacquire_after: int = 8,
        acquire_stride: int = 2,
        acquire_budget: int = 300,
        gravity: float = 0.0,
    ) -> None:
        self.yolo = yolo
        self.motion = motion or MotionBallDetector()
        self.release_detector = release_detector
        # SAHI tiled model — used to ACQUIRE the tiny ball full-frame (plain YOLO
        # can't see it); after acquisition, ROI crops make it detectable cheaply.
        self.sahi_model = sahi_model
        self.sahi_slice = sahi_slice
        self.accept_conf = accept_conf
        self.max_misses = max_misses
        self.base_roi = base_roi
        self.max_speed = max_speed
        self.accel_noise = accel_noise
        # release-velocity seeding: after wrist-init the ball has NO observed
        # velocity, so a v=0 filter coasts in place while the ball flies away.
        # Seed a downward, down-pitch velocity so the search region chases it,
        # and widen the region for the first few post-release frames.
        self.release_speed = release_speed
        self.seed_frames = seed_frames
        self.seed_boost = seed_boost
        # state-machine thresholds: LOST(expand_after)->grow ROI;
        # LOST(reacquire_after)->full-frame SAHI re-acquire; LOST(max_misses)->terminate.
        self.expand_after = expand_after
        self.reacquire_after = reacquire_after
        self.acquire_stride = acquire_stride      # throttle full-frame SAHI scanning
        self.acquire_budget = acquire_budget      # give up acquiring after this many attempts
        self.gravity = gravity                    # px/frame^2 downward; 0 disables gravity prior
        self.metrics: dict = {}

    # ---- public contract (matches BlurAwareBallDetector) ----------------- #
    def detect_sequence(self, frames: Sequence[Frame]) -> List[Optional[BallDetection]]:
        """ONLINE state machine: ACQUIRE -> LOCKED -> (LOST: expand ROI ->
        re-acquire via SAHI -> terminate). Detection and tracking advance
        together, frame by frame; predictions fill short detector gaps."""
        frames = list(frames)
        n = len(frames)
        results: List[Optional[BallDetection]] = [None] * n
        self.metrics = {"n_frames": n, "acquired": False, "acquire_frame": None,
                        "detected": 0, "predicted": 0, "recoveries": 0,
                        "expand_events": 0, "reacquire_attempts": 0,
                        "terminated_frame": None, "tracked_span": 0, "final_state": "NEVER_ACQUIRED"}
        if n == 0:
            return results

        start_i, init_pos, init_vel, init_conf = self._acquire(frames, 0)
        if start_i is None:
            return results

        kf = KalmanBall2D(accel_noise=self.accel_noise)
        kf.init_state(init_pos, vel=init_vel)
        last_size = self._init_size(frames[start_i], init_pos)
        results[start_i] = BallDetection(
            frame_index=frames[start_i].index, timestamp=frames[start_i].timestamp,
            bbox=(init_pos[0] - 9, init_pos[1] - 9, init_pos[0] + 9, init_pos[1] + 9),
            confidence=init_conf or 0.3, source="yolo")
        self.metrics.update(acquired=True, acquire_frame=start_i, final_state="LOCKED")
        self.metrics["detected"] += 1
        misses, have_velocity, last_tracked = 0, False, start_i

        H, W = frames[0].image.shape[:2]
        i = start_i + 1
        while i < n:
            f = frames[i]
            kf.predict(self.gravity)                        # PREDICT (gravity-aware)
            self._stabilize(kf, W, H)                       # keep the coast physical (no blow-up)
            pred = kf.pos
            boost = self.seed_boost if (i - start_i) < self.seed_frames else 1.0
            roi = self._roi(pred, kf, misses, boost)       # ROI (grows with misses)
            best = self._pick(self._observe(frames, i, roi), kf, have_velocity)

            if best is not None:                           # LOCKED / recovered
                if misses >= self.reacquire_after:
                    self.metrics["recoveries"] += 1
                kf.update(best.center, self._meas_var(best))
                misses = 0
                last_size = (best.bbox[2] - best.bbox[0], best.bbox[3] - best.bbox[1])
                results[i] = BallDetection(frame_index=f.index, timestamp=f.timestamp,
                                           bbox=best.bbox, confidence=best.confidence, source=best.source)
                self.metrics["detected"] += 1
                if np.hypot(*kf.vel) > 2.0:
                    have_velocity = True
            else:                                          # LOST
                misses += 1
                if misses == self.expand_after:
                    self.metrics["expand_events"] += 1
                reacq = None
                if misses >= self.reacquire_after:         # full-frame SAHI re-acquire
                    self.metrics["reacquire_attempts"] += 1
                    c = self._acquire_detect(f)
                    if c is not None and self._plausible_reacq((c[0], c[1]), kf, misses):
                        reacq = c
                if reacq is not None:
                    kf.x[0, 0], kf.x[1, 0] = reacq[0], reacq[1]
                    kf.P[0, 0] = kf.P[1, 1] = 25.0
                    misses = 0
                    self.metrics["recoveries"] += 1
                    results[i] = BallDetection(frame_index=f.index, timestamp=f.timestamp,
                                               bbox=(reacq[0] - 9, reacq[1] - 9, reacq[0] + 9, reacq[1] + 9),
                                               confidence=reacq[2], source="reacquired")
                    self.metrics["detected"] += 1
                else:
                    cx, cy = pred; w, h = last_size         # coast (prediction fills the gap)
                    results[i] = BallDetection(frame_index=f.index, timestamp=f.timestamp,
                                               bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                                               confidence=0.1, source="predicted")
                    self.metrics["predicted"] += 1
                    if misses > self.max_misses:            # TERMINATE
                        self.metrics.update(terminated_frame=f.index, final_state="LOST")
                        break
            last_tracked = i
            i += 1

        self.metrics["tracked_span"] = last_tracked - start_i + 1
        return results

    def _stabilize(self, kf: KalmanBall2D, W: int, H: int) -> None:
        """Keep the predicted state physical so a sparse/noisy track can't blow
        up: cap velocity to max_speed, cap acceleration, and clamp the predicted
        position to a margin around the frame. Without this the constant-accel
        model extrapolates quadratically and the coast flies thousands of px
        off-screen."""
        max_a = 6.0
        kf.x[2, 0] = float(np.clip(kf.x[2, 0], -self.max_speed, self.max_speed))
        kf.x[3, 0] = float(np.clip(kf.x[3, 0], -self.max_speed, self.max_speed))
        kf.x[4, 0] = float(np.clip(kf.x[4, 0], -max_a, max_a))
        kf.x[5, 0] = float(np.clip(kf.x[5, 0], -max_a, max_a))
        m = 0.25
        kf.x[0, 0] = float(np.clip(kf.x[0, 0], -m * W, W + m * W))
        kf.x[1, 0] = float(np.clip(kf.x[1, 0], -m * H, H + m * H))

    def _plausible_reacq(self, pos, kf: KalmanBall2D, misses: int) -> bool:
        """A re-acquired detection must be reachable from the prediction (no
        teleport) and not a big backward jump against the travel direction."""
        px, py = kf.pos
        dx, dy = pos[0] - px, pos[1] - py
        if np.hypot(dx, dy) > self.max_speed * max(misses, 1) * 1.5:
            return False
        vx, vy = kf.vel
        if np.hypot(vx, vy) > 4.0 and (dx * vx + dy * vy) < 0:   # moving opposite travel
            return False
        return True

    # ---- acquisition ----------------------------------------------------- #
    def _acquire(self, frames: Sequence[Frame], from_i: int = 0):
        """Return (start_index, init_pos, init_vel, init_conf).

        Prefer MediaPipe release (init at wrist + a seeded down-pitch velocity).
        Else scan full-frame SAHI/YOLO — THROTTLED by ``acquire_stride`` and
        capped at ``acquire_budget`` attempts so a video with no ball fails fast
        instead of running SAHI on every frame. Velocity is seeded from the
        first two hits so the ROI immediately chases the moving ball.
        """
        if from_i == 0 and self.release_detector is not None:
            try:
                ev = self.release_detector.detect(frames)
            except Exception as e:  # pragma: no cover - pose runtime issues
                logger.warning("Release detection failed (%s); using YOLO acquisition.", e)
                ev = None
            if ev is not None:
                H, W = frames[0].image.shape[:2]
                vel = self._seed_velocity(ev.wrist_xy, W, H)
                for i, f in enumerate(frames):
                    if f.index == ev.frame_index:
                        return i, ev.wrist_xy, vel, ev.confidence

        hits = []          # (i, cx, cy, conf)
        attempts = 0
        for i in range(from_i, len(frames), max(1, self.acquire_stride)):
            if attempts >= self.acquire_budget:
                break
            attempts += 1
            c = self._acquire_detect(frames[i])
            if c is not None:
                hits.append((i, c[0], c[1], c[2]))
                if len(hits) >= 2:
                    (i0, x0, y0, _), (i1, x1, y1, c1) = hits[-2], hits[-1]
                    dt = max(i1 - i0, 1)
                    return i1, (x1, y1), ((x1 - x0) / dt, (y1 - y0) / dt), c1
        if hits:           # only one detection found
            i0, x0, y0, c0 = hits[0]
            return i0, (x0, y0), (0.0, 0.0), c0
        return None, None, None, None

    def _acquire_detect(self, frame):
        """Best ball detection on a full frame (SAHI if available), or None."""
        if self.sahi_model is not None:
            from sahi.predict import get_sliced_prediction
            res = get_sliced_prediction(
                frame.image[:, :, ::-1], self.sahi_model,
                slice_height=self.sahi_slice, slice_width=self.sahi_slice,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                perform_standard_pred=False, verbose=0)
            best = max(res.object_prediction_list, key=lambda o: o.score.value, default=None)
            if best is not None and best.score.value >= self.accept_conf:
                b = best.bbox
                return ((b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2, best.score.value)
            return None
        if self.yolo is not None:
            dets = self.yolo.detect_frame(frame.image, frame.index, frame.timestamp)
            best = max(dets, key=lambda d: d.confidence, default=None)
            if best is not None and best.confidence >= self.accept_conf:
                return (best.center[0], best.center[1], best.confidence)
        return None

    def _seed_velocity(self, wrist, W, H):
        """Prior release velocity: aim down-pitch (toward bottom-centre of the
        frame) at ``release_speed`` px/frame, so the ROI moves off the hand."""
        dx, dy = (W * 0.5 - wrist[0]), (H * 0.95 - wrist[1])
        norm = float(np.hypot(dx, dy)) or 1.0
        return (self.release_speed * dx / norm, self.release_speed * dy / norm)

    # ---- observation sources (ROI-limited) ------------------------------- #
    def _observe(self, frames: Sequence[Frame], i: int, roi) -> List[_Cand]:
        f = frames[i]
        x1, y1, x2, y2 = roi
        cands: List[_Cand] = []

        # 1) Detector inside the ROI. Prefer SAHI tiling of the crop (same recipe
        #    that actually detects the tiny ball); fall back to plain YOLO-on-crop.
        crop = f.image[y1:y2, x1:x2]
        if crop.size and self.sahi_model is not None:
            from sahi.predict import get_sliced_prediction
            sl = max(160, min(self.sahi_slice, crop.shape[0], crop.shape[1]))
            res = get_sliced_prediction(crop[:, :, ::-1], self.sahi_model,
                                        slice_height=sl, slice_width=sl,
                                        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                        perform_standard_pred=True, verbose=0)
            for o in res.object_prediction_list:
                b = o.bbox
                bx = (b.minx + x1, b.miny + y1, b.maxx + x1, b.maxy + y1)
                cands.append(_Cand(((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), bx, o.score.value, "yolo"))
        elif crop.size and self.yolo is not None:
            for d in self.yolo.detect_frame(crop, f.index, f.timestamp):
                bx = (d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2] + x1, d.bbox[3] + y1)
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                cands.append(_Cand((cx, cy), bx, d.confidence, "yolo"))

        # 2) Motion inside the ROI (needs previous frame)
        if i > 0:
            prev = frames[i - 1].image[y1:y2, x1:x2]
            cur = f.image[y1:y2, x1:x2]
            if prev.size and cur.size and prev.shape == cur.shape:
                for d in self.motion.detect(prev, cur, f.index, f.timestamp):
                    bx = (d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2] + x1, d.bbox[3] + y1)
                    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                    cands.append(_Cand((cx, cy), bx, d.confidence, "motion"))
        return cands

    def _pick(self, cands: List[_Cand], kf: KalmanBall2D, have_velocity: bool) -> Optional[_Cand]:
        """Choose the observation that best corrects the prediction, after
        rejecting physically impossible ones."""
        pred = kf.pos
        vx, vy = kf.vel
        speed = np.hypot(vx, vy)
        valid: List[Tuple[float, _Cand]] = []
        for c in cands:
            dx, dy = c.center[0] - pred[0], c.center[1] - pred[1]
            jump = np.hypot(dx, dy)
            # gate 1: impossible jump / unrealistic speed relative to prediction
            if jump > self.max_speed:
                continue
            # gate 2: no sudden backward motion once travelling
            if have_velocity and speed > 4.0:
                if (dx * vx + dy * vy) < -0.5 * speed * jump:
                    continue
            # score: prefer YOLO, high confidence, close to prediction
            score = c.confidence + (0.3 if c.source == "yolo" else 0.0) - 0.002 * jump
            valid.append((score, c))
        if not valid:
            return None
        valid.sort(key=lambda t: t[0], reverse=True)
        return valid[0][1]

    # ---- geometry / noise helpers ---------------------------------------- #
    def _roi(self, pred: Tuple[float, float], kf: KalmanBall2D, misses: int, boost: float = 1.0):
        """Search box: base size + positional uncertainty + a per-miss growth
        (and an early-frame ``boost`` right after release), biased forward along
        the velocity."""
        sx, sy = kf.pos_std
        grow = (1.0 + 0.6 * misses) * boost
        half_x = (self.base_roi + 2 * sx) * grow
        half_y = (self.base_roi + 2 * sy) * grow
        # bias toward where the ball is heading
        vx, vy = kf.vel
        cx, cy = pred[0] + 0.5 * vx, pred[1] + 0.5 * vy
        x1, y1 = int(max(cx - half_x, 0)), int(max(cy - half_y, 0))
        x2, y2 = int(cx + half_x), int(cy + half_y)
        return x1, y1, x2, y2

    @staticmethod
    def _meas_var(c: _Cand) -> float:
        # smaller variance for confident YOLO boxes; motion is noisier
        base = 9.0 if c.source == "yolo" else 36.0
        return base / max(c.confidence, 0.1)

    @staticmethod
    def _init_size(frame: Frame, pos: Tuple[float, float]) -> Tuple[float, float]:
        return 18.0, 18.0  # nominal ball box until a real detection refines it


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def build_predictive_tracker(
    weights: Optional[str] = None,
    use_pose: bool = True,
    conf_threshold: float = 0.15,
    imgsz: int = 640,
    **kwargs,
) -> PredictiveBallTracker:
    """Wire a tracker from weights + optional pose, reusing existing detectors.

    ``use_pose=True`` attaches a :class:`ReleaseDetector`; if MediaPipe isn't
    installed it silently degrades to full-frame YOLO acquisition.
    """
    yolo = None
    if weights:
        yolo = YOLOBallDetector(weights, conf_threshold=conf_threshold, imgsz=imgsz)
    release = None
    if use_pose:
        from pipeline.release_detector import ReleaseDetector
        rd = ReleaseDetector()
        release = rd if rd.available() else None
    return PredictiveBallTracker(yolo=yolo, release_detector=release,
                                 accept_conf=conf_threshold, **kwargs)
