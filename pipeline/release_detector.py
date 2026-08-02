"""Release detection from bowling-arm pose (OPTIONAL, MediaPipe-backed).

Given the frames of a single delivery, estimate **when and where** the ball
leaves the bowler's hand:

    ReleaseEvent(frame_index, wrist position in pixels, confidence)

The predictive tracker (:mod:`pipeline.tracking`) uses this to initialise the
ball at the wrist instead of searching the whole frame before release.

MediaPipe is an OPTIONAL dependency. If it is not installed (or pose fails on
the clip), :meth:`ReleaseDetector.detect` returns ``None`` and the tracker
falls back to full-frame YOLO acquisition — so the existing pipeline still runs
unchanged. Nothing here is imported at package import time unless requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pipeline.ingest import Frame

logger = logging.getLogger("cricket_ai.release")

# MediaPipe Pose landmark indices we care about.
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16


@dataclass
class ReleaseEvent:
    """Estimated ball-release moment for one delivery."""

    frame_index: int
    """``Frame.index`` of the release frame."""

    timestamp: float
    wrist_xy: Tuple[float, float]
    """Bowling-hand wrist position at release, in **pixels**."""

    confidence: float
    """0..1 — pose visibility around release × shape of the speed peak."""

    arm: str = "unknown"  # "right" | "left"


class ReleaseDetector:
    """Estimate the release frame + wrist position via MediaPipe Pose.

    Heuristic (single fixed camera, one bowler in view): track both wrists
    through the delivery, pick the *bowling arm* as the wrist that swings
    highest above the shoulders, and place release at the frame of **peak wrist
    speed** while the hand is at/above shoulder height (the hand is fastest as
    the ball is let go, at the top of the over-arm arc).

    Parameters
    ----------
    min_visibility:
        Minimum landmark visibility to trust a wrist sample.
    model_complexity:
        MediaPipe Pose complexity (0/1/2). 1 is a good speed/accuracy balance.
    smooth_win:
        Frames for the moving-average smoothing of the wrist path.
    """

    def __init__(
        self,
        min_visibility: float = 0.5,
        model_complexity: int = 1,
        smooth_win: int = 3,
        release_window: int = 12,
        min_lift: float = 20.0,
        min_confidence: float = 0.6,
    ) -> None:
        self.min_visibility = min_visibility
        self.model_complexity = model_complexity
        self.smooth_win = max(1, smooth_win)
        self.release_window = release_window
        self.min_lift = min_lift          # px the wrist must rise above the shoulder
        self.min_confidence = min_confidence
        self._available: Optional[bool] = None

    # -- availability ------------------------------------------------------ #
    def available(self) -> bool:
        """True if MediaPipe can be imported (checked once, cached)."""
        if self._available is None:
            try:
                import mediapipe  # noqa: F401
                self._available = True
            except Exception as e:  # pragma: no cover - env dependent
                logger.info("MediaPipe unavailable (%s); release detection disabled.", e)
                self._available = False
        return self._available

    # -- main -------------------------------------------------------------- #
    def detect(self, frames: Sequence[Frame]) -> Optional[ReleaseEvent]:
        """Return the estimated :class:`ReleaseEvent`, or ``None`` if pose is
        unavailable / the action can't be resolved confidently."""
        if not self.available() or len(frames) < 5:
            return None
        try:
            import mediapipe as mp
        except Exception:  # pragma: no cover
            return None

        H, W = frames[0].image.shape[:2]
        # per-wrist tracks in pixels; NaN where not visible
        n = len(frames)
        lw = np.full((n, 2), np.nan); rw = np.full((n, 2), np.nan)
        lw_vis = np.zeros(n); rw_vis = np.zeros(n)
        ls_y = np.full(n, np.nan); rs_y = np.full(n, np.nan)

        Pose = mp.solutions.pose.Pose
        with Pose(static_image_mode=False, model_complexity=self.model_complexity,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for i, f in enumerate(frames):
                rgb = f.image[:, :, ::-1]  # BGR -> RGB
                res = pose.process(rgb)
                lm = getattr(res, "pose_landmarks", None)
                if not lm:
                    continue
                pts = lm.landmark
                if pts[_L_WRIST].visibility >= self.min_visibility:
                    lw[i] = (pts[_L_WRIST].x * W, pts[_L_WRIST].y * H); lw_vis[i] = pts[_L_WRIST].visibility
                if pts[_R_WRIST].visibility >= self.min_visibility:
                    rw[i] = (pts[_R_WRIST].x * W, pts[_R_WRIST].y * H); rw_vis[i] = pts[_R_WRIST].visibility
                ls_y[i] = pts[_L_SHOULDER].y * H
                rs_y[i] = pts[_R_SHOULDER].y * H

        # choose the bowling arm: the wrist that reaches highest (min y) above
        # its shoulder, with enough visible samples.
        cand = []
        for name, wrist, vis, sh in (("left", lw, lw_vis, ls_y), ("right", rw, rw_vis, rs_y)):
            ok = ~np.isnan(wrist[:, 1])
            if ok.sum() < 5:
                continue
            apex = np.nanmin(wrist[:, 1])            # highest point (smallest y)
            sh_med = np.nanmedian(sh)
            lift = (sh_med - apex)                    # how far above shoulder it swings
            cand.append((lift, name, wrist, vis, sh_med))
        if not cand:
            return None
        cand.sort(reverse=True, key=lambda c: c[0])
        lift, arm, wrist, vis, sh_med = cand[0]
        # arm must actually swing over the top; otherwise pose isn't seeing a
        # bowling action (e.g. bowler too far / facing away) -> don't guess.
        if lift < self.min_lift:
            logger.info("Release: arm never rises above shoulder (lift=%.0fpx); skipping.", lift)
            return None

        rel = self._release_from_wrist(wrist, vis)
        if rel is None:
            return None
        idx, conf, wx, wy = rel
        if conf < self.min_confidence:
            logger.info("Release: low confidence (%.2f); skipping.", conf)
            return None
        return ReleaseEvent(
            frame_index=frames[idx].index,
            timestamp=frames[idx].timestamp,
            wrist_xy=(wx, wy),   # interpolated -> never NaN
            confidence=float(conf),
            arm=arm,
        )

    # -- internals --------------------------------------------------------- #
    def _release_from_wrist(
        self, wrist: np.ndarray, vis: np.ndarray
    ) -> Optional[Tuple[int, float, float, float]]:
        """Release = peak wrist speed near the top of the arm arc."""
        n = len(wrist)
        # interpolate short gaps so speed is well-defined
        t = np.arange(n)
        good = ~np.isnan(wrist[:, 0])
        if good.sum() < 5:
            return None
        x = np.interp(t, t[good], wrist[good, 0])
        y = np.interp(t, t[good], wrist[good, 1])
        if self.smooth_win > 1:
            k = np.ones(self.smooth_win) / self.smooth_win
            x = np.convolve(x, k, mode="same"); y = np.convolve(y, k, mode="same")
        speed = np.hypot(np.gradient(x), np.gradient(y))
        # release happens within a short window just after the arm's apex, not
        # anywhere in the rest of the clip — this rejects end-of-clip noise.
        apex = int(np.argmin(y))
        # if the arm is already at the top when the clip starts, the wind-up
        # isn't visible and we can't localise release reliably -> abstain.
        if apex <= 1:
            return None
        hi = min(apex + self.release_window, n)
        window = np.arange(apex, hi)
        if len(window) < 2:
            return None
        idx = int(window[np.argmax(speed[window])])
        # a release at the very last frames is almost always spurious
        if idx >= n - 2:
            return None
        # confidence: local visibility × how peaked the speed is
        vpk = speed[idx]
        peakiness = float(np.clip((vpk - np.median(speed)) / (vpk + 1e-6), 0, 1))
        local_vis = float(np.nanmean(vis[max(0, idx - 2): idx + 3]) or 0.0)
        conf = float(np.clip(0.5 * peakiness + 0.5 * local_vis, 0, 1))
        return idx, conf, float(x[idx]), float(y[idx])
