"""3D trajectory reconstruction.

Given noisy world-frame ball observations for a single delivery, fit a
physically-consistent projectile-with-bounce model and derive:

* a smooth 3D trajectory (``TrajectoryPoint`` list),
* the bounce location on the pitch (``PitchPoint``),
* the 2D line/length summary (``Pitch2D``),
* release speed in km/h.

Coordinate frame (world / pitch frame — an upstream calibration stage is
responsible for mapping pixels into it):

    origin  : striker's stumps, at ground level
    x axis  : lateral. +x = off side for a right-handed batter
    y axis  : down the pitch toward the bowler (metres from striker)
    z axis  : height above the ground (metres)

Physics model
-------------
Horizontal motion (x, y) is treated as constant-velocity (air drag ignored
for the MVP), so x(t) and y(t) are linear. Vertical motion z(t) is a parabola
under gravity, ``z = z0 + vz·t - ½·g·t²``. A delivery that bounces is two such
parabolas joined at the bounce, so we detect the bounce and fit each side
independently. A full toss (no bounce) is fit as a single segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from models.delivery import (
    Delivery,
    DeliveryStatus,
    Length,
    Line,
    Pitch2D,
    PitchPoint,
    TrajectoryPoint,
)

logger = logging.getLogger("cricket_ai.trajectory")

GRAVITY = 9.81  # m/s²
_MS_TO_KMPH = 3.6


# --------------------------------------------------------------------------- #
# Inputs & geometry
# --------------------------------------------------------------------------- #
@dataclass
class BallObservation:
    """A single detected ball position in the world frame."""

    timestamp: float  # seconds (clip-relative)
    x: float
    y: float
    z: float


@dataclass
class PitchGeometry:
    """Pitch dimensions and classification thresholds (all metres).

    Length zones are distances of the bounce from the striker's stumps.
    Defaults are approximate and intended to be tuned per competition/format.
    """

    half_stump_width: float = 0.15  # ~middle-stump tolerance for "middle" line
    # length zone upper bounds, measured from striker's stumps (metres):
    yorker_max: float = 1.5
    full_max: float = 4.0
    good_max: float = 7.0
    # beyond good_max => short


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ReconstructionResult:
    """Everything the reconstruction stage produces for a delivery."""

    points: List[TrajectoryPoint] = field(default_factory=list)
    pitch_point: Optional[PitchPoint] = None
    bounce_time: Optional[float] = None
    speed_kmph: Optional[float] = None
    line: Optional[Line] = None
    length: Optional[Length] = None
    residual: float = 0.0
    """RMS fit residual in metres — a proxy for reconstruction quality."""


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_line(x: float, geom: PitchGeometry, batter_hand: str = "right") -> Line:
    """Map a bounce x-offset to off/middle/leg.

    +x is the off side for a right-hander; mirror for a left-hander.
    """
    if abs(x) <= geom.half_stump_width:
        return Line.MIDDLE
    off_positive = batter_hand.lower().startswith("r")
    is_off = (x > 0) == off_positive
    return Line.OFF if is_off else Line.LEG


def classify_length(y: float, geom: PitchGeometry) -> Length:
    """Map a bounce distance-from-striker to a length zone."""
    if y <= geom.yorker_max:
        return Length.YORKER
    if y <= geom.full_max:
        return Length.FULL
    if y <= geom.good_max:
        return Length.GOOD
    return Length.SHORT


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def _fit_linear(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return [slope, intercept] for v ≈ slope·t + intercept."""
    return np.polyfit(t, v, 1)


def _fit_parabola(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return [a, b, c] for z ≈ a·t² + b·t + c."""
    return np.polyfit(t, z, 2)


def _detect_bounce_index(obs: Sequence[BallObservation]) -> Optional[int]:
    """Index of the bounce = interior minimum in height (z).

    Returns None when the minimum is at an endpoint (i.e. no bounce captured —
    a full toss or a clip that ends before/after the bounce).
    """
    zs = np.array([o.z for o in obs])
    idx = int(np.argmin(zs))
    if idx == 0 or idx == len(zs) - 1:
        return None
    return idx


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
class TrajectoryReconstructor:
    """Fits a smooth 3D trajectory to ball observations for one delivery."""

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        batter_hand: str = "right",
        sample_hz: float = 120.0,
    ) -> None:
        self.geom = geometry or PitchGeometry()
        self.batter_hand = batter_hand
        self.sample_hz = sample_hz

    def reconstruct(self, observations: Sequence[BallObservation]) -> ReconstructionResult:
        obs = sorted(observations, key=lambda o: o.timestamp)
        if len(obs) < 3:
            logger.warning("Need >= 3 observations to fit a trajectory; got %d", len(obs))
            return ReconstructionResult()

        t = np.array([o.timestamp for o in obs])
        x = np.array([o.x for o in obs])
        y = np.array([o.y for o in obs])
        z = np.array([o.z for o in obs])

        bounce_idx = _detect_bounce_index(obs)

        # Build the smooth sample grid.
        t0, t1 = float(t[0]), float(t[-1])
        n_samples = max(int(round((t1 - t0) * self.sample_hz)) + 1, len(obs))
        grid = np.linspace(t0, t1, n_samples)

        if bounce_idx is None:
            # Single-segment (full toss or partial capture).
            fx, fy, fz = _fit_linear(t, x), _fit_linear(t, y), _fit_parabola(t, z)
            xg, yg, zg = np.polyval(fx, grid), np.polyval(fy, grid), np.polyval(fz, grid)
            bounce_time = None
            pitch_point = None
            residual = self._residual(t, x, y, z, fx, fy, fz)
            vx, vy, vz = fx[0], fy[0], 2 * fz[0] * t0 + fz[1]
        else:
            bounce_time = float(t[bounce_idx])
            # Fit each side of the bounce independently, then stitch on the grid.
            pre = slice(0, bounce_idx + 1)
            post = slice(bounce_idx, len(obs))
            fx_a, fy_a, fz_a = _fit_linear(t[pre], x[pre]), _fit_linear(t[pre], y[pre]), _fit_parabola(t[pre], z[pre])
            fx_b, fy_b, fz_b = _fit_linear(t[post], x[post]), _fit_linear(t[post], y[post]), _fit_parabola(t[post], z[post])

            mask = grid <= bounce_time
            xg = np.where(mask, np.polyval(fx_a, grid), np.polyval(fx_b, grid))
            yg = np.where(mask, np.polyval(fy_a, grid), np.polyval(fy_b, grid))
            zg = np.where(mask, np.polyval(fz_a, grid), np.polyval(fz_b, grid))
            zg = np.clip(zg, 0.0, None)  # ball never goes below ground

            pitch_point = PitchPoint(
                x=float(np.polyval(fx_a, bounce_time)),
                y=float(np.polyval(fy_a, bounce_time)),
            )
            residual = 0.5 * (
                self._residual(t[pre], x[pre], y[pre], z[pre], fx_a, fy_a, fz_a)
                + self._residual(t[post], x[post], y[post], z[post], fx_b, fy_b, fz_b)
            )
            # Release velocity from the pre-bounce segment at t0.
            vx, vy, vz = fx_a[0], fy_a[0], 2 * fz_a[0] * t0 + fz_a[1]

        points = [
            TrajectoryPoint(x=float(xi), y=float(yi), z=float(zi), timestamp=float(ti))
            for ti, xi, yi, zi in zip(grid, xg, yg, zg)
        ]

        speed = float(np.sqrt(vx**2 + vy**2 + vz**2) * _MS_TO_KMPH)

        result = ReconstructionResult(
            points=points,
            pitch_point=pitch_point,
            bounce_time=bounce_time,
            speed_kmph=round(speed, 1),
            residual=round(float(residual), 4),
        )
        if pitch_point is not None:
            result.line = classify_line(pitch_point.x, self.geom, self.batter_hand)
            result.length = classify_length(pitch_point.y, self.geom)
        return result

    @staticmethod
    def _residual(t, x, y, z, fx, fy, fz) -> float:
        """RMS 3D distance between observations and the fitted curves."""
        dx = np.polyval(fx, t) - x
        dy = np.polyval(fy, t) - y
        dz = np.polyval(fz, t) - z
        return float(np.sqrt(np.mean(dx**2 + dy**2 + dz**2)))


# --------------------------------------------------------------------------- #
# Delivery integration
# --------------------------------------------------------------------------- #
def apply_to_delivery(delivery: Delivery, result: ReconstructionResult) -> Delivery:
    """Return a copy of ``delivery`` enriched with reconstruction output.

    Fills ``trajectory_3d``, ``events.pitch_point``, ``pitch_2d`` and
    ``speed_kmph``. If a clean bounce was found and the fit is tight, upgrades
    a ``LOW_CONFIDENCE`` status to ``SUCCESS``.
    """
    events = delivery.events.model_copy(update={"pitch_point": result.pitch_point})

    pitch_2d = None
    if result.line is not None and result.length is not None:
        pitch_2d = Pitch2D(line=result.line, length=result.length)

    status = delivery.status
    if (
        status == DeliveryStatus.LOW_CONFIDENCE
        and result.pitch_point is not None
        and result.residual < 0.1  # metres RMS
    ):
        status = DeliveryStatus.SUCCESS

    return delivery.model_copy(
        update={
            "events": events,
            "trajectory_3d": result.points,
            "pitch_2d": pitch_2d,
            "speed_kmph": result.speed_kmph,
            "status": status,
        }
    )


def reconstruct_delivery(
    delivery: Delivery,
    observations: Sequence[BallObservation],
    geometry: Optional[PitchGeometry] = None,
    batter_hand: str = "right",
    sample_hz: float = 120.0,
) -> Delivery:
    """One-shot: reconstruct from observations and enrich the delivery."""
    result = TrajectoryReconstructor(
        geometry=geometry, batter_hand=batter_hand, sample_hz=sample_hz
    ).reconstruct(observations)
    return apply_to_delivery(delivery, result)
