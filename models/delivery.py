"""Delivery data model.

A `Delivery` describes a single ball bowled: the source video window, the
key detected events, the reconstructed 3D trajectory, the 2D pitch-map
summary (line & length), and quality/status metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DeliveryStatus(str, Enum):
    """Overall outcome of processing this delivery."""

    SUCCESS = "success"
    FAILED = "failed"
    LOW_CONFIDENCE = "low_confidence"


class Line(str, Enum):
    """Categorical line relative to the stumps (batter's perspective)."""

    OFF = "off"
    MIDDLE = "middle"
    LEG = "leg"


class Length(str, Enum):
    """Categorical length zone where the ball pitches."""

    SHORT = "short"
    GOOD = "good"
    FULL = "full"
    YORKER = "yorker"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class TrajectoryPoint(BaseModel):
    """A single smoothed point on the 3D ball path.

    Coordinates are in metres in the pitch reference frame; `timestamp` is
    seconds relative to the start of the delivery clip.
    """

    x: float
    y: float
    z: float
    timestamp: float = Field(..., ge=0.0, description="Seconds from clip start")


class PitchPoint(BaseModel):
    """Where the ball bounced on the pitch (2D pitch-plane coordinates, metres)."""

    x: float
    y: float


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
class Events(BaseModel):
    """Key temporal/spatial events detected within the delivery."""

    release_frame: Optional[int] = Field(
        default=None,
        ge=0,
        description="Frame index of ball release; may be estimated / absent.",
    )
    release_estimated: bool = Field(
        default=False,
        description="True when release_frame was inferred rather than detected.",
    )
    pitch_point: Optional[PitchPoint] = Field(
        default=None,
        description=(
            "Bounce location of the ball on the pitch. None until the "
            "trajectory-reconstruction stage computes it."
        ),
    )


# --------------------------------------------------------------------------- #
# Pitch map (2D)
# --------------------------------------------------------------------------- #
class Pitch2D(BaseModel):
    """2D pitch-map summary: line and length.

    `line` accepts either a categorical `Line` (off/middle/leg) or a numeric
    x-offset in metres from middle stump (negative = leg, positive = off,
    by convention). `length` is a categorical zone.
    """

    line: Union[Line, float]
    length: Length

    @field_validator("line")
    @classmethod
    def _coerce_line(cls, v: Union[Line, float]) -> Union[Line, float]:
        # Allow the categorical strings to parse into the Line enum while
        # leaving numeric x-offsets untouched.
        if isinstance(v, str):
            return Line(v)
        return v


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
class Delivery(BaseModel):
    """A single bowled delivery reconstructed from video."""

    video_id: str
    start_time: float = Field(..., ge=0.0, description="Clip start, seconds into source video")
    end_time: float = Field(..., ge=0.0, description="Clip end, seconds into source video")

    events: Events

    trajectory_3d: List[TrajectoryPoint] = Field(
        default_factory=list,
        description="Ordered, smoothed 3D ball-path points.",
    )

    pitch_2d: Optional[Pitch2D] = Field(
        default=None,
        description="2D line/length summary. None until derived from the trajectory.",
    )

    speed_kmph: Optional[float] = Field(default=None, ge=0.0)

    confidence_score: float = Field(..., ge=0.0, le=1.0)

    status: DeliveryStatus

    @field_validator("end_time")
    @classmethod
    def _end_after_start(cls, end_time: float, info) -> float:
        start = info.data.get("start_time")
        if start is not None and end_time < start:
            raise ValueError("end_time must be >= start_time")
        return end_time
