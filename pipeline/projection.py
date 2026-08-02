"""Pixel-to-world projection — the camera-calibration seam.

Ball detection produces boxes in **pixel** space; trajectory reconstruction
consumes **world-frame** ``BallObservation``s (metres). The mapping between the
two is camera calibration.

Calibration is intentionally *not* implemented yet. Instead this module defines
the interface the future calibration module must satisfy —
:class:`ObservationProjector` — and ships a temporary, deliberately non-metric
:class:`PixelPlaneProjector` so the end-to-end pipeline runs and every stage can
be verified to communicate.

To add real calibration later, implement ``ObservationProjector`` (e.g.
``CameraCalibrationProjector``) and pass it to the pipeline. Nothing else
changes.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, Sequence

from pipeline.detection import BallDetection
from pipeline.ingest import IngestedVideo, VideoMetadata
from physics.trajectory import BallObservation

logger = logging.getLogger("cricket_ai.projection")


class ObservationProjector(Protocol):
    """Maps a pixel-space ball track into world-frame observations.

    Any implementation (temporary placeholder or real camera calibration)
    takes the ordered ball detections for one delivery plus the source video
    metadata and returns ``BallObservation``s in the pitch world frame.

    Lifecycle
    ---------
    ``prepare(video)`` is called once per video, after ingestion and before any
    ``project`` call. Session-level setup that needs image data — e.g. camera
    calibration — happens here. Stateless projectors implement it as a no-op.
    """

    def prepare(self, video: IngestedVideo) -> None:  # pragma: no cover - interface
        ...

    def project(
        self, track: Sequence[BallDetection], metadata: VideoMetadata
    ) -> List[BallObservation]:  # pragma: no cover - interface
        ...


class PixelPlaneProjector:
    """TEMPORARY, NON-METRIC projector — placeholder for camera calibration.

    Produces ``BallObservation``s so the trajectory stage has something to fit,
    but the coordinates are **not** physically calibrated:

    * ``x`` (lateral)      : image x offset from centre × ``meters_per_pixel``
    * ``z`` (height)       : inverted image y × ``meters_per_pixel`` (screen-up)
    * ``y`` (down-pitch)   : elapsed time × ``nominal_speed_mps`` (a guess)

    Depth is unrecoverable from a single uncalibrated view, so ``y`` is faked
    from timing. Expect bounce/line/length/speed to be meaningless until a real
    :class:`ObservationProjector` (camera calibration) replaces this.
    """

    def __init__(
        self,
        meters_per_pixel: float = 0.01,
        nominal_speed_mps: float = 30.0,
    ) -> None:
        self.meters_per_pixel = meters_per_pixel
        self.nominal_speed_mps = nominal_speed_mps
        self._warned = False

    def prepare(self, video: IngestedVideo) -> None:
        """No calibration needed — stateless placeholder."""
        return None

    def project(
        self, track: Sequence[BallDetection], metadata: VideoMetadata
    ) -> List[BallObservation]:
        if not self._warned:
            logger.warning(
                "PixelPlaneProjector is a non-metric placeholder; results are "
                "not physically calibrated. Insert a real camera-calibration "
                "ObservationProjector for meaningful output."
            )
            self._warned = True

        if not track:
            return []

        mpp = self.meters_per_pixel
        cx0 = metadata.width / 2.0
        height = float(metadata.height)
        t0 = track[0].timestamp

        observations: List[BallObservation] = []
        for det in track:
            px, py = det.center
            observations.append(
                BallObservation(
                    timestamp=det.timestamp,
                    x=(px - cx0) * mpp,
                    y=(det.timestamp - t0) * self.nominal_speed_mps,
                    z=max((height - py) * mpp, 0.0),
                )
            )
        return observations
