"""End-to-end cricket-AI pipeline orchestration.

Wires the existing modules into a single flow:

    Video
      -> Video Ingestion        (pipeline.ingest)
      -> Delivery Segmentation  (pipeline.splitter)
      -> Ball Detection         (pipeline.detection)
      -> Pixel-space track      (temporary)
      -> Projection interface   (pipeline.projection)  <-- calibration seam
      -> Trajectory Reconstruction (physics.trajectory)
      -> Delivery output

Every stage is injectable, so components can be swapped without touching the
orchestration. In particular the ``projector`` is where a future camera
calibration module slots in — see :mod:`pipeline.projection`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from models.delivery import Delivery
from physics.trajectory import TrajectoryReconstructor, apply_to_delivery
from pipeline.detection import BallDetection, BlurAwareBallDetector
from pipeline.ingest import Frame, IngestedVideo, VideoMetadata, ingest_video
from pipeline.projection import ObservationProjector, PixelPlaneProjector
from pipeline.splitter import DeliverySplitter

logger = logging.getLogger("cricket_ai.pipeline")

# Stable path for the trained ball detector weights (produced by
# scripts/train_ball_it1.py; see AUTONOMOUS_LOG.md / HANDOFF.md).
BALL_WEIGHTS = os.path.join("models", "ball_detector.pt")


def _default_ball_detector():
    """Trained YOLO detector if available, else motion-only fallback.

    Guarded so environments without ultralytics/weights (e.g. Python 3.14) still
    run the pipeline. NOTE: the trained detector is benchmark-validated but does
    NOT reliably fire on the ball in the current behind-the-batsman footage
    (see AUTONOMOUS_LOG.md 'BLOCKER'); it is wired in so the pipeline is ready
    for suitable (side-on / high-fps) footage.
    """
    from pipeline.detection import BlurAwareBallDetector
    if os.path.isfile(BALL_WEIGHTS):
        try:
            import ultralytics  # noqa: F401  (verify availability)
            from pipeline.detection import YOLOBallDetector
            yolo = YOLOBallDetector(BALL_WEIGHTS, conf_threshold=0.15, imgsz=1280)
            # accept YOLO detections readily; motion remains the fallback per-frame
            return BlurAwareBallDetector(yolo=yolo, accept_threshold=0.15)
        except Exception as e:  # ultralytics missing / load error -> motion-only
            logger.warning("Trained YOLO unavailable (%s); using motion detector.", e)
    return BlurAwareBallDetector(yolo=None)


# --------------------------------------------------------------------------- #
# Result / diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class DeliveryDiagnostics:
    """Per-delivery counts, for verifying stages communicate."""

    index: int
    window: str
    n_frames: int
    n_detections: int
    n_observations: int
    n_trajectory_points: int
    status: str


@dataclass
class PipelineResult:
    """Aggregate output of a pipeline run."""

    metadata: VideoMetadata
    deliveries: List[Delivery] = field(default_factory=list)
    diagnostics: List[DeliveryDiagnostics] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class CricketPipeline:
    """Runs video -> deliveries end to end.

    All collaborators are injectable. Defaults use the motion-only ball
    detector (no YOLO weights required) and the temporary
    :class:`PixelPlaneProjector`. Swap ``projector`` for a real calibration
    implementation when available — no other change is needed.
    """

    def __init__(
        self,
        splitter: Optional[DeliverySplitter] = None,
        ball_detector: Optional[BlurAwareBallDetector] = None,
        projector: Optional[ObservationProjector] = None,
        reconstructor: Optional[TrajectoryReconstructor] = None,
        sample_rate: int = 1,
        min_observations: int = 3,
    ) -> None:
        self.splitter = splitter or DeliverySplitter()
        # Default detector: use the trained YOLO ball model if weights exist AND
        # ultralytics is importable (GPU venv); otherwise fall back to motion-only
        # so the pipeline still runs anywhere. Detection wiring only.
        self.ball_detector = ball_detector or _default_ball_detector()
        self.projector = projector or PixelPlaneProjector()
        self.reconstructor = reconstructor or TrajectoryReconstructor()
        self.sample_rate = sample_rate
        self.min_observations = min_observations

    def run(self, video_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        # 1. Ingestion
        video: IngestedVideo = ingest_video(video_path, sample_rate=self.sample_rate)
        logger.info("Ingested %d frames from %s", len(video), video_path)

        # 1b. Projector setup (calibration seam). May raise CalibrationError,
        #     which rejects the video with an explanation.
        self.projector.prepare(video)

        # 2. Segmentation
        deliveries = self.splitter.split(video)
        logger.info("Segmented %d deliveries", len(deliveries))

        result = PipelineResult(metadata=video.metadata)
        for i, delivery in enumerate(deliveries):
            enriched, diag = self._process_delivery(i, delivery, video)
            result.deliveries.append(enriched)
            result.diagnostics.append(diag)

        # 6. Output
        if output_dir:
            self._write_output(result, output_dir)

        return result

    # -- per-delivery flow ------------------------------------------------- #
    def _process_delivery(
        self, index: int, delivery: Delivery, video: IngestedVideo
    ) -> tuple[Delivery, DeliveryDiagnostics]:
        # frames belonging to this delivery's time window
        frames = self._frames_for(video.frames, delivery)

        # 3. Ball detection
        detections = self.ball_detector.detect_sequence(frames)

        # 4. Temporary pixel-space track (drop the misses)
        track: List[BallDetection] = [d for d in detections if d is not None]

        # 5a. Projection (calibration seam): pixels -> world observations
        observations = self.projector.project(track, video.metadata)

        # 5b. Trajectory reconstruction (via the reconstructor interface)
        if len(observations) >= self.min_observations:
            recon = self.reconstructor.reconstruct(observations)
            delivery = apply_to_delivery(delivery, recon)
        else:
            logger.info(
                "Delivery %d: only %d observations (< %d); skipping reconstruction",
                index, len(observations), self.min_observations,
            )

        diag = DeliveryDiagnostics(
            index=index,
            window=f"{delivery.start_time:.2f}-{delivery.end_time:.2f}s",
            n_frames=len(frames),
            n_detections=len(track),
            n_observations=len(observations),
            n_trajectory_points=len(delivery.trajectory_3d),
            status=delivery.status.value,
        )
        return delivery, diag

    @staticmethod
    def _frames_for(frames: Sequence[Frame], delivery: Delivery) -> List[Frame]:
        return [
            f for f in frames
            if delivery.start_time <= f.timestamp <= delivery.end_time
        ]

    @staticmethod
    def _write_output(result: PipelineResult, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "deliveries.json")
        payload = [d.model_dump(mode="json") for d in result.deliveries]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Wrote %d deliveries to %s", len(payload), path)


def run_pipeline(video_path: str, output_dir: Optional[str] = None, **kwargs) -> PipelineResult:
    """Convenience wrapper around :class:`CricketPipeline`."""
    return CricketPipeline(**kwargs).run(video_path, output_dir=output_dir)
