"""Data ingest, transform, and orchestration."""

from pipeline.ingest import (
    Frame,
    IngestedVideo,
    VideoIngestError,
    VideoIngestor,
    VideoMetadata,
    ingest_video,
)
from pipeline.splitter import (
    BowlingActionDetector,
    DeliverySplitter,
    HeuristicBowlingDetector,
    compute_motion_energy,
    split_video,
)
from pipeline.detection import (
    BallDetection,
    BallDetector,
    BlurAwareBallDetector,
    MotionBallDetector,
    YOLOBallDetector,
)
from pipeline.projection import ObservationProjector, PixelPlaneProjector
from pipeline.release_detector import ReleaseDetector, ReleaseEvent
from pipeline.tracking import (
    KalmanBall2D,
    PredictiveBallTracker,
    build_predictive_tracker,
)
from pipeline.calibration import (
    CalibrationError,
    CalibrationResult,
    CameraCalibrationProjector,
    CameraCalibrator,
    StumpPitchDetector,
    PitchModel,
    PitchReferenceDetector,
    ReferenceDetection,
    focal_from_homography,
)
from pipeline.pipeline import (
    CricketPipeline,
    DeliveryDiagnostics,
    PipelineResult,
    run_pipeline,
)

__all__ = [
    "Frame",
    "IngestedVideo",
    "VideoIngestError",
    "VideoIngestor",
    "VideoMetadata",
    "ingest_video",
    "BowlingActionDetector",
    "DeliverySplitter",
    "HeuristicBowlingDetector",
    "compute_motion_energy",
    "split_video",
    "BallDetection",
    "BallDetector",
    "BlurAwareBallDetector",
    "MotionBallDetector",
    "YOLOBallDetector",
    "ObservationProjector",
    "PixelPlaneProjector",
    "ReleaseDetector",
    "ReleaseEvent",
    "KalmanBall2D",
    "PredictiveBallTracker",
    "build_predictive_tracker",
    "CalibrationError",
    "CalibrationResult",
    "CameraCalibrationProjector",
    "CameraCalibrator",
    "HeuristicPitchDetector",
    "PitchModel",
    "PitchReferenceDetector",
    "ReferenceDetection",
    "focal_from_homography",
    "CricketPipeline",
    "DeliveryDiagnostics",
    "PipelineResult",
    "run_pipeline",
]
