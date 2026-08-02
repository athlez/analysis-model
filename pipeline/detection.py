"""Cricket-ball detection.

Per-frame detection of the cricket ball, returning a bounding box plus
confidence. A fast delivery blurs the ball into a faint streak that generic
YOLO weights frequently miss, so detection is a **two-tier** strategy:

1. :class:`YOLOBallDetector` — primary. Wraps an Ultralytics YOLO model
   (ideally fine-tuned on cricket balls; falls back to COCO's ``sports ball``
   class otherwise). Lazy-loads weights so the module imports without the
   dependency installed.

2. :class:`MotionBallDetector` — motion-blur recovery. When YOLO returns
   nothing confident, frame-differencing finds moving blobs; a blurred ball
   shows up as an elongated blob, so the size/aspect filters are deliberately
   permissive.

:class:`BlurAwareBallDetector` fuses the two and adds a temporal gate (predict
the next position from recent velocity) to pick the right blob, plus optional
gap interpolation to keep the track continuous through heavy blur.

The output — a per-frame :class:`BallDetection` in **pixel** coordinates — is
what a downstream calibration stage turns into the world-frame
``BallObservation`` consumed by :mod:`physics.trajectory`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from pipeline.ingest import Frame

logger = logging.getLogger("cricket_ai.detection")


# --------------------------------------------------------------------------- #
# Detection type
# --------------------------------------------------------------------------- #
@dataclass
class BallDetection:
    """A single ball detection in image (pixel) space."""

    frame_index: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    source: str = "yolo"  # "yolo" | "motion" | "interpolated"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """Longer side / shorter side — >1 indicates motion-blur elongation."""
        w, h = max(self.width, 1e-6), max(self.height, 1e-6)
        return max(w, h) / min(w, h)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class BallDetector(Protocol):
    """Detects ball candidates in one frame."""

    def detect_frame(
        self, image: np.ndarray, frame_index: int, timestamp: float
    ) -> List[BallDetection]:  # pragma: no cover - interface
        ...


# --------------------------------------------------------------------------- #
# Primary: YOLO
# --------------------------------------------------------------------------- #
class YOLOBallDetector:
    """Ultralytics YOLO wrapper filtered to the cricket ball.

    Parameters
    ----------
    model_path:
        Path to weights (e.g. a fine-tuned ``cricket_ball.pt``, or ``yolov8n.pt``
        to use the COCO ``sports ball`` proxy).
    conf_threshold:
        Minimum detector confidence to keep a box.
    ball_class_names:
        Class names treated as "the ball" (matched against ``model.names``).
    device:
        Torch device string ("cpu", "cuda:0", ...). None => Ultralytics default.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        ball_class_names: Sequence[str] = ("cricket-ball", "cricket_ball", "ball", "sports ball"),
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.ball_class_names = {n.lower() for n in ball_class_names}
        self.device = device
        self.imgsz = imgsz
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO  # imported lazily
            except ImportError as e:  # pragma: no cover - env dependent
                raise ImportError(
                    "ultralytics is required for YOLOBallDetector. "
                    "Install with `pip install ultralytics`."
                ) from e
            logger.info("Loading YOLO weights: %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def detect_frame(
        self, image: np.ndarray, frame_index: int, timestamp: float
    ) -> List[BallDetection]:
        model = self._load()
        results = model.predict(
            image, conf=self.conf_threshold, imgsz=self.imgsz, device=self.device, verbose=False
        )
        detections: List[BallDetection] = []
        names = model.names  # {class_id: name}
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                name = str(names.get(cls_id, "")).lower()
                if name not in self.ball_class_names:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    BallDetection(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        source="yolo",
                    )
                )
        return detections


# --------------------------------------------------------------------------- #
# Recovery: motion (handles blur)
# --------------------------------------------------------------------------- #
class MotionBallDetector:
    """Frame-difference blob detector for fast / motion-blurred balls.

    A blurred ball is a small, faint, often elongated moving region. This finds
    such regions via consecutive-frame differencing and returns them as
    candidates with a heuristic confidence.

    Parameters
    ----------
    diff_threshold:
        Pixel-intensity change treated as motion.
    min_area / max_area:
        Blob area bounds (px²) — reject specks and large body/bat motion.
    max_aspect:
        Max elongation kept; blur stretches the ball, so this is generous.
    """

    def __init__(
        self,
        diff_threshold: int = 18,
        min_area: float = 6.0,
        max_area: float = 1500.0,
        max_aspect: float = 6.0,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_aspect = max_aspect

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect(
        self,
        prev_image: np.ndarray,
        cur_image: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> List[BallDetection]:
        prev, cur = self._gray(prev_image), self._gray(cur_image)
        diff = cv2.absdiff(cur, prev)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[BallDetection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect:
                continue
            # Heuristic confidence: how solidly the blob fills its box
            # (a compact ball fills more than a random motion sliver).
            fill = area / max(w * h, 1)
            confidence = float(np.clip(0.2 + 0.4 * fill, 0.0, 0.6))
            candidates.append(
                BallDetection(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=confidence,
                    source="motion",
                )
            )
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates


# --------------------------------------------------------------------------- #
# Fused, blur-aware detector
# --------------------------------------------------------------------------- #
class BlurAwareBallDetector:
    """Runs YOLO first, recovers missed (blurred) frames via motion + gating.

    Parameters
    ----------
    yolo:
        Primary detector (any ``BallDetector``). Optional — omit to run
        motion-only (useful for testing / no-weights environments).
    motion:
        Recovery detector. Defaults to :class:`MotionBallDetector`.
    accept_threshold:
        YOLO confidence at/above which the YOLO box is taken outright.
    max_jump:
        Max plausible ball-centre displacement (px) between consecutive frames;
        used to gate motion candidates against the predicted position.
    interpolate_gaps:
        Linearly interpolate ball centres across short runs of missed frames.
    """

    def __init__(
        self,
        yolo: Optional[BallDetector] = None,
        motion: Optional[MotionBallDetector] = None,
        accept_threshold: float = 0.35,
        max_jump: float = 150.0,
        interpolate_gaps: bool = True,
    ) -> None:
        self.yolo = yolo
        self.motion = motion or MotionBallDetector()
        self.accept_threshold = accept_threshold
        self.max_jump = max_jump
        self.interpolate_gaps = interpolate_gaps

    def detect_sequence(self, frames: Sequence[Frame]) -> List[Optional[BallDetection]]:
        """Detect the ball across an ordered frame sequence.

        Returns one entry per frame: a ``BallDetection`` or ``None`` when the
        ball could not be located (before optional interpolation).
        """
        results: List[Optional[BallDetection]] = []
        prev_image: Optional[np.ndarray] = None
        prev_center: Optional[Tuple[float, float]] = None
        last_center: Optional[Tuple[float, float]] = None

        for f in frames:
            det = self._detect_single(f, prev_image, prev_center, last_center)

            if det is not None:
                prev_center = last_center
                last_center = det.center
            results.append(det)
            prev_image = f.image

        if self.interpolate_gaps:
            self._interpolate(results)
        n_found = sum(d is not None for d in results)
        logger.info("Ball located in %d / %d frames", n_found, len(frames))
        return results

    # -- internals --------------------------------------------------------- #
    def _detect_single(
        self,
        frame: Frame,
        prev_image: Optional[np.ndarray],
        prev_center: Optional[Tuple[float, float]],
        last_center: Optional[Tuple[float, float]],
    ) -> Optional[BallDetection]:
        # 1) YOLO primary
        if self.yolo is not None:
            yolo_dets = self.yolo.detect_frame(frame.image, frame.index, frame.timestamp)
            best = max(yolo_dets, key=lambda d: d.confidence, default=None)
            if best is not None and best.confidence >= self.accept_threshold:
                return best

        # 2) Motion recovery (needs a predecessor frame)
        if prev_image is None:
            return None
        candidates = self.motion.detect(prev_image, frame.image, frame.index, frame.timestamp)
        if not candidates:
            return None

        # 3) Temporal gate: prefer the candidate nearest the predicted centre.
        predicted = self._predict(prev_center, last_center)
        if predicted is not None:
            gated = [
                d for d in candidates
                if _dist(d.center, predicted) <= self.max_jump
            ]
            if gated:
                return min(gated, key=lambda d: _dist(d.center, predicted))
            return None  # all candidates implausibly far => reject
        # No motion history yet: take the most ball-like blob.
        return candidates[0]

    @staticmethod
    def _predict(
        prev_center: Optional[Tuple[float, float]],
        last_center: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if last_center is None:
            return None
        if prev_center is None:
            return last_center  # constant-position prior
        # constant-velocity prediction
        vx = last_center[0] - prev_center[0]
        vy = last_center[1] - prev_center[1]
        return (last_center[0] + vx, last_center[1] + vy)

    def _interpolate(self, results: List[Optional[BallDetection]]) -> None:
        """Fill ``None`` gaps between two detections by linear interpolation."""
        known = [i for i, d in enumerate(results) if d is not None]
        for a, b in zip(known, known[1:]):
            if b - a <= 1:
                continue
            da, db = results[a], results[b]
            assert da is not None and db is not None
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                cx = da.center[0] + t * (db.center[0] - da.center[0])
                cy = da.center[1] + t * (db.center[1] - da.center[1])
                # borrow an averaged box size
                hw = (da.width + db.width) / 4.0
                hh = (da.height + db.height) / 4.0
                results[k] = BallDetection(
                    frame_index=k,
                    timestamp=da.timestamp + t * (db.timestamp - da.timestamp),
                    bbox=(cx - hw, cy - hh, cx + hw, cy + hh),
                    confidence=0.15,
                    source="interpolated",
                )


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
