"""Delivery splitter.

Takes a full ingested video and automatically carves it into individual
deliveries — no manual trimming. Detection fuses two signals:

1. **Motion cues** (implemented here): per-frame motion energy from frame
   differencing locates bursts of activity (bowler run-up + ball flight) and
   the precise start/end of ball movement.
2. **AI action detection** (pluggable): a ``BowlingActionDetector`` that
   confirms a burst is a genuine bowling action. A trained pose /
   action-recognition model implements this interface; until one is wired in,
   ``HeuristicBowlingDetector`` uses the motion signal itself as a proxy.

The two are combined in :class:`DeliverySplitter`, which emits a list of
``Delivery`` objects (see :mod:`models.delivery`).

Note
----
Segmentation only fills the *temporal* fields of a ``Delivery`` (window +
estimated release) plus a detection ``confidence_score``. The 3D trajectory,
bounce ``pitch_point`` and ``pitch_2d`` line/length are produced by later
pipeline stages; here they carry placeholders and ``status`` reflects that the
delivery is detected-but-not-yet-analysed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from models.delivery import Delivery, DeliveryStatus, Events
from pipeline.ingest import Frame, IngestedVideo

logger = logging.getLogger("cricket_ai.splitter")


# --------------------------------------------------------------------------- #
# Motion cue extraction
# --------------------------------------------------------------------------- #
def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_motion_energy(frames: Sequence[Frame]) -> np.ndarray:
    """Per-frame motion energy via consecutive-frame differencing.

    Returns an array aligned with ``frames`` (energy[0] == 0.0 by definition,
    since motion needs a predecessor). Values are mean absolute pixel change,
    normalised to [0, 1] against the sequence maximum.
    """
    if len(frames) < 2:
        return np.zeros(len(frames), dtype=np.float32)

    energy = np.zeros(len(frames), dtype=np.float32)
    prev = _to_gray(frames[0].image).astype(np.int16)
    for i in range(1, len(frames)):
        cur = _to_gray(frames[i].image).astype(np.int16)
        energy[i] = np.abs(cur - prev).mean()
        prev = cur

    peak = energy.max()
    if peak > 0:
        energy /= peak
    return energy


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple centred moving-average smoother."""
    if window <= 1 or signal.size == 0:
        return signal
    window = min(window, signal.size)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal, kernel, mode="same")


# --------------------------------------------------------------------------- #
# AI detector interface (pluggable)
# --------------------------------------------------------------------------- #
class BowlingActionDetector(Protocol):
    """Confirms which motion bursts are genuine bowling actions.

    Implementations receive the decoded frames plus the motion-energy signal
    and return candidate ``(start_index, end_index)`` windows (inclusive of
    start, exclusive of end) in frame-index space.

    A production implementation wraps a trained action-recognition / pose
    model. The default :class:`HeuristicBowlingDetector` uses only motion.
    """

    def detect(
        self, frames: Sequence[Frame], energy: np.ndarray
    ) -> List[Tuple[int, int]]:  # pragma: no cover - interface
        ...


@dataclass
class HeuristicBowlingDetector:
    """Motion-only bowling-action detector (fully automatic, no manual trimming).

    A bowling action is the burst of high motion from the delivery stride
    through release and follow-through. Its *duration* varies enormously with
    style (a whippy fast action can be <0.2 s; a slow looping action longer), so
    an absolute minimum-duration gate is the wrong tool — it silently drops
    genuine short actions (the original cause of missed deliveries).

    Instead this uses a **peak-relative activation threshold**: a frame is
    "active" when its smoothed motion energy rises a fraction ``activation_ratio``
    of the way from the clip's baseline (median) up to its peak. This adapts to
    each clip and each bowling style. Fragments of one action (individual
    run-up strides, release, follow-through) are then joined with a generous
    ``merge_gap_s`` so one action becomes one window, and only 1-2 frame spikes
    are discarded. Windows are returned strongest-first.

    ``threshold_std`` is retained for backward compatibility (used only as a
    secondary noise floor).
    """

    threshold_std: float = 1.0
    smoothing_window: int = 5
    min_duration_s: float = 0.10       # floor: reject only 1-2 frame spikes
    merge_gap_s: float = 0.6           # join run-up -> release -> follow-through
    activation_ratio: float = 0.30     # threshold height between baseline and peak

    def detect(self, frames: Sequence[Frame], energy: np.ndarray) -> List[Tuple[int, int]]:
        if len(frames) < 2:
            return []

        smooth = _smooth(energy, self.smoothing_window)
        peak = float(smooth.max())
        baseline = float(np.median(smooth))
        if peak <= baseline:
            return []

        # Peak-relative activation, floored by the adaptive noise level so that
        # near-static clips don't light up everything.
        relative = baseline + self.activation_ratio * (peak - baseline)
        noise_floor = float(smooth.mean() + 0.5 * self.threshold_std * smooth.std())
        threshold = max(relative, min(noise_floor, relative * 1.5))
        active = smooth > threshold

        # Estimate frame period from timestamps (median dt is robust to gaps).
        times = np.array([f.timestamp for f in frames], dtype=np.float64)
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        merge_gap_frames = int(round(self.merge_gap_s / dt)) if dt > 0 else 0
        min_frames = int(round(self.min_duration_s / dt)) if dt > 0 else 1

        windows = self._contiguous_regions(active)
        windows = self._merge_close(windows, merge_gap_frames)
        windows = [(s, e) for (s, e) in windows if (e - s) >= max(min_frames, 1)]

        # Guarantee recall: if a clip has clear motion structure but nothing
        # survived (e.g. a single very short burst), keep the region around the
        # global motion peak so a genuine action is never dropped to zero.
        if not windows:
            pk = int(np.argmax(smooth))
            half = max(min_frames, 1)
            windows = [(max(0, pk - half), min(len(smooth), pk + half + 1))]

        # Strongest first (by peak energy within the window).
        windows.sort(key=lambda w: float(energy[w[0]:w[1]].max()), reverse=True)
        return windows

    @staticmethod
    def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
        regions: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, on in enumerate(mask):
            if on and start is None:
                start = i
            elif not on and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, len(mask)))
        return regions

    @staticmethod
    def _merge_close(
        regions: List[Tuple[int, int]], gap: int
    ) -> List[Tuple[int, int]]:
        if not regions:
            return []
        merged = [regions[0]]
        for s, e in regions[1:]:
            ps, pe = merged[-1]
            if s - pe <= gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        return merged


# --------------------------------------------------------------------------- #
# Splitter
# --------------------------------------------------------------------------- #
@dataclass
class DeliverySegment:
    """Internal representation of one detected delivery window."""

    start_index: int
    end_index: int  # exclusive
    start_time: float
    end_time: float
    release_index: int
    release_time: float
    confidence: float


class DeliverySplitter:
    """Splits a full ingested video into individual deliveries.

    Parameters
    ----------
    action_detector:
        Strategy that confirms bowling actions. Defaults to
        :class:`HeuristicBowlingDetector`. Inject a trained model here to use
        real AI detection without changing the rest of the pipeline.
    low_confidence_threshold:
        Detections below this confidence are still emitted but flagged
        ``LOW_CONFIDENCE`` rather than ``SUCCESS``.
    min_confidence:
        Detections below this are dropped entirely (treated as noise).
    """

    def __init__(
        self,
        action_detector: Optional[BowlingActionDetector] = None,
        low_confidence_threshold: float = 0.6,
        min_confidence: float = 0.3,
        secondary_min_confidence: float = 0.85,
        min_separation_s: float = 1.0,
        min_secondary_duration_s: float = 0.2,
    ) -> None:
        # The strongest motion burst is always emitted (recall guarantee).
        # Additional bursts are only emitted when they are strongly prominent
        # (>= secondary_min_confidence), long enough to be a real action
        # (>= min_secondary_duration_s), and clearly separated in time
        # (> min_separation_s) — this admits genuine multi-delivery clips while
        # not counting run-up strides or follow-through as extra deliveries.
        self.secondary_min_confidence = secondary_min_confidence
        self.min_separation_s = min_separation_s
        self.min_secondary_duration_s = min_secondary_duration_s
        self.action_detector: BowlingActionDetector = (
            action_detector or HeuristicBowlingDetector()
        )
        self.low_confidence_threshold = low_confidence_threshold
        self.min_confidence = min_confidence

    def split(self, video: IngestedVideo) -> List[Delivery]:
        """Detect and return every delivery in ``video`` as a ``Delivery``."""
        frames = video.frames
        if len(frames) < 2:
            logger.warning("Video has < 2 frames; nothing to split.")
            return []

        energy = compute_motion_energy(frames)
        windows = self.action_detector.detect(frames, energy)
        logger.info("Detected %d candidate bowling action(s)", len(windows))

        video_id = self._video_id(video)

        # Build candidate segments (windows arrive strongest-first).
        segments = [
            self._to_segment(frames, energy, s, e) for (s, e) in windows
        ]
        segments = [s for s in segments if s.confidence >= self.min_confidence]

        # Selection: always keep the strongest; add prominent, well-separated
        # secondaries only.
        segments.sort(key=lambda s: s.confidence, reverse=True)
        kept: List[DeliverySegment] = []
        for seg in segments:
            if not kept:
                kept.append(seg)
                continue
            if seg.confidence < self.secondary_min_confidence:
                continue
            if (seg.end_time - seg.start_time) < self.min_secondary_duration_s:
                continue
            separated = all(
                seg.start_time > k.end_time + self.min_separation_s
                or seg.end_time < k.start_time - self.min_separation_s
                for k in kept
            )
            if separated:
                kept.append(seg)

        kept.sort(key=lambda s: s.start_time)
        deliveries = [self._build_delivery(video_id, seg) for seg in kept]
        logger.info("Emitting %d deliveries", len(deliveries))
        return deliveries

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _video_id(video: IngestedVideo) -> str:
        import os

        return os.path.splitext(os.path.basename(video.metadata.path))[0]

    def _to_segment(
        self,
        frames: Sequence[Frame],
        energy: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> DeliverySegment:
        """Refine a raw window into a segment.

        The detector's activation threshold has already trimmed idle lead-in, so
        the window bounds *are* the action; we keep them (guaranteeing a
        non-empty span) rather than re-trimming to the half-peak, which used to
        collapse spiky windows to a single frame. Release is estimated as the
        motion peak inside the window.
        """
        window = energy[start_idx:end_idx]
        move_start = start_idx
        move_end = max(end_idx, start_idx + 2)  # never empty

        # Release ≈ motion peak within the window.
        peak_offset = int(np.argmax(window)) if window.size else 0
        release_index = start_idx + peak_offset

        # Confidence: how strongly this window's peak stands out globally.
        local_peak = float(window.max()) if window.size else 0.0
        global_peak = float(energy.max()) if energy.size else 0.0
        confidence = float(local_peak / global_peak) if global_peak > 0 else 0.0

        return DeliverySegment(
            start_index=move_start,
            end_index=move_end,
            start_time=frames[move_start].timestamp,
            end_time=frames[min(move_end, len(frames)) - 1].timestamp,
            release_index=frames[release_index].source_index,
            release_time=frames[release_index].timestamp,
            confidence=round(confidence, 3),
        )

    def _build_delivery(self, video_id: str, seg: DeliverySegment) -> Delivery:
        status = (
            DeliveryStatus.SUCCESS
            if seg.confidence >= self.low_confidence_threshold
            else DeliveryStatus.LOW_CONFIDENCE
        )
        return Delivery(
            video_id=video_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            events=Events(
                release_frame=seg.release_index,
                release_estimated=True,  # motion-peak estimate, not detected
                pitch_point=None,  # filled by trajectory-reconstruction stage
            ),
            trajectory_3d=[],  # filled by 3D reconstruction stage
            pitch_2d=None,  # derived from the trajectory later
            speed_kmph=None,
            confidence_score=seg.confidence,
            status=status,
        )


def split_video(video: IngestedVideo, **kwargs) -> List[Delivery]:
    """Convenience wrapper around :class:`DeliverySplitter`."""
    return DeliverySplitter(**kwargs).split(video)
