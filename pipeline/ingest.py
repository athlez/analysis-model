"""Video ingestion.

Accepts an mp4 file, decodes frames with OpenCV, records each frame's
timestamp, and packages everything into an `IngestedVideo` ready to be
handed to delivery segmentation.

Two access patterns are supported:

* ``VideoIngestor.iter_frames()``  — lazy generator, O(1) memory. Preferred
  for long match footage that won't fit in RAM.
* ``VideoIngestor.ingest()``       — eager, returns an ``IngestedVideo`` with
  all frames materialised. Convenient for short clips / tests.

Example
-------
>>> with VideoIngestor("over.mp4", sample_rate=2) as ing:
...     for frame in ing.iter_frames():
...         process(frame.image, frame.timestamp)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("cricket_ai.ingest")

_SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov"}


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """A single decoded frame plus where it sits in time."""

    index: int
    """Zero-based position of this frame in the *decoded* stream."""

    source_index: int
    """Zero-based position in the original video (differs when sampling)."""

    timestamp: float
    """Seconds from the start of the video."""

    image: np.ndarray
    """Decoded BGR frame (OpenCV's native channel order)."""


@dataclass
class VideoMetadata:
    """Static properties of the source video."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        """Approximate length in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class IngestedVideo:
    """Everything segmentation needs: metadata, frames, and their timestamps."""

    metadata: VideoMetadata
    frames: List[Frame] = field(default_factory=list)

    @property
    def timestamps(self) -> List[float]:
        """Timestamps (seconds) of the retained frames, in order."""
        return [f.timestamp for f in self.frames]

    def __len__(self) -> int:
        return len(self.frames)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class VideoIngestError(Exception):
    """Raised when a video cannot be opened or decoded."""


# --------------------------------------------------------------------------- #
# Ingestor
# --------------------------------------------------------------------------- #
class VideoIngestor:
    """Opens an mp4 and yields timestamped frames.

    Parameters
    ----------
    path:
        Path to the source video.
    sample_rate:
        Keep every Nth frame (1 = every frame). Reduces work for
        segmentation when full frame-rate isn't needed.
    grayscale:
        If True, decoded frames are converted to single-channel grayscale.
    """

    def __init__(self, path: str, sample_rate: int = 1, grayscale: bool = False) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")

        self.path = path
        self.sample_rate = sample_rate
        self.grayscale = grayscale
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> "VideoIngestor":
        if not os.path.isfile(self.path):
            raise VideoIngestError(f"File not found: {self.path}")

        ext = os.path.splitext(self.path)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise VideoIngestError(
                f"Unsupported extension {ext!r}; expected one of {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise VideoIngestError(f"OpenCV could not open video: {self.path}")

        self._cap = cap
        self._metadata = VideoMetadata(
            path=self.path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        logger.info(
            "Opened %s: %.2f fps, %d frames, %dx%d",
            self.path,
            self._metadata.fps,
            self._metadata.frame_count,
            self._metadata.width,
            self._metadata.height,
        )
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoIngestor":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- properties -------------------------------------------------------- #
    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise VideoIngestError("Video not opened; call open() first.")
        return self._metadata

    # -- extraction -------------------------------------------------------- #
    def _timestamp_for(self, source_index: int) -> float:
        """Timestamp in seconds for a source frame index.

        Prefers the decoder's reported position (handles variable frame
        rate); falls back to index / fps when unavailable.
        """
        assert self._cap is not None
        pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and pos_ms > 0:
            return pos_ms / 1000.0
        fps = self.metadata.fps
        return source_index / fps if fps > 0 else 0.0

    def iter_frames(self) -> Iterator[Frame]:
        """Lazily yield timestamped frames, honouring ``sample_rate``."""
        if self._cap is None:
            raise VideoIngestError("Video not opened; use `with VideoIngestor(...)` or call open().")

        source_index = 0
        kept_index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break

            if source_index % self.sample_rate == 0:
                timestamp = self._timestamp_for(source_index)
                if self.grayscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                yield Frame(
                    index=kept_index,
                    source_index=source_index,
                    timestamp=timestamp,
                    image=image,
                )
                kept_index += 1

            source_index += 1

        logger.info("Decoded %d source frames, kept %d", source_index, kept_index)

    def ingest(self) -> IngestedVideo:
        """Eagerly decode all (sampled) frames into an ``IngestedVideo``.

        Prepares the video for delivery segmentation: bundles metadata,
        frames, and their timestamps in a single object.
        """
        frames = list(self.iter_frames())
        return IngestedVideo(metadata=self.metadata, frames=frames)


def ingest_video(path: str, sample_rate: int = 1, grayscale: bool = False) -> IngestedVideo:
    """Convenience wrapper: open, decode, close, return an ``IngestedVideo``."""
    with VideoIngestor(path, sample_rate=sample_rate, grayscale=grayscale) as ing:
        return ing.ingest()
