"""Instrumented pipeline runner for validation.

Runs the *existing* pipeline modules on one video, capturing every intermediate
so we can measure where things break. This does NOT modify the pipeline — it
wires the same components together with instrumentation, and falls back to the
non-metric ``PixelPlaneProjector`` when calibration is rejected so that the
downstream stages can still be measured.

Intermediates saved per video (under ``<out>/<clip>/``):

* ``deliveries.json``      — final enriched Delivery objects
* ``ball_detections.json`` — per-frame detections (incl. misses) per delivery
* ``ball_track.json``      — the kept ball track per delivery
* ``calibration.json``     — calibration diagnostics or rejection reasons
* ``trajectory.json``      — reconstructed 3D trajectory per delivery
* ``metrics.json``         — per-stage metrics + failure attribution
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.delivery import Delivery
from physics.trajectory import ReconstructionResult, TrajectoryReconstructor, apply_to_delivery
from pipeline.calibration import CalibrationError, CameraCalibrationProjector
from pipeline.detection import BallDetection, BlurAwareBallDetector
from pipeline.ingest import Frame, IngestedVideo, ingest_video
from pipeline.projection import ObservationProjector, PixelPlaneProjector
from pipeline.splitter import DeliverySplitter, compute_motion_energy

from validation import metrics as M

logger = logging.getLogger("cricket_ai.validation")


@dataclass
class VideoEvaluation:
    """Aggregated result for one video."""

    video: str
    stage_metrics: Dict[str, dict] = field(default_factory=dict)
    first_failure: Optional[str] = None
    notes: List[str] = field(default_factory=list)


_MEM_BUDGET_BYTES = 2.0e9  # ~2 GB of decoded grayscale frames per video


def _memory_safe_sample_rate(video_path: str) -> int:
    """Choose a frame stride bounding decoded-grayscale memory (and ~30 fps).

    Reads size/count via capture properties (no full decode). Returns 1 for
    normal clips; >1 only for large/high-fps footage.
    """
    import math
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    gray_bytes = max(w * h * n, 1)
    rate_mem = math.ceil(gray_bytes / _MEM_BUDGET_BYTES)
    rate_fps = max(1, round(fps / 30.0))
    return max(1, rate_mem, rate_fps)


def evaluate_video(
    video_path: str,
    out_root: str,
    annotation: Optional[dict] = None,
    debug: bool = False,
) -> VideoEvaluation:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    vdir = os.path.join(out_root, stem)
    os.makedirs(vdir, exist_ok=True)

    ev = VideoEvaluation(video=stem)

    # -- 1. Ingestion ------------------------------------------------------ #
    # Memory guard (validation-runtime only, NOT a pipeline change): decode in
    # grayscale — the motion-only pipeline converts to gray internally, so
    # results are identical — and temporally subsample very large clips so a
    # single 4K video cannot exhaust RAM. Uses ingest_video's existing options.
    sample_rate = _memory_safe_sample_rate(video_path)
    if sample_rate > 1:
        ev.notes.append(f"Large video: temporally subsampled (sample_rate={sample_rate}) for memory.")
    video = ingest_video(video_path, sample_rate=sample_rate, grayscale=True)
    ev.stage_metrics["ingestion"] = M.ingestion_metrics(video)
    if not ev.stage_metrics["ingestion"]["ok"]:
        ev.notes.append("Too few frames to process.")
        ev.first_failure = M.first_failure(ev.stage_metrics)
        _dump(vdir, "metrics.json", _eval_payload(ev))
        return ev

    # -- 2. Calibration (with graceful fallback) --------------------------- #
    projector: ObservationProjector = CameraCalibrationProjector()
    calib_ok, calib_info = True, {}
    try:
        projector.prepare(video)
        cal = projector.calibration  # type: ignore[attr-defined]
        calib_info = {
            "focal_px": round(cal.focal, 1),
            "reprojection_rms_px": round(cal.reprojection_rms, 3),
            "confidence": round(cal.confidence, 3),
            "camera_center_m": [round(float(c), 3) for c in cal.camera_center],
        }
    except CalibrationError as e:
        calib_ok = False
        calib_info = {"reasons": e.reasons}
        ev.notes.append("Calibration rejected; downstream measured with non-metric fallback.")
        projector = PixelPlaneProjector()
        projector.prepare(video)
    ev.stage_metrics["calibration"] = M.calibration_metrics(calib_ok, calib_info)
    _dump(vdir, "calibration.json", ev.stage_metrics["calibration"])

    # -- 3. Segmentation --------------------------------------------------- #
    splitter = DeliverySplitter()
    deliveries = splitter.split(video)
    ev.stage_metrics["segmentation"] = M.segmentation_metrics(deliveries, annotation)

    # -- 4/5/6. Detection -> track -> projection -> reconstruction --------- #
    detector = BlurAwareBallDetector(yolo=None)  # motion-only (no weights)
    reconstructor = TrajectoryReconstructor()

    detections_all: List[List[Optional[BallDetection]]] = []
    tracks_all: List[List[BallDetection]] = []
    recons_all: List[Optional[ReconstructionResult]] = []
    enriched: List[Delivery] = []
    per_delivery_frames: List[List[Frame]] = []

    for d in deliveries:
        frames = [f for f in video.frames if d.start_time <= f.timestamp <= d.end_time]
        per_delivery_frames.append(frames)

        dets = detector.detect_sequence(frames)
        track = [x for x in dets if x is not None]
        observations = projector.project(track, video.metadata)

        recon = None
        if len(observations) >= M.MIN_OBSERVATIONS:
            recon = reconstructor.reconstruct(observations)
            d = apply_to_delivery(d, recon)

        detections_all.append(dets)
        tracks_all.append(track)
        recons_all.append(recon)
        enriched.append(d)

    ev.stage_metrics["detection"] = M.detection_metrics(detections_all)
    ev.stage_metrics["tracking"] = M.tracking_metrics(tracks_all)
    ev.stage_metrics["reconstruction"] = M.trajectory_metrics(recons_all, annotation)

    # -- persist intermediates -------------------------------------------- #
    _dump(vdir, "deliveries.json", [d.model_dump(mode="json") for d in enriched])
    _dump(vdir, "ball_detections.json", _serialize_detections(detections_all))
    _dump(vdir, "ball_track.json", _serialize_tracks(tracks_all))
    _dump(vdir, "trajectory.json", _serialize_trajectories(recons_all))

    ev.first_failure = M.first_failure(ev.stage_metrics)
    _dump(vdir, "metrics.json", _eval_payload(ev))

    # -- developer-only visualizations ------------------------------------ #
    if debug:
        _render_debug(vdir, video, deliveries, per_delivery_frames, detections_all, recons_all)

    return ev


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _det_to_dict(d: Optional[BallDetection]) -> Optional[dict]:
    if d is None:
        return None
    return {
        "frame_index": d.frame_index,
        "timestamp": round(d.timestamp, 4),
        "bbox": [round(v, 2) for v in d.bbox],
        "center": [round(c, 2) for c in d.center],
        "confidence": round(d.confidence, 3),
        "source": d.source,
    }


def _serialize_detections(detections_all) -> dict:
    return {str(i): [_det_to_dict(d) for d in dets] for i, dets in enumerate(detections_all)}


def _serialize_tracks(tracks_all) -> dict:
    return {str(i): [_det_to_dict(d) for d in track] for i, track in enumerate(tracks_all)}


def _serialize_trajectories(recons_all) -> dict:
    out = {}
    for i, r in enumerate(recons_all):
        if r is None:
            out[str(i)] = None
            continue
        out[str(i)] = {
            "points": [p.model_dump() for p in r.points],
            "pitch_point": (None if r.pitch_point is None
                            else {"x": r.pitch_point.x, "y": r.pitch_point.y}),
            "bounce_time": r.bounce_time,
            "speed_kmph": r.speed_kmph,
            "line": None if r.line is None else r.line.value,
            "length": None if r.length is None else r.length.value,
            "residual_m": r.residual,
        }
    return out


def _eval_payload(ev: VideoEvaluation) -> dict:
    return {
        "video": ev.video,
        "first_failure": ev.first_failure,
        "notes": ev.notes,
        "stages": ev.stage_metrics,
    }


def _dump(vdir: str, name: str, payload) -> None:
    with open(os.path.join(vdir, name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _render_debug(vdir, video, deliveries, per_delivery_frames, detections_all, recons_all) -> None:
    """Best-effort dev visualizations; skipped if matplotlib is unavailable."""
    try:
        from validation import visualize as V
    except Exception as e:  # pragma: no cover
        logger.warning("Debug visualizations unavailable: %s", e)
        return

    dbg = os.path.join(vdir, "debug")
    os.makedirs(dbg, exist_ok=True)
    energy = compute_motion_energy(video.frames)
    V.save_motion_energy(dbg, video.frames, energy, deliveries)
    for i, (frames, dets, recon) in enumerate(zip(per_delivery_frames, detections_all, recons_all)):
        V.save_detection_overlay(dbg, i, frames, dets)
        if recon is not None:
            V.save_trajectory_plot(dbg, i, recon)
