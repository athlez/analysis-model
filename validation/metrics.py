"""Per-stage evaluation metrics for the pipeline.

Every function returns a plain dict (JSON-serialisable). Metrics degrade
gracefully: with ground-truth annotations they report accuracy; without them
they report quality/proxy signals (detection rate, fit residual, ...).

Thresholds below decide whether a stage is considered to have "passed" for the
failure report. They are conservative MVP defaults, not tuned values.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# --- pass/fail thresholds (MVP defaults) ----------------------------------- #
MIN_FRAMES = 2
MIN_DETECTION_RATE = 0.30          # fraction of delivery frames with a ball
MIN_TRACK_RUN = 5                  # longest contiguous detected run (frames)
MIN_OBSERVATIONS = 3               # needed to fit a trajectory
SPEED_PLAUSIBLE_KMPH = (40.0, 170.0)
MAX_TRAJ_RESIDUAL_M = 0.50


def ingestion_metrics(video) -> Dict:
    m = video.metadata
    return {
        "n_frames": len(video),
        "fps": round(m.fps, 2),
        "duration_s": round(m.duration, 2),
        "resolution": f"{m.width}x{m.height}",
        "ok": len(video) >= MIN_FRAMES,
    }


def calibration_metrics(calib_ok: bool, info: Dict) -> Dict:
    out = {"ok": calib_ok, **info}
    return out


def segmentation_metrics(deliveries, annotation: Optional[Dict]) -> Dict:
    n = len(deliveries)
    out = {"n_deliveries": n, "ok": n > 0}
    if annotation and "num_deliveries" in annotation:
        gt = annotation["num_deliveries"]
        out["gt_deliveries"] = gt
        out["count_error"] = n - gt
    return out


def detection_metrics(detections_per_delivery: List[List[Optional[object]]]) -> Dict:
    """`detections_per_delivery`: for each delivery, a per-frame list where each
    item is a BallDetection or None."""
    rates, confidences, sources, gaps = [], [], {}, []
    for dets in detections_per_delivery:
        if not dets:
            continue
        found = [d for d in dets if d is not None]
        rates.append(len(found) / len(dets))
        for d in found:
            confidences.append(d.confidence)
            sources[d.source] = sources.get(d.source, 0) + 1
        gaps.append(_longest_gap([d is not None for d in dets]))

    mean_rate = sum(rates) / len(rates) if rates else 0.0
    return {
        "mean_detection_rate": round(mean_rate, 3),
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "source_breakdown": sources,
        "max_gap_frames": max(gaps) if gaps else 0,
        "ok": bool(rates) and mean_rate >= MIN_DETECTION_RATE,
    }


def tracking_metrics(tracks_per_delivery: List[List[object]]) -> Dict:
    """`tracks_per_delivery`: for each delivery, the ordered non-None detections."""
    longest_runs, monotonic_flags = [], []
    for track in tracks_per_delivery:
        longest_runs.append(len(track))  # track is already the kept detections
        xs = [d.center[0] for d in track]
        monotonic_flags.append(_mostly_monotonic(xs))
    best_run = max(longest_runs) if longest_runs else 0
    return {
        "best_track_length": best_run,
        "monotonic_fraction": round(sum(monotonic_flags) / len(monotonic_flags), 3)
        if monotonic_flags else 0.0,
        "ok": best_run >= MIN_TRACK_RUN,
    }


def trajectory_metrics(recons: List[Optional[object]], annotation: Optional[Dict]) -> Dict:
    """`recons`: per-delivery ReconstructionResult or None."""
    n_points, residuals, speeds, bounces = [], [], [], 0
    for r in recons:
        if r is None:
            continue
        n_points.append(len(r.points))
        residuals.append(r.residual)
        if r.speed_kmph is not None:
            speeds.append(r.speed_kmph)
        if r.pitch_point is not None:
            bounces += 1

    plausible_speed = [
        s for s in speeds if SPEED_PLAUSIBLE_KMPH[0] <= s <= SPEED_PLAUSIBLE_KMPH[1]
    ]
    mean_resid = sum(residuals) / len(residuals) if residuals else None
    out = {
        "n_reconstructed": len(n_points),
        "bounces_found": bounces,
        "mean_residual_m": round(mean_resid, 4) if mean_resid is not None else None,
        "speeds_kmph": [round(s, 1) for s in speeds],
        "plausible_speed_fraction": round(len(plausible_speed) / len(speeds), 3)
        if speeds else 0.0,
        "ok": bool(n_points) and (mean_resid is not None and mean_resid <= MAX_TRAJ_RESIDUAL_M),
    }
    if annotation and "deliveries" in annotation:
        gt_speeds = [d["speed_kmph"] for d in annotation["deliveries"] if "speed_kmph" in d]
        if gt_speeds and speeds:
            # naive positional pairing for a coarse speed-error signal
            pairs = list(zip(speeds, gt_speeds))
            out["mean_speed_abs_error_kmph"] = round(
                sum(abs(a - b) for a, b in pairs) / len(pairs), 1
            )
    return out


# --------------------------------------------------------------------------- #
# Failure attribution
# --------------------------------------------------------------------------- #
STAGE_ORDER = ["ingestion", "calibration", "segmentation", "detection", "tracking", "reconstruction"]


def first_failure(stage_metrics: Dict[str, Dict]) -> Optional[str]:
    """Earliest stage (in pipeline order) whose ``ok`` is explicitly False."""
    for stage in STAGE_ORDER:
        m = stage_metrics.get(stage)
        if m is not None and m.get("ok") is False:
            return stage
    return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _longest_gap(present: Sequence[bool]) -> int:
    longest = cur = 0
    for p in present:
        cur = 0 if p else cur + 1
        longest = max(longest, cur)
    return longest


def _mostly_monotonic(values: Sequence[float], tol: float = 0.9) -> bool:
    if len(values) < 2:
        return False
    inc = sum(1 for a, b in zip(values, values[1:]) if b >= a)
    dec = len(values) - 1 - inc
    return max(inc, dec) / (len(values) - 1) >= tol
