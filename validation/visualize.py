"""Developer-only debugging visualizations.

Only invoked when the validator runs with ``--debug``. Requires matplotlib
(and OpenCV, already a pipeline dependency). All functions are best-effort and
write PNGs into the video's ``debug/`` folder — they are diagnostics for
developers, not user-facing output.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger("cricket_ai.validation.viz")


def _plt():
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    return plt


def save_motion_energy(dbg_dir: str, frames, energy: np.ndarray, deliveries) -> None:
    """Motion-energy curve with detected delivery windows shaded."""
    plt = _plt()
    t = [f.timestamp for f in frames]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, energy, lw=1, color="tab:blue")
    for d in deliveries:
        ax.axvspan(d.start_time, d.end_time, color="tab:orange", alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("motion energy")
    ax.set_title("Segmentation: motion energy + detected deliveries")
    fig.tight_layout()
    fig.savefig(os.path.join(dbg_dir, "motion_energy.png"), dpi=110)
    plt.close(fig)


def save_detection_overlay(
    dbg_dir: str, index: int, frames, detections: Sequence[Optional[object]], max_frames: int = 6
) -> None:
    """Montage of sampled delivery frames with the ball box drawn."""
    if not frames:
        return
    step = max(1, len(frames) // max_frames)
    picks = list(range(0, len(frames), step))[:max_frames]

    tiles = []
    for i in picks:
        img = frames[i].image
        img = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        det = detections[i] if i < len(detections) else None
        if det is not None:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            colour = {"yolo": (0, 255, 0), "motion": (0, 200, 255),
                      "interpolated": (200, 200, 200)}.get(det.source, (255, 0, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img, f"{det.source} {det.confidence:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        tiles.append(img)

    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    montage = np.hstack(tiles)
    cv2.imwrite(os.path.join(dbg_dir, f"delivery{index}_detections.png"), montage)


def save_trajectory_plot(dbg_dir: str, index: int, recon) -> None:
    """Side view (down-pitch vs height) and top view (lateral vs down-pitch)."""
    plt = _plt()
    pts = recon.points
    if not pts:
        return
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    xs = [p.x for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ys, zs, "-o", ms=2, color="tab:green")
    if recon.pitch_point is not None:
        ax1.scatter([recon.pitch_point.y], [0], color="red", zorder=5, label="bounce")
        ax1.legend()
    ax1.set_xlabel("down-pitch y (m)"); ax1.set_ylabel("height z (m)")
    ax1.set_title(f"Delivery {index} — side view")

    ax2.plot(xs, ys, "-o", ms=2, color="tab:purple")
    ax2.set_xlabel("lateral x (m)"); ax2.set_ylabel("down-pitch y (m)")
    ax2.set_title(f"speed={recon.speed_kmph} km/h  resid={recon.residual} m")

    fig.tight_layout()
    fig.savefig(os.path.join(dbg_dir, f"delivery{index}_trajectory.png"), dpi=110)
    plt.close(fig)
