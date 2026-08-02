"""cricket-ai-mvp entry point.

Runs the end-to-end pipeline on a source video:

    Video -> Ingestion -> Segmentation -> Ball Detection
          -> (pixel track) -> Projection seam -> Trajectory -> Deliveries

Usage:
    python main.py <video.mp4> [output_dir]
"""

from __future__ import annotations

import logging
import sys

from pipeline import CalibrationError, CricketPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cricket_ai")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    video_path = argv[1]
    output_dir = argv[2] if len(argv) > 2 else "output"

    # Swap PixelPlaneProjector -> CameraCalibrationProjector with one line:
    #   pipeline = CricketPipeline(projector=CameraCalibrationProjector())
    pipeline = CricketPipeline()
    try:
        result = pipeline.run(video_path, output_dir=output_dir)
    except CalibrationError as e:
        print("\nVIDEO REJECTED — could not calibrate the camera:")
        for reason in e.reasons:
            print(f"  - {reason}")
        print(
            "\nRecord again with the whole pitch clearly visible from a fixed "
            "position behind the bowler, in good lighting, and keep the camera "
            "still for the session."
        )
        return 1

    print(f"\nVideo: {result.metadata.path}")
    print(f"  {result.metadata.fps:.1f} fps, {result.metadata.duration:.2f}s")
    print(f"  Deliveries detected: {len(result.deliveries)}\n")
    for diag in result.diagnostics:
        print(
            f"  D{diag.index} [{diag.window}] "
            f"frames={diag.n_frames} detections={diag.n_detections} "
            f"observations={diag.n_observations} traj_pts={diag.n_trajectory_points} "
            f"status={diag.status}"
        )
    print(f"\nWrote results to {output_dir}/deliveries.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
