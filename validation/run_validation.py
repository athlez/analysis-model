"""Validation CLI: run the pipeline over a folder of real videos and report.

Usage:
    python -m validation.run_validation [--data-dir DIR] [--out DIR] [--debug]

Steps:
    1. discover videos in --data-dir (default data/test_videos)
    2. load optional ground truth from <data-dir>/annotations/<clip>.json
    3. run the instrumented pipeline on each, saving intermediates
    4. write one aggregate failure report (report.md + summary.json)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import List, Optional

from validation.harness import VideoEvaluation, evaluate_video
from validation.report import write_report

logger = logging.getLogger("cricket_ai.validation")

_VIDEO_EXTS = ("*.mp4", "*.m4v", "*.mov")


def _discover(data_dir: str) -> List[str]:
    files: List[str] = []
    for ext in _VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    return sorted(files)


def _load_annotation(data_dir: str, video_path: str) -> Optional[dict]:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(data_dir, "annotations", f"{stem}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the cricket pipeline on real videos.")
    parser.add_argument("--data-dir", default=os.path.join("data", "test_videos"))
    parser.add_argument("--out", default=os.path.join("output", "validation"))
    parser.add_argument("--debug", action="store_true", help="render developer visualizations")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    videos = _discover(args.data_dir)
    if not videos:
        print(
            f"No videos found in {args.data_dir!r}.\n"
            f"Drop real cricket clips there (see {args.data_dir}/README.md) and re-run."
        )
        return 1

    os.makedirs(args.out, exist_ok=True)
    evaluations: List[VideoEvaluation] = []
    for path in videos:
        logger.info("Evaluating %s", path)
        annotation = _load_annotation(args.data_dir, path)
        try:
            ev = evaluate_video(path, args.out, annotation=annotation, debug=args.debug)
        except Exception as e:  # keep going; record the crash
            logger.exception("Unhandled error on %s", path)
            ev = VideoEvaluation(video=os.path.basename(path))
            ev.first_failure = "crashed"
            ev.notes.append(f"Unhandled exception: {e}")
        evaluations.append(ev)

    report_path = write_report(evaluations, args.out)
    print(f"\nEvaluated {len(evaluations)} video(s).")
    print(f"Report:  {report_path}")
    print(f"Summary: {os.path.join(args.out, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
