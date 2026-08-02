"""Aggregate failure report across all validated videos."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import List

from validation.harness import VideoEvaluation
from validation.metrics import STAGE_ORDER


def write_report(evaluations: List[VideoEvaluation], out_root: str) -> str:
    """Write ``report.md`` + ``summary.json``; return the report path."""
    md = _build_markdown(evaluations)
    report_path = os.path.join(out_root, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    summary = {
        "n_videos": len(evaluations),
        "failure_stage_counts": dict(_failure_counts(evaluations)),
        "videos": [
            {"video": e.video, "first_failure": e.first_failure, "notes": e.notes}
            for e in evaluations
        ],
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return report_path


def _failure_counts(evaluations) -> Counter:
    c = Counter()
    for e in evaluations:
        c[e.first_failure or "none (completed)"] += 1
    return c


def _build_markdown(evaluations: List[VideoEvaluation]) -> str:
    n = len(evaluations)
    lines: List[str] = []
    lines.append("# Pipeline validation report\n")
    lines.append(f"Videos evaluated: **{n}**\n")

    # --- where the pipeline fails (the headline) ---
    counts = _failure_counts(evaluations)
    lines.append("## Where the pipeline fails\n")
    lines.append("| First failing stage | Videos | Share |")
    lines.append("|---|---:|---:|")
    for stage, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {stage} | {cnt} | {cnt / n:.0%} |" if n else f"| {stage} | {cnt} | - |")
    lines.append("")

    # --- stage pass rates ---
    lines.append("## Stage pass rates\n")
    lines.append("| Stage | Passed | Rate |")
    lines.append("|---|---:|---:|")
    for stage in STAGE_ORDER:
        passed = sum(
            1 for e in evaluations
            if e.stage_metrics.get(stage, {}).get("ok") is True
        )
        applicable = sum(
            1 for e in evaluations
            if e.stage_metrics.get(stage, {}).get("ok") is not None
        )
        rate = f"{passed / applicable:.0%}" if applicable else "n/a"
        lines.append(f"| {stage} | {passed}/{applicable} | {rate} |")
    lines.append("")

    # --- per-video breakdown ---
    lines.append("## Per-video breakdown\n")
    lines.append(
        "| Video | Deliveries | Calib | Det.rate | Best track | Recon | Fail stage |"
    )
    lines.append("|---|---:|:---:|---:|---:|:---:|---|")
    for e in evaluations:
        sm = e.stage_metrics
        deliveries = sm.get("segmentation", {}).get("n_deliveries", "-")
        calib = "ok" if sm.get("calibration", {}).get("ok") else "FAIL"
        det = sm.get("detection", {}).get("mean_detection_rate", "-")
        track = sm.get("tracking", {}).get("best_track_length", "-")
        recon = "ok" if sm.get("reconstruction", {}).get("ok") else "no"
        fail = e.first_failure or "—"
        lines.append(
            f"| {e.video} | {deliveries} | {calib} | {det} | {track} | {recon} | {fail} |"
        )
    lines.append("")

    # --- notes ---
    noted = [e for e in evaluations if e.notes]
    if noted:
        lines.append("## Notes\n")
        for e in noted:
            for note in e.notes:
                lines.append(f"- **{e.video}**: {note}")
        lines.append("")

    lines.append(
        "> Metrics reflect the pipeline *as-is* (no tuning). "
        "Where ground-truth annotations were absent, quality/proxy metrics were "
        "used instead of accuracy.\n"
    )
    return "\n".join(lines)
