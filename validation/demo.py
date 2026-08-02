"""Demonstration renderer — visualize the CURRENT pipeline outputs on a video.

This does NOT modify or improve any algorithm. It runs the existing components
exactly as they are (calibration with the existing PixelPlaneProjector fallback,
segmentation, ball detection, tracking, trajectory reconstruction), captures
every prediction, and overlays them frame by frame. It shows exactly what the
model currently believes — failures included.

Outputs:
  output/demo/<name>_output.mp4    annotated video + 5s summary screen
  output/demo/<name>_report.json   every prediction used to draw it

Usage:
  python -m validation.demo data/raw/own_recordings/bowling2.mp4
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from pipeline.ingest import ingest_video
from pipeline.calibration import CameraCalibrationProjector, CalibrationError
from pipeline.projection import PixelPlaneProjector
from pipeline.splitter import DeliverySplitter
from pipeline.detection import BlurAwareBallDetector
from physics.trajectory import TrajectoryReconstructor, apply_to_delivery

# colours (BGR)
GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
CYAN = (255, 220, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
GREY = (170, 170, 170)

MIN_TRACK_RUN = 5  # mirrors validation.metrics (not an algorithm change)


# --------------------------------------------------------------------------- #
# Run the pipeline exactly as-is, capturing every intermediate
# --------------------------------------------------------------------------- #
def run_pipeline_capture(video_path: str, weights: str = None, conf: float = 0.15,
                         det_imgsz: int = 1280) -> dict:
    t0 = time.time()
    video = ingest_video(video_path)  # native resolution / colour, sample_rate=1
    meta = video.metadata

    # Calibration: attempt the real projector, fall back exactly as implemented.
    projector = CameraCalibrationProjector()
    calib = {"ok": False, "reason": None, "focal": None, "rms": None, "camera_center": None}
    try:
        projector.prepare(video)
        c = projector.calibration
        calib.update(ok=True, focal=round(c.focal, 1),
                     rms=round(c.reprojection_rms, 3),
                     camera_center=[round(float(v), 3) for v in c.camera_center])
    except CalibrationError as e:
        calib["reason"] = e.reasons[0]
        projector = PixelPlaneProjector()
        projector.prepare(video)

    splitter = DeliverySplitter()
    deliveries = splitter.split(video)

    # Detection: plug the trained YOLO in as the primary detector when weights
    # are supplied (falls back to motion-only otherwise). Detection wiring only.
    yolo = None
    if weights:
        from pipeline.detection import YOLOBallDetector
        yolo = YOLOBallDetector(weights, conf_threshold=conf, imgsz=det_imgsz)
    detector = BlurAwareBallDetector(yolo=yolo, accept_threshold=conf)
    reconstructor = TrajectoryReconstructor()

    reproject = _make_reprojector(projector, calib["ok"], meta)

    captured = []
    frame_det: Dict[int, dict] = {}          # source frame index -> detection
    frame_delivery: Dict[int, int] = {}      # source frame index -> delivery idx
    total_window_frames = 0
    total_detections = 0

    for i, d in enumerate(deliveries):
        window = [f for f in video.frames if d.start_time <= f.timestamp <= d.end_time]
        detections = detector.detect_sequence(window)
        track = [x for x in detections if x is not None]
        observations = projector.project(track, meta)

        recon = None
        if len(observations) >= 3:
            recon = reconstructor.reconstruct(observations)
            d = apply_to_delivery(d, recon)

        # map detections to their source frame index (robust to interp indexing)
        det_list = []
        for f, det in zip(window, detections):
            frame_delivery[f.index] = i
            if det is not None:
                info = {"frame_index": f.index, "timestamp": round(det.timestamp, 4),
                        "bbox": [round(v, 1) for v in det.bbox],
                        "center": [round(v, 1) for v in det.center],
                        "confidence": round(det.confidence, 3), "source": det.source}
                frame_det[f.index] = info
                det_list.append(info)
        total_window_frames += len(window)
        total_detections += len(track)

        # reproject trajectory + bounce for drawing
        traj_uv = []
        traj_pts = []
        if recon is not None:
            for p in recon.points:
                uv = reproject((p.x, p.y, p.z))
                traj_uv.append(uv)
                traj_pts.append({"x": round(p.x, 3), "y": round(p.y, 3),
                                 "z": round(p.z, 3), "t": round(p.timestamp, 4)})
        bounce_uv = None
        if recon is not None and recon.bounce_time is not None and recon.points:
            nearest = min(recon.points, key=lambda p: abs(p.timestamp - recon.bounce_time))
            bounce_uv = reproject((nearest.x, nearest.y, nearest.z))

        release_frame = d.events.release_frame
        release_uv = None
        if release_frame is not None and release_frame in frame_det:
            release_uv = frame_det[release_frame]["center"]

        captured.append({
            "index": i,
            "start_time": round(d.start_time, 3), "end_time": round(d.end_time, 3),
            "start_frame": window[0].index if window else None,
            "end_frame": window[-1].index if window else None,
            "status": d.status.value,
            "confidence_score": round(d.confidence_score, 3),
            "speed_kmph": d.speed_kmph,
            "release_frame": release_frame,
            "release_uv": release_uv,
            "pitch_point": (None if d.events.pitch_point is None
                            else {"x": round(d.events.pitch_point.x, 3),
                                  "y": round(d.events.pitch_point.y, 3)}),
            "pitch_2d": (None if d.pitch_2d is None
                         else {"line": str(getattr(d.pitch_2d.line, "value", d.pitch_2d.line)),
                               "length": d.pitch_2d.length.value}),
            "bounce_time": None if recon is None else recon.bounce_time,
            "residual_m": None if recon is None else round(recon.residual, 4),
            "track": det_list,
            "trajectory_3d": traj_pts,
            "_traj_uv": traj_uv,        # drawing only (stripped from report)
            "_bounce_uv": bounce_uv,
            "n_track": len(track),
        })

    elapsed = time.time() - t0

    tracking_ok = any(c["n_track"] >= MIN_TRACK_RUN for c in captured)
    trajectory_ok = any(len(c["trajectory_3d"]) > 0 for c in captured)
    det_rate = (total_detections / total_window_frames) if total_window_frames else 0.0
    speeds = [c["speed_kmph"] for c in captured if c["speed_kmph"] is not None]

    return {
        "video": os.path.basename(video_path),
        "meta": {"width": meta.width, "height": meta.height, "fps": round(meta.fps, 3),
                 "frames": len(video), "duration_s": round(meta.duration, 3)},
        "calibration": calib,
        "deliveries": captured,
        "summary": {
            "n_deliveries": len(captured),
            "detection_rate": round(det_rate, 3),
            "tracking_ok": tracking_ok,
            "trajectory_ok": trajectory_ok,
            "calibration_ok": calib["ok"],
            "speed_kmph": (round(float(np.mean(speeds)), 1) if speeds else None),
            "confidence": (round(float(np.mean([c["confidence_score"] for c in captured])), 3)
                           if captured else None),
            "processing_time_s": round(elapsed, 2),
        },
        "_video": video,  # frames for rendering (stripped from report)
        "_frame_det": frame_det,
        "_frame_delivery": frame_delivery,
    }


def _make_reprojector(projector, calib_ok, meta):
    if calib_ok:
        cal = projector.calibration
        rvec, _ = cv2.Rodrigues(cal.R)
        tvec = np.asarray(cal.t, dtype=float).reshape(3, 1)
        K = cal.K

        def rp(pt3):
            uv, _ = cv2.projectPoints(np.array([pt3], dtype=float), rvec, tvec, K, None)
            u, v = uv.reshape(2)
            return (float(u), float(v))
        return rp

    # PixelPlaneProjector fallback: invert its (x,z)->(u,v) mapping.
    mpp = getattr(projector, "meters_per_pixel", 0.01)
    cx0 = meta.width / 2.0
    Hh = meta.height

    def rp(pt3):
        x, _, z = pt3
        return (x / mpp + cx0, Hh - z / mpp)
    return rp


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w].copy()
    cv2.rectangle(sub, (0, 0), (w, h), (0, 0, 0), -1)
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, alpha, img[y:y + h, x:x + w], 1 - alpha, 0)


def _text(img, s, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _clip_pt(uv, W, H):
    return int(np.clip(uv[0], -1e4, 1e4)), int(np.clip(uv[1], -1e4, 1e4))


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render(data: dict, out_path: str) -> None:
    video = data["_video"]
    meta = video.metadata
    W, H = meta.width, meta.height
    fps = meta.fps or 30.0
    frame_det = data["_frame_det"]
    frame_delivery = data["_frame_delivery"]
    calib = data["calibration"]
    deliveries = data["deliveries"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for f in video.frames:
        img = f.image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        fi = f.index
        di = frame_delivery.get(fi)
        active = deliveries[di] if di is not None else None

        stage = "Segmentation (scanning — no delivery here)"
        if active is not None:
            stage = f"Delivery {active['index']}: Detection + Tracking"
            if active["trajectory_3d"]:
                stage += " + Trajectory"

        # --- trajectory (reprojected predicted path) ---
        if active is not None and active["_traj_uv"]:
            pts = [_clip_pt(uv, W, H) for uv in active["_traj_uv"]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, CYAN, 2, cv2.LINE_AA)

        # --- ball trail (tracked history up to now) ---
        if active is not None:
            trail = [d["center"] for d in active["track"] if d["frame_index"] <= fi]
            for a, b in zip(trail, trail[1:]):
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), ORANGE, 2, cv2.LINE_AA)

        # --- bounce point ---
        if active is not None and active["_bounce_uv"] is not None \
                and active["bounce_time"] is not None and f.timestamp >= active["bounce_time"]:
            bx, by = _clip_pt(active["_bounce_uv"], W, H)
            cv2.drawMarker(img, (bx, by), YELLOW, cv2.MARKER_TILTED_CROSS, 22, 2)
            _text(img, "BOUNCE", (bx + 8, by), YELLOW, 0.45)

        # --- release point ---
        if active is not None and active["release_uv"] is not None \
                and active["release_frame"] is not None and fi >= active["release_frame"]:
            rx, ry = int(active["release_uv"][0]), int(active["release_uv"][1])
            cv2.circle(img, (rx, ry), 7, CYAN, 2)
            _text(img, "RELEASE", (rx + 8, ry), CYAN, 0.45)

        # --- current ball bbox + center ---
        det = frame_det.get(fi)
        source_label = "-"
        if det is not None:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            cx, cy = int(det["center"][0]), int(det["center"][1])
            cv2.circle(img, (cx, cy), 4, RED, -1)
            source_label = {"yolo": "YOLO", "motion": "Motion",
                            "interpolated": "Interpolated"}.get(det["source"], det["source"])

        # --- top-left overlay ---
        _panel(img, 6, 6, min(300, W - 12), 96)
        _text(img, f"Video: {data['video']}", (12, 24), WHITE, 0.5)
        _text(img, f"Frame: {fi}/{len(video.frames) - 1}", (12, 44), WHITE, 0.45)
        _text(img, f"Time:  {f.timestamp:6.2f}s", (12, 62), WHITE, 0.45)
        src_col = {"YOLO": GREEN, "Motion": ORANGE, "Interpolated": GREY}.get(source_label, GREY)
        _text(img, f"Detection: {source_label}", (12, 80), src_col, 0.45)
        _text(img, stage, (12, 96), YELLOW, 0.4)

        # --- active delivery info panel (top-right) ---
        if active is not None:
            pw = min(232, W - 12)
            _panel(img, W - pw - 6, 6, pw, 92)
            bx = W - pw + 2
            _text(img, f"Delivery {active['index']}", (bx, 24), WHITE, 0.5)
            _text(img, f"{active['start_time']:.2f}-{active['end_time']:.2f}s", (bx, 42), WHITE, 0.42)
            sp = "n/a" if active["speed_kmph"] is None else f"{active['speed_kmph']:.1f} km/h"
            _text(img, f"Speed: {sp}", (bx, 60), WHITE, 0.42)
            _text(img, f"Conf: {active['confidence_score']:.2f}", (bx, 76), WHITE, 0.42)
            _text(img, f"Status: {active['status']}", (bx, 92), WHITE, 0.42)

        # --- fallback banner ---
        if not calib["ok"]:
            _panel(img, 6, H - 118, min(360, W - 12), 22)
            _text(img, "Metric calibration unavailable (fallback mode)", (12, H - 102), RED, 0.45)

        # --- bottom-right PASS/FAIL ---
        pw, ph = 150, 84
        _panel(img, W - pw - 6, H - ph - 6, pw, ph)
        bx = W - pw + 2
        def pf(ok): return ("PASS", GREEN) if ok else ("FAIL", RED)
        cv, cc = pf(calib["ok"])
        tv, tc = pf(data["summary"]["tracking_ok"])
        jv, jc = pf(data["summary"]["trajectory_ok"])
        _text(img, f"Calibration: {cv}", (bx, H - ph + 14), cc, 0.45)
        _text(img, f"Tracking:    {tv}", (bx, H - ph + 36), tc, 0.45)
        _text(img, f"Trajectory:  {jv}", (bx, H - ph + 58), jc, 0.45)

        vw.write(img)

    # --- 5-second summary screen ---
    for frame in _summary_frames(data, W, H, int(round(fps * 5))):
        vw.write(frame)

    vw.release()


def _summary_frames(data, W, H, n_frames):
    s = data["summary"]
    calib = data["calibration"]
    base = np.full((H, W, 3), 20, np.uint8)
    y = int(H * 0.16)
    _text(base, "PIPELINE SUMMARY", (int(W * 0.06), y), WHITE, 0.8, 2)
    lines = [
        (f"Video: {data['video']}", WHITE),
        (f"Deliveries detected: {s['n_deliveries']}", WHITE),
        (f"Detection rate: {s['detection_rate']:.2f}"
         + ("  (no deliveries)" if s["n_deliveries"] == 0 else ""), WHITE),
        ("Tracking: " + ("SUCCESS" if s["tracking_ok"] else "FAIL"),
         GREEN if s["tracking_ok"] else RED),
        ("Calibration: " + ("PASS" if s["calibration_ok"] else "FAIL (fallback mode)"),
         GREEN if s["calibration_ok"] else RED),
        ("Trajectory: " + ("PASS" if s["trajectory_ok"] else "FAIL"),
         GREEN if s["trajectory_ok"] else RED),
        (f"Estimated speed: {s['speed_kmph']} km/h" if s["speed_kmph"] is not None
         else "Estimated speed: n/a", WHITE),
        (f"Confidence: {s['confidence']}" if s["confidence"] is not None
         else "Confidence: n/a", WHITE),
        (f"Total processing time: {s['processing_time_s']}s", WHITE),
    ]
    dy = y + 40
    for text, col in lines:
        _text(base, text, (int(W * 0.06), dy), col, 0.6, 1)
        dy += 42
    if not calib["ok"] and calib["reason"]:
        _text(base, "Reason: " + calib["reason"][:52], (int(W * 0.06), dy), GREY, 0.42)
    return [base] * n_frames


# --------------------------------------------------------------------------- #
def _strip_for_report(data: dict) -> dict:
    out = {k: v for k, v in data.items() if not k.startswith("_")}
    for d in out["deliveries"]:
        d.pop("_traj_uv", None)
        d.pop("_bounce_uv", None)
    return out


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir", nargs="?", default=os.path.join("output", "demo"))
    ap.add_argument("--weights", default=None, help="trained YOLO weights to plug into detection")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--det-imgsz", type=int, default=1280)
    args = ap.parse_args(argv[1:])
    video_path = args.video
    out_dir = args.out_dir
    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    data = run_pipeline_capture(video_path, weights=args.weights, conf=args.conf, det_imgsz=args.det_imgsz)
    out_video = os.path.join(out_dir, f"{stem}_output.mp4")
    render(data, out_video)

    report = _strip_for_report(data)
    with open(os.path.join(out_dir, f"{stem}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    s = report["summary"]
    print(f"Wrote {out_video}")
    print(f"Wrote {os.path.join(out_dir, f'{stem}_report.json')}")
    print(f"Deliveries={s['n_deliveries']} calib={'PASS' if s['calibration_ok'] else 'FAIL'} "
          f"tracking={'PASS' if s['tracking_ok'] else 'FAIL'} traj={'PASS' if s['trajectory_ok'] else 'FAIL'} "
          f"time={s['processing_time_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
