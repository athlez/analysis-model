"""Run the full pipeline demo (trained detector wired in) on every own_recordings
clip and save an overlay video per clip. Robust: continues on per-clip errors.

Output: output/demo_all/<clip>_output.mp4 + <clip>_report.json, plus a summary.
"""
import glob, json, os, time, traceback
from validation.demo import run_pipeline_capture, render, _strip_for_report

SRC = "data/raw/own_recordings"
OUT = "output/demo_all"
WEIGHTS = "models/ball_detector.pt"

os.makedirs(OUT, exist_ok=True)
clips = sorted(glob.glob(os.path.join(SRC, "*.mp4")),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
summary = []
for i, video in enumerate(clips, 1):
    stem = os.path.splitext(os.path.basename(video))[0]
    t0 = time.time()
    print(f"[{i}/{len(clips)}] {stem} ...", flush=True)
    try:
        data = run_pipeline_capture(video, weights=WEIGHTS, conf=0.15, det_imgsz=1280)
        render(data, os.path.join(OUT, f"{stem}_output.mp4"))
        rep = _strip_for_report(data)
        with open(os.path.join(OUT, f"{stem}_report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        s = rep["summary"]
        # detection-source breakdown across deliveries
        srcs = {}
        for d in rep["deliveries"]:
            for tk in d["track"]:
                srcs[tk["source"]] = srcs.get(tk["source"], 0) + 1
        row = {"clip": stem, "deliveries": s["n_deliveries"],
               "calib": s["calibration_ok"], "tracking": s["tracking_ok"],
               "traj": s["trajectory_ok"], "speed_kmph": s["speed_kmph"],
               "det_sources": srcs, "secs": round(time.time() - t0, 1)}
        print(f"    OK {row}", flush=True)
    except Exception as e:
        row = {"clip": stem, "error": str(e)}
        print(f"    ERROR: {e}\n{traceback.format_exc()}", flush=True)
    summary.append(row)

with open(os.path.join(OUT, "_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n===== BATCH SUMMARY =====")
for r in summary:
    print(r)
print(f"\nSaved {sum('error' not in r for r in summary)}/{len(clips)} overlay videos to {OUT}/")
