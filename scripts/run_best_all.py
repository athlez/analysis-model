"""Run best.pt (yolo11m ball detector) + ByteTrack over ALL provided videos,
render an overlay per clip, score each by how continuously it tracks the ball,
and write a ranking so the best outputs are easy to find.

Overlay: bbox (green) + confidence + track ID + ball centre (red) + trail.
Single process, model loaded once (avoids per-clip reload / fork issues).
"""
import glob, json, os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO

SRC = "data/raw/own_recordings"
OUT = "output/detect_best"
WEIGHTS = "best.pt"
CONF = 0.2
IMGSZ = 1280


def process(model, path):
    stem = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if W == 0 or total == 0:
        return {"clip": stem, "error": "unreadable"}

    vw = cv2.VideoWriter(os.path.join(OUT, f"{stem}_det.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = defaultdict(lambda: deque(maxlen=25))
    track_frames = defaultdict(int)          # track_id -> #frames seen
    n_with_det = 0; n_frames = 0

    for fi, r in enumerate(model.track(source=path, tracker="bytetrack.yaml", persist=True,
                                       conf=CONF, imgsz=IMGSZ, stream=True, verbose=False)):
        img = r.orig_img.copy(); n_frames += 1
        boxes = r.boxes
        got = False
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0]); tid = int(b.id[0]) if b.id is not None else -1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
                cv2.circle(img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(img, f"ball#{tid} {conf:.2f}", (int(x1), max(int(y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
                if tid >= 0:
                    trails[tid].append((int(cx), int(cy))); track_frames[tid] += 1
                got = True
        if got:
            n_with_det += 1
        for pts in trails.values():
            for a, b in zip(pts, list(pts)[1:]):
                cv2.line(img, a, b, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{stem}  f{fi}/{max(total-1,0)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    longest = max(track_frames.values()) if track_frames else 0
    return {"clip": stem, "frames": n_frames,
            "detection_rate": round(n_with_det / max(n_frames, 1), 3),
            "n_tracks": len(track_frames),
            "longest_track": longest,
            "longest_track_frac": round(longest / max(n_frames, 1), 3)}


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(WEIGHTS)
    clips = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
    rows = []
    for i, c in enumerate(clips, 1):
        try:
            r = process(model, c)
        except Exception as e:
            r = {"clip": os.path.basename(c), "error": str(e)}
        rows.append(r)
        print(f"[{i}/{len(clips)}] {r}", flush=True)

    ok = [r for r in rows if "error" not in r]
    # 'best' = continuous tracking of a single ball: rank by longest_track_frac then detection_rate
    ok.sort(key=lambda r: (r["longest_track_frac"], r["detection_rate"]), reverse=True)
    with open(os.path.join(OUT, "_scores.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(OUT, "RANKING.md"), "w") as f:
        f.write("# best.pt detection ranking (best ball-tracking first)\n\n")
        f.write("Score = longest continuous track fraction (proxy for cleanly following one ball).\n\n")
        f.write("| Rank | Clip | det.rate | longest track | frames | #tracks |\n|---|---|---:|---:|---:|---:|\n")
        for i, r in enumerate(ok, 1):
            f.write(f"| {i} | {r['clip']} | {r['detection_rate']} | "
                    f"{r['longest_track']} ({int(r['longest_track_frac']*100)}%) | {r['frames']} | {r['n_tracks']} |\n")
        errs = [r for r in rows if "error" in r]
        if errs:
            f.write("\n## Skipped\n")
            for r in errs:
                f.write(f"- {r['clip']}: {r['error']}\n")
    print("\nTOP 8:")
    for r in ok[:8]:
        print(f"  {r['clip']}: det={r['detection_rate']} longest_track={r['longest_track']} ({int(r['longest_track_frac']*100)}%)")
    print(f"\nwrote {OUT}/RANKING.md + overlays")


if __name__ == "__main__":
    main()
