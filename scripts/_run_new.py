import json, glob, os
from ultralytics import YOLO
import run_best_all as R   # reuse process(), OUT, WEIGHTS
import sys; sys.path.insert(0, "scripts")
from run_best_all import process, OUT
os.makedirs(OUT, exist_ok=True)
model = YOLO("best.pt")
clips = [f"data/raw/own_recordings/bowling{i}.mp4" for i in range(16,24)]
rows=[]
for i,c in enumerate(clips,1):
    try: r=process(model,c)
    except Exception as e: r={"clip":os.path.basename(c),"error":str(e)}
    rows.append(r); print(f"[{i}/{len(clips)}] {r}", flush=True)
json.dump(rows, open(os.path.join(OUT,"_scores_new.json"),"w"), indent=2)
