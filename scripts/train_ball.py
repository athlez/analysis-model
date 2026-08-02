"""Train the YOLO cricket-ball detector (Stage 4).

Small-object / fast-ball oriented settings:
  - high imgsz (default 960) for tiny balls
  - mosaic + wide scale jitter so the ball is seen at many sizes
  - HSV jitter for lighting robustness; albumentations blur (if installed) for
    motion-blur robustness
  - single class ('ball')

Reports mAP50, mAP50-95, precision, recall, and FP/FN from the confusion matrix.

Usage:
  python scripts/train_ball.py --model yolo11n.pt --imgsz 960 --epochs 80 --batch 8 --name it0_yolo11n
"""
import argparse, json, os
import numpy as np
from ultralytics import YOLO

DATA = "data/yolo_ball/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="it0")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=DATA, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, project="runs/ball", name=args.name, exist_ok=True,
        single_cls=True, patience=args.patience, cos_lr=True, cache=False, workers=8,
        # --- small-object / fast-ball augmentation ---
        mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1,
        fliplr=0.5, flipud=0.0, degrees=5.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )

    # Evaluate on the val split and pull FP/FN from the confusion matrix.
    m = model.val(data=DATA, imgsz=args.imgsz, split="val", device=args.device,
                  project="runs/ball", name=args.name + "_val", exist_ok=True)
    cm = m.confusion_matrix.matrix  # rows=pred (ball,bg), cols=true (ball,bg)
    # ultralytics: matrix[i,j] = predicted i, actual j; index 0=ball,1=background
    tp = float(cm[0, 0]); fp = float(cm[0, 1]); fn = float(cm[1, 0])
    out = {
        "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "TP": tp, "FP": fp, "FN": fn,
        "weights": f"runs/ball/{args.name}/weights/best.pt",
    }
    os.makedirs("runs/ball", exist_ok=True)
    with open(f"runs/ball/{args.name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n===== METRICS =====")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
