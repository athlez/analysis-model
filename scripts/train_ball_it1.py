"""Iteration 1: retrain with MOTION-BLUR augmentation (targets the fast, blurred
ball that iteration 0 missed). Warm-starts from the it0 weights and fine-tunes.

The ball in our footage is a faint motion-blurred streak; iteration 0 (trained on
mostly-sharp balls) never fired on it. We inject albumentations MotionBlur/Blur
(plus brightness & JPEG-compression jitter for phone footage) into ultralytics'
augmentation so the model learns blurred-ball appearance.

Data unchanged (no rebalance this iteration, per instruction). Downstream frozen.
"""
import argparse, json, os
import albumentations as A
import ultralytics.data.augment as aug

DATA = "data/yolo_ball/data.yaml"


def patch_albumentations():
    """Inject blur-heavy transforms via ultralytics' own (transforms=...) path so
    Compose/bbox_params/spatial handling stays correct across versions."""
    _orig = aug.Albumentations.__init__

    def _init(self, p=1.0, transforms=None):
        if transforms is None:
            transforms = [
                A.MotionBlur(blur_limit=(3, 15), p=0.35),   # fast-ball streak
                A.Blur(blur_limit=5, p=0.10),
                A.MedianBlur(blur_limit=5, p=0.10),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                A.ImageCompression(quality_range=(50, 90), p=0.15),  # phone-video artifacts
            ]
            print(">> Patched Albumentations: MotionBlur(0.35)+Blur+MedianBlur+bright+jpeg")
        _orig(self, p=p, transforms=transforms)

    aug.Albumentations.__init__ = _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/detect/runs/ball/it0_yolo11n/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="it1_blur")
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 avoids Windows paging-file DLL crashes with spawned workers")
    ap.add_argument("--cache", default="False",
                    help="'ram' caches decoded images in RAM -> fast epochs with workers=0")
    ap.add_argument("--mosaic", type=float, default=1.0,
                    help="mosaic aug; set 0 for tiny objects that mosaic destroys")
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--optimizer", default="auto",
                    help="explicit optimizer (e.g. AdamW/SGD) so lr0 is honoured")
    args = ap.parse_args()
    data_cfg = args.data

    patch_albumentations()
    from ultralytics import YOLO
    model = YOLO(args.init)  # warm-start from iteration 0
    model.train(
        data=data_cfg, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, project="runs/ball", name=args.name, exist_ok=True,
        single_cls=True, patience=10, cos_lr=True, optimizer=args.optimizer,
        cache=(str(args.cache).lower() if str(args.cache).lower() in ("ram", "disk") else False), workers=args.workers,
        mosaic=args.mosaic, close_mosaic=(8 if args.mosaic > 0 else 0),
        scale=0.5, translate=0.1,
        fliplr=0.5, flipud=0.0, degrees=5.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        lr0=args.lr0,
    )
    m = model.val(data=data_cfg, imgsz=args.imgsz, split="val", device=args.device,
                  project="runs/ball", name=args.name + "_val", exist_ok=True)
    cm = m.confusion_matrix.matrix
    out = {
        "iteration": 1, "init": args.init, "imgsz": args.imgsz, "epochs": args.epochs,
        "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
        "TP": float(cm[0, 0]), "FP": float(cm[0, 1]), "FN": float(cm[1, 0]),
    }
    with open(f"runs/ball/{args.name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n===== IT1 METRICS =====")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
