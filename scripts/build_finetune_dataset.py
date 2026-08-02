"""Build a fine-tuning dataset config that combines:
  - Roboflow appearance data (data/yolo_ball train)  [keeps general 'ball' ability]
  - in-domain labeled frames, OVERSAMPLED             [adapts to our ball]
with one labeled video HELD OUT as validation (leave-one-video-out) for an
honest "detects the ball in an unseen clip" signal.

Writes data/ball_dataset/finetune.yaml. Uses ultralytics' list-of-dirs `train`
so the large Roboflow set isn't copied; only the in-domain oversample is copied.
"""
import argparse, glob, os, shutil, yaml

ROOT = "data/ball_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="bowling3", help="video used for val (not trained on)")
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--roboflow", default="data/yolo_ball/images/train")
    ap.add_argument("--roboflow_subset", type=int, default=0,
                    help="if >0, use a random N-image Roboflow subset (fast fine-tune)")
    args = ap.parse_args()

    # Optional: build a small random Roboflow subset to keep epochs fast while
    # still preventing catastrophic forgetting of the general 'ball' appearance.
    if args.roboflow_subset > 0:
        import random
        sub = os.path.join(ROOT, "_ft_roboflow_sub")
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(os.path.join(sub, "images")); os.makedirs(os.path.join(sub, "labels"))
        imgs = glob.glob(os.path.join(args.roboflow, "*"))
        random.seed(0); random.shuffle(imgs)
        for img in imgs[:args.roboflow_subset]:
            lbl = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            if not os.path.isfile(lbl):
                continue
            shutil.copy(img, os.path.join(sub, "images", os.path.basename(img)))
            shutil.copy(lbl, os.path.join(sub, "labels", os.path.basename(lbl)))
        args.roboflow = os.path.join(sub, "images")
        print(f"roboflow subset: {len(os.listdir(os.path.join(sub,'images')))} images")

    ov = os.path.join(ROOT, "_ft_indomain")
    shutil.rmtree(ov, ignore_errors=True)
    os.makedirs(os.path.join(ov, "images")); os.makedirs(os.path.join(ov, "labels"))

    train_vids, n_copies = [], 0
    for vid in sorted(os.listdir(os.path.join(ROOT, "sources"))):
        if vid == args.holdout:
            continue
        train_vids.append(vid)
        for img in glob.glob(os.path.join(ROOT, "sources", vid, "images", "*.jpg")):
            lbl = img.replace("images", "labels").replace(".jpg", ".txt")
            stem = os.path.splitext(os.path.basename(img))[0]
            for k in range(args.oversample):
                shutil.copy(img, os.path.join(ov, "images", f"{stem}_d{k}.jpg"))
                shutil.copy(lbl, os.path.join(ov, "labels", f"{stem}_d{k}.txt"))
                n_copies += 1

    ov_posix = ov.replace("\\", "/")
    data = {
        "path": os.path.abspath(".").replace("\\", "/"),
        "train": [args.roboflow.replace("\\", "/"), f"{ov_posix}/images"],
        "val": [f"{ROOT}/sources/{args.holdout}/images".replace("\\", "/")],
        "nc": 1, "names": ["ball"],
    }
    with open(os.path.join(ROOT, "finetune.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    n_val = len(glob.glob(os.path.join(ROOT, "sources", args.holdout, "images", "*.jpg")))
    n_rob = len(glob.glob(os.path.join(args.roboflow, "*")))
    print(f"train sources: Roboflow({n_rob}) + in-domain[{train_vids}] x{args.oversample} = {n_copies} copies")
    print(f"val (held-out): {args.holdout} = {n_val} frames")
    print(f"wrote {ROOT}/finetune.yaml")
    print(open(os.path.join(ROOT, "finetune.yaml")).read())


if __name__ == "__main__":
    main()
