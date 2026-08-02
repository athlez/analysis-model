"""Train TrackNetV2 on the packaged dataset. Designed for a single cloud GPU
(Colab). Held-out clips are excluded so you get an honest generalization number.

    python tracknet/train.py --data tracknet/dataset --epochs 40 \
        --holdout bowling3,Test13,bowling_5,bowling_18,bowling_47,bowling_99
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))
from tracknet.model import TrackNetV2
from tracknet.dataset import TrackNetDataset


def wbce(pred, target, pos_w=None):
    """Weighted BCE — the heatmap is mostly background, so up-weight positives."""
    eps = 1e-6
    if pos_w is None:
        pos_w = (1 - target.mean()) / (target.mean() + eps)
        pos_w = float(torch.clamp(pos_w, 1, 500))
    loss = -(pos_w * target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
    return loss.mean()


def peak_recall(model, loader, dev, tol_px=8):
    """Fraction of visible-ball frames where the heatmap peak is within tol of GT."""
    model.eval(); hit = tot = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev)).cpu().numpy()
            y = y.numpy()
            for b in range(p.shape[0]):
                for c in range(p.shape[1]):
                    if y[b, c].max() < 0.5:
                        continue
                    tot += 1
                    gy, gx = np.unravel_index(y[b, c].argmax(), y[b, c].shape)
                    py, px = np.unravel_index(p[b, c].argmax(), p[b, c].shape)
                    if p[b, c].max() > 0.3 and np.hypot(px - gx, py - gy) <= tol_px:
                        hit += 1
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tracknet/dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--out", default="tracknet/tracknet_best.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    ds = TrackNetDataset(args.data)
    hold = set(s for s in args.holdout.split(",") if s)
    tr_idx = [i for i, (c, _) in enumerate(ds.samples) if c not in hold]
    va_idx = [i for i, (c, _) in enumerate(ds.samples) if c in hold]
    print(f"samples: {len(ds)}  train {len(tr_idx)}  held-out {len(va_idx)} ({sorted(hold)})")
    tl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, num_workers=2) if va_idx else None

    model = TrackNetV2(3).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); p = model(x); loss = wbce(p, y); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        rec = peak_recall(model, vl, dev) if vl else float("nan")
        print(f"epoch {ep:3d}  loss {tot/len(tl):.4f}  held-out peak-recall {rec:.3f}", flush=True)
        if vl and rec > best:
            best = rec; torch.save(model.state_dict(), args.out)
            print(f"   saved {args.out} (held-out recall {best:.3f})", flush=True)
    if not vl:
        torch.save(model.state_dict(), args.out)
    print("done. best held-out peak-recall:", round(best, 3))


if __name__ == "__main__":
    main()
