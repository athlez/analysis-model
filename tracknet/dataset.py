"""PyTorch Dataset: builds 3-consecutive-frame inputs + Gaussian heatmap targets
from the packaged manifest. Runs on Colab (or locally)."""
import csv, os
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def gaussian_heatmap(h, w, cx, cy, sigma=3.0):
    """1-amplitude Gaussian blob centred at (cx,cy) px; zeros if cx<0 (no ball)."""
    hm = np.zeros((h, w), np.float32)
    if cx < 0 or cy < 0:
        return hm
    r = int(3 * sigma)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return hm
    xs = np.arange(x0, x1)[None, :]; ys = np.arange(y0, y1)[:, None]
    hm[y0:y1, x0:x1] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm


class TrackNetDataset(Dataset):
    def __init__(self, root, n_frames=3, in_w=288, in_h=512, sigma=3.0):
        self.root, self.n, self.W, self.H, self.sigma = root, n_frames, in_w, in_h, sigma
        per = defaultdict(dict)   # clip -> {idx: (x,y,vis)}
        with open(os.path.join(root, "manifest.csv")) as f:
            for r in csv.DictReader(f):
                per[r["clip"]][int(r["idx"])] = (float(r["x_norm"]), float(r["y_norm"]), int(r["visible"]))
        # a sample = n consecutive indices that all have frames on disk
        self.samples = []
        for clip, d in per.items():
            idxs = sorted(d)
            present = set(idxs)
            for i in idxs:
                win = list(range(i - n_frames + 1, i + 1))
                if all(w in present for w in win):
                    self.samples.append((clip, win))
        self.per = per

    def __len__(self):
        return len(self.samples)

    def _load(self, clip, idx):
        img = cv2.imread(os.path.join(self.root, "frames", clip, f"{idx:05d}.jpg"))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, k):
        clip, win = self.samples[k]
        imgs, hms = [], []
        for idx in win:
            imgs.append(self._load(clip, idx))
            x, y, vis = self.per[clip][idx]
            cx, cy = (x * self.W, y * self.H) if vis else (-1, -1)
            hms.append(gaussian_heatmap(self.H, self.W, cx, cy, self.sigma))
        x = np.concatenate([im.transpose(2, 0, 1) for im in imgs], 0)   # (3n,H,W)
        y = np.stack(hms, 0)                                            # (n,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)
