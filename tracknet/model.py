"""TrackNetV2 — heatmap ball tracker for tiny, fast, blurred balls.

Input : N consecutive RGB frames stacked on the channel axis (3*N channels).
Output: N sigmoid heatmaps (one per input frame); the ball is the bright blob.
Because it sees several frames at once, it learns the ball's *motion streak* —
which is exactly the cue a per-frame object detector (YOLO) throws away, and why
this works on a 4-14px ball where YOLO plateaus.

Architecture: VGG16-style encoder + symmetric upsampling decoder (the original
TrackNetV2 design), ~11M params, trains fine on a single cloud GPU.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _cbr(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout))


class TrackNetV2(nn.Module):
    def __init__(self, n_frames: int = 3):
        super().__init__()
        self.n_frames = n_frames
        cin = 3 * n_frames
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.e1 = nn.Sequential(_cbr(cin, 64), _cbr(64, 64))
        self.e2 = nn.Sequential(_cbr(64, 128), _cbr(128, 128))
        self.e3 = nn.Sequential(_cbr(128, 256), _cbr(256, 256), _cbr(256, 256))
        self.e4 = nn.Sequential(_cbr(256, 512), _cbr(512, 512), _cbr(512, 512))

        self.d3 = nn.Sequential(_cbr(512 + 256, 256), _cbr(256, 256), _cbr(256, 256))
        self.d2 = nn.Sequential(_cbr(256 + 128, 128), _cbr(128, 128))
        self.d1 = nn.Sequential(_cbr(128 + 64, 64), _cbr(64, 64))
        self.head = nn.Conv2d(64, n_frames, 1)

    def forward(self, x):
        e1 = self.e1(x)                 # H
        e2 = self.e2(self.pool(e1))     # H/2
        e3 = self.e3(self.pool(e2))     # H/4
        e4 = self.e4(self.pool(e3))     # H/8 (bottleneck, no further pool)
        d3 = self.d3(torch.cat([self.up(e4), e3], 1))   # H/4
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))   # H/2
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))   # H
        return torch.sigmoid(self.head(d1))             # (B, n_frames, H, W)


if __name__ == "__main__":
    m = TrackNetV2(3)
    x = torch.randn(2, 9, 512, 288)
    y = m(x)
    print("params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 1))
    print("in", tuple(x.shape), "-> out", tuple(y.shape))
