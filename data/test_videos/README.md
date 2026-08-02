# Test video dataset

Drop **real cricket bowling videos** here to validate the pipeline.

```
data/test_videos/
├── README.md
├── <clip1>.mp4
├── <clip2>.mp4
└── annotations/                 # OPTIONAL ground truth, one per clip
    └── <clip1>.json
```

## Recording guidance (matches the MVP calibration assumptions)

- Single **fixed** smartphone camera, **behind the bowler**, looking down the pitch.
- Camera **stationary** for the whole session.
- Whole **22-yard pitch** clearly visible, good lighting.
- One or more deliveries per clip is fine — the pipeline segments them.

## Optional ground-truth annotation schema

If you know the truth for a clip, add `annotations/<clip>.json`. Every field is
optional; metrics that need a field are simply skipped when it's absent.

```json
{
  "num_deliveries": 3,
  "deliveries": [
    {
      "start_time": 12.1,
      "end_time": 13.4,
      "speed_kmph": 132.0,
      "line": "off",
      "length": "good",
      "bounce": { "x": 0.2, "y": 6.1 }
    }
  ]
}
```

The validator runs even with **no annotations** — it then reports quality/proxy
metrics (detection rate, calibration confidence, fit residual, ...) instead of
accuracy against truth.

> No videos are committed to the repo. This folder is intentionally empty except
> for this README; `git` ignores the media files (see `.gitignore`).
