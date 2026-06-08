# Data

Dataset **không nằm trong Git** (chỉ `data/README.md` được track). Tải/đặt dữ liệu theo layout dưới đây.
Đường dẫn là default của script; có thể override bằng các cờ `--shot-root`, `--clipshots-root`, …

## SHOT / AutoShot (video)

`autoshotv2.prepare_metadata` (default `--shot-root ./data/ShotDataset`) kỳ vọng:

```text
data/ShotDataset/
├── train/GT/ground_truth.txt
├── train/videos/<name>.mp4
├── test/GT/ground_truth.txt
└── test/videos/<name>.mp4
```

GT SHOT test dùng cho eval:
`artifacts/experiments/published_sweeps/gt_scenes_dict_baseline_v2.pickle`.

## ClipShots

`autoshotv2.prepare_metadata` (default `--clipshots-root ./data/ClipShots`) kỳ vọng:

```text
data/ClipShots/
├── annotations/{train,test}.json
├── video_lists/{train,test}.txt
└── videos/{train,test}/<name>.mp4
```

GT ClipShots test:
`artifacts/experiments/published_sweeps/clipshots_test_gt_scenes.pickle`.

## BBC

```text
data/BBCDataset/
├── annotations/
└── bbc_01.mp4 … bbc_11.mp4
```

GT BBC: `artifacts/experiments/published_sweeps/bbc_shots_gt_scenes.pickle`.

## Hiện có trong bundle local (tham khảo)

- `data/BBCDataset/` — 11 video + annotations (đầy đủ).
- `data/ClipShots/` — annotations / video_lists / videos.
- `data/ShotData/` — **feature `.npy` precomputed** (không phải video mp4). Đây không phải layout
  `data/ShotDataset/...` mà `prepare_metadata` cần; nếu muốn chạy lại từ video SHOT thì đặt mp4 theo mục SHOT ở trên.

## Tái lập không cần video

Để chạy lại calibration/eval mà không cần dataset video, chỉ cần các logit cache + GT trong
`artifacts/experiments/published_sweeps/` và
`artifacts/experiments/ablation_full/`. Xem `docs/ARTIFACTS_MANIFEST.md`.
