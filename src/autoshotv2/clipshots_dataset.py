"""ClipShots training dataset for full-model fine-tuning (EMA study).

Reads `<clipshots_root>/annotations/train.json`, converts the `transitions`
field to scene boundaries, and yields 100-frame windows (the centered 50 frames
are supervised, mirroring the inference pipeline). Used by `autoshotv2.train_ema`.
"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from autoshotv2.utils import ffmpeg, get_frames, scenes2zero_one_representation

# Errors get_frames can raise on a corrupt/missing video: ffmpeg decode failures,
# subprocess fallback errors, and shape/reshape errors on truncated streams.
_VIDEO_LOAD_ERRORS = (RuntimeError, OSError, ValueError, subprocess.CalledProcessError)
if ffmpeg is not None:
    _VIDEO_LOAD_ERRORS = (*_VIDEO_LOAD_ERRORS, ffmpeg.Error)

WINDOW = 100
CENTER = 50  # we predict / supervise the middle 50 frames


def transitions_to_scenes(transitions, n_frames: int) -> np.ndarray:
    """Convert ClipShots-style `transitions=[[s,e], ...]` to scene boundaries."""
    if not transitions:
        return np.array([[0, max(0, n_frames - 1)]], dtype=np.int32)
    t = np.array(transitions, dtype=np.int32)
    scenes = [[0, int(t[0, 0])]]
    for j in range(1, len(t)):
        scenes.append([int(t[j - 1, 1]), int(t[j, 0])])
    end = n_frames - 1 if n_frames > 0 else int(t[-1, 1]) + 1
    scenes.append([int(t[-1, 1]), end])
    arr = np.array(scenes, dtype=np.int32)
    if n_frames > 0:
        arr = np.clip(arr, 0, n_frames - 1)
    arr = arr[arr[:, 0] <= arr[:, 1]]
    if len(arr) == 0:
        arr = np.array([[0, 0]], dtype=np.int32)
    return arr


class ClipShotsTrainDataset(Dataset):
    """One sample = one random 100-frame window from one ClipShots train video."""

    def __init__(self, clipshots_root, annotation_name: str = "train.json",
                 video_subdir: str = "videos/train", limit: int | None = None):
        self.root = Path(clipshots_root)
        ann_path = self.root / "annotations" / annotation_name
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation not found: {ann_path}")
        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.video_dir = self.root / video_subdir
        if not self.video_dir.is_dir():
            raise FileNotFoundError(f"Video folder not found: {self.video_dir}")

        self.entries = []
        skipped = 0
        for fname, entry in raw.items():
            vpath = self.video_dir / fname
            if not vpath.is_file():
                skipped += 1
                continue
            transitions = entry.get("transitions", [])
            n_frames = int(entry.get("frame_num", 0))
            self.entries.append((fname, str(vpath), transitions, n_frames))
            if limit is not None and len(self.entries) >= limit:
                break
        print(f"[ClipShotsTrainDataset] usable={len(self.entries)} skipped={skipped}", flush=True)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_one(self, idx: int):
        fname, vpath, transitions, n_frames_meta = self.entries[idx]
        frames = get_frames(vpath)  # (N, 27, 48, 3) uint8
        n_frames = len(frames)
        if n_frames < 26:
            # too short for even one center-supervised window
            raise RuntimeError(f"video too short ({n_frames} frames): {fname}")

        scenes = transitions_to_scenes(transitions, n_frames)
        one_hot, many_hot = scenes2zero_one_representation(scenes, n_frames)

        # Choose a random WINDOW=100 frame slice; pad if video is shorter than WINDOW.
        if n_frames < WINDOW:
            pad = WINDOW - n_frames
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
            one_hot = np.concatenate([one_hot, np.zeros(pad, dtype=one_hot.dtype)])
            many_hot = np.concatenate([many_hot, np.zeros(pad, dtype=many_hot.dtype)])
            start = 0
        else:
            start = random.randint(0, n_frames - WINDOW)

        win_frames = frames[start:start + WINDOW]                      # (100, 27, 48, 3)
        win_oh = one_hot[start + 25:start + 25 + CENTER]               # (50,)
        win_mh = many_hot[start + 25:start + 25 + CENTER]              # (50,)

        # Convert to (3, 100, 27, 48) float tensor; .copy() avoids the non-writable
        # warning from torch.from_numpy on ffmpeg-backed numpy buffers.
        frames_t = torch.from_numpy(
            win_frames.transpose((3, 0, 1, 2)).copy()
        ).float()
        oh_t = torch.from_numpy(win_oh.astype(np.float32))
        mh_t = torch.from_numpy(win_mh.astype(np.float32))
        return frames_t, oh_t, mh_t

    def __getitem__(self, idx: int):
        # Retry with another random index if ffmpeg / numpy fails on a corrupt
        # or missing video file. ClipShots train has a few non-decodable mp4s.
        attempts = 0
        last_err = None
        cur = idx
        while attempts < 8:
            try:
                return self._load_one(cur)
            except _VIDEO_LOAD_ERRORS as e:
                last_err = e
                fname = self.entries[cur][0]
                print(f"[ClipShotsTrainDataset] skip {fname}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                attempts += 1
                cur = random.randint(0, len(self.entries) - 1)
        raise RuntimeError(f"Failed to load any sample after 8 retries; last error: {last_err}")
