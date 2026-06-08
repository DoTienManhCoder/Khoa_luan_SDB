# Kaggle Training Notes

Use this when running `Shot train + ClipShots train` on Kaggle P100.

## Inputs

Attach these Kaggle datasets to the notebook:

- Source code dataset containing this repository.
- `DatasetShot`, with `train/` and `test/` under the dataset root.
- `clipshots`, with `annotations/`, `video_lists/`, and `videos/`.
- Optional previous run output dataset for resume.

Kaggle input paths are read-only. The notebook copies source and previous outputs into `/kaggle/working` automatically.

## Notebook Cells

The notebook is already configured for these dataset paths and can be run with `Run All`:

- `/kaggle/input/datasets/domanh704/dataset-clipshots/ClipShots`
- `/kaggle/input/datasets/domanh704/shotdivide/DatasetShot`

It auto-detects the source dataset by searching `/kaggle/input` for `pyproject.toml`. `ckpt_0_200_0.pth` can be in source or any attached checkpoint input dataset.

The notebook installs `ffmpeg-python` at startup. If Kaggle cannot install/import it, the notebook writes a small local `ffmpeg.py` shim in `/kaggle/working/autoshotv2` as fallback. Kaggle still needs the `ffmpeg` command-line binary, which is normally available.

For T4, the notebook keeps Kaggle's built-in PyTorch. For P100, the notebook force-installs PyTorch `2.4.1` with CUDA `11.8` from the official PyTorch wheel index because some recent Kaggle PyTorch builds only include `sm_70+`, while P100 is `sm_60`. Enable Kaggle Internet for this install step. If Internet is disabled or P100 still fails, switch Accelerator to T4.

Warnings are suppressed with `PYTHONWARNINGS=ignore`. The full run uses `--stop-after-minutes 660` and saves partial cache every 25 videos so Kaggle can stop cleanly and resume in the next session instead of being killed by `SIGKILL`. The command also uses `--no-resume` so stale head-training state does not block a fresh train after the sample cache finishes.

```python
from pathlib import Path
import os
import shutil

SHOT_ROOT = Path("/kaggle/input/datasets/domanh704/shotdivide/DatasetShot")
CLIPSHOTS_ROOT = Path("/kaggle/input/datasets/domanh704/dataset-clipshots/ClipShots")
RUN_SMOKE_TEST = True
RUN_FULL_TRAIN = True

source_candidates = [p.parent for p in Path("/kaggle/input").rglob("pyproject.toml")]
if not source_candidates:
    raise FileNotFoundError("Attach source dataset containing pyproject.toml")
SOURCE_INPUT = sorted(source_candidates, key=lambda p: len(str(p)))[0]

previous_candidates = [
    p.parent for p in Path("/kaggle/input").rglob("phase2_shot_clipshots_resume.pt")
]
PREVIOUS_OUTPUT = previous_candidates[0] if previous_candidates else None

WORK_DIR = Path("/kaggle/working/autoshotv2")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
shutil.copytree(SOURCE_INPUT, WORK_DIR)
if not (WORK_DIR / "ckpt_0_200_0.pth").exists():
    ckpt_candidates = list(Path("/kaggle/input").rglob("ckpt_0_200_0.pth"))
    if not ckpt_candidates:
        raise FileNotFoundError("Attach ckpt_0_200_0.pth in source or a checkpoint input dataset")
    shutil.copy2(ckpt_candidates[0], WORK_DIR / "ckpt_0_200_0.pth")

if PREVIOUS_OUTPUT and PREVIOUS_OUTPUT.exists():
    for name in [
        "shot_clipshots_trainval.pickle",
        "shot_clipshots_phase2_sample_cache.pkl",
        "shot_clipshots_phase2_sample_cache.pkl.partial.pkl",
        "shot_clipshots_phase2_sample_cache.pkl.parts",
        "phase2_shot_clipshots_resume.pt",
        "phase2_shot_clipshots_checkpoints",
        "eval_cache_shot_clipshots",
    ]:
        src = PREVIOUS_OUTPUT / name
        dst = WORK_DIR / name
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

os.chdir(WORK_DIR)
print(WORK_DIR)
```

Source now lives in `src/autoshotv2/` (not flat scripts). The setup cell sets
`PYTHONPATH=src` so `python -m autoshotv2.*` works **without installing** (no Internet needed):

```python
os.environ["PYTHONPATH"] = str(WORK_DIR / "src")
```

Alternatively, if Internet is enabled you can install the package instead:

```python
!pip install -e . --no-deps
```

```python
!python -m autoshotv2.prepare_metadata \
  --shot-root "{SHOT_ROOT}" \
  --clipshots-root "{CLIPSHOTS_ROOT}" \
  --out /kaggle/working/autoshotv2/shot_clipshots_trainval.pickle \
  --val-ratio 0.10 \
  --max-val-videos 200
```

Run full training. `--stop-after-minutes 660` exits safely before Kaggle's 12-hour limit and skips stale head-training resume state.

```python
!python -m autoshotv2.train_phase2 \
  --meta /kaggle/working/autoshotv2/shot_clipshots_trainval.pickle \
  --base-ckpt /kaggle/working/autoshotv2/ckpt_0_200_0.pth \
  --sample-cache /kaggle/working/autoshotv2/shot_clipshots_phase2_sample_cache.pkl \
  --resume-state /kaggle/working/autoshotv2/phase2_shot_clipshots_resume.pt \
  --checkpoint-dir /kaggle/working/autoshotv2/phase2_shot_clipshots_checkpoints \
  --out-ckpt /kaggle/working/autoshotv2/ckpt_phase2_shot_clipshots_best.pth \
  --results /kaggle/working/autoshotv2/phase2_shot_clipshots_results.pkl \
  --eval-cache-dir /kaggle/working/autoshotv2/eval_cache_shot_clipshots \
  --epochs 20 \
  --batch-size 512 \
  --max-samples-per-video 160 \
  --save-every-videos 25 \
  --save-every-epochs 1 \
  --log-every-batches 100 \
  --stop-after-minutes 660 \
  --no-resume
```

For a quick Kaggle smoke test:

```python
!python -m autoshotv2.train_phase2 \
  --meta /kaggle/working/autoshotv2/shot_clipshots_trainval.pickle \
  --base-ckpt /kaggle/working/autoshotv2/ckpt_0_200_0.pth \
  --epochs 1 \
  --max-train-videos 2 \
  --max-total-samples 100 \
  --max-samples-per-video 50 \
  --max-val-videos 1 \
  --max-test-videos 1 \
  --sample-cache /kaggle/working/autoshotv2/smoke_sample_cache.pkl \
  --out-ckpt /kaggle/working/autoshotv2/smoke_ckpt.pth \
  --results /kaggle/working/autoshotv2/smoke_results.pkl \
  --eval-cache-dir /kaggle/working/autoshotv2/smoke_eval_cache \
  --rebuild-sample-cache \
  --no-eval-cache \
  --no-resume
```

## Resume Workflow

At the end of each Kaggle run, save notebook output as a Kaggle dataset. In the next run, attach that output dataset; the notebook auto-detects previous outputs if it finds resume files under `/kaggle/input`. The training script will resume from:

- `shot_clipshots_phase2_sample_cache.pkl.parts/` while feature cache is incomplete.
- `shot_clipshots_phase2_sample_cache.pkl` once feature cache is complete.
- `phase2_shot_clipshots_resume.pt` for epoch-level head training.
- `phase2_shot_clipshots_checkpoints/epoch_*.pt` as extra epoch snapshots.

Final important outputs:

- `ckpt_phase2_shot_clipshots_best.pth`
- `phase2_shot_clipshots_results.pkl`
- `eval_cache_shot_clipshots/shot_test_logits.pkl`
