# AutoShotV2 Inference / Evaluation

Run from the repo root (after `pip install -e .`):

```powershell
python -m autoshotv2.eval
```

The default command evaluates the cached SHOT logits against the included ground truth file.

To run inference again from videos, put videos anywhere and pass the folder:

```powershell
python -m autoshotv2.eval --videos-dir .\videos
```

For evaluation, video file stems must match the ground-truth keys, for example
`31602670982.mp4` maps to key `31602670982`.

Relevant source (in the `autoshotv2` package):

- `autoshotv2.model.supernet`: model definition (`TransNetV2Supernet`).
- `autoshotv2.model.linear`: model helper layer (`Linear_`).
- `autoshotv2.eval`: main inference/evaluation command (`python -m autoshotv2.eval`).

Artifacts needed for cached evaluation (not in Git — see `docs/ARTIFACTS_MANIFEST.md`):

- `ckpt_phase2_shot_clipshots_best.pth`: final model checkpoint.
- `gt_scenes_dict_baseline_v2.pickle` (hoặc trong
  `artifacts/experiments/published_sweeps/`): ground truth.
- `eval_cache_shot_clipshots/shot_test_logits.pkl`: cached logits for the 200-video test set.

Runtime dependencies: Python, PyTorch, NumPy, SciPy, einops, and the `ffmpeg`
command-line tool when using `--videos-dir`.
