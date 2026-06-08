# AutoShotV2 Merged Report Bundle

This folder is the merged report bundle of:

- `C:\Users\dotie\Documents\GitHub\archive1\autoshotv2`
- `C:\Users\dotie\Documents\GitHub\archive2\autoshotv2`
- `C:\Users\dotie\Documents\GitHub\archive2\inference`

## What Was Kept

- Training code from `archive2\autoshotv2`.
- Kaggle training notes from `archive1\autoshotv2`, because that copy contains the newer `--stop-after-minutes 660` and `--no-resume` guidance.
- Kaggle notebooks from `archive2\autoshotv2`, including the extra smoke-only notebook.
- Inference/evaluation code from `archive2\inference`.

## Included Run Artifacts

This report bundle includes the source code plus the run artifacts needed to
inspect or reproduce the completed training/evaluation outputs:

- final checkpoints: `ckpt_phase2_shot_clipshots_best.pth`, `phase2_shot_clipshots_resume.pt`
- epoch checkpoints: `phase2_shot_clipshots_checkpoints/`
- train/eval result files: `phase2_shot_clipshots_results.pkl`, `phase2_eval_gt_scenes_dict_baseline_v2_results.json`
- metadata and sample caches: `shot_clipshots_trainval.pickle`, `shot_clipshots_phase2_sample_cache.pkl`, `*.parts/`
- deploy result JSON under `reports/deploy_results/` (SHOT, ClipShots, BBC)
- cached logits under `eval_cache_*`

The original artifact copies are still available in the archive folders.

## Main Result Files

- `phase2_eval_gt_scenes_dict_baseline_v2_results.json`: SHOT 200-video evaluation summary.
- `phase2_shot_clipshots_results.pkl`: phase2 training result payload.
- `reports/deploy_results/inference_results.json`: SHOT inference/evaluation result.
- `reports/deploy_results/clipshot_test_inference_results.json`: ClipShots test result.
- `reports/deploy_results/bbc_shot_inference_results.json`: BBC result.

## Packaging (Wave 2)

The earlier flat layout duplicated `linear.py` / `supernet_*.py` between the repo root and
`inference/` because `inference/run_inference_eval.py` mixed flat imports (cwd=inference) with
`inference.`-prefixed imports (cwd=root). This duplication has been **removed**: all source now lives
in the `autoshotv2` package (`pip install -e .`), so there is a single copy of each module.

- `inference/run_inference_eval.py` -> `autoshotv2.eval`
- `supernet_*.py` / `linear.py` -> `autoshotv2.model.supernet` / `autoshotv2.model.linear`
- deploy result JSON -> `reports/deploy_results/`

## Main Commands

Prepare train/validation metadata:

```powershell
python -m autoshotv2.prepare_metadata --shot-root <DatasetShot> --clipshots-root <ClipShots> --out shot_clipshots_trainval.pickle
```

Train phase 2:

```powershell
python -m autoshotv2.train_phase2 --meta shot_clipshots_trainval.pickle --base-ckpt ckpt_0_200_0.pth
```

Run inference/evaluation (from repo root):

```powershell
python -m autoshotv2.eval --checkpoint ckpt_phase2_shot_clipshots_best.pth --videos-dir <videos>
```
