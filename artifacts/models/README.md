# Model Checkpoints

Place the deployment checkpoint at:

```text
artifacts/models/deploy.pth
```

Place the original backbone checkpoint at `artifacts/models/ckpt_0_200_0.pth`.
Training checkpoints and resume state default to `artifacts/models/training/`.

Checkpoint binaries are intentionally excluded from Git. Set
`AUTOSHOT_MODEL_PATH` when using another filename or mount location.

With `backend=auto`, the web API uses the OpenCV baseline when the checkpoint
is missing or invalid. With `backend=phase2`, the job fails with a model error.
