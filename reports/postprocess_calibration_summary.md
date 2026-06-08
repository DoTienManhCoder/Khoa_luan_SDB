# Post-process Calibration (no retraining)

- Method: 5-fold cross-validation, seed 42. Post-process knobs
  (temperature, Gaussian sigma, decision threshold) are chosen on calibration folds and
  measured on the held-out fold (micro-averaged tp/fp/fn). This avoids tuning on the test set.
- `cv_f1` is the honest, cross-validated deploy F1. `ceiling_f1` tunes on the full test set
  (optimistic, shown only to expose the optimism gap).
- The deploy-checkpoint logits behind the README table are not in the bundle; these models are
  A0 baseline + the Phase2 controlled runs (A1, B5) whose caches ship with the bundle.

## Cross-validated deploy F1 (honest)

| Model | shot | clipshots | bbc |
|---|---|---|---|
| `A0_autoshot_baseline` | 0.8443 | 0.7881 | 0.9538 |
| `A1_phase2_control` | 0.8540 | 0.7575 | 0.9631 |
| `B5_phase2_full` | 0.8539 | 0.7555 | 0.9584 |

## Optimism gap (ceiling tuned on test vs CV)

| Model | shot | clipshots | bbc |
|---|---|---|---|
| `A0_autoshot_baseline` | 0.8475 / 0.8443 | 0.7896 / 0.7881 | 0.9561 / 0.9538 |
| `A1_phase2_control` | 0.8540 / 0.8540 | 0.7599 / 0.7575 | 0.9631 / 0.9631 |
| `B5_phase2_full` | 0.8550 / 0.8539 | 0.7607 / 0.7555 | 0.9603 / 0.9584 |

## Headline: does calibrated Phase2 beat A0 on ClipShots? (cross-validated)

- A0 baseline CV F1: 0.7881
- B5 Phase2 CV F1: 0.7555
- Delta (B5 - A0): -0.0326 -> calibrated Phase2 **does NOT beat** A0 on ClipShots.
