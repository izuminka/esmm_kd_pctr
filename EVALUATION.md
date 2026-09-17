# ESMM evaluation protocol

`configs/esmm.yaml`, `esmm_half.yaml`, and `esmm_quarter.yaml` use the same
training and evaluation settings. Only `data.train_cohort` differs. The quarter
cohort is nested in half; validation and test always use their full populations.

| Model | Conversion score in training | Validation selection | Logged pCTR files needed |
| --- | --- | --- | --- |
| ESMM full / half / quarter | `q*v` | Native CTCVR PR-AUC (`q*v`) | Test |
| Logged-pCTR KD / teacher variants | `q*v`; `p` supplies the CTR soft target | Logged CTCVR PR-AUC (`p*v`) | Train, validation, test |
| Logged-pCTR product-only | `p*v` | Logged CTCVR PR-AUC (`p*v`) | Train, validation, test |

Logged-score models (KD, teacher, and product-only) validate and select on `p*v`.
For KD this is an explicit selection rule; its training conversion loss still
uses `q*v`. Native ESMM models neither load nor use validation logged pCTR. The
pipeline rejects `logged_ctcvr_prauc` selection for native ESMM. Train reports do not
include `p*v`; test reports include it when `evaluation.is_logged_ctcvr` is true.
Turning that flag off does not remove logged-model validation: KD and product-only
still require validation logged scores for checkpoint selection.

For the full/half/quarter controls:

- Seed Python, NumPy, and Torch with `training.seed` before constructing the
  model. All three use seed 42 and the same vocabulary and architecture.
  GPU kernels are not forced to be deterministic.
- Validate after every epoch; maximize complete-validation native CTCVR average
  precision (`ctcvr_prauc`).
- Train for at most five epochs. Stop after two consecutive epochs without a
  strictly higher score (`min_delta: 0.0`).
- Restore the best checkpoint before reporting any final split metrics.
- Record AUC, PR-AUC, NE, and CE for CTR, native CTCVR, and clicked pCVR each
  epoch in `history.csv`. KD additionally records logged CTCVR; product-only
  records logged CTCVR and clicked pCVR.
- Evaluate test after checkpoint selection; test metrics never select models.

Different arms may retain different epochs. Equal passes through differently
sized cohorts do not imply equal numbers of updates. For a separate fixed-epoch
experiment, use `keep_checkpoint: last`: it always runs the configured number
of epochs and exports the final weights. Early stopping is disabled in this
mode, even if `early_stopping.enabled` is true. With `keep_checkpoint: best`,
`early_stopping.enabled: false` runs the full budget but restores the best
validation checkpoint.

## Score populations

| Population | Score | Label | Rows |
| --- | --- | --- | --- |
| CTR | `q` | `y` | All impressions |
| Native CTCVR | `q*v` | `y*z` | All impressions |
| Logged CTCVR | `p*v` | `y*z` | Allowed splits above |
| Clicked pCVR | `v` | `z` | Only `y == 1` |

PR-AUC is average precision. NE divides log loss by the population's label
entropy; CE is sum(predictions) / sum(labels). The validation logger and final
reports use the same evaluator. A head absent from a model has blank history
columns and no metric row. AUC is undefined on a population with only one class.

## Existing runs

These settings apply to new training runs. Existing checkpoints and saved
metrics are not retrained or rewritten by changing configs. In particular, old
quarter runs selected on `p*v` are not matched controls under this protocol.
Re-running the report script applies the population rules above, but cannot
change how an old checkpoint was selected.

The best-model registry refuses comparisons across different selection metrics.
For a fresh matched study, pass the same new registry path to each training
command, for example `--best-models best_models_standardized.yaml`, and compare
the resulting run folders. Existing run folders retain their configs and code
diffs for provenance.
