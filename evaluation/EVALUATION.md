# Train once, evaluate many pCTR sources

Active recipes live in `configs/<model>/train.yaml`: ESMM, the selected
`dcn_matched_stats_wd3e6`, and full-data KD–0.3, KD–0.5, KD–0.7, and KD–teacher.
Exploratory and appendix recipes are preserved in `configs_old/<model>/`.
Downsampled recipes use the same layout under `configs_downsampled/`.
Run names come from the parent directory of `train.yaml`; existing flat
configs and run snapshots remain supported.

Plain ESMM training configs have no logged-pCTR source or evaluation flag.
They train and select on their native scores. KD configs retain the teacher
inputs needed by their training and validation objectives.

## Separate test evaluation

All active training recipes set `training.evaluate_test: false`: training
selects the checkpoint on validation and reports train/validation metrics.
Run `scripts/evaluate_run.py --run-dir <run>` afterward for native test metrics.
Add one or more `--logged-config` arguments to evaluate logged CTCVR. The add-on
works with any pCVR predictor exporting `v`; no ESMM model-name check is used.
A new architecture still needs the usual model builder and prediction support.

## Independent logged evaluation

Each `eval_<pctr_model>.yaml` names one frozen producer:

```yaml
pctr_model: dcn_matched_stats_wd3e6
logged_pctr:
  dcn_run: artifacts/dcn_matched_stats_wd3e6/20260917T232959Z
  predictions:
    test: artifacts/dcn_matched_stats_wd3e6/20260917T232959Z/predictions/test.parquet
  score_column: score
```

From the project root:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 code/.venv/bin/python code/scripts/evaluate_run.py \
  --run-dir artifacts_downsampled/esmm_retention_5pct/20260922T234215Z \
  --logged-config code/configs_downsampled/esmm/eval_dcn_matched_stats_wd3e6.yaml \
  --gpus 1
```

The evaluator reuses `predictions/test.parquet` if present, or loads the
selected checkpoint and generates all native heads once on the full test set.
It reports native metrics before evaluating optional logged sources. Subsequent sources reuse these predictions without loading the
model or encoded feature arrays into a GPU. They join logged pCTR by sample ID,
verify unique IDs and matching click labels, and score `p*v` against `y*z`.
No training or checkpoint reselection occurs. No correction is applied to
logged pCTR or pCVR; the downsampling correction belongs only to native pCTR.

```text
<trained_run>/
├── best.pt
├── config.yaml                         # Original training snapshot
├── metrics/                            # Original native metrics
├── metrics.json                        # Original aggregate report
├── predictions/
│   ├── test.parquet                     # sample_id, y, z, native score columns
│   └── test.provenance.json             # Checkpoint/input identity and file hash
├── metrics_logged/
│   └── <pctr_model>/
│       ├── config.yaml
│       ├── provenance.json
│       └── test.json
└── metrics_combined.json               # Explicit combined report
```

The cache binds predictions to checkpoint/config hashes and encoded test/common
file sizes and modification timestamps. Encoded data should be immutable;
reprocessing or replacing them invalidates the cache. Cache contents are also
hashed. A stale/incomplete cache requires `--refresh-cache`. Writes are locked
to prevent concurrent evaluations from building inconsistent caches.

Each logged output records the selected checkpoint hash, cache hash, producer run,
and producer prediction-file hash. Reusing a model name for a different source
is rejected; use a distinct name such as `dcn_model_run2` to preserve both.
Checkpoints, training snapshots, and run status are preserved. Native test
metrics are created when missing, or recomputed with `--refresh-cache`.
Legacy prediction exports without a sidecar are adopted after checking their
IDs, labels, and age against the checkpoint/config; their provenance is trusted
as an export of the finished run. Subsequent reuse checks the recorded hashes.

To display native and separately logged metrics together:

```bash
code/.venv/bin/python code/scripts/report_metrics.py \
  --run-dir artifacts_downsampled/esmm_retention_5pct/20260922T234215Z \
  --include-logged
```

The command labels each external source and writes `metrics_combined.json`.
Without `--include-logged`, reporting retains the existing native-report path.

## Score populations and selection

| Population | Score | Label | Rows |
| --- | --- | --- | --- |
| CTR | Native `q` (posterior-corrected when downsampled) | `y` | All impressions |
| Native CTCVR | `q*v` | `y*z` | All impressions |
| Logged CTCVR | Frozen `p*v` | `y*z` | Full test population |
| Clicked pCVR | `v` | `z` | Only `y == 1` |

PR-AUC is average precision. NE divides log loss by label entropy; CE is
sum(predictions) / sum(labels). AUC is undefined for a single-class population.

Active ESMM runs select the lowest validation native CTCVR NE; active KD runs
select the lowest validation logged CTCVR NE. Both train five epochs and
restore the selected checkpoint before final evaluation; ties retain the
earlier checkpoint. KD's conversion loss still uses `q*v`.
Test metrics never select the checkpoint. The best-run registry refuses
comparisons across different selection metrics.

Existing saved configs with logged evaluation enabled retain their historical
behavior. The separate evaluator adds new results without changing that history.
