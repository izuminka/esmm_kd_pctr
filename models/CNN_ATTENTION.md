# CNN + attention multitask model (2025)

This implements the architecture described by El-Sherief, Khafagy, and Sweidan,
*Multitask Model with an Attention Mechanism for Sequentially Dependent Online
User Behaviors to Enhance Audience Targeting*, IJACSA 16(4), 2025.
[Paper and DOI](https://doi.org/10.14569/IJACSA.2025.01604112).
Table II reports Ali-CCP click ROC-AUC **0.661**. This is a reference target,
not a reproduced result or a smoke-test acceptance threshold.

## Architecture and assumptions

The paper specifies shared embeddings of width 5, a CNN click tower, a dense
conversion tower, click-information transfer through attention, dense widths
128/64/32, batch size 1,000, and five training epochs. It does not give a complete
CNN architecture, optimizer recipe, initialization, dropout rate, or calibration
weight. The implementation makes these choices explicit:

| Component | Implementation |
| --- | --- |
| Shared input | Weighted-sum EmbeddingBag per field; fields sorted as in the existing pipeline |
| CNN input | Concatenated embeddings shaped `[B, 1, F * 5]`; convolution over features, not a temporal user sequence |
| Click tower | Conv1d channels 32/64/32, kernel 3, stride 1, padding 1, each with ReLU and dropout 0.2; flatten; dense 128/64/32 with ReLU and dropout 0.2 |
| Conversion tower | Shared flattened input to dense 128/64/32 with ReLU and dropout 0.2 |
| Transfer | Linear 32-to-32 projection of the click representation; stack with the conversion representation |
| Attention | Linear query/key/value projections; each vector's Q·K divided by sqrt(32), softmax across the two vectors, weighted sum of V (equations 2–4) |
| Heads | Independent linear-to-one click and direct conversion logits; sigmoid for reporting |
| Initialization | Existing ESMM embedding rule U(-1/sqrt(K), 1/sqrt(K)), where K includes unknown ID 0; PyTorch defaults for dense and convolution layers |
| Optimization | Adam, learning rate 0.001, no weight decay or gradient clipping, calibration alpha 1, seed 42 |

The loss is the sum of mean click BCE and mean conversion BCE, computed with
`binary_cross_entropy_with_logits`, plus
`alpha * mean(relu(p_conversion - p_click))`.
Equation 6 prints the opposite sign; we correct it to implement the prose's
constraint that conversion should not exceed click probability. This is a soft
penalty, so predictions are not guaranteed to satisfy that inequality.
Conversion gradients flow through attention into the click tower without detach.

## Data and evaluation

Reuse `ali-ccp/encoded`, including its train-only vocabulary, weighted feature
bags, exclusion of conversion-without-click rows, 10% hash-based validation split
from the official training file, and untouched official test population. No
rare-feature filtering or history construction is added. This differs from the
paper's stated 50/10/40 partition and removal of features occurring fewer than
ten times, so this is an architecture implementation on our existing protocol,
not an exact reproduction of the published experiment.

`forward(batch)` returns two `[B]` logit tensors. Prediction files contain
`sample_id, y, z, q, c`, where q is click probability and c is direct
impression-level conversion probability. Report CTR on `(q, y)` and CTCVR on
`(c, z)`, both over all impressions. The model has no conditional pCVR head;
do not multiply c by q or logged pCTR and do not label c as pCVR.

Run all five epochs, select the lowest complete-validation CTR normalized entropy
(`ctr_ne`, ties keep the earlier checkpoint), and restore that checkpoint before
final scoring. Test data never choose the epoch or rank runs. The history
records validation AUC, PR-AUC, NE, and CE for both heads. The existing `ctr_auc`
option remains available for earlier runs; it requires both validation click classes.
Changing the default does not alter an active run's snapshotted selection metric
or relabel existing AUC-selected checkpoints as NE-selected.

The test CTR metric row includes `reference_auc: 0.661` and
`auc_delta: measured_auc - 0.661`; the report also prints them with the protocol
caveat. Conversion metrics may be undefined on a tiny fixture with no positives.
The default recipe writes metrics only, while `save_predictions: true` enables
prediction parquets. Test scoring uses a separate process after training releases
the dataset and optimizer. The new model supports the encoded-data pipeline;
the legacy list-of-Impression training interface is not supported.

## Run

From the project root, on a machine with the encoded dataset and sufficient
memory (preferably an allocated GPU):

```bash
code/.venv/bin/python code/scripts/train.py \
  --config code/configs/cnn_attention.yaml \
  --best-models artifacts/cnn_attention_best_models.yaml
```

Training creates `artifacts/cnn_attention/<timestamp>/` with the copied config,
run provenance, `history.csv`, `best.pt`, per-split metrics, and `metrics.json`.
CNN checkpoints include all constructor settings; encoded prediction restores
these rather than taking architecture settings from a subsequently edited YAML.
This command launches a full five-epoch run; it is not part of the smoke test.

From `code/`, run the model and real-data integration tests with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_cnn_attention.py tests/test_train_cnn_attention.py
```

The integration tests use the existing fixture carved from real Ali-CCP CSVs,
the full default architecture, two short CPU epochs, and temporary artifacts and
best-model registries. They cover both metrics-only and prediction-export modes,
checkpoint restoration, independent test scoring, and complete-validation NE.
