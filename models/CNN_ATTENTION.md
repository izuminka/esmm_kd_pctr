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

| Component | Robust architecture (version 2) |
| --- | --- |
| Shared input | Weighted-mean EmbeddingBag per field; inactive/OOV bags use the trainable reserved ID-0 vector; append one presence bit per field |
| Cold-start training | Replace field 101 by its missing state independently on 33.1509% of training rows |
| CNN input | Concatenated embedding-plus-presence vectors shaped `[B, 1, F * 6]`; convolution over features, not a temporal sequence |
| Click tower | Conv1d channels 32/64/32, kernel 3, stride 1, padding 1, each with ReLU and dropout 0.2; flatten; dense 128/64/32 |
| Conversion tower | Shared flattened input to dense 128/64/32 |
| Transfer | Linear 32-to-32 click projection stacked with the conversion representation; normalize each task vector |
| Attention | Q/K cosine similarities bounded to [-1, 1], softmax across tasks, weighted sum of V; add a residual from the normalized conversion representation |
| Heads | Independent linear-to-one click and direct impression-level conversion logits |
| Initialization | ID 0 shares the normal embedding initialization and is trained by natural missing rows plus field masking |
| Optimization | Adam, learning rate 0.001, no weight decay or gradient clipping, calibration alpha 1, seed 42 |

Architecture version 1 preserves the original weighted-sum, all-zero missing
state, unbounded dot-product attention, and no-residual tensor shapes. Prediction
automatically assigns unversioned checkpoints to version 1, so prior runs remain
reproducible. Version 2 is a robustness recipe motivated by measured failures;
it is not claimed to be a verbatim implementation of unpublished paper details.

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

`forward(batch, forced_missing=None)` returns two `[B]` logit tensors.
`forced_missing` is used only for deterministic robustness evaluation; ordinary
prediction passes no override. Prediction files contain `sample_id, y, z, q, c`,
where q is click probability and c is direct
impression-level conversion probability. Report CTR on `(q, y)` and CTCVR on
`(c, z)`, both over all impressions. The model has no conditional pCVR head;
do not multiply c by q or logged pCTR and do not label c as pCVR.

Run all five epochs and select the lowest cold-start CTR normalized entropy
(`cold_ctr_ne`, ties keep the earlier checkpoint). Each validation call creates a
deterministic deployment-like mixture: 33.1509% of rows are scored with field 101
forced through the learned missing state and the rest are scored normally. This
rate is an input-coverage measurement, not a metric fitted on test labels.

History records ordinary and cold-start CTR metrics plus mean click/conversion
attention weights, attention entropy, and the minimum task weight. Bounded cosine
scores cannot numerically underflow a task weight as the version-1 checkpoint did.
Test labels never choose an epoch or rank runs. Ordinary `ctr_ne`, `ctr_auc`,
and `ctr_prauc` selection remain available for controlled ablations.

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
  --config code/configs_old/cnn_attention/train.yaml \
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
best-model registries. They cover metrics-only and prediction-export modes, old
and new checkpoint restoration, learned missing-state gradients, bounded
attention, independent test scoring, and complete cold-start validation NE.
