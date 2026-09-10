# kd_pctr_to_pcvr

Reproduction code for the Ali-CCP experiments in
[`../paper/frozen_pctr_iclr2027.tex`](../paper/frozen_pctr_iclr2027.tex)
("Serving-Aligned Entire-Space Conversion Modeling with an External
Click-through Softlabel Knowledge Distillation"). Currently: the ESMM and
DCN-V2 models, their losses, evaluation metrics, and the data pipeline that
feeds them, each built and validated against real data one chunk at a time.

## Layout

```
code/
  data/
    raw.py        # streams + parses the official Ali-CCP tarballs, joins
                   # skeleton rows to common features, drops z>y rows
    vocab.py       # per-field vocab (train-only, lexicographic sort, ID 0 =
                   # unseen) and batch encoding into EmbeddingBag-ready tensors
  model/
    esmm.py        # ESMM: shared embeddings + CTR/CVR towers, esmm_loss
    dcn.py         # DCN-V2 pCTR: weighted-mean pooling, cross + deep
                   # branches, dcn_loss (reuses data/raw.py and
                   # data/vocab.py unchanged -- only pooling differs from ESMM)
  metrics.py        # AUC, PR-AUC, normalized entropy, calibration ratio --
                     # framework-agnostic (plain arrays, no torch dependency)
  evaluate.py        # scores an ESMM model's CTR/CTCVR/pCVR heads via metrics.py
  conftest.py         # makes code/ importable as the test root; shared
                       # real_batch fixture (~2,000 real Ali-CCP rows)
  tests/               # one test file per module above, plus real-data
                        # integration tests using the real_batch fixture
  requirements.txt      # pinned dependency versions
```

Data files (`ali-cpp/sample_{train,test}.tar.gz`, the official Ali-CCP
release) live one level up, outside this directory.

## Design notes

- **DCN-V2 reuses the ESMM data pipeline unchanged.** The paper specifies
  that DCN-V2 uses the same join, vocab, and token encoding as ESMM, and
  differs only in pooling (weighted mean vs. ESMM's weighted sum) -- so
  `data/raw.py` and `data/vocab.py` are shared as-is; only `model/dcn.py`
  does its own weight normalization before pooling.
- **`model/esmm.py` and `model/dcn.py` do not import from each other.**
  Both independently call the same PyTorch init functions
  (`nn.init.kaiming_uniform_(..., mode="fan_out")` for embeddings,
  `nn.Linear` defaults or `nn.init.kaiming_normal_` for dense layers) rather
  than sharing a custom helper -- see each file's inline comments for why
  those specific library calls reproduce the paper's init formulas exactly.
- **`metrics.py` has no torch dependency.** It operates on plain arrays so
  it stays reusable outside a torch training loop; `evaluate.py` is the one
  place that converts model tensors to numpy before calling it.
- Every module is validated against real data (a ~2,000-row Ali-CCP slice,
  shared across test files via the `real_batch` fixture in `conftest.py`),
  not just synthetic fixtures -- see each `tests/test_*.py` for the specific
  per-component and integration checks.

## Running tests

```
pip install -r requirements.txt
pytest tests/
```

The real-data integration tests (anything using the `real_batch` fixture)
take a few minutes, dominated by scanning the common-features file for the
join -- this is expected, not a hang.

## Status

Built so far: data pipeline, ESMM, DCN-V2, evaluation metrics, and
ESMM-head evaluation (`evaluate.py`). Not yet built: a training loop,
checkpointing, DCN-V2 evaluation wiring, the half/quarter cohort-selection
logic, and the paper's KD/Direct-IPW treatment objectives -- these are
future work, not gaps in what's described above.
