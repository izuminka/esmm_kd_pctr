# kd_pctr_pcvr

Reproduction code for the Ali-CCP experiments on Entire-Space Conversion
Modeling with an External Click-through Softlabel Knowledge Distillation.
Currently: the ESMM and DCN-V2 models, their losses, evaluation metrics, and
the data pipeline that feeds them, each built and validated against real data
one chunk at a time.

The matched ESMM controls and logged-score validation rules are documented in
[EVALUATION.md](evaluation/EVALUATION.md).

The 2025 CNN + attention multitask model is available as `cnn_attention`.
See [CNN_ATTENTION.md](models/CNN_ATTENTION.md) for the runnable recipe, architecture
assumptions, direct-conversion evaluation, and the paper's 0.661 CTR AUC reference.

## Layout

```
code/
  data/
    raw.py           # streams + parses the official Ali-CCP tarballs, joins
                     # skeleton rows to common features, drops z>y rows
    vocab.py         # per-field vocab (train-only, lexicographic sort, ID 0 =
                     # unseen) and batch encoding into EmbeddingBag-ready tensors
  model/
    esmm.py          # ESMM: shared embeddings + CTR/CVR towers, esmm_loss
    dcn.py           # DCN-V2 pCTR: weighted-mean pooling, cross + deep
                     # branches, dcn_loss (reuses data/raw.py and
                     # data/vocab.py unchanged -- only pooling differs from ESMM)
  metrics.py         # AUC, PR-AUC, normalized entropy, calibration ratio --
                     # framework-agnostic (plain arrays, no torch dependency)
  evaluate.py        # scores an ESMM model's CTR/CTCVR/pCVR heads via metrics.py
  conftest.py        # makes code/ importable as the test root; shared
                     # real_batch fixture (~2,000 real Ali-CCP rows)
  tests/             # one test file per module above, plus real-data
                     # integration tests using the real_batch fixture
  requirements.txt   # pinned dependency versions
```

Data files (`ali-ccp/sample_{train,test}.tar.gz`, the official Ali-CCP
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

## ESMM with a DCN CTR head

`esmm_dcn` replaces the existing ESMM CTR head with the matched DCN
architecture. It retains ESMM's shared embeddings, pCVR tower, loss
`BCE(y, q) + BCE(y*z, q*v)`, and optimizer settings. Both heads and their shared
embeddings train together from scratch without KD or teacher scores.
The standalone DCN optimizer is not used: this baseline keeps the rest of
the ESMM training recipe unchanged. Its recipe is
`configs/esmm_dcn/train.yaml`.

From the project root, prepare a reproducible four-seed study with:

```bash
code/.venv/bin/python code/scripts/run_esmm_dcn_study.py
```

Preparation saves configs, the complete working source, its Git diff, and a
submission manifest under `artifacts/esmm_dcn_studies/`. Add `--submit`
to launch seeds 42–45 as four concurrent Slurm GPU jobs plus a dependent CPU
report. Each seed trains for five epochs, keeps the minimum validation native
CTCVR NE checkpoint, and then evaluates pCVR, CTR, native CTCVR `q*v`, and
logged CTCVR `p*v` using the original frozen seed-42 DCN. `--report-study DIR`
regenerates per-seed results and mean ± SEM from the exact recorded run paths.

For a different prespecified seed set, add `--seeds 42 43 44 45 46 47 48 49 50 51`.
Array tasks map to the seeds recorded in the study manifest.

## Final tables (main.tex Tables 1 and 2)

The final tables are regenerated under `artifacts_final/` (full population) and
`artifacts_final_downsampled/` (5% nonclick retention) through Slurm:
`scripts/run_final_study.py --spec studies/final.yaml submit teacher`, then
`submit students <variant ids>`, then `report` and `tables`;
`scripts/verify_final_study.py` recomputes every stored score from the saved
predictions. Data, features, splits, architectures, optimizers and checkpoint
rules are those of the earlier draft; two training changes apply to every
model. The frozen DCN-V2 is retrained the same way, at the draft's seed 42; every
KD, KD-tau and Direct student distills from it, and it is the serving factor p
in both tables.

- **Output heads start at their training base rates.** Each final bias is the
  log-odds of P(click) = 0.03887418 and of P(conversion | click) = 0.0053562476
  on the training rows (`model_params.output_bias_init`). Under 5% nonclick
  retention the click start is the retained-set rate, 0.44697409; conversion
  given a click is unchanged because every click is retained. `scripts/train.py`
  refuses a start that does not match the training rows. With the heads at 0.5,
  the pCVR head first collapses and then oscillates through much of the first
  epoch, which left the selected checkpoints' calibration close to random and
  the 5% runs above NE 1.
- **Batches are stratified by label** (`training.stratify_batches`). Nonclicks,
  clicks without conversion and conversions are each spread evenly over the
  epoch, so every batch carries its share of the rare conversions (13-14 per
  batch at full population, against 2-28 under plain shuffling).

**Splits and seeds, against the literature.** The official Ali-CCP train/test
files are a time split: "the first 1/2 data in the time sequence [is the]
training set while the rest [is the] test set" (Ma et al., SIGIR 2018, ESMM,
Sec. 3.1). Validation is a random 10% of the training file, as in AITM
("We randomly take 10% of the train set as the validation set", Xi et al.,
KDD 2021, Sec. 5.1) and ESCM2 (Wang et al., SIGIR 2022, Sec. 5.1.1); ours is
deterministic by a hash of the sample id, giving the 38.07M / 4.23M / 43.02M
split AITM reports. Ten seeds follow ESMM ("All experiments are repeated 10
times", Sec. 3.1).

**Unseen users at test time.** Because test is the later half, 149,979 of the
295,787 test users (field 101) have no training impressions; they carry 32.9%
of test impressions. This is the time split, not corrupted data: both files
draw user ids from one dictionary with the same range, and these users have
full profiles and long behavior histories, but fewer impressions (94 vs 197 per
user). Their user id is unseen, so models fall back on the remaining user
fields there; validation, whose users all appear in training, cannot see it.

An AITM/ESCM2-style feature protocol (17 single-valued fields, tokens seen
fewer than ten times mapped to a shared OOV embedding) was also tried and
dropped: without the six multi-valued behavior-history fields, ranking was
worse for every model and the 5% runs stayed above NE 1.

## Ten-seed manuscript results

The combined study covers full-data ESMM, ESMM + DCN, KD at alpha 0, 0.3,
0.5, and 0.7, and both 5% nonclick-retention variants in Table 2. It uses
student seeds 42–51, one frozen seed-42 teacher, and sampling seed 2026.
Existing finished runs are pinned before submission; only missing runs train.

From the project root:

```bash
code/.venv/bin/python code/scripts/run_paper_seed_study.py --prepare artifacts/paper_seed_studies/NEW_STUDY
code/.venv/bin/python code/scripts/run_paper_seed_study.py --submit artifacts/paper_seed_studies/NEW_STUDY
code/.venv/bin/python code/scripts/report_paper_ten_seeds.py --study-dir artifacts/paper_seed_studies/NEW_STUDY
code/.venv/bin/python code/scripts/plot_esmm_comparison_auc_ne.py --summary artifacts/paper_seed_studies/NEW_STUDY/seed_summary.json
code/.venv/bin/python code/scripts/plot_downsampled_esmm_comparison_acm_format.py --summary artifacts/paper_seed_studies/NEW_STUDY/seed_summary.json
```

Preparation snapshots tracked and untracked working source, records hashes,
and creates private per-run registries. The Slurm array runs up to four jobs
at once, each with one GPU and four CPU cores. After each GPU task exports
test predictions, a corresponding CPU task scores them, with up to eight
scoring tasks running concurrently. GPUs can start later seeds while CPU
scoring continues. A failed task can resume its
recorded run without selecting a different checkpoint or consulting `latest`.
The report requires all ten seeds for every variant, matching training
configurations, five epochs, the prescribed validation checkpoint rule,
identical test populations, and checkpoint-bound serving metrics from the
fixed teacher. It writes per-seed values, means, and SEMs (sample standard
deviation divided by sqrt(10)), plus both manuscript tables. An incomplete
study cannot produce a new aggregate.

The direct-product baseline has a separate ten-seed study. It trains only
the conversion tower through `BCE(z, stopgrad(p) * v)`, using the same frozen
teacher and five-epoch budget. The original four runs are pinned; missing
seeds use the same GPU prediction and CPU scoring phases:

```bash
code/.venv/bin/python code/scripts/run_direct_product_study.py --prepare artifacts/direct_product_studies/NEW_STUDY
code/.venv/bin/python code/scripts/run_direct_product_study.py --submit artifacts/direct_product_studies/NEW_STUDY
code/.venv/bin/python code/scripts/report_direct_product_ten_seeds.py --study-dir artifacts/direct_product_studies/NEW_STUDY
```

DIRECT reports pCVR and serving CTCVR only because it has no auxiliary pCTR
tower. The report requires all ten seeds, the fixed teacher's train/validation/
test score hashes, and matching checkpoint and prediction provenance.

## Status

Built so far: data pipeline, ESMM, DCN-V2, evaluation metrics, and
ESMM-head evaluation (`evaluate.py`). Not yet built: a training loop,
checkpointing, DCN-V2 evaluation wiring, the half/quarter cohort-selection
logic, and the paper's KD/Direct-IPW treatment objectives -- these are
future work, not gaps in what's described above.
