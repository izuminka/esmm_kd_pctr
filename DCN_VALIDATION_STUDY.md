# DCN versus ESMM CTR: validation study

## Acceptance criterion

The requested improvement is **at least 0.5% lower validation CTR normalized
entropy (NE)** than the ESMM CTR tower. This means
`DCN_NE <= 0.995 * ESMM_NE`. Against the seed-42 reference, the target is
0.941221443. No completed candidate below meets this target.

## Comparison protocol

- Train on all 38,070,359 encoded training impressions. Validate on the same
  4,229,546 impressions, including 164,302 clicks, for both models.
- Use the same encoded source fields, 20-dimensional embeddings, batch size
  65,536, and row shuffle rule. Seed model construction before initialization.
- Run all five epochs; disable early stopping and restore the checkpoint with
  minimum validation CTR NE. Compute CTR NE on all validation impressions.
- The original seed-42 ESMM run selected clicked-pCVR NE; its selected epoch 1
  also independently minimizes its recorded CTR NE over all five epochs.
  The seed-43 reference explicitly selects CTR NE.
- Models keep their own architecture, objective, and tuned learning rate.
  ESMM uses joint CTR/CTCVR loss; DCN uses click BCE. This compares predictive
  models, not an isolated change to architecture under an identical loss.
- New tuning configurations set both `evaluate_test: false` and
  `load_test: false`. Reports also skip test artifacts when test evaluation is
  disabled. Checkpoint export must reproduce the selected validation NE.
- NE is log loss divided by the entropy of the observed validation click rate.
  Lower is better. CE in the metric files is the prediction/label sum ratio,
  not cross-entropy.

## Completed five-epoch results

Values below are minimum epoch validation CTR NE. Reloaded/exported metrics
can differ in the final decimal places due to prediction serialization.

| Model/config | Seed | Best NE | Epoch | Run folder under `artifacts/` |
|---|---:|---:|---:|---|
| ESMM reference | 42 | 0.945951199 | 1 | `esmm/20260917T213034Z` |
| DCN mean + bag statistics | 42 | 0.945847523 | 1 | `dcn_matched_stats/20260917T222847Z` |
| DCN weight decay 1e-4 | 42 | 0.949043027 | 5 | `dcn_matched_stats_wd1e4/20260917T231025Z` |
| DCN weight decay 1e-6 | 42 | 0.944903250 | 1 | `dcn_matched_stats_wd1e6/20260917T231439Z` |
| DCN weight decay 3e-6 | 42 | 0.944234032 | 2 | `dcn_matched_stats_wd3e6/20260917T232959Z` |
| DCN weight decay 1e-5 | 42 | 0.944776065 | 2 | `dcn_matched_stats_wd1e5/20260917T232959Z` |
| Same unregularized DCN, cross layers removed | 42 | 0.947040632 | 1 | `dcn_matched_stats_no_cross/20260917T231620Z` |
| ESMM CTR-selected reference | 43 | 0.945347582 | 1 | `esmm_ctr_reference_seed43/20260917T232216Z` |
| DCN weight decay 1e-6 | 43 | 0.944848730 | 1 | `dcn_matched_stats_wd1e6_seed43/20260917T232216Z` |
| Stacked DCN weight decay 1e-6 | 42 | 0.944366388 | 1 | `dcn_matched_stats_wd1e6_stacked/20260917T233749Z` |

The regularized DCN improves NE by about 0.111% at seed 42 and 0.053% at seed
43. Neither reaches 0.5%. Removing crosses worsens the otherwise matched
unregularized model, supporting retention of the explicit cross network.
These comparisons do not establish statistical significance.

The intermediate weight-decay runs also completed five training epochs. The
best is 3e-6 at epoch 2, a 0.182% NE reduction against seed-42 ESMM; the 0.5%
target remains unmet. Both intermediate settings worsen after epoch 2.
Both runs are FINISHED, and their restored-checkpoint validation NE agrees
with the selection score within 1e-6. The 3e-6 model has validation ROC-AUC
0.693054029, PR-AUC 0.087550186, NE 0.944234018, and calibration ratio
1.016065975. Neither run contains a test prediction artifact or test metrics.

Training loss continues to decrease while validation NE deteriorates after
epoch 1 in the lightly regularized runs. For seed 42, weight decay 1e-6 gives
validation NE 0.94490, 0.94647, 0.97326, 1.08168, 1.26618 over five epochs.
This supports overfitting as a remaining issue. Stronger weight decay 1e-4
reduces that deterioration but gives a worse best validation result.

## Experiments in flight

- Slurm 446 finished five training epochs and is exporting the selected
  checkpoint. Its stacked structure improved on parallel at weight decay 1e-6,
  but did not beat the parallel 3e-6 model. Validation NE reached 1.5090 by
  epoch 5, despite falling training BCE, so overfitting remains.
- Slurm 447: 64-dimensional embeddings with weight decay 3e-6. Only embedding
  width changes from job 444; the raw inputs and evaluation population remain
  fixed. This tests additional capacity under regularization, not equality of
  model parameter counts with ESMM.
- Slurm 448, after 446: embedding learning rate 1e-4, dense learning rate
  unchanged at 5e-4, otherwise identical to the best parallel 3e-6 recipe.
  The hypothesis is that slower embedding updates reduce memorization while
  preserving the dense network's learning rate. The option is DCN-only and
  defaults to the existing single-rate Adam behavior.

Use `bash_scripts/train_dcn.sh` with `DCN_CONFIG` pointing to the corresponding
tracked YAML. The seed-43 comparison is in `bash_scripts/compare_ctr_seed43.sh`.
`compare_ctr_seeds.sh` prepares seeds 44–46 but has **not** been submitted;
additional confirmation is deferred until a candidate approaches the target.
Each job requests one GPU. Every run saves its configuration and code diff.

## Research and limitations

The [DCN-V2 paper, section 3.4](https://arxiv.org/html/2008.13535v2#S3.SS4)
defines both stacked and parallel structures and finds their relative
performance depends on the dataset. Its reported benchmark gains on Criteo and
MovieLens do not guarantee a 0.5% NE gain on this Ali-CCP split.

The previously cited DCN-V2 Ali-CCP AUC of 0.6289 is a **CVR** result in the
[source article](https://pinnaclepubs.com/index.php/JSISI/article/view/730), so it
cannot benchmark our CTR model. Published results with different labels,
encoders, or split rules are not numerical targets for this comparison.

Earlier exploration inspected test metrics and test OOV statistics and used
them to motivate some candidates. That exposure cannot be undone: the entire
search must not be described as test-independent. The regularization,
cross-removal, independent-seed, and stacked experiments documented here use
training/validation evidence and disable test evaluation. Validation is also
being used repeatedly for tuning; its improvement is not an unbiased estimate
of unseen-data improvement.

## Verification

- Test-disabled training must not load test data even if `load_test: true`;
  its report must ignore stale test artifacts. Integration regression passed.
- Legacy and encoded ESMM selection paths now agree on CTR NE, CTR PR-AUC,
  clicked-pCVR NE, and CTCVR PR-AUC. All 38 protocol tests passed.
- Stacked cross-to-MLP data flow, gradients, and default parallel checkpoint
  compatibility pass the DCN tests (21 passed; raw-data test excluded).
- Training and encoded-export constructors reproduce identical predictions
  from a restored stacked state dictionary.
- Combined regression run: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  .venv/bin/python -m pytest -q tests/test_dcn.py tests/test_esmm_protocol.py
  tests/test_runs.py -k 'not real_data'` passed 79 tests, with one raw-data
  test deselected.
- Separate-rate optimizer tests verify that embedding updates are scaled as
  configured while dense updates remain identical, all parameters are covered
  exactly once, and invalid rates are rejected. These tests plus the ESMM
  protocol suite passed 43 tests after introducing the optional rate.
