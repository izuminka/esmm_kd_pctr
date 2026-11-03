"""Numeric definitions and row-identity guards for saved-score alignment."""

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.report_ctr_alignment import (
    AlignmentSums, ProductAlignmentSums, alignment_for_file, check_alignment, check_ordered_ids,
    check_configuration, load_teacher, ROOT, study_sources,
)


def test_soft_teacher_bce_uses_probabilities_not_observed_click_labels():
    p, q = np.array([.2, .8]), np.array([.3, .6])
    sums = AlignmentSums()
    sums.add(p, q)
    observed = sums.result()
    expected = (-.2*math.log(.3)-.8*math.log(.7)-.8*math.log(.6)-.2*math.log(.4))/2
    hard_label_bce = (-math.log(.7)-math.log(.6))/2
    assert observed['soft_bce'] == pytest.approx(expected)
    assert observed['soft_bce'] != pytest.approx(hard_label_bce)


def test_identical_interior_probabilities_reach_teacher_entropy_floor():
    p = np.array([.2, .5, .8])
    sums = AlignmentSums()
    sums.add(p, p)
    observed = sums.result()
    assert observed['soft_bce'] == pytest.approx(observed['teacher_entropy'])
    assert observed['bernoulli_kl'] == pytest.approx(0, abs=1e-15)
    assert observed['mae'] == 0
    assert observed['q_clipped_count'] == 0


def test_boundary_teacher_is_raw_while_only_student_is_clipped():
    sums = AlignmentSums()
    sums.add(np.array([0., 1.]), np.array([0., 1.]))
    observed = sums.result()
    assert observed['teacher_entropy'] == 0
    assert observed['soft_bce'] == pytest.approx(-math.log1p(-1e-7))
    assert observed['q_clipped_count'] == 2
    assert observed['mae'] == 0


def test_unequal_batch_sizes_weight_rows_instead_of_batch_means():
    p, q = np.array([.1, .2, .4, .8]), np.array([.2, .3, .5, .7])
    together, batched = AlignmentSums(), AlignmentSums()
    together.add(p, q)
    batched.add(p[:1], q[:1])
    batched.add(p[1:], q[1:])
    assert batched.result() == pytest.approx(together.result())


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, 1.01])
@pytest.mark.parametrize('target', ['teacher', 'student'])
def test_invalid_probabilities_fail_instead_of_being_silently_clipped(bad, target):
    p = np.array([bad if target == 'teacher' else .5])
    q = np.array([bad if target == 'student' else .5])
    with pytest.raises(ValueError, match='probability'):
        AlignmentSums().add(p, q)


@pytest.mark.parametrize('ids,previous', [([1, 1], None), ([2, 1], None), ([3, 4], 3)])
def test_duplicate_or_unordered_teacher_ids_fail_within_and_across_batches(ids, previous):
    with pytest.raises(ValueError, match='duplicated or not strictly increasing'):
        check_ordered_ids(np.array(ids, dtype=np.int64), previous)


@pytest.mark.parametrize('field', ['sample_id', 'y', 'z'])
def test_equal_row_counts_do_not_bypass_identity_and_label_checks(field):
    teacher = {'sample_id': np.array([1, 2], dtype=np.int64), 'y': np.array([0, 1]), 'z': np.array([0, 0])}
    student = {name: value.copy() for name, value in teacher.items()}
    student[field][1] += 1
    with pytest.raises(ValueError, match=field + ': alignment mismatch at row 11'):
        check_alignment(teacher, student, offset=10)


def test_streaming_parquet_alignment_and_cross_batch_id_guard(tmp_path):
    common = {'sample_id': pa.array([1, 2, 3], type=pa.int64()),
              'y': pa.array([0, 1, 1], type=pa.int8()), 'z': pa.array([0, 0, 1], type=pa.int8())}
    teacher_file, student_file = tmp_path/'teacher.parquet', tmp_path/'student.parquet'
    p = np.array([.2, .4, .8], dtype=np.float32)
    q = np.array([.3, .5, .7], dtype=np.float32)
    pq.write_table(pa.table({**common, 'score': p}), teacher_file)
    pq.write_table(pa.table({**common, 'q': q, 'v': q}), student_file)
    teacher = load_teacher(teacher_file, batch_size=2)
    expected = AlignmentSums()
    expected.add(p, q)
    observed = alignment_for_file(student_file, teacher, batch_size=2)
    assert {name: observed[name] for name in expected.result()} == pytest.approx(expected.result())
    products = ProductAlignmentSums()
    products.add(p, q, q)
    assert {name: observed[name] for name in products.result()} == pytest.approx(products.result())
    pq.write_table(pa.table({**common, 'sample_id': pa.array([1, 3, 2], type=pa.int64()), 'q': q, 'v': q}), student_file)
    with pytest.raises(ValueError, match='sample_id: alignment mismatch'):
        alignment_for_file(student_file, teacher, batch_size=2)
    pq.write_table(pa.table({**common, 'sample_id': pa.array([1, 2, 2], type=pa.int64()), 'score': p}), teacher_file)
    with pytest.raises(ValueError, match='across batches'):
        load_teacher(teacher_file, batch_size=2)


@pytest.mark.parametrize('variant,model,criterion,alpha', [
    ('esmm', 'esmm', 'ctcvr_ne', None),
    ('esmm_dcn', 'esmm_dcn', 'ctcvr_ne', None),
    ('kd03', 'esmm_logged_pctr', 'logged_ctcvr_ne', .3),
    ('kd05', 'esmm_logged_pctr', 'logged_ctcvr_ne', .5),
    ('kd07', 'esmm_logged_pctr', 'logged_ctcvr_ne', .7),
    ('kd0', 'esmm_logged_pctr', 'logged_ctcvr_ne', 0.),
])
def test_each_full_variant_keeps_its_model_alpha_and_checkpoint_rule(variant, model, criterion, alpha):
    selected = {'metric': criterion, 'split': 'val', 'epoch': 1, 'value': .9}
    source = {'variant': variant, 'seed': 42, 'selected_on': selected}
    run = {'status': 'FINISHED', 'selected_on': selected}
    teacher = ROOT/'artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z/predictions/test.parquet'
    config = {'model': model, 'training': {'seed': 42, 'epochs': 5, 'checkpoint_metric': criterion},
              'model_params': {'dcn_params': {'cross_rank': 128}} if variant == 'esmm_dcn' else {},
              'data': {}}
    if alpha is not None:
        config['model_params'] = {'objective': 'kd', 'alpha': alpha}
        config['logged_pctr'] = {'dcn_run': 'artifacts/dcn_matched_stats_wd3e6/20260917T232959Z'}
    check_configuration(source, run, config, teacher)
    if alpha is not None:
        config['model_params']['alpha'] = (alpha+.1) % 1
        with pytest.raises(ValueError, match='wrong KD objective or alpha'):
            check_configuration(source, run, config, teacher)
    else:
        config['training']['checkpoint_metric'] = 'logged_ctcvr_ne'
        with pytest.raises(ValueError, match='wrong checkpoint selection rule'):
            check_configuration(source, run, config, teacher)


def test_full_study_requires_ten_distinct_seeds_for_every_six_variants():
    variants = ['esmm', 'esmm_dcn', 'kd03', 'kd05', 'kd07', 'kd0']
    rows = [{'variant': variant, 'seed': seed, 'run_dir': str(Path('/tmp/ctr-test')/variant/str(seed))}
            for variant in variants for seed in range(42, 52)]
    # Downsampled data must not enter the full-population comparison.
    rows.append({'variant': 'down_esmm', 'seed': 42, 'run_dir': '/tmp/ctr-test/down'})
    assert len(study_sources({'sources': rows})) == 60
    incomplete = [row for row in rows if not (row['variant'] == 'esmm_dcn' and row['seed'] == 51)]
    with pytest.raises(ValueError, match='esmm_dcn: expected ten distinct seeds'):
        study_sources({'sources': incomplete})
    duplicated = [dict(row) for row in rows]
    duplicated[59]['seed'] = 50
    with pytest.raises(ValueError, match='kd0: expected ten distinct seeds'):
        study_sources({'sources': duplicated})


def test_full_study_rejects_reused_run_even_with_valid_seed_labels():
    rows = [{'variant': variant, 'seed': seed, 'run_dir': f'/tmp/ctr-test/{variant}/{seed}'}
            for variant in ['esmm', 'esmm_dcn', 'kd03', 'kd05', 'kd07', 'kd0'] for seed in range(42, 52)]
    rows[10]['run_dir'] = rows[0]['run_dir']
    with pytest.raises(ValueError, match='reuses a run directory'):
        study_sources({'sources': rows})


def test_product_bce_uses_native_target_and_serving_prediction_in_that_order():
    p, q, v = (np.array(values, dtype=np.float32) for values in ([.2, .7], [.3, .6], [.4, .2]))
    sums = ProductAlignmentSums()
    sums.add(p, q, v)
    observed = sums.result()
    target, prediction = q.astype('float64')*v.astype('float64'), p.astype('float64')*v.astype('float64')
    expected = np.mean(-target*np.log(prediction)-(1-target)*np.log1p(-prediction))
    reversed_bce = np.mean(-prediction*np.log(target)-(1-prediction)*np.log1p(-target))
    assert observed['product_soft_bce'] == pytest.approx(expected, abs=1e-15)
    assert abs(observed['product_soft_bce']-reversed_bce) > .001
    assert observed['product_mae'] == pytest.approx(np.mean(np.abs(target-prediction)))


def test_raw_product_below_one_e_minus_seven_is_not_clipped():
    p, q, v = (np.array([value], dtype=np.float32) for value in [1e-9, 2e-9, .1])
    target = float(q[0])*float(v[0])
    prediction = float(p[0])*float(v[0])
    sums = ProductAlignmentSums()
    sums.add(p, q, v)
    result = sums.result()
    expected = -target*math.log(prediction)-(1-target)*math.log1p(-prediction)
    assert result['product_soft_bce'] == pytest.approx(expected, rel=1e-14)
    assert result['product_prediction_min'] == prediction < 1e-7
    assert result['product_prediction_zero_count'] == 0
    clipped = -target*math.log(1e-7)-(1-target)*math.log1p(-1e-7)
    assert abs(result['product_soft_bce']-clipped) > 9e-8


def test_product_endpoints_use_zero_log_zero_convention_exactly():
    sums = ProductAlignmentSums()
    sums.add(np.array([0., 1., .2]), np.array([0., 1., .7]), np.array([1., 1., 0.]))
    result = sums.result()
    assert result['product_soft_bce'] == 0
    assert result['product_native_entropy'] == 0
    assert result['product_bernoulli_kl'] == 0
    assert result['product_mae'] == 0
    assert result['product_prediction_zero_count'] == 2
    assert result['product_prediction_one_count'] == 1


@pytest.mark.parametrize('p,q', [(0., .2), (1., .8)])
def test_incompatible_product_endpoint_fails_instead_of_arbitrary_clipping(p, q):
    with pytest.raises(ValueError, match='raw product BCE is nonfinite'):
        ProductAlignmentSums().add(np.array([p]), np.array([q]), np.array([1.]))


def test_product_entropy_is_native_model_specific_and_batch_weighting_is_per_row():
    p = np.array([.2, .4, .8], dtype=np.float32)
    q = np.array([.3, .5, .7], dtype=np.float32)
    v = np.array([.1, .2, .3], dtype=np.float32)
    all_rows, batched, other_model = ProductAlignmentSums(), ProductAlignmentSums(), ProductAlignmentSums()
    all_rows.add(p, q, v)
    batched.add(p[:1], q[:1], v[:1])
    batched.add(p[1:], q[1:], v[1:])
    other_model.add(p, p, v)
    assert batched.result() == pytest.approx(all_rows.result())
    assert abs(other_model.result()['product_native_entropy']-all_rows.result()['product_native_entropy']) > .001
