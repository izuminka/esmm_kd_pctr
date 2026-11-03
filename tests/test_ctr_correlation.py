"""Head/product Pearson definitions, batch merging, identity and filtered reports."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import pearsonr

from scripts import report_ctr_correlation as correlation
from scripts.report_ctr_correlation import CenteredCovariance, correlation_for_file


@pytest.mark.parametrize('scale,offset,expected', [(1., 0., 1.), (3., 4., 1.), (-2., .5, -1.)])
def test_identity_and_affine_relationships(scale, offset, expected):
    p = np.array([.1, .2, .5, .8])
    moments = CenteredCovariance()
    moments.add(p, p*scale+offset)
    assert moments.result() == pytest.approx(expected, abs=1e-14)


def test_unequal_batches_match_full_scipy_correlation():
    p, q = np.array([.1, .3, .2, .8, .6, .5]), np.array([.2, .1, .4, .9, .4, .7])
    together, batched = CenteredCovariance(), CenteredCovariance()
    together.add(p, q)
    for start, end in [(0, 1), (1, 5), (5, 6)]:
        batched.add(p[start:end], q[start:end])
    expected = pearsonr(p, q).statistic
    assert together.result() == pytest.approx(expected, abs=1e-14)
    assert batched.result() == pytest.approx(expected, abs=1e-14)


def test_centered_merge_handles_large_common_offsets_without_raw_moment_cancellation():
    p = 1e9+np.array([0., .25, .5, 1., 2.])
    q = 1e9+np.array([.5, 0., 1., 1.5, 3.])
    moments = CenteredCovariance()
    moments.add(p[:2], q[:2])
    moments.add(p[2:], q[2:])
    assert moments.result() == pytest.approx(pearsonr(p, q).statistic, abs=2e-8)


@pytest.mark.parametrize('p,q', [([.2, .2, .2], [.1, .2, .3]), ([.1, .2, .3], [.5, .5, .5])])
def test_constant_variance_is_explicitly_undefined(p, q):
    moments = CenteredCovariance()
    moments.add(p, q)
    with pytest.raises(ValueError, match='undefined for constant'):
        moments.result()


def test_single_row_cannot_define_correlation():
    moments = CenteredCovariance()
    moments.add([.2], [.3])
    with pytest.raises(ValueError, match='at least two'):
        moments.result()


@pytest.mark.parametrize('p,q', [([np.nan, .2], [.1, .2]), ([.1, .2], [.2, np.inf]), ([.1], [.2, .3])])
def test_invalid_values_or_lengths_fail(p, q):
    with pytest.raises(ValueError, match='nonfinite|equal one-dimensional'):
        CenteredCovariance().add(p, q)


@pytest.mark.parametrize('batch_size', [1, 3, 100])
def test_parquet_pass_checks_all_ids_and_matches_scipy(tmp_path, batch_size):
    p = np.array([.1, .3, .6, .9], dtype=np.float32).astype('float64')
    q = np.array([.2, .4, .5, .8], dtype=np.float32)
    v = np.array([.9, .1, .8, .2], dtype=np.float32)
    teacher = {'sample_id': np.array([1, 2, 3, 4], dtype='int64'),
               'y': np.array([0, 1, 1, 0], dtype='int8'), 'z': np.array([0, 0, 1, 0], dtype='int8'), 'score': p}
    columns = {name: pa.array(values) for name, values in teacher.items() if name != 'score'}
    path = tmp_path/'student.parquet'
    pq.write_table(pa.table({**columns, 'q': q, 'v': v}), path)
    result = correlation_for_file(path, teacher, batch_size=batch_size)
    assert result['rows'] == 4
    assert result['pearson'] == pytest.approx(pearsonr(p, q.astype('float64')).statistic, abs=1e-14)
    expected_product = pearsonr(q.astype('float64')*v.astype('float64'), p*v.astype('float64')).statistic
    assert result['product_pearson'] == pytest.approx(expected_product, abs=1e-14)
    assert result['product_scipy_pearson'] == pytest.approx(expected_product, abs=1e-14)
    assert result['product_absolute_difference'] <= 1e-14
    assert result['product_pearson'] != pytest.approx(result['pearson'], abs=1e-3)
    pq.write_table(pa.table({**columns, 'sample_id': pa.array([1, 2, 4, 3], type=pa.int64()), 'q': q, 'v': v}), path)
    with pytest.raises(ValueError, match='sample_id: alignment mismatch at row 2'):
        correlation_for_file(path, teacher, batch_size=batch_size)


def saved_probabilities(tmp_path, p=None, q=None, v=None):
    p = np.array([.1, .3, .6, .9] if p is None else p, dtype=np.float32)
    q = np.array([.2, .4, .5, .8] if q is None else q, dtype=np.float32)
    v = np.array([.9, .1, .8, .2] if v is None else v, dtype=np.float32)
    teacher = {'sample_id': np.arange(1, len(p)+1, dtype=np.int64),
               'y': np.ones(len(p), dtype=np.int8), 'z': np.zeros(len(p), dtype=np.int8), 'score': p}
    columns = {name: values for name, values in teacher.items() if name != 'score'}
    path = tmp_path/'student.parquet'
    pq.write_table(pa.table({**columns, 'q': q, 'v': v}), path)
    return path, teacher


def test_products_cast_heads_to_float64_before_multiplication(tmp_path):
    p = np.array([1e-30, 2e-30, 4e-30, 7e-30], dtype=np.float32)
    q = np.array([3e-30, 1e-30, 5e-30, 2e-30], dtype=np.float32)
    v = np.array([1e-20, 4e-20, 2e-20, 3e-20], dtype=np.float32)
    assert np.all(p*v == 0) and np.all(q*v == 0)  # A float32 product destroys all variation.
    path, teacher = saved_probabilities(tmp_path, p, q, v)
    result = correlation_for_file(path, teacher, batch_size=3)
    expected = pearsonr(q.astype('float64')*v.astype('float64'),
                        p.astype('float64')*v.astype('float64')).statistic
    assert result['product_pearson'] == pytest.approx(expected, abs=1e-14)
    assert result['product_scipy_pearson'] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize('column,value', [('score', np.nan), ('score', 1.1),
                                         ('q', np.inf), ('q', -.1),
                                         ('v', np.nan), ('v', 1.1), ('v', -.1)])
def test_head_probabilities_must_be_finite_and_in_unit_interval(tmp_path, column, value):
    path, teacher = saved_probabilities(tmp_path)
    if column == 'score':
        teacher['score'][0] = value
    else:
        table = pq.read_table(path)
        values = table.column(column).to_numpy().copy()
        values[0] = value
        table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
        pq.write_table(table, path)
    with pytest.raises(ValueError, match='nonfinite probability|probability outside'):
        correlation_for_file(path, teacher, batch_size=3)


@pytest.mark.parametrize('problem', ['missing', 'wrong_type', 'null'])
def test_v_column_is_required_with_float32_nonnull_probabilities(tmp_path, problem):
    path, teacher = saved_probabilities(tmp_path)
    table = pq.read_table(path)
    if problem == 'missing':
        table = table.drop(['v'])
    else:
        values = pa.array([.2, .3, .4, .5], type=pa.float64()) if problem == 'wrong_type' else pa.array(
            [.2, None, .4, .5], type=pa.float32())
        table = table.set_column(table.schema.get_field_index('v'), 'v', values)
    pq.write_table(table, path)
    with pytest.raises(ValueError, match='expected v column of type float32|v: null values'):
        correlation_for_file(path, teacher, batch_size=3)


def test_constant_products_are_rejected_even_with_nonconstant_heads(tmp_path):
    path, teacher = saved_probabilities(tmp_path, v=[0, 0, 0, 0])
    with pytest.raises(ValueError, match='undefined for constant'):
        correlation_for_file(path, teacher, batch_size=3)


@pytest.mark.parametrize('name', ['y', 'z'])
def test_product_correlation_checks_labels_beyond_batch_boundary(tmp_path, name):
    path, teacher = saved_probabilities(tmp_path)
    table = pq.read_table(path)
    labels = table.column(name).to_numpy().copy()
    labels[3] = 1-labels[3]
    table = table.set_column(table.schema.get_field_index(name), name, pa.array(labels))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=f'{name}: alignment mismatch at row 3'):
        correlation_for_file(path, teacher, batch_size=3)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def audited_study(tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    path, teacher = saved_probabilities(project)
    teacher_path = project/'teacher.parquet'
    pq.write_table(pa.table(teacher), teacher_path)
    sources, identities = [], []
    for variant in correlation.alignment.VARIANTS:
        for seed in correlation.alignment.SEEDS:
            run = project/'runs'/variant/str(seed)
            run.mkdir(parents=True)
            selected = {'metric': 'ctcvr_ne', 'split': 'val', 'epoch': 1, 'value': .8}
            write_json(run/'run.json', {'status': 'FINISHED', 'selected_on': selected})
            (run/'config.yaml').write_text(f'model: {variant}\nseed: {seed}\n')
            cache = run/'predictions/test.parquet'
            cache.parent.mkdir()
            cache.write_bytes(path.read_bytes())
            checkpoint_hash, config_hash = f'checkpoint-{variant}-{seed}', correlation.alignment.digest(run/'config.yaml')
            cache_hash = correlation.alignment.digest(cache)
            sidecar = run/'predictions/test.provenance.json'
            write_json(sidecar, {'identity': {'checkpoint_sha256': checkpoint_hash, 'config_sha256': config_hash},
                                 'sha256': cache_hash, 'rows': 4})
            sources.append({'variant': variant, 'seed': seed, 'run_dir': str(run.relative_to(project)),
                            'selected_on': selected})
            identities.append({'variant': variant, 'seed': seed, 'run_dir': str(run), 'selected_on': selected,
                               'config_sha256': config_hash, 'checkpoint_sha256': checkpoint_hash,
                               'predictions_sha256': cache_hash, 'cache_provenance_sha256': correlation.alignment.digest(sidecar)})
    summary = project/'summary.json'
    write_json(summary, {'sources': sources})
    prior = project/'alignment.json'
    write_json(prior, {'inputs': {'summary_sha256': correlation.alignment.digest(summary),
                                  'script_sha256': correlation.alignment.digest(Path(correlation.alignment.__file__)),
                                  'teacher_predictions': str(teacher_path),
                                  'teacher_sha256': correlation.alignment.digest(teacher_path)},
                       'run_provenance': identities})
    return project, summary, prior, teacher_path


def test_filtered_report_validates_sixty_sources_but_hashes_and_computes_only_ten_caches(audited_study, monkeypatch):
    project, summary, prior, teacher = audited_study
    hashed, digest = [], correlation.alignment.digest

    def track(path):
        hashed.append(Path(path))
        return digest(path)

    monkeypatch.setattr(correlation.alignment, 'digest', track)
    result = correlation.report(summary, prior, teacher, project/'result', batch_size=3,
                                variant='kd05', project_root=project)
    assert len(result['per_seed']) == len(result['run_provenance']) == result['validation']['runs'] == 10
    assert result['validation']['source_runs'] == 60
    assert {row['variant'] for row in result['per_seed']} == {'kd05'}
    assert len(result['summary']) == 1 and result['summary'][0]['n'] == 10
    row = result['summary'][0]
    assert row['product_mean'] == pytest.approx(np.mean(row['product_values']))
    assert row['product_sem'] == pytest.approx(np.std(row['product_values'], ddof=1)/np.sqrt(10))
    assert row['product_scipy_mean'] == pytest.approx(row['product_mean'], abs=1e-14)
    cache_hashes = [path for path in hashed if path.name == 'test.parquet']
    assert len(cache_hashes) == 10 and {path.parent.parent.parent.name for path in cache_hashes} == {'kd05'}
    assert len([path for path in hashed if path.name == 'test.provenance.json']) == 60
    assert (project/'result/correlation.json').is_file()


def test_default_report_preserves_all_six_variants_and_old_summary_fields(audited_study):
    project, summary, prior, teacher = audited_study
    result = correlation.report(summary, prior, teacher, project/'result', batch_size=3, project_root=project)
    assert result['validation']['runs'] == result['validation']['source_runs'] == 60
    assert len(result['summary']) == 6
    for row in result['summary']:
        assert {'variant', 'n', 'mean', 'sem', 'scipy_mean', 'values', 'product_mean',
                'product_sem', 'product_scipy_mean', 'product_values'} <= row.keys()


def test_variant_filter_still_rejects_unselected_run_provenance_changes(audited_study):
    project, summary, prior, teacher = audited_study
    (project/'runs/kd0/51/config.yaml').write_text('modified: true\n')
    with pytest.raises(ValueError, match='configuration changed'):
        correlation.report(summary, prior, teacher, project/'result', variant='esmm', project_root=project)
    assert not (project/'result').exists()


def test_unknown_variant_is_rejected(audited_study):
    project, summary, prior, teacher = audited_study
    with pytest.raises(ValueError, match='unknown variant'):
        correlation.report(summary, prior, teacher, project/'result', variant='missing', project_root=project)
