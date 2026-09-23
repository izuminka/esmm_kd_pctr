"""Independent logged-score evaluations share one immutable model prediction cache."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from evaluation import logged_ctcvr as logged
from evaluation import predictions
from evaluation.metrics import compute_metrics
from scripts.report_metrics import logged_ctcvr_splits
from scripts.train import load_config
from training.runs import run_name
from test_logged_pctr import _dcn_run, IDS, YS, SCORES

CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def cached_run(tmp_path, monkeypatch):
    run = tmp_path / 'artifacts/esmm/run1'
    (run / 'metrics').mkdir(parents=True)
    (run / 'best.pt').write_bytes(b'checkpoint')
    (run / 'config.yaml').write_text(yaml.safe_dump({'model': 'esmm', 'data': {'encoded_dir': 'encoded'}}))
    (run / 'run.json').write_text(json.dumps({'model': 'esmm', 'status': 'FINISHED'}))
    for folder in ('test', 'common_test'):
        (tmp_path / 'encoded' / folder).mkdir(parents=True)
    np.save(tmp_path / 'encoded/test/sample_id.npy', IDS)
    frame = pd.DataFrame({'sample_id': IDS, 'y': YS, 'z': [1, 0, 0, 0, 0],
                          'v': np.array([.8, .2, .3, .4, .1], dtype='float32')})
    np.save(tmp_path / 'encoded/test/y.npy', frame.y.to_numpy())
    np.save(tmp_path / 'encoded/test/z.npy', frame.z.to_numpy())
    calls = []
    def predict(*args):
        calls.append(True)
        return frame.copy()
    monkeypatch.setattr(predictions, 'predict_test', predict)
    native = [{'split': 'test', 'population': 'pCVR', 'rows': 2, 'positives': 1,
               **compute_metrics(np.array([.8, .4]), np.array([1, 0]))}]
    for path in (run / 'metrics/test.json', run / 'metrics.json'):
        path.write_text(json.dumps(native))
    return run, frame, calls


def source_config(root, name='teacher1', scores=SCORES):
    block = _dcn_run(root, rows=(IDS[::-1], YS[::-1], scores[::-1]), name=name)
    block['predictions'] = {'test': block['predictions']['test']}
    path = root / f'eval_{name}.yaml'
    path.write_text(yaml.safe_dump({'pctr_model': name, 'logged_pctr': block}))
    return path


def test_many_sources_reuse_inference_and_preserve_native_results(tmp_path, cached_run):
    run, frame, calls = cached_run
    original = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    for name, p in [('teacher1', SCORES), ('teacher2', SCORES * .5)]:
        config = source_config(tmp_path, name, p)
        output = logged.evaluate_logged(run, config, tmp_path)
        row, = json.loads(output.read_text())
        expected = compute_metrics(p * frame.v.to_numpy(), frame.z.to_numpy() * YS)
        for metric, value in expected.items():
            assert row[metric] == pytest.approx(value)
        assert row['pctr_model'] == name
        assert output == run / 'metrics_logged' / name / 'test.json'
        assert logged.evaluate_logged(run, config, tmp_path) == output
    assert len(calls) == 1
    assert all((run / path).read_bytes() == content for path, content in original.items())
    assert len(logged.logged_metric_rows(run)) == 2
    done = subprocess.run([sys.executable, str(CODE / 'scripts/report_metrics.py'),
                           '--run-dir', str(run), '--include-logged'], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert len(json.loads((run / 'metrics_combined.json').read_text())) == 3
    assert (run / 'metrics.json').read_bytes() == original['metrics.json']


@pytest.mark.parametrize('change', ['checkpoint', 'inputs', 'cache'])
def test_stale_cache_is_refused(tmp_path, cached_run, change):
    run, _, calls = cached_run
    predictions.test_predictions(run, tmp_path)
    path = {'checkpoint': run / 'best.pt', 'inputs': tmp_path / 'encoded/test/sample_id.npy',
            'cache': run / 'predictions/test.parquet'}[change]
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='stale or damaged'):
        predictions.test_predictions(run, tmp_path)
    assert len(calls) == 1


def test_source_name_conflict_and_unsafe_names(tmp_path, cached_run):
    run, _, calls = cached_run
    config = source_config(tmp_path)
    logged.evaluate_logged(run, config, tmp_path)
    changed = yaml.safe_load(config.read_text())
    changed['logged_pctr']['score_column'] = 'different'
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='different source'):
        logged.evaluate_logged(run, config, tmp_path)
    changed['pctr_model'] = '../escape'
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='safe directory'):
        logged.evaluate_logged(run, config, tmp_path)
    assert len(calls) == 1


def test_existing_source_file_mutation_is_detected(tmp_path, cached_run):
    run, _, _ = cached_run
    config = source_config(tmp_path)
    output = logged.evaluate_logged(run, config, tmp_path)
    before = output.read_bytes()
    source = tmp_path / yaml.safe_load(config.read_text())['logged_pctr']['predictions']['test']
    frame = pd.read_parquet(source)
    frame['score'] *= .8
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match='inputs changed'):
        logged.evaluate_logged(run, config, tmp_path)
    assert output.read_bytes() == before


@pytest.mark.parametrize('name', ['esmm', 'esmm_half', 'dcn_matched_stats_wd3e6'])
def test_run_names_preserve_variants_and_flat_compatibility(name):
    assert run_name(f'/configs/{name}/train.yaml') == name
    assert run_name(f'/configs/{name}.yaml') == name


def test_all_live_training_configs_are_nested_and_plain_esmm_has_no_logged_info():
    for folder in (CODE / 'configs', CODE / 'configs_old', CODE / 'configs_downsampled'):
        assert not list(folder.glob('*.yaml'))
        configs = list(folder.glob('*/train.yaml'))
        assert configs
        for path in configs:
            cfg = load_config(path)
            if cfg['model'] == 'esmm':
                assert 'logged_pctr' not in cfg
                assert 'is_logged_ctcvr' not in cfg.get('evaluation', {})
                assert not logged_ctcvr_splits(cfg)
            if cfg['model'] == 'esmm_logged_pctr':
                assert cfg['logged_pctr']
                assert 'val' in logged_ctcvr_splits(cfg)


def test_real_checkpoint_builds_reusable_full_test_cache(encoded, tmp_path, monkeypatch):
    from scripts.train import build_model
    from training.dataset import NumpyDataset
    encoded_dir, _, _ = encoded
    run = tmp_path / 'esmm/run1'
    run.mkdir(parents=True)
    cfg = load_config(CODE / 'configs_downsampled/esmm/train.yaml')
    data = NumpyDataset.load(encoded_dir, device=torch.device('cpu'), load_test=True)
    cfg['data']['encoded_dir'] = str(encoded_dir)
    cfg['model_params'] = {'emb_dim': 2, 'tower_dims': [len(data.field_ids) * 2, 8, 2]}
    cfg['training'].update(gpus=0, batch_size=64, eval_batch_size=64)
    model = build_model(cfg, data.vocab_sizes).eval()
    torch.save({'model_type': 'esmm', 'model_state_dict': model.state_dict(),
                'vocab_sizes': data.vocab_sizes, 'field_ids': data.field_ids,
                'model_params': cfg['model_params']}, run / 'best.pt')
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
    cache, _ = predictions.test_predictions(run, tmp_path, gpus=0)
    assert np.array_equal(cache.sample_id, data.test.sample_ids.numpy())
    batch, _, _ = data.test.get_batch(torch.arange(len(data.test)))
    with torch.no_grad():
        _, expected = model(batch)
    np.testing.assert_allclose(cache.v, expected.numpy(), rtol=1e-6)
    monkeypatch.setattr(predictions, 'predict_test', lambda *args: pytest.fail('cache must bypass inference'))
    again, _ = predictions.test_predictions(run, tmp_path, gpus=0)
    pd.testing.assert_frame_equal(cache, again)
    (run / 'run.json').write_text(json.dumps({'model': 'esmm', 'status': 'FINISHED'}))
    before = (run / 'predictions/test.parquet').stat().st_mtime_ns
    checkpoint = logged.digest(run / 'best.pt')
    for name in ('teacher_a', 'teacher_b'):
        source = tmp_path / name
        (source / 'predictions').mkdir(parents=True)
        (source / 'run.json').write_text(json.dumps({'model': 'dcn', 'status': 'FINISHED', 'run': name}))
        scores = pd.DataFrame({'sample_id': data.test.sample_ids.numpy(),
                               'y': data.test.ys.numpy(), 'score': np.full(len(data.test), .2, dtype='float32')})
        scores.iloc[::-1].to_parquet(source / 'predictions/test.parquet', index=False)
        config = tmp_path / f'eval_{name}.yaml'
        config.write_text(yaml.safe_dump({'pctr_model': name, 'logged_pctr': {
            'dcn_run': str(source), 'predictions': {'test': str(source / 'predictions/test.parquet')},
            'score_column': 'score'}}))
        done = subprocess.run([sys.executable, str(CODE / 'scripts/evaluate_run.py'),
                               '--run-dir', str(run), '--logged-config', str(config), '--gpus', '0'],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        row, = json.loads((run / 'metrics_logged' / name / 'test.json').read_text())
        assert row['rows'] == len(data.test)
    assert (run / 'predictions/test.parquet').stat().st_mtime_ns == before
    assert logged.digest(run / 'best.pt') == checkpoint


def test_other_pcvr_architecture_uses_same_logged_addon(tmp_path, cached_run):
    run, frame, calls = cached_run
    (run / 'run.json').write_text(json.dumps({'model': 'future_pcvr', 'status': 'FINISHED'}))
    (run / 'metrics/test.json').unlink()
    config = source_config(tmp_path)
    output = logged.evaluate_logged(run, config, tmp_path)
    assert output.is_file()
    native = json.loads((run / 'metrics/test.json').read_text())
    assert [r['population'] for r in native] == ['pCVR']
    assert len(calls) == 1


def test_existing_export_is_adopted_without_inference(tmp_path, cached_run, monkeypatch):
    run, frame, calls = cached_run
    (run / 'predictions').mkdir()
    path = run / 'predictions/test.parquet'
    frame.to_parquet(path, index=False)
    original = path.read_bytes()
    monkeypatch.setattr(predictions, 'predict_test', lambda *a: pytest.fail('existing export'))
    result, _ = predictions.test_predictions(run, tmp_path)
    pd.testing.assert_frame_equal(result, frame)
    assert path.read_bytes() == original
    assert not calls


def test_ctr_only_model_evaluates_native_and_rejects_logged(tmp_path, cached_run, monkeypatch):
    run, frame, _ = cached_run
    frame = frame.rename(columns={'v': 'score'})
    (run / 'run.json').write_text(json.dumps({'model': 'dcn', 'status': 'FINISHED'}))
    (run / 'metrics/test.json').unlink()
    monkeypatch.setattr(predictions, 'predict_test', lambda *a: frame)
    predictions.evaluate_native(run, tmp_path)
    native = json.loads((run / 'metrics/test.json').read_text())
    assert [r['population'] for r in native] == ['CTR']
    with pytest.raises(ValueError, match='pCVR prediction column'):
        logged.evaluate_logged(run, source_config(tmp_path), tmp_path)


def test_active_recipes_defer_test_to_evaluation():
    assert {p.name for p in (CODE / 'configs').iterdir() if p.is_dir()} == {
        'esmm', 'dcn_matched_stats_wd3e6', 'esmm_logged_pctr_kd03',
        'esmm_logged_pctr_kd05', 'esmm_logged_pctr_kd07', 'esmm_logged_pctr_kd_teacher'}
    for folder in ('configs', 'configs_downsampled'):
        for config in (CODE / folder).glob('*/train.yaml'):
            assert load_config(config)['training']['evaluate_test'] is False


def test_logged_metrics_against_independent_math(tmp_path, cached_run):
    run, frame, _ = cached_run
    row, = json.loads(logged.evaluate_logged(run, source_config(tmp_path), tmp_path).read_text())
    scores = SCORES * frame.v.to_numpy()
    labels = YS * frame.z.to_numpy()
    pos, neg = scores[labels == 1], scores[labels == 0]
    auc = np.mean((pos[:, None] > neg).astype(float) + .5 * (pos[:, None] == neg))
    order = np.argsort(-scores)
    ranked = labels[order]
    ap = np.sum(np.cumsum(ranked) / np.arange(1, len(labels) + 1) * ranked) / labels.sum()
    loss = -np.mean(labels * np.log(scores) + (1 - labels) * np.log1p(-scores))
    rate = labels.mean()
    entropy = -rate * np.log(rate) - (1 - rate) * np.log1p(-rate)
    for key, value in {'auc': auc, 'pr_auc': ap, 'ne': loss / entropy,
                       'ce': scores.sum() / labels.sum()}.items():
        assert row[key] == pytest.approx(value, rel=1e-6)


def test_training_to_multiple_logged_evaluations_end_to_end(encoded, tmp_path):
    import csv
    import shutil
    from scripts.predict_encoded import build_model
    from training.dataset import NumpyDataset

    # Real encoded features, controlled two-class labels for meaningful smoke metrics.
    enc = tmp_path / 'encoded'
    shutil.copytree(encoded[0], enc)
    for split in ('train', 'val', 'test'):
        ids = np.load(enc / split / 'sample_id.npy')
        y = (np.arange(len(ids)) % 2 == 0).astype('int8')
        z = (np.arange(len(ids)) % 4 == 0).astype('int8')
        np.save(enc / split / 'y.npy', y)
        np.save(enc / split / 'z.npy', z)
    cfg = load_config(CODE / 'configs_downsampled/esmm/train.yaml')
    cfg['training'].update(epochs=2, gpus=0, batch_size=64, eval_batch_size=128,
                           out_dir=str(tmp_path / 'runs'), save_predictions=False)
    cfg['data']['encoded_dir'] = str(enc)
    fields = json.loads((enc / 'field_ids.json').read_text())
    cfg['model_params'] = {'emb_dim': 2, 'tower_dims': [2 * len(fields), 8, 2]}
    config = tmp_path / 'smoke.yaml'
    config.write_text(yaml.safe_dump(cfg))

    def cli(script, *args):
        done = subprocess.run([sys.executable, str(CODE / 'scripts' / script), *map(str, args)],
                              capture_output=True, text=True, timeout=180)
        with (tmp_path / 'end_to_end.log').open('a') as log:
            log.write(done.stdout + done.stderr)
        assert done.returncode == 0, done.stdout + done.stderr

    cli('train.py', '--config', config, '--best-models', tmp_path / 'best.yaml')
    checkpoint, = (tmp_path / 'runs/smoke').glob('*/best.pt')
    run = checkpoint.parent
    assert {r['split'] for r in json.loads((run / 'metrics.json').read_text())} == {'train', 'val'}
    assert not (run / 'predictions/test.parquet').exists()
    assert not (run / 'metrics/test.json').exists()
    history = list(csv.DictReader((run / 'history.csv').open()))
    selected = json.loads((run / 'run.json').read_text())['selected_on']
    assert selected['epoch'] == int(min(history, key=lambda r: float(r['val_score']))['epoch'])
    before = predictions.digest(checkpoint)
    cli('evaluate_run.py', '--run-dir', run, '--gpus', 0)
    export = run / 'predictions/test.parquet'
    saved = pd.read_parquet(export)
    mtime = export.stat().st_mtime_ns
    native = (run / 'metrics/test.json').read_bytes()
    data = NumpyDataset.load(str(enc), device=torch.device('cpu'), load_test=True)
    ckpt = torch.load(checkpoint, weights_only=False)
    model = build_model(ckpt['model_type'], ckpt['vocab_sizes'], ckpt['model_params']).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    batch, _, _ = data.test.get_batch(torch.arange(len(data.test)))
    with torch.no_grad():
        q, v = model(batch)
    rate = cfg['data']['nonclick_sampling']['rate']
    np.testing.assert_allclose(saved.q, (rate*q / (1-q+rate*q)).numpy(), rtol=1e-6)
    np.testing.assert_allclose(saved.v, v.numpy(), rtol=1e-6)
    args = []
    for name, value in [('first', .2), ('second', .4)]:
        source = tmp_path / name
        (source / 'predictions').mkdir(parents=True)
        (source / 'run.json').write_text(json.dumps({'model': 'dcn', 'status': 'FINISHED', 'run': name}))
        scores = saved[['sample_id', 'y']].assign(score=np.float32(value))
        path = source / 'predictions/test.parquet'
        scores.iloc[::-1].to_parquet(path, index=False)
        evaluation = tmp_path / f'eval_{name}.yaml'
        evaluation.write_text(yaml.safe_dump({'pctr_model': name, 'logged_pctr': {
            'dcn_run': str(source), 'predictions': {'test': str(path)}, 'score_column': 'score'}}))
        args.extend(['--logged-config', evaluation])
    cli('evaluate_run.py', '--run-dir', run, '--gpus', 0, *args)
    assert export.stat().st_mtime_ns == mtime
    assert predictions.digest(checkpoint) == before
    assert (run / 'metrics/test.json').read_bytes() == native
    combined = json.loads((run / 'metrics_combined.json').read_text())
    assert {r.get('pctr_model') for r in combined if r['population'] == 'CTCVR-logged'} == {'first', 'second'}
