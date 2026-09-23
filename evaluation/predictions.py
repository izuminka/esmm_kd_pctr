"""Shared post-training test prediction and native evaluation pipeline."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml



def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prediction_identity(run_dir, root):
    """Hash model/config; track immutable encoded inputs by size and mtime."""
    cfg = yaml.safe_load((run_dir / 'config.yaml').read_text())
    encoded = root / cfg['data']['encoded_dir']
    files = sorted(p for folder in ('test', 'common_test')
                   for p in (encoded / folder).rglob('*') if p.is_file())
    if not files or not (encoded / 'test/sample_id.npy').is_file():
        raise ValueError('encoded test inputs are missing')
    return {'version': 1, 'checkpoint_sha256': digest(run_dir / 'best.pt'),
            'config_sha256': digest(run_dir / 'config.yaml'),
            'encoded_dir': str(encoded.resolve()),
            'inputs': {str(p.relative_to(encoded)): [p.stat().st_size, p.stat().st_mtime_ns]
                       for p in files}}


def predict_test(run_dir, root, gpus):
    """Reuse native prediction, including saved downsampling correction for q."""
    from scripts.predict_encoded import build_model
    from training.dataset import CommonTable, SplitData
    from training import downsampling
    from training.downsampling import NONCLICK_SAMPLING_KEY
    from training.trainer import Trainer, resolve_device

    cfg = yaml.safe_load((run_dir / 'config.yaml').read_text())
    ckpt = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    device = resolve_device(gpus)
    encoded = root / cfg['data']['encoded_dir']
    common = CommonTable(str(encoded / 'common_test'), ckpt['field_ids'], device)
    split = SplitData(str(encoded / 'test'), ckpt['field_ids'], common, device)
    params = ckpt.get('model_params', cfg.get('model_params', {}))
    model = build_model(ckpt['model_type'], ckpt['vocab_sizes'], params).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    sampling = ckpt.get(NONCLICK_SAMPLING_KEY, cfg.get('data', {}).get(NONCLICK_SAMPLING_KEY))
    inference_cfg = {**cfg, 'model': ckpt['model_type'],
                     'data': {NONCLICK_SAMPLING_KEY: sampling},
                     'evaluation': {'is_logged_ctcvr': False},
                     'training': {**cfg['training'], 'gpus': gpus}}
    ids, y, z, scores = Trainer(inference_cfg, str(root))._predict_numpy(
        model, split, split_name='test', is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space['test'])
    return pd.DataFrame({'sample_id': ids, 'y': y.astype('int8'), 'z': z.astype('int8'),
                         **{k: v.astype('float32') for k, v in scores.items()}})


def test_predictions(run_dir, root, gpus=1, refresh=False):
    """Reuse predictions/test.parquet; generate only when missing or refreshed.

    Old exports without a sidecar are adopted only after full ID/label alignment
    checks and an age check against the selected checkpoint/config. Such exports
    are trusted outputs of their finished run; hashes bind subsequent reuse.
    """
    directory = run_dir / 'predictions'
    with locked(directory):
        identity = prediction_identity(run_dir, root)
        data_path, meta_path = directory / 'test.parquet', directory / 'test.provenance.json'
        if not refresh and data_path.exists():
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta['identity'] != identity or meta['sha256'] != digest(data_path):
                    raise ValueError('stale or damaged test predictions; use --refresh-cache')
                return pd.read_parquet(data_path), meta
            # Preserve existing exports, rather than rerunning the network.
            if data_path.stat().st_mtime_ns < max((run_dir / name).stat().st_mtime_ns
                                                 for name in ('best.pt', 'config.yaml')):
                raise ValueError('test predictions predate checkpoint/config; use --refresh-cache')
            frame = pd.read_parquet(data_path)
        else:
            frame = predict_test(run_dir, root, gpus)
        required = {'sample_id', 'y', 'z'}
        cfg = yaml.safe_load((run_dir / 'config.yaml').read_text())
        if not required.issubset(frame.columns) or frame.sample_id.duplicated().any() or not len(frame):
            raise ValueError('test predictions need native heads, labels and unique sample IDs')
        encoded = root / cfg['data']['encoded_dir'] / 'test'
        for column, filename in (('sample_id', 'sample_id'), ('y', 'y'), ('z', 'z')):
            expected = np.load(encoded / f'{filename}.npy', mmap_mode='r')
            if not np.array_equal(frame[column].to_numpy(), expected):
                raise ValueError('saved test predictions do not match encoded IDs/labels')
        if prediction_identity(run_dir, root) != identity:
            raise ValueError('checkpoint or test inputs changed during prediction')
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            if refresh or not data_path.exists():
                saved = Path(tmp) / data_path.name
                frame.to_parquet(saved, index=False)
                saved.replace(data_path)
            meta = {'identity': identity, 'sha256': digest(data_path), 'rows': len(frame)}
            staged = Path(tmp) / meta_path.name
            staged.write_text(json.dumps(meta, indent=2) + '\n')
            staged.replace(meta_path)
        return frame, meta


def native_metrics(run_dir, frame, refresh=False):
    """Report existing native test metrics, or compute them once from saved scores."""
    from scripts.report_metrics import metric_rows
    path = run_dir / 'metrics/test.json'
    if path.exists() and not refresh:
        rows = [r for r in json.loads(path.read_text()) if r['population'] != 'CTCVR-logged']
    else:
        scores = {k: frame[k].to_numpy() for k in frame if k not in ('sample_id', 'y', 'z', 'p')}
        model = json.loads((run_dir / 'run.json').read_text())['model']
        rows = metric_rows('test', model, frame.y.to_numpy(), frame.z.to_numpy(), scores)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print('native test', row['population'],
              ' '.join(f'{k}={row[k]:.6f}' for k in ('auc', 'pr_auc', 'ne', 'ce')))
    return rows


def evaluate_native(run_dir, root, gpus=1, refresh=False):
    """Evaluate the selected checkpoint on test, independently of training."""
    run_dir, root = Path(run_dir), Path(root)
    if json.loads((run_dir / 'run.json').read_text())['status'] != 'FINISHED':
        raise ValueError('evaluation requires a finished training run')
    frame, provenance = test_predictions(run_dir, root, gpus, refresh)
    native_metrics(run_dir, frame, refresh)
    return frame, provenance
