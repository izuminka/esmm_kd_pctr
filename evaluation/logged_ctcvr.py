"""Optional logged-pCTR evaluation for any predictor exporting a pCVR column v."""

import json
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
import yaml

from data.logged_pctr import load_logged_pctr
from evaluation.metrics import compute_metrics
from evaluation.predictions import digest, locked, prediction_identity, evaluate_native


def evaluate_logged(run_dir, config_path, root, gpus=1, refresh=False, predictions=None):
    """Native evaluation followed by p*v; preserve checkpoint and prior source results."""
    run_dir, config_path, root = map(Path, (run_dir, config_path, root))
    cfg = yaml.safe_load(config_path.read_text())
    name = cfg.get('pctr_model')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('pctr_model must be a single safe directory name')
    block = cfg.get('logged_pctr')
    if not isinstance(block, dict) or set(cfg) != {'pctr_model', 'logged_pctr'}:
        raise ValueError('evaluation config requires only pctr_model and logged_pctr')
    if set(block.get('predictions', {})) != {'test'}:
        raise ValueError('logged evaluation requires exactly the full test split')
    run = json.loads((run_dir / 'run.json').read_text())
    if run['status'] != 'FINISHED':
        raise ValueError('evaluation requires a finished training run')
    target = run_dir / 'metrics_logged' / name
    with locked(run_dir / 'metrics_logged'):
        if target.exists() and yaml.safe_load((target / 'config.yaml').read_text()) != cfg:
            raise ValueError('pctr_model already names a different source; use a distinct name')
        frame, cache = predictions if predictions is not None else evaluate_native(run_dir, root, gpus, refresh)
        if 'v' not in frame:
            raise ValueError('logged CTCVR requires a pCVR prediction column named v')
        source_path = root / block['predictions']['test']
        source_hash = digest(source_path)
        p, source = load_logged_pctr(block, str(root), 'test',
                                    torch.tensor(frame.sample_id.to_numpy()),
                                    torch.tensor(frame.y.to_numpy()))
        values = frame.v.to_numpy()
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('pCVR scores must be finite probabilities')
        if not bool(torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
            raise ValueError('logged pCTR scores must be finite probabilities')
        provenance = {'checkpoint_sha256': cache['identity']['checkpoint_sha256'],
                      'predictions_sha256': cache['sha256'], 'pctr_scores_sha256': source_hash,
                      'pctr_model': name, 'logged_pctr': source}
        if target.exists():
            if json.loads((target / 'provenance.json').read_text()) != provenance:
                raise ValueError('evaluation inputs changed; use a distinct pctr_model name')
        labels = frame.y.to_numpy() * frame.z.to_numpy()
        row = {'split': 'test', 'population': 'CTCVR-logged', 'pctr_model': name,
               'rows': len(frame), 'positives': int(labels.sum()),
               'logged_pctr_run': source['dcn_run'], 'logged_pctr_file': source['files']['test']['file'],
               **compute_metrics(p.numpy() * values, labels)}
        if (digest(source_path) != source_hash
                or prediction_identity(run_dir, root) != cache['identity']):
            raise ValueError('evaluation inputs changed while scoring')
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                staged = Path(tmp) / name
                staged.mkdir()
                (staged / 'config.yaml').write_text(config_path.read_text())
                (staged / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
                (staged / 'test.json').write_text(json.dumps([row], indent=2) + '\n')
                staged.rename(target)
        return target / 'test.json'


def logged_metric_rows(run_dir):
    """Read separately evaluated sources for an explicitly combined report."""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / 'metrics_logged').glob('*/test.json'))
    checkpoint = digest(run_dir / 'best.pt') if paths else None
    rows = []
    for path in paths:
        provenance = json.loads((path.parent / 'provenance.json').read_text())
        if provenance['checkpoint_sha256'] != checkpoint:
            raise ValueError(f'stale logged evaluation: {path}')
        rows.extend(json.loads(path.read_text()))
    return rows
