"""Read canonical split reports; legacy aggregates are read-only fallbacks."""

import json
from pathlib import Path


def read_metric_rows(run_dir, include_logged=False):
    run_dir = Path(run_dir)
    paths = [run_dir / 'metrics' / f'{split}.json' for split in ('train', 'val', 'test')]
    legacy = run_dir / 'metrics.json'
    fallback = json.loads(legacy.read_text()) if legacy.exists() and not all(p.exists() for p in paths) else []
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text()) if path.exists()
                    else [row for row in fallback if row['split'] == path.stem])
    if any((run_dir / 'metrics_logged').glob('*/test.json')):
        rows = [row for row in rows if not (row['split'] == 'test' and row['population'] == 'CTCVR-logged')]
    if include_logged:
        from evaluation.logged_ctcvr import logged_metric_rows
        rows.extend(logged_metric_rows(run_dir))
    return rows


def study_test_metrics(run_dir):
    """Legacy studies expect one row per population; never choose a teacher silently."""
    rows = [row for row in read_metric_rows(run_dir, include_logged=True) if row['split'] == 'test']
    metrics = {}
    for row in rows:
        if row['population'] in metrics:
            raise ValueError(f'{run_dir}: multiple logged producers; select a source explicitly')
        metrics[row['population']] = row
    return metrics
