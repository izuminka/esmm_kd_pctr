"""Test-only inspection of canonical files, independent of the report loader."""
import io
import json
from pathlib import Path


def canonical_rows(run_dir, include_logged=False):
    run_dir = Path(run_dir)
    paths = [run_dir / 'metrics' / f'{split}.json' for split in ('train', 'val', 'test')]
    if include_logged:
        paths += sorted((run_dir / 'metrics_logged').glob('*/test.json'))
    return [row for path in paths if path.exists() for row in json.loads(path.read_text())]


def metric_report(run_dir):
    return io.StringIO(json.dumps(canonical_rows(run_dir)))
