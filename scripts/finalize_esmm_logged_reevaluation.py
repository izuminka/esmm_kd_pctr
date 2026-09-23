"""Index completed inference-only ESMM logged-score results under artifacts/."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.reports import study_test_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study-dir', type=Path, required=True)
    args = ap.parse_args()
    study = args.study_dir.resolve()
    root = Path(__file__).resolve().parents[2]
    if not study.is_relative_to(root / 'artifacts'):
        ap.error('study must be inside the project artifacts directory')
    rows, arms = [], {}
    dcn_source = None
    for arm in ('esmm', 'esmm_half', 'esmm_third'):
        folder = study / arm
        run = json.loads((folder / 'run.json').read_text())
        if run.get('evaluation_status') != 'FINISHED' or not run.get('evaluation_only'):
            raise RuntimeError(f'{arm}: inference and invariance checks have not completed')
        source = run['logged_pctr']['dcn_run']
        if dcn_source is not None and source != dcn_source:
            raise RuntimeError('arms have different logged pCTR sources')
        dcn_source = source
        metrics = list(study_test_metrics(folder).values())
        assert {r['population'] for r in metrics} == {'CTR', 'pCVR', 'CTCVR', 'CTCVR-logged'}
        assert all(r['split'] == 'test' for r in metrics)
        train_rows = run.get('train_cohort', {}).get('n_rows', run['data']['train']['n_rows'])
        rows.extend(dict(r, arm=arm, train_rows=train_rows) for r in metrics)
        arms[arm] = {
            'source_run': run['evaluation_of'],
            'evaluation_dir': str(folder.relative_to(root)),
            'checkpoint_sha256': run['checkpoint_sha256'],
            'train_rows': train_rows,
        }
    dcn = root / dcn_source
    dcn_run = json.loads((dcn / 'run.json').read_text())
    dcn_metrics = json.loads((dcn / 'metrics/test.json').read_text())
    assert len(dcn_metrics) == 1 and dcn_metrics[0]['population'] == 'CTR'
    rows.append(dict(dcn_metrics[0], arm='dcn', train_rows=dcn_run['data']['train']['n_rows']))
    manifest = {
        'run': study.name, 'status': 'FINISHED', 'training_performed': False,
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'dcn_run': dcn_source, 'dcn_selected_on': dcn_run['selected_on'], 'arms': arms,
    }
    (study / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (study / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    columns = ('arm', 'population', 'split', 'train_rows', 'rows', 'positives',
               'auc', 'pr_auc', 'ne', 'ce')
    with (study / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for arm, info in arms.items():
        index = Path(info['source_run']) / 'evaluations'
        index.mkdir(exist_ok=True)
        (index / f'logged_pctr_{dcn.name}.json').write_text(json.dumps({
            **info, 'dcn_run': dcn_source,
            'study_manifest': str((study / 'manifest.json').relative_to(root)),
        }, indent=2) + '\n')
    (study.parent / 'latest.json').write_text(json.dumps({
        'run': study.name, 'finished_at': manifest['finished_at'],
        'manifest': str((study / 'manifest.json').relative_to(root)),
    }, indent=2) + '\n')
    print(study / 'metrics.csv')


if __name__ == '__main__':
    main()
