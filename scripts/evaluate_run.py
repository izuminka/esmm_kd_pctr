"""Evaluate a finished model once, with optional logged-pCTR sources."""

import argparse
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluation.predictions import evaluate_native
from evaluation.logged_ctcvr import evaluate_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--logged-config', action='append', default=[], type=Path,
                        help='optional eval_<pctr_model>.yaml; repeat for multiple sources')
    parser.add_argument('--gpus', type=int, default=1, help='inference device; 0 for CPU')
    parser.add_argument('--refresh-cache', action='store_true')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    predictions = evaluate_native(run_dir, ROOT, args.gpus, args.refresh_cache)
    for config in args.logged_config:
        print(evaluate_logged(run_dir, config.resolve(), ROOT, predictions=predictions))
    subprocess.run([sys.executable, str(CODE / 'scripts/report_metrics.py'),
                    '--run-dir', str(run_dir),
                    *(['--include-logged'] if args.logged_config else [])], check=True)


if __name__ == '__main__':
    main()
