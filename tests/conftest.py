"""Shared fixture: a small encoded dataset carved from the real Ali-CCP CSVs.

Real rows rather than synthetic ones: the delimiter nesting, the skeleton/common
split and the ragged field layout are exactly what a synthetic fixture would
paper over. Session-scoped because encoding it runs preprocess.py end to end.
"""

import os
import subprocess
import sys

import pytest
import yaml

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(CODE_DIR)
CSV_DIR = os.path.join(ROOT_DIR, "ali-ccp", "extracted")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

_N_BAGS = 4000         # common_features rows to copy into the fixture
_N_ROWS = 400          # skeleton rows (that reference those bags) to keep
_SCAN_LIMIT = 400_000  # skeleton lines to scan looking for matches


def carve_split(csv_dir: str, split: str, out_dir: str) -> int:
    """Copy a small but self-consistent slice of one split into out_dir.

    Head-of-file slices of the two files barely intersect (the skeleton
    references bags scattered through common_features), so take the bags first
    and then keep only skeleton rows pointing at them.
    """
    cf_src = os.path.join(csv_dir, f"common_features_{split}.csv")
    sk_src = os.path.join(csv_dir, f"sample_skeleton_{split}.csv")

    kept_idx = set()
    with open(cf_src, encoding="utf-8") as fin, \
         open(os.path.join(out_dir, f"common_features_{split}.csv"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= _N_BAGS:
                break
            kept_idx.add(line.split(",", 1)[0])
            fout.write(line)

    n = 0
    with open(sk_src, encoding="utf-8") as fin, \
         open(os.path.join(out_dir, f"sample_skeleton_{split}.csv"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if n >= _N_ROWS or i >= _SCAN_LIMIT:
                break
            parts = line.split(",", 5)
            if len(parts) == 6 and parts[3] in kept_idx and int(parts[2]) <= int(parts[1]):
                fout.write(line)
                n += 1
    return n


@pytest.fixture(scope="session")
def encoded(tmp_path_factory):
    """Run the real preprocess.py over a fixture carved from the real CSVs.

    Returns (encoded_dir, csv_dir, n_train_rows_carved).
    """
    if not os.path.exists(os.path.join(CSV_DIR, "common_features_train.csv")):
        pytest.skip("extracted Ali-CCP CSVs not present")

    base = tmp_path_factory.mktemp("aliccp")
    csv_dir = base / "csv"
    csv_dir.mkdir()
    enc_dir = base / "encoded"

    n_train = carve_split(CSV_DIR, "train", str(csv_dir))
    n_test = carve_split(CSV_DIR, "test", str(csv_dir))
    if n_train < 50 or n_test < 10:
        pytest.skip(f"too few joinable rows in fixture (train={n_train}, test={n_test})")

    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": "dcn",
        "training": {"epochs": 1, "batch_size": 64,
                     "eval_batch_size": 128, "lr": 1e-3, "grad_clip_norm": 0,
                     "shuffle_block_size": 128,
                     "gpus": 0, "out_dir": str(base / "artifacts")},
        "data": {"csv_dir": str(csv_dir), "encoded_dir": str(enc_dir)},
    }))

    subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "scripts", "preprocess.py"),
         "--config", str(cfg_path)],
        check=True, cwd=CODE_DIR, capture_output=True, text=True, timeout=900,
    )
    return str(enc_dir), str(csv_dir), n_train
