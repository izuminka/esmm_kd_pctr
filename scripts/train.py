"""Training entrypoint.

Usage:
    python scripts/train.py --config configs/dcn.yaml

Orchestrates: load config → build dataset → build model → fit.
All training logic lives in training/trainer.py.
"""

import argparse
import hashlib
import os
import pickle
import sys

import yaml

# Ensure code/ is on sys.path when the script is run directly
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import iter_impressions
from data.vocab import build_vocabs
from training.trainer import Dataset, Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _discover_field_ids(impressions: list) -> list:
    """Return sorted list of all field IDs seen in impressions."""
    fields = set()
    for imp in impressions:
        fields.update(imp.fields.keys())
    return sorted(fields)


def _load_impressions(tar_path: str, split: str, cache_dir: str) -> list:
    """Load impressions from tar, using a pickle cache keyed by file mtime.

    First call: parses the full tar and writes a cache file (~minutes).
    Subsequent calls: loads the pickle directly (~seconds).
    Set cache_dir=None to disable caching.
    """
    if cache_dir is None:
        return list(iter_impressions(tar_path, split))

    os.makedirs(cache_dir, exist_ok=True)
    mtime = str(os.path.getmtime(tar_path))
    key = hashlib.md5(f"{tar_path}:{split}:{mtime}".encode()).hexdigest()[:10]
    cache_file = os.path.join(cache_dir, f"{os.path.basename(tar_path).split('.')[0]}_{split}_{key}.pkl")

    if os.path.exists(cache_file):
        print(f"  Loading from cache: {os.path.basename(cache_file)}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    impressions = list(iter_impressions(tar_path, split))
    with open(cache_file, "wb") as f:
        pickle.dump(impressions, f)
    print(f"  Cached to: {os.path.basename(cache_file)}")
    return impressions


def build_dataset(cfg: dict, root_dir: str) -> Dataset:
    train_tar = os.path.join(root_dir, cfg["data"]["train_tar"])
    test_tar = os.path.join(root_dir, cfg["data"]["test_tar"])
    val_frac = cfg["training"]["val_frac"]

    raw_cache = cfg["data"].get("cache_dir")
    cache_dir = os.path.join(root_dir, raw_cache) if raw_cache else None

    print("Loading train impressions...")
    all_train = _load_impressions(train_tar, "train", cache_dir)
    n_train = int(len(all_train) * (1 - val_frac))
    train_imps, val_imps = all_train[:n_train], all_train[n_train:]
    print(f"  {len(all_train)} impressions  →  train={len(train_imps)}, val={len(val_imps)}")

    field_ids = cfg["data"].get("field_ids") or _discover_field_ids(train_imps)
    print(f"  {len(field_ids)} fields")

    print("Building vocabs from train split...")
    vocabs = build_vocabs(train_imps, field_ids)

    print("Loading test impressions...")
    test_imps = _load_impressions(test_tar, "test", cache_dir)
    print(f"  {len(test_imps)} test impressions")

    return Dataset(train=train_imps, val=val_imps, test=test_imps, vocabs=vocabs, field_ids=field_ids)


def build_model(cfg: dict, vocab_sizes: dict):
    params = cfg.get("model_params", {})
    if cfg["model"] == "dcn":
        from models.dcn import DCNv2
        return DCNv2(vocab_sizes, **params)
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def build_metric_fn(cfg: dict):
    if cfg["model"] == "dcn":
        from evaluation.dcn import evaluate_dcn
        return lambda model, enc, y: evaluate_dcn(model, enc, y)["ctr"]["auc"]
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def main():
    parser = argparse.ArgumentParser(description="Train a recommendation model on Ali-CCP.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/dcn.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: model={cfg['model']}  epochs={cfg['training']['epochs']}"
          f"  batch_size={cfg['training']['batch_size']}  lr={cfg['training']['lr']}")

    dataset = build_dataset(cfg, _ROOT_DIR)
    model = build_model(cfg, dataset.vocab_sizes)
    metric_fn = build_metric_fn(cfg)
    trainer = Trainer(cfg, _ROOT_DIR)
    print(f"Devices: {trainer.devices}")

    checkpoint = trainer.fit(model, dataset, metric_fn)
    print(f"\nDone. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
