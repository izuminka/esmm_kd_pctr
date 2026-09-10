"""Training entrypoint.

Usage:
    python scripts/train.py --config configs/dcn.yaml

Orchestrates: load config → build dataset → build model → fit.
All training logic lives in training/trainer.py.
"""

import argparse
import hashlib
import logging
import os
import pickle
import sys
import time

import yaml

# Ensure code/ is on sys.path when the script is run directly
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import iter_impressions_from_csv
from data.vocab import build_vocabs
from training.trainer import Dataset, Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _discover_field_ids(impressions: list) -> list:
    fields = set()
    for imp in impressions:
        fields.update(imp.fields.keys())
    return sorted(fields)


def _load_impressions(csv_dir: str, split: str, cache_dir: str | None) -> list:
    """Load impressions from CSV dir, with optional pickle cache keyed on CSV mtime."""
    cf_path = os.path.join(csv_dir, f"common_features_{split}.csv")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        mtime = str(os.path.getmtime(cf_path))
        key = hashlib.md5(f"{cf_path}:{split}:{mtime}".encode()).hexdigest()[:10]
        cache_file = os.path.join(cache_dir, f"{split}_{key}.pkl")
        if os.path.exists(cache_file):
            logger.info("[%s] Loading from pickle cache: %s", split, os.path.basename(cache_file))
            t0 = time.perf_counter()
            with open(cache_file, "rb") as f:
                imps = pickle.load(f)
            logger.info("[%s] Cache load done in %.1fs — %d impressions", split, time.perf_counter() - t0, len(imps))
            return imps

    n_workers = os.cpu_count() or 8
    logger.info("[%s] Reading CSVs with %d workers from %s", split, n_workers, csv_dir)
    t0 = time.perf_counter()
    impressions = list(iter_impressions_from_csv(csv_dir, split, n_workers=n_workers))
    logger.info("[%s] Loaded %d impressions in %.1fs", split, len(impressions), time.perf_counter() - t0)

    if cache_dir is not None:
        logger.info("[%s] Writing pickle cache: %s", split, os.path.basename(cache_file))
        t1 = time.perf_counter()
        with open(cache_file, "wb") as f:
            pickle.dump(impressions, f)
        logger.info("[%s] Cache written in %.1fs", split, time.perf_counter() - t1)

    return impressions


def build_dataset(cfg: dict, root_dir: str) -> Dataset:
    csv_dir = os.path.join(root_dir, cfg["data"]["csv_dir"])
    val_frac = cfg["training"]["val_frac"]
    cache_dir = os.path.join(root_dir, cfg["data"]["cache_dir"]) if cfg["data"].get("cache_dir") else None

    logger.info("Loading train impressions (csv_dir=%s, cache_dir=%s)", csv_dir, cache_dir)
    all_train = _load_impressions(csv_dir, "train", cache_dir)
    n_train = int(len(all_train) * (1 - val_frac))
    train_imps, val_imps = all_train[:n_train], all_train[n_train:]
    logger.info("Split: train=%d, val=%d", len(train_imps), len(val_imps))

    field_ids = cfg["data"].get("field_ids") or _discover_field_ids(train_imps)
    logger.info("Using %d fields", len(field_ids))

    logger.info("Building vocabs from train split...")
    vocabs = build_vocabs(train_imps, field_ids)

    logger.info("Loading test impressions...")
    test_imps = _load_impressions(csv_dir, "test", cache_dir)

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
    logger.info(
        "Config: model=%s  epochs=%d  batch_size=%d  lr=%s  devices=auto",
        cfg["model"], cfg["training"]["epochs"], cfg["training"]["batch_size"], cfg["training"]["lr"],
    )

    t0 = time.perf_counter()
    dataset = build_dataset(cfg, _ROOT_DIR)
    logger.info("Dataset ready in %.1fs", time.perf_counter() - t0)

    model = build_model(cfg, dataset.vocab_sizes)
    metric_fn = build_metric_fn(cfg)
    trainer = Trainer(cfg, _ROOT_DIR)
    logger.info("Devices: %s", trainer.devices)

    checkpoint = trainer.fit(model, dataset, metric_fn)
    logger.info("Done. Checkpoint: %s", checkpoint)


if __name__ == "__main__":
    main()
