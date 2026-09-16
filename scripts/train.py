"""Training entrypoint.

Usage:
    python scripts/train.py --config configs/dcn.yaml

Prerequisites:
    python scripts/preprocess.py --config configs/dcn.yaml  (run once)

Orchestrates: load config → load encoded dataset → build model → fit.
All training logic lives in training/trainer.py.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

import torch
import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from training.trainer import Trainer, resolve_device
from training.dataset import NumpyDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, vocab_sizes: dict):
    params = cfg.get("model_params", {})
    if cfg["model"] == "dcn":
        from models.dcn import DCNv2
        return DCNv2(vocab_sizes, **params)
    if cfg["model"] == "esmm":
        from models.esmm import ESMM
        return ESMM(vocab_sizes, **params)
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def build_metric_fn(cfg: dict):
    if cfg["model"] == "dcn":
        from evaluation.dcn import evaluate_dcn
        return lambda model, enc, y: evaluate_dcn(model, enc, y)["ctr"]["auc"]
    if cfg["model"] == "esmm":
        from evaluation.esmm import evaluate_esmm
        return lambda model, enc, y, z: evaluate_esmm(model, enc, y, z)["ctcvr"]["pr_auc"]
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def main():
    parser = argparse.ArgumentParser(description="Train a recommendation model on Ali-CCP.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/dcn.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(
        "Config: model=%s  epochs=%d  batch_size=%d  lr=%s",
        cfg["model"], cfg["training"]["epochs"], cfg["training"]["batch_size"], cfg["training"]["lr"],
    )

    encoded_dir = os.path.join(_ROOT_DIR, cfg["data"]["encoded_dir"])
    if not os.path.exists(encoded_dir):
        raise RuntimeError(
            f"Encoded data not found at {encoded_dir}. "
            "Run: python scripts/preprocess.py --config <config>"
        )

    device = resolve_device(cfg["training"].get("gpus", -1))
    logger.info("Loading encoded dataset from %s onto %s...", encoded_dir, device)
    t0 = time.perf_counter()
    load_test = cfg["training"]["load_test"]
    dataset = NumpyDataset.load(encoded_dir, device=device, load_test=load_test)
    logger.info("Dataset ready in %.1fs  (train=%d, val=%d, test=%s)",
                time.perf_counter() - t0, len(dataset.train), len(dataset.val),
                len(dataset.test) if dataset.test is not None else "not loaded")

    model = build_model(cfg, dataset.vocab_sizes)
    metric_fn = build_metric_fn(cfg)
    trainer = Trainer(cfg, _ROOT_DIR)

    checkpoint = trainer.fit(model, dataset, metric_fn)
    logger.info("Checkpoint: %s", checkpoint)

    # Release the training state before touching the test split: together they
    # exceed one card, which is why load_test defaults to false.
    del dataset, model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not load_test:
        _run(["predict_encoded.py", "--config", args.config, "--split", "test"],
             "Exporting test predictions")
    _run(["report_metrics.py", "--config-model", cfg["model"],
          "--artifacts", os.path.join(_ROOT_DIR, cfg["training"]["out_dir"]),
          "--json-out", os.path.join(_ROOT_DIR, cfg["training"]["out_dir"],
                                     f"metrics_{cfg['model']}.json")],
         "Metrics (paper evaluation recipe)")
    logger.info("Done.")


def _run(script_args: list, banner: str) -> None:
    """Run a sibling script in its own process.

    Subprocesses rather than in-process calls: the prediction pass needs the
    device memory training just gave up, and only process exit guarantees the
    caching allocator has actually returned it.
    """
    cmd = [sys.executable, "-u", os.path.join(_CODE_DIR, "scripts", script_args[0])]
    cmd += script_args[1:]
    logger.info("=== %s ===", banner)
    result = subprocess.run(cmd, cwd=_CODE_DIR)
    if result.returncode != 0:
        logger.error("%s failed (exit %d): %s", banner, result.returncode, " ".join(cmd))
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
