"""Training entrypoint.

Usage:
    python scripts/train.py --config configs/dcn_matched_stats_wd3e6/train.yaml

Prerequisites:
    python scripts/preprocess.py --config configs/dcn_matched_stats_wd3e6/train.yaml  (run once)

Orchestrates: load config → start a run folder → load encoded dataset →
build model → fit → report train/validation → update the pointers.
Legacy configs can additionally evaluate test; active recipes use evaluate_run.py.
All training logic lives in training/trainer.py; the run folder layout and
pointer files are described in training/runs.py.
"""

import argparse
import logging
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.logged_pctr import attach_logged_pctr
from scripts.report_metrics import logged_ctcvr_splits
from training import cohorts, downsampling, runs
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
    # Seed before constructors draw embedding and tower weights. fit() alone is
    # too late to make initialization reproducible across the cohort arms.
    seed = cfg.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    params = cfg.get("model_params", {})
    if cfg["model"] == "dcn":
        from models.dcn import DCNv2
        return DCNv2(vocab_sizes, **params)
    if cfg["model"] == "esmm":
        from models.esmm import ESMM
        return ESMM(vocab_sizes, **params)
    if cfg["model"] == "esmm_logged_pctr":
        from models.esmm_logged_pctr import LoggedPctrESMM
        return LoggedPctrESMM(vocab_sizes, **params)
    if cfg["model"] == "cnn_attention":
        from models.cnn_attention import CNNAttention
        return CNNAttention(vocab_sizes, **params)
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def build_metric_fn(cfg: dict):
    if cfg["model"] == "dcn":
        from evaluation.dcn import evaluate_dcn
        metric = "ne" if cfg["training"].get("checkpoint_metric") == "ctr_ne" else "pr_auc"
        return lambda model, enc, y: evaluate_dcn(model, enc, y)["ctr"][metric]
    if cfg["model"] == "esmm":
        from evaluation.esmm import evaluate_esmm
        metric_name = cfg["training"].get("checkpoint_metric", "ctcvr_prauc")
        population, metric = {
            "pcvr_ne": ("pcvr", "ne"),
            "ctr_ne": ("ctr", "ne"),
            "ctr_prauc": ("ctr", "pr_auc"),
            "ctcvr_prauc": ("ctcvr", "pr_auc"),
            "ctcvr_ne": ("ctcvr", "ne"),
        }[metric_name]
        sampling = downsampling.from_config(cfg)
        rate = sampling["rate"] if sampling else downsampling.DEFAULT_NONCLICK_RATE
        return lambda model, enc, y, z: evaluate_esmm(
            model, enc, y, z, is_logged_ctcvr=False, nonclick_rate=rate)[population][metric]
    if cfg["model"] in ("esmm_logged_pctr", "cnn_attention"):
        return None  # encoded data path only, where the trainer scores validation itself
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def main():
    # SLURM cancellation and time limits send SIGTERM. Its default action ends the
    # process without unwinding, which would leave the run marked RUNNING; raising
    # SystemExit instead lets the handler below record it as FAILED.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    parser = argparse.ArgumentParser(description="Train a recommendation model on Ali-CCP.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/dcn_matched_stats_wd3e6/train.yaml)")
    parser.add_argument("--best-models", default=os.path.join(_CODE_DIR, "best_models.yaml"),
                        help="git-tracked record of each model's best run")
    args = parser.parse_args()
    # Prediction subprocesses run from code/, which may differ from the caller's
    # cwd. Resolve the config before passing it across that directory boundary.
    args.config = os.path.abspath(args.config)

    cfg = load_config(args.config)
    sampling = downsampling.from_config(cfg)
    name = runs.run_name(args.config)
    logger.info(
        "Config: %s  model=%s  epochs=%d  batch_size=%d  lr=%s",
        name, cfg["model"], cfg["training"]["epochs"], cfg["training"]["batch_size"], cfg["training"]["lr"],
    )
    # Logged-model training and validation use p; native ESMM only uses it for
    # optional test reporting.
    logged_splits = logged_ctcvr_splits(cfg)
    is_logged_ctcvr = bool(logged_splits)
    is_logged_pctr_used = is_logged_ctcvr or cfg["model"] == "esmm_logged_pctr"
    if is_logged_pctr_used and "logged_pctr" not in cfg:
        raise ValueError(f"{args.config}: an {cfg['model']} config needs a logged_pctr block "
                         "naming the DCN run and predictions files the logged pCTR comes from "
                         "(or evaluation.is_logged_ctcvr: false for plain ESMM)")
    if cfg["training"].get("checkpoint_metric") in ("logged_ctcvr_prauc", "logged_ctcvr_ne") and "val" not in logged_splits:
        raise ValueError("logged CTCVR checkpoint selection requires esmm_logged_pctr: "
                         "native ESMM validates on q*v")

    encoded_dir = os.path.join(_ROOT_DIR, cfg["data"]["encoded_dir"])
    if not os.path.exists(encoded_dir):
        raise RuntimeError(
            f"Encoded data not found at {encoded_dir}. "
            "Run: python scripts/preprocess.py --config <config>"
        )

    artifacts_root = os.path.join(_ROOT_DIR, cfg["training"]["out_dir"])
    run_dir = runs.start_run(artifacts_root, cfg, args.config, _CODE_DIR, encoded_dir)
    logger.info("Run folder: %s", run_dir)

    try:
        device = resolve_device(cfg["training"].get("gpus", -1))
        logger.info("Loading encoded dataset from %s onto %s...", encoded_dir, device)
        t0 = time.perf_counter()
        evaluate_test = cfg["training"].get("evaluate_test", True)
        load_test = cfg["training"]["load_test"] and evaluate_test
        dataset = NumpyDataset.load(encoded_dir, device=device, load_test=load_test)
        logger.info("Dataset ready in %.1fs  (train=%d, val=%d, test=%s)",
                    time.perf_counter() - t0, len(dataset.train), len(dataset.val),
                    len(dataset.test) if dataset.test is not None else "not loaded")

        logged_pctr_source = None
        needed_splits = {s: split for s, split in
                         {"train": dataset.train, "val": dataset.val, "test": dataset.test}.items()
                         if split is not None and
                         (s in logged_splits or (s == "train" and cfg["model"] == "esmm_logged_pctr"))}
        if needed_splits:
            logged_pctr_source = attach_logged_pctr(
                cfg["logged_pctr"], _ROOT_DIR, needed_splits)
            runs.record_logged_pctr(run_dir, logged_pctr_source)

        # After logged pCTR: its predictions file matches the full train split row for
        # row, and the cohort reads it through restricted to the kept rows.
        if "train_cohort" in cfg["data"]:
            spec = cfg["data"]["train_cohort"]
            dataset.train = cohorts.restrict(dataset.train, spec)
            census = dataset.train.census()
            runs.update_run(run_dir, train_cohort={**spec, **census})
            logger.info("Train cohort: %d of %d rows (fraction %s, per-range %s)",
                        census["n_rows"], len(dataset.train.split), spec["fraction"],
                        census["rank_rows"])

        if sampling is not None:
            original_rows = len(dataset.train)
            dataset.train = downsampling.restrict(dataset.train, sampling)
            np.save(os.path.join(run_dir, "retained_sample_ids.npy"),
                    dataset.train.sample_ids.cpu().numpy())
            counts = {k: v for k, v in dataset.train.census().items() if k != "rank_rows"}
            runs.update_run(run_dir, **{downsampling.NONCLICK_SAMPLING_KEY: {
                **sampling, **counts, "original_rows": original_rows,
                "sample_ids_file": "retained_sample_ids.npy",
                "reported_ctr": "full_population_posterior",
            }})
            logger.info("Nonclick retention %s: %d/%d train rows, %d clicks, %d conversions",
                        sampling["rate"], len(dataset.train), original_rows,
                        counts["n_clicks"], counts["n_conversions"])

        model = build_model(cfg, dataset.vocab_sizes)
        metric_fn = build_metric_fn(cfg)
        trainer = Trainer(cfg, _ROOT_DIR)
        trainer.out_dir = run_dir
        trainer.logged_pctr_source = logged_pctr_source
        checkpoint_metric = trainer.checkpoint_metric
        save_predictions = trainer.save_predictions
        evaluate_test = cfg["training"].get("evaluate_test", True)

        trainer.fit(model, dataset, metric_fn)

        # Release the training state before touching the test split: together they
        # exceed one card, which is why load_test defaults to false.
        del dataset, model, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not load_test and evaluate_test:
            if save_predictions:
                _run(["predict_encoded.py", "--config", args.config, "--split", "test",
                      "--run-dir", run_dir], "Exporting test predictions")
            else:
                _run(["predict_encoded.py", "--config", args.config, "--split", "test",
                      "--run-dir", run_dir, "--metrics-only"], "Scoring test")
        _run(["report_metrics.py", "--model", cfg["model"], "--run-dir", run_dir],
             "Metrics (paper evaluation recipe)")

        # Inside the tracked block: a failure while recording the result must mark
        # the run FAILED too, rather than leave its run.json at RUNNING forever.
        run = runs.finish_run(run_dir, runs.FINISHED,
                              selected_on=runs.selection(run_dir, checkpoint_metric),
                              test=runs.read_test_metrics(run_dir))
        runs.update_latest(artifacts_root, name, run_dir)
        if runs.update_best(args.best_models, name, runs.best_entry(run)):
            logger.info("New best %s run (%s %.6f); %s updated -- review and commit.",
                        name, checkpoint_metric, run["selected_on"]["value"],
                        args.best_models)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        # A run that already finished stays FINISHED: only the bookkeeping after it
        # failed, and the pointers it wrote must keep pointing at a finished run.
        if runs.read_run(run_dir)["status"] == runs.FINISHED:
            runs.update_run(run_dir, error=error)
        else:
            runs.finish_run(run_dir, runs.FAILED, error=error)
        raise

    logger.info("Done. %s is now latest.", run["run"])


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
