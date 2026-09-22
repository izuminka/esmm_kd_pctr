"""Export predictions for one encoded split, or only score it, in its own process.

Usage:
    python scripts/predict_encoded.py --config configs/dcn.yaml --split test
    python scripts/predict_encoded.py --config configs/dcn.yaml --split val \
        --run-dir artifacts/dcn/20260916T020356Z
    python scripts/predict_encoded.py --config configs/esmm.yaml --split test --metrics-only

Reads <run>/best.pt and writes <run>/predictions/<split>.parquet, for the
model's latest run unless --run-dir names one. --metrics-only writes the split's
metric rows to <run>/metrics/<split>.json instead of its predictions.

Training holds train+val and the optimizer state on the device, which leaves no
room for the test split alongside them. Exporting it from a separate process is
simpler than releasing and reloading device memory mid-run, and guarantees the
training allocations are actually gone rather than merely freed to a cache.
"""

import argparse
import logging
import os
import sys
import time

import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import torch  # noqa: E402

from data.logged_pctr import load_logged_pctr  # noqa: E402
from training import cohorts, downsampling  # noqa: E402
from scripts.report_metrics import logged_ctcvr_splits  # noqa: E402
from training.dataset import CommonTable, SplitData  # noqa: E402
from training.runs import record_logged_pctr, resolve_run, run_name  # noqa: E402
from training.trainer import Trainer, resolve_device, save_scores, write_split  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("predict_encoded")

# train and val share the train source's bags; test has its own.
_COMMON_FOR = {"train": "common_train", "val": "common_train", "test": "common_test"}


def build_model(model_type: str, vocab_sizes: dict, params: dict):
    if model_type == "dcn":
        from models.dcn import DCNv2
        return DCNv2(vocab_sizes, **params)
    if model_type == "esmm":
        from models.esmm import ESMM
        return ESMM(vocab_sizes, **params)
    if model_type == "esmm_logged_pctr":
        from models.esmm_logged_pctr import LoggedPctrESMM
        return LoggedPctrESMM(vocab_sizes, **params)
    if model_type == "cnn_attention":
        from models.cnn_attention import CNNAttention
        return CNNAttention(vocab_sizes, **params)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def main():
    ap = argparse.ArgumentParser(description="Export predictions for one encoded split.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--run-dir", default=None,
                    help="defaults to --checkpoint's folder, else the model's latest run")
    ap.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    ap.add_argument("--out", default=None, help="defaults to <run>/predictions/<split>.parquet")
    ap.add_argument("--metrics-only", action="store_true",
                    help="write <run>/metrics/<split>.json instead of predictions")
    args = ap.parse_args()
    if args.metrics_only and args.out:
        ap.error("--out names a predictions file; --metrics-only writes none")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    encoded_dir = os.path.join(_ROOT_DIR, cfg["data"]["encoded_dir"])
    artifacts_root = os.path.join(_ROOT_DIR, cfg["training"]["out_dir"])
    if args.run_dir:
        run_dir = args.run_dir
    elif args.checkpoint:
        # The checkpoint's own folder, so its architecture is read from the config it
        # was trained with -- not from whichever run happens to be latest.
        run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    else:
        run_dir = resolve_run(artifacts_root, run_name(args.config))
    ckpt_path = args.checkpoint or os.path.join(run_dir, "best.pt")

    device = resolve_device(cfg["training"].get("gpus", -1))
    ckpt = torch.load(ckpt_path, weights_only=False)

    logger.info("Loading %s split onto %s...", args.split, device)
    t0 = time.perf_counter()
    common = CommonTable(os.path.join(encoded_dir, _COMMON_FOR[args.split]),
                         ckpt["field_ids"], device)
    split = SplitData(os.path.join(encoded_dir, args.split),
                      ckpt["field_ids"], common, device)
    logger.info("  %d rows, %d common bags (%.1fs)", len(split), len(common),
                time.perf_counter() - t0)

    # Architecture and logged pCTR source come from the run's own config, not
    # today's: the checkpoint doesn't carry every constructor argument, configs/
    # may have moved on, and this split must be scored with the same DCN run the
    # model was trained and selected with.
    run_config = os.path.join(run_dir, "config.yaml")
    if os.path.exists(run_config):
        with open(run_config) as f:
            run_cfg = yaml.safe_load(f)
    else:
        run_cfg = cfg
    params = (ckpt["model_params"] if ckpt["model_type"] == "cnn_attention"
              else run_cfg.get("model_params", {}))
    model = build_model(ckpt["model_type"], ckpt["vocab_sizes"], params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    is_logged_ctcvr = "logged_pctr" in run_cfg and args.split in logged_ctcvr_splits(run_cfg)
    if is_logged_ctcvr:
        split.logged_pctr, source = load_logged_pctr(run_cfg["logged_pctr"], _ROOT_DIR, args.split,
                                                     split.sample_ids, split.ys)
        if os.path.exists(os.path.join(run_dir, "run.json")):
            record_logged_pctr(run_dir, source)

    # The cohort restriction too: a run trained on a cohort must export the rows it
    # trained on, not the whole split. Applied after the logged pCTR, which is
    # attached to the full split and read through the cohort.
    if args.split == "train" and "train_cohort" in run_cfg.get("data", {}):
        split = cohorts.restrict(split, run_cfg["data"]["train_cohort"])
        logger.info("  train cohort: %d rows", len(split))

    # Saved settings own the correction, even if today's config has changed.
    sampling = ckpt.get(downsampling.NONCLICK_SAMPLING_KEY,
                        run_cfg.get("data", {}).get(downsampling.NONCLICK_SAMPLING_KEY))
    if args.split == "train" and sampling is not None:
        split = downsampling.restrict(split, sampling)
        logger.info("  retained training rows: %d", len(split))

    t0 = time.perf_counter()
    # The run's model and its own logged-CTCVR setting decide the score columns,
    # so an older run without logged pCTR exports exactly what it did before.
    trainer_cfg = {**cfg, "model": ckpt["model_type"], "model_params": run_cfg.get("model_params", {}),
                   "data": {downsampling.NONCLICK_SAMPLING_KEY: sampling},
                   "evaluation": {"is_logged_ctcvr": is_logged_ctcvr}}
    ids, ys, zs, scores = Trainer(trainer_cfg, _ROOT_DIR)._predict_numpy(model, split, split_name=args.split)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_scores(ids, ys, zs, scores, args.out)
        out_path = args.out
    else:
        out_path = write_split(run_dir, args.split, ckpt["model_type"], ids, ys, zs, scores,
                               save_predictions=not args.metrics_only)
    logger.info("%s: %d rows -> %s (%.1fs)", args.split, len(ids), out_path,
                time.perf_counter() - t0)
    if torch.cuda.is_available():
        logger.info("peak VRAM %.2f GB", torch.cuda.max_memory_allocated(device) / 1024**3)


if __name__ == "__main__":
    main()
