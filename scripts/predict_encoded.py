"""Export predictions for one encoded split, in its own process.

Usage:
    python scripts/predict_encoded.py --config configs/dcn.yaml --split test

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

from training.dataset import CommonTable, SplitData  # noqa: E402
from training.trainer import Trainer, resolve_device, save_predictions  # noqa: E402

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
    raise ValueError(f"predict_encoded supports dcn only, got {model_type!r}")


def main():
    ap = argparse.ArgumentParser(description="Export predictions for one encoded split.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    encoded_dir = os.path.join(_ROOT_DIR, cfg["data"]["encoded_dir"])
    out_dir = os.path.join(_ROOT_DIR, cfg["training"]["out_dir"])
    ckpt_path = args.checkpoint or os.path.join(out_dir, f"{cfg['model']}_model.pt")
    out_path = args.out or os.path.join(out_dir, f"predictions_{args.split}.parquet")

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

    model = build_model(ckpt["model_type"], ckpt["vocab_sizes"],
                        cfg.get("model_params", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    t0 = time.perf_counter()
    ids, ys, zs, scores = Trainer(cfg, _ROOT_DIR)._predict_numpy(model, split)
    save_predictions(ids, ys, zs, scores, out_path)
    logger.info("%s: %d rows -> %s (%.1fs)", args.split, len(ids), out_path,
                time.perf_counter() - t0)
    if torch.cuda.is_available():
        logger.info("peak VRAM %.2f GB", torch.cuda.max_memory_allocated(device) / 1024**3)


if __name__ == "__main__":
    main()
