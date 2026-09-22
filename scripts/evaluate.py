"""Evaluate a saved checkpoint on a data split.

Usage:
    python scripts/evaluate.py --checkpoint artifacts/dcn/20260916T020356Z/best.pt --split val
    python scripts/evaluate.py --checkpoint artifacts/dcn/20260916T020356Z/best.pt --split test \\
        --data-dir /path/to/ali-ccp
"""

import argparse
import os
import sys

import torch

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import is_validation, iter_impressions
from data.vocab import encode_batch
from training.downsampling import DEFAULT_NONCLICK_RATE, NONCLICK_SAMPLING_KEY


def _load_checkpoint(path: str):
    ckpt = torch.load(path, weights_only=False)
    model_type = ckpt["model_type"]
    if model_type == "dcn":
        from models.dcn import DCNv2
        model = DCNv2(ckpt["vocab_sizes"])
    elif model_type == "esmm":
        from models.esmm import ESMM
        model = ESMM(ckpt["vocab_sizes"], **ckpt.get("model_params", {}))
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type!r}")
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def _select_split(imps: list, split: str) -> list:
    """Select a split by the same per-row hash rule preprocess.py applied.

    The split is defined by example id, not by position, so slicing a fraction
    off the front would evaluate on rows the model trained on.
    """
    if split == "test":
        return imps  # the official test population is never resplit
    want_val = split == "val"
    return [imp for imp in imps if is_validation(imp.sample_id) == want_val]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=os.path.join(_ROOT_DIR, "ali-ccp"),
                        help="Directory containing sample_{train,test}.tar.gz")
    args = parser.parse_args()

    model, ckpt = _load_checkpoint(args.checkpoint)
    model.eval()

    tar_split = "train" if args.split in ("train", "val") else "test"
    tar_path = os.path.join(args.data_dir, f"sample_{tar_split}.tar.gz")
    print(f"Loading {args.split} impressions from {tar_path}...")
    all_imps = list(iter_impressions(tar_path, tar_split))
    imps = _select_split(all_imps, args.split)
    print(f"  {len(imps)} impressions in split {args.split!r}")

    encoded = encode_batch(imps, ckpt["field_ids"], ckpt["vocabs"])
    y = torch.tensor([imp.y for imp in imps], dtype=torch.float32)
    z = torch.tensor([imp.z for imp in imps], dtype=torch.float32)

    if ckpt["model_type"] == "dcn":
        from evaluation.dcn import evaluate_dcn
        results = evaluate_dcn(model, encoded, y)
    elif ckpt["model_type"] == "esmm":
        from evaluation.esmm import evaluate_esmm
        results = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False,
                                nonclick_rate=ckpt.get(NONCLICK_SAMPLING_KEY, {}).get("rate", DEFAULT_NONCLICK_RATE))
    else:
        raise ValueError(f"Unknown model_type: {ckpt['model_type']!r}")

    for population, metrics in results.items():
        row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{population}:  {row}")


if __name__ == "__main__":
    main()
