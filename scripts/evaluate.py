"""Evaluate a saved checkpoint on a data split.

Usage:
    python scripts/evaluate.py --checkpoint artifacts/dcn_model.pt --split val
    python scripts/evaluate.py --checkpoint artifacts/dcn_model.pt --split test \\
        --data-dir /path/to/ali-cpp
"""

import argparse
import os
import sys

import torch

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import iter_impressions
from data.vocab import encode_batch


def _load_checkpoint(path: str):
    ckpt = torch.load(path, weights_only=False)
    model_type = ckpt["model_type"]
    if model_type == "dcn":
        from models.dcn import DCNv2
        model = DCNv2(ckpt["vocab_sizes"])
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type!r}")
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def _select_split(imps: list, split: str, val_frac: float) -> list:
    n_train = int(len(imps) * (1 - val_frac))
    if split == "train":
        return imps[:n_train]
    if split == "val":
        return imps[n_train:]
    return imps  # "test" already the full test set


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=os.path.join(_ROOT_DIR, "ali-cpp"),
                        help="Directory containing sample_{train,test}.tar.gz")
    args = parser.parse_args()

    model, ckpt = _load_checkpoint(args.checkpoint)
    model.eval()
    val_frac = ckpt["val_frac"]

    tar_split = "train" if args.split in ("train", "val") else "test"
    tar_path = os.path.join(args.data_dir, f"sample_{tar_split}.tar.gz")
    print(f"Loading {args.split} impressions from {tar_path}...")
    all_imps = list(iter_impressions(tar_path, tar_split))
    imps = _select_split(all_imps, args.split, val_frac)
    print(f"  {len(imps)} impressions (val_frac={val_frac})")

    encoded = encode_batch(imps, ckpt["field_ids"], ckpt["vocabs"])
    y = torch.tensor([imp.y for imp in imps], dtype=torch.float32)

    if ckpt["model_type"] == "dcn":
        from evaluation.dcn import evaluate_dcn
        results = evaluate_dcn(model, encoded, y)
    else:
        raise ValueError(f"Unknown model_type: {ckpt['model_type']!r}")

    for population, metrics in results.items():
        row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{population}:  {row}")


if __name__ == "__main__":
    main()
