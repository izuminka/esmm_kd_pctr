"""Export predictions from a saved checkpoint to Parquet.

Usage:
    python scripts/predict.py --checkpoint artifacts/dcn/20260916T020356Z/best.pt --split train
    python scripts/predict.py --checkpoint artifacts/dcn/20260916T020356Z/best.pt --split test \\
        --out /tmp/predictions_test.parquet
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
from training.trainer import Trainer, save_predictions, save_esmm_predictions
from training import downsampling
from training.downsampling import NONCLICK_SAMPLING_KEY


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
    parser = argparse.ArgumentParser(description="Export model predictions to Parquet.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None,
                        help="Output .parquet path (default: <checkpoint dir>/predictions/<split>.parquet)")
    parser.add_argument("--data-dir", default=os.path.join(_ROOT_DIR, "ali-ccp"),
                        help="Directory containing sample_{train,test}.tar.gz")
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), "predictions", f"{args.split}.parquet"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    model, ckpt = _load_checkpoint(args.checkpoint)

    tar_split = "train" if args.split in ("train", "val") else "test"
    tar_path = os.path.join(args.data_dir, f"sample_{tar_split}.tar.gz")
    print(f"Loading {args.split} impressions from {tar_path}...")
    all_imps = list(iter_impressions(tar_path, tar_split))
    imps = _select_split(all_imps, args.split)
    print(f"  {len(imps)} impressions in split {args.split!r}")

    cfg = {"model": ckpt["model_type"],
           "data": {NONCLICK_SAMPLING_KEY: ckpt.get(NONCLICK_SAMPLING_KEY)},
           "evaluation": {"is_logged_ctcvr": False},
           "training": {"epochs": 1, "lr": 0.001, "shuffle_block_size": 1,
                        "batch_size": 4096, "eval_batch_size": 4096,
                        "gpus": -1, "out_dir": ".", "grad_clip_norm": 0}}
    trainer = Trainer(cfg, _ROOT_DIR)
    model = model.to(trainer.device)
    if ckpt["model_type"] == "esmm":
        ids, ys, zs, qs, vs = trainer.predict_esmm(model, imps, ckpt["vocabs"], ckpt["field_ids"],
                                              is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space[args.split])
        save_esmm_predictions(ids, ys, zs, qs, vs, out_path)
    else:
        ids, ys, zs, scores = trainer.predict(model, imps, ckpt["vocabs"], ckpt["field_ids"])
        save_predictions(ids, ys, zs, scores, out_path)
    print(f"Saved {len(ids)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
