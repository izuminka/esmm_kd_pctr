"""Smoke test for the ESMM training pipeline.

Runs 1 epoch on a tiny slice of real Ali-CCP data and verifies:
- checkpoint saved with correct keys and model_type
- all three prediction parquets exist with correct schema and valid score range
- train/val sizes match the configured 80/20 split
"""

import os

import pandas as pd
import pytest
import torch

from data.raw import iter_impressions
from data.vocab import build_vocabs
from models.esmm import ESMM
from training.trainer import Dataset, Trainer, save_esmm_predictions
from evaluation.esmm import evaluate_esmm

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ali-ccp")
_TRAIN_TAR = os.path.join(_DATA_DIR, "sample_train.tar.gz")
_TEST_TAR = os.path.join(_DATA_DIR, "sample_test.tar.gz")


def _make_cfg(out_dir: str, epochs: int = 1) -> dict:
    return {
        "model": "esmm",
        "training": {
            "epochs": epochs,
            "batch_size": 256,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "grad_clip_norm": 0, "shuffle_block_size": 128,
            "seed": 42,
            "gpus": 0,    # CPU for tests (no GPU dependency)
            "out_dir": str(out_dir),
        },
        "evaluation": {"is_logged_ctcvr": False},  # this Impression path carries no logged pCTR
    }


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("artifacts"))
    cfg = _make_cfg(out)

    all_train = list(iter_impressions(_TRAIN_TAR, "train", limit=200, common_features_limit=200_000))
    n_train = int(len(all_train) * 0.8)
    train_imps, val_imps = all_train[:n_train], all_train[n_train:]
    field_ids = sorted({f for imp in train_imps for f in imp.fields})
    vocabs = build_vocabs(train_imps, field_ids)
    test_imps = list(iter_impressions(_TEST_TAR, "test", limit=50, common_features_limit=200_000))

    dataset = Dataset(train=train_imps, val=val_imps, test=test_imps, vocabs=vocabs, field_ids=field_ids)
    model = ESMM(dataset.vocab_sizes)
    metric_fn = lambda m, enc, y: evaluate_esmm(m, enc, y, torch.zeros_like(y), is_logged_ctcvr=False)["ctr"]["auc"]

    trainer = Trainer(cfg, root_dir=out)
    trainer.out_dir = out
    trainer.fit(model, dataset, metric_fn)
    return out


def test_checkpoint_keys(artifacts):
    ckpt = torch.load(os.path.join(artifacts, "best.pt"), weights_only=False)
    assert set(ckpt.keys()) == {"model_type", "model_state_dict", "vocabs", "field_ids", "vocab_sizes", "split_rule", "model_params"}
    assert ckpt["model_type"] == "esmm"
    assert len(ckpt["field_ids"]) > 0
    assert set(ckpt["vocab_sizes"].keys()) == set(ckpt["field_ids"])
    assert "emb_dim" in ckpt["model_params"]
    assert "tower_dims" in ckpt["model_params"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_predictions_schema_and_range(artifacts, split):
    path = os.path.join(artifacts, "predictions", f"{split}.parquet")
    assert os.path.exists(path), f"missing predictions/{split}.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["sample_id", "y", "z", "q", "v"]
    assert df["q"].dtype == "float32"
    assert df["v"].dtype == "float32"
    assert df["q"].between(0.0, 1.0).all(), "q scores must be valid probabilities"
    assert df["v"].between(0.0, 1.0).all(), "v scores must be valid probabilities"
    assert len(df) > 0


def test_train_val_sizes_match_80_20_split(artifacts):
    n_train = len(pd.read_parquet(os.path.join(artifacts, "predictions", "train.parquet")))
    n_val = len(pd.read_parquet(os.path.join(artifacts, "predictions", "val.parquet")))
    total = n_train + n_val
    assert abs(n_train / total - 0.8) < 0.05
