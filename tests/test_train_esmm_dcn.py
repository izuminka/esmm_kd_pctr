"""The DCN tower swap must preserve the canonical ESMM training contract."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from evaluation.predictions import evaluate_native
from models import esmm
from scripts.predict_encoded import build_model as restore_model
from scripts.report_metrics import logged_ctcvr_splits
from scripts.train import build_model, build_metric_fn, load_config
from training import runs
from training.dataset import NumpyDataset
from training.trainer import Trainer

CODE = Path(__file__).resolve().parents[1]


def _config(tmp_path, n_fields=1):
    cfg = load_config(CODE / "configs/esmm_dcn/train.yaml")
    cfg["training"].update(epochs=2, batch_size=4, eval_batch_size=4,
                           gpus=0, out_dir=str(tmp_path))
    cfg["model_params"] = {
        "emb_dim": 2, "tower_dims": [2 * n_fields, 4, 2], "aux_weight": 1.0,
        "dcn_params": {"emb_dim": 2, "cross_rank": 2, "num_cross_layers": 1,
                       "deep_hidden_dims": [4], "dropout": 0.1,
                       "pooling": "mean", "bag_stats": True},
    }
    return cfg


class TinySplit:
    device = torch.device("cpu")
    logged_pctr = None

    def __init__(self):
        self.sample_ids = torch.arange(8)
        self.ys = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int8)
        self.zs = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.int8)

    def __len__(self):
        return 8

    def get_batch(self, idx):
        batch = {"f": (idx + 1, torch.ones(len(idx)), torch.arange(len(idx)))}
        return batch, self.ys[idx].float(), self.zs[idx].float()


def test_recipe_is_exactly_esmm_plus_dcn_architecture():
    cfg = load_config(CODE / "configs/esmm_dcn/train.yaml")
    baseline = load_config(CODE / "configs/esmm/train.yaml")
    reference = load_config(CODE / "configs/dcn_matched_stats_wd3e6/train.yaml")
    assert cfg["model"] == "esmm_dcn"
    assert cfg["model_params"].pop("dcn_params") == reference["model_params"]
    cfg["model"] = "esmm"
    assert cfg == baseline
    assert "dcn_optimizer" not in cfg["training"]
    assert logged_ctcvr_splits(cfg) == frozenset()


def test_whole_model_uses_vanilla_esmm_adam_once_without_extra_clipping(tmp_path):
    cfg = _config(tmp_path)
    model = build_model(cfg, {"f": 8})
    trainer = Trainer(cfg, str(tmp_path))
    optimizer = trainer._make_optimizer(model)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.001 and group["weight_decay"] == 0
    assert group["betas"] == (0.9, 0.999)
    parameters = group["params"]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    assert model.ctr_tower.embeddings is model.embeddings
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.)
    trainer._clip_dense_grads(model)
    assert all(torch.equal(p.grad, torch.full_like(p, 100.)) for p in model.parameters())


def test_production_train_step_calls_existing_esmm_loss(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg["model_params"]["aux_weight"] = 1.7
    model = build_model(cfg, {"f": 8})
    calls = []
    original = esmm.esmm_loss

    def checked_loss(y, z, q, v, aux_weight):
        value = original(y, z, q, v, aux_weight)
        calls.append((aux_weight, value.item()))
        return value

    monkeypatch.setattr(esmm, "esmm_loss", checked_loss)
    trainer = Trainer(cfg, str(tmp_path))
    value = trainer._train_step_numpy(model, TinySplit(), torch.arange(8))
    assert calls == [(1.7, value)]
    assert any(torch.any(bag.weight.grad != 0) for bag in model.embeddings.values())
    assert model.cvr_tower[-1].weight.grad is not None
    assert model.ctr_tower.head.weight.grad is not None


def test_native_selection_training_and_checkpoint_keep_sharing(tmp_path):
    cfg = _config(tmp_path)
    split = TinySplit()
    data = NumpyDataset(split, split, None, {"f": {str(i): i for i in range(1, 9)}}, ["f"])
    model = build_model(cfg, data.vocab_sizes)
    metric_fn = build_metric_fn(cfg)
    batch, y, z = split.get_batch(torch.arange(8))
    legacy_score = metric_fn(model, batch, y, z)
    trainer = Trainer(cfg, str(tmp_path))
    score, _ = trainer._val_score_numpy(model, split)
    assert score == pytest.approx(legacy_score, abs=1e-6)
    checkpoint_path = trainer.fit(model, data, metric_fn)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_type"] == "esmm_dcn"
    assert checkpoint["model_params"] == model.model_params
    assert "logged_pctr" not in checkpoint
    history = list(csv.DictReader((tmp_path / "history.csv").open()))
    for row in history:
        assert float(row["val_score"]) == pytest.approx(float(row["val_ctcvr_ne"]), abs=1e-6)
    restored = restore_model("esmm_dcn", data.vocab_sizes, checkpoint["model_params"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    assert restored.ctr_tower.embeddings is restored.embeddings
    for actual, expected in zip(restored(batch), model(batch)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _run(script, *args):
    return subprocess.run([sys.executable, str(CODE / "scripts" / script), *map(str, args)],
                          cwd=CODE, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})


def test_real_data_cli_training_and_both_restore_paths(encoded, tmp_path):
    encoded_dir, _, _ = encoded
    n_fields = len(json.loads((Path(encoded_dir) / "field_ids.json").read_text()))
    cfg = _config(tmp_path / "artifacts", n_fields)
    # The small real-data fixture has no validation conversions. AP remains
    # defined; the full-data recipe and the mixed-label test above select by NE.
    cfg["training"].update(epochs=1, batch_size=64, eval_batch_size=128,
                           checkpoint_metric="ctcvr_prauc")
    cfg["data"] = {"encoded_dir": encoded_dir}
    config_path = tmp_path / "esmm_dcn.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    done = _run("train.py", "--config", config_path, "--best-models", tmp_path / "best.yaml")
    assert done.returncode == 0, done.stderr[-5000:]
    run = Path(runs.resolve_run(str(tmp_path / "artifacts"), "esmm_dcn"))
    assert json.loads((run / "run.json").read_text())["status"] == "FINISHED"
    assert not (run / "metrics/test.json").exists()
    stale = yaml.safe_load((run / "config.yaml").read_text())
    stale["model_params"]["tower_dims"] = [999, 7, 2]
    (run / "config.yaml").write_text(yaml.safe_dump(stale))
    exported = tmp_path / "restored.parquet"
    done = _run("predict_encoded.py", "--config", config_path, "--run-dir", run,
                "--split", "test", "--out", exported)
    assert done.returncode == 0, done.stderr[-5000:]
    frame, _ = evaluate_native(run, CODE.parent, gpus=0)
    direct = pd.read_parquet(exported)
    for column in frame:
        np.testing.assert_array_equal(frame[column], direct[column])
    populations = {r["population"] for r in json.loads((run / "metrics/test.json").read_text())}
    assert populations == {"CTR", "CTCVR", "pCVR"}
