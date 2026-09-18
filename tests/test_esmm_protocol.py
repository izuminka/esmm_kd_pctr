"""Matched ESMM protocols, reproducible initialization, and checkpoint restoration."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train import build_model, build_metric_fn, load_config
from scripts.report_metrics import logged_ctcvr_splits, metric_rows
from training.dataset import NumpyDataset
from training.trainer import Trainer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_matched_cohort_arms_differ_only_in_training_cohort():
    full = load_config(CONFIGS / "esmm.yaml")
    for name, fraction in (("esmm_half", .5), ("esmm_third", "1/3"),
                           ("esmm_quarter", .25)):
        cfg = load_config(CONFIGS / f"{name}.yaml")
        assert cfg["data"].pop("train_cohort")["fraction"] == fraction
        if name == "esmm_quarter":
            # Historical quarter config retains its original test-only teacher.
            # The current full/half/third study shares the newly frozen DCN.
            assert {k: v for k, v in cfg.items() if k != "logged_pctr"} == {
                k: v for k, v in full.items() if k != "logged_pctr"}
        else:
            assert cfg == full
    assert full["training"]["checkpoint_metric"] == "ctcvr_ne"
    assert full["training"]["keep_checkpoint"] == "best"
    assert full["training"]["early_stopping"] == {"enabled": False, "patience": 2, "min_delta": 0.0}
    assert set(full["logged_pctr"]["predictions"]) == {"test"}


@pytest.mark.parametrize("config", sorted(CONFIGS.glob("esmm*.yaml")), ids=lambda p: p.stem)
def test_native_and_logged_configs_use_their_prescribed_validation_score(config):
    cfg = load_config(config)
    logged = cfg["model"] == "esmm_logged_pctr"
    native_metric = "ctcvr_ne"
    if config.stem.startswith("esmm_ctr_reference_"):
        native_metric = "ctr_ne"
        assert cfg["training"]["evaluate_test"] is False
        assert cfg["training"]["load_test"] is False
    kd_ne = config.stem.startswith(("esmm_logged_pctr_kd_teacher", "esmm_logged_pctr_kd05", "esmm_logged_pctr_kd03", "esmm_logged_pctr_kd07")) and "2x_aux" not in config.stem
    logged_metric = "logged_ctcvr_ne" if kd_ne else "logged_ctcvr_prauc"
    assert cfg["training"]["checkpoint_metric"] == (logged_metric if logged else native_metric)
    assert ("val" in logged_ctcvr_splits(cfg)) == logged
    assert "train" not in logged_ctcvr_splits(cfg)


def tiny_cfg(tmp_path):
    cfg = load_config(CONFIGS / "esmm.yaml")
    cfg["training"].update(gpus=0, batch_size=3, eval_batch_size=3, out_dir=str(tmp_path))
    cfg["model_params"] = {"emb_dim": 2, "tower_dims": [2, 4, 2]}
    return cfg


def test_model_initialization_uses_configured_seed(tmp_path):
    cfg = tiny_cfg(tmp_path)
    first = build_model(cfg, {"f": 8}).state_dict()
    torch.rand(200)  # unrelated RNG consumption cannot change the next run
    second = build_model(cfg, {"f": 8}).state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    cfg["training"]["seed"] += 1
    third = build_model(cfg, {"f": 8}).state_dict()
    assert any(not torch.equal(first[k], third[k]) for k in first)


class TinySplit:
    device = torch.device("cpu")

    def __init__(self):
        self.sample_ids = torch.arange(8)
        self.ys = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int8)
        self.zs = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.int8)
        self.logged_pctr = None

    def __len__(self):
        return 8

    def get_batch(self, idx):
        return {"f": (idx + 1, torch.ones(len(idx)), torch.arange(len(idx)))}, self.ys[idx].float(), self.zs[idx].float()


@pytest.mark.parametrize("metric", ["ctr_ne", "ctr_prauc", "pcvr_ne", "ctcvr_prauc", "ctcvr_ne"])
def test_legacy_and_encoded_esmm_selection_agree(tmp_path, metric):
    cfg = tiny_cfg(tmp_path)
    cfg["training"]["checkpoint_metric"] = metric
    cfg["evaluation"]["is_logged_ctcvr"] = False
    split = TinySplit()
    model = build_model(cfg, {"f": 8})
    batch, y, z = split.get_batch(torch.arange(len(split)))
    legacy = build_metric_fn(cfg)(model, batch, y, z)
    encoded, _ = Trainer(cfg, str(tmp_path))._val_score_numpy(model, split)
    assert legacy == pytest.approx(encoded, abs=1e-6)


@pytest.mark.parametrize("selection", ["logged_ctcvr_prauc", "logged_ctcvr_ne"])
@pytest.mark.parametrize("objective", [None, "kd", "product"])
def test_validation_and_export_use_only_allowed_logged_scores(tmp_path, objective, selection):
    cfg = tiny_cfg(tmp_path)
    if objective:
        cfg["model"] = "esmm_logged_pctr"
        cfg["model_params"]["objective"] = objective
        if objective == "kd":
            cfg["model_params"].update(alpha=.5, aux_weight=1.)
        cfg["training"]["checkpoint_metric"] = selection
    split = TinySplit()
    # Any accidental use of p during native validation will fail metrics.
    split.logged_pctr = torch.full((8,), float("nan"))
    if objective is not None:
        split.logged_pctr = torch.linspace(.1, .8, 8)
    model = build_model(cfg, {"f": 8})
    trainer = Trainer(cfg, str(tmp_path))
    score, extras = trainer._val_score_numpy(model, split)
    _, y, z, scores = trainer._predict_numpy(model, split, split_name="val")
    assert ("p" in scores) == (objective is not None)
    expected = {r["population"]: r for r in metric_rows("val", cfg["model"], y.astype("int8"), z.astype("int8"), scores)}
    population = "CTCVR-logged" if objective is not None else "CTCVR"
    assert score == pytest.approx(expected[population]["pr_auc" if objective and selection.endswith("prauc") else "ne"], abs=1e-6)
    assert extras["val_pcvr_ne"] == pytest.approx(expected["pCVR"]["ne"])
    assert any("logged" in c for c in extras) == (objective is not None)
    _, _, _, train_scores = trainer._predict_numpy(model, split, split_name="train")
    assert "p" not in train_scores
    split.logged_pctr = torch.linspace(.1, .8, 8)
    _, _, _, test_scores = trainer._predict_numpy(model, split, split_name="test")
    assert np.array_equal(test_scores["p"], split.logged_pctr.numpy())


def test_native_esmm_cannot_select_on_logged_product(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg["training"]["checkpoint_metric"] = "logged_ctcvr_prauc"
    with pytest.raises(ValueError, match="native ESMM validates on"):
        Trainer(cfg, str(tmp_path))._selection_inputs({}, torch.ones(1), torch.ones(1))


def test_native_fit_with_loaded_test_needs_only_test_logged_scores(tmp_path):
    import json

    cfg = tiny_cfg(tmp_path)
    cfg["training"]["epochs"] = 1
    train, val, test = TinySplit(), TinySplit(), TinySplit()
    test.logged_pctr = torch.linspace(.1, .8, 8)
    data = NumpyDataset(train, val, test, {"f": {str(i): i for i in range(1, 9)}}, ["f"])
    trainer = Trainer(cfg, str(tmp_path))
    trainer.fit(build_model(cfg, data.vocab_sizes), data, None)
    for split in ("train", "val", "test"):
        rows = json.loads((tmp_path / "metrics" / f"{split}.json").read_text())
        assert any(r["population"] == "CTCVR-logged" for r in rows) == (split == "test")


@pytest.mark.parametrize("metric", ["ctcvr_prauc", "pcvr_ne", "ctr_ne", "ctcvr_ne"])
@pytest.mark.parametrize("keep,enabled,expected_epochs,expected_kept", [
    ("best", True, 4, 2), ("best", False, 5, 2),
    ("last", True, 5, 5), ("last", False, 5, 5), ("last", None, 5, 5),
])
def test_stopping_restores_the_selected_weights(tmp_path, monkeypatch, keep, enabled, expected_epochs, expected_kept, metric):
    cfg = tiny_cfg(tmp_path)
    # Exercise stopping and restoration independently of the default run budget.
    cfg["training"]["epochs"] = 5
    cfg["training"]["checkpoint_metric"] = metric
    cfg["training"]["keep_checkpoint"] = keep
    if enabled is None:
        cfg["training"]["early_stopping"].pop("enabled", None)
    else:
        cfg["training"]["early_stopping"]["enabled"] = enabled
    cfg["evaluation"]["is_logged_ctcvr"] = False
    data = NumpyDataset(TinySplit(), TinySplit(), None, {"f": {str(i): i for i in range(1, 9)}}, ["f"])
    model = build_model(cfg, data.vocab_sizes)
    trainer = Trainer(cfg, str(tmp_path))
    scores = iter([1.1, .8, .9, 1.0, 1.2] if metric.endswith("_ne")
                  else [.1, .2, .15, .14, .13])

    def train_epoch(model, split, optimizer, epoch):
        with torch.no_grad():
            next(model.parameters()).fill_(epoch + 1)
        return .1

    monkeypatch.setattr(trainer, "_train_epoch_numpy", train_epoch)
    monkeypatch.setattr(trainer, "_val_score_numpy", lambda *args: (next(scores), {}))
    path = trainer.fit(model, data, None)
    checkpoint = torch.load(path, weights_only=False)
    assert next(iter(checkpoint["model_state_dict"].values())).flatten()[0] == expected_kept
    assert next(model.parameters()).flatten()[0] == expected_kept
    with (tmp_path / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected_epochs
    assert int([r for r in rows if r["saved"] == "True"][-1]["epoch"]) == expected_kept
