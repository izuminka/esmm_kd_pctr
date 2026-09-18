"""CNN + attention through the existing encoded training/prediction/report CLIs."""

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

from evaluation.metrics import compute_metrics
from scripts.predict_encoded import build_model as restore_model
from scripts.report_metrics import logged_ctcvr_splits, metric_rows, populations
from scripts.train import build_model
from training import runs
from training.dataset import NumpyDataset
from training.trainer import Trainer, write_split

CODE_DIR = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((CODE_DIR / "configs/cnn_attention.yaml").read_text())


def _run(script, *args):
    return subprocess.run([sys.executable, str(CODE_DIR / "scripts" / script), *map(str, args)],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})


def test_reporting_uses_direct_conversion_on_all_impressions(tmp_path):
    y, z = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    scores = {"q": np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
              "c": np.array([0.05, 0.3, 0.4, 0.15], dtype=np.float32)}
    routed = populations("cnn_attention", y, z, scores)
    assert set(routed) == {"CTR", "CTCVR"}
    np.testing.assert_array_equal(routed["CTCVR"][0], scores["c"])
    np.testing.assert_array_equal(routed["CTCVR"][1], z)
    assert logged_ctcvr_splits(_config()) == frozenset()

    ids = np.arange(len(y))
    parquet = write_split(str(tmp_path / "preds"), "test", "cnn_attention", ids, y, z, scores, True)
    metrics = write_split(str(tmp_path / "metrics"), "test", "cnn_attention", ids, y, z, scores, False)
    df = pd.read_parquet(parquet)
    rows = metric_rows("test", "cnn_attention", df.y, df.z, {k: df[k] for k in scores})
    assert rows == json.loads(Path(metrics).read_text())
    assert rows[0]["reference_auc"] == 0.661
    assert rows[0]["auc_delta"] == pytest.approx(rows[0]["auc"] - 0.661)
    for key, value in compute_metrics(scores["c"], z).items():
        assert rows[1][key] == pytest.approx(value)
    assert "reference_auc" not in metric_rows("val", "cnn_attention", y, z, scores)[0]


class _ValidationSplit:
    device = torch.device("cpu")

    def __init__(self, y=None):
        self.y = torch.tensor([0., 1., 0., 1.] if y is None else y)

    def __len__(self):
        return 4

    def get_batch(self, idx):
        return idx, self.y[idx], torch.tensor([0., 1., 0., 0.])[idx]


class _FixedModel(torch.nn.Module):
    def forward(self, idx):
        q = torch.tensor([0.1, 0.2, 0.8, 0.9])[idx]
        c = torch.tensor([0.05, 0.1, 0.15, 0.2])[idx]
        return torch.logit(q), torch.logit(c)


def test_selection_uses_full_validation_roc_auc_not_mean_batch_auc():
    cfg = _config()
    cfg["training"].update(gpus=0, eval_batch_size=2, checkpoint_metric="ctr_auc")
    score, extras = Trainer(cfg, "/tmp")._val_score_numpy(_FixedModel(), _ValidationSplit())
    # Each batch has AUC 1; full-population AUC is 3/4. PR-AUC is also different.
    assert score == pytest.approx(0.75)
    assert score == extras["val_ctr_auc"]
    assert set(extras) == {f"val_{p}_{m}" for p in ("ctr", "ctcvr")
                          for m in ("auc", "prauc", "ne", "ce")}


def test_auc_selection_requires_two_validation_classes():
    cfg = _config()
    cfg["training"].update(gpus=0, checkpoint_metric="ctr_auc")
    with pytest.raises(ValueError, match="both click classes"):
        Trainer(cfg, "/tmp")._val_score_numpy(_FixedModel(), _ValidationSplit([1., 1., 1., 1.]))


def test_default_selection_is_complete_validation_ctr_ne():
    cfg = _config()
    cfg["training"].update(gpus=0, eval_batch_size=3)
    assert cfg["training"]["checkpoint_metric"] == "ctr_ne"
    trainer = Trainer(cfg, "/tmp")
    score, extras = trainer._val_score_numpy(_FixedModel(), _ValidationSplit())
    # Normalize once by the full population's entropy, not per-batch entropy.
    expected = -(2 * np.log(0.9) + 2 * np.log(0.2)) / (4 * np.log(2))
    assert score == pytest.approx(expected)
    assert score == pytest.approx(extras["val_ctr_ne"])

    class OverconfidentModel(_FixedModel):
        def forward(self, idx):
            click, conversion = super().forward(idx)
            return click * 3, conversion

    worse, worse_extras = trainer._val_score_numpy(OverconfidentModel(), _ValidationSplit())
    assert worse_extras["val_ctr_auc"] == extras["val_ctr_auc"]
    assert worse > score  # unchanged ranking can still have worse NE


def test_cnn_rejects_conditional_cvr_selection():
    cfg = _config()
    cfg["training"].update(gpus=0, checkpoint_metric="pcvr_ne")
    with pytest.raises(ValueError, match="checkpoint_metric"):
        Trainer(cfg, "/tmp")


def test_seeded_construction_is_reproducible():
    cfg = _config()
    first = build_model(cfg, {"a": 3})
    second = build_model(cfg, {"a": 3})
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("metric,values,selected_epoch,saved", [
    ("ctr_auc", [0.75, 0.75, 0.5, 0.6], 1, [True, False, False, False]),
    ("ctr_ne", [0.95, 0.90, 0.90, 0.96], 2, [True, True, False, False]),
])
def test_selection_keeps_earlier_tie_and_restores_before_export(
    tmp_path, monkeypatch, metric, values, selected_epoch, saved,
):
    class SmallSplit(_ValidationSplit):
        sample_ids = torch.arange(4)

        def get_batch(self, idx):
            _, y, z = super().get_batch(idx)
            return {"a": (idx + 1, torch.ones(len(idx)), torch.arange(len(idx)))}, y, z

    cfg = _config()
    cfg["training"].update(epochs=len(values), gpus=0, out_dir=str(tmp_path), checkpoint_metric=metric)
    data = NumpyDataset(SmallSplit(), SmallSplit(), SmallSplit(),
                        {"a": {str(i): i for i in range(1, 5)}}, ["a"])
    model = build_model(cfg, data.vocab_sizes)
    trainer = Trainer(cfg, str(tmp_path))
    validation = iter(values)

    def fake_epoch(model, split, optimizer, epoch):
        with torch.no_grad():
            model.click_head.bias.fill_(epoch + 1)
        return 0.1

    exported_epochs = []
    predict = trainer._predict_numpy

    def record_export(model, split, *, split_name):
        exported_epochs.append(model.click_head.bias.item())
        return predict(model, split, split_name=split_name)

    monkeypatch.setattr(trainer, "_train_epoch_numpy", fake_epoch)
    monkeypatch.setattr(trainer, "_val_score_numpy", lambda *args: (next(validation), {}))
    monkeypatch.setattr(trainer, "_predict_numpy", record_export)
    checkpoint = trainer.fit(model, data, None)
    assert torch.load(checkpoint, weights_only=False)["model_state_dict"]["click_head.bias"].item() == selected_epoch
    assert exported_epochs == [selected_epoch] * 3
    assert pd.read_csv(tmp_path / "history.csv").saved.tolist() == saved


@pytest.mark.parametrize("save_predictions", [False, True])
def test_real_data_training_checkpoint_reload_and_reporting(encoded, tmp_path, save_predictions):
    enc_dir, _, _ = encoded
    cfg = _config()
    cfg["data"]["encoded_dir"] = enc_dir
    cfg["training"].update(epochs=2, batch_size=64, eval_batch_size=64, gpus=0,
                           out_dir=str(tmp_path / "artifacts"), save_predictions=save_predictions)
    config_path = tmp_path / "cnn_attention_smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    best_file = tmp_path / "best_models.yaml"
    done = _run("train.py", "--config", config_path, "--best-models", best_file)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-5000:]

    run_dir = Path(runs.resolve_run(cfg["training"]["out_dir"], config_path.stem))
    run = runs.read_run(str(run_dir))
    assert run["status"] == runs.FINISHED
    assert run["model"] == "cnn_attention" and "logged_pctr" not in run
    assert run["selected_on"]["metric"] == "ctr_ne"
    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 2
    assert run["selected_on"]["value"] == pytest.approx(history.val_ctr_ne.min())
    np.testing.assert_allclose(history.val_score, history.val_ctr_ne)
    assert not any("pcvr" in c or "logged" in c for c in history.columns)

    ckpt = torch.load(run_dir / "best.pt", weights_only=False)
    assert ckpt["model_type"] == "cnn_attention"
    assert ckpt["model_params"] == cfg["model_params"]
    model = restore_model(ckpt["model_type"], ckpt["vocab_sizes"], ckpt["model_params"])
    model.load_state_dict(ckpt["model_state_dict"])
    dataset = NumpyDataset.load(enc_dir, torch.device("cpu"), load_test=True)
    trainer = Trainer(cfg, str(CODE_DIR.parent))
    restored_ne, _ = trainer._val_score_numpy(model, dataset.val)
    assert restored_ne == pytest.approx(history.val_ctr_ne.min())

    rows = json.loads((run_dir / "metrics.json").read_text())
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        assert [r["population"] for r in subset] == ["CTR", "CTCVR"]
        data = getattr(dataset, split)
        assert all(r["rows"] == len(data) for r in subset)
        if save_predictions:
            predictions = pd.read_parquet(run_dir / "predictions" / f"{split}.parquet")
            assert list(predictions.columns) == ["sample_id", "y", "z", "q", "c"]
            assert predictions[["q", "c"]].ge(0).all().all()
            assert predictions[["q", "c"]].le(1).all().all()
        else:
            assert (run_dir / "metrics" / f"{split}.json").is_file()
            assert not (run_dir / "predictions").exists()
    test_ctr = next(r for r in rows if r["split"] == "test" and r["population"] == "CTR")
    assert test_ctr["reference_auc"] == 0.661
    assert test_ctr["auc_delta"] == pytest.approx(test_ctr["auc"] - 0.661)
    assert "Paper CTR AUC reference: 0.661" in done.stdout
    assert "Test CTR AUC delta:" in done.stdout

    # Deliberately change today's architecture: the exporter must restore the
    # checkpoint's architecture and reproduce its predictions, not use this YAML.
    cfg["model_params"].update(emb_dim=2, tower_dims=[4])
    config_path.write_text(yaml.safe_dump(cfg))
    exported = tmp_path / "restored_test.parquet"
    done = _run("predict_encoded.py", "--config", config_path, "--run-dir", run_dir,
                "--split", "test", "--out", exported)
    assert done.returncode == 0, done.stderr[-5000:]
    df = pd.read_parquet(exported)
    ids, y, z, scores = trainer._predict_numpy(model, dataset.test, split_name="test")
    np.testing.assert_array_equal(df.sample_id, ids)
    np.testing.assert_array_equal(df.y, y)
    np.testing.assert_array_equal(df.z, z)
    for column in ("q", "c"):
        np.testing.assert_allclose(df[column], scores[column], rtol=1e-6, atol=1e-7)
