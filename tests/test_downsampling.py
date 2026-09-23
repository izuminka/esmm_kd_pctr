"""Downsampling tests: sampling, posterior recovery, training, and inference."""

import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data.raw import Impression
from data.vocab import build_vocabs, encode_batch
from evaluation import esmm as evaluator
from evaluation.esmm import evaluate_esmm
from models.esmm import ESMM, esmm_loss
from scripts import evaluate, predict
from scripts.predict_encoded import build_model as restore_model
from scripts.train import build_metric_fn, build_model, load_config
from training import cohorts, downsampling, trainer
from training.dataset import NumpyDataset
from training.trainer import Trainer
from test_cohorts import _FakeSplit
from test_esmm_protocol import TinySplit, tiny_cfg
from test_evaluate import _make_synthetic_batch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
CONFIG = CODE / "configs_downsampled/esmm/train.yaml"


# Sampling and posterior unit tests


@pytest.mark.parametrize("rate", [0.05, 0.5, 1.0])
def test_sampling_preserves_clicks_and_alignment(rate):
    full = _FakeSplit(10000)
    full.logged_pctr = torch.linspace(0, 1, len(full))
    spec = {"rate": rate, "seed": 2026}
    kept = downsampling.restrict(full, spec)
    again = downsampling.restrict(full, spec)
    assert torch.equal(kept.sample_ids, again.sample_ids)
    assert set(full.sample_ids[full.ys == 1].tolist()) <= set(kept.sample_ids.tolist())
    assert kept.ys.sum() == full.ys.sum()
    assert kept.zs.sum() == full.zs.sum()
    assert torch.equal(kept.logged_pctr, full.logged_pctr[kept.rows])
    idx = torch.arange(min(20, len(kept)))
    _, y, z = kept.get_batch(idx)
    assert torch.equal(y, full.ys[kept.rows[idx]].float())
    assert torch.equal(z, full.zs[kept.rows[idx]].float())
    if rate == 1:
        assert torch.equal(kept.sample_ids, full.sample_ids)
    else:
        retained_fraction = (kept.ys == 0).sum() / (full.ys == 0).sum()
        assert float(retained_fraction) == pytest.approx(rate, abs=.02)
        other = downsampling.restrict(full, {**spec, "seed": 2027})
        assert not torch.equal(kept.sample_ids, other.sample_ids)


@pytest.mark.parametrize("rate", [0, -0.5, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_retention_rejected(rate):
    with pytest.raises(ValueError, match="retention rate"):
        downsampling.from_config({"model": "esmm", "data": {
            "nonclick_sampling": {"rate": rate, "seed": 2026}}})


@pytest.mark.parametrize("rate", [.05, .5, 1.])
def test_posterior_recovers_known_population(rate):
    # Derive the retained CTR from class counts, independently of the inverse.
    clicked = torch.tensor([0, 1, 10, 50, 100], dtype=torch.float64)
    nonclicked = 100 - clicked
    retained_ctr = clicked / (clicked + rate * nonclicked)
    recovered = downsampling.recover_ctr(retained_ctr, rate)
    torch.testing.assert_close(recovered, clicked / 100)
    assert recovered[0] == 0 and recovered[-1] == 1
    if rate == 1:
        assert recovered is retained_ctr


def test_config_default_and_model_guard():
    assert downsampling.from_config({"model": "esmm"}) is None
    cfg = {"model": "esmm_logged_pctr", "data": {
        "nonclick_sampling": {"rate": .5, "seed": 2026}}}
    with pytest.raises(ValueError, match="plain ESMM"):
        downsampling.from_config(cfg)


# Boundary cases and exact row selection


@pytest.mark.parametrize("rate", [None, 1, 1.0])
def test_full_rate_bypasses_all_inference_paths(tmp_path, monkeypatch, rate):
    cfg = tiny_cfg(tmp_path)
    cfg["evaluation"]["is_logged_ctcvr"] = False
    if rate is not None:
        cfg["data"]["nonclick_sampling"] = {"rate": rate, "seed": 2026}
    split = TinySplit()
    model = build_model(cfg, {"f": 8}).eval()
    trainer = Trainer(cfg, str(tmp_path))
    idx = torch.arange(len(split))
    batch, y, z = split.get_batch(idx)
    with torch.no_grad():
        raw_q, raw_v = model(batch)

    def unexpected(*args):
        raise AssertionError("full retention must never call recover_ctr")

    monkeypatch.setattr(downsampling, "recover_ctr", unexpected)
    monkeypatch.setattr(evaluator, "recover_ctr", unexpected)
    for name in ("train", "val", "test"):
        _, _, _, scores = trainer._predict_numpy(model, split, split_name=name)
        np.testing.assert_array_equal(scores["q"], raw_q.numpy())
        np.testing.assert_array_equal(scores["v"], raw_v.numpy())
        np.testing.assert_array_equal(scores["q"] * scores["v"], (raw_q * raw_v).numpy())
    trainer._val_score_numpy(model, split)
    rows = [Impression(i, int(y[i]), int(z[i]), str(i), {"f": [(i + 1, 1.)]})
            for i in range(len(split))]
    vocab = {"f": {str(i): i for i in range(1, 9)}}
    _, _, _, q, v = trainer.predict_esmm(model, rows, vocab, ["f"])
    np.testing.assert_array_equal(q, raw_q.numpy())
    np.testing.assert_array_equal(v, raw_v.numpy())
    kwargs = {} if rate is None else {"nonclick_rate": rate}
    evaluator.evaluate_esmm(model, batch, y, z, is_logged_ctcvr=False, **kwargs)


@pytest.mark.parametrize("rate", [.05, .5, .999])
def test_recovery_applied_once_to_ctr_only(tmp_path, monkeypatch, rate):
    cfg = tiny_cfg(tmp_path)
    cfg["evaluation"]["is_logged_ctcvr"] = False
    cfg["data"]["nonclick_sampling"] = {"rate": rate, "seed": 2026}
    cfg["training"]["batch_size"] = 8
    split = TinySplit()
    model = build_model(cfg, {"f": 8}).eval()
    trainer = Trainer(cfg, str(tmp_path))
    batch, y, z = split.get_batch(torch.arange(len(split)))
    with torch.no_grad():
        raw_q, raw_v = model(batch)
    # Independent odds-space reference, evaluated at higher precision.
    expected = torch.sigmoid(torch.logit(raw_q.double()) + math.log(rate)).float()
    recover = downsampling.recover_ctr
    calls = []

    def observed(q, rho):
        calls.append((q.clone(), rho))
        return recover(q, rho)

    monkeypatch.setattr(downsampling, "recover_ctr", observed)
    monkeypatch.setattr(evaluator, "recover_ctr", observed)
    for name in ("train", "val", "test"):
        calls.clear()
        _, _, _, scores = trainer._predict_numpy(model, split, split_name=name)
        assert len(calls) == 1
        torch.testing.assert_close(calls[0][0], raw_q, rtol=0, atol=0)
        assert calls[0][1] == rate
        np.testing.assert_allclose(scores["q"], expected.numpy(), rtol=2e-6)
        assert (scores["q"] < raw_q.numpy()).all()
        np.testing.assert_array_equal(scores["v"], raw_v.numpy())
        np.testing.assert_allclose(scores["q"] * scores["v"], (expected * raw_v).numpy(), rtol=2e-6)
    rows = [Impression(i, int(y[i]), int(z[i]), str(i), {"f": [(i + 1, 1.)]})
            for i in range(len(split))]
    calls.clear()
    _, _, _, q, v = trainer.predict_esmm(model, rows, {"f": {str(i): i for i in range(1, 9)}}, ["f"])
    assert len(calls) == 1
    np.testing.assert_allclose(q, expected.numpy(), rtol=2e-6)
    np.testing.assert_array_equal(v, raw_v.numpy())
    calls.clear()
    evaluator.evaluate_esmm(model, batch, y, z, is_logged_ctcvr=False, nonclick_rate=rate)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("rate", [1, 1.0, .5, .05, 1e-12, 1e-50])
def test_probability_boundaries_and_no_mutation(dtype, rate):
    eps = torch.finfo(dtype).eps
    q = torch.tensor([0., eps, .1, .5, .9, 1 - eps, 1.], dtype=dtype)
    before = q.clone()
    actual = downsampling.recover_ctr(q, rate)
    expected = torch.sigmoid(torch.logit(q.double()) + math.log(rate)).to(dtype)
    rtol = 2e-3 if dtype == torch.float16 else (2e-6 if dtype == torch.float32 else 1e-13)
    # Permit only one subnormal ULP of absolute error at underflow boundaries.
    atol = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    assert (actual[1:] >= actual[:-1]).all()
    assert actual[0] == 0 and actual[-1] == 1
    torch.testing.assert_close(q, before, rtol=0, atol=0)
    if rate == 1:
        assert actual is q


@pytest.mark.parametrize("rate", [1, 1.0, .05, 1e-50])
def test_numpy_scores_and_scalar_endpoints(rate):
    q = np.array([0., .1, .5, 1.], dtype=np.float32)
    before = q.copy()
    actual = downsampling.recover_ctr(q, rate)
    assert np.isfinite(actual).all()
    assert actual[0] == 0 and actual[-1] == 1
    np.testing.assert_array_equal(q, before)
    assert downsampling.recover_ctr(0., rate) == 0
    assert downsampling.recover_ctr(1., rate) == 1
    if rate == 1:
        assert actual is q


@pytest.mark.parametrize("rate", [0, -1, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_rate_rejected_by_all_entry_points(tmp_path, rate):
    cfg = tiny_cfg(tmp_path)
    cfg["evaluation"]["is_logged_ctcvr"] = False
    cfg["data"]["nonclick_sampling"] = {"rate": rate, "seed": 0}
    with pytest.raises(ValueError, match="retention rate"):
        Trainer(cfg, str(tmp_path))
    with pytest.raises(ValueError, match="retention rate"):
        downsampling.restrict(_FakeSplit(8), cfg["data"]["nonclick_sampling"])
    with pytest.raises(ValueError, match="retention rate"):
        downsampling.recover_ctr(torch.tensor([.5]), rate)
    split = TinySplit()
    batch, y, z = split.get_batch(torch.arange(len(split)))
    with pytest.raises(ValueError, match="retention rate"):
        evaluator.evaluate_esmm(build_model(cfg, {"f": 8}), batch, y, z,
                                is_logged_ctcvr=False, nonclick_rate=rate)


@pytest.mark.parametrize("spec", [False, [], "sampling", {}, {"rate": .5},
    {"rate": .5, "seed": 0, "extra": 1}, {"rate": .5, "seed": -1},
    {"rate": .5, "seed": True}, {"rate": .5, "seed": 1.5}])
def test_malformed_sampling_spec_rejected(spec):
    with pytest.raises(ValueError):
        downsampling.from_config({"model": "esmm", "data": {"nonclick_sampling": spec}})
    with pytest.raises(ValueError):
        downsampling.restrict(_FakeSplit(8), spec)


def test_exact_rows_and_threshold_boundary(monkeypatch):
    # Controlled draws: clicks always survive; nonclick draws == rate are rejected.
    split = _FakeSplit(6)
    split.ys = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int8)
    split.zs = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int8)
    draws = np.array([.99, 0., .4999, .99, .5, .999])

    class Draws:
        def random(self, n):
            assert n == 6
            return draws

    monkeypatch.setattr(downsampling.np.random, "default_rng", lambda seed: Draws())
    kept = downsampling.restrict(split, {"rate": .5, "seed": 1})
    assert kept.rows.tolist() == [0, 1, 2, 3]
    assert kept.sample_ids.tolist() == [100, 101, 102, 103]
    assert len(kept) == 4
    assert kept.census()["n_clicks"] == 2
    assert kept.census()["n_conversions"] == 1


@pytest.mark.parametrize("clicks", [[1], [1, 1, 1], [0, 0, 0], [1, 0, 0]])
@pytest.mark.parametrize("rate", [1, 1.0, .5, 1e-12])
def test_tiny_populations(clicks, rate):
    split = _FakeSplit(len(clicks))
    split.ys = torch.tensor(clicks, dtype=torch.int8)
    split.zs = split.ys.clone()
    # Seed 0 draws [0.63696, 0.26979, 0.04097]; at .5 keep nonclick rows 1,2.
    expected = [i for i, y in enumerate(clicks) if y or rate == 1 or (rate == .5 and i in (1, 2))]
    if not expected:
        with pytest.raises(ValueError, match="no training rows"):
            downsampling.restrict(split, {"rate": rate, "seed": 0})
    else:
        kept = downsampling.restrict(split, {"rate": rate, "seed": 0})
        assert kept.rows.tolist() == expected
        assert len(kept) == len(expected)
        assert kept.zs.sum() == split.zs.sum()


@pytest.mark.parametrize("rate", [1, .5])
def test_empty_population_rejected(rate):
    with pytest.raises(ValueError, match="no training rows"):
        downsampling.restrict(_FakeSplit(0), {"rate": rate, "seed": 0})


def test_nested_cohort_preserves_feature_and_score_alignment():
    full = _FakeSplit(60)
    full.logged_pctr = torch.linspace(0, 1, 60)
    original_ids = full.sample_ids.clone()
    pool = cohorts.restrict(full, {"fraction": .5, "ranks": 4, "block_size": 4, "seed": 1})
    kept = downsampling.restrict(pool, {"rate": .5, "seed": 2026})
    original_rows = pool.rows[kept.rows]
    # FakeSplit records the actual feature row indices requested by the view.
    kept.get_batch(torch.arange(len(kept)))
    assert torch.equal(full.requested, original_rows)
    assert torch.equal(kept.sample_ids, full.sample_ids[original_rows])
    assert torch.equal(kept.logged_pctr, full.logged_pctr[original_rows])
    assert kept.ys.sum() == pool.ys.sum()
    assert kept.zs.sum() == pool.zs.sum()
    assert len(torch.unique(kept.sample_ids)) == len(kept)
    assert torch.equal(full.sample_ids, original_ids)


@pytest.mark.parametrize("rate", [.05, .5, .95])
def test_retained_count_matches_bernoulli_distribution(rate):
    split = _FakeSplit(10000)
    positives = int(split.ys.sum())
    negatives = len(split) - positives
    counts = np.array([len(downsampling.restrict(split, {"rate": rate, "seed": seed})) - positives
                       for seed in range(100)])
    expected_mean = negatives * rate
    expected_variance = negatives * rate * (1 - rate)
    assert abs(counts.mean() - expected_mean) < 5 * math.sqrt(expected_variance / len(counts))
    assert counts.var(ddof=1) == pytest.approx(expected_variance, rel=.5)
    assert len(np.unique(counts)) > 1  # Bernoulli sampling is not fixed-size sampling.


def test_sampling_is_independent_of_global_rng():
    split = _FakeSplit(1000)
    spec = {"rate": .5, "seed": 2026}
    before = downsampling.restrict(split, spec)
    np.random.seed(999)
    np.random.random(1000)
    torch.manual_seed(888)
    torch.rand(1000)
    after = downsampling.restrict(split, spec)
    assert torch.equal(before.rows, after.rows)


# GPU behavior and missing CTR heads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("rate", [1, 1.0, .05, 1e-12, 1e-50])
def test_cuda_preserves_device_dtype_endpoints_and_input(dtype, rate):
    q = torch.tensor([0., .01, .1, .5, .99, 1.], dtype=dtype, device="cuda")
    original = q.clone()
    result = downsampling.recover_ctr(q, rate)
    expected = torch.sigmoid(torch.logit(q.double()) + math.log(rate)).to(dtype)
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()
    assert result.device == q.device and result.dtype == q.dtype
    assert result[0] == 0 and result[-1] == 1
    assert torch.equal(q, original)
    if rate == 1:
        assert result is q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sampling_keeps_the_same_ids_on_cpu_and_gpu():
    cpu = TinySplit()
    gpu = TinySplit()
    gpu.device = torch.device("cuda:0")
    for name in ("sample_ids", "ys", "zs"):
        setattr(gpu, name, getattr(gpu, name).to(gpu.device))
    spec = {"rate": .5, "seed": 2026}
    a = downsampling.restrict(cpu, spec)
    b = downsampling.restrict(gpu, spec)
    assert b.rows.device == gpu.device
    assert torch.equal(a.sample_ids, b.sample_ids.cpu())
    assert torch.equal(a.ys, b.ys.cpu())
    assert torch.equal(a.zs, b.zs.cpu())


def test_missing_ctr_head_bypasses_recovery(tmp_path, monkeypatch):
    split = TinySplit()

    class ProductOnly(torch.nn.Module):
        def forward(self, batch):
            return None, torch.full((len(batch["f"][0]),), .25)

    def unexpected(*args):
        raise AssertionError("no CTR head to recover")

    monkeypatch.setattr(downsampling, "recover_ctr", unexpected)
    monkeypatch.setattr(evaluator, "recover_ctr", unexpected)
    cfg = tiny_cfg(tmp_path)
    cfg["model"] = "esmm_logged_pctr"
    cfg["evaluation"]["is_logged_ctcvr"] = False
    cfg["training"]["checkpoint_metric"] = "logged_ctcvr_ne"
    split.logged_pctr = torch.full((len(split),), .1)
    trainer = Trainer(cfg, str(tmp_path))
    _, _, _, scores = trainer._predict_numpy(ProductOnly(), split, split_name="test")
    assert set(scores) == {"v"}
    np.testing.assert_array_equal(scores["v"], np.full(len(split), .25))
    batch, y, z = split.get_batch(torch.arange(len(split)))
    metrics = evaluator.evaluate_esmm(ProductOnly(), batch, y, z, is_logged_ctcvr=False, nonclick_rate=.5)
    assert set(metrics) == {"pcvr"}


# Training and checkpoint integration


def test_recovery_is_evaluation_only_and_shared(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg["evaluation"]["is_logged_ctcvr"] = False
    cfg["data"]["nonclick_sampling"] = {"rate": .5, "seed": 2026}
    split = TinySplit()
    model = build_model(cfg, {"f": 8})
    trainer = Trainer(cfg, str(tmp_path))
    batch, y, z = split.get_batch(torch.arange(len(split)))
    model.eval()
    with torch.no_grad():
        q_raw, v_raw = model(batch)
    _, _, _, scores = trainer._predict_numpy(model, split, split_name="test")
    np.testing.assert_allclose(scores["q"], q_raw.numpy() / (2 - q_raw.numpy()), rtol=1e-6)
    np.testing.assert_array_equal(scores["v"], v_raw.numpy())
    legacy = evaluate_esmm(model, batch, y, z, is_logged_ctcvr=False, nonclick_rate=.5)
    selected, metrics = trainer._val_score_numpy(model, split)
    assert selected == pytest.approx(legacy["ctcvr"]["ne"])
    assert build_metric_fn(cfg)(model, batch, y, z) == pytest.approx(selected)
    assert metrics["val_pcvr_ne"] == pytest.approx(legacy["pcvr"]["ne"])
    assert not np.allclose(scores["q"] * scores["v"],
                           downsampling.recover_ctr(q_raw * v_raw, .5).numpy())

    # Training must still use ordinary retained-population ESMM gradients.
    reference = copy.deepcopy(model)
    q, v = reference(batch)
    expected = esmm_loss(y, z, q, v, reference.aux_weight)
    expected.backward()
    actual = trainer._train_step_numpy(model, split, torch.arange(len(split)))
    assert actual == pytest.approx(expected.item())
    for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)


def test_real_data_train_and_reexport(encoded, tmp_path):
    """Actual CLI train, test scoring, then re-export using a conflicting config."""
    encoded_dir, _, _ = encoded
    cfg = load_config(CONFIG)
    cfg["training"].update(epochs=1, batch_size=64, eval_batch_size=128, gpus=0, evaluate_test=True,
                           out_dir=str(tmp_path / "runs"), save_predictions=True)
    fields = json.loads((Path(encoded_dir) / "field_ids.json").read_text())
    cfg["model_params"] = {"emb_dim": 2, "tower_dims": [2 * len(fields), 8, 2], "aux_weight": 1.0}
    cfg["data"]["encoded_dir"] = encoded_dir
    config = tmp_path / "esmm_smoke.yaml"
    config.write_text(yaml.safe_dump(cfg))

    def run(script, *args):
        result = subprocess.run([sys.executable, str(CODE / "scripts" / script), *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, timeout=180)
        (tmp_path / f"{script}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr

    run("train.py", "--config", config, "--best-models", tmp_path / "best_models.yaml")
    run_dir, = (tmp_path / "runs" / "esmm_smoke").glob("*/best.pt")
    run_dir = run_dir.parent
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "FINISHED"
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)
    assert ckpt["nonclick_sampling"] == cfg["data"]["nonclick_sampling"]
    data = NumpyDataset.load(encoded_dir, device=torch.device("cpu"), load_test=True)
    expected_train = downsampling.restrict(data.train, ckpt["nonclick_sampling"])
    train = pd.read_parquet(run_dir / "predictions/train.parquet")
    np.testing.assert_array_equal(train.sample_id, expected_train.sample_ids.numpy())
    np.testing.assert_array_equal(np.load(run_dir / "retained_sample_ids.npy"), train.sample_id)
    assert metadata["nonclick_sampling"]["n_clicks"] == int(data.train.ys.sum())
    assert metadata["nonclick_sampling"]["n_conversions"] == int(data.train.zs.sum())
    model = restore_model("esmm", ckpt["vocab_sizes"], ckpt["model_params"])
    model.load_state_dict(ckpt["model_state_dict"])
    trainer = Trainer(cfg, str(ROOT))
    val_score, _ = trainer._val_score_numpy(model, data.val)
    with (run_dir / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    assert float(history[0]["val_score"]) == pytest.approx(val_score)
    for name in ("val", "test"):
        table = pd.read_parquet(run_dir / "predictions" / f"{name}.parquet")
        split = getattr(data, name)
        np.testing.assert_array_equal(table.sample_id, split.sample_ids.numpy())
        _, _, _, scores = trainer._predict_numpy(model, split, split_name=name)
        np.testing.assert_allclose(table.q, scores["q"], rtol=1e-6)
        np.testing.assert_allclose(table.v, scores["v"], rtol=1e-6)

    cfg["data"]["nonclick_sampling"]["rate"] = .05
    config.write_text(yaml.safe_dump(cfg))
    run("predict_encoded.py", "--config", config, "--run-dir", run_dir,
        "--split", "test", "--out", tmp_path / "reexport.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "reexport.parquet"),
                                  pd.read_parquet(run_dir / "predictions/test.parquet"))


# Legacy inference commands


def test_checkpoint_correction_in_legacy_commands(tmp_path, monkeypatch, capsys):
    rows = [Impression(i, y, z, str(i), {"f": [(i, 1.)]})
            for i, (y, z) in enumerate([(1, 1), (1, 0), (0, 0), (0, 0)])]
    vocab = build_vocabs(rows, ["f"])
    params = {"emb_dim": 2, "tower_dims": [2, 4, 2], "aux_weight": 1.0}
    model = ESMM({"f": len(vocab["f"])}, **params)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_type": "esmm", "model_params": params,
                "vocab_sizes": {"f": len(vocab["f"])}, "vocabs": vocab,
                "field_ids": ["f"], "model_state_dict": model.state_dict(),
                "nonclick_sampling": {"rate": .5, "seed": 2026}}, checkpoint)
    monkeypatch.setattr(trainer, "resolve_device", lambda *_: torch.device("cpu"))
    monkeypatch.setattr(predict, "iter_impressions", lambda *_: iter(rows))
    monkeypatch.setattr(evaluate, "iter_impressions", lambda *_: iter(rows))
    monkeypatch.setattr(sys, "argv", ["predict.py", "--checkpoint", str(checkpoint), "--split", "test"])
    predict.main()
    scores = pd.read_parquet(tmp_path / "predictions/test.parquet")
    model.eval()
    with torch.no_grad():
        q, v = model(encode_batch(rows, ["f"], vocab))
    np.testing.assert_allclose(scores.q, q.numpy() / (2 - q.numpy()), rtol=1e-6)
    np.testing.assert_array_equal(scores.v, v.numpy())

    from evaluation.metrics import compute_metrics
    expected = compute_metrics(scores.q.to_numpy() * scores.v.to_numpy(), np.array([1, 0, 0, 0]))
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(checkpoint), "--split", "test"])
    evaluate.main()
    output = capsys.readouterr().out
    assert "ctcvr:  " + "  ".join(f"{k}={value:.4f}" for k, value in expected.items()) in output


def test_full_retention_bypasses_recovery(monkeypatch):
    from evaluation import esmm as esmm_evaluation

    encoded, y, z, vocab_sizes = _make_synthetic_batch()
    model = ESMM(vocab_sizes, emb_dim=460)

    def unexpected_recovery(*args):
        raise AssertionError("rate=1 must bypass recover_ctr entirely")

    monkeypatch.setattr(esmm_evaluation, "recover_ctr", unexpected_recovery)
    default = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False)
    explicit = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False, nonclick_rate=1.0)
    assert explicit == default
