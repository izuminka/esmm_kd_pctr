"""Logged pCTR end to end through scripts/train.py on the real-data fixture.

A small DCN is trained first and becomes the logged pCTR source. ESMM-based
runs then point at it: the logged-pCTR KD arm, standalone ESMM Direct Product,
and plain ESMM controls selected on the native CTCVR product. Each run must train, record
exactly which DCN files it used, and report CTCVR on the logged product.
"""

from metric_reports import canonical_rows, metric_report

import csv
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from training import runs

LOGGED_TRAINING_MODELS = {"esmm_logged_pctr", "esmm_direct_product"}
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("train", "val", "test")


def _train(config_path, best_file):
    return subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "train.py"),
                           "--config", str(config_path), "--best-models", best_file],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)


def _write(tmp_path, name, cfg):
    path = tmp_path / "configs" / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _training(artifacts, checkpoint_metric, save_predictions=True):
    return {"epochs": 1, "batch_size": 64, "eval_batch_size": 128, "lr": 1e-3,
            "grad_clip_norm": 0, "shuffle_block_size": 1, "seed": 42,
            "checkpoint_metric": checkpoint_metric, "save_predictions": save_predictions,
            "gpus": 0, "load_test": False, "out_dir": str(artifacts),
            "early_stopping": {"patience": 2, "min_delta": 0.0}}


@pytest.fixture(scope="module")
def logged_runs(encoded, tmp_path_factory):
    """Train the DCN source, then every ESMM-based run against it; returns their folders."""
    enc_dir, _, _ = encoded
    tmp = tmp_path_factory.mktemp("logged")
    artifacts, best_file = tmp / "artifacts", str(tmp / "best_models.yaml")

    dcn = _write(tmp, "dcn", {
        "model": "dcn", "training": _training(artifacts, "ctr_prauc"),
        "model_params": {"emb_dim": 8, "cross_rank": 4, "num_cross_layers": 1,
                         "deep_hidden_dims": [8], "dropout": 0.1},
        "data": {"encoded_dir": enc_dir}})
    done = _train(dcn, best_file)
    assert done.returncode == 0, done.stderr[-3000:]
    dcn_run = runs.resolve_run(str(artifacts), "dcn")
    logged_pctr = {"dcn_run": dcn_run,
                   "predictions": {s: os.path.join(dcn_run, "predictions", f"{s}.parquet") for s in SPLITS},
                   "score_column": "score"}

    with open(os.path.join(enc_dir, "field_ids.json")) as f:
        n_fields = len(json.load(f))
    esmm_params = {"emb_dim": 4, "tower_dims": [4 * n_fields, 8, 2]}
    arms = {
        "esmm_logged_pctr_kd_tau": ("esmm_logged_pctr", "logged_ctcvr_prauc", True,
                                    {**esmm_params, "objective": "kd_tau", "alpha": 0.5, "aux_weight": 1.0}),
        "esmm_logged_pctr_kd": ("esmm_logged_pctr", "logged_ctcvr_prauc", True,
                                {**esmm_params, "objective": "kd", "alpha": 0.5, "aux_weight": 2.0}),
        "esmm_direct_product": ("esmm_direct_product", "logged_ctcvr_prauc", False,
                                esmm_params),
        "esmm_native_select": ("esmm", "ctcvr_prauc", True, esmm_params),
        "esmm_native_2x": ("esmm", "ctcvr_prauc", False, {**esmm_params, "aux_weight": 2.0}),
    }
    folders = {"dcn": dcn_run}
    for name, (model, metric, save_predictions, params) in arms.items():
        path = _write(tmp, name, {"model": model,
                                  "training": _training(artifacts, metric, save_predictions),
                                  "model_params": params, "logged_pctr": {**logged_pctr, "predictions":
                                      {s: f for s, f in logged_pctr["predictions"].items()
                                       if s == "test" or (s == "train" and model in LOGGED_TRAINING_MODELS)
                                       or (s == "val" and model in LOGGED_TRAINING_MODELS)}},
                                  "data": {"encoded_dir": enc_dir, **({"nonclick_sampling": {"rate": .5, "seed": 2026}}
                                           if params.get("objective") == "kd_tau" else {})}})
        done = _train(path, best_file)
        assert done.returncode == 0, f"{name}: {done.stderr[-3000:]}"
        folders[name] = runs.resolve_run(str(artifacts), name)

    # A cohort run: its train export must carry the cohort's rows, not the whole split.
    cohort = {"fraction": 0.5, "ranks": 4, "block_size": 32, "seed": 42}
    path = _write(tmp, "esmm_cohort", {"model": "esmm",
                                       "training": _training(artifacts, "ctcvr_prauc", True),
                                       "model_params": esmm_params, "logged_pctr": logged_pctr,
                                       "data": {"encoded_dir": enc_dir, "train_cohort": cohort}})
    done = _train(path, best_file)
    assert done.returncode == 0, f"esmm_cohort: {done.stderr[-3000:]}"
    folders["esmm_cohort"] = runs.resolve_run(str(artifacts), "esmm_cohort")

    # Plain ESMM with the logged CTCVR turned off: no logged_pctr block at all.
    path = _write(tmp, "esmm_no_logged", {"model": "esmm",
                                          "training": _training(artifacts, "ctcvr_prauc", True),
                                          "model_params": esmm_params,
                                          "evaluation": {"is_logged_ctcvr": False},
                                          "data": {"encoded_dir": enc_dir}})
    done = _train(path, best_file)
    assert done.returncode == 0, f"esmm_no_logged: {done.stderr[-3000:]}"
    folders["esmm_no_logged"] = runs.resolve_run(str(artifacts), "esmm_no_logged")

    return {"runs": folders, "tmp": tmp, "best_file": best_file, "logged_pctr": logged_pctr,
            "arms": arms, "enc_dir": enc_dir}


ESMM_RUNS = ["esmm_logged_pctr_kd_tau", "esmm_logged_pctr_kd", "esmm_direct_product", "esmm_native_select", "esmm_native_2x"]


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_each_run_finishes_in_its_config_named_folder(logged_runs, name):
    run_dir = logged_runs["runs"][name]
    model, metric, _, _ = logged_runs["arms"][name]
    run = runs.read_run(run_dir)

    assert os.path.basename(os.path.dirname(run_dir)) == name
    assert run["status"] == runs.FINISHED
    assert run["name"] == name and run["model"] == model
    assert run["selected_on"]["metric"] == metric


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_every_esmm_run_reports_ctcvr_on_the_logged_product(logged_runs, name):
    run_dir = logged_runs["runs"][name]
    with metric_report(run_dir) as f:
        rows = json.load(f)
    expected = (["CTCVR-logged", "pCVR"] if name == "esmm_direct_product"
                else ["CTR", "CTCVR", "CTCVR-logged", "pCVR"])

    for split in SPLITS:
        assert [r["population"] for r in rows if r["split"] == split] == [
            p for p in expected if split == "test" or p != "CTCVR-logged"
            or (split == "val" and logged_runs["arms"][name][0] in LOGGED_TRAINING_MODELS)], split
    assert set(runs.read_run(run_dir)["test"]) == set(expected)


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_every_output_names_the_dcn_files_used(logged_runs, name):
    run_dir = logged_runs["runs"][name]
    block = logged_runs["logged_pctr"]
    dcn_run_id = os.path.basename(logged_runs["runs"]["dcn"])

    source = runs.read_run(run_dir)["logged_pctr"]
    assert source["dcn_run"] == block["dcn_run"] and source["run_id"] == dcn_run_id
    used = {"train", "test"} if logged_runs["arms"][name][0] in LOGGED_TRAINING_MODELS else {"test"}
    if logged_runs["arms"][name][0] in LOGGED_TRAINING_MODELS:
        used.add("val")
    assert {s: f["file"] for s, f in source["files"].items()} == {s: block["predictions"][s] for s in used}

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    if "train" in used:
        assert ckpt["logged_pctr"]["run_id"] == dcn_run_id
        assert set(ckpt["logged_pctr"]["files"]) == used - {"test"}
    else:
        assert "logged_pctr" not in ckpt  # test scores are loaded after checkpoint selection

    with metric_report(run_dir) as f:
        logged = [r for r in json.load(f) if r["population"] == "CTCVR-logged"]
    assert {r["split"]: r["logged_pctr_file"] for r in logged} == {s: block["predictions"][s] for s in used - {"train"}}
    assert all(r["logged_pctr_run"] == block["dcn_run"] for r in logged)

    with open(logged_runs["best_file"]) as f:
        assert yaml.safe_load(f)[name]["logged_pctr"] == source


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_history_tracks_the_prescribed_validation_metrics(logged_runs, name):
    with open(os.path.join(logged_runs["runs"][name], "history.csv"), newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert any("logged" in c for c in row) == (logged_runs["arms"][name][0] in LOGGED_TRAINING_MODELS)
    with metric_report(logged_runs["runs"][name]) as f:
        pcvr = next(r for r in json.load(f) if r["split"] == "val" and r["population"] == "pCVR")
    for metric, column in (("auc", "auc"), ("pr_auc", "prauc"), ("ne", "ne"), ("ce", "ce")):
        # This small real-data fixture has no validation conversions; AUC is undefined.
        assert float(row[f"val_pcvr_{column}"]) == pytest.approx(pcvr[metric], nan_ok=True)
    if name == "esmm_direct_product":
        assert row["val_ctcvr_prauc"] == ""
    else:
        assert float(row["val_ctcvr_prauc"]) >= 0
    _, metric, _, _ = logged_runs["arms"][name]
    column = {"ctcvr_prauc": "val_ctcvr_prauc", "logged_ctcvr_prauc": "val_ctcvr_logged_prauc"}[metric]
    assert float(row["val_score"]) == float(row[column])


def test_exported_predictions_carry_the_aligned_logged_pctr(logged_runs):
    dcn_run = logged_runs["runs"]["dcn"]
    for name in ("esmm_logged_pctr_kd", "esmm_native_select"):
        for split in SPLITS:
            df = pd.read_parquet(os.path.join(logged_runs["runs"][name], "predictions", f"{split}.parquet"))
            logged = split == "test" or (split == "val" and name == "esmm_logged_pctr_kd")
            assert list(df.columns) == ["sample_id", "y", "z", "q", "v"] + (["p"] if logged else [])
            if not logged:
                continue
            dcn = pd.read_parquet(os.path.join(dcn_run, "predictions", f"{split}.parquet")).set_index("sample_id")
            assert (df["p"].to_numpy() == dcn.loc[df["sample_id"], "score"].to_numpy()).all(), (name, split)


def test_checkpoint_rebuilds_the_arm(logged_runs):
    ckpt = torch.load(os.path.join(logged_runs["runs"]["esmm_logged_pctr_kd"], "best.pt"), weights_only=False)
    assert ckpt["model_type"] == "esmm_logged_pctr"
    assert {k: ckpt["model_params"][k] for k in ("objective", "alpha", "aux_weight")} == {
        "objective": "kd", "alpha": 0.5, "aux_weight": 2.0}

    direct = torch.load(
        os.path.join(logged_runs["runs"]["esmm_direct_product"], "best.pt"),
        weights_only=False)
    assert direct["model_type"] == "esmm_direct_product"
    assert set(direct["model_params"]) == {"emb_dim", "tower_dims"}

    esmm = torch.load(os.path.join(logged_runs["runs"]["esmm_native_2x"], "best.pt"), weights_only=False)
    assert esmm["model_type"] == "esmm" and esmm["model_params"]["aux_weight"] == 2.0


def test_plain_esmm_with_logged_ctcvr_off_reports_no_logged_product(logged_runs):
    run_dir = logged_runs["runs"]["esmm_no_logged"]
    run = runs.read_run(run_dir)
    assert run["status"] == runs.FINISHED and "logged_pctr" not in run
    with metric_report(run_dir) as f:
        rows = json.load(f)
    for split in SPLITS:
        assert [r["population"] for r in rows if r["split"] == split] == ["CTR", "CTCVR", "pCVR"]
    with open(os.path.join(run_dir, "history.csv"), newline="") as f:
        columns = csv.DictReader(f).fieldnames
        assert not any("logged" in c for c in columns)
        assert {f"val_{pop}_{metric}" for pop in ("ctr", "ctcvr", "pcvr")
                for metric in ("auc", "prauc", "ne", "ce")} <= set(columns)
    for split in SPLITS:
        df = pd.read_parquet(os.path.join(run_dir, "predictions", f"{split}.parquet"))
        assert list(df.columns) == ["sample_id", "y", "z", "q", "v"]
    assert "logged_pctr" not in torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)


@pytest.mark.parametrize("is_logged_ctcvr", [False, True])
def test_native_selection_on_the_logged_product_is_refused(logged_runs, is_logged_ctcvr):
    tmp = logged_runs["tmp"]
    cfg = yaml.safe_load(open(tmp / "configs" / "esmm_native_2x.yaml"))
    cfg["evaluation"] = {"is_logged_ctcvr": is_logged_ctcvr}
    cfg["training"]["checkpoint_metric"] = "logged_ctcvr_prauc"
    path = _write(tmp, "esmm_off_but_logged_select", cfg)

    done = _train(path, logged_runs["best_file"])

    assert done.returncode != 0
    assert "native ESMM validates on q*v" in done.stderr


def test_an_esmm_config_without_a_logged_pctr_source_is_refused(logged_runs):
    tmp = logged_runs["tmp"]
    cfg = yaml.safe_load(open(tmp / "configs" / "esmm_native_select.yaml"))
    del cfg["logged_pctr"]
    cfg["evaluation"] = {"is_logged_ctcvr": True}  # Explicit legacy opt-in needs a source.
    path = _write(tmp, "esmm_no_source", cfg)

    done = _train(path, logged_runs["best_file"])

    assert done.returncode != 0
    assert "needs a logged_pctr block" in done.stderr
    assert not os.path.exists(os.path.join(cfg["training"]["out_dir"], "esmm_no_source"))


# --- regressions ---


def test_predict_uses_the_runs_own_logged_pctr_setting(logged_runs):
    """A run trained without logged pCTR is re-scored without it, whatever today's config says."""
    run_dir = logged_runs["runs"]["esmm_no_logged"]
    logged_config = logged_runs["tmp"] / "configs" / "esmm_native_select.yaml"  # wants p

    done = subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "predict_encoded.py"),
                           "--config", str(logged_config), "--split", "val", "--run-dir", run_dir],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)

    assert done.returncode == 0, done.stderr[-3000:]
    df = pd.read_parquet(os.path.join(run_dir, "predictions", "val.parquet"))
    assert list(df.columns) == ["sample_id", "y", "z", "q", "v"]


def test_predict_exports_the_cohort_rows_of_a_cohort_run(logged_runs):
    run_dir = logged_runs["runs"]["esmm_cohort"]
    n_cohort = runs.read_run(run_dir)["train_cohort"]["n_rows"]
    n_full = runs.read_run(run_dir)["data"]["train"]["n_rows"]
    assert n_cohort < n_full

    done = subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "predict_encoded.py"),
                           "--config", str(logged_runs["tmp"] / "configs" / "esmm_cohort.yaml"),
                           "--split", "train", "--run-dir", run_dir],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)

    assert done.returncode == 0, done.stderr[-3000:]
    df = pd.read_parquet(os.path.join(run_dir, "predictions", "train.parquet"))
    assert len(df) == n_cohort, "the full split overwrote the cohort's train predictions"


@pytest.mark.parametrize("name", ["esmm_logged_pctr_kd", "esmm_no_logged"])
def test_report_metrics_resolves_a_run_by_config_name(logged_runs, name):
    """Run folders are keyed by config name, and the model comes from the run itself."""
    artifacts = os.path.dirname(os.path.dirname(logged_runs["runs"][name]))

    done = subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "report_metrics.py"),
                           "--artifacts", artifacts, "--name", name],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)

    assert done.returncode == 0, done.stderr[-3000:]
    assert f"artifacts/{name}" in done.stdout or name in done.stdout
    with metric_report(logged_runs["runs"][name]) as f:
        populations = {r["population"] for r in json.load(f)}
    assert ("CTCVR-logged" in populations) == (name != "esmm_no_logged")


def test_report_metrics_needs_a_run_to_read():
    done = subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "report_metrics.py")],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)
    assert done.returncode != 0 and "--run-dir" in done.stderr


def test_a_finished_run_stays_finished_when_the_bookkeeping_fails(logged_runs, tmp_path):
    """The best-models path is a directory here, so recording the result raises."""
    cfg_path = logged_runs["tmp"] / "configs" / "esmm_no_logged.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    artifacts = cfg["training"]["out_dir"]
    blocked = tmp_path / "best_models.yaml"
    blocked.mkdir()

    done = _train(cfg_path, str(blocked))

    assert done.returncode != 0
    run_dir = runs.resolve_run(artifacts, "esmm_no_logged")  # only finished runs are latest
    run = runs.read_run(run_dir)
    assert run["status"] == runs.FINISHED and "IsADirectoryError" in run["error"]


def test_tau_checkpoint_and_retained_teacher_alignment(logged_runs):
    from pathlib import Path
    import numpy as np
    from scripts.predict_encoded import build_model
    run = Path(logged_runs['runs']['esmm_logged_pctr_kd_tau'])
    ckpt = torch.load(run/'best.pt', weights_only=False)
    assert ckpt['nonclick_sampling'] == {'rate': .5, 'seed': 2026}
    restored = build_model(ckpt['model_type'], ckpt['vocab_sizes'], ckpt['model_params'])
    assert restored.objective == 'kd_tau' and restored.nonclick_rate == .5
    enc = Path(logged_runs['enc_dir'])/'train'
    y = np.load(enc/'y.npy')
    ids = np.load(enc/'sample_id.npy')
    keep = (y == 1) | (np.random.default_rng(2026).random(len(y)) < .5)
    saved = pd.read_parquet(run/'predictions/train.parquet')
    assert np.array_equal(saved.sample_id.to_numpy(), ids[keep])
    saved_val = pd.read_parquet(run/'predictions/val.parquet')
    teacher = pd.read_parquet(logged_runs['logged_pctr']['predictions']['val']).set_index('sample_id')
    assert np.array_equal(saved_val.p.to_numpy(), teacher.loc[saved_val.sample_id, 'score'].to_numpy())


def test_tau_separate_logged_evaluation_pipeline(logged_runs):
    from pathlib import Path
    from evaluation.predictions import evaluate_native
    from evaluation.logged_ctcvr import evaluate_logged
    run = Path(logged_runs['runs']['esmm_logged_pctr_kd_tau'])
    root = Path(CODE_DIR).parent
    predictions = evaluate_native(run, root, gpus=0, refresh=True)
    saved = pd.read_parquet(run/'predictions/test.parquet')
    assert 'p' not in saved.columns  # native inference needs no external score
    recipe = Path(logged_runs['tmp'])/'eval_tau_teacher.yaml'
    recipe.write_text(yaml.safe_dump({'pctr_model':'fixture_dcn', 'logged_pctr':{
        **logged_runs['logged_pctr'], 'predictions':{'test':logged_runs['logged_pctr']['predictions']['test']}}}))
    path = evaluate_logged(run, recipe, root, predictions=predictions)
    row, = json.loads(path.read_text())
    teacher = pd.read_parquet(logged_runs['logged_pctr']['predictions']['test']).set_index('sample_id')
    scores = teacher.loc[saved.sample_id, 'score'].to_numpy()*saved.v.to_numpy()
    import numpy as np
    positives = (saved.y.to_numpy()*saved.z.to_numpy()).sum()
    if positives:
        assert row['ce'] == pytest.approx(scores.sum()/positives)
    assert row['rows'] == len(saved)
    assert row['pctr_model'] == 'fixture_dcn'
