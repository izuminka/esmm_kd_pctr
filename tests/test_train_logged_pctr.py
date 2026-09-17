"""Logged pCTR end to end through scripts/train.py on the real-data fixture.

A small DCN is trained first and becomes the logged pCTR source. ESMM-based
runs then point at it: the logged-pCTR arms (kd and product), and the plain
ESMM control selected on each CTCVR product. Each run must train, record
exactly which DCN files it used, and report CTCVR on the logged product.
"""

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
        "esmm_logged_pctr_kd": ("esmm_logged_pctr", "logged_ctcvr_prauc", True,
                                {**esmm_params, "objective": "kd", "alpha": 0.5, "aux_weight": 2.0}),
        "esmm_logged_pctr_product": ("esmm_logged_pctr", "logged_ctcvr_prauc", False,
                                     {**esmm_params, "objective": "product"}),
        "esmm_native_select": ("esmm", "ctcvr_prauc", True, esmm_params),
        "esmm_logged_select": ("esmm", "logged_ctcvr_prauc", False, {**esmm_params, "aux_weight": 2.0}),
    }
    folders = {"dcn": dcn_run}
    for name, (model, metric, save_predictions, params) in arms.items():
        path = _write(tmp, name, {"model": model,
                                  "training": _training(artifacts, metric, save_predictions),
                                  "model_params": params, "logged_pctr": logged_pctr,
                                  "data": {"encoded_dir": enc_dir}})
        done = _train(path, best_file)
        assert done.returncode == 0, f"{name}: {done.stderr[-3000:]}"
        folders[name] = runs.resolve_run(str(artifacts), name)

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


ESMM_RUNS = ["esmm_logged_pctr_kd", "esmm_logged_pctr_product", "esmm_native_select", "esmm_logged_select"]


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
    with open(os.path.join(run_dir, "metrics.json")) as f:
        rows = json.load(f)
    expected = (["CTCVR-logged", "pCVR"] if name == "esmm_logged_pctr_product"
                else ["CTR", "CTCVR", "CTCVR-logged", "pCVR"])

    for split in SPLITS:
        assert [r["population"] for r in rows if r["split"] == split] == expected, split
    assert set(runs.read_run(run_dir)["test"]) == set(expected)


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_every_output_names_the_dcn_files_used(logged_runs, name):
    run_dir = logged_runs["runs"][name]
    block = logged_runs["logged_pctr"]
    dcn_run_id = os.path.basename(logged_runs["runs"]["dcn"])

    source = runs.read_run(run_dir)["logged_pctr"]
    assert source["dcn_run"] == block["dcn_run"] and source["run_id"] == dcn_run_id
    assert {s: f["file"] for s, f in source["files"].items()} == block["predictions"]

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    assert ckpt["logged_pctr"]["run_id"] == dcn_run_id
    assert set(ckpt["logged_pctr"]["files"]) == {"train", "val"}  # test is scored after training

    with open(os.path.join(run_dir, "metrics.json")) as f:
        logged = [r for r in json.load(f) if r["population"] == "CTCVR-logged"]
    assert {r["split"]: r["logged_pctr_file"] for r in logged} == block["predictions"]
    assert all(r["logged_pctr_run"] == block["dcn_run"] for r in logged)

    with open(logged_runs["best_file"]) as f:
        assert yaml.safe_load(f)[name]["logged_pctr"] == source


@pytest.mark.parametrize("name", ESMM_RUNS)
def test_history_tracks_both_ctcvr_products_per_epoch(logged_runs, name):
    with open(os.path.join(logged_runs["runs"][name], "history.csv"), newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert float(row["val_ctcvr_logged_prauc"]) >= 0
    if name == "esmm_logged_pctr_product":
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
            assert list(df.columns) == ["sample_id", "y", "z", "q", "v", "p"]
            dcn = pd.read_parquet(os.path.join(dcn_run, "predictions", f"{split}.parquet")).set_index("sample_id")
            assert (df["p"].to_numpy() == dcn.loc[df["sample_id"], "score"].to_numpy()).all(), (name, split)


def test_checkpoint_rebuilds_the_arm(logged_runs):
    ckpt = torch.load(os.path.join(logged_runs["runs"]["esmm_logged_pctr_kd"], "best.pt"), weights_only=False)
    assert ckpt["model_type"] == "esmm_logged_pctr"
    assert {k: ckpt["model_params"][k] for k in ("objective", "alpha", "aux_weight")} == {
        "objective": "kd", "alpha": 0.5, "aux_weight": 2.0}

    esmm = torch.load(os.path.join(logged_runs["runs"]["esmm_logged_select"], "best.pt"), weights_only=False)
    assert esmm["model_type"] == "esmm" and esmm["model_params"]["aux_weight"] == 2.0


def test_plain_esmm_with_logged_ctcvr_off_reports_no_logged_product(logged_runs):
    run_dir = logged_runs["runs"]["esmm_no_logged"]
    run = runs.read_run(run_dir)
    assert run["status"] == runs.FINISHED and "logged_pctr" not in run
    with open(os.path.join(run_dir, "metrics.json")) as f:
        rows = json.load(f)
    for split in SPLITS:
        assert [r["population"] for r in rows if r["split"] == split] == ["CTR", "CTCVR", "pCVR"]
    with open(os.path.join(run_dir, "history.csv"), newline="") as f:
        assert csv.DictReader(f).fieldnames == ["epoch", "loss", "val_score", "seconds", "saved",
                                                "val_ctcvr_prauc"]
    for split in SPLITS:
        df = pd.read_parquet(os.path.join(run_dir, "predictions", f"{split}.parquet"))
        assert list(df.columns) == ["sample_id", "y", "z", "q", "v"]
    assert "logged_pctr" not in torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)


def test_selecting_on_the_logged_product_with_logged_ctcvr_off_is_refused(logged_runs):
    tmp = logged_runs["tmp"]
    cfg = yaml.safe_load(open(tmp / "configs" / "esmm_logged_select.yaml"))
    cfg["evaluation"] = {"is_logged_ctcvr": False}
    path = _write(tmp, "esmm_off_but_logged_select", cfg)

    done = _train(path, logged_runs["best_file"])

    assert done.returncode != 0
    assert "logged_ctcvr_prauc needs evaluation.is_logged_ctcvr on" in done.stderr


def test_an_esmm_config_without_a_logged_pctr_source_is_refused(logged_runs):
    tmp = logged_runs["tmp"]
    cfg = yaml.safe_load(open(tmp / "configs" / "esmm_native_select.yaml"))
    del cfg["logged_pctr"]
    path = _write(tmp, "esmm_no_source", cfg)

    done = _train(path, logged_runs["best_file"])

    assert done.returncode != 0
    assert "needs a logged_pctr block" in done.stderr
    assert not os.path.exists(os.path.join(cfg["training"]["out_dir"], "esmm_no_source"))
