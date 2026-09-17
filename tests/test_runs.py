"""Run folders, the `latest` pointer and the best-model record (training/runs.py).

The unit tests pin each rule on its own. The lifecycle test at the bottom drives
the real scripts/train.py twice over the shared real-data fixture -- once to
finish, once to fail -- and checks the pointers move only for the run that
finished.
"""

import csv
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from training import runs

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MST = timezone(timedelta(hours=-7))


def _cfg(model="dcn"):
    return {"model": model, "training": {"lr": 1e-3}}


def _start(tmp_path, root, cfg=None, code_dir=CODE_DIR):
    """start_run on a config written to disk, the way train.py calls it."""
    cfg = cfg or _cfg()
    # A run is named by its config file, so each config is <model>.yaml in its own folder.
    config_dir = tmp_path / f"cfg-{len(os.listdir(tmp_path))}"
    config_dir.mkdir()
    config_path = config_dir / f"{cfg['model']}.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return runs.start_run(str(root), cfg, str(config_path), code_dir, str(tmp_path))


def _entry(run, value, metric="ctr_prauc"):
    return {"run": run, "checkpoint": f"dcn/{run}/best.pt",
            "selected_on": {"metric": metric, "split": "val", "value": value, "epoch": 1}}


# ---------------------------------------------------------------------------
# run folders
# ---------------------------------------------------------------------------

def test_run_id_is_utc_start_time_in_iso_basic_format(tmp_path):
    local_start = datetime(2026, 9, 15, 19, 3, 56, tzinfo=MST)
    path = runs.new_run_dir(str(tmp_path), "dcn", local_start)
    assert os.path.basename(path) == "20260916T020356Z"
    assert os.path.dirname(path) == str(tmp_path / "dcn")


def test_runs_started_in_the_same_second_get_distinct_folders(tmp_path):
    t = datetime(2026, 9, 16, 2, 3, 56, tzinfo=timezone.utc)
    names = [os.path.basename(runs.new_run_dir(str(tmp_path), "dcn", t)) for _ in range(3)]
    assert names == ["20260916T020356Z", "20260916T020356Z-2", "20260916T020356Z-3"]


def test_start_run_copies_the_config_verbatim_and_marks_running(tmp_path):
    cfg = {"model": "esmm", "training": {"lr": 0.001, "seed": 42}}
    source = "model: esmm\ntraining:\n  lr: 0.001   # paper's ESMM learning rate\n  seed: 42\n"
    config_path = tmp_path / "esmm.yaml"
    config_path.write_text(source)
    run_dir = runs.start_run(str(tmp_path / "artifacts"), cfg, str(config_path), CODE_DIR,
                             str(tmp_path / "none"))

    assert os.path.dirname(run_dir) == str(tmp_path / "artifacts" / "esmm")
    with open(os.path.join(run_dir, "config.yaml")) as f:
        assert f.read() == source  # comments survive
    run = runs.read_run(run_dir)
    assert run["status"] == runs.RUNNING and run["finished_at"] is None
    assert run["run"] == os.path.basename(run_dir)
    assert run["name"] == "esmm" and run["model"] == "esmm"


def test_runs_are_named_by_config_file_not_model(tmp_path):
    cfg = {"model": "esmm_logged_pctr", "training": {"lr": 0.001}}
    arms = []
    for arm in ("esmm_logged_pctr_kd05", "esmm_logged_pctr_product"):
        config_path = tmp_path / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        arms.append(runs.start_run(str(tmp_path / "artifacts"), cfg, str(config_path), CODE_DIR,
                                   str(tmp_path / "none")))

    assert [os.path.basename(os.path.dirname(d)) for d in arms] == [
        "esmm_logged_pctr_kd05", "esmm_logged_pctr_product"]
    run = runs.finish_run(arms[0], runs.FINISHED, selected_on={}, data={})
    assert runs.best_entry(run)["checkpoint"] == f"esmm_logged_pctr_kd05/{run['run']}/best.pt"


def test_logged_pctr_source_merges_split_files_into_run_json(tmp_path):
    run_dir = _start(tmp_path, tmp_path / "artifacts")
    source = {"dcn_run": "artifacts/dcn/r1", "run_id": "r1", "commit": "abc",
              "selected_on": None, "score_column": "score"}
    runs.record_logged_pctr(run_dir, {**source, "files": {"train": {"file": "t"}, "val": {"file": "v"}}})
    runs.record_logged_pctr(run_dir, {**source, "files": {"test": {"file": "x"}}})

    recorded = runs.read_run(run_dir)["logged_pctr"]
    assert recorded == {**source, "files": {"train": {"file": "t"}, "val": {"file": "v"},
                                            "test": {"file": "x"}}}
    run = runs.finish_run(run_dir, runs.FINISHED, selected_on={}, data={})
    assert runs.best_entry(run)["logged_pctr"] == recorded
    assert len(run["git"]["commit"]) == 40
    assert {"host", "torch", "gpu", "slurm_job_id"} <= set(run["environment"])


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def test_git_diff_saved_only_when_the_tree_is_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.py").write_text("x = 1\n")
    _git(repo, "add", "f.py")
    _git(repo, "commit", "-qm", "init")

    clean = _start(tmp_path, tmp_path / "a", code_dir=str(repo))
    assert not os.path.exists(os.path.join(clean, "git.diff"))
    assert runs.read_run(clean)["git"]["dirty"] is False

    (repo / "f.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    dirty = _start(tmp_path, tmp_path / "b", code_dir=str(repo))
    with open(os.path.join(dirty, "git.diff")) as f:
        assert "+x = 2" in f.read()
    git = runs.read_run(dirty)["git"]
    assert git["dirty"] is True and git["untracked"] == ["new.py"]


def test_data_census_reads_split_meta(tmp_path):
    for split, n in (("train", 10), ("val", 2)):
        (tmp_path / split).mkdir()
        (tmp_path / split / "split_meta.json").write_text(json.dumps(
            {"split": split, "n_rows": n, "n_clicks": 1, "n_conversions": 0}))
    assert runs.data_census(str(tmp_path)) == {
        "train": {"n_rows": 10, "n_clicks": 1, "n_conversions": 0},
        "val": {"n_rows": 2, "n_clicks": 1, "n_conversions": 0},
    }


def test_finish_run_records_status_time_and_fields(tmp_path):
    run_dir = _start(tmp_path, tmp_path / "artifacts")
    run = runs.finish_run(run_dir, runs.FAILED, error="OutOfMemoryError: boom")
    assert run == runs.read_run(run_dir)
    assert run["status"] == runs.FAILED and run["error"] == "OutOfMemoryError: boom"
    assert run["finished_at"].endswith("Z")


def test_selection_is_the_last_saved_epoch(tmp_path):
    with open(tmp_path / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "val_score", "seconds", "saved"])
        w.writerows([[1, 0.16, 0.0786, 188.6, True], [2, 0.14, 0.0801, 188.2, True],
                     [3, 0.11, 0.0632, 188.1, False]])
    assert runs.selection(str(tmp_path), "ctr_prauc") == {
        "metric": "ctr_prauc", "split": "val", "value": 0.0801, "epoch": 2}


# ---------------------------------------------------------------------------
# latest pointer
# ---------------------------------------------------------------------------

def test_only_a_finished_run_can_become_latest(tmp_path):
    root = str(tmp_path / "artifacts")
    run_dir = _start(tmp_path, root)
    with pytest.raises(ValueError, match="RUNNING"):
        runs.update_latest(root, "dcn", run_dir)
    assert not os.path.exists(os.path.join(root, "dcn", "latest.json"))

    runs.finish_run(run_dir, runs.FINISHED)
    runs.update_latest(root, "dcn", run_dir)
    with open(os.path.join(root, "dcn", "latest.json")) as f:
        pointer = json.load(f)
    assert pointer["run"] == os.path.basename(run_dir)
    assert not os.path.isabs(pointer["run"])
    assert runs.resolve_run(root, "dcn") == run_dir
    assert runs.resolve_run(root, "dcn", pointer["run"]) == run_dir


def test_command_line_prints_the_latest_run(tmp_path):
    root = str(tmp_path / "artifacts")
    run_dir = _start(tmp_path, root)
    runs.finish_run(run_dir, runs.FINISHED)
    runs.update_latest(root, "dcn", run_dir)
    out = subprocess.run([sys.executable, "-m", "training.runs", "latest", "dcn",
                          "--artifacts", root], cwd=CODE_DIR, capture_output=True,
                         text=True, check=True).stdout
    assert out.strip() == run_dir


# ---------------------------------------------------------------------------
# best-model record
# ---------------------------------------------------------------------------

def test_best_is_replaced_only_by_a_strictly_better_validation_score(tmp_path):
    best = str(tmp_path / "best_models.yaml")
    assert runs.update_best(best, "dcn", _entry("r1", 0.0786)) is True
    assert runs.update_best(best, "dcn", _entry("r2", 0.0700)) is False   # worse
    assert runs.update_best(best, "dcn", _entry("r3", 0.0786)) is False   # tie
    with open(best) as f:
        assert yaml.safe_load(f)["dcn"]["run"] == "r1"

    assert runs.update_best(best, "dcn", _entry("r4", 0.0801)) is True
    with open(best) as f:
        assert yaml.safe_load(f)["dcn"]["run"] == "r4"


def test_best_ignores_a_run_selected_on_a_different_metric(tmp_path, caplog):
    best = str(tmp_path / "best_models.yaml")
    runs.update_best(best, "dcn", _entry("r1", 0.05))
    before = open(best).read()
    assert runs.update_best(best, "dcn", _entry("r2", 0.90, metric="ctr_auc")) is False
    assert open(best).read() == before
    assert "not comparable" in caplog.text


def test_best_keeps_models_separate_and_keeps_its_header(tmp_path):
    best = str(tmp_path / "best_models.yaml")
    runs.update_best(best, "dcn", _entry("d1", 0.06))
    runs.update_best(best, "esmm", _entry("e1", 0.0003, metric="ctcvr_prauc"))
    text = open(best).read()
    assert text.startswith("# Best finished run per model")
    assert set(yaml.safe_load(text)) == {"dcn", "esmm"}


def _record_best(best_file, model, barrier):
    barrier.wait()  # release every writer at once
    runs.update_best(best_file, model, _entry(f"{model}-run", 0.5))


def test_concurrent_runs_of_different_models_all_land_in_the_best_file(tmp_path):
    best = str(tmp_path / "best_models.yaml")
    models = [f"model{i}" for i in range(16)]
    barrier = mp.get_context("fork").Barrier(len(models))
    procs = [mp.get_context("fork").Process(target=_record_best, args=(best, m, barrier))
             for m in models]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    with open(best) as f:
        assert set(yaml.safe_load(f)) == set(models), "a concurrent write lost another model's entry"


# ---------------------------------------------------------------------------
# end to end through scripts/train.py
# ---------------------------------------------------------------------------

def _train(cfg: dict, tmp_path, name: str, best_file: str) -> subprocess.CompletedProcess:
    """train.py on `cfg` written as <name>/<model>.yaml, so its runs land under the model."""
    (tmp_path / name).mkdir()
    cfg_path = tmp_path / name / f"{cfg['model']}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return subprocess.run([sys.executable, os.path.join(CODE_DIR, "scripts", "train.py"),
                           "--config", str(cfg_path), "--best-models", best_file],
                          cwd=CODE_DIR, capture_output=True, text=True, timeout=900)


def _small_dcn_cfg(encoded_dir, artifacts, epochs=1):
    return {
        "model": "dcn",
        "training": {"epochs": epochs, "batch_size": 64, "eval_batch_size": 128, "lr": 1e-3,
                     "grad_clip_norm": 10, "shuffle_block_size": 1, "seed": 42,
                     "checkpoint_metric": "ctr_prauc", "gpus": 0, "load_test": False,
                     "out_dir": str(artifacts),
                     "early_stopping": {"patience": epochs + 1, "min_delta": 0.0}},
        "model_params": {"emb_dim": 8, "cross_rank": 4, "num_cross_layers": 1,
                         "deep_hidden_dims": [8], "dropout": 0.1},
        "data": {"encoded_dir": encoded_dir},
    }


def test_train_py_records_a_finished_run_and_a_failed_one(encoded, tmp_path):
    enc_dir, _, _ = encoded
    artifacts = tmp_path / "artifacts"
    best_file = str(tmp_path / "best_models.yaml")
    cfg = _small_dcn_cfg(enc_dir, artifacts)

    ok = _train(cfg, tmp_path, "ok", best_file)
    assert ok.returncode == 0, ok.stderr[-3000:]

    run_dir = runs.resolve_run(str(artifacts), "dcn")
    for name in ("config.yaml", "run.json", "history.csv", "best.pt", "metrics.json",
                 "predictions/train.parquet", "predictions/val.parquet",
                 "predictions/test.parquet"):
        assert os.path.exists(os.path.join(run_dir, name)), name

    with open(os.path.join(run_dir, "config.yaml")) as f, open(tmp_path / "ok" / "dcn.yaml") as g:
        assert f.read() == g.read()

    run = runs.read_run(run_dir)
    assert run["status"] == runs.FINISHED
    assert run["selected_on"]["metric"] == "ctr_prauc" and run["selected_on"]["epoch"] == 1
    assert set(run["test"]["CTR"]) == {"auc", "pr_auc", "ne", "ce"}
    assert run["data"]["train"]["n_rows"] > 0

    with open(best_file) as f:
        best = yaml.safe_load(f)["dcn"]
    assert best["run"] == run["run"]
    assert best["checkpoint"] == f"dcn/{run['run']}/best.pt"
    best_before = open(best_file).read()

    # An explicit --checkpoint is exported with the architecture it was trained
    # with, even when today's config differs and no artifacts root has a latest run.
    wider = {**cfg, "model_params": {**cfg["model_params"], "emb_dim": 16},
             "training": {**cfg["training"], "out_dir": str(tmp_path / "empty")}}
    (tmp_path / "wider.yaml").write_text(yaml.safe_dump(wider))
    predicted = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "scripts", "predict_encoded.py"),
         "--config", str(tmp_path / "wider.yaml"), "--split", "val",
         "--checkpoint", os.path.join(run_dir, "best.pt"), "--out", "val.parquet"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert predicted.returncode == 0, predicted.stderr[-3000:]
    assert os.path.exists(tmp_path / "val.parquet")

    # Same config with a constructor argument the model doesn't accept: fails
    # inside the tracked block, after its run folder exists.
    bad = {**cfg, "model_params": {**cfg["model_params"], "no_such_arg": 1}}
    failed = _train(bad, tmp_path, "bad", best_file)
    assert failed.returncode != 0

    folders = sorted(d for d in os.listdir(artifacts / "dcn") if d != "latest.json")
    assert len(folders) == 2
    failed_run = runs.read_run(str(artifacts / "dcn" / folders[-1]))
    assert failed_run["status"] == runs.FAILED
    assert "TypeError" in failed_run["error"]

    assert runs.resolve_run(str(artifacts), "dcn") == run_dir, "a failed run became latest"
    assert open(best_file).read() == best_before, "a failed run touched the best file"


def test_sigterm_records_the_run_as_failed(encoded, tmp_path):
    """SLURM cancellation and time limits arrive as SIGTERM before any SIGKILL."""
    enc_dir, _, _ = encoded
    artifacts = tmp_path / "artifacts"
    cfg_path = tmp_path / "dcn.yaml"  # runs are named by config file
    cfg_path.write_text(yaml.safe_dump(_small_dcn_cfg(enc_dir, artifacts, epochs=100_000)))

    proc = subprocess.Popen([sys.executable, os.path.join(CODE_DIR, "scripts", "train.py"),
                             "--config", str(cfg_path),
                             "--best-models", str(tmp_path / "best_models.yaml")],
                            cwd=CODE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 300
    while not list(artifacts.glob("dcn/*/run.json")):
        assert proc.poll() is None, "train.py exited before its run started"
        assert time.monotonic() < deadline, "run folder never appeared"
        time.sleep(0.2)

    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=120) != 0

    (run_json,) = (artifacts / "dcn").glob("*/run.json")
    run = json.loads(run_json.read_text())
    assert run["status"] == runs.FAILED and "SystemExit" in run["error"]
    assert not (artifacts / "dcn" / "latest.json").exists()
