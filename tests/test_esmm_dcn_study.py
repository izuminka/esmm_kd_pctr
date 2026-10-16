import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from scripts import run_esmm_dcn_study as study_cli


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture(params=[-1, 1])
def prepared(tmp_path, request):
    model, gpus = "esmm_dcn", request.param
    code = tmp_path / "code"
    code.mkdir()
    for args in (("init", "-q"),):
        subprocess.run(["git", "-C", str(code), *args], check=True)
    (code / "tracked.py").write_text("ORIGINAL = True\n")
    subprocess.run(["git", "-C", str(code), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(code), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    (code / "tracked.py").write_text("ORIGINAL = False\n")
    (code / "new_model.py").write_text("JOINT = True\n")
    cfg = {
        "model": model,
        "training": {"seed": 42, "epochs": 5, "batch_size": 65536, "gpus": gpus,
                     "checkpoint_metric": "ctcvr_ne", "keep_checkpoint": "best",
                     "evaluate_test": False, "load_test": False, "out_dir": "artifacts",
                     "early_stopping": {"enabled": False}},
        "data": {"encoded_dir": "ali-ccp/encoded"},
    }
    logged = {"pctr_model": "teacher42", "logged_pctr": {
        "dcn_run": "artifacts/teacher/run1", "score_column": "score",
        "predictions": {"test": "artifacts/teacher/run1/predictions/test.parquet"}}}
    config, eval_config = tmp_path / "train.yaml", tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(cfg))
    eval_config.write_text(yaml.safe_dump(logged))
    study = study_cli.prepare(tmp_path / "study", config, eval_config,
                             code=code, root=tmp_path, python="/example/python")
    return study, tmp_path, code


def test_prepare_snapshots_uncommitted_source_and_private_seed_configs(prepared):
    study, root, code = prepared
    manifest = study_cli._manifest(study)
    variant = manifest["model"]
    assert manifest["variant"] == variant
    assert [job["seed"] for job in manifest["jobs"]] == [42, 43, 44, 45]
    assert manifest["untracked"] == ["new_model.py"]
    snapshot = Path(manifest["source_code"])
    assert (snapshot / "tracked.py").read_text() == "ORIGINAL = False\n"
    assert (snapshot / "new_model.py").read_text() == "JOINT = True\n"
    assert "+ORIGINAL = False" in (study / "code.diff").read_text()
    assert (snapshot.parent / "artifacts").is_symlink()
    for job in manifest["jobs"]:
        assert job["arm"] == f"{variant}_seed{job['seed']}"
        assert yaml.safe_load(Path(job["config"]).read_text())["training"]["seed"] == job["seed"]
        assert not Path(job["registry"]).exists()
    train = (study / "train.sbatch").read_text()
    assert "--array=0-3%4" in train and "--gres=gpu:1" in train
    assert "--run-index" in train and "--gres" not in (study / "report.sbatch").read_text()
    (code / "new_model.py").write_text("CHANGED = True\n")
    study_cli._verify_inputs(manifest)  # Active working-tree edits cannot alter queued jobs.
    (snapshot / "new_model.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="source snapshot changed"):
        study_cli._verify_inputs(manifest)
    assert not (study / "submission_jobs.json").exists()


def test_cli_defaults_to_esmm_dcn_and_its_study_root(prepared, monkeypatch):
    study, root, code = prepared
    manifest = study_cli._manifest(study)
    model_config = code / "configs/esmm_dcn/train.yaml"
    model_config.parent.mkdir(parents=True)
    cfg = yaml.safe_load(Path(manifest["jobs"][0]["config"]).read_text())
    cfg["model"] = "esmm_dcn"
    cfg["training"]["gpus"] = -1
    model_config.write_text(yaml.safe_dump(cfg))
    logged_config = model_config.parent / "eval_dcn_matched_stats_wd3e6_seed42.yaml"
    logged_config.write_text(Path(manifest["logged_config"]).read_text())
    old_latest = root / "artifacts/esmm_seed42/latest.json"
    old_latest.parent.mkdir(parents=True)
    old_latest.write_text('{"run": "existing_esmm_baseline"}')
    calls = []
    monkeypatch.setattr(study_cli, "CODE", code)
    monkeypatch.setattr(study_cli, "ROOT", root)
    monkeypatch.setattr(study_cli, "prepare", lambda *args, **kwargs: calls.append(args) or args[0])
    study_cli.main([])
    selected_study, selected_config, selected_logged = calls[0]
    assert selected_study.parent == root / "artifacts/esmm_dcn_studies"
    assert selected_config == model_config
    assert selected_logged == logged_config
    assert json.loads(old_latest.read_text())["run"] == "existing_esmm_baseline"


def test_canonical_esmm_dcn_rejects_a_branch_specific_optimizer(prepared):
    study, _, _ = prepared
    manifest = study_cli._manifest(study)
    cfg = yaml.safe_load(Path(manifest["jobs"][0]["config"]).read_text())
    cfg["model"] = "esmm_dcn"
    cfg["training"]["dcn_optimizer"] = {"lr": .0005}
    with pytest.raises(ValueError, match="retains the ESMM optimizer"):
        study_cli._check_config(cfg)


def test_prepare_continuation_seeds_maps_array_indices_through_manifest(prepared, monkeypatch):
    study, root, code = prepared
    original = study_cli._manifest(study)
    seeds = (46, 47, 48, 49, 50, 51)
    continuation = study_cli.prepare(root / "continuation", study / "configs/base_train.yaml",
                                     original["logged_config"], code=code, root=root, seeds=seeds)
    manifest = study_cli._manifest(continuation)
    assert manifest["seeds"] == list(seeds)
    assert [job["seed"] for job in manifest["jobs"]] == list(seeds)
    assert "--array=0-5%4" in (continuation / "train.sbatch").read_text()
    assert "42 +" not in (continuation / "train.sbatch").read_text()
    calls = []
    monkeypatch.setattr(study_cli, "run_seed", lambda location, seed: calls.append((location, seed)))
    study_cli.run_index(continuation, 0)
    study_cli.run_index(continuation, 5)
    assert calls == [(continuation, 46), (continuation, 51)]
    with pytest.raises(ValueError, match="outside study jobs"):
        study_cli.run_index(continuation, -1)
    with pytest.raises(ValueError, match="outside study jobs"):
        study_cli.run_index(continuation, 6)


@pytest.mark.parametrize("seeds", [(), (42,), (42, 42), (-1, 42), (True, 42), (42, 43.0)])
def test_prepare_rejects_invalid_seed_lists_before_creating_study(prepared, seeds):
    study, root, code = prepared
    manifest = study_cli._manifest(study)
    target = root / "invalid"
    with pytest.raises(ValueError, match="at least two unique nonnegative integers"):
        study_cli.prepare(target, study / "configs/base_train.yaml", manifest["logged_config"],
                          code=code, root=root, seeds=seeds)
    assert not target.exists()


def test_cli_passes_explicit_seeds_to_prepare(prepared, monkeypatch):
    study, _, _ = prepared
    calls = []
    monkeypatch.setattr(study_cli, "prepare", lambda *args, **kwargs: calls.append(kwargs) or study)
    study_cli.main(["--config", str(study / "configs/base_train.yaml"),
                    "--seeds", "46", "47", "48", "49", "50", "51"])
    assert calls[0]["seeds"] == [46, 47, 48, 49, 50, 51]


def test_run_seed_uses_private_registry_then_immutable_record(prepared, monkeypatch):
    study, root, _ = prepared
    manifest = study_cli._manifest(study)
    job = manifest["jobs"][0]
    run = root / "artifacts" / job["arm"] / "exact_run"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(Path(job["config"]).read_text())
    write_json(run / "run.json", {"status": "FINISHED"})
    Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": "exact_run"}}))
    write_json(run.parent / "latest.json", {"run": "unrelated_newer_run"})
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(study_cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert study_cli.run_seed(study, 42) == run
    assert len(calls) == 1 and calls[0][3:5] == ["--run-dir", str(run)]
    Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": "another_run"}}))
    assert study_cli.run_seed(study, 42) == run
    assert json.loads(Path(job["run_record"]).read_text())["run_dir"] == str(run)


@pytest.mark.parametrize("available, count", [(False, 0), (True, 2)])
def test_run_seed_requires_exactly_one_cuda_gpu(prepared, monkeypatch, available, count):
    study, _, _ = prepared
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(RuntimeError, match="exactly one visible CUDA GPU"):
        study_cli.run_seed(study, 42)
    assert not list(study.glob("run_seed*.json"))


def test_submit_records_array_and_afterok_report(prepared, monkeypatch):
    study, _, _ = prepared
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "123;cluster\n" if len(calls) == 1 else "124\n")

    monkeypatch.setattr(study_cli.subprocess, "run", sbatch)
    assert study_cli.submit(study) == ("123", "124")
    assert calls[1][2] == "--dependency=afterok:123"
    assert json.loads((study / "submission_jobs.json").read_text())["report_job_id"] == "124"
    with pytest.raises(FileExistsError):
        study_cli.submit(study)
    assert len(calls) == 2


@pytest.fixture
def finished(prepared):
    study, root, _ = prepared
    return finish_study(study, root)


def finish_study(study, root):
    manifest = study_cli._manifest(study)
    for job in manifest["jobs"]:
        run = root / "artifacts" / job["arm"] / "r1"
        (run / "metrics").mkdir(parents=True)
        (run / "config.yaml").write_text(Path(job["config"]).read_text())
        (run / "best.pt").write_bytes(b"checkpoint")
        write_json(run / "run.json", {"status": "FINISHED", "selected_on": {
            "metric": "ctcvr_ne", "split": "val", "epoch": 2, "value": .8}})
        (run / "history.csv").write_text(
            "epoch,val_score,val_ctcvr_ne\n1,1.2,1.2\n2,.8,.8\n3,.8,.8\n4,.9,.9\n5,1.1,1.1\n")
        rows = [{"split": "test", "population": pop, "rows": 20 if pop == "pCVR" else 100,
                 "positives": 20 if pop == "CTR" else 3,
                 "auc": job["seed"] / 100, "pr_auc": .01, "ne": .9, "ce": 1.}
                for pop in study_cli.POPULATIONS]
        write_json(run / "metrics/test.json", rows[:3])
        target = run / "metrics_logged/teacher42"
        target.mkdir(parents=True)
        write_json(target / "test.json", [dict(rows[3], pctr_model="teacher42")])
        (target / "config.yaml").write_text(Path(manifest["logged_config"]).read_text())
        write_json(target / "provenance.json", {
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "pctr_model": "teacher42", "pctr_scores_sha256": "scores_hash",
            "logged_pctr": {"dcn_run": "artifacts/teacher/run1"}})
        write_json(Path(job["run_record"]), {"seed": job["seed"], "arm": job["arm"],
                                              "run_dir": str(run),
                                              "config_sha256": study_cli._hash(run / "config.yaml")})
    return study, root, manifest


def test_ten_seed_report_requires_every_requested_seed_and_computes_sem(prepared):
    old_study, root, code = prepared
    seeds = tuple(range(42, 52))
    study = study_cli.prepare(root / "ten_seeds", old_study / "configs/base_train.yaml",
                             study_cli._manifest(old_study)["logged_config"],
                             code=code, root=root, seeds=seeds)
    _, _, manifest = finish_study(study, root)
    # A ten-seed study cannot silently report its first nine finished runs.
    missing = manifest["jobs"].pop()
    write_json(study / "submission.json", manifest)
    with pytest.raises(ValueError, match="exactly manifest seeds 42–51"):
        study_cli.report(study)
    assert not (study / "seed_summary.json").exists()
    manifest["jobs"].append(missing)
    write_json(study / "submission.json", manifest)
    path = study_cli.report(study)
    summary = json.loads((study / "seed_summary.json").read_text())
    assert len(summary["per_seed"]) == 40
    assert [source["seed"] for source in summary["sources"]] == list(seeds)
    assert all(row["n"] == 10 for row in summary["summary"])
    auc = next(row for row in summary["summary"] if row["metric"] == "auc")
    assert auc["mean"] == pytest.approx(.465)
    assert auc["sem"] == pytest.approx(.009574271077563381)
    assert "seeds 42–51 (n=10)" in path.read_text()


def test_report_aggregates_all_four_populations_without_latest_pointers(finished):
    study, _, _ = finished
    path = study_cli.report(study)
    result = json.loads((study / "seed_summary.json").read_text())
    assert len(result["per_seed"]) == 16 and len(result["summary"]) == 16
    variant = study_cli._manifest(study)["variant"]
    assert result["variant"] == variant
    assert {row["variant"] for row in result["summary"]} == {variant}
    assert {row["variant"] for row in result["sources"]} == {variant}
    assert [row["seed"] for row in result["sources"]] == [42, 43, 44, 45]
    auc = next(row for row in result["summary"] if row["metric"] == "auc")
    assert auc["values"] == [.42, .43, .44, .45]
    assert auc["mean"] == pytest.approx(.435)
    assert auc["sem"] == pytest.approx(.006454972243679028)
    assert "native CTCVR" in path.read_text()
    with (study / "per_seed.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 16


@pytest.mark.parametrize("problem, message", [
    ("unfinished", "run is RUNNING"),
    ("wrong_epoch", "minimum validation"),
    ("wrong_selection_metric", "minimum validation"),
    ("short_history", "five epochs"),
    ("teacher", "teacher identity differs"),
    ("census", "census inconsistent"),
    ("missing_native", "populations missing"),
    ("seed", "run record seed differs"),
])
def test_report_rejects_protocol_violations_without_partial_summary(finished, problem, message):
    study, _, manifest = finished
    job = manifest["jobs"][1]
    run = Path(json.loads(Path(job["run_record"]).read_text())["run_dir"])
    if problem in {"unfinished", "wrong_epoch", "wrong_selection_metric"}:
        path = run / "run.json"
        data = json.loads(path.read_text())
        if problem == "unfinished":
            data["status"] = "RUNNING"
        elif problem == "wrong_epoch":
            data["selected_on"]["epoch"] = 3  # First minimum must win ties.
        else:
            data["selected_on"]["metric"] = "logged_ctcvr_ne"
    elif problem == "short_history":
        path = run / "history.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif problem == "teacher":
        path = run / "metrics_logged/teacher42/provenance.json"
        data = json.loads(path.read_text())
        data["pctr_scores_sha256"] = "different_teacher_scores"
    elif problem in {"census", "missing_native"}:
        path = run / "metrics/test.json"
        data = json.loads(path.read_text())
        if problem == "census":
            data[0]["rows"] += 1
        else:
            data.pop()
    else:
        path = Path(job["run_record"])
        data = json.loads(path.read_text())
        data["seed"] = 999
    if problem != "short_history":
        write_json(path, data)
    with pytest.raises(ValueError, match=message):
        study_cli.report(study)
    assert not (study / "seed_summary.json").exists()


def test_report_rejects_manifests_without_the_model_identity(finished):
    study, _, manifest = finished
    manifest.pop("model")
    manifest.pop("variant")
    write_json(study / "submission.json", manifest)
    with pytest.raises(ValueError, match="study model and variant must identify esmm_dcn"):
        study_cli.report(study)
