import json
from pathlib import Path
import shutil

import pytest
import torch
import yaml

from scripts import run_paper_seed_phase as phase
from scripts import run_paper_seed_study as original_runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study_task(tmp_path, monkeypatch):
    study = tmp_path / "study"
    source = study / "source/code"
    (source / "scripts").mkdir(parents=True)
    frozen_runner = source / "scripts/run_paper_seed_study.py"
    shutil.copyfile(original_runner.__file__, frozen_runner)
    config = study / "train.yaml"
    config.write_text(yaml.safe_dump({"model": "esmm_logged_pctr", "training": {
        "seed": 46, "out_dir": "artifacts", "evaluate_test": False}}))
    logged = study / "logged_eval.yaml"
    logged.write_text("pctr_model: fixed_teacher\n")
    job = {"task": 0, "seed": 46, "arm": "esmm_logged_pctr_kd03_seed46",
           "config": str(config), "config_sha256": phase._digest(config),
           "run_record": str(study / "runs/seed46.json"), "registry": str(study / "private.yaml")}
    (study / "runs").mkdir()
    manifest = {"project_root": str(tmp_path), "python": "/original/python", "source_code": str(source),
                "source_sha256": {"scripts/run_paper_seed_study.py": phase._digest(frozen_runner)},
                "logged_config": str(logged), "logged_config_sha256": phase._digest(logged), "jobs": [job]}
    write_json(study / "manifest.json", manifest)
    run = tmp_path / "artifacts" / job["arm"] / "exact_run"
    write_json(run / "run.json", {"status": "FINISHED"})
    (run / "config.yaml").write_text(config.read_text())
    write_json(run.parent / "latest.json", {"run": "other_run"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    return study, manifest, job, run


def record_run(job, run):
    write_json(Path(job["run_record"]), {"seed": job["seed"], "arm": job["arm"], "run_dir": str(run)})


def cache_run(run):
    (run / "predictions").mkdir()
    (run / "predictions/test.parquet").write_bytes(b"validated predictions")
    write_json(run / "predictions/test.provenance.json", {"identity": "validated by frozen evaluator"})


def test_gpu_phase_trains_original_recipe_then_exports_without_metric_scoring(study_task, monkeypatch):
    study, manifest, job, run = study_task
    calls = []

    def execute(command, **kwargs):
        assert kwargs == {"cwd": Path(manifest["source_code"]), "check": True}
        calls.append(command)
        if command[2].endswith("/train.py"):
            Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": run.name}}))

    monkeypatch.setattr(phase.subprocess, "run", execute)
    assert phase.gpu_task(study, 0) == run
    assert calls[0] == [manifest["python"], "-u", str(Path(manifest["source_code"]) / "scripts/train.py"),
                        "--config", job["config"], "--best-models", job["registry"]]
    assert calls[1][:3] == [manifest["python"], "-u", "-c"]
    assert "sys.path.insert(0, sys.argv[1])" in calls[1][3]
    assert "from evaluation.predictions import test_predictions" in calls[1][3]
    assert "native_metrics" not in calls[1][3] and "evaluate_native" not in calls[1][3]
    assert calls[1][4:] == [manifest["source_code"], str(run), manifest["project_root"]]
    assert json.loads(Path(job["run_record"]).read_text())["run_dir"] == str(run)


@pytest.mark.parametrize("existing_record", [False, True])
def test_gpu_resume_never_retrains_or_follows_latest_registry(study_task, monkeypatch, existing_record):
    study, _, job, run = study_task
    Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": run.name}}))
    if existing_record:
        record_run(job, run)
        Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": "newer_run"}}))
    calls = []
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert phase.gpu_task(study, 0) == run
    assert len(calls) == 1 and calls[0][2] == "-c"
    assert calls[0][-2] == str(run)


def test_cpu_scores_exact_run_with_original_source_and_explicit_cpu_device(study_task, monkeypatch):
    study, manifest, job, run = study_task
    record_run(job, run)
    cache_run(run)
    calls = []
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert phase.cpu_task(study, 0) == run
    assert calls == [([manifest["python"], "-u", str(Path(manifest["source_code"]) / "scripts/evaluate_run.py"),
                      "--run-dir", str(run), "--gpus", "0", "--logged-config", manifest["logged_config"]],
                     {"cwd": Path(manifest["source_code"]), "check": True})]


@pytest.mark.parametrize("missing", ["test.parquet", "test.provenance.json"])
def test_cpu_refuses_inference_when_prediction_cache_is_incomplete(study_task, monkeypatch, missing):
    study, _, job, run = study_task
    record_run(job, run)
    cache_run(run)
    (run / "predictions" / missing).unlink()
    calls = []
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="CPU inference is disabled"):
        phase.cpu_task(study, 0)
    assert not calls


@pytest.mark.parametrize("problem, message", [("seed", "record does not match"),
                                               ("config", "configuration differs"),
                                               ("status", "run is unfinished")])
def test_resume_rejects_wrong_or_changed_run(study_task, monkeypatch, problem, message):
    study, _, job, run = study_task
    record_run(job, run)
    if problem == "seed":
        saved = json.loads(Path(job["run_record"]).read_text())
        saved["seed"] += 1
        write_json(Path(job["run_record"]), saved)
    elif problem == "config":
        (run / "config.yaml").write_text("model: incorrect\n")
    else:
        write_json(run / "run.json", {"status": "RUNNING"})
    calls = []
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match=message):
        phase.gpu_task(study, 0)
    assert not calls


def test_frozen_verification_rejects_modified_training_config(study_task, monkeypatch):
    study, _, job, _ = study_task
    Path(job["config"]).write_text("model: changed\n")
    calls = []
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="configuration changed"):
        phase.gpu_task(study, 0)
    assert not calls


@pytest.mark.parametrize("available, count", [(False, 0), (True, 2)])
def test_gpu_phase_requires_exactly_one_visible_gpu(study_task, monkeypatch, available, count):
    study, _, _, _ = study_task
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(RuntimeError, match="exactly one visible CUDA GPU"):
        phase.gpu_task(study, 0)
