import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_paper_seed_study as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reference_config():
    return {
        "model": "esmm_logged_pctr",
        "training": {"seed": 42, "epochs": 5, "batch_size": 65536,
                     "lr": .001, "gpus": -1, "evaluate_test": True,
                     "load_test": False, "out_dir": "artifacts_downsampled",
                     "checkpoint_metric": "logged_ctcvr_ne",
                     "keep_checkpoint": "best", "early_stopping": {"enabled": False}},
        "model_params": {"emb_dim": 20, "tower_dims": [460, 360, 200, 80, 2],
                         "objective": "kd_tau", "alpha": .5, "aux_weight": 1.0},
        "data": {"encoded_dir": "ali-ccp/encoded",
                 "nonclick_sampling": {"rate": .05, "seed": 2026}},
        "evaluation": {"is_logged_ctcvr": True},
        "logged_pctr": {"dcn_run": runner.LEGACY_TEACHER, "score_column": "score",
                        "predictions": {
                            split: f"{runner.LEGACY_TEACHER}/predictions/{split}.parquet"
                            for split in ("train", "val", "test")}},
    }


def test_training_config_changes_student_seed_without_changing_sampling_or_recipe(reference_config):
    before = copy.deepcopy(reference_config)
    config = runner.training_config(reference_config, 51)
    assert reference_config == before
    assert config["training"] == {**before["training"], "seed": 51,
                                  "evaluate_test": False, "gpus": 1}
    assert config["data"] == before["data"]
    assert config["data"]["nonclick_sampling"]["seed"] == 2026
    assert config["model_params"] == before["model_params"]
    assert config["evaluation"] == before["evaluation"]
    assert config["model_params"]["objective"] == "kd_tau"
    assert config["model_params"]["alpha"] == .5
    # Mutating the generated recipe cannot change the pinned reference recipe.
    config["data"]["nonclick_sampling"]["rate"] = .25
    assert reference_config == before


def test_training_teacher_is_canonical_and_test_scores_are_reserved_for_separate_evaluation(reference_config):
    config = runner.training_config(reference_config, 46)
    assert config["logged_pctr"]["dcn_run"] == runner.TEACHER
    assert config["logged_pctr"]["predictions"] == {
        split: f"{runner.TEACHER}/predictions/{split}.parquet" for split in ("train", "val")}
    assert config["training"]["evaluate_test"] is False
    assert config["training"]["load_test"] is False


def test_vanilla_esmm_training_does_not_load_a_logged_teacher(reference_config):
    reference_config["model"] = "esmm"
    reference_config["training"]["checkpoint_metric"] = "ctcvr_ne"
    config = runner.training_config(reference_config, 46)
    assert "logged_pctr" not in config
    assert "evaluation" not in config
    assert config["training"]["checkpoint_metric"] == "ctcvr_ne"
    assert config["data"]["nonclick_sampling"] == {"rate": .05, "seed": 2026}


def test_existing_run_uses_finished_run_and_rejects_multiple_finished_runs(tmp_path):
    arm = tmp_path / "artifacts/esmm_dcn_seed46"
    write_json(arm / "unfinished/run.json", {"status": "RUNNING"})
    assert runner.existing_run(tmp_path, "artifacts", "esmm_dcn", 46, "original") is None
    write_json(arm / "finished/run.json", {"status": "FINISHED"})
    assert runner.existing_run(tmp_path, "artifacts", "esmm_dcn", 46, "original") == arm / "finished"
    write_json(arm / "another/run.json", {"status": "FINISHED"})
    with pytest.raises(ValueError, match="ambiguous existing runs"):
        runner.existing_run(tmp_path, "artifacts", "esmm_dcn", 46, "original")


@pytest.mark.parametrize("base, directory", [("esmm_dcn", "esmm_dcn_seed42"),
                                             ("esmm_logged_pctr_kd05", "esmm_logged_pctr_kd05")])
def test_original_seed42_is_pinned_to_paper_run_instead_of_latest(tmp_path, base, directory):
    arm = tmp_path / "artifacts" / directory
    write_json(arm / "original/run.json", {"status": "FINISHED"})
    write_json(arm / "newer/run.json", {"status": "FINISHED"})
    write_json(arm / "latest.json", {"run": "newer"})
    assert runner.existing_run(tmp_path, "artifacts", base, 42, "original") == arm / "original"


def test_original_seed42_must_be_finished(tmp_path):
    write_json(tmp_path / "artifacts/esmm_dcn_seed42/original/run.json", {"status": "RUNNING"})
    with pytest.raises(ValueError, match="original paper run is unfinished"):
        runner.existing_run(tmp_path, "artifacts", "esmm_dcn", 42, "original")


@pytest.fixture
def task_study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    source = study / "source/code"
    source.mkdir(parents=True)
    (source / "tracked.py").write_text("SNAPSHOT = True\n")
    config = study / "train.yaml"
    config.write_text(yaml.safe_dump({"model": "esmm_dcn", "training": {
        "seed": 46, "out_dir": "artifacts", "evaluate_test": False}}))
    logged_config = study / "logged_eval.yaml"
    logged_config.write_text("pctr_model: fixed_teacher\n")
    job = {"variant": "esmm_dcn", "seed": 46, "arm": "esmm_dcn_seed46",
           "task": 0, "config": str(config), "config_sha256": runner.digest(config),
           "registry": str(study / "private.yaml"), "run_record": str(study / "record.json")}
    manifest = {"source_code": str(source), "project_root": str(tmp_path), "python": "/python",
                "source_sha256": {"tracked.py": runner.digest(source / "tracked.py")},
                "logged_config": str(logged_config), "logged_config_sha256": runner.digest(logged_config),
                "jobs": [job]}
    write_json(study / "manifest.json", manifest)
    run = tmp_path / "artifacts" / job["arm"] / "selected_run"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(config.read_text())
    write_json(run / "run.json", {"status": "FINISHED"})
    write_json(run.parent / "latest.json", {"run": "some_other_run"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    return study, manifest, job, run


def test_existing_private_registry_resumes_without_retraining_and_record_pins_exact_run(task_study, monkeypatch):
    study, manifest, job, run = task_study
    registry = Path(job["registry"])
    registry.write_text(yaml.safe_dump({job["arm"]: {"run": run.name}}))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    runner.run_task(study, 0)
    assert len(calls) == 1
    assert calls[0][2] == str(Path(manifest["source_code"]) / "scripts/evaluate_run.py")
    assert calls[0][3:5] == ["--run-dir", str(run)]
    assert calls[0][-2:] == ["--logged-config", manifest["logged_config"]]
    assert runner.read(job["run_record"])["run_dir"] == str(run)
    registry.write_text(yaml.safe_dump({job["arm"]: {"run": "later_run"}}))
    runner.run_task(study, 0)
    assert len(calls) == 2 and calls[1][3:5] == ["--run-dir", str(run)]


def test_new_task_trains_with_private_registry_then_evaluates_that_exact_run(task_study, monkeypatch):
    study, manifest, job, run = task_study
    calls = []

    def execute(command, **kwargs):
        assert kwargs["cwd"] == Path(manifest["source_code"])
        assert kwargs["check"] is True
        calls.append(command)
        if command[2].endswith("/train.py"):
            assert command[-2:] == ["--best-models", job["registry"]]
            Path(job["registry"]).write_text(yaml.safe_dump({job["arm"]: {"run": run.name}}))

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.run_task(study, 0)
    assert len(calls) == 2
    assert calls[0][3:5] == ["--config", job["config"]]
    assert calls[1][3:5] == ["--run-dir", str(run)]


def test_private_registry_cannot_contain_an_unrelated_arm(task_study, monkeypatch):
    study, _, job, run = task_study
    Path(job["registry"]).write_text(yaml.safe_dump({
        job["arm"]: {"run": run.name}, "unrelated": {"run": "other"}}))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="private registry"):
        runner.run_task(study, 0)
    assert not calls and not Path(job["run_record"]).exists()


@pytest.mark.parametrize("task", [-1, 1])
def test_invalid_task_index_cannot_select_another_run(task_study, monkeypatch, task):
    study, _, _, _ = task_study
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="invalid task index"):
        runner.run_task(study, task)
    assert not calls


@pytest.mark.parametrize("problem, message", [
    ("seed", "run record does not match"),
    ("arm", "run record does not match"),
    ("status", "recorded run is unfinished"),
    ("config", "recorded run configuration differs"),
])
def test_resume_rejects_wrong_or_changed_recorded_run(task_study, monkeypatch, problem, message):
    study, _, job, run = task_study
    saved = {"seed": job["seed"], "arm": job["arm"], "run_dir": str(run)}
    if problem == "seed":
        saved["seed"] += 1
    elif problem == "arm":
        saved["arm"] = "another_arm"
    elif problem == "status":
        write_json(run / "run.json", {"status": "RUNNING"})
    else:
        (run / "config.yaml").write_text("model: different_model\n")
    write_json(Path(job["run_record"]), saved)
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match=message):
        runner.run_task(study, 0)
    assert not calls


@pytest.mark.parametrize("changed, message", [("source", "source changed"),
                                               ("config", "configuration changed"),
                                               ("logged", "logged evaluation config changed")])
def test_mutated_snapshot_or_config_is_rejected_before_starting_task(task_study, monkeypatch, changed, message):
    study, manifest, job, _ = task_study
    path = {"source": Path(manifest["source_code"]) / "tracked.py",
            "config": Path(job["config"]), "logged": Path(manifest["logged_config"])}[changed]
    path.write_text(path.read_text() + "\n# altered\n")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match=message):
        runner.run_task(study, 0)
    assert not calls


@pytest.mark.parametrize("missing_jobs", [True, False])
def test_submit_records_gpu_cpu_and_report_dependencies(tmp_path, monkeypatch, missing_jobs):
    study = tmp_path / "study"
    manifest = {"jobs": [{"seed": 46}] if missing_jobs else []}
    write_json(study / "manifest.json", manifest)
    verified, calls = [], []
    monkeypatch.setattr(runner, "verify", lambda value: verified.append(value))
    monkeypatch.setattr(runner.sys, "argv", ["run_paper_seed_study.py", "--submit", str(study)])

    def sbatch(command, **kwargs):
        assert kwargs == {"text": True}
        calls.append(command)
        if len(calls) == 2:
            assert runner.read(study / "submission_jobs.json") == {
                "train_job_id": "123", "state": "training_submitted"}
        elif len(calls) == 3:
            assert runner.read(study / "submission_jobs.json") == {
                "train_job_id": "123", "score_job_id": "124", "state": "scoring_submitted"}
        return f"{122 + len(calls)};cluster\n"

    monkeypatch.setattr(runner.subprocess, "check_output", sbatch)
    runner.main()
    assert verified == [manifest]
    expected = {"state": "submitted", "report_job_id": "125" if missing_jobs else "123"}
    if missing_jobs:
        expected["train_job_id"] = "123"
        expected["score_job_id"] = "124"
        assert calls == [["sbatch", "--parsable", str(study / "train.sbatch")],
                         ["sbatch", "--parsable", "--dependency=aftercorr:123", str(study / "score.sbatch")],
                         ["sbatch", "--parsable", "--dependency=afterok:124", str(study / "report.sbatch")]]
    else:
        assert calls == [["sbatch", "--parsable", str(study / "report.sbatch")]]
    assert runner.read(study / "submission_jobs.json") == expected
    with pytest.raises(FileExistsError):
        runner.main()
    assert len(calls) == (3 if missing_jobs else 1)  # Repeated submission cannot duplicate jobs.


@pytest.mark.parametrize("all_reused", [True, False])
def test_prepare_separates_gpu_and_cpu_arrays_and_reused_study_has_only_report(tmp_path, monkeypatch, all_reused):
    code = tmp_path / "code"
    code.mkdir()
    (code / "snapshot.py").write_text("SOURCE = True\n")
    teacher_scores = tmp_path / runner.TEACHER / "predictions/test.parquet"
    teacher_scores.parent.mkdir(parents=True)
    teacher_scores.write_bytes(b"fixed teacher scores")
    baseline = tmp_path / "artifacts/esmm_dcn_seed42/original"
    write_json(baseline / "run.json", {"status": "FINISHED"})
    (baseline / "config.yaml").write_text(yaml.safe_dump({
        "model": "esmm_dcn", "training": {"seed": 42, "out_dir": "artifacts",
                                          "checkpoint_metric": "ctcvr_ne"}}))
    if all_reused:
        write_json(tmp_path / "artifacts/esmm_dcn_seed43/reused/run.json", {"status": "FINISHED"})
    monkeypatch.setattr(runner, "SEEDS", [42, 43])
    monkeypatch.setattr(runner, "RECIPES", [
        ("esmm_dcn", "ESMM + DCN", "full", "artifacts", "esmm_dcn", "original")])

    def git(_code, *arguments):
        return {("rev-parse", "HEAD"): "fixture_commit\n",
                ("ls-files", "-z"): "snapshot.py\0",
                ("ls-files", "--others", "--exclude-standard", "-z"): "",
                ("diff", "HEAD"): ""}[arguments]

    monkeypatch.setattr(runner, "git", git)
    study = tmp_path / "study"
    manifest = runner.prepare(study, root=tmp_path, code=code, python="/example/python")
    report = (study / "report.sbatch").read_text()
    assert "--gres" not in report and "--gpus" not in report
    assert "report_paper_ten_seeds.py" in report
    assert f"--study-dir {study}" in report
    assert f"--project-root {tmp_path}" in report
    assert len(manifest["jobs"]) == (0 if all_reused else 1)
    assert (study / "train.sbatch").exists() is (not all_reused)
    assert (study / "score.sbatch").exists() is (not all_reused)
    if not all_reused:
        train = (study / "train.sbatch").read_text()
        score = (study / "score.sbatch").read_text()
        assert "--array=0-0%4" in train and "--gpu-task" in train
        assert "run_paper_seed_phase.py" in train
        assert "--array=0-0%8" in score and "--cpu-task" in score
        assert "run_paper_seed_phase.py" in score
        assert "--gres" not in score and "--gpus" not in score
        assert manifest["jobs"][0]["seed"] == 43
