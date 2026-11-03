import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_direct_product_study as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def original_study(tmp_path):
    baseline = tmp_path / "artifacts/direct_product_studies/20261015T235815Z"
    baseline.mkdir(parents=True)
    teacher = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"
    for split in ("train", "val", "test"):
        scores = tmp_path / teacher / "predictions" / f"{split}.parquet"
        scores.parent.mkdir(parents=True, exist_ok=True)
        scores.write_bytes(f"fixed {split} teacher scores".encode())
    reference = {
        "model": "esmm_direct_product", "model_params": {"emb_dim": 20, "tower_dims": [460, 360, 200, 80, 2]},
        "training": {"seed": 42, "epochs": 5, "batch_size": 65536, "eval_batch_size": 65536,
                     "lr": .001, "gpus": 1, "evaluate_test": False, "load_test": False,
                     "out_dir": "artifacts", "checkpoint_metric": "logged_ctcvr_ne", "keep_checkpoint": "best",
                     "early_stopping": {"enabled": False}, "save_predictions": False},
        "data": {"encoded_dir": "ali-ccp/encoded", "csv_dir": "ali-ccp/extracted"},
        "evaluation": {"is_logged_ctcvr": True},
        "logged_pctr": {"dcn_run": teacher, "score_column": "score", "predictions": {
            split: f"{teacher}/predictions/{split}.parquet" for split in ("train", "val", "test")}},
    }
    rows, runs = [], {}
    for seed in range(42, 46):
        run = tmp_path / "artifacts" / f"esmm_direct_product_seed{seed}" / "pinned_original"
        config = copy.deepcopy(reference)
        config["training"]["seed"] = seed
        write_json(run / "run.json", {"status": "FINISHED", "selected_on": {
            "metric": "logged_ctcvr_ne", "split": "val", "epoch": 1, "value": .8}})
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        (run / "history.csv").write_text("epoch,val_score\n1,.8\n2,.9\n3,1.0\n4,1.1\n5,1.2\n")
        registry = {f"esmm_direct_product_seed{seed}": {"run": run.name}}
        (baseline / f"best_models_seed{seed}.yaml").write_text(yaml.safe_dump(registry))
        rows.extend({"seed": seed, "run": str(run), "population": population}
                    for population in ("pCVR", "CTCVR-logged"))
        runs[seed] = run
    write_json(baseline / "seed_summary.json", {"per_seed": rows})
    # The older unsuffixed seed-42 arm and mutable latest pointers must not win.
    write_json(tmp_path / "artifacts/esmm_direct_product/older/run.json", {"status": "FINISHED"})
    for run in runs.values():
        write_json(run.parent / "later/run.json", {"status": "FINISHED"})
        write_json(run.parent / "latest.json", {"run": "later"})
    code = tmp_path / "code"
    code.mkdir()
    (code / "tracked.py").write_text("FROZEN = True\n")
    return baseline, tmp_path, code, reference, runs


@pytest.fixture
def prepared(original_study, monkeypatch):
    baseline, root, code, reference, runs = original_study
    (code / "untracked.py").write_text("UNTRACKED = True\n")

    def git(_code, *arguments):
        return {("rev-parse", "HEAD"): "fixture_commit\n",
                ("ls-files", "-z"): "tracked.py\0",
                ("ls-files", "--others", "--exclude-standard", "-z"): "untracked.py\0",
                ("diff", "HEAD"): "fixture working diff\n"}[arguments]

    monkeypatch.setattr(runner, "git", git)
    study = root / "new_study"
    manifest = runner.prepare(study, root=root, code=code,
                              python="/original/python", baseline_study=baseline)
    return study, manifest, root, code, reference, runs


def test_original_runs_are_pinned_by_summary_and_private_registries(original_study):
    baseline, root, _, _, runs = original_study
    assert runner.pinned_original_runs(baseline, root) == runs
    assert runner.pinned_original_runs(baseline, root)[42].parent.name == "esmm_direct_product_seed42"


def test_original_summary_cannot_identify_two_runs_for_one_seed(original_study):
    baseline, root, _, _, runs = original_study
    path = baseline / "seed_summary.json"
    summary = json.loads(path.read_text())
    summary["per_seed"][0]["run"] = str(runs[42].parent / "later")
    write_json(path, summary)
    with pytest.raises(ValueError, match="conflicting runs"):
        runner.pinned_original_runs(baseline, root)


def test_original_private_registry_must_agree_with_recorded_summary(original_study):
    baseline, root, _, _, _ = original_study
    (baseline / "best_models_seed42.yaml").write_text(yaml.safe_dump({
        "esmm_direct_product_seed42": {"run": "later"}}))
    with pytest.raises(ValueError, match="private registry disagrees"):
        runner.pinned_original_runs(baseline, root)


def test_prepare_preserves_four_runs_and_trains_only_six_new_seeds(prepared):
    study, manifest, root, _, reference, runs = prepared
    assert manifest["seeds"] == list(range(42, 52))
    assert len(manifest["runs"]) == 10
    assert [job["seed"] for job in manifest["jobs"]] == list(range(46, 52))
    assert [job["task"] for job in manifest["jobs"]] == list(range(6))
    assert {item["seed"]: Path(item["run_dir"]) for item in manifest["runs"][:4]} == runs
    for job in manifest["jobs"]:
        config = yaml.safe_load(Path(job["config"]).read_text())
        expected = copy.deepcopy(reference)
        expected["training"]["seed"] = job["seed"]
        expected["logged_pctr"]["predictions"].pop("test")
        assert config == expected
        assert job["config_sha256"] == runner.digest(job["config"])
        assert job["run_dir"] is None
        assert not Path(job["registry"]).exists()
    teacher = manifest["teacher"]
    assert set(teacher["predictions_sha256"]) == {"train", "val", "test"}
    for split, path in teacher["predictions"].items():
        assert teacher["predictions_sha256"][split] == runner.digest(path)
    logged = yaml.safe_load(Path(manifest["logged_config"]).read_text())
    assert set(logged["logged_pctr"]["predictions"]) == {"test"}
    assert logged["logged_pctr"]["dcn_run"] == runner.TEACHER
    train, score, report = [(study / name).read_text() for name in (
        "train.sbatch", "score.sbatch", "report.sbatch")]
    assert "--array=0-5%4" in train and "--gres=gpu:1" in train and "--gpu-task" in train
    assert "--array=0-5%8" in score and "--cpu-task" in score and "--gres" not in score
    assert "report_direct_product_ten_seeds.py" in report and "--gres" not in report
    assert f"--project-root {root}" in report


def test_source_snapshot_includes_untracked_files_and_is_independent_of_later_edits(prepared):
    study, manifest, _, code, _, _ = prepared
    snapshot = Path(manifest["source_code"])
    assert (snapshot / "untracked.py").read_text() == "UNTRACKED = True\n"
    assert (study / "source.diff").read_text() == "fixture working diff\n"
    (code / "tracked.py").write_text("LATER_WORKING_CHANGE = True\n")
    runner.verify(manifest)
    (snapshot / "tracked.py").write_text("ALTERED_SNAPSHOT = True\n")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify(manifest)


def test_preparation_cannot_overwrite_an_existing_study(prepared):
    study, _, root, code, _, _ = prepared
    with pytest.raises(FileExistsError):
        runner.prepare(study, root=root, code=code)


def test_submit_gpu_to_cpu_aftercorr_then_report_afterok_and_prevents_duplicates(prepared, monkeypatch):
    study, _, _, _, _, _ = prepared
    calls = []

    def sbatch(command, **kwargs):
        assert kwargs == {"text": True}
        calls.append(command)
        return f"{700 + len(calls)};cluster\n"

    monkeypatch.setattr(runner.subprocess, "check_output", sbatch)
    assert runner.submit(study) == {"train_job_id": "701", "score_job_id": "702", "report_job_id": "703"}
    assert calls == [["sbatch", "--parsable", str(study / "train.sbatch")],
                     ["sbatch", "--parsable", "--dependency=aftercorr:701", str(study / "score.sbatch")],
                     ["sbatch", "--parsable", "--dependency=afterok:702", str(study / "report.sbatch")]]
    assert json.loads((study / "submission_jobs.json").read_text()) == {
        "train_job_id": "701", "score_job_id": "702", "report_job_id": "703", "state": "submitted"}
    with pytest.raises(FileExistsError):
        runner.submit(study)
    assert len(calls) == 3


def test_changed_training_teacher_scores_prevent_submission(prepared, monkeypatch):
    study, manifest, _, _, _, _ = prepared
    Path(manifest["teacher"]["predictions"]["train"]).write_bytes(b"changed teacher")
    calls = []
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="fixed teacher train predictions changed"):
        runner.submit(study)
    assert not calls and not (study / "submission_jobs.json").exists()
