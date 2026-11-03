"""Reject partial DIRECT studies and stale or differently sourced evaluations."""

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import pytest
import yaml

from scripts.report_direct_product_ten_seeds import (
    MODEL, SEEDS, normalized_config, report, validate_run,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def study(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    teacher_dir = tmp_path / "teacher/seed42"
    teacher_dir.mkdir(parents=True)
    teacher = {"run_dir": str(teacher_dir), "predictions": {}, "predictions_sha256": {}}
    for split in ("train", "val", "test"):
        path = teacher_dir / "predictions" / f"{split}.parquet"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"fixed teacher {split} probabilities".encode())
        teacher["predictions"][split] = str(path)
        teacher["predictions_sha256"][split] = digest(path)
    teacher.update(test_predictions=teacher["predictions"]["test"],
                   pctr_scores_sha256=teacher["predictions_sha256"]["test"])
    write_json(teacher_dir / "run.json", {"model": "dcn", "status": "FINISHED",
               "selected_on": {"metric": "ctr_ne", "split": "val"}})
    encoded = tmp_path / "encoded"
    ids = encoded / "test/sample_id.npy"
    ids.parent.mkdir(parents=True)
    ids.write_bytes(b"encoded test IDs")
    manifest = {"seeds": SEEDS, "model": MODEL, "checkpoint_metric": "logged_ctcvr_ne",
                "epochs": 5, "teacher": teacher, "runs": []}
    for seed in SEEDS:
        run_dir = tmp_path / "runs" / str(seed)
        run_dir.mkdir(parents=True)
        cfg = {"model": MODEL, "training": {"seed": seed, "epochs": 5,
               "checkpoint_metric": "logged_ctcvr_ne", "keep_checkpoint": "best",
               "gpus": 1, "out_dir": "artifacts", "evaluate_test": False, "lr": .001},
               "model_params": {"emb_dim": 20}, "data": {"encoded_dir": str(encoded)},
               "logged_pctr": {"dcn_run": str(teacher_dir), "score_column": "score",
                               "predictions": dict(teacher["predictions"])},
               "evaluation": {"is_logged_ctcvr": True}}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        if seed == 42:
            manifest["reference_config"] = str(config_path)
        metadata = {"model": MODEL, "status": "FINISHED", "selected_on": {
                    "metric": "logged_ctcvr_ne", "split": "val", "epoch": 2, "value": .8},
                    "logged_pctr": {"dcn_run": str(teacher_dir), "files": {
                        split: {"file": teacher["predictions"][split]} for split in ("train", "val")}},
                    "data": {"test": {"n_rows": 100, "n_clicks": 20, "n_conversions": 3}}}
        write_json(run_dir / "run.json", metadata)
        (run_dir / "history.csv").write_text("epoch,val_score\n1,1.2\n2,.8\n3,.8\n4,.9\n5,1.1\n")
        (run_dir / "best.pt").write_bytes(f"DIRECT checkpoint {seed}".encode())
        predictions = run_dir / "predictions/test.parquet"
        predictions.parent.mkdir()
        predictions.write_bytes(f"DIRECT v probabilities {seed}".encode())
        checkpoint_hash, prediction_hash = digest(run_dir / "best.pt"), digest(predictions)
        cache = {"identity": {"version": 1, "checkpoint_sha256": checkpoint_hash,
                              "config_sha256": digest(config_path), "encoded_dir": str(encoded),
                              "inputs": {"test/sample_id.npy": [ids.stat().st_size, ids.stat().st_mtime_ns]}},
                 "sha256": prediction_hash, "rows": 100}
        write_json(run_dir / "predictions/test.provenance.json", cache)
        native = {"split": "test", "population": "pCVR", "rows": 20, "positives": 3,
                  "auc": seed / 100, "pr_auc": .01, "ne": .9, "ce": 1.1}
        logged = {**native, "population": "CTCVR-logged", "rows": 100,
                  "logged_pctr_run": str(teacher_dir), "logged_pctr_file": teacher["test_predictions"]}
        write_json(run_dir / "metrics/test.json", [native])
        logged_dir = run_dir / "metrics_logged/teacher"
        write_json(logged_dir / "test.json", [logged])
        provenance = {"checkpoint_sha256": checkpoint_hash, "predictions_sha256": prediction_hash,
                      "pctr_scores_sha256": teacher["pctr_scores_sha256"],
                      "logged_pctr": {"dcn_run": str(teacher_dir)}}
        write_json(logged_dir / "provenance.json", provenance)
        # A better-scoring other teacher cannot replace the fixed producer.
        write_json(run_dir / "metrics_logged/other/test.json", [{**logged, "auc": .999}])
        write_json(run_dir / "metrics_logged/other/provenance.json", {
            **provenance, "pctr_scores_sha256": "other source"})
        entry = {"seed": seed, "run_dir": str(run_dir)}
        if seed == 51:
            record = study_dir / "records/51.json"
            write_json(record, {"seed": seed, "run_dir": str(run_dir)})
            entry.update(run_dir=None, run_record=str(record))
        manifest["runs"].append(entry)
    write_json(study_dir / "manifest.json", manifest)
    return study_dir, tmp_path


def test_report_has_only_two_legitimate_populations_and_exact_ten_seed_sample_sem(study):
    study_dir, root = study
    output = report(study_dir, root)
    result = read_json(output)
    assert len(result["sources"]) == 10
    assert len(result["summary"]) == 8
    assert len(result["per_seed"]) == 20
    assert {r["population"] for r in result["summary"]} == {"pCVR", "CTCVR-logged"}
    assert [r["seed"] for r in result["sources"]] == SEEDS
    values = [seed / 100 for seed in SEEDS]
    for row in result["summary"]:
        assert row["variant"] == "direct" and row["n"] == 10
        if row["metric"] == "auc":
            assert row["values"] == values
            assert row["mean"] == pytest.approx(statistics.mean(values))
            assert row["sem"] == pytest.approx(statistics.stdev(values) / math.sqrt(10))
    assert output.with_suffix(".csv").exists()
    assert "no auxiliary pCTR tower or native q*v output" in output.with_suffix(".md").read_text()


@pytest.mark.parametrize("problem,match", [
    ("missing_seed", "exactly one pinned"), ("duplicate_seed", "exactly one pinned"),
    ("wrong_seed_range", "seeds 42 through 51"), ("wrong_model", "prescribe DIRECT"),
    ("wrong_checkpoint_metric", "prescribe DIRECT"), ("missing_record", "Missing required artifact"),
    ("wrong_record_seed", "record contains a different seed"),
])
def test_rejects_incomplete_or_inconsistent_manifest_without_outputs(study, problem, match):
    study_dir, root = study
    path = study_dir / "manifest.json"
    manifest = read_json(path)
    if problem == "missing_seed":
        manifest["runs"].pop()
    elif problem == "duplicate_seed":
        manifest["runs"][-1]["seed"] = 42
    elif problem == "wrong_seed_range":
        manifest["seeds"] = list(range(42, 46))
    elif problem == "wrong_model":
        manifest["model"] = "esmm"
    elif problem == "wrong_checkpoint_metric":
        manifest["checkpoint_metric"] = "ctcvr_ne"
    elif problem == "missing_record":
        Path(manifest["runs"][-1]["run_record"]).unlink()
    else:
        write_json(Path(manifest["runs"][-1]["run_record"]), {"seed": 50, "run_dir": str(root / "runs/51")})
    write_json(path, manifest)
    with pytest.raises(ValueError, match=match):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


@pytest.mark.parametrize("problem,match", [
    ("unfinished", "expected a finished"), ("wrong_seed", "declared seed"),
    ("different_config", "configuration differs"), ("four_epochs", "five complete"),
    ("selected_tie", "first minimum validation"), ("selected_on_test", "first minimum validation"),
    ("wrong_selected_value", "disagrees with history"),
    ("different_training_teacher_record", "recorded training teacher differs"),
    ("stale_checkpoint", "native prediction checkpoint"),
    ("stale_config_binding", "native prediction configuration"),
    ("damaged_predictions", "damaged native test predictions"),
    ("changed_encoded_input", "native encoded input changed"),
    ("stale_serving_checkpoint", "serving evaluation checkpoint"),
    ("different_serving_predictions", "different native predictions"),
    ("different_serving_teacher", "serving teacher prediction hash differs"),
    ("extra_auxiliary_metrics", "only native pCVR"),
    ("nonfinite_metric", "invalid auc"), ("wrong_test_census", "inconsistent test census"),
])
def test_rejects_invalid_protocol_and_stale_evaluation_inputs(study, problem, match):
    study_dir, root = study
    run_dir = root / "runs/43"
    path = run_dir / "run.json"
    metadata = read_json(path)
    if problem == "unfinished":
        metadata["status"] = "RUNNING"
    elif problem == "selected_tie":
        metadata["selected_on"]["epoch"] = 3
    elif problem == "selected_on_test":
        metadata["selected_on"]["split"] = "test"
    elif problem == "wrong_selected_value":
        metadata["selected_on"]["value"] = .7
    elif problem == "different_training_teacher_record":
        metadata["logged_pctr"]["dcn_run"] = "other-teacher"
    elif problem == "wrong_test_census":
        metadata["data"]["test"]["n_clicks"] = 21
    write_json(path, metadata)
    if problem in ("wrong_seed", "different_config"):
        path = run_dir / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["training"]["seed" if problem == "wrong_seed" else "lr"] = 999
        path.write_text(yaml.safe_dump(cfg))
    elif problem == "four_epochs":
        path = run_dir / "history.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif problem == "stale_checkpoint":
        (run_dir / "best.pt").write_bytes(b"new checkpoint")
    elif problem == "stale_config_binding":
        path = run_dir / "predictions/test.provenance.json"
        cache = read_json(path)
        cache["identity"]["config_sha256"] = "different-config"
        write_json(path, cache)
    elif problem == "damaged_predictions":
        (run_dir / "predictions/test.parquet").write_bytes(b"changed probabilities")
    elif problem == "changed_encoded_input":
        (root / "encoded/test/sample_id.npy").write_bytes(b"changed IDs")
    elif problem in ("stale_serving_checkpoint", "different_serving_predictions", "different_serving_teacher"):
        path = run_dir / "metrics_logged/teacher/provenance.json"
        provenance = read_json(path)
        key = {"stale_serving_checkpoint": "checkpoint_sha256",
               "different_serving_predictions": "predictions_sha256",
               "different_serving_teacher": "pctr_scores_sha256"}[problem]
        provenance[key] = "different input"
        write_json(path, provenance)
    elif problem in ("extra_auxiliary_metrics", "nonfinite_metric"):
        path = run_dir / "metrics/test.json"
        rows = read_json(path)
        if problem == "extra_auxiliary_metrics":
            rows.append({**rows[0], "population": "CTR"})
        else:
            rows[0]["auc"] = float("nan")
        write_json(path, rows)
    with pytest.raises(ValueError, match=match):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_fixed_teacher_split_is_content_bound(study, split):
    study_dir, root = study
    teacher = read_json(study_dir / "manifest.json")["teacher"]
    Path(teacher["predictions"][split]).write_bytes(b"changed teacher scores")
    with pytest.raises(ValueError, match=f"teacher {split} prediction hash changed"):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


def test_allowed_evaluation_only_config_differences_do_not_change_training_contract(study):
    study_dir, root = study
    manifest = read_json(study_dir / "manifest.json")
    run_dir = root / "runs/43"
    path = run_dir / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    original = copy.deepcopy(cfg)
    cfg["training"].update(gpus=-1, out_dir="another-output", evaluate_test=True)
    cfg.pop("evaluation")
    cfg["logged_pctr"]["predictions"].pop("test")
    assert normalized_config(cfg, root) == normalized_config(original, root)
    path.write_text(yaml.safe_dump(cfg))
    cache_path = run_dir / "predictions/test.provenance.json"
    cache = read_json(cache_path)
    cache["identity"]["config_sha256"] = digest(path)
    write_json(cache_path, cache)
    metadata, metrics = validate_run(run_dir, 43, manifest["reference_config"], manifest["teacher"], root)
    assert metadata["selected_on"]["epoch"] == 2
    assert set(metrics) == {"pCVR", "CTCVR-logged"}
