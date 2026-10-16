"""Guard against partial seed summaries and mixing serving teachers/checkpoints."""

import copy
import hashlib
import json
import math
import shutil
import statistics

import pytest
import yaml

from scripts.report_paper_ten_seeds import (
    SEEDS, TEACHER_ALIAS, TEACHER_CANONICAL, VARIANTS,
    normalized_config, read_test_metrics, report, validate_run,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def study(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    teacher_dir = tmp_path / "teacher/run"
    teacher_dir.mkdir(parents=True)
    predictions = teacher_dir / "test.parquet"
    predictions.write_bytes(b"fixed teacher test probabilities")
    teacher_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()

    def rows(seed):
        return [{"split": "test", "population": population,
                 "rows": 20 if population == "pCVR" else 100,
                 "positives": 20 if population == "CTR" else 3,
                 "auc": seed / 100, "pr_auc": .01, "ne": .9, "ce": 1.1}
                for population in ("CTR", "pCVR", "CTCVR", "CTCVR-logged")]

    write_json(teacher_dir / "metrics/test.json", rows(42)[:1])
    manifest = {"seeds": SEEDS, "teacher": {
        "run_dir": str(teacher_dir), "test_predictions": str(predictions),
        "pctr_scores_sha256": teacher_hash}, "variants": []}
    for variant_id in VARIANTS:
        regime = "downsampled" if variant_id.startswith("down_") else "full"
        criterion = "ctcvr_ne" if variant_id in ("esmm", "esmm_dcn", "down_esmm") else "logged_ctcvr_ne"
        variant = {"id": variant_id, "label": variant_id, "regime": regime,
                   "checkpoint_metric": criterion, "runs": []}
        for seed in SEEDS:
            run_dir = tmp_path / "runs" / variant_id / str(seed)
            run_dir.mkdir(parents=True)
            config = {"model": "esmm" if criterion == "ctcvr_ne" else "esmm_logged_pctr",
                      "model_params": {"emb_dim": 20},
                      "training": {"seed": seed, "epochs": 5, "gpus": 1,
                                   "lr": .001, "checkpoint_metric": criterion,
                                   "keep_checkpoint": "best", "out_dir": "artifacts"}}
            if regime == "downsampled":
                config["data"] = {"nonclick_sampling": {"rate": .05, "seed": 2026}}
                (run_dir / "retained_sample_ids.npy").write_bytes(b"fixed retained sample IDs")
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
            if seed == 42:
                variant["reference_config"] = str(run_dir / "config.yaml")
            write_json(run_dir / "run.json", {"status": "FINISHED", "selected_on": {
                "metric": criterion, "split": "val", "epoch": 2, "value": .8}})
            (run_dir / "history.csv").write_text("epoch,val_score\n1,1.2\n2,.8\n3,.8\n4,.9\n5,1.1\n")
            checkpoint = f"checkpoint {variant_id} {seed}".encode()
            (run_dir / "best.pt").write_bytes(checkpoint)
            metrics = rows(seed)
            # A misleading legacy inline result must never override the bound report.
            write_json(run_dir / "metrics/test.json", metrics[:3] + [dict(metrics[3], auc=.99)])
            logged_dir = run_dir / "metrics_logged/teacher"
            write_json(logged_dir / "test.json", [dict(metrics[3], logged_pctr_run=str(teacher_dir))])
            provenance = {"checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                          "pctr_scores_sha256": teacher_hash,
                          "logged_pctr": {"dcn_run": str(teacher_dir)}}
            write_json(logged_dir / "provenance.json", provenance)
            # This other producer scores better; it is not the specified teacher.
            write_json(run_dir / "metrics_logged/other/test.json", [dict(metrics[3], auc=.999)])
            write_json(run_dir / "metrics_logged/other/provenance.json", {
                **provenance, "pctr_scores_sha256": "different teacher",
                "logged_pctr": {"dcn_run": str(tmp_path / "other-teacher")}})
            entry = {"seed": seed, "run_dir": str(run_dir), "config": str(run_dir / "config.yaml")}
            if seed == 51:
                record_path = study_dir / "records" / f"{variant_id}_{seed}.json"
                write_json(record_path, {"run_dir": str(run_dir)})
                entry.update(run_dir=None, run_record=str(record_path))
            variant["runs"].append(entry)
        manifest["variants"].append(variant)
    write_json(study_dir / "manifest.json", manifest)
    return study_dir, tmp_path


def test_reports_exact_ten_seed_means_and_sample_sem_with_fixed_teacher(study):
    study_dir, root = study
    output = report(study_dir, root)
    result = read_json(output)
    assert len(result["sources"]) == 80
    assert len(result["summary"]) == 128
    expected = [seed / 100 for seed in SEEDS]
    for row in result["summary"]:
        assert row["n"] == 10
        if row["metric"] == "auc":
            assert row["values"] == expected
            assert row["mean"] == pytest.approx(statistics.mean(expected))
            assert row["sem"] == pytest.approx(statistics.stdev(expected) / math.sqrt(10))
    assert "sqrt(10)" in output.with_suffix(".md").read_text()
    assert output.with_suffix(".csv").exists()
    full = (study_dir / "tables/full.tex").read_text()
    down = (study_dir / "tables/downsampled.tex").read_text()
    assert "42--51" in full and "42--51" in down
    assert full.count(r"\multirow{4}") == 6
    assert down.count(r"\multirow{4}") == 2
    assert r"\shortstack{ESMM\\+ DCN}" in full
    assert "DCN-V2 & $p_i$" in full
    assert r"q_i^{\mathrm{full}}" in down


@pytest.mark.parametrize("problem,match", [
    ("missing_seed", "exactly one run"),
    ("duplicate_seed", "exactly one run"),
    ("wrong_seed_declaration", "exactly seeds"),
    ("missing_variant", "eight manuscript variants"),
    ("wrong_regime", "wrong training regime"),
    ("missing_run_record", "Missing required artifact"),
])
def test_rejects_incomplete_or_mismatched_manifest_without_report(study, problem, match):
    study_dir, root = study
    path = study_dir / "manifest.json"
    manifest = read_json(path)
    if problem == "missing_seed":
        manifest["variants"][0]["runs"].pop()
    elif problem == "duplicate_seed":
        manifest["variants"][0]["runs"][-1]["seed"] = 42
    elif problem == "wrong_seed_declaration":
        manifest["seeds"] = list(range(42, 46))
    elif problem == "missing_variant":
        manifest["variants"].pop()
    elif problem == "wrong_regime":
        manifest["variants"][0]["regime"] = "downsampled"
    else:
        manifest["variants"][0]["runs"][-1]["run_record"] = str(root / "absent.json")
    write_json(path, manifest)
    with pytest.raises(ValueError, match=match):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


@pytest.mark.parametrize("problem,match", [
    ("unfinished", "run is RUNNING"),
    ("four_epochs", "five complete"),
    ("selected_tie", "first minimum validation"),
    ("test_selection", "first minimum validation"),
    ("wrong_selected_value", "disagrees with history"),
    ("wrong_seed", "declared seed"),
    ("different_configuration", "configuration differs"),
    ("stale_checkpoint", "stale logged evaluation checkpoint"),
    ("different_teacher_hash", "prediction hash does not match"),
    ("missing_fixed_teacher", "exactly one fixed-teacher logged report"),
    ("duplicate_fixed_teacher", "exactly one fixed-teacher logged report"),
    ("nonfinite", "invalid auc"),
    ("different_census", "inconsistent test census"),
])
def test_rejects_invalid_run_or_unbound_metrics(study, problem, match):
    study_dir, root = study
    run_dir = root / "runs/esmm/43"
    path = run_dir / "run.json"
    run = read_json(path)
    if problem == "unfinished":
        run["status"] = "RUNNING"
    elif problem == "selected_tie":
        run["selected_on"]["epoch"] = 3
    elif problem == "test_selection":
        run["selected_on"]["split"] = "test"
    elif problem == "wrong_selected_value":
        run["selected_on"]["value"] = .7
    write_json(path, run)
    if problem == "four_epochs":
        path = run_dir / "history.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    if problem in ("wrong_seed", "different_configuration"):
        path = run_dir / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["training"]["seed" if problem == "wrong_seed" else "lr"] = 999
        path.write_text(yaml.safe_dump(cfg))
    if problem == "stale_checkpoint":
        (run_dir / "best.pt").write_bytes(b"changed checkpoint")
    if problem in ("different_teacher_hash", "missing_fixed_teacher", "duplicate_fixed_teacher"):
        path = run_dir / "metrics_logged/teacher/provenance.json"
        provenance = read_json(path)
        if problem == "different_teacher_hash":
            provenance["pctr_scores_sha256"] = "changed"
        elif problem == "missing_fixed_teacher":
            provenance["pctr_scores_sha256"] = "other"
            provenance["logged_pctr"]["dcn_run"] = str(root / "other")
        else:
            write_json(run_dir / "metrics_logged/duplicate/provenance.json", provenance)
            write_json(run_dir / "metrics_logged/duplicate/test.json",
                       read_json(run_dir / "metrics_logged/teacher/test.json"))
        write_json(path, provenance)
    if problem in ("nonfinite", "different_census"):
        path = run_dir / "metrics/test.json"
        rows = read_json(path)
        if problem == "nonfinite":
            rows[0]["auc"] = float("nan")
        else:
            for row in rows:
                row["rows"] *= 2
                row["positives"] *= 2
            logged_path = run_dir / "metrics_logged/teacher/test.json"
            logged = read_json(logged_path)
            logged[0]["rows"] *= 2
            logged[0]["positives"] *= 2
            write_json(logged_path, logged)
        write_json(path, rows)
    with pytest.raises(ValueError, match=match):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


def test_training_only_validation_does_not_require_test_reports(study):
    study_dir, root = study
    variant = read_json(study_dir / "manifest.json")["variants"][0]
    run_dir = root / "runs/esmm/42"
    (run_dir / "metrics/test.json").unlink()
    run = validate_run(run_dir, variant, 42, variant["reference_config"], root,
                       training_only=True)
    assert run["selected_on"]["epoch"] == 2


def test_rejects_teacher_scores_changed_after_manifest_was_prepared(study):
    study_dir, root = study
    teacher = read_json(study_dir / "manifest.json")["teacher"]
    with open(teacher["test_predictions"], "wb") as stream:
        stream.write(b"new teacher scores")
    with pytest.raises(ValueError, match="teacher test predictions changed"):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


@pytest.mark.parametrize("problem,match", [
    ("different_ids", "retained sample IDs differ"),
    ("missing_ids", "missing retained sample IDs"),
    ("missing_reference", "Missing retained sample IDs for reference"),
    ("wrong_sampling_spec", "fixed nonclick sampling rate"),
])
def test_downsampling_mask_is_fixed_across_variants_and_student_seeds(study, problem, match):
    study_dir, root = study
    run_dir = root / "runs/down_kd_tau/51"
    if problem == "different_ids":
        (run_dir / "retained_sample_ids.npy").write_bytes(b"different retained sample IDs")
    elif problem == "missing_ids":
        (run_dir / "retained_sample_ids.npy").unlink()
    elif problem == "missing_reference":
        (root / "runs/down_esmm/42/retained_sample_ids.npy").unlink()
    else:
        # Even an internally consistent but changed reference is not this study.
        for seed in SEEDS:
            path = root / "runs/down_kd_tau" / str(seed) / "config.yaml"
            cfg = yaml.safe_load(path.read_text())
            cfg["data"]["nonclick_sampling"]["seed"] = 999
            path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        report(study_dir, root)
    assert not (study_dir / "seed_summary.json").exists()


@pytest.mark.parametrize("conflicting", [False, True])
def test_known_migrated_teacher_duplicates_require_identical_metrics(study, conflicting):
    study_dir, root = study
    teacher = read_json(study_dir / "manifest.json")["teacher"]
    teacher["run_dir"] = str(root / TEACHER_CANONICAL)
    run_dir = root / "runs/esmm/42"
    canonical = run_dir / "metrics_logged/dcn_matched_stats_wd3e6_seed42"
    (run_dir / "metrics_logged/teacher").rename(canonical)
    path = canonical / "provenance.json"
    provenance = read_json(path)
    provenance["logged_pctr"]["dcn_run"] = TEACHER_CANONICAL
    write_json(path, provenance)
    path = canonical / "test.json"
    rows = read_json(path)
    rows[0]["logged_pctr_run"] = TEACHER_CANONICAL
    write_json(path, rows)
    alias = run_dir / "metrics_logged/dcn_matched_stats_wd3e6"
    shutil.copytree(canonical, alias)
    provenance["logged_pctr"]["dcn_run"] = TEACHER_ALIAS
    write_json(alias / "provenance.json", provenance)
    rows[0]["logged_pctr_run"] = TEACHER_ALIAS
    if conflicting:
        rows[0]["auc"] = .95
    write_json(alias / "test.json", rows)
    if conflicting:
        with pytest.raises(ValueError, match="conflicting canonical and historical"):
            read_test_metrics(run_dir, teacher, root)
    else:
        metrics = read_test_metrics(run_dir, teacher, root)
        assert metrics["CTCVR-logged"]["logged_pctr_run"] == TEACHER_CANONICAL
        assert metrics["CTCVR-logged"]["auc"] == .42


def test_normalization_allows_only_known_reporting_and_teacher_migration_differences(tmp_path):
    old = {"model": "esmm_logged_pctr", "training": {"seed": 42, "out_dir": "old", "gpus": 1},
           "evaluation": {"is_logged_ctcvr": True}, "logged_pctr": {
               "dcn_run": TEACHER_ALIAS, "score_column": "score",
               "predictions": {split: f"{TEACHER_ALIAS}/predictions/{split}.parquet"
                               for split in ("train", "val", "test")}}}
    new = copy.deepcopy(old)
    new["training"].update(seed=51, out_dir="new", gpus=-1, evaluate_test=False)
    new.pop("evaluation")
    new["logged_pctr"]["dcn_run"] = str(tmp_path / TEACHER_CANONICAL)
    new["logged_pctr"]["predictions"] = {
        split: str(tmp_path / TEACHER_CANONICAL / "predictions" / f"{split}.parquet")
        for split in ("train", "val")}
    before = copy.deepcopy(old)
    assert normalized_config(old, tmp_path) == normalized_config(new, tmp_path)
    assert old == before
    new["logged_pctr"]["predictions"]["train"] = "a-different-train-file"
    assert normalized_config(old, tmp_path) != normalized_config(new, tmp_path)
    old["model"] = "esmm"
    old["logged_pctr"]["predictions"] = {"test": "only-test"}
    native = {"model": "esmm", "training": {"gpus": -1}}
    assert normalized_config(old, tmp_path) == normalized_config(native, tmp_path)
    native["model_params"] = {"alpha": .3}
    assert normalized_config(old, tmp_path) != normalized_config(native, tmp_path)
