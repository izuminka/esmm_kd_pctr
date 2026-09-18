import csv
import json

import pytest
import yaml

from scripts.report_esmm_study import report


@pytest.fixture
def study(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    jobs, registry = [], {}
    for arm in ("esmm", "esmm_half", "esmm_quarter"):
        config = study_dir / f"{arm}.yaml"
        config.write_text(yaml.safe_dump({"training": {"out_dir": "artifacts"}}))
        jobs.append({"arm": arm, "config": str(config)})
        registry[arm] = {"run": "r1"}
        run_dir = tmp_path / "artifacts" / arm / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run": "r1", "status": "FINISHED",
            "selected_on": {"metric": "pcvr_ne", "epoch": 2},
        }))
        (run_dir / "history.csv").write_text(
            "epoch,val_pcvr_ne\n1,1.2\n2,0.8\n3,0.8\n"
        )
        metrics = [{"split": "test", "population": population,
                    "rows": 20 if population == "pCVR" else 100,
                    "positives": 3, "auc": .7, "pr_auc": .2, "ne": .9, "ce": 1.1}
                   for population in ("CTR", "CTCVR", "CTCVR-logged", "pCVR")]
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (study_dir / "submission.json").write_text(json.dumps({
        "jobs": jobs, "epochs": 3, "checkpoint_metric": "pcvr_ne",
    }))
    (study_dir / "best_models.yaml").write_text(yaml.safe_dump(registry))
    return study_dir, tmp_path


def test_report_exports_all_arms_and_populations_with_first_minimum(study):
    study_dir, root = study
    path = report(study_dir, root)
    assert "lowest validation clicked-pCVR NE" in path.read_text()
    with path.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["arm"] for row in rows} == {"esmm", "esmm_half", "esmm_quarter"}
    assert {row["selected_epoch"] for row in rows} == {"2"}
    assert {row["population"] for row in rows} == {
        "pCVR", "pCTR", "CTCVR (q*v)", "logged CTCVR (p*v)",
    }


@pytest.mark.parametrize("problem", ["unfinished", "wrong_epoch", "different_census"])
def test_report_rejects_invalid_study_results(study, problem):
    study_dir, root = study
    run_dir = root / "artifacts" / "esmm_half" / "r1"
    if problem == "different_census":
        path = run_dir / "metrics.json"
        metrics = json.loads(path.read_text())
        metrics[0]["rows"] += 1
        path.write_text(json.dumps(metrics))
        message = "inconsistent test census"
    else:
        path = run_dir / "run.json"
        run = json.loads(path.read_text())
        if problem == "unfinished":
            run["status"] = "RUNNING"
            message = "run is RUNNING"
        else:
            run["selected_on"]["epoch"] = 3
            message = "not the minimum validation"
        path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match=message):
        report(study_dir, root)
    assert not (study_dir / "comparison.csv").exists()


@pytest.mark.parametrize('metric', ['ctr_ne', 'ctcvr_ne', 'logged_ctcvr_ne', 'logged_ctcvr_prauc'])
def test_report_checks_selection_direction_and_retains_first_tie(study, metric):
    study_dir, root = study
    path = study_dir / 'submission.json'
    manifest = json.loads(path.read_text())
    manifest['checkpoint_metric'] = metric
    path.write_text(json.dumps(manifest))
    values = (1.2, .8, .8) if metric.endswith('_ne') else (.2, .4, .4)
    for job in manifest['jobs']:
        run_dir = root / 'artifacts' / job['arm'] / 'r1'
        run_path = run_dir / 'run.json'
        run = json.loads(run_path.read_text())
        run['selected_on']['metric'] = metric
        run_path.write_text(json.dumps(run))
        (run_dir / 'history.csv').write_text('epoch,val_score\n' + ''.join(
            f'{epoch},{value}\n' for epoch, value in enumerate(values, 1)))
    report(study_dir, root)
    run['selected_on']['epoch'] = 3
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='does not optimize'):
        report(study_dir, root)
