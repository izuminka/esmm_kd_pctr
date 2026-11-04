"""Verify every stored score of a final study against its inputs.

Usage (from the project root):
    code/.venv/bin/python code/scripts/verify_final_study.py --spec code/studies/final.yaml
    code/.venv/bin/python code/scripts/verify_final_study.py --spec code/studies/downsampled.yaml

For the teacher and every finished student run listed in <study_dir>/runs, it
checks that the run's config is the study's config for that seed, that the test
predictions are bound by hash to the run's checkpoint and config, recomputes
every native test metric from those predictions, and recomputes the serving
product p*v from the teacher's test scores, whose hash is taken from the file on
disk. It also refuses serving scores from any teacher other than the study's.
Nothing is written; a mismatch is reported and the exit status is nonzero.
"""

import argparse
import glob
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluation.metrics import compute_metrics  # noqa: E402
from scripts.report_metrics import populations  # noqa: E402

METRICS = ("auc", "pr_auc", "ne", "ce")


def sha(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(stored: dict, recomputed: dict) -> float:
    return max(abs(stored[m] - recomputed[m]) for m in METRICS)


def check_student(args) -> dict:
    record, study_dir, pctr_model, teacher_scores, model = args
    run = Path(record["run_dir"])
    out = {"arm": record["arm"], "problems": [], "max_diff": 0.0}
    study_cfg = yaml.safe_load(Path(record["config"]).read_text())
    if yaml.safe_load((run / "config.yaml").read_text()) != study_cfg:
        out["problems"].append("run config differs from the study config")
    meta = json.loads((run / "predictions/test.provenance.json").read_text())
    if meta["identity"]["checkpoint_sha256"] != sha(run / "best.pt"):
        out["problems"].append("test predictions not bound to this checkpoint")
    if meta["identity"]["config_sha256"] != sha(run / "config.yaml"):
        out["problems"].append("test predictions not bound to this config")
    if meta["sha256"] != sha(run / "predictions/test.parquet"):
        out["problems"].append("test predictions file changed after it was recorded")
    frame = pd.read_parquet(run / "predictions/test.parquet")
    y, z = frame.y.to_numpy(), frame.z.to_numpy()
    scores = {c: frame[c].to_numpy() for c in frame if c not in ("sample_id", "y", "z", "p")}
    stored = {r["population"]: r for r in json.loads((run / "metrics/test.json").read_text())
              if r["population"] != "CTCVR-logged"}
    for population, (score, label) in populations(model, y, z, scores).items():
        diff = compare(stored[population], compute_metrics(score, label))
        out["max_diff"] = max(out["max_diff"], diff)
        out[f"{population}_ne"] = stored[population]["ne"]
    sources = sorted(p.name for p in (run / "metrics_logged").iterdir() if p.is_dir())
    if sources != [pctr_model]:
        out["problems"].append(f"serving scores from {sources}, expected only {pctr_model}")
    logged = run / "metrics_logged" / pctr_model
    provenance = json.loads((logged / "provenance.json").read_text())
    teacher = pd.read_parquet(teacher_scores, columns=["sample_id", "score"])
    if provenance["pctr_scores_sha256"] != sha(teacher_scores):
        out["problems"].append("serving score made with a different teacher file")
    if provenance["predictions_sha256"] != meta["sha256"] or provenance["checkpoint_sha256"] != sha(run / "best.pt"):
        out["problems"].append("serving score made from different predictions or checkpoint")
    if not np.array_equal(teacher.sample_id.to_numpy(), frame.sample_id.to_numpy()):
        out["problems"].append("teacher and run test rows differ")
    serving = json.loads((logged / "test.json").read_text())[0]
    diff = compare(serving, compute_metrics(teacher.score.to_numpy() * scores["v"], y * z))
    out["max_diff"] = max(out["max_diff"], diff)
    out["serving_ne"] = serving["ne"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    study_dir = ROOT / spec["study_dir"]
    pctr_model = spec["teacher"]["pctr_model"]
    teacher_run = Path(json.loads((study_dir / "teacher.json").read_text())["selected"]["run_dir"])
    teacher_scores = teacher_run / "predictions/test.parquet"
    failures = 0

    # The teacher's own reported metrics, recomputed from its exported scores.
    for split in ("val", "test"):
        frame = pd.read_parquet(teacher_run / f"predictions/{split}.parquet", columns=["y", "score"])
        stored = json.loads((teacher_run / f"metrics/{split}.json").read_text())[0]
        diff = compare(stored, compute_metrics(frame.score.to_numpy(), frame.y.to_numpy()))
        failures += diff > 1e-9
        print(f"teacher {split:4s}: stored NE {stored['ne']:.6f}  recomputed max |diff| {diff:.1e}")
    meta = json.loads((teacher_run / "predictions/test.provenance.json").read_text())
    if meta["identity"]["checkpoint_sha256"] != sha(teacher_run / "best.pt") or meta["sha256"] != sha(teacher_scores):
        failures += 1
        print("teacher test scores are not bound to its checkpoint")

    models = {v["id"]: yaml.safe_load((ROOT / v["recipe"]).read_text())["model"] for v in spec["variants"]}
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(study_dir / "runs/*.json")))]
    students = [(r, study_dir, pctr_model, teacher_scores, models[r["variant"]])
                for r in records if r.get("variant") in models and r.get("evaluated")
                and (Path(r["run_dir"]) / "metrics_logged" / pctr_model / "test.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(check_student, students))
    for r in results:
        bad = r["problems"] or r["max_diff"] > 1e-9
        failures += bool(bad)
        native = "  ".join(f"{k.split('_ne')[0]} {v:.4f}" for k, v in r.items() if k.endswith("_ne") and k != "serving_ne")
        print(f"{r['arm']:<22} {'FAIL' if bad else 'ok':4}  recomputed max |diff| {r['max_diff']:.1e}  "
              f"serving NE {r['serving_ne']:.4f}  {native}  {'; '.join(r['problems'])}")
    print(f"\n{len(results)} student runs and the teacher checked; {failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
