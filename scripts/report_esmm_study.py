"""Combine the three finished Slurm arms' test metrics without reselecting runs."""

import argparse
import csv
import json
from pathlib import Path

import yaml


def report(study: Path, root: Path) -> Path:
    registry = yaml.safe_load((study / "best_models.yaml").read_text())
    manifest = json.loads((study / "submission.json").read_text())
    rows = []
    populations = {"pCVR": "pCVR", "CTR": "pCTR", "CTCVR": "CTCVR (q*v)",
                   "CTCVR-logged": "logged CTCVR (p*v)"}
    census = {}
    for job in manifest["jobs"]:
        arm = job["arm"]
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        run_dir = root / cfg["training"]["out_dir"] / arm / registry[arm]["run"]
        run = json.loads((run_dir / "run.json").read_text())
        if run["status"] != "FINISHED":
            raise ValueError(f"{arm}: run is {run['status']}")
        with (run_dir / "history.csv").open(newline="") as f:
            history = list(csv.DictReader(f))
        if (len(history) != manifest["epochs"]
                or run["selected_on"]["metric"] != manifest["checkpoint_metric"]):
            raise ValueError(f"{arm}: run does not match the study selection protocol")
        if manifest["checkpoint_metric"] == "pcvr_ne":
            best = min(history, key=lambda row: float(row["val_pcvr_ne"]))
            if run["selected_on"]["epoch"] != int(best["epoch"]):
                raise ValueError(f"{arm}: checkpoint is not the minimum validation pCVR NE")
        metrics = {r["population"]: r for r in json.loads((run_dir / "metrics.json").read_text())
                   if r["split"] == "test"}
        for key, label in populations.items():
            metric = metrics[key]  # fail if any requested population is missing
            counts = (metric["rows"], metric["positives"])
            if census.setdefault(key, counts) != counts:
                raise ValueError(f"{arm}: inconsistent test census for {key}")
            rows.append({"arm": arm, "run": run["run"],
                         "selected_epoch": run["selected_on"]["epoch"],
                         "population": label, "rows": counts[0], "positives": counts[1],
                         **{k: metric[k] for k in ("auc", "pr_auc", "ne", "ce")}})
    with (study / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    criterion = {"pcvr_ne": "lowest validation clicked-pCVR NE",
                 "ctcvr_prauc": "highest validation q*v PR-AUC"}[manifest["checkpoint_metric"]]
    lines = [f"{manifest['epochs']} epochs per arm; checkpoint selected by {criterion}.",
             "Test metrics do not participate in checkpoint selection.", "",
             "| Arm | Kept epoch | Population | AUC | PR-AUC | NE | CE |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['selected_epoch']} | {row['population']} | "
                     f"{row['auc']:.6f} | {row['pr_auc']:.8f} | "
                     f"{row['ne']:.6f} | {row['ce']:.6f} |")
    lines += ["", "pCVR uses clicked impressions; the other scores use all test impressions."]
    for key, counts in census.items():
        lines.append(f"- {populations[key]}: {counts[0]:,} rows, {counts[1]:,} positives.")
    lines += ["", "Runs:"]
    for job in manifest["jobs"]:
        arm = job["arm"]
        lines.append(f"- {arm}: {registry[arm]['run']}")
    path = study / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = report(args.study_dir.resolve(), args.project_root.resolve())
    print(output.read_text())
    print(f"Wrote {output} and {output.with_suffix('.csv')}")
