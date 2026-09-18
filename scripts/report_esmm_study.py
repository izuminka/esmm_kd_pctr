"""Combine finished Slurm arms' test metrics without reselecting runs."""

import argparse
import csv
import json
import math
import statistics
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
        elif manifest["checkpoint_metric"] in ("ctr_ne", "ctcvr_ne", "logged_ctcvr_prauc", "logged_ctcvr_ne"):
            select = min if manifest["checkpoint_metric"].endswith("_ne") else max
            best = select(history, key=lambda row: float(row["val_score"]))
            if run["selected_on"]["epoch"] != int(best["epoch"]):
                raise ValueError(f"{arm}: checkpoint does not optimize the validation selection metric")
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
    criterion = {"ctr_ne": "lowest validation pCTR NE",
                 "ctcvr_ne": "lowest validation native q*v CTCVR NE",
                 "pcvr_ne": "lowest validation clicked-pCVR NE",
                 "ctcvr_prauc": "highest validation q*v PR-AUC",
                 "logged_ctcvr_ne": "lowest validation p*v NE",
                 "logged_ctcvr_prauc": "highest validation p*v PR-AUC"}[manifest["checkpoint_metric"]]
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
    if "seed_baselines" in manifest or "seed_groups" in manifest:
        report_seed_summary(study, root, manifest, registry)
    return path


def report_seed_summary(study, root, manifest, registry):
    """Aggregate every prespecified seed, including the original seed 42."""
    groups = {}
    sources = []
    for base, baseline in manifest.get("seed_baselines", manifest.get("seed_groups", {})).items():
        paths = [root / baseline] if baseline else []
        for job in manifest["jobs"]:
            if job["arm"].rsplit("_seed", 1)[0] == base:
                cfg = yaml.safe_load(Path(job["config"]).read_text())
                paths.append(root / cfg["training"]["out_dir"] / job["arm"] / registry[job["arm"]]["run"])
        seeds = []
        reference = None
        for run_dir in paths:
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
            seed = cfg["training"].pop("seed")
            if reference is None:
                reference = cfg
            if cfg != reference:
                raise ValueError(f"Configuration differs beyond seed: {run_dir}")
            seeds.append(seed)
            run = json.loads((run_dir / "run.json").read_text())
            history = list(csv.DictReader((run_dir / "history.csv").open()))
            assert run["status"] == "FINISHED" and len(history) == 5
            assert run["selected_on"]["metric"] == manifest["checkpoint_metric"]
            assert run["selected_on"]["split"] == "val"
            assert run["selected_on"]["epoch"] == int(min(history, key=lambda r: float(r["val_score"]))["epoch"])
            metrics = json.loads((run_dir / "metrics/test.json").read_text())
            assert {r["population"] for r in metrics} == {"CTR", "pCVR", "CTCVR", "CTCVR-logged"}
            sources.append({"variant": base, "seed": seed, "run_dir": str(run_dir), "selected_on": run["selected_on"]})
            for row in metrics:
                for metric in ("auc", "pr_auc", "ne", "ce"):
                    key = (base, row["population"], metric)
                    group = groups.setdefault(key, {"values": [], "counts": (row["rows"], row["positives"])})
                    assert group["counts"] == (row["rows"], row["positives"])
                    group["values"].append(row[metric])
        assert sorted(seeds) == [42, 43, 44, 45], seeds
    summary = []
    for (variant, population, metric), group in groups.items():
        values = group["values"]
        summary.append({"variant": variant, "population": population, "metric": metric,
                        "n": len(values), "mean": statistics.mean(values),
                        "sem": statistics.stdev(values) / math.sqrt(len(values)),
                        "values": values})
    (study / "seed_summary.json").write_text(json.dumps({"sources": sources, "summary": summary}, indent=2) + "\n")
    with (study / "seed_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["Test metrics: mean ± SEM across seeds 42, 43, 44, 45 (n=4 per variant).",
             "SEM = sample standard deviation / sqrt(4); this measures training-seed variability with a fixed DCN teacher and test split.", "",
             "| Variant | Population | AUC | PR-AUC | NE | CE |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for variant, population in dict.fromkeys((r["variant"], r["population"]) for r in summary):
        cells = []
        for metric in ("auc", "pr_auc", "ne", "ce"):
            r = next(r for r in summary if (r["variant"], r["population"], r["metric"]) == (variant, population, metric))
            digits = 8 if metric == "pr_auc" else 6
            cells.append(f"{r['mean']:.{digits}f} ± {r['sem']:.{digits}f}")
        lines.append(f"| {variant} | {population} | " + " | ".join(cells) + " |")
    (study / "seed_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = report(args.study_dir.resolve(), args.project_root.resolve())
    print(output.read_text())
    print(f"Wrote {output} and {output.with_suffix('.csv')}")
