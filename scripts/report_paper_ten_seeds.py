"""Validate and aggregate the prespecified ten-seed manuscript study.

The manifest fixes the seeds, run directories, checkpoint criteria, and serving
teacher before this script reads test scores. A partial study produces no report.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import yaml


SEEDS = list(range(42, 52))
VARIANTS = ("esmm", "esmm_dcn", "kd03", "kd05", "kd07", "kd0",
            "down_esmm", "down_kd_tau")
POPULATIONS = ("CTR", "pCVR", "CTCVR", "CTCVR-logged")
METRICS = ("auc", "pr_auc", "ne", "ce")
TEACHER_ALIAS = "artifacts/dcn_matched_stats_wd3e6/20260917T232959Z"
TEACHER_CANONICAL = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"
TEX_LABELS = {"esmm": "ESMM", "esmm_dcn": r"\shortstack{ESMM\\+ DCN}",
              "kd03": "KD--0.3", "kd05": "KD--0.5", "kd07": "KD--0.7",
              "kd0": "KD--0", "down_esmm": "ESMM",
              "down_kd_tau": r"KD$_{\tau}$--0.5"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_path(value, root):
    """Resolve paths, including only the documented seed-42 teacher migration."""
    path = (Path(root) / value).resolve()
    old, new = Path(root) / TEACHER_ALIAS, Path(root) / TEACHER_CANONICAL
    if path == old or path.is_relative_to(old):
        path = new / path.relative_to(old)
    return path


def normalized_config(config, root):
    """Remove only seed, output location, and known evaluation-only differences."""
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    for key in ("seed", "out_dir", "evaluate_test"):
        training.pop(key, None)
    if training.get("gpus", -1) in (-1, 1):
        training["gpus"] = 1
    evaluation = cfg.get("evaluation", {})
    evaluation.pop("is_logged_ctcvr", None)
    if not evaluation:
        cfg.pop("evaluation", None)
    block = cfg.get("logged_pctr")
    if isinstance(block, dict):
        predictions = block.get("predictions", {})
        predictions.pop("test", None)
        if (not predictions and cfg.get("model") in ("esmm", "esmm_dcn")
                and set(block) <= {"dcn_run", "predictions", "score_column"}):
            cfg.pop("logged_pctr")
        else:
            if "dcn_run" in block:
                block["dcn_run"] = str(canonical_path(block["dcn_run"], root))
            block["predictions"] = {split: str(canonical_path(path, root))
                                    for split, path in predictions.items()}
    return cfg


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing required artifact: {path}") from error


def _read_config(path):
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing required configuration: {path}") from error
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return cfg


def validate_run(run_dir, variant, seed, reference_config, root, teacher=None,
                 training_only=False):
    """Validate a run's protocol; optionally also load its fixed-teacher scores."""
    run_dir = Path(run_dir)
    name = f"{variant['id']} seed {seed} ({run_dir})"
    run = _read_json(run_dir / "run.json")
    if run.get("status") != "FINISHED":
        raise ValueError(f"{name}: run is {run.get('status')}; expected FINISHED")
    cfg = _read_config(run_dir / "config.yaml")
    reference = (_read_config(reference_config) if isinstance(reference_config, (str, Path))
                 else reference_config)
    if normalized_config(cfg, root) != normalized_config(reference, root):
        raise ValueError(f"{name}: training configuration differs from reference")
    training = cfg.get("training", {})
    if training.get("seed") != seed or training.get("epochs") != 5:
        raise ValueError(f"{name}: expected declared seed and five training epochs")
    if (variant["regime"] == "downsampled"
            and cfg.get("data", {}).get("nonclick_sampling") != {"rate": .05, "seed": 2026}):
        raise ValueError(f"{name}: expected fixed nonclick sampling rate 0.05 and seed 2026")
    criterion = variant["checkpoint_metric"]
    if (training.get("checkpoint_metric") != criterion
            or training.get("keep_checkpoint", "best") != "best"):
        raise ValueError(f"{name}: configuration has a different checkpoint rule")
    try:
        with (run_dir / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in history]
        scores = [float(row["val_score"]) for row in history]
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise ValueError(f"{name}: missing or invalid validation history") from error
    if epochs != [1, 2, 3, 4, 5] or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"{name}: expected five complete finite validation epochs")
    best = min(range(5), key=scores.__getitem__)  # First minimum matches training.
    selected = run.get("selected_on", {})
    if (selected.get("metric") != criterion or selected.get("split") != "val"
            or selected.get("epoch") != epochs[best]):
        raise ValueError(f"{name}: checkpoint is not the first minimum validation {criterion}")
    if "value" in selected and not math.isclose(float(selected["value"]), scores[best],
                                                rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{name}: selected validation value disagrees with history")
    if training_only:
        return run
    if teacher is None:
        raise ValueError("Fixed serving teacher is required when reading metrics")
    return run, read_test_metrics(run_dir, teacher, root)


def _validate_metric(row, name):
    if row.get("split") != "test":
        raise ValueError(f"{name}: expected only test metrics")
    if (not isinstance(row.get("rows"), int) or not isinstance(row.get("positives"), int)
            or not 0 <= row["positives"] <= row["rows"] or row["rows"] <= 0):
        raise ValueError(f"{name}: invalid test census")
    for metric in METRICS:
        value = row.get(metric)
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (metric in ("auc", "pr_auc") and value > 1)):
            raise ValueError(f"{name}: invalid {metric} value")


def read_test_metrics(run_dir, teacher, root):
    """Choose serving scores by declared producer identity, never by test metric."""
    run_dir = Path(run_dir)
    native = _read_json(run_dir / "metrics/test.json")
    metrics = {}
    for row in native:
        population = row.get("population")
        if population == "CTCVR-logged":
            continue  # Legacy inline scores have no independently bound provenance.
        if population not in POPULATIONS[:3] or population in metrics:
            raise ValueError(f"{run_dir}: duplicate or unexpected native population {population}")
        _validate_metric(row, str(run_dir))
        metrics[population] = row
    if set(metrics) != set(POPULATIONS[:3]):
        raise ValueError(f"{run_dir}: missing native test populations")
    candidates = []
    for path in sorted((run_dir / "metrics_logged").glob("*/test.json")):
        provenance = _read_json(path.parent / "provenance.json")
        source = provenance.get("logged_pctr", {}).get("dcn_run")
        same_run = source is not None and canonical_path(source, root) == canonical_path(teacher["run_dir"], root)
        source_hash = provenance.get("pctr_scores_sha256")
        same_hash = source_hash is not None and source_hash == teacher["pctr_scores_sha256"]
        if not (same_run or same_hash):
            continue
        if source_hash != teacher["pctr_scores_sha256"]:
            raise ValueError(f"{path}: fixed teacher prediction hash does not match")
        candidates.append((path, provenance))
    # Some migrated runs retain both the historical producer name and its
    # canonical seed-42 name. Prefer the canonical name only when the bound
    # checkpoint and every reported number agree; do not pool duplicate rows.
    producer = Path(teacher["run_dir"]).parent.name
    canonical = [(path, provenance) for path, provenance in candidates
                 if path.parent.name == producer]
    known_names = {"dcn_matched_stats_wd3e6", "dcn_matched_stats_wd3e6_seed42"}
    if (len(candidates) > 1 and len(canonical) == 1
            and {path.parent.name for path, _ in candidates} <= known_names):
        columns = ("split", "population", "rows", "positives", *METRICS)
        signatures = [([tuple(row.get(key) for key in columns) for row in _read_json(path)],
                       provenance.get("checkpoint_sha256"))
                      for path, provenance in candidates]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"{run_dir}: conflicting canonical and historical teacher reports")
        candidates = canonical
    if len(candidates) != 1:
        raise ValueError(f"{run_dir}: expected exactly one fixed-teacher logged report; found {len(candidates)}")
    path, provenance = candidates[0]
    if provenance.get("checkpoint_sha256") != digest(run_dir / "best.pt"):
        raise ValueError(f"{path}: stale logged evaluation checkpoint hash")
    rows = _read_json(path)
    if len(rows) != 1 or rows[0].get("population") != "CTCVR-logged":
        raise ValueError(f"{path}: expected one serving CTCVR test row")
    logged = rows[0]
    source = provenance.get("logged_pctr", {}).get("dcn_run")
    if (logged.get("logged_pctr_run") is not None and source is not None
            and canonical_path(logged["logged_pctr_run"], root) != canonical_path(source, root)):
        raise ValueError(f"{path}: logged row and provenance identify different teachers")
    _validate_metric(logged, str(path))
    metrics["CTCVR-logged"] = logged
    counts = lambda population: (metrics[population]["rows"], metrics[population]["positives"])
    if (counts("CTCVR") != counts("CTCVR-logged")
            or metrics["CTR"]["rows"] != metrics["CTCVR"]["rows"]
            or metrics["CTR"]["positives"] != metrics["pCVR"]["rows"]
            or metrics["CTCVR"]["positives"] != metrics["pCVR"]["positives"]):
        raise ValueError(f"{run_dir}: inconsistent census across test populations")
    return metrics


def _table(variants, summary, teacher, regime):
    lookup = {(row["variant"], row["population"], row["metric"]): row["mean"]
              for row in summary}
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.0}",
             r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}clrrrr@{}}", r"\hline",
             r"Model & Output & AUC $\uparrow$ & PR-AUC $\uparrow$ & NE $\downarrow$ & CE $\to 1$ \\", r"\hline",
             "DCN-V2 & $p_i$ & " + " & ".join(f"{teacher[m]:.4g}" for m in METRICS) + r" \\", r"\hline"]
    q = r"q_i^{\mathrm{full}}" if regime == "downsampled" else "q_i"
    outputs = (f"${q}$", "$v_i$", f"${q} v_i$", "$p_i v_i$")
    for variant in variants:
        if variant["regime"] != regime:
            continue
        for index, (population, output) in enumerate(zip(POPULATIONS, outputs)):
            label = (r"\multirow{4}{*}{" + TEX_LABELS[variant["id"]] + "}"
                     if index == 0 else "")
            values = [lookup[variant["id"], population, metric] for metric in METRICS]
            lines.append(f"{label} & {output} & " + " & ".join(f"{v:.4g}" for v in values) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular*}")
    if regime == "full":
        lines.extend([r"\caption{Full-population Ali-CCP test means over seeds 42--51 ($n=10$ per variant; DCN-V2: one fixed run).",
                      r"Outputs are frozen pCTR $p_i$, auxiliary pCTR $q_i$, pCVR $v_i$, native CTCVR $q_i v_i$, and serving CTCVR $p_i v_i$.",
                      r"ESMM + DCN replaces ESMM's pCTR tower with DCN-V2 and shares embeddings with its pCVR tower.}",
                      r"\label{tab:aliccp-full-seeds-column}"])
    else:
        lines.extend([r"\caption{Ali-CCP test means at 5\% nonclick retention over seeds 42--51 ($n=10$ per variant; KD$_{\tau}$--0.5: $\alpha=0.5$).",
                      r"Auxiliary pCTR $q_i^{\mathrm{full}}$ is corrected for downsampling; frozen DCN-V2 is unchanged from Table~\ref{tab:aliccp-full-seeds-column}.}",
                      r"\label{tab:aliccp-downsampled-5pct-column}"])
    return "\n".join(lines + [r"\end{table}", ""])


def report(study_dir, root):
    study_dir, root = Path(study_dir), Path(root)
    manifest = _read_json(study_dir / "manifest.json")
    variants = manifest["variants"]
    if manifest.get("seeds") != SEEDS:
        raise ValueError("Study must declare exactly seeds 42 through 51, in order")
    if len(variants) != 8 or {v["id"] for v in variants} != set(VARIANTS):
        raise ValueError("Study must declare each of the eight manuscript variants exactly once")
    variants = sorted(variants, key=lambda variant: VARIANTS.index(variant["id"]))
    teacher = manifest["teacher"]
    if digest(canonical_path(teacher["test_predictions"], root)) != teacher["pctr_scores_sha256"]:
        raise ValueError("Fixed teacher test predictions changed after study preparation")
    down_reference = next(variant for variant in variants if variant["id"] == "down_esmm")
    reference_ids = (root / down_reference["reference_config"]).parent / "retained_sample_ids.npy"
    if not reference_ids.is_file():
        raise ValueError(f"Missing retained sample IDs for reference downsampled ESMM seed 42: {reference_ids}")
    reference_mask_hash = digest(reference_ids)
    sources, groups, census = [], {}, {}
    for variant in variants:
        expected_regime = "downsampled" if variant["id"].startswith("down_") else "full"
        expected_criterion = "ctcvr_ne" if variant["id"] in ("esmm", "esmm_dcn", "down_esmm") else "logged_ctcvr_ne"
        if variant["regime"] != expected_regime or variant["checkpoint_metric"] != expected_criterion:
            raise ValueError(f"{variant['id']}: wrong training regime or checkpoint criterion")
        runs = variant["runs"]
        if len(runs) != 10 or sorted(item["seed"] for item in runs) != SEEDS:
            raise ValueError(f"{variant['id']}: expected exactly one run for each of seeds 42 through 51")
        for item in sorted(runs, key=lambda item: item["seed"]):
            run_dir = item.get("run_dir")
            if run_dir is None:
                record = _read_json(root / item["run_record"])
                run_dir = record.get("run_dir")
            if not run_dir:
                raise ValueError(f"{variant['id']} seed {item['seed']}: no completed run directory")
            run_dir = (root / run_dir).resolve()
            run, metrics = validate_run(run_dir, variant, item["seed"],
                                        root / variant["reference_config"], root, teacher)
            if variant["regime"] == "downsampled":
                retained_ids = run_dir / "retained_sample_ids.npy"
                if not retained_ids.is_file():
                    raise ValueError(f"{run_dir}: missing retained sample IDs")
                if digest(retained_ids) != reference_mask_hash:
                    raise ValueError(f"{run_dir}: retained sample IDs differ from downsampled ESMM seed 42")
            sources.append({"variant": variant["id"], "seed": item["seed"],
                            "run_dir": str(run_dir), "selected_on": run["selected_on"]})
            for population in POPULATIONS:
                row = metrics[population]
                counts = row["rows"], row["positives"]
                if census.setdefault(population, counts) != counts:
                    raise ValueError(f"{run_dir}: inconsistent test census for {population}")
                for metric in METRICS:
                    groups.setdefault((variant["id"], population, metric), []).append(row[metric])
    summary = [{"variant": variant, "population": population, "metric": metric,
                "n": 10, "mean": statistics.mean(values),
                "sem": statistics.stdev(values) / math.sqrt(10), "values": values}
               for (variant, population, metric), values in groups.items()]
    teacher_rows = _read_json(canonical_path(teacher["run_dir"], root) / "metrics/test.json")
    if len(teacher_rows) != 1 or teacher_rows[0].get("population") != "CTR":
        raise ValueError("Fixed teacher must have exactly one CTR test report")
    teacher_row = teacher_rows[0]
    _validate_metric(teacher_row, "Fixed teacher")
    if (teacher_row["rows"], teacher_row["positives"]) != census["CTR"]:
        raise ValueError("Fixed teacher has a different test census")
    # Write only after every seed, population, producer, and checkpoint passes.
    output = study_dir / "seed_summary.json"
    output.write_text(json.dumps({"sources": sources, "summary": summary}, indent=2) + "\n")
    with (study_dir / "seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    lines = ["Test metrics: mean ± SEM across seeds 42–51 (n=10 per variant).",
             "SEM = sample standard deviation / sqrt(10), with a fixed serving teacher and test split.",
             "", "| Variant | Population | AUC | PR-AUC | NE | CE |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    lookup = {(r["variant"], r["population"], r["metric"]): r for r in summary}
    for variant in variants:
        for population in POPULATIONS:
            cells = []
            for metric in METRICS:
                row = lookup[variant["id"], population, metric]
                cells.append(f"{row['mean']:.6g} ± {row['sem']:.6g}")
            lines.append(f"| {variant['label']} | {population} | " + " | ".join(cells) + " |")
    (study_dir / "seed_summary.md").write_text("\n".join(lines) + "\n")
    (study_dir / "tables").mkdir(exist_ok=True)
    for regime in ("full", "downsampled"):
        (study_dir / "tables" / f"{regime}.tex").write_text(_table(variants, summary, teacher_row, regime))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        print(report(args.study_dir.resolve(), args.project_root.resolve()))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Cannot report ten-seed study: {error}\n")
