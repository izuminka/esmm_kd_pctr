"""Report ten prespecified DIRECT seeds using a fixed serving pCTR teacher.

DIRECT has no auxiliary pCTR tower: its only test populations are clicked
pCVR (v) and serving CTCVR (p*v). Test scores never select runs or checkpoints.
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
MODEL = "esmm_direct_product"
POPULATIONS = ("pCVR", "CTCVR-logged")
METRICS = ("auc", "pr_auc", "ne", "ce")
TEACHER_ALIAS = "artifacts/dcn_matched_stats_wd3e6/20260917T232959Z"
TEACHER_CANONICAL = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing required artifact: {path}") from error


def read_config(path):
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing required configuration: {path}") from error
    require(isinstance(cfg, dict), f"Configuration must be a mapping: {path}")
    return cfg


def digest(path):
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError as error:
        raise ValueError(f"Missing required artifact: {path}") from error


def canonical_path(path, root):
    path = (Path(root) / path).resolve()
    old = Path(root) / TEACHER_ALIAS
    if path == old or path.is_relative_to(old):
        path = Path(root) / TEACHER_CANONICAL / path.relative_to(old)
    return path


def normalized_config(config, root):
    """Ignore only the student seed, output location, and evaluation options."""
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
    block = cfg.get("logged_pctr", {})
    if "dcn_run" in block:
        block["dcn_run"] = str(canonical_path(block["dcn_run"], root))
    if "predictions" in block:
        block["predictions"] = {split: str(canonical_path(path, root))
                                for split, path in block["predictions"].items()
                                if split != "test"}
    return cfg


def validate_teacher(teacher, root):
    """Pin both training/validation targets and the serving test factor."""
    predictions, hashes = teacher.get("predictions", {}), teacher.get("predictions_sha256", {})
    require(set(predictions) == set(hashes) == {"train", "val", "test"},
            "Fixed teacher requires train/val/test prediction paths and hashes")
    teacher_dir = canonical_path(teacher["run_dir"], root)
    source = read_json(teacher_dir / "run.json")
    require(source.get("model") == "dcn" and source.get("status") == "FINISHED",
            "Fixed teacher must be a finished DCN run")
    require(source.get("selected_on", {}).get("metric") == "ctr_ne"
            and source["selected_on"].get("split") == "val",
            "Fixed teacher must be selected by validation pCTR NE")
    for split in ("train", "val", "test"):
        path = canonical_path(predictions[split], root)
        require(path.is_relative_to(teacher_dir), f"Teacher {split} scores are outside the fixed run")
        require(digest(path) == hashes[split], f"Fixed teacher {split} prediction hash changed")
    require(canonical_path(teacher["test_predictions"], root) == canonical_path(predictions["test"], root)
            and teacher["pctr_scores_sha256"] == hashes["test"],
            "Teacher test identity disagrees with its split manifest")


def validate_metric(row, population, name):
    require(row.get("split") == "test" and row.get("population") == population,
            f"{name}: expected exactly the {population} test population")
    require(type(row.get("rows")) is int and type(row.get("positives")) is int
            and 0 <= row["positives"] <= row["rows"] and row["rows"] > 0,
            f"{name}: invalid test census")
    for metric in METRICS:
        value = row.get(metric)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                and (metric not in ("auc", "pr_auc") or value <= 1),
                f"{name}: invalid {metric} value")


def validate_run(run_dir, seed, reference_config, teacher, root):
    """Validate training selection and the common checkpoint/prediction binding."""
    run_dir = Path(run_dir).resolve()
    name = f"DIRECT seed {seed} ({run_dir})"
    run = read_json(run_dir / "run.json")
    require(run.get("status") == "FINISHED" and run.get("model") == MODEL,
            f"{name}: expected a finished {MODEL} run")
    config_path = run_dir / "config.yaml"
    cfg = read_config(config_path)
    reference = read_config(reference_config) if isinstance(reference_config, (str, Path)) else reference_config
    require(cfg.get("model") == MODEL, f"{name}: incorrect model configuration")
    require(normalized_config(cfg, root) == normalized_config(reference, root),
            f"{name}: training configuration differs from reference")
    training = cfg.get("training", {})
    require(training.get("seed") == seed and training.get("epochs") == 5,
            f"{name}: expected declared seed and five training epochs")
    require(training.get("checkpoint_metric") == "logged_ctcvr_ne"
            and training.get("keep_checkpoint", "best") == "best",
            f"{name}: expected best validation serving CTCVR NE checkpoint")
    require(cfg.get("data", {}).get("nonclick_sampling") is None,
            f"{name}: DIRECT study requires the full training population")
    block = cfg.get("logged_pctr", {})
    require(canonical_path(block.get("dcn_run", ""), root) == canonical_path(teacher["run_dir"], root)
            and block.get("score_column") == "score", f"{name}: different training teacher")
    recorded_teacher = run.get("logged_pctr", {})
    require(canonical_path(recorded_teacher.get("dcn_run", ""), root) == canonical_path(teacher["run_dir"], root),
            f"{name}: recorded training teacher differs")
    for split in ("train", "val"):
        expected = canonical_path(teacher["predictions"][split], root)
        configured = block.get("predictions", {}).get(split, "")
        recorded = recorded_teacher.get("files", {}).get(split, {}).get("file", "")
        require(canonical_path(configured, root) == expected and canonical_path(recorded, root) == expected,
                f"{name}: {split} teacher predictions differ")
    try:
        with (run_dir / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in history]
        scores = [float(row["val_score"]) for row in history]
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise ValueError(f"{name}: missing or invalid validation history") from error
    require(epochs == [1, 2, 3, 4, 5] and all(math.isfinite(value) for value in scores),
            f"{name}: expected five complete finite validation epochs")
    best = min(range(5), key=scores.__getitem__)
    selected = run.get("selected_on", {})
    require(selected.get("metric") == "logged_ctcvr_ne" and selected.get("split") == "val"
            and selected.get("epoch") == epochs[best],
            f"{name}: checkpoint is not the first minimum validation serving CTCVR NE")
    require(math.isclose(float(selected.get("value", math.nan)), scores[best], rel_tol=1e-10, abs_tol=1e-12),
            f"{name}: selected validation value disagrees with history")

    checkpoint_hash = digest(run_dir / "best.pt")
    cache = read_json(run_dir / "predictions/test.provenance.json")
    identity = cache.get("identity", {})
    require(identity.get("checkpoint_sha256") == checkpoint_hash,
            f"{name}: stale native prediction checkpoint hash")
    require(identity.get("config_sha256") == digest(config_path),
            f"{name}: stale native prediction configuration hash")
    encoded = (Path(root) / cfg.get("data", {}).get("encoded_dir", "")).resolve()
    require(identity.get("version") == 1 and identity.get("encoded_dir") == str(encoded),
            f"{name}: native prediction encoded-data identity differs")
    inputs = identity.get("inputs", {})
    require("test/sample_id.npy" in inputs, f"{name}: missing native test input provenance")
    for relative, signature in inputs.items():
        path = (encoded / relative).resolve()
        require(path.is_relative_to(encoded) and path.is_file(),
                f"{name}: missing native encoded input {relative}")
        stat = path.stat()
        require(signature == [stat.st_size, stat.st_mtime_ns],
                f"{name}: native encoded input changed: {relative}")
    prediction_hash = digest(run_dir / "predictions/test.parquet")
    require(cache.get("sha256") == prediction_hash, f"{name}: damaged native test predictions")
    native = read_json(run_dir / "metrics/test.json")
    require(len(native) == 1, f"{name}: DIRECT has only native pCVR; no auxiliary CTR or q*v output")
    validate_metric(native[0], "pCVR", name)

    producer = canonical_path(teacher["run_dir"], root).parent.name
    logged_dir = run_dir / "metrics_logged" / producer
    provenance = read_json(logged_dir / "provenance.json")
    require(provenance.get("checkpoint_sha256") == checkpoint_hash,
            f"{name}: stale serving evaluation checkpoint hash")
    require(provenance.get("predictions_sha256") == prediction_hash,
            f"{name}: serving evaluation uses different native predictions")
    require(provenance.get("pctr_scores_sha256") == teacher["pctr_scores_sha256"],
            f"{name}: serving teacher prediction hash differs")
    require(canonical_path(provenance.get("logged_pctr", {}).get("dcn_run", ""), root)
            == canonical_path(teacher["run_dir"], root), f"{name}: different serving teacher")
    logged = read_json(logged_dir / "test.json")
    require(len(logged) == 1, f"{name}: expected one serving CTCVR test row")
    validate_metric(logged[0], "CTCVR-logged", name)
    require(canonical_path(logged[0].get("logged_pctr_run", ""), root) == canonical_path(teacher["run_dir"], root)
            and canonical_path(logged[0].get("logged_pctr_file", ""), root) == canonical_path(teacher["test_predictions"], root),
            f"{name}: serving metric row identifies a different teacher")
    test = run.get("data", {}).get("test", {})
    require((native[0]["rows"], native[0]["positives"]) == (test.get("n_clicks"), test.get("n_conversions"))
            and (logged[0]["rows"], logged[0]["positives"]) == (test.get("n_rows"), test.get("n_conversions"))
            and cache.get("rows") == test.get("n_rows"), f"{name}: inconsistent test census")
    return run, {"pCVR": native[0], "CTCVR-logged": logged[0]}


def report(study_dir, root):
    study_dir, root = Path(study_dir).resolve(), Path(root).resolve()
    manifest = read_json(study_dir / "manifest.json")
    require(manifest.get("seeds") == SEEDS and manifest.get("model") == MODEL
            and manifest.get("epochs") == 5 and manifest.get("checkpoint_metric") == "logged_ctcvr_ne",
            "Study must prescribe DIRECT seeds 42 through 51, five epochs, and validation serving CTCVR NE")
    items = manifest.get("runs", [])
    require(len(items) == 10 and sorted(item["seed"] for item in items) == SEEDS,
            "Expected exactly one pinned DIRECT run for each seed 42 through 51")
    teacher = manifest["teacher"]
    validate_teacher(teacher, root)
    groups, sources, per_seed, census, seen_paths = {}, [], [], {}, set()
    for item in sorted(items, key=lambda item: item["seed"]):
        location = item.get("run_dir")
        if location is None:
            record = read_json(root / item["run_record"])
            require(record.get("seed", item["seed"]) == item["seed"], "Run record contains a different seed")
            location = record.get("run_dir")
        require(bool(location), f"DIRECT seed {item['seed']}: no completed run directory")
        run_dir = (root / location).resolve()
        require(run_dir not in seen_paths, f"DIRECT seed {item['seed']}: duplicate pinned run directory")
        seen_paths.add(run_dir)
        run, metrics = validate_run(run_dir, item["seed"], root / manifest["reference_config"], teacher, root)
        sources.append({"variant": "direct", "seed": item["seed"], "run_dir": str(run_dir),
                        "selected_on": run["selected_on"]})
        for population in POPULATIONS:
            row = metrics[population]
            counts = row["rows"], row["positives"]
            require(census.setdefault(population, counts) == counts,
                    f"{run_dir}: inconsistent test census across seeds for {population}")
            per_seed.append({"variant": "direct", "seed": item["seed"], "run_dir": str(run_dir), **row})
            for metric in METRICS:
                groups.setdefault((population, metric), []).append(row[metric])
    summary = [{"variant": "direct", "population": population, "metric": metric, "n": 10,
                "mean": statistics.mean(values), "sem": statistics.stdev(values) / math.sqrt(10),
                "values": values} for (population, metric), values in groups.items()]
    result = {"design": {"model": MODEL, "objective": "mean BCE(z, stopgrad(p) * v)",
                         "seeds": SEEDS, "checkpoint_metric": "logged_ctcvr_ne", "teacher": teacher},
              "sources": sources, "summary": summary, "per_seed": per_seed}
    # All validation finishes before any output is replaced.
    output = study_dir / "seed_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (study_dir / "seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, "values": json.dumps(row["values"])})
    lookup = {(row["population"], row["metric"]): row for row in summary}
    lines = ["DIRECT test metrics: mean ± SEM across seeds 42–51 (n=10).",
             "SEM = sample standard deviation / sqrt(10), with a fixed teacher and test split.",
             "Checkpoints minimize validation serving CTCVR NE; test scores do not select checkpoints.",
             "DIRECT has no auxiliary pCTR tower or native q*v output.", "",
             "| Output | AUC | PR-AUC | NE | CE |", "| --- | ---: | ---: | ---: | ---: |"]
    for population in POPULATIONS:
        cells = []
        for metric in METRICS:
            row = lookup[population, metric]
            digits = 8 if metric == "pr_auc" else 6
            cells.append(f"{row['mean']:.{digits}f} ± {row['sem']:.{digits}f}")
        lines.append(f"| {population} | " + " | ".join(cells) + " |")
    (study_dir / "seed_summary.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        print(report(args.study_dir, args.project_root))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Cannot report DIRECT ten-seed study: {error}\n")
