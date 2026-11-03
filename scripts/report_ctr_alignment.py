"""Audit frozen-pCTR/auxiliary-pCTR alignment from existing test predictions.

No training or prediction is performed. Reads the ten-seed manuscript study,
checks prediction/checkpoint/config and fixed-teacher provenance, and streams
all test rows with exact sample-ID and label matching. BCE and KL use natural
logarithms (nats per impression), raw teacher probabilities, and q clipped to
[1e-7, 1-1e-7]. Also reports native-product q*v as a soft target for serving
p*v, without clipping either product and using the exact 0*log(0) convention.
Product targets and predictions are multiplied
in float64 from the original float32 heads. Reports per-seed values and
arithmetic means +/- one SEM; native-product entropy varies across models.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(42, 52))
VARIANTS = ("esmm", "esmm_dcn", "kd03", "kd05", "kd07", "kd0")
KD_ALPHA = {"kd03": .3, "kd05": .5, "kd07": .7, "kd0": 0.}
TEACHER = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"
TEACHER_ALIAS = "artifacts/dcn_matched_stats_wd3e6/20260917T232959Z"
Q_EPSILON = 1e-7
ALIGNMENT_METRICS = ("soft_bce", "teacher_entropy", "bernoulli_kl", "mae")
PRODUCT_ALIGNMENT_METRICS = ("product_soft_bce", "product_native_entropy",
                             "product_bernoulli_kl", "product_mae")
REPORT_METRICS = (*ALIGNMENT_METRICS, *PRODUCT_ALIGNMENT_METRICS,
                  "pcvr_pr_auc", "serving_pctcvr_pr_auc")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    """Hash inputs without loading a prediction file or checkpoint into RAM."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_path(path, root=ROOT):
    path = (Path(root) / path).resolve()
    alias = (Path(root) / TEACHER_ALIAS).resolve()
    if path == alias or path.is_relative_to(alias):
        return (Path(root) / TEACHER / path.relative_to(alias)).resolve()
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_probability(values, name):
    values = np.asarray(values)
    require(values.ndim == 1, f"{name}: expected one-dimensional probabilities")
    require(np.all(np.isfinite(values)), f"{name}: nonfinite probability")
    require(np.all((values >= 0) & (values <= 1)), f"{name}: probability outside [0,1]")


def check_ordered_ids(ids, previous=None):
    """Strict monotonicity establishes both ordering and uniqueness, across batches."""
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype == np.dtype("int64"),
            "sample_id must be a one-dimensional int64 array")
    require(len(ids) == 0 or previous is None or ids[0] > previous,
            "teacher sample_id is duplicated or not strictly increasing across batches")
    require(np.all(ids[1:] > ids[:-1]),
            "teacher sample_id is duplicated or not strictly increasing")
    return int(ids[-1]) if len(ids) else previous


def check_alignment(teacher, student, offset=0):
    """Never infer identity from equal row counts or assume matching order."""
    for name in ("sample_id", "y", "z"):
        expected, observed = teacher[name], student[name]
        require(len(expected) == len(observed), f"{name}: row-count mismatch at offset {offset}")
        unequal = np.flatnonzero(expected != observed)
        require(len(unequal) == 0,
                f"{name}: alignment mismatch at row {offset + int(unequal[0]) if len(unequal) else offset}")


@dataclass
class AlignmentSums:
    rows: int = 0
    clipped_q: int = 0
    soft_bce_sum: float = 0.0
    teacher_entropy_sum: float = 0.0
    absolute_error_sum: float = 0.0

    def add(self, p, q):
        check_probability(p, "teacher p")
        check_probability(q, "student q")
        require(len(p) == len(q), "p/q row-count mismatch")
        p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
        q_safe = np.clip(q, Q_EPSILON, 1 - Q_EPSILON)
        self.rows += len(p)
        self.clipped_q += int(np.count_nonzero(q != q_safe))
        self.soft_bce_sum += float(np.sum(-p * np.log(q_safe) - (1-p) * np.log1p(-q_safe), dtype=np.float64))
        # Define 0 log(0) = 0, without clipping or changing teacher targets.
        interior = p[(p > 0) & (p < 1)]
        self.teacher_entropy_sum += float(np.sum(
            -interior * np.log(interior) - (1-interior) * np.log1p(-interior), dtype=np.float64))
        self.absolute_error_sum += float(np.sum(np.abs(p-q), dtype=np.float64))

    def result(self):
        require(self.rows > 0, "empty prediction population")
        bce = self.soft_bce_sum / self.rows
        entropy = self.teacher_entropy_sum / self.rows
        return {"rows": self.rows, "q_clipped_count": self.clipped_q,
                "soft_bce": bce, "teacher_entropy": entropy,
                "bernoulli_kl": bce - entropy,
                "mae": self.absolute_error_sum / self.rows}


class ProductAlignmentSums:
    """Native q*v target, serving p*v prediction; both use the same raw v."""

    def __init__(self):
        self.rows = 0
        self.bce_sum = self.entropy_sum = self.mae_sum = 0.
        self.prediction_min, self.prediction_max = math.inf, -math.inf
        self.prediction_zero_count = self.prediction_one_count = 0

    def add(self, p, q, v):
        for values, name in [(p, "teacher p"), (q, "student q"), (v, "student v")]:
            check_probability(values, name)
        require(len(p) == len(q) == len(v), "p/q/v row-count mismatch")
        # Cast each original head before multiplication, not after a float32 product.
        p, q, v = (np.asarray(values, dtype=np.float64) for values in (p, q, v))
        target, prediction = q*v, p*v
        # Evaluate only nonzero coefficients, so 0*log(0)=0 at endpoints.
        terms = np.zeros(len(target), dtype=np.float64)
        positive, negative = target > 0, target < 1
        with np.errstate(divide="ignore"):
            terms[positive] -= target[positive] * np.log(prediction[positive])
            terms[negative] -= (1-target[negative]) * np.log1p(-prediction[negative])
        require(np.all(np.isfinite(terms)),
                "raw product BCE is nonfinite: incompatible native target and serving prediction endpoint")
        interior = target[(target > 0) & (target < 1)]
        self.rows += len(target)
        self.bce_sum += float(np.sum(terms, dtype=np.float64))
        self.entropy_sum += float(np.sum(-interior*np.log(interior) - (1-interior)*np.log1p(-interior), dtype=np.float64))
        self.mae_sum += float(np.sum(np.abs(target-prediction), dtype=np.float64))
        if len(prediction):
            self.prediction_min = min(self.prediction_min, float(prediction.min()))
            self.prediction_max = max(self.prediction_max, float(prediction.max()))
        self.prediction_zero_count += int(np.count_nonzero(prediction == 0))
        self.prediction_one_count += int(np.count_nonzero(prediction == 1))

    def result(self):
        require(self.rows > 0, "empty product prediction population")
        bce, entropy = self.bce_sum/self.rows, self.entropy_sum/self.rows
        require(math.isfinite(bce) and math.isfinite(entropy), "raw product aggregate is nonfinite")
        return {"product_prediction_min": self.prediction_min,
                "product_prediction_max": self.prediction_max,
                "product_prediction_zero_count": self.prediction_zero_count,
                "product_prediction_one_count": self.prediction_one_count,
                "product_soft_bce": bce, "product_native_entropy": entropy,
                "product_bernoulli_kl": bce-entropy, "product_mae": self.mae_sum/self.rows}


def read_batch(batch, probability_column):
    result = {}
    for name in ("sample_id", "y", "z", probability_column):
        column = batch.column(batch.schema.get_field_index(name))
        require(column.null_count == 0, f"{name}: null values in predictions")
        result[name] = column.to_numpy(zero_copy_only=False)
    for name in ("y", "z"):
        require(np.all((result[name] == 0) | (result[name] == 1)), f"{name}: labels must be binary")
    require(np.all(result["z"] <= result["y"]), "conversion label exceeds click label")
    check_probability(result[probability_column], probability_column)
    return result


def parquet(path, probability_column):
    file = pq.ParquetFile(path)
    for name, dtype in [("sample_id", pa.int64()), ("y", pa.int8()), ("z", pa.int8()),
                        (probability_column, pa.float32())]:
        require(name in file.schema_arrow.names and file.schema_arrow.field(name).type == dtype,
                f"{path}: expected {name} column of type {dtype}")
    return file


def load_teacher(path, batch_size):
    file = parquet(path, "score")
    rows = file.metadata.num_rows
    require(rows > 0, "teacher predictions are empty")
    arrays = {name: np.empty(rows, dtype=dtype) for name, dtype in
              [("sample_id", np.int64), ("y", np.int8), ("z", np.int8), ("score", np.float32)]}
    offset, previous = 0, None
    for batch in file.iter_batches(batch_size=batch_size, columns=list(arrays)):
        values = read_batch(batch, "score")
        previous = check_ordered_ids(values["sample_id"], previous)
        end = offset + batch.num_rows
        for name in arrays:
            arrays[name][offset:end] = values[name]
        offset = end
    require(offset == rows, "teacher Parquet row count changed")
    return arrays


def alignment_for_file(path, teacher, batch_size):
    file = parquet(path, "q")
    require(file.metadata.num_rows == len(teacher["score"]), f"{path}: teacher/student row-count mismatch")
    # The same v head supplies both native and serving product probabilities.
    require("v" in file.schema_arrow.names and file.schema_arrow.field("v").type == pa.float32(),
            f"{path}: expected v column of type float32")
    totals, products, offset = AlignmentSums(), ProductAlignmentSums(), 0
    for batch in file.iter_batches(batch_size=batch_size, columns=["sample_id", "y", "z", "q", "v"]):
        student = read_batch(batch, "q")
        v = batch.column(batch.schema.get_field_index("v"))
        require(v.null_count == 0, "v: null values in predictions")
        v_values = v.to_numpy(zero_copy_only=False)
        check_probability(v_values, "student v")
        end = offset + batch.num_rows
        reference = {name: values[offset:end] for name, values in teacher.items()}
        check_alignment(reference, student, offset)
        totals.add(reference["score"], student["q"])
        products.add(reference["score"], student["q"], v_values)
        offset = end
    require(offset == len(teacher["score"]), "student Parquet row count changed")
    return {**totals.result(), **products.result()}


def check_configuration(source, run, config, teacher_path, root=ROOT):
    """Keep native ESMM/DCN selection distinct from each KD serving criterion."""
    variant, seed = source["variant"], source["seed"]
    require(variant in VARIANTS, f"unknown full-population variant: {variant}")
    name = f"{variant} seed {seed}"
    require(run.get("status") == "FINISHED", f"{name}: run is not FINISHED")
    require(run.get("selected_on") == source.get("selected_on"), f"{name}: selected checkpoint differs from summary")
    training = config.get("training", {})
    criterion = "logged_ctcvr_ne" if variant in KD_ALPHA else "ctcvr_ne"
    require(training.get("seed") == seed and training.get("epochs") == 5, f"{name}: wrong seed or epoch cap")
    expected_model = "esmm_logged_pctr" if variant in KD_ALPHA else variant
    require(config.get("model") == expected_model, f"{name}: wrong model")
    require(not config.get("data", {}).get("nonclick_sampling"), f"{name}: run is not full-population")
    require(training.get("checkpoint_metric") == criterion and training.get("keep_checkpoint", "best") == "best",
            f"{name}: wrong checkpoint selection rule")
    selected = run["selected_on"]
    require(selected.get("metric") == criterion and selected.get("split") == "val", f"{name}: wrong selected criterion")
    if variant in KD_ALPHA:
        require(config.get("model_params", {}).get("objective") == "kd"
                and config["model_params"].get("alpha") == KD_ALPHA[variant],
                f"{name}: wrong KD objective or alpha")
        require(canonical_path(config["logged_pctr"]["dcn_run"], root) == teacher_path.parent.parent,
                f"{name}: training teacher differs")
    if variant == "esmm_dcn":
        require(bool(config.get("model_params", {}).get("dcn_params")),
                f"{name}: missing joint DCN tower configuration")


def verify_inputs(source, summary, teacher_path, teacher_hash, root=ROOT):
    """Bind each cache to the declared model, selected checkpoint and test reports."""
    variant, seed = source["variant"], source["seed"]
    run_dir = canonical_path(source["run_dir"], root)
    run, config = read_json(run_dir / "run.json"), yaml.safe_load((run_dir / "config.yaml").read_text())
    name = f"{variant} seed {seed}"
    check_configuration(source, run, config, teacher_path, root)
    selected = run["selected_on"]
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    scores = [float(row["val_score"]) for row in history]
    require([int(row["epoch"]) for row in history] == [1, 2, 3, 4, 5]
            and all(math.isfinite(score) for score in scores), f"{name}: incomplete validation history")
    best = min(range(5), key=scores.__getitem__)
    require(selected.get("epoch") == best+1 and math.isclose(selected["value"], scores[best], rel_tol=1e-10, abs_tol=1e-12),
            f"{name}: checkpoint does not minimize validation criterion")
    cache = run_dir / "predictions/test.parquet"
    cache_provenance = run_dir / "predictions/test.provenance.json"
    provenance = read_json(cache_provenance)
    identity = provenance.get("identity", {})
    checkpoint_hash, config_hash, cache_hash = digest(run_dir / "best.pt"), digest(run_dir / "config.yaml"), digest(cache)
    require(identity.get("version") == 1 and identity.get("checkpoint_sha256") == checkpoint_hash,
            f"{name}: prediction cache checkpoint hash mismatch")
    require(identity.get("config_sha256") == config_hash, f"{name}: prediction cache config hash mismatch")
    require(provenance.get("sha256") == cache_hash, f"{name}: prediction cache hash mismatch")
    require(identity.get("inputs") and identity.get("encoded_dir"), f"{name}: missing cache input identity")
    for relative, expected in identity["inputs"].items():
        stat = (Path(identity["encoded_dir"]) / relative).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, f"{name}: encoded test input changed: {relative}")

    candidates = []
    for path in sorted((run_dir / "metrics_logged").glob("*/test.json")):
        prov = read_json(path.parent / "provenance.json")
        producer = prov.get("logged_pctr", {}).get("dcn_run")
        if producer is not None and canonical_path(producer, root) == teacher_path.parent.parent:
            require(prov.get("pctr_scores_sha256") == teacher_hash, f"{name}: logged teacher hash mismatch")
            require(prov.get("checkpoint_sha256") == checkpoint_hash, f"{name}: logged checkpoint hash mismatch")
            # Historical aliases predate prediction-cache hashes. They cannot
            # establish cache identity and must not substitute for a bound report.
            if prov.get("predictions_sha256") is None:
                continue
            require(prov.get("predictions_sha256") == cache_hash, f"{name}: logged evaluation uses a different prediction cache")
            candidates.append((path, prov))
    require(candidates, f"{name}: missing fixed-teacher serving report")
    # Documented historical/canonical duplicate reports must agree numerically.
    reports = [read_json(path) for path, _ in candidates]
    signatures = [[{k: row[k] for k in ["split", "population", "rows", "positives", "auc", "pr_auc", "ne", "ce"]}
                   for row in rows] for rows in reports]
    require(all(value == signatures[0] for value in signatures), f"{name}: conflicting serving reports")
    choice = next((index for index, (path, _) in enumerate(candidates) if path.parent.name == Path(TEACHER).parent.name), 0)
    logged_path, _ = candidates[choice]
    logged = reports[choice]
    require(len(logged) == 1 and logged[0].get("population") == "CTCVR-logged" and logged[0].get("split") == "test",
            f"{name}: invalid serving test population")
    native = read_json(run_dir / "metrics/test.json")
    pcvr = [row for row in native if row.get("population") == "pCVR" and row.get("split") == "test"]
    require(len(pcvr) == 1, f"{name}: missing/duplicate pCVR test report")
    scores = {}
    for population, row, key in [("pCVR", pcvr[0], "pcvr_pr_auc"), ("CTCVR-logged", logged[0], "serving_pctcvr_pr_auc")]:
        target = [r for r in summary["summary"] if (r["variant"], r["population"], r["metric"]) == (variant, population, "pr_auc")]
        require(len(target) == 1 and target[0]["n"] == 10 and len(target[0]["values"]) == 10,
                f"{name}: missing/invalid PR-AUC summary")
        require(target[0]["values"][SEEDS.index(seed)] == row["pr_auc"], f"{name}: PR-AUC differs from raw test report")
        require(math.isclose(statistics.mean(target[0]["values"]), target[0]["mean"], rel_tol=1e-14), f"{name}: PR-AUC mean is inconsistent")
        scores[key] = row["pr_auc"]
    hashes = {"checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
              "predictions_sha256": cache_hash, "cache_provenance_sha256": digest(cache_provenance),
              "native_test_sha256": digest(run_dir / "metrics/test.json"),
              "serving_test_sha256": digest(logged_path),
              "serving_provenance_sha256": digest(logged_path.parent / "provenance.json")}
    return cache, provenance["rows"], scores, {"run_dir": str(run_dir), "selected_on": selected,
            "prediction_cache": str(cache), "serving_metrics": str(logged_path), **hashes}


def aggregate(rows):
    result = []
    for variant in VARIANTS:
        group = sorted((row for row in rows if row["variant"] == variant), key=lambda row: row["seed"])
        require([row["seed"] for row in group] == SEEDS, f"{variant}: expected ten distinct seeds 42--51")
        for metric in REPORT_METRICS:
            values = [row[metric] for row in group]
            result.append({"variant": variant, "metric": metric, "n": 10,
                           "mean": statistics.mean(values), "sem": statistics.stdev(values)/math.sqrt(10), "values": values})
    return result


def study_sources(summary):
    """Require all six full-data variants, ten distinct seeds and distinct runs."""
    sources = [row for row in summary["sources"] if row["variant"] in VARIANTS]
    for variant in VARIANTS:
        require(sorted(row["seed"] for row in sources if row["variant"] == variant) == SEEDS,
                f"{variant}: expected ten distinct seeds 42--51")
    paths = [canonical_path(row["run_dir"]) for row in sources]
    require(len(set(paths)) == len(paths), "study reuses a run directory for multiple variant/seed entries")
    return sorted(sources, key=lambda row: (VARIANTS.index(row["variant"]), row["seed"]))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(summary_path, teacher_path, output_dir, batch_size=262144):
    summary_path, teacher_path, output_dir = Path(summary_path).resolve(), Path(teacher_path).resolve(), Path(output_dir).resolve()
    require(batch_size > 0, "batch size must be positive")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "output directory must be new or empty")
    summary = read_json(summary_path)
    sources = study_sources(summary)
    missing = [str(canonical_path(source["run_dir"]) / relative) for source in sources
               for relative in ("predictions/test.parquet", "predictions/test.provenance.json")
               if not (canonical_path(source["run_dir"]) / relative).is_file()]
    require(not missing, "missing prediction caches; export the fixed checkpoints first:\n" + "\n".join(missing))
    design_path = ROOT / "paper/main/results/experiment_design.json"
    design = read_json(design_path)
    require(teacher_path == canonical_path(design["fixed_teacher_run"]) / "predictions/test.parquet", "wrong fixed serving teacher")
    teacher_hash = digest(teacher_path)
    require(teacher_hash == design["teacher_input_sha256"][TEACHER + "/predictions/test.parquet"], "teacher prediction hash mismatch")
    teacher = load_teacher(teacher_path, batch_size)
    teacher_state = teacher_path.stat()
    rows, identities = [], []
    for source in sources:
        cache, expected_rows, scores, identity = verify_inputs(source, summary, teacher_path, teacher_hash)
        require(expected_rows == len(teacher["score"]), "cache provenance census differs from teacher")
        before = cache.stat()
        row = {"variant": source["variant"], "seed": source["seed"],
               **alignment_for_file(cache, teacher, batch_size), **scores}
        after = cache.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "prediction cache changed during analysis")
        rows.append(row)
        identities.append({"variant": source["variant"], "seed": source["seed"], **identity})
        print(f"{source['variant']} seed {source['seed']}: n={row['rows']} BCE={row['soft_bce']:.10g} KL={row['bernoulli_kl']:.10g} product_BCE={row['product_soft_bce']:.10g} product_KL={row['product_bernoulli_kl']:.10g} clipped_q={row['q_clipped_count']} pv_min={row['product_prediction_min']:.4g} pv_zeros={row['product_prediction_zero_count']}", flush=True)
    final_teacher_state = teacher_path.stat()
    require((teacher_state.st_size, teacher_state.st_mtime_ns) == (final_teacher_state.st_size, final_teacher_state.st_mtime_ns), "teacher cache changed during analysis")
    grouped = aggregate(rows)
    paired = []
    for variant in VARIANTS[1:]:
        base = sorted((row for row in rows if row["variant"] == "esmm"), key=lambda row: row["seed"])
        compared = sorted((row for row in rows if row["variant"] == variant), key=lambda row: row["seed"])
        for metric in REPORT_METRICS:
            differences = [b[metric]-a[metric] for a,b in zip(base,compared)]
            baseline = statistics.mean(row[metric] for row in base)
            paired.append({"variant": variant, "reference_variant": "esmm", "metric": metric,
                           "variant_minus_esmm_mean": statistics.mean(differences),
                           "paired_difference_sem": statistics.stdev(differences)/math.sqrt(10),
                           "relative_mean_change_percent": 100*statistics.mean(differences)/baseline if baseline else None})
    product_diagnostics = {variant: {
        "min": min(row["product_prediction_min"] for row in rows if row["variant"] == variant),
        "max": max(row["product_prediction_max"] for row in rows if row["variant"] == variant),
        "zero_count_across_seeds": sum(row["product_prediction_zero_count"] for row in rows if row["variant"] == variant),
        "one_count_across_seeds": sum(row["product_prediction_one_count"] for row in rows if row["variant"] == variant)} for variant in VARIANTS}
    result = {"definition": {"soft_bce": "mean[-p*ln(clip(q))-(1-p)*ln(1-clip(q))]", "teacher_entropy": "mean[-p*ln(p)-(1-p)*ln(1-p)], with 0*ln(0)=0", "bernoulli_kl": "soft_bce - teacher_entropy", "mae": "mean(abs(p-q)); raw p and q", "product_soft_bce": "mean[-(q*v)*ln(p*v)-(1-q*v)*ln(1-p*v)]; native product is the soft target, serving product is the prediction; 0*ln(0)=0", "product_native_entropy": "mean[H(q*v)]; varies by model and seed", "product_bernoulli_kl": "product_soft_bce - product_native_entropy = mean KL(Bern(q*v)||Bern(p*v))", "product_mae": "mean(abs(q*v-p*v)); raw products", "product_arithmetic": "cast original float32 p, q and v to float64 before multiplication; use the same model/seed v in both products", "units": "natural-log nats per test impression for BCE, entropy and KL", "q_clip": [Q_EPSILON, 1-Q_EPSILON], "product_prediction_clip": None, "teacher_clipped": False, "product_native_target_clipped": False, "aggregation": "arithmetic mean over ten training seeds; SEM=sample SD/sqrt(10)", "row_alignment": "exact ordered sample_id/y/z equality on every row; teacher IDs strictly increasing and unique", "training_or_prediction_performed": False},
              "inputs": {"summary": str(summary_path), "summary_sha256": digest(summary_path), "variants": list(VARIANTS),
                         "experiment_design": str(design_path), "experiment_design_sha256": digest(design_path),
                         "teacher_predictions": str(teacher_path), "teacher_sha256": teacher_hash,
                         "rows": len(teacher["score"]), "script_sha256": digest(Path(__file__))},
              "per_seed": rows, "summary": grouped, "paired_comparison": paired,
              "product_prediction_diagnostics": product_diagnostics, "run_provenance": identities}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment.json").write_text(json.dumps(result, indent=2)+"\n")
    write_csv(output_dir / "per_seed.csv", rows)
    write_csv(output_dir / "summary.csv", [{k:v for k,v in row.items() if k != "values"} for row in grouped])
    write_csv(output_dir / "paired_comparison.csv", paired)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "paper/main/results/ten_seed_summary.json")
    parser.add_argument("--teacher-predictions", type=Path, default=ROOT / TEACHER / "predictions/test.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=262144)
    args = parser.parse_args()
    try:
        report(args.summary, args.teacher_predictions, args.output_dir, args.batch_size)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Alignment audit failed: {error}\n")


if __name__ == "__main__":
    main()
