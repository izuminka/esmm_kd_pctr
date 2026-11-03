"""Pearson(p, q) and Pearson(q*v, p*v) for full-data manuscript variants.

Reads only saved predictions. Reuses the completed alignment audit's checkpoint
identity, verifies each prediction/cache-provenance hash without rehashing model
weights, and matches every sample ID and label to the fixed teacher. A streaming
float64 centered-covariance merge is independently compared with scipy.pearsonr
on the same loaded probabilities, avoiding another Parquet decoding pass.
Both products use the same model/seed v, with every head cast to float64 before
multiplication and no clipping. Each correlation uses all test impressions.
An optional variant selects ten runs to compute while validating the full
sixty-run source census and the prior audit's config/cache-sidecar identities.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.stats import pearsonr

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts import report_ctr_alignment as alignment

ROOT = CODE.parent
TOLERANCE = 1e-12


@dataclass
class CenteredCovariance:
    """Chan's centered moment merge; unequal batches are weighted by row count."""

    rows: int = 0
    mean_p: float = 0.
    mean_q: float = 0.
    m2_p: float = 0.
    m2_q: float = 0.
    cross: float = 0.
    p_min: float = math.inf
    p_max: float = -math.inf
    q_min: float = math.inf
    q_max: float = -math.inf

    def add(self, p, q):
        p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
        alignment.require(p.ndim == q.ndim == 1 and len(p) == len(q),
                          "correlation inputs need equal one-dimensional lengths")
        alignment.require(np.all(np.isfinite(p)) and np.all(np.isfinite(q)),
                          "correlation input is nonfinite")
        if not len(p):
            return
        n = len(p)
        self.p_min, self.p_max = min(self.p_min, float(p.min())), max(self.p_max, float(p.max()))
        self.q_min, self.q_max = min(self.q_min, float(q.min())), max(self.q_max, float(q.max()))
        mean_p, mean_q = float(p.mean()), float(q.mean())
        p_centered, q_centered = p-mean_p, q-mean_q
        p_sum = float(np.einsum('i,i->', p_centered, p_centered))
        q_sum = float(np.einsum('i,i->', q_centered, q_centered))
        cross = float(np.einsum('i,i->', p_centered, q_centered))
        total = self.rows+n
        dp, dq = mean_p-self.mean_p, mean_q-self.mean_q
        weight = self.rows*n/total
        self.m2_p += p_sum+dp*dp*weight
        self.m2_q += q_sum+dq*dq*weight
        self.cross += cross+dp*dq*weight
        self.mean_p += dp*n/total
        self.mean_q += dq*n/total
        self.rows = total

    def result(self):
        alignment.require(self.rows >= 2, "Pearson correlation requires at least two rows")
        alignment.require(self.p_min < self.p_max and self.q_min < self.q_max
                          and self.m2_p > 0 and self.m2_q > 0,
                          "Pearson correlation is undefined for constant probabilities")
        value = self.cross/math.sqrt(self.m2_p)/math.sqrt(self.m2_q)
        alignment.require(math.isfinite(value) and abs(value) <= 1+TOLERANCE,
                          "invalid Pearson correlation aggregate")
        return float(np.clip(value, -1., 1.))


def correlation_for_file(path, teacher, batch_size):
    """Decode q/v once; retain float64 heads/products for independent SciPy checks."""
    alignment.require(batch_size > 0, "batch size must be positive")
    alignment.check_probability(teacher["score"], "teacher p")
    file = alignment.parquet(path, "q")
    alignment.require("v" in file.schema_arrow.names
                      and file.schema_arrow.field("v").type == alignment.pa.float32(),
                      f"{path}: expected v column of type float32")
    rows = len(teacher["score"])
    alignment.require(file.metadata.num_rows == rows, "teacher/student row-count mismatch")
    q_all = np.empty(rows, dtype=np.float64)
    qv_all, pv_all = np.empty_like(q_all), np.empty_like(q_all)
    moments, products, offset = CenteredCovariance(), CenteredCovariance(), 0
    for batch in file.iter_batches(batch_size=batch_size, columns=["sample_id", "y", "z", "q", "v"]):
        student = alignment.read_batch(batch, "q")
        column = batch.column(batch.schema.get_field_index("v"))
        alignment.require(column.null_count == 0, "v: null values in predictions")
        v = column.to_numpy(zero_copy_only=False)
        alignment.check_probability(v, "student v")
        end = offset+batch.num_rows
        reference = {name: values[offset:end] for name, values in teacher.items()}
        alignment.check_alignment(reference, student, offset)
        q_all[offset:end] = student["q"]
        p, q, v = (np.asarray(values, dtype=np.float64)
                   for values in (reference["score"], student["q"], v))
        qv_all[offset:end], pv_all[offset:end] = q*v, p*v
        moments.add(p, q)
        products.add(qv_all[offset:end], pv_all[offset:end])
        offset = end
    alignment.require(offset == rows, "student Parquet row count changed")
    value, product_value = moments.result(), products.result()
    reference = float(pearsonr(np.asarray(teacher["score"], dtype=np.float64), q_all).statistic)
    product_reference = float(pearsonr(qv_all, pv_all).statistic)
    difference, product_difference = abs(value-reference), abs(product_value-product_reference)
    for label, result, independent, delta in (("p/q", value, reference, difference),
                                              ("q*v/p*v", product_value, product_reference, product_difference)):
        alignment.require(math.isfinite(independent) and delta <= TOLERANCE,
                          f"{label} streaming/SciPy Pearson mismatch: {result} versus {independent}")
    return {"rows": rows, "pearson": value, "scipy_pearson": reference,
            "absolute_difference": difference, "product_pearson": product_value,
            "product_scipy_pearson": product_reference, "product_absolute_difference": product_difference}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(summary_path, alignment_path, teacher_path, output_dir, batch_size=1048576,
           variant=None, project_root=ROOT):
    summary_path, alignment_path, teacher_path, output_dir = map(
        lambda path: Path(path).resolve(), (summary_path, alignment_path, teacher_path, output_dir))
    alignment.require(batch_size > 0, "batch size must be positive")
    alignment.require(variant is None or variant in alignment.VARIANTS, f"unknown variant: {variant}")
    project_root = Path(project_root).resolve()
    alignment.require(not output_dir.exists() or not any(output_dir.iterdir()),
                      "output directory must be new or empty")
    prior = alignment.read_json(alignment_path)
    summary = alignment.read_json(summary_path)
    summary_hash, prior_hash = alignment.digest(summary_path), alignment.digest(alignment_path)
    alignment.require(summary_hash == prior["inputs"]["summary_sha256"],
                      "summary differs from the completed alignment audit")
    helper_hash = alignment.digest(Path(alignment.__file__))
    alignment.require(helper_hash == prior["inputs"]["script_sha256"],
                      "alignment helper source differs from the completed audit")
    # Make relative artifact paths independent of where this script is frozen.
    source_summary = {**summary, "sources": [
        {**row, "run_dir": str(alignment.canonical_path(row["run_dir"], project_root))}
        for row in summary["sources"]]}
    sources = alignment.study_sources(source_summary)
    identities = {(row["variant"], row["seed"]): row for row in prior["run_provenance"]}
    alignment.require(len(identities) == len(prior["run_provenance"]) == len(sources) == 60,
                      "prior alignment provenance must contain sixty distinct runs")
    alignment.require(set(identities) == {(row["variant"], row["seed"]) for row in sources},
                      "prior alignment provenance differs from the sixty source runs")
    alignment.require(teacher_path == Path(prior["inputs"]["teacher_predictions"]).resolve(),
                      "teacher differs from the completed alignment audit")
    teacher_state = teacher_path.stat()
    teacher_hash = alignment.digest(teacher_path)
    alignment.require(teacher_hash == prior["inputs"]["teacher_sha256"], "teacher prediction hash mismatch")
    teacher = alignment.load_teacher(teacher_path, batch_size)
    # One shared float64 p vector lets SciPy perform its independent full-array
    # calculation without rereading predictions; every q is released after its run.
    teacher["score"] = teacher["score"].astype(np.float64)
    rows, provenance = [], []
    for source in sources:
        key = source["variant"], source["seed"]
        alignment.require(key in identities, f"missing prior alignment provenance for {key}")
        identity = identities[key]
        run = alignment.canonical_path(source["run_dir"], project_root)
        alignment.require(run == Path(identity["run_dir"]).resolve(), f"run identity changed for {key}")
        record = alignment.read_json(run/"run.json")
        alignment.require(record.get("status") == "FINISHED"
                          and record.get("selected_on") == source["selected_on"] == identity["selected_on"],
                          f"selected checkpoint identity changed for {key}")
        alignment.require(alignment.digest(run/"config.yaml") == identity["config_sha256"],
                          f"configuration changed for {key}")
        path, sidecar = run/"predictions/test.parquet", run/"predictions/test.provenance.json"
        alignment.require(alignment.digest(sidecar) == identity["cache_provenance_sha256"],
                          f"cache provenance changed for {key}")
        meta = alignment.read_json(sidecar)
        alignment.require(meta["identity"]["checkpoint_sha256"] == identity["checkpoint_sha256"]
                          and meta["identity"]["config_sha256"] == identity["config_sha256"]
                          and meta["sha256"] == identity["predictions_sha256"],
                          f"cache checkpoint/config identity changed for {key}")
        alignment.require(meta["rows"] == len(teacher["score"]), f"cache census differs for {key}")
        before = path.stat()
        if variant is not None and key[0] != variant:
            continue  # All identities are validated; only selected caches are rehashed/decoded.
        cache_hash = alignment.digest(path)
        alignment.require(cache_hash == identity["predictions_sha256"] == meta["sha256"],
                          f"prediction cache hash mismatch for {key}")
        row = {"variant": key[0], "seed": key[1], **correlation_for_file(path, teacher, batch_size)}
        after = path.stat()
        alignment.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                          f"prediction cache changed during correlation for {key}")
        rows.append(row)
        provenance.append({"variant": key[0], "seed": key[1], "run_dir": str(run),
                           "predictions": str(path), "predictions_sha256": cache_hash,
                           "prediction_bytes": before.st_size, "prediction_mtime_ns": before.st_mtime_ns,
                           "cache_provenance_sha256": identity["cache_provenance_sha256"],
                           "checkpoint_sha256_from_prior_audit": identity["checkpoint_sha256"],
                           "config_sha256": identity["config_sha256"], "selected_on": source["selected_on"]})
        print(f"{key[0]} seed {key[1]}: n={row['rows']} Pearson(p,q)={row['pearson']:.12g} "
              f"Pearson(q*v,p*v)={row['product_pearson']:.12g} "
              f"SciPy_deltas={row['absolute_difference']:.3g}/{row['product_absolute_difference']:.3g}", flush=True)
    final_state = teacher_path.stat()
    alignment.require((teacher_state.st_size, teacher_state.st_mtime_ns) == (final_state.st_size, final_state.st_mtime_ns),
                      "teacher cache changed during correlation")
    alignment.require(alignment.digest(summary_path) == summary_hash and alignment.digest(alignment_path) == prior_hash,
                      "study inputs changed during correlation")
    grouped = []
    variants = (variant,) if variant is not None else alignment.VARIANTS
    for selected_variant in variants:
        group = [row for row in rows if row["variant"] == selected_variant]
        alignment.require([row["seed"] for row in group] == alignment.SEEDS,
                          f"{selected_variant}: expected ten distinct seeds 42--51")
        values = [row["pearson"] for row in group]
        references = [row["scipy_pearson"] for row in group]
        product_values = [row["product_pearson"] for row in group]
        grouped.append({"variant": selected_variant, "n": 10, "mean": statistics.mean(values),
                        "sem": statistics.stdev(values)/math.sqrt(10),
                        "scipy_mean": statistics.mean(references), "values": values,
                        "product_mean": statistics.mean(product_values),
                        "product_sem": statistics.stdev(product_values)/math.sqrt(10),
                        "product_scipy_mean": statistics.mean(row["product_scipy_pearson"] for row in group),
                        "product_values": product_values})
    result = {"definition": {"metric": "Pearson correlation between raw frozen teacher p and auxiliary q, on all test impressions",
                             "product_metric": "Pearson correlation between raw native q*v and serving p*v, on all test impressions",
                             "product_arithmetic": "cast original p, q and v to float64 before multiplication; both products use the same model/seed v",
                             "rows": "every test impression, including nonclicks; no filtering by label",
                             "algorithm": "float64 batch-centered covariance with Chan merging",
                             "independent_check": "scipy.stats.pearsonr on the same full float64 p/q and q*v/p*v arrays",
                             "absolute_tolerance": TOLERANCE, "clipping": None,
                             "aggregation": "arithmetic mean over ten training seeds; SEM=sample SD/sqrt(10)",
                             "constant_variance": "undefined; raises an error",
                             "row_alignment": "exact sample_id/y/z equality for every row; teacher IDs strictly increasing and unique",
                             "checkpoint_validation": "reuse checkpoint SHA from completed alignment audit, verify cache/config identities; do not rehash model weights",
                             "training_or_inference_performed": False},
              "inputs": {"summary": str(summary_path), "summary_sha256": summary_hash,
                         "alignment": str(alignment_path), "alignment_sha256": prior_hash,
                         "teacher_predictions": str(teacher_path), "teacher_sha256": teacher_hash,
                         "script_sha256": alignment.digest(Path(__file__)), "alignment_helper_sha256": helper_hash,
                         "project_root": str(project_root), "variants": list(variants),
                         "numpy_version": np.__version__, "scipy_version": __import__('scipy').__version__},
              "per_seed": rows, "summary": grouped, "run_provenance": provenance,
              "validation": {"status": "PASS", "runs": len(rows), "source_runs": len(sources),
                             "full_source_validation": "all sixty source/config/cache-sidecar identities; rehash/decode prediction bytes only for selected runs",
                             "max_absolute_streaming_scipy_difference": max(row["absolute_difference"] for row in rows),
                             "max_absolute_product_streaming_scipy_difference": max(row["product_absolute_difference"] for row in rows)}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"correlation.json").write_text(json.dumps(result, indent=2)+"\n")
    write_csv(output_dir/"per_seed.csv", rows)
    write_csv(output_dir/"summary.csv", [{key:value for key,value in row.items()
                                        if key not in ("values", "product_values")} for row in grouped])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT/"paper/main/results/ten_seed_summary.json")
    parser.add_argument("--alignment", type=Path, default=ROOT/"artifacts/alignment_studies/20261017_all_full_with_product/alignment.json")
    parser.add_argument("--teacher-predictions", type=Path, default=ROOT/alignment.TEACHER/"predictions/test.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1048576)
    parser.add_argument("--variant", choices=alignment.VARIANTS,
                        help="compute one variant's ten seeds while validating all sixty source identities")
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="project root for relative source paths; pass explicitly when running a frozen copy")
    args = parser.parse_args()
    try:
        report(args.summary, args.alignment, args.teacher_predictions, args.output_dir, args.batch_size,
               variant=args.variant, project_root=args.project_root)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"CTR correlation audit failed: {error}\n")


if __name__ == "__main__":
    main()
