"""Aggregate the fixed-DCN teacher/serving swap over ESMM student seeds."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import yaml

from report_teacher_swap import LABELS, METRICS, build


def _student_seed(run_dir: Path) -> int:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return int(cfg["training"]["seed"])


def aggregate(seed_runs: list[tuple[int, Path, Path, Path]],
              p_a_model: str, p_b_model: str) -> dict:
    reports = []
    for seed, v0, va, vb in seed_runs:
        resolved = tuple(path.resolve() for path in (v0, va, vb))
        observed = tuple(_student_seed(path) for path in resolved)
        if observed != (seed, seed, seed):
            raise ValueError(
                f"student seed {seed} has run seeds {observed}, expected all {seed}")
        reports.append((seed, build(*resolved, p_a_model, p_b_model)))

    reference = reports[0][1]["design"]
    for seed, report in reports[1:]:
        design = report["design"]
        for key in ("pA", "pB", "test_rows", "test_positives"):
            if design[key] != reference[key]:
                raise ValueError(f"student seed {seed} differs on fixed design field {key}")

    summaries = []
    for conversion_head in ("v0", "vA", "vB"):
        for serving_pctr in ("pA", "pB"):
            seed_rows = []
            for seed, report in reports:
                row = next(
                    item for item in report["rows"]
                    if item["conversion_head"] == conversion_head
                    and item["serving_pctr"] == serving_pctr
                )
                seed_rows.append((seed, row))
            for metric in METRICS:
                values = [row[metric] for _, row in seed_rows]
                summaries.append({
                    "conversion_head": conversion_head,
                    "serving_pctr": serving_pctr,
                    "relation": seed_rows[0][1]["relation"],
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "sem": statistics.stdev(values) / math.sqrt(len(values)),
                    "values": values,
                })

    return {
        "design": {
            "student_seeds": [seed for seed, _ in reports],
            "pA": reference["pA"],
            "pB": reference["pB"],
            "test_rows": reference["test_rows"],
            "test_positives": reference["test_positives"],
            "runs": {
                str(seed): report["design"]
                for seed, report in reports
            },
        },
        "summary": summaries,
    }


def write(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher_swap_multiseed.json").write_text(
        json.dumps(report, indent=2) + "\n")
    with (output_dir / "teacher_swap_multiseed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["summary"][0]))
        writer.writeheader()
        writer.writerows(report["summary"])

    lookup = {
        (row["conversion_head"], row["serving_pctr"], row["metric"]): row
        for row in report["summary"]
    }
    digits = {"auc": 6, "pr_auc": 8, "ne": 6, "ce": 6}
    lines = [
        "# Multi-seed DCN teacher/serving-factor swap",
        "",
        f"Student seeds: {report['design']['student_seeds']}.",
        f"Complete test split: {report['design']['test_rows']:,} impressions, "
        f"{report['design']['test_positives']:,} conversions.",
    ]
    for metric in METRICS:
        precision = digits[metric]
        lines += [
            "",
            f"## {LABELS[metric]}",
            "",
            "| Conversion head | Serve with pA | Serve with pB |",
            "| --- | ---: | ---: |",
        ]
        for conversion_head in ("v0", "vA", "vB"):
            cells = []
            for serving_pctr in ("pA", "pB"):
                row = lookup[conversion_head, serving_pctr, metric]
                cells.append(
                    f"{row['mean']:.{precision}f} ± {row['sem']:.{precision}f}")
            lines.append(f"| {conversion_head} | {cells[0]} | {cells[1]} |")
    (output_dir / "teacher_swap_multiseed.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-run", action="append", nargs=4, required=True,
        metavar=("SEED", "V0_RUN", "VA_RUN", "VB_RUN"),
        help="student seed followed by its baseline, pA-trained, and pB-trained runs",
    )
    parser.add_argument("--p-a-model", required=True)
    parser.add_argument("--p-b-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    seed_runs = [
        (int(seed), Path(v0), Path(va), Path(vb))
        for seed, v0, va, vb in args.seed_run
    ]
    if len({seed for seed, *_ in seed_runs}) != len(seed_runs):
        raise ValueError("student seeds must be unique")
    report = aggregate(seed_runs, args.p_a_model, args.p_b_model)
    write(report, args.output_dir.resolve())
    print((args.output_dir / "teacher_swap_multiseed.md").resolve().read_text())


if __name__ == "__main__":
    main()
