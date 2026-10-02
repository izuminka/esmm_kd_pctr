"""Build the pCTR-teacher/serving-factor swap tables from finished runs.

The design has three fixed conversion heads (v0, vA, vB) crossed with two
serving pCTR models (pA, pB).  vA and vB must be balanced-KD students with the
same student seed/configuration and different logged-pCTR teachers.
"""

import argparse
import csv
import json
from pathlib import Path

import yaml


METRICS = ("auc", "pr_auc", "ne", "ce")
LABELS = {"auc": "AUC", "pr_auc": "PR-AUC", "ne": "NE", "ce": "CE"}


def _run(run_dir: Path) -> tuple[dict, dict]:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if run["status"] != "FINISHED":
        raise ValueError(f"{run_dir}: run is {run['status']}")
    return run, cfg


def _logged_row(run_dir: Path, pctr_model: str) -> dict:
    path = run_dir / "metrics_logged" / pctr_model / "test.json"
    rows = json.loads(path.read_text())
    if len(rows) != 1 or rows[0]["population"] != "CTCVR-logged":
        raise ValueError(f"{path}: expected one CTCVR-logged row")
    return rows[0]


def build(v0: Path, va: Path, vb: Path, p_a: str, p_b: str) -> dict:
    run0, cfg0 = _run(v0)
    runa, cfga = _run(va)
    runb, cfgb = _run(vb)
    if cfg0["model"] != "esmm":
        raise ValueError("v0 must be a native ESMM baseline")
    for label, cfg in (("vA", cfga), ("vB", cfgb)):
        params = cfg.get("model_params", {})
        if (cfg["model"] != "esmm_logged_pctr" or params.get("objective") != "kd"
                or params.get("alpha") != 0.5):
            raise ValueError(f"{label} must be a balanced KD-0.5 student")
    comparable_a, comparable_b = dict(cfga), dict(cfgb)
    teacher_a = comparable_a.pop("logged_pctr")
    teacher_b = comparable_b.pop("logged_pctr")
    # The historical vA run predates the explicit separated-test flag.  This
    # controls only whether test scoring follows training and cannot affect the
    # fitted checkpoint, so exclude it from the training-recipe comparison.
    for comparable in (comparable_a, comparable_b):
        comparable["training"] = dict(comparable["training"])
        comparable["training"].pop("evaluate_test", None)
    if comparable_a != comparable_b:
        raise ValueError("vA and vB configurations differ beyond logged_pctr")
    if teacher_a["dcn_run"] == teacher_b["dcn_run"]:
        raise ValueError("vA and vB use the same teacher")

    run_dirs = {"v0": v0, "vA": va, "vB": vb}
    rows = []
    census = None
    pctr_runs = {}
    for v_label, run_dir in run_dirs.items():
        for p_label, pctr_model in (("pA", p_a), ("pB", p_b)):
            row = _logged_row(run_dir, pctr_model)
            counts = (row["rows"], row["positives"])
            if census is None:
                census = counts
            if counts != census:
                raise ValueError("crossed evaluations use different test populations")
            if pctr_runs.setdefault(p_label, row["logged_pctr_run"]) != row["logged_pctr_run"]:
                raise ValueError(f"{p_label} refers to inconsistent DCN runs")
            relation = ("baseline" if v_label == "v0" else
                        "matched" if v_label[-1] == p_label[-1] else "mismatched")
            rows.append({
                "conversion_head": v_label,
                "serving_pctr": p_label,
                "relation": relation,
                **{metric: row[metric] for metric in METRICS},
            })
    return {
        "design": {
            "v0": {"run_dir": str(v0), "selected_on": run0["selected_on"]},
            "vA": {"run_dir": str(va), "selected_on": runa["selected_on"],
                   "training_teacher": teacher_a["dcn_run"]},
            "vB": {"run_dir": str(vb), "selected_on": runb["selected_on"],
                   "training_teacher": teacher_b["dcn_run"]},
            "pA": pctr_runs["pA"],
            "pB": pctr_runs["pB"],
            "test_rows": census[0],
            "test_positives": census[1],
        },
        "rows": rows,
    }


def write(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher_swap.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "teacher_swap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])

    lookup = {(row["conversion_head"], row["serving_pctr"]): row
              for row in report["rows"]}
    lines = [
        "# DCN teacher/serving-factor swap",
        "",
        f"Complete test split: {report['design']['test_rows']:,} impressions, "
        f"{report['design']['test_positives']:,} conversions.",
        "Matched-teacher cells are bold. v0 is the native ESMM baseline.",
    ]
    digits = {"auc": 6, "pr_auc": 8, "ne": 6, "ce": 6}
    for metric in METRICS:
        lines += ["", f"## {LABELS[metric]}", "",
                  "| Conversion head | Serve with pA | Serve with pB |",
                  "| --- | ---: | ---: |"]
        for v_label in ("v0", "vA", "vB"):
            cells = []
            for p_label in ("pA", "pB"):
                row = lookup[v_label, p_label]
                value = f"{row[metric]:.{digits[metric]}f}"
                cells.append(f"**{value}**" if row["relation"] == "matched" else value)
            lines.append(f"| {v_label} | {cells[0]} | {cells[1]} |")
    lines += ["", "Runs:"]
    for label in ("pA", "pB", "v0", "vA", "vB"):
        source = report["design"][label]
        lines.append(f"- {label}: {source if isinstance(source, str) else source['run_dir']}")
    (output_dir / "teacher_swap.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-run", required=True, type=Path)
    parser.add_argument("--va-run", required=True, type=Path)
    parser.add_argument("--vb-run", required=True, type=Path)
    parser.add_argument("--p-a-model", required=True)
    parser.add_argument("--p-b-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(*(path.resolve() for path in (args.v0_run, args.va_run, args.vb_run)),
                   args.p_a_model, args.p_b_model)
    write(report, args.output_dir.resolve())
    print((args.output_dir / "teacher_swap.md").resolve().read_text())


if __name__ == "__main__":
    main()
