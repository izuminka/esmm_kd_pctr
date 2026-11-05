"""Report the paper's evaluation table for a run.

Usage:
    python scripts/report_metrics.py --name dcn
    python scripts/report_metrics.py --run-dir artifacts/esmm_logged_pctr_kd05/20260916T232126Z

For each of train, val and test, scores <run>/predictions/<split>.parquet, or,
for a model configured not to export predictions, reads the rows already scored
in memory from <run>/metrics/<split>.json. Reads the latest run of the config
named by --name unless --run-dir names a run, and prints the table without writing an aggregate report. Computes AUC, PR-AUC, NE and CE per population, matching
the evaluation recipe in
paper/frozen_pctr_iclr2027.tex. Metrics are computed over the complete
population before rounding, so the test row is directly comparable to the
paper's reported values, which are printed alongside for reference.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from training.runs import read_run, resolve_run  # noqa: E402
from evaluation.metrics import compute_metrics


# Table 2, DCN-V2 pCTR row: the target this reproduction is measured against.
_PAPER_DCN_CTR = {"auc": 0.620567, "pr_auc": 0.059938, "ne": 0.980200, "ce": 1.014134}
_PAPER_TEST_CENSUS = {"rows": 43_016_614, "positives": 1_673_447}
_LABEL_COLUMNS = ("sample_id", "y", "z")


# Models reporting ESMM-shaped outputs (optional q, required v) rather than a
# single score. Logged-training models need p on train and validation because p
# participates in their objective and checkpoint metric.
ESMM_MODELS = ("esmm", "esmm_logged_pctr", "esmm_direct_product", "esmm_dcn")
LOGGED_PCTR_TRAINING_MODELS = ("esmm_logged_pctr", "esmm_direct_product")


def logged_ctcvr_splits(cfg: dict) -> frozenset[str]:
    """Logged-score models select on p*v; native ESMM uses it only on test."""
    evaluation = cfg.get("evaluation", {})
    if cfg["model"] not in ESMM_MODELS:
        return frozenset()
    # Native ESMM defaults to native metrics; explicit legacy flags still work.
    legacy_logged = cfg["model"] in LOGGED_PCTR_TRAINING_MODELS or "logged_pctr" in cfg
    splits = {"test"} if evaluation.get("is_logged_ctcvr", legacy_logged) else set()
    if cfg["model"] in LOGGED_PCTR_TRAINING_MODELS:
        splits.add("val")
    return frozenset(splits)


def populations(
    model: str,     # model name, which fixes the heads reported
    y,              # click label
    z,              # conversion label; also the CTCVR label
    scores: dict,   # prediction columns: q, v and p for ESMM, score for a single head
) -> dict:
    """Population name -> (score, label) arrays.

    `scores` holds the model's prediction columns: q and v for ESMM, score for a
    single-head model. An ESMM model with no auxiliary CTR head has no q, and
    drops the two populations scored on it. When the logged pCTR p is present,
    CTCVR-logged scores the product it is served with, p * v, beside the
    auxiliary-head product q * v.
    """
    if "v" in scores:
        clicked = y == 1
        rows = {}
        if "q" in scores:
            rows["CTR"] = (scores["q"], y)
            rows["CTCVR"] = (scores["q"] * scores["v"], y * z)
        if "p" in scores:
            rows["CTCVR-logged"] = (scores["p"] * scores["v"], y * z)
        rows["pCVR"] = (scores["v"][clicked], z[clicked])  # clicked impressions only
        return rows
    return {"CTR": (scores["score"], y)}


def metric_rows(
    split: str,    # train, val or test, recorded on every row
    model: str,    # model name, which fixes the heads reported
    y,             # click label
    z,             # conversion label; also the CTCVR label
    scores: dict,  # prediction columns, as in populations()
) -> list[dict]:
    """One row per population: its size, positives and the four metrics."""
    y, z = np.asarray(y), np.asarray(z)
    scores = {c: np.asarray(s) for c, s in scores.items()}
    rows = [
        {"split": split, "population": population,
         "rows": len(label), "positives": int(label.sum()), **compute_metrics(score, label)}
        for population, (score, label) in populations(model, y, z, scores).items()
    ]
    return rows


def _attach_logged_pctr_source(rows: list[dict], logged_pctr: dict | None) -> None:
    """Name the DCN run and file behind every CTCVR-logged row, from the run's own record.

    A logged-product row with no recorded source can't be traced to the scores it
    used, so it is refused rather than reported.
    """
    for r in rows:
        if r["population"] != "CTCVR-logged":
            continue
        source = (logged_pctr or {}).get("files", {}).get(r["split"])
        if source is None:
            raise SystemExit(f"{r['split']} CTCVR-logged has no logged_pctr source in run.json")
        expected = {"logged_pctr_run": logged_pctr["dcn_run"], "logged_pctr_file": source["file"]}
        for key, value in expected.items():
            if key in r and r[key] != value:
                raise ValueError(f"{r['split']}: logged producer disagrees with run metadata")
            r[key] = value


def main():
    ap = argparse.ArgumentParser(description="Report metrics from prediction parquets.")
    ap.add_argument("--artifacts", default=os.path.join(_ROOT_DIR, "artifacts"),
                    help="artifacts root, used to find the latest run")
    ap.add_argument("--model", default=None, choices=["dcn", *ESMM_MODELS],
                    help="which populations to report; defaults to the run's own model")
    ap.add_argument("--name", default=None,
                    help="config name whose latest run to read, e.g. esmm_logged_pctr_kd05")
    ap.add_argument("--run-dir", default=None, help="defaults to the latest run of --name")
    ap.add_argument("--include-logged", action="store_true",
                    help="display metrics_logged/<pctr_model> alongside native metrics")
    args = ap.parse_args()
    if not (args.run_dir or args.name):
        ap.error("pass --run-dir, or --name to take that config's latest run")
    # Runs live under their config's name, which need not be the model's: several
    # configs (cohorts, loss variants) share one model class.
    run_dir = args.run_dir or resolve_run(args.artifacts, args.name)
    print(f"Run: {run_dir}")
    model = args.model or read_run(run_dir)["model"]
    with open(os.path.join(run_dir, "config.yaml")) as f:
        report_cfg = yaml.safe_load(f)
    logged_splits = logged_ctcvr_splits(report_cfg)
    separated_logged = any(Path(run_dir, "metrics_logged").glob("*/test.json"))
    if separated_logged:
        logged_splits = logged_splits - {"test"}

    rows = []
    # A separate evaluator explicitly binds test predictions to this run.
    # Validation-only training must still ignore stray legacy test files.
    evaluated_test = os.path.exists(os.path.join(run_dir, "predictions", "test.provenance.json"))
    report_splits = (("train", "val", "test")
                     if report_cfg.get("training", {}).get("evaluate_test", True) or evaluated_test
                     else ("train", "val"))
    for split in report_splits:
        parquet = os.path.join(run_dir, "predictions", f"{split}.parquet")
        scored = os.path.join(run_dir, "metrics", f"{split}.json")
        if os.path.exists(scored):
            with open(scored) as f:
                stored_rows = json.load(f)
            before = json.dumps(stored_rows)
            split_rows = [r for r in stored_rows
                          if split in logged_splits or r["population"] != "CTCVR-logged"]
            _attach_logged_pctr_source(split_rows, read_run(run_dir).get("logged_pctr"))
            if json.dumps(stored_rows) != before:
                with open(scored, "w") as f:
                    json.dump(stored_rows, f, indent=2)
            rows += split_rows
        elif os.path.exists(parquet):
            df = pd.read_parquet(parquet)
            scores = {c: df[c].to_numpy() for c in df.columns if c not in _LABEL_COLUMNS}
            if split not in logged_splits:
                scores.pop("p", None)
            split_rows = metric_rows(split, model, df["y"].to_numpy(), df["z"].to_numpy(), scores)
            _attach_logged_pctr_source(split_rows, read_run(run_dir).get("logged_pctr"))
            os.makedirs(os.path.dirname(scored), exist_ok=True)
            with open(scored, "w") as f:
                json.dump(split_rows, f, indent=2)
            rows += split_rows
        else:
            print(f"  (no predictions or metrics for {split}, skipping)")

    if not rows:
        raise SystemExit(f"no predictions or metrics found in {run_dir}")
    _attach_logged_pctr_source(rows, read_run(run_dir).get("logged_pctr"))
    if args.include_logged:
        from evaluation.logged_ctcvr import logged_metric_rows
        rows += logged_metric_rows(run_dir)

    hdr = f"{'split':<6} {'pop':<12} {'rows':>12} {'pos':>10} " \
          f"{'AUC':>10} {'PR-AUC':>11} {'NE':>10} {'CE':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split']:<6} {r['population']:<12} {r['rows']:>12,} {r['positives']:>10,} "
              f"{r['auc']:>10.6f} {r['pr_auc']:>11.6f} {r['ne']:>10.6f} {r['ce']:>10.6f}"
              + (f"  [{r['pctr_model']}]" if "pctr_model" in r else ""))
    for r in rows:
        if r["population"] == "CTCVR-logged":
            print(f"  {r['split']} CTCVR-logged: p from dcn run {r['logged_pctr_run']}, "
                  f"{r['logged_pctr_file']}")

    if model == "dcn":
        p = _PAPER_DCN_CTR
        print("-" * len(hdr))
        print(f"{'paper':<6} {'CTR':<12} {_PAPER_TEST_CENSUS['rows']:>12,} "
              f"{_PAPER_TEST_CENSUS['positives']:>10,} "
              f"{p['auc']:>10.6f} {p['pr_auc']:>11.6f} {p['ne']:>10.6f} {p['ce']:>10.6f}")
        test = next((r for r in rows if r["split"] == "test"), None)
        if test:
            print(f"{'delta':<6} {'':<12} {test['rows'] - _PAPER_TEST_CENSUS['rows']:>12,} "
                  f"{test['positives'] - _PAPER_TEST_CENSUS['positives']:>10,} "
                  f"{test['auc'] - p['auc']:>+10.6f} {test['pr_auc'] - p['pr_auc']:>+11.6f} "
                  f"{test['ne'] - p['ne']:>+10.6f} {test['ce'] - p['ce']:>+10.6f}")


if __name__ == "__main__":
    main()
