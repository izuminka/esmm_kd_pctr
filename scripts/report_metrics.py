"""Report the paper's evaluation table for a run.

Usage:
    python scripts/report_metrics.py --model dcn
    python scripts/report_metrics.py --model dcn --run-dir artifacts/dcn/20260916T020356Z

For each of train, val and test, scores <run>/predictions/<split>.parquet, or,
for a model configured not to export predictions, reads the rows already scored
in memory from <run>/metrics/<split>.json. Uses the model's latest run unless
--run-dir names one, and writes the table to <run>/metrics.json. Computes AUC,
PR-AUC, NE and CE per population, matching the evaluation recipe in
paper/frozen_pctr_iclr2027.tex. Metrics are computed over the complete
population before rounding, so the test row is directly comparable to the
paper's reported values, which are printed alongside for reference.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

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


# Models reporting ESMM's heads (q, v) rather than a single score.
ESMM_MODELS = ("esmm", "esmm_logged_pctr")


def populations(model: str, y, z, scores: dict) -> dict:
    """Population name -> (score, label) arrays.

    `scores` holds the model's prediction columns: q and v for ESMM, score for a
    single-head model. An ESMM model with no auxiliary CTR head has no q, and
    drops the two populations scored on it. When the logged pCTR p is present,
    CTCVR-logged scores the product it is served with, p * v, beside the
    auxiliary-head product q * v.
    """
    if model in ESMM_MODELS:
        clicked = y == 1
        rows = {}
        if "q" in scores:
            rows["CTR"] = (scores["q"], y)
            rows["CTCVR"] = (scores["q"] * scores["v"], z)
        if "p" in scores:
            rows["CTCVR-logged"] = (scores["p"] * scores["v"], z)
        rows["pCVR"] = (scores["v"][clicked], z[clicked])  # clicked impressions only
        return rows
    return {"CTR": (scores["score"], y)}


def metric_rows(split: str, model: str, y, z, scores: dict) -> list[dict]:
    """One row per population: its size, positives and the four metrics."""
    y, z = np.asarray(y), np.asarray(z)
    scores = {c: np.asarray(s) for c, s in scores.items()}
    return [
        {"split": split, "population": population,
         "rows": len(label), "positives": int(label.sum()), **compute_metrics(score, label)}
        for population, (score, label) in populations(model, y, z, scores).items()
    ]


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
        r["logged_pctr_run"] = logged_pctr["dcn_run"]
        r["logged_pctr_file"] = source["file"]


def main():
    ap = argparse.ArgumentParser(description="Report metrics from prediction parquets.")
    ap.add_argument("--artifacts", default=os.path.join(_ROOT_DIR, "artifacts"),
                    help="artifacts root, used to find the latest run")
    ap.add_argument("--model", default="dcn", choices=["dcn", *ESMM_MODELS])
    ap.add_argument("--run-dir", default=None, help="defaults to the model's latest run")
    args = ap.parse_args()
    run_dir = args.run_dir or resolve_run(args.artifacts, args.model)

    rows = []
    for split in ("train", "val", "test"):
        parquet = os.path.join(run_dir, "predictions", f"{split}.parquet")
        scored = os.path.join(run_dir, "metrics", f"{split}.json")
        if os.path.exists(parquet):
            df = pd.read_parquet(parquet)
            scores = {c: df[c].to_numpy() for c in df.columns if c not in _LABEL_COLUMNS}
            rows += metric_rows(split, args.model, df["y"].to_numpy(), df["z"].to_numpy(), scores)
        elif os.path.exists(scored):
            with open(scored) as f:
                rows += json.load(f)
        else:
            print(f"  (no predictions or metrics for {split}, skipping)")

    if not rows:
        raise SystemExit(f"no predictions or metrics found in {run_dir}")
    is_recorded = os.path.exists(os.path.join(run_dir, "run.json"))
    _attach_logged_pctr_source(rows, read_run(run_dir).get("logged_pctr") if is_recorded else None)

    hdr = f"{'split':<6} {'pop':<12} {'rows':>12} {'pos':>10} " \
          f"{'AUC':>10} {'PR-AUC':>11} {'NE':>10} {'CE':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split']:<6} {r['population']:<12} {r['rows']:>12,} {r['positives']:>10,} "
              f"{r['auc']:>10.6f} {r['pr_auc']:>11.6f} {r['ne']:>10.6f} {r['ce']:>10.6f}")
    for r in rows:
        if r["population"] == "CTCVR-logged":
            print(f"  {r['split']} CTCVR-logged: p from dcn run {r['logged_pctr_run']}, "
                  f"{r['logged_pctr_file']}")

    if args.model == "dcn":
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

    out = os.path.join(run_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
