"""Report the paper's evaluation table from exported prediction parquets.

Usage:
    python scripts/report_metrics.py --artifacts artifacts
    python scripts/report_metrics.py --artifacts artifacts --model dcn

Reads predictions_{train,val,test}.parquet written by scripts/train.py and
computes AUC, PR-AUC, NE and CE per population, matching the evaluation recipe
in paper/frozen_pctr_iclr2027.tex. Metrics are computed over the complete
population before rounding, so the test row is directly comparable to the
paper's reported values, which are printed alongside for reference.
"""

import argparse
import json
import os
import sys

import pandas as pd

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from evaluation.metrics import compute_metrics  # noqa: E402

# Table 2, DCN-V2 pCTR row: the target this reproduction is measured against.
_PAPER_DCN_CTR = {"auc": 0.620567, "pr_auc": 0.059938, "ne": 0.980200, "ce": 1.014134}
_PAPER_TEST_CENSUS = {"rows": 43_016_614, "positives": 1_673_447}


def _populations(df: pd.DataFrame, model: str) -> dict:
    """Score/label pairs per evaluation population, per the paper's contracts."""
    if model == "esmm":
        return {
            "CTR": (df["q"], df["y"]),
            "CTCVR": (df["q"] * df["v"], df["z"]),
            # clicked-only population for pCVR
            "pCVR": (df.loc[df["y"] == 1, "v"], df.loc[df["y"] == 1, "z"]),
        }
    return {"CTR": (df["score"], df["y"])}


def main():
    ap = argparse.ArgumentParser(description="Report metrics from prediction parquets.")
    ap.add_argument("--artifacts", default=os.path.join(_ROOT_DIR, "artifacts"))
    ap.add_argument("--model", default="dcn", choices=["dcn", "esmm"])
    ap.add_argument("--json-out", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(args.artifacts, f"predictions_{split}.parquet")
        if not os.path.exists(path):
            print(f"  (missing {os.path.basename(path)}, skipping)")
            continue
        df = pd.read_parquet(path)
        for population, (score, label) in _populations(df, args.model).items():
            m = compute_metrics(score.to_numpy(), label.to_numpy())
            rows.append({"split": split, "population": population,
                         "rows": len(label), "positives": int(label.sum()), **m})

    if not rows:
        raise SystemExit(f"no prediction parquets found in {args.artifacts}")

    hdr = f"{'split':<6} {'pop':<6} {'rows':>12} {'pos':>10} " \
          f"{'AUC':>10} {'PR-AUC':>11} {'NE':>10} {'CE':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split']:<6} {r['population']:<6} {r['rows']:>12,} {r['positives']:>10,} "
              f"{r['auc']:>10.6f} {r['pr_auc']:>11.6f} {r['ne']:>10.6f} {r['ce']:>10.6f}")

    if args.model == "dcn":
        p = _PAPER_DCN_CTR
        print("-" * len(hdr))
        print(f"{'paper':<6} {'CTR':<6} {_PAPER_TEST_CENSUS['rows']:>12,} "
              f"{_PAPER_TEST_CENSUS['positives']:>10,} "
              f"{p['auc']:>10.6f} {p['pr_auc']:>11.6f} {p['ne']:>10.6f} {p['ce']:>10.6f}")
        test = next((r for r in rows if r["split"] == "test"), None)
        if test:
            print(f"{'delta':<6} {'':<6} {test['rows'] - _PAPER_TEST_CENSUS['rows']:>12,} "
                  f"{test['positives'] - _PAPER_TEST_CENSUS['positives']:>10,} "
                  f"{test['auc'] - p['auc']:>+10.6f} {test['pr_auc'] - p['pr_auc']:>+11.6f} "
                  f"{test['ne'] - p['ne']:>+10.6f} {test['ce'] - p['ce']:>+10.6f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
