"""Figure 2 inputs for the final study: click-head and product alignment.

Run from the project root:
    code/.venv/bin/python code/scripts/report_final_alignment.py
    code/.venv/bin/python code/scripts/plot_ctr_alignment.py

For ESMM, ESMM + DCN-V2 and KD-0.3, every seed of the final study is scored
against the study's frozen teacher p on all test impressions. The teacher side
is the reference distribution on both panels, with q' = clip(q, 1e-7, 1-1e-7)
and 0 ln 0 = 0:
  KL(p||q)   = mean KL(Bern(p) || Bern(q'))
  KL(pv||qv) = mean KL(Bern(p*v) || Bern(q'*v))
  Pearson r(p, q) and r(p*v, q*v) on raw float64 scores.
Both reports are written in the format plot_ctr_alignment.py validates.
"""

import argparse
import hashlib
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
Q_EPSILON = 1e-7
VARIANTS = ("esmm", "esmm_dcn", "kd03")
KL_DIRECTIONS = {"bernoulli_kl": "KL(Bern(p)||Bern(q'))",
                 "product_bernoulli_kl": "KL(Bern(pv)||Bern(q'v))"}


def sha(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def xlogy(a, b):
    """a * ln(b), with 0 ln(anything) = 0."""
    out = np.zeros_like(a)
    nz = a != 0
    out[nz] = a[nz] * np.log(b[nz])
    return out


def bernoulli_kl(target, prediction):
    return float(np.mean(xlogy(target, target) - xlogy(target, prediction)
                         + xlogy(1 - target, 1 - target) - xlogy(1 - target, 1 - prediction)))


def score_run(args):
    variant, seed, run_dir, teacher_path = args
    run = Path(run_dir)
    frame = pd.read_parquet(run / "predictions/test.parquet", columns=["sample_id", "y", "z", "q", "v"])
    teacher = pd.read_parquet(teacher_path, columns=["sample_id", "y", "score"])
    if not (np.array_equal(frame.sample_id.to_numpy(), teacher.sample_id.to_numpy())
            and np.array_equal(frame.y.to_numpy(), teacher.y.to_numpy())):
        raise ValueError(f"{variant}/{seed}: test rows differ from the teacher's")
    p = teacher.score.to_numpy(np.float64)
    q = frame.q.to_numpy(np.float64)
    v = frame.v.to_numpy(np.float64)
    q_clipped = np.clip(q, Q_EPSILON, 1 - Q_EPSILON)
    pv, qv = p * v, q * v
    meta = json.loads((run / "predictions/test.provenance.json").read_text())
    history = pd.read_csv(run / "history.csv")
    return {
        "variant": variant, "seed": seed, "rows": len(frame),
        "q_clipped_count": int(np.count_nonzero(q_clipped != q)),
        "bernoulli_kl": bernoulli_kl(p, q_clipped),
        "product_bernoulli_kl": bernoulli_kl(pv, q_clipped * v),
        "pearson": float(np.corrcoef(p, q)[0, 1]),
        "product_pearson": float(np.corrcoef(pv, qv)[0, 1]),
        "provenance": {
            "variant": variant, "seed": seed, "run_dir": str(run),
            "predictions_sha256": meta["sha256"],
            "config_sha256": meta["identity"]["config_sha256"],
            "cache_provenance_sha256": sha(run / "predictions/test.provenance.json"),
            "selected_on": f"epoch {int(history[history.saved].epoch.max())}",
            "checkpoint_sha256": meta["identity"]["checkpoint_sha256"],
        },
    }


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sem": statistics.stdev(values) / len(values) ** 0.5, "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts_final/study")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/main/results")
    args = parser.parse_args()
    teacher_run = Path(json.loads((args.study / "teacher.json").read_text())["selected"]["run_dir"])
    teacher_path = teacher_run / "predictions/test.parquet"
    jobs = []
    for variant in VARIANTS:
        for record in sorted(args.study.glob(f"runs/{variant}_seed*.json")):
            r = json.loads(record.read_text())
            jobs.append((variant, r["seed"], r["run_dir"], str(teacher_path)))
    with ProcessPoolExecutor(10) as pool:
        rows = sorted(pool.map(score_run, jobs), key=lambda r: (r["variant"], r["seed"]))
    for variant in VARIANTS:
        if [r["seed"] for r in rows if r["variant"] == variant] != list(range(42, 52)):
            raise ValueError(f"{variant}: expected seeds 42-51")
    if len({r["rows"] for r in rows}) != 1:
        raise ValueError("runs score different test populations")

    inputs = {"teacher_sha256": sha(teacher_path), "teacher_predictions": str(teacher_path),
              "summary_sha256": sha(args.study / "summary.json"), "rows": rows[0]["rows"],
              "script_sha256": sha(__file__)}
    provenance = [r["provenance"] for r in rows]
    alignment = {
        "definition": __doc__,
        "kl_directions": KL_DIRECTIONS,
        "inputs": inputs,
        "per_seed": [{k: r[k] for k in ("variant", "seed", "rows", "q_clipped_count",
                                        "bernoulli_kl", "product_bernoulli_kl")}
                     for r in rows],
        "summary": [{"variant": v, "metric": m, **stats([r[m] for r in rows if r["variant"] == v])}
                    for v in VARIANTS for m in ("bernoulli_kl", "product_bernoulli_kl")],
        "run_provenance": provenance,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    alignment_path = args.out / "ctr_alignment_final.json"
    alignment_path.write_text(json.dumps(alignment, indent=1) + "\n")

    summary = []
    for v in VARIANTS:
        plain = stats([r["pearson"] for r in rows if r["variant"] == v])
        product = stats([r["product_pearson"] for r in rows if r["variant"] == v])
        summary.append({"variant": v, **plain, **{f"product_{k}": x for k, x in product.items() if k != "n"}})
    correlation = {
        "definition": __doc__,
        "inputs": {**{k: inputs[k] for k in ("teacher_sha256", "teacher_predictions", "summary_sha256")},
                   "alignment_sha256": sha(alignment_path), "alignment_helper_sha256": inputs["script_sha256"]},
        "per_seed": [{k: r[k] for k in ("variant", "seed", "rows", "pearson", "product_pearson")} for r in rows],
        "summary": summary,
        "run_provenance": [{**p, "checkpoint_sha256_from_prior_audit": p["checkpoint_sha256"]} for p in provenance],
    }
    (args.out / "ctr_correlation_final.json").write_text(json.dumps(correlation, indent=1) + "\n")
    for v in VARIANTS:
        print(v, {m: round(next(s["mean"] for s in alignment["summary"] if s["variant"] == v and s["metric"] == m), 8)
                  for m in ("bernoulli_kl", "product_bernoulli_kl")},
              {k: round(next(s for s in summary if s["variant"] == v)[k], 4) for k in ("mean", "product_mean")})


if __name__ == "__main__":
    main()
