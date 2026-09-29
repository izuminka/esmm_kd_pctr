"""Compare two DCN test-prediction exports that differ only by training seed.

The comparison verifies configuration equivalence and exact sample/label
alignment, reports the four canonical CTR metrics, and quantifies paired
per-impression score and ranking stability.  Block-level inference is a
diagnostic for local dependence; it is not a substitute for user- or
auction-clustered inference when those identifiers are available.

Example, from the project root::

    code/.venv/bin/python code/scripts/compare_dcn_seeds.py \
      --run-a artifacts/dcn_matched_stats_wd3e6/20260917T232959Z \
      --run-b artifacts/dcn_matched_stats_wd3e6_seed43/20260929T214036Z
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml


METRICS = ("auc", "pr_auc", "ne", "ce")
PREDICTION_COLUMNS = ("sample_id", "y", "z", "score")


def _load_seed_config(run_dir: Path) -> tuple[int, dict]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if cfg.get("model") != "dcn":
        raise ValueError(f"{run_dir}: expected a DCN config")
    comparable = dict(cfg)
    comparable["training"] = dict(cfg["training"])
    seed = int(comparable["training"].pop("seed"))
    return seed, comparable


def _validate_runs(run_a: Path, run_b: Path) -> tuple[int, int]:
    for run_dir in (run_a, run_b):
        run = json.loads((run_dir / "run.json").read_text())
        if run["status"] != "FINISHED":
            raise ValueError(f"{run_dir}: run is {run['status']}, not FINISHED")
        for relative in ("predictions/test.parquet", "metrics/test.json"):
            if not (run_dir / relative).is_file():
                raise ValueError(f"{run_dir}: missing {relative}")
    seed_a, config_a = _load_seed_config(run_a)
    seed_b, config_b = _load_seed_config(run_b)
    if seed_a == seed_b:
        raise ValueError(f"both runs use seed {seed_a}")
    if config_a != config_b:
        raise ValueError("run configurations differ beyond training.seed")
    return seed_a, seed_b


def _metric_row(run_dir: Path) -> dict:
    rows = json.loads((run_dir / "metrics/test.json").read_text())
    return next(row for row in rows if row["population"] == "CTR")


def _block_inference(values: np.ndarray, block_size: int) -> dict:
    n_full = len(values) // block_size
    if n_full < 2:
        raise ValueError("block_size leaves fewer than two complete blocks")
    block_means = values[: n_full * block_size].reshape(n_full, block_size).mean(
        axis=1, dtype=np.float64
    )
    return _inference_from_block_means(
        float(np.mean(values, dtype=np.float64)), block_means, block_size
    )


def _inference_from_block_means(mean: float, block_means, block_size: int) -> dict:
    block_means = np.asarray(block_means, dtype=np.float64)
    sem = float(stats.sem(block_means))
    low, high = stats.t.interval(
        0.95, len(block_means) - 1, loc=mean, scale=sem
    )
    test = stats.ttest_1samp(block_means, 0.0)
    return {
        "mean": mean,
        "block_size": block_size,
        "n_full_blocks": len(block_means),
        "block_t_statistic": float(test.statistic),
        "block_p_value": float(test.pvalue),
        "block_95pct_ci": [float(low), float(high)],
    }


def _pearson_correlation(x: np.ndarray, y: np.ndarray,
                         chunk_size: int = 1_000_000) -> float:
    n = len(x)
    sx = sy = sxx = syy = sxy = 0.0
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        xc = x[start:stop].astype(np.float64)
        yc = y[start:stop].astype(np.float64)
        sx += xc.sum()
        sy += yc.sum()
        sxx += np.dot(xc, xc)
        syy += np.dot(yc, yc)
        sxy += np.dot(xc, yc)
    numerator = n * sxy - sx * sy
    denominator = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return numerator / denominator


def _top_overlap(x: np.ndarray, y: np.ndarray, fraction: float) -> dict:
    threshold_x, threshold_y = np.quantile(x, 1 - fraction), np.quantile(y, 1 - fraction)
    selected_x, selected_y = x >= threshold_x, y >= threshold_y
    rows_x, rows_y = int(selected_x.sum()), int(selected_y.sum())
    intersection = int(np.count_nonzero(selected_x & selected_y))
    union = rows_x + rows_y - intersection
    return {
        "fraction": fraction,
        "run_a_rows": rows_x,
        "run_b_rows": rows_y,
        "intersection": intersection,
        "overlap_fraction_of_smaller_set": intersection / min(rows_x, rows_y),
        "jaccard": intersection / union,
        "run_a_threshold": float(threshold_x),
        "run_b_threshold": float(threshold_y),
    }


def compare(run_a: Path, run_b: Path, output: Path, block_size: int,
            rank_sample_rows: int) -> dict:
    seed_a, seed_b = _validate_runs(run_a, run_b)
    label_a, label_b = f"seed{seed_a}", f"seed{seed_b}"
    delta_label = f"{label_b}_minus_{label_a}"

    frame_a = pd.read_parquet(
        run_a / "predictions/test.parquet", columns=PREDICTION_COLUMNS
    )
    frame_b = pd.read_parquet(
        run_b / "predictions/test.parquet", columns=PREDICTION_COLUMNS
    )
    if len(frame_a) != len(frame_b):
        raise ValueError("test prediction row counts differ")
    alignment = {
        "sample_ids_exactly_equal": np.array_equal(frame_a.sample_id, frame_b.sample_id),
        "click_labels_exactly_equal": np.array_equal(frame_a.y, frame_b.y),
        "conversion_labels_exactly_equal": np.array_equal(frame_a.z, frame_b.z),
    }
    if not all(alignment.values()):
        raise ValueError("test IDs or labels do not align exactly")

    sample_ids = frame_a.sample_id.to_numpy()
    labels = frame_a.y.to_numpy()
    score_a = frame_a.score.to_numpy()
    score_b = frame_b.score.to_numpy()
    delta = score_b - score_a
    abs_delta = np.abs(delta)
    rows = len(delta)
    prevalence = float(np.mean(labels, dtype=np.float64))
    mean_a = float(np.mean(score_a, dtype=np.float64))
    mean_b = float(np.mean(score_b, dtype=np.float64))
    mean_delta = float(np.mean(delta, dtype=np.float64))
    std_delta = float(np.std(delta, dtype=np.float64, ddof=1))

    quantile_probs = np.asarray(
        [0, .01, .05, .25, .5, .75, .9, .95, .99, .999, 1.0]
    )
    abs_quantiles = np.quantile(abs_delta, quantile_probs)
    signed_quantiles = np.quantile(delta, quantile_probs)
    thresholds = (1e-4, 1e-3, .005, .01, .02, .05)

    rank_rows = min(rank_sample_rows, rows)
    rank_indices = np.linspace(0, rows - 1, rank_rows, dtype=np.int64)
    spearman = stats.spearmanr(score_a[rank_indices], score_b[rank_indices])

    # Compute paired proper-loss differences in bounded-memory blocks. Complete
    # block means support the dependence-aware diagnostic; all rows contribute
    # to the exact overall mean, including the final partial block.
    eps = 1e-7
    logloss_block_means = []
    brier_block_means = []
    logloss_sum = brier_sum = 0.0
    for start in range(0, rows, block_size):
        stop = min(rows, start + block_size)
        y = labels[start:stop].astype(np.float64)
        a = np.clip(score_a[start:stop].astype(np.float64), eps, 1 - eps)
        b = np.clip(score_b[start:stop].astype(np.float64), eps, 1 - eps)
        loss_a = -(y * np.log(a) + (1 - y) * np.log1p(-a))
        loss_b = -(y * np.log(b) + (1 - y) * np.log1p(-b))
        logloss_delta = loss_b - loss_a
        brier_delta = (b - y) ** 2 - (a - y) ** 2
        logloss_sum += float(logloss_delta.sum())
        brier_sum += float(brier_delta.sum())
        if stop - start == block_size:
            logloss_block_means.append(float(logloss_delta.mean()))
            brier_block_means.append(float(brier_delta.mean()))

    entropy = -(prevalence * math.log(prevalence)
                + (1 - prevalence) * math.log(1 - prevalence))
    logloss = _inference_from_block_means(
        logloss_sum / rows, logloss_block_means, block_size
    )
    ne = _inference_from_block_means(
        (logloss_sum / rows) / entropy,
        np.asarray(logloss_block_means) / entropy,
        block_size,
    )
    brier = _inference_from_block_means(
        brier_sum / rows, brier_block_means, block_size
    )

    metrics_a, metrics_b = _metric_row(run_a), _metric_row(run_b)
    report = {
        "runs": {label_a: str(run_a), label_b: str(run_b)},
        "configuration": {
            "seeds": [seed_a, seed_b],
            "equal_except_training_seed": True,
        },
        "alignment": {
            "rows": rows,
            **alignment,
            "first_sample_id": int(sample_ids[0]),
            "last_sample_id": int(sample_ids[-1]),
        },
        "test_metrics": {
            label_a: {key: metrics_a[key] for key in METRICS},
            label_b: {key: metrics_b[key] for key in METRICS},
            delta_label: {
                key: metrics_b[key] - metrics_a[key] for key in METRICS
            },
        },
        "prediction_comparison": {
            "delta_definition": delta_label,
            "prevalence": prevalence,
            f"mean_score_{label_a}": mean_a,
            f"mean_score_{label_b}": mean_b,
            "mean_signed_delta": mean_delta,
            "mean_absolute_delta": float(np.mean(abs_delta, dtype=np.float64)),
            "root_mean_squared_delta": float(
                np.sqrt(np.mean(np.square(delta), dtype=np.float64))
            ),
            "std_signed_delta": std_delta,
            "paired_standardized_mean_delta_cohen_dz": mean_delta / std_delta,
            "pearson_correlation_exact": _pearson_correlation(score_a, score_b),
            "spearman_correlation_even_sample": float(spearman.statistic),
            "spearman_sample_rows": rank_rows,
            "exactly_equal_fraction": float(np.mean(delta == 0)),
            "abs_delta_quantiles": {
                str(q): float(value)
                for q, value in zip(quantile_probs, abs_quantiles)
            },
            "signed_delta_quantiles": {
                str(q): float(value)
                for q, value in zip(quantile_probs, signed_quantiles)
            },
            "fraction_abs_delta_above": {
                str(threshold): float(np.mean(abs_delta > threshold))
                for threshold in thresholds
            },
            "by_click_label": {
                "nonclick": {
                    "rows": int(np.count_nonzero(labels == 0)),
                    "mean_signed_delta": float(
                        np.mean(delta[labels == 0], dtype=np.float64)
                    ),
                    "mean_absolute_delta": float(
                        np.mean(abs_delta[labels == 0], dtype=np.float64)
                    ),
                },
                "click": {
                    "rows": int(np.count_nonzero(labels == 1)),
                    "mean_signed_delta": float(
                        np.mean(delta[labels == 1], dtype=np.float64)
                    ),
                    "mean_absolute_delta": float(
                        np.mean(abs_delta[labels == 1], dtype=np.float64)
                    ),
                },
            },
            "top_score_overlap": [
                _top_overlap(score_a, score_b, fraction)
                for fraction in (.01, .05, .10)
            ],
            "paired_mean_delta_block_inference": _block_inference(
                delta, block_size
            ),
            "paired_log_loss_delta": logloss,
            "paired_ne_delta": ne,
            "paired_brier_delta": brier,
            "inference_note": (
                f"Confidence intervals and p-values use contiguous {block_size:,}-"
                "impression block means. They remain diagnostics because the "
                "prediction parquet does not expose a user/auction clustering key."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="default: <run-b>/comparisons/seed<run-a seed>_test.json",
    )
    parser.add_argument("--block-size", type=int, default=100_000)
    parser.add_argument("--rank-sample-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    run_a, run_b = args.run_a.resolve(), args.run_b.resolve()
    if args.block_size <= 0 or args.rank_sample_rows <= 1:
        parser.error("block-size must be positive and rank-sample-rows must exceed one")
    seed_a, _ = _load_seed_config(run_a)
    output = (args.output.resolve() if args.output else
              run_b / "comparisons" / f"seed{seed_a}_test.json")
    report = compare(run_a, run_b, output, args.block_size, args.rank_sample_rows)
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
