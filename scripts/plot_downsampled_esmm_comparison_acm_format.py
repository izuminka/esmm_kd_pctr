"""Plot the 5% retention comparison in the stacked ACM Figure 2 format.

Run from project root:
    code/.venv/bin/python code/scripts/plot_downsampled_esmm_comparison_acm_format.py

The data-loading and validation routines come from
``plot_downsampled_esmm_comparison.py``. This script changes only the output
names and layout: pCVR is stacked above logged CTCVR, matching Figure 1.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from plot_downsampled_esmm_comparison import (
    DEFAULT_SEED_STUDIES,
    METRICS,
    OUT,
    POPS,
    RUNS,
    load_records,
    load_seed_records,
)

STEM = "esmm_downsampled_5pct_comparison_acm_format"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--seed-studies",
        nargs=2,
        type=Path,
        metavar=("ESMM", "KD_TAU"),
        default=DEFAULT_SEED_STUDIES,
        help="Study directories containing four-seed summaries, in model order",
    )
    source.add_argument(
        "--single-seed",
        action="store_true",
        help="Plot the canonical seed-42 runs without error bars",
    )
    args = parser.parse_args()

    records = (
        load_records()
        if args.single_seed
        else load_seed_records(args.seed_studies)
    )
    stem = STEM + "_single_seed" if args.single_seed else STEM
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{stem}_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lookup = {
        (r["model"], r["population"], r["metric"]): r["value"] for r in records
    }
    errors = {
        (r["model"], r["population"], r["metric"]): r.get("sem", 0)
        for r in records
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.4))
    for ax, (population, _), title in zip(
        axes.flat, POPS, ("a) CVR", "b) Logged CTCVR (p × v)")
    ):
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.04,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=15,
            linespacing=1.05,
        )
        for j, (metric, label) in enumerate(METRICS):
            sub = ax.inset_axes(
                [0.10 + (j % 2) * 0.52, 0.57 - (j // 2) * 0.55, 0.36, 0.34]
            )
            values = [lookup[model, population, metric] for model, _, _, _ in RUNS]
            sems = [errors[model, population, metric] for model, _, _, _ in RUNS]
            low = min(value - sem for value, sem in zip(values, sems))
            high = max(value + sem for value, sem in zip(values, sems))
            pad = max(0.08 * (high - low), 1e-8)
            sub.bar(
                np.arange(len(RUNS)),
                values,
                color=[run[2] for run in RUNS],
                width=0.72,
                yerr=sems if not args.single_seed else None,
                error_kw={
                    "elinewidth": 0.8,
                    "capsize": 2,
                    "ecolor": "#333333",
                },
            )
            sub.set_ylim(low - pad, high + pad)
            if metric in ("ne", "ce") and low - pad <= 1 <= high + pad:
                sub.axhline(1, color="#555555", linestyle="--", linewidth=0.9)
            sub.set_xticks([])
            sub.set_title(label, fontsize=14, pad=8)
            sub.yaxis.set_major_locator(MaxNLocator(nbins=3))
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_scientific(metric == "pr_auc")
            if metric == "pr_auc":
                formatter.set_powerlimits((0, 0))
            sub.yaxis.set_major_formatter(formatter)
            sub.tick_params(axis="y", labelsize=18)
            sub.yaxis.get_offset_text().set_fontsize(18)
            sub.set_axisbelow(True)
            sub.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            sub.plot(
                [-0.012, 0.012],
                [-0.025, 0.025],
                transform=sub.transAxes,
                color="#333333",
                linewidth=1,
                clip_on=False,
            )

    fig.legend(
        [Patch(facecolor=run[2]) for run in RUNS],
        [run[0] for run in RUNS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(RUNS),
        frameon=False,
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.99, bottom=0.10, hspace=0.24
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Wrote {OUT / stem} (PNG, PDF, SVG, CSV)")


if __name__ == "__main__":
    main()
