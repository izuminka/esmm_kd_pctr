"""Rebuild Table 1 grouped plots in a vertically stacked ACM layout.

Run from project root:
    code/.venv/bin/python code/scripts/plot_esmm_comparison_acm_format.py

Requires matplotlib and numpy. Outputs PNG, PDF, SVG, and the plotted CSV.
The ESMM and KD--0.5 values and source summaries are identical to
``plot_esmm_comparison.py``; this version omits KD--teacher and stacks the two
population panels for the ACM layout.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
MODELS = [
    ("ESMM", "20260918T182214Z_4aUNTa", "#0072B2"),
    ("KD–0.5", "20260918T180705Z_KBm7rX", "#CC79A7"),
]
POPS = [("pCVR", "pCVR"), ("CTCVR-logged", "Logged CTCVR\n(p × v)")]
METRICS = [
    ("auc", "AUC ↑"),
    ("pr_auc", "PR-AUC ↑"),
    ("ne", "NE ↓"),
    ("ce", "CE (ideal = 1)"),
]


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    lookup = {}
    for model, study, _ in MODELS:
        source = ROOT / "artifacts/esmm_studies" / study / "seed_summary.json"
        data = json.loads(source.read_text())
        assert sorted(s["seed"] for s in data["sources"]) == [42, 43, 44, 45]
        for pop, _ in POPS:
            for metric, _ in METRICS:
                row = next(
                    r
                    for r in data["summary"]
                    if r["population"] == pop and r["metric"] == metric
                )
                assert row["n"] == 4
                lookup[model, pop, metric] = (row["mean"], row["sem"])
                records.append(
                    dict(
                        model=model,
                        population=pop,
                        metric=metric,
                        mean=row["mean"],
                        sem=row["sem"],
                        n=4,
                        source=str(source.relative_to(ROOT)),
                    )
                )
    with (OUT / "esmm_comparison_acm_format_data.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

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

    def population_panel(ax, pop, title):
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
            means, sems = np.array(
                [lookup[model, pop, metric] for model, _, _ in MODELS]
            ).T
            low, high = min(means - sems), max(means + sems)
            pad = 0.08 * (high - low)
            sub.bar(
                np.arange(len(MODELS)),
                means,
                yerr=sems,
                color=[color for _, _, color in MODELS],
                width=0.72,
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
            fmt = ScalarFormatter(useOffset=False, useMathText=True)
            fmt.set_scientific(metric == "pr_auc")
            if metric == "pr_auc":
                fmt.set_powerlimits((0, 0))
            sub.yaxis.set_major_formatter(fmt)
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

    def panel(ax, metric, title):
        x = np.arange(len(POPS))
        width = 0.15
        for i, (model, _, color) in enumerate(MODELS):
            means, sems = np.array(
                [lookup[model, pop, metric] for pop, _ in POPS]
            ).T
            # Positive baseline for log bars; other panels use a zero baseline.
            bottom = 1e-4 if metric == "pr_auc" else 0
            ax.bar(
                x + (i - (len(MODELS) - 1) / 2) * width,
                means - bottom,
                width,
                bottom=bottom,
                yerr=sems,
                label=model,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                error_kw={
                    "elinewidth": 0.8,
                    "capsize": 2,
                    "capthick": 0.8,
                    "ecolor": "#333333",
                },
            )
        ax.set_xticks(x, [label for _, label in POPS])
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        bounds = np.array(
            [
                lookup[model, pop, metric]
                for model, _, _ in MODELS
                for pop, _ in POPS
            ]
        )
        low = np.min(bounds[:, 0] - bounds[:, 1])
        high = np.max(bounds[:, 0] + bounds[:, 1])
        if metric == "pr_auc":
            ax.set_yscale("log")
            padding = 0.06 * (np.log10(high) - np.log10(low))
            ax.set_ylim(
                10 ** (np.log10(low) - padding),
                10 ** (np.log10(high) + padding),
            )
        else:
            if metric in ("ne", "ce"):
                low, high = min(low, 1), max(high, 1)
                ax.axhline(1, color="#555555", linestyle="--", linewidth=0.9)
            padding = 0.03 * (high - low)
            ax.set_ylim(low - padding, high + padding)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        # Mark the cropped vertical range at the lower-left axis boundary.
        ax.plot(
            [-0.008, 0.008],
            [-0.012, 0.012],
            transform=ax.transAxes,
            color="#333333",
            linewidth=1,
            clip_on=False,
        )

    note = (
        "Test means ± SEM across seeds 42–45 (n = 4). Raw metrics; each subplot "
        "has its own zoomed, nonzero y-axis.\n"
        "ESMM: minimum validation q×v NE; KD: minimum validation p×v NE. "
        "KD α is the click-label weight; teacher-only α = 0."
    )

    # ACM format: stack the two population panels instead of placing them side
    # by side. Each population panel retains the original 2×2 metric layout.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.4))
    for ax, (pop, _), title in zip(
        axes.flat, POPS, ("a) CVR", "b) Logged CTCVR (p × v)")
    ):
        population_panel(ax, pop, title)
    handles = [Patch(facecolor=color) for _, _, color in MODELS]
    labels = [model for model, _, _ in MODELS]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(MODELS),
        frameon=False,
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.99, bottom=0.10, hspace=0.24
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"esmm_comparison_panels_acm_format.{ext}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    panel(ax, "auc", "Test AUC · ESMM and knowledge distillation")
    ax.legend(
        ncol=len(MODELS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5, 0.025, note, ha="center", va="bottom", fontsize=8, color="#444444"
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"esmm_auc_bars_acm_format.{ext}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
