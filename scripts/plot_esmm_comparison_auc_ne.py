"""Create side-by-side full-population CVR and CTCVR panels with AUC and NE.

Copied from plot_esmm_comparison_acm_format.py; preserves its source data,
colors, error bars, axis limits, and grid styling; sizes for one column.
Run from project root:
    code/.venv/bin/python code/scripts/plot_esmm_comparison_auc_ne.py
Outputs PNG, PDF, SVG, and plotted CSV in paper/main/figures.
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
OUT = ROOT / "paper/main/figures"
MODELS = [
    ("ESMM", "20260918T182214Z_4aUNTa", "#0072B2"),
    ("KD–0.5", "20260918T180705Z_KBm7rX", "#CC79A7"),
]
POPS = [("pCVR", "pCVR"), ("CTCVR-logged", "Logged CTCVR\n(p × v)")]
METRICS = [
    ("auc", "AUC ↑"),
    ("ne", "NE ↓"),
]


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    lookup = {}
    for model, study, _ in MODELS:
        source = ROOT / "artifacts/esmm_studies" / study / "seed_summary.json"
        if not source.is_file():
            source = ROOT / "artifacts copy/esmm_studies" / study / "seed_summary.json"
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
    with (OUT / "esmm_comparison_auc_ne_data.csv").open(
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

    # Lay out at the final printed size so tick labels have real space.
    fig = plt.figure(figsize=(3.45, 1.45))
    positions = (0.085, 0.33, 0.585, 0.84)
    for k, (pop, metric, title) in enumerate(
        (pop, metric, title) for pop, _ in POPS for metric, title in METRICS
    ):
        ax = fig.add_axes([positions[k], 0.36, 0.14, 0.48])
        means, sems = np.array(
            [lookup[model, pop, metric] for model, _, _ in MODELS]
        ).T
        low, high = min(means - sems), max(means + sems)
        pad = 0.08 * (high - low)
        ax.bar(
            np.arange(len(MODELS)), means, yerr=sems,
            color=[color for _, _, color in MODELS], width=0.72,
            error_kw={"elinewidth": 0.5, "capsize": 1.3,
                      "capthick": 0.5, "ecolor": "#333333"},
        )
        ax.set_ylim(low - pad, high + pad)
        if metric == "ne" and low - pad <= 1 <= high + pad:
            ax.axhline(1, color="#555555", linestyle="--", linewidth=0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=6.5, pad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(axis="y", labelsize=5.5, pad=1.5, length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4)
        ax.plot(
            [-0.012, 0.012], [-0.025, 0.025], transform=ax.transAxes,
            color="#333333", linewidth=0.5, clip_on=False,
        )

    for x, title in ((0.28, "(a) CVR"), (0.78, "(b) CTCVR (p × v)")):
        fig.text(x, 0.24, title, ha="center", va="center",
                 fontsize=6.5, fontweight="bold")
    fig.legend(
        [Patch(facecolor=color) for _, _, color in MODELS],
        [model for model, _, _ in MODELS],
        loc="lower center", bbox_to_anchor=(0.5, 0.055),
        ncol=len(MODELS), frameon=False, fontsize=6.5,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.5,
        borderaxespad=0,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"esmm_comparison_auc_ne.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
