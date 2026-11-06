"""Plot click-head and product alignment in the one-column Figure 2 style.

Copied from plot_esmm_comparison_auc_ne.py; retains its fonts, grid, colors,
mean +/- SEM bars, truncated axes, and shared legend. No predictions are read.
Run from project root:
    code/.venv/bin/python code/scripts/plot_ctr_alignment.py
Outputs ctr_alignment.{png,pdf,svg,csv,json} in paper/main/figures.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/main/figures"
MODELS = [
    ("ESMM", "esmm", "#0072B2"),
    ("ESMM+DCN-V2", "esmm_dcn", "#E69F00"),
    ("KD–0.3", "kd03", "#CC79A7"),
]
POPS = [("p_q", "(a) p vs q"), ("pv_qv", "(b) pv vs qv")]
# The frozen teacher side is the reference distribution on both panels.
KL_DIRECTIONS = {"bernoulli_kl": "KL(Bern(p)||Bern(q'))",
                 "product_bernoulli_kl": "KL(Bern(pv)||Bern(q'v))"}
METRICS = [
    ("kl", "KL ↓"),
    ("pearson", "Pearson ↑"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_stats(values, mean, sem, name):
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name}: expected ten finite per-seed values")
    expected = statistics.mean(values), statistics.stdev(values)/math.sqrt(10)
    if not all(math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-18)
               for actual, value in zip((mean, sem), expected)):
        raise ValueError(f"{name}: summary mean/SEM disagrees with per-seed values")
    return mean, sem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=ROOT/"paper/main/results/ctr_alignment_final.json")
    parser.add_argument("--correlation", type=Path, default=ROOT/"paper/main/results/ctr_correlation_final.json")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    records = []
    lookup = {}
    alignment_path, correlation_path = args.alignment.resolve(), args.correlation.resolve()
    data = json.loads(alignment_path.read_text())
    correlation = json.loads(correlation_path.read_text())
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (alignment_path, correlation_path)}
    if data.get("kl_directions") != KL_DIRECTIONS:
        raise ValueError(f"Alignment report KL directions {data.get('kl_directions')} are not {KL_DIRECTIONS}")
    if correlation["inputs"]["alignment_sha256"] != digest(alignment_path):
        raise ValueError("Correlation is bound to a different alignment report")
    for key in ("teacher_sha256", "teacher_predictions", "summary_sha256"):
        if data["inputs"][key] != correlation["inputs"][key]:
            raise ValueError(f"Alignment/correlation source identity mismatch: {key}")
    if correlation["inputs"]["alignment_helper_sha256"] != data["inputs"]["script_sha256"]:
        raise ValueError("Alignment calculation source identity mismatch")
    provenance = [{(r["variant"], r["seed"]): r for r in report["run_provenance"]}
                  for report in (data, correlation)]
    for model, variant, _ in MODELS:
        samples = [sorted((r for r in report["per_seed"] if r["variant"] == variant), key=lambda r:r["seed"])
                   for report in (data, correlation)]
        for rows in samples:
            if [r["seed"] for r in rows] != list(range(42,52)) or any(r["rows"] != data["inputs"]["rows"] for r in rows):
                raise ValueError(f"{variant}: expected ten distinct seeds and matching test census")
        for seed in range(42,52):
            first, second = [rows[variant,seed] for rows in provenance]
            for key in ("run_dir", "predictions_sha256", "config_sha256", "cache_provenance_sha256", "selected_on"):
                if first[key] != second[key]:
                    raise ValueError(f"{variant}/{seed}: run identity mismatch in {key}")
            if first["checkpoint_sha256"] != second["checkpoint_sha256_from_prior_audit"]:
                raise ValueError(f"{variant}/{seed}: checkpoint identity mismatch")
        for pop, _ in POPS:
            for metric, _ in METRICS:
                if metric == "kl":
                    source_metric = "bernoulli_kl" if pop == "p_q" else "product_bernoulli_kl"
                    row = next(r for r in data["summary"] if r["variant"] == variant and r["metric"] == source_metric)
                    values = [r[source_metric] for r in samples[0]]
                    source, mean, sem = alignment_path, row["mean"], row["sem"]
                    if row["values"] != values:
                        raise ValueError(f"{variant}/{source_metric}: summary values mismatch")
                else:
                    source_metric = "pearson" if pop == "p_q" else "product_pearson"
                    row = next(r for r in correlation["summary"] if r["variant"] == variant)
                    values = [r[source_metric] for r in samples[1]]
                    prefix = "" if pop == "p_q" else "product_"
                    source, mean, sem = correlation_path, row[prefix+"mean"], row[prefix+"sem"]
                    if row[prefix+"values"] != values:
                        raise ValueError(f"{variant}/{source_metric}: summary values mismatch")
                if row["n"] != 10:
                    raise ValueError(f"{variant}: expected ten observations")
                lookup[model,pop,metric] = checked_stats(values, mean, sem, f"{variant}/{pop}/{metric}")
                records.append(
                    dict(
                        model=model,
                        variant=variant,
                        comparison=pop,
                        metric=metric,
                        source_metric=source_metric,
                        mean=mean,
                        sem=sem,
                        n=10,
                        source=str(source.relative_to(ROOT)),
                    )
                )
    with (OUT / "ctr_alignment.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
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
    fig = plt.figure(figsize=(3.45, 1.55))
    positions = (0.075, 0.33, 0.585, 0.84)
    for k, (pop, metric, title) in enumerate(
        (pop, metric, title) for pop, _ in POPS for metric, title in METRICS
    ):
        ax = fig.add_axes([positions[k], 0.34, 0.14, 0.48])
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
        ax.set_xticks([])
        ax.set_title(title, fontsize=6.5, y=1.17, pad=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 3, 4, 5, 10]))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        if metric == "kl":
            # Scale tick labels only; bars, error bars, and exported data stay raw.
            if pop == "p_q":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:.2f}"))
                scale_label = "×10⁻³"
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e6:.1f}"))
                scale_label = "×10⁻⁶"
            ax.text(0.5, 1.035, scale_label, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5, pad=1.5, length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4)

    for x, title in ((0.27, POPS[0][1]), (0.78, POPS[1][1])):
        fig.text(x, 0.24, title, ha="center", va="center",
                 fontsize=6.5, fontweight="bold")
    fig.legend(
        [Patch(facecolor=color) for _, _, color in MODELS],
        [model for model, _, _ in MODELS],
        loc="lower center", bbox_to_anchor=(0.5, 0.055),
        ncol=len(MODELS), frameon=False, fontsize=6.5,
        handlelength=1.2, handletextpad=0.5, columnspacing=1.2,
        borderaxespad=0,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"ctr_alignment.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        if ext == "svg":
            path = OUT / f"ctr_alignment.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    (OUT/"ctr_alignment.json").write_text(json.dumps({
        "source_sha256": source_hashes,
        "script_sha256": digest(Path(__file__)),
        "copied_from": "code/scripts/plot_esmm_comparison_auc_ne.py",
        "template_sha256": digest(Path(__file__).with_name("plot_esmm_comparison_auc_ne.py")),
        "teacher_sha256": data["inputs"]["teacher_sha256"],
        "summary_sha256": data["inputs"]["summary_sha256"],
        "validation": "12 bar means and SEMs recomputed from ten matched seeds; teacher, source, run, cache and checkpoint identities match",
        "kl_directions": {"p_q": KL_DIRECTIONS["bernoulli_kl"], "pv_qv": KL_DIRECTIONS["product_bernoulli_kl"]},
        "raw_bar_values": records,
        "truncated_axes": True,
        "axis_break_markers": False,
        "kl_tick_scaling_only": {"p_q": "1e-3", "pv_qv": "1e-6"},
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
