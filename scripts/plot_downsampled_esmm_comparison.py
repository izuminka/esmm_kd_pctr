"""Plot the 5% retention ESMM and KD-tau results from fixed canonical run reports.

Run from project root:
    code/.venv/bin/python code/scripts/plot_downsampled_esmm_comparison.py
Writes PNG, PDF, SVG, and a source-attributed CSV under paper/figures.
By default plots means ± SEM across training seeds 42–45 (sampling seed
2026) for paper Figure 2, with a _seeds output suffix. SEM uses the sample
standard deviation divided by sqrt(n). Pass --seed-studies ESMM_STUDY
KD_TAU_STUDY to override the summary directories, or --single-seed to
reproduce the seed-42 plot without error bars and without the _seeds suffix.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import yaml

from plot_esmm_comparison import METRICS, POPS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
RUNS = [
    ("ESMM", "esmm_retention_5pct/20260923T014746Z", "#0072B2", "ctcvr_ne"),
    ("KD–τ (α = 0.5)", "esmm_logged_pctr_kd_tau_retention_5pct/20260923T235217Z",
     "#CC79A7", "logged_ctcvr_ne"),
]
TEACHER = "dcn_matched_stats_wd3e6"
STEM = "esmm_downsampled_5pct_comparison"
DEFAULT_SEED_STUDIES = (
    ROOT / "artifacts_downsampled/esmm/_support/studies/20260924T001757Z_XXiMUo",
    ROOT / "artifacts_downsampled/esmm_logged_pctr_kd_tau/_support/studies/20260924T001757Z_vmxDKV",
)


def load_records():
    records, census, teacher_runs = [], {}, set()
    for model, relative, _, selection in RUNS:
        run = ROOT / "artifacts_downsampled" / relative
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        metadata = json.loads((run / "run.json").read_text())
        assert metadata["status"] == "FINISHED", run
        assert metadata["selected_on"]["metric"] == selection, run
        assert metadata["selected_on"]["split"] == "val", run
        assert cfg["data"]["nonclick_sampling"] == {"rate": .05, "seed": 2026}, run
        assert cfg["training"]["seed"] == 42 and cfg["training"]["epochs"] == 5, run
        if cfg["model"] == "esmm_logged_pctr":
            assert cfg["model_params"]["objective"] == "kd_tau", run
            assert cfg["model_params"]["alpha"] == .5, run
        for population, _ in POPS:
            source = (run / "metrics/test.json" if population == "pCVR" else
                      run / "metrics_logged" / TEACHER / "test.json")
            row, = [r for r in json.loads(source.read_text())
                    if r["split"] == "test" and r["population"] == population]
            counts = (row["rows"], row["positives"])
            assert census.setdefault(population, counts) == counts, source
            if population == "CTCVR-logged":
                assert row["pctr_model"] == TEACHER, source
                teacher_runs.add(row["logged_pctr_run"])
            for metric, _ in METRICS:
                value = row[metric]
                assert np.isfinite(value), (source, metric)
                records.append(dict(model=model, population=population, metric=metric,
                                    value=value, n=1, seed=42, retention=.05,
                                    selected_epoch=metadata["selected_on"]["epoch"],
                                    selection_metric=selection,
                                    source=str(source.relative_to(ROOT))))
    assert len(teacher_runs) == 1
    assert census == {"pCVR": (1673447, 9195), "CTCVR-logged": (43016614, 9195)}
    return records


def load_seed_records(studies):
    """Read verified four-seed summaries produced by report_esmm_study.py."""
    records = []
    for study, (model, relative, _, selection) in zip(studies, RUNS):
        data = json.loads((study / "seed_summary.json").read_text())
        base = relative.split("/")[0]
        assert sorted(r["seed"] for r in data["sources"]) == [42, 43, 44, 45]
        assert all(r["selected_on"]["metric"] == selection for r in data["sources"])
        assert all(r["variant"] == base for r in data["sources"])
        assert any(Path(r["run_dir"]).resolve() == (ROOT / "artifacts_downsampled" / relative).resolve()
                   for r in data["sources"])
        for source in data["sources"]:
            cfg = yaml.safe_load((Path(source["run_dir"]) / "config.yaml").read_text())
            assert cfg["data"]["nonclick_sampling"] == {"rate": .05, "seed": 2026}
        for population, _ in POPS:
            for metric, _ in METRICS:
                row, = [r for r in data["summary"] if r["variant"] == base
                        and r["population"] == population and r["metric"] == metric]
                assert row["n"] == 4 and len(row["values"]) == 4
                assert np.isclose(row["mean"], np.mean(row["values"]))
                assert np.isclose(row["sem"], np.std(row["values"], ddof=1) / 2)
                records.append(dict(model=model, population=population, metric=metric,
                                    value=row["mean"], sem=row["sem"], n=4,
                                    source=str(study / "seed_summary.json")))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--seed-studies", nargs=2, type=Path, metavar=("ESMM", "KD_TAU"),
                        default=DEFAULT_SEED_STUDIES,
                        help="Study directories containing four-seed summaries, in model order")
    source.add_argument("--single-seed", action="store_true",
                        help="Reproduce the seed-42 plot without error bars")
    args = parser.parse_args()
    records = load_records() if args.single_seed else load_seed_records(args.seed_studies)
    stem = STEM + "_seeds" if not args.single_seed else STEM
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{stem}_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lookup = {(r["model"], r["population"], r["metric"]): r["value"] for r in records}
    errors = {(r["model"], r["population"], r["metric"]): r.get("sem", 0) for r in records}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (population, _), title in zip(axes, POPS, ("a) CVR", "b) Logged CTCVR (p × v)")):
        ax.set_axis_off()
        ax.text(.5, -.04, title, transform=ax.transAxes, ha="center", va="top",
                fontweight="bold", fontsize=15)
        for j, (metric, label) in enumerate(METRICS):
            sub = ax.inset_axes([.10 + (j % 2)*.52, .57 - (j // 2)*.55, .36, .34])
            values = [lookup[model, population, metric] for model, _, _, _ in RUNS]
            sem = [errors[model, population, metric] for model, _, _, _ in RUNS]
            low = min(v - e for v, e in zip(values, sem))
            high = max(v + e for v, e in zip(values, sem))
            pad = max(.08*(high-low), 1e-8)
            sub.bar(np.arange(len(RUNS)), values, color=[r[2] for r in RUNS], width=.72,
                    yerr=sem if not args.single_seed else None, capsize=4,
                    error_kw={"elinewidth": 1.2})
            sub.set_ylim(low-pad, high+pad)
            if metric in ("ne", "ce") and low-pad <= 1 <= high+pad:
                sub.axhline(1, color="#555555", linestyle="--", linewidth=.9)
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
            sub.grid(axis="y", color="#e5e5e5", linewidth=.6)
            sub.plot([-.012, .012], [-.025, .025], transform=sub.transAxes,
                     color="#333333", linewidth=1, clip_on=False)
    fig.legend([Patch(facecolor=r[2]) for r in RUNS], [r[0] for r in RUNS],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.07, right=.98, top=.94, bottom=.19, wspace=.25)
    title = ("5% nonclick retention · mean ± SEM (seeds 42–45, n = 4)"
             if not args.single_seed else "5% nonclick retention · single seed (42)")
    fig.suptitle(title, fontsize=13, y=1.02)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / stem} (PNG, PDF, SVG, CSV)")


if __name__ == "__main__":
    main()
