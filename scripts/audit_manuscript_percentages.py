#!/usr/bin/env python3
"""Audit manuscript percentage claims against displayed and unrounded results.

Run from any directory: python3 code/scripts/audit_manuscript_percentages.py
Writes paper/main/results/percentage_ce_audit.json. Source summaries contain
the ten individual seed results, which are averaged independently here.
"""

import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/main/main.tex"
FULL = ROOT / "artifacts_final/study/summary.json"
DOWN = ROOT / "artifacts_final_downsampled/study/summary.json"
FIG = ROOT / "paper/main/figures/ctr_alignment.json"
OUT = ROOT / "paper/main/results/percentage_ce_audit.json"
METRICS = ("auc", "pr_auc", "ne", "ce")
MODELS = {"ESMM": "esmm", "ESMM+DCN-V2": "esmm_dcn", "KD-0.7": "kd07",
          "KD-0.5": "kd05", "KD-0.3": "kd03", "KD-0": "kd0", "Direct": "direct",
          "ESMM_R": "down_esmm", "KD_R-0.3": "down_kd0.3_tau", "DCN-V2": "teacher"}


def read_json(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomments(text):
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def label_from_cell(cell, downsampled):
    if "DCN-V2" in cell:
        return "ESMM+DCN-V2" if "ESMM" in cell else "DCN-V2"
    if "Direct" in cell:
        return "Direct"
    if "KD" in cell:
        number = re.search(r"--(0(?:\.[0-9]+)?)", cell).group(1)
        return f"KD_R-{number}" if downsampled else f"KD-{number}"
    if "ESMM" in cell:
        return "ESMM_R" if downsampled else "ESMM"
    raise ValueError(f"Unknown model cell: {cell}")


def parse_table(text, label, downsampled=False):
    stop = text.index(r"\label{" + label + "}")
    start = text.rfind(r"\begin{table}", 0, stop)
    if start < 0:
        raise ValueError(f"Missing table for {label}")
    block = text[start:stop]
    rows, model = {}, None
    for line in block.splitlines():
        cells = [part.strip() for part in line.split("&")]
        if len(cells) != 6 or not re.search(r"\d", cells[2]):
            continue
        if cells[0]:
            model = label_from_cell(cells[0], downsampled)
        output_cell = cells[1].replace(" ", "")
        output = {"$v_i$": "pCVR", "$p_iv_i$": "CTCVR-logged", "$q_i$": "CTR",
                  "$p_i$": "CTR"}.get(output_cell)
        if model is None or output is None:
            raise ValueError(f"Unknown table row: {line}")
        values = [Decimal(re.search(r"[0-9]+\.[0-9]+", cell).group()) for cell in cells[2:]]
        rows[(model, output)] = dict(zip(METRICS, values))
    return rows


def relative(after, before):
    return (after - before) / before * 100


def round_one(number):
    return number.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def source_means(full, down, variant, population):
    if variant == "teacher":
        return {key: full["teacher"]["test"][key] for key in METRICS}
    collection = down if variant.startswith("down_") else full
    seeds = collection["variants"][variant]["seeds"]
    if sorted(seed["seed"] for seed in seeds) != list(range(42, 52)):
        raise ValueError(f"Expected seeds 42-51 for {variant}")
    return {key: sum(seed["test"][population][key] for seed in seeds) / len(seeds)
            for key in METRICS}


def compare(rows, means, baseline, treatment, population):
    result = {}
    for metric in METRICS:
        a, b = rows[(baseline, population)][metric], rows[(treatment, population)][metric]
        ua, ub = means[(baseline, population)][metric], means[(treatment, population)][metric]
        item = {"baseline_displayed": a, "treatment_displayed": b,
                "displayed_relative_change_pct": relative(b, a),
                "baseline_unrounded": ua, "treatment_unrounded": ub,
                "unrounded_relative_change_pct": relative(ub, ua)}
        if metric == "ce":
            item.update({"displayed_baseline_distance_from_one": abs(a - 1),
                         "displayed_treatment_distance_from_one": abs(b - 1),
                         "displayed_distance_change_pct": relative(abs(b - 1), abs(a - 1)),
                         "unrounded_baseline_distance_from_one": abs(ua - 1),
                         "unrounded_treatment_distance_from_one": abs(ub - 1),
                         "unrounded_distance_change_pct": relative(abs(ub - 1), abs(ua - 1))})
        result[metric] = item
    return result


def section(text, start, end):
    pos = text.index(start)
    return text[pos:text.index(end, pos)]


def values_in(text):
    return [Decimal(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\\%", text)]


def main():
    original = TEX.read_text()
    text = uncomments(original)
    full, down, fig = read_json(FULL), read_json(DOWN), read_json(FIG)
    rows = {}
    for label, sampled in (("tab:aliccp-full-seeds-column", False),
                           ("tab:aliccp-full-ctr", False),
                           ("tab:aliccp-downsampled-5pct-column", True)):
        parsed = parse_table(text, label, sampled)
        if rows.keys() & parsed.keys():
            raise ValueError(f"Duplicate rows in {label}")
        rows.update(parsed)
    means = {(model, pop): source_means(full, down, MODELS[model], pop)
             for model, pop in rows}
    table_checks = []
    for key, values in rows.items():
        for metric, displayed in values.items():
            source = means[key][metric]
            table_checks.append({"model": key[0], "output": key[1], "metric": metric,
                                 "displayed": displayed, "unrounded_mean": source,
                                 "matches_display_rounding": source.quantize(displayed, rounding=ROUND_HALF_UP) == displayed})

    comparisons = {}
    for baseline, treatment, populations in (
        ("ESMM", "KD-0.3", ("pCVR", "CTCVR-logged", "CTR")),
        ("ESMM+DCN-V2", "KD-0.3", ("pCVR", "CTCVR-logged", "CTR")),
        ("Direct", "KD-0.3", ("pCVR", "CTCVR-logged")),
        ("DCN-V2", "KD-0.3", ("CTR",)),
        ("ESMM_R", "KD_R-0.3", ("pCVR", "CTCVR-logged"))):
        for pop in populations:
            comparisons[f"{treatment} vs {baseline} / {pop}"] = compare(rows, means, baseline, treatment, pop)

    def calc(key, metric, source="unrounded_relative_change_pct"):
        value = comparisons[key][metric][source]
        return -value if metric == "ne" else value

    full_key = "KD-0.3 vs ESMM / "
    down_key = "KD_R-0.3 vs ESMM_R / "
    def paired(prefix, source):
        return [calc(prefix + pop, metric, source) for metric in ("auc", "pr_auc", "ne")
                for pop in ("pCVR", "CTCVR-logged")]
    def triple(base, pop, source):
        return [calc(f"KD-0.3 vs {base} / {pop}", metric, source) for metric in ("auc", "pr_auc", "ne")]

    bars = {(row["variant"], row["comparison"], row["metric"]): row["mean"]
            for row in fig["raw_bar_values"]}
    kl = [-relative(bars[("kd03", k, "kl")], bars[("esmm_dcn", k, "kl")])
          for k in ("p_q", "pv_qv")]
    def specifications(source):
        f, d = paired(full_key, source), paired(down_key, source)
        # The retained-space NE claim says "both", so validate both separately.
        return [
            ("abstract full", section(text, "On the Ali-CCP dataset, relative", "Additionally, reflecting"), f),
            ("abstract downsampled", section(text, "Relative to retained-space ESMM", r"\end{abstract}"), d[:5]),
            ("introduction full", section(text, r"\item \textbf{Full-population evidence.}", r"\item \textbf{Negative-downsampling evidence.}"), f),
            ("introduction downsampled", section(text, r"\item \textbf{Negative-downsampling evidence.}", r"\end{itemize}"), [Decimal(5)] + d[:5]),
            ("serving results", section(text, r"\paragraph{KD improves serving pCTCVR", r"\paragraph{KD aligns"), triple("ESMM", "CTCVR-logged", source) + triple("ESMM+DCN-V2", "CTCVR-logged", source) + triple("Direct", "CTCVR-logged", source)),
            ("alignment results", section(text, r"\paragraph{KD aligns", r"\paragraph{KD improves pCVR"), kl),
            ("pCVR results", section(text, r"\paragraph{KD improves pCVR", r"\begin{table}"), triple("ESMM", "pCVR", source) + triple("ESMM+DCN-V2", "pCVR", source) + triple("Direct", "pCVR", source)),
            ("downsampled results", section(text, r"\paragraph{Ranking gains persist", r"\begingroup"), [Decimal(5)] + d[:5]),
            ("conclusion", section(text, r"\section{Conclusion}", r"\bibliographystyle"), [f[1], f[0], f[3], f[2]]),
        ]

    specs = specifications("unrounded_relative_change_pct")
    displayed_specs = specifications("displayed_relative_change_pct")
    checks = []
    for (name, excerpt, expected), (display_name, _, displayed_expected) in zip(specs, displayed_specs):
        assert name == display_name
        reported = values_in(excerpt)
        if len(reported) != len(expected):
            raise ValueError(f"{name}: found {len(reported)} percentages; expected {len(expected)}")
        for index, (observed, value, displayed_value) in enumerate(zip(reported, expected, displayed_expected), 1):
            checks.append({"section": name, "position_in_section": index, "reported_pct": observed,
                           "unrounded_computed_pct": value, "matches_unrounded": observed == round_one(value),
                           "displayed_computed_pct": displayed_value,
                           "matches_displayed": observed == round_one(displayed_value)})
    for name in ("abstract downsampled", "introduction downsampled", "downsampled results"):
        check = next(c for c in checks if c["section"] == name and c["position_in_section"] == (5 if name == "abstract downsampled" else 6))
        for source, field in (("unrounded_relative_change_pct", "matches_unrounded_both_outputs"),
                              ("displayed_relative_change_pct", "matches_displayed_both_outputs")):
            check[field] = check["reported_pct"] == round_one(calc(down_key + "CTCVR-logged", "ne", source))

    inventory = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for value in values_in(line):
            inventory.append({"line": lineno, "reported_pct": value, "text": line.strip()})
    validation_pct = Decimal("4.23") / (Decimal("38.07") + Decimal("4.23")) * 100
    report = {"source_files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (TEX, FULL, DOWN, FIG)},
              "definitions": {"relative_change_pct": "100 * (treatment - baseline) / baseline",
                              "ce_distance_change_pct": "100 * (|treatment CE - 1| - |baseline CE - 1|) / |baseline CE - 1|",
                              "interpretation": "CE distance from one measures aggregate bias, not full probability calibration.",
                              "rounding": "percentages rounded to one decimal, half up; table cells use their displayed precision"},
              "percentage_occurrences": inventory, "performance_claim_checks": checks,
              "validation_split_pct_from_displayed_counts": validation_pct,
              "table_cell_checks": table_checks, "comparisons": comparisons,
              "figure_kl_reductions_pct": dict(zip(("p_q", "pv_qv"), kl))}
    OUT.write_text(json.dumps(report, indent=2, default=lambda x: str(x) if isinstance(x, Decimal) else x) + "\n")
    failures = [c for c in checks if not c["matches_unrounded"] or c.get("matches_unrounded_both_outputs") is False]
    display_failures = [c for c in checks if not c["matches_displayed"] or c.get("matches_displayed_both_outputs") is False]
    table_failures = [c for c in table_checks if not c["matches_display_rounding"]]
    print(f"Saved {OUT.relative_to(ROOT)}; {len(inventory)} percentage occurrences, "
          f"{len(checks)} mapped claims, {len(table_checks)} table cells.")
    print(f"Claim mismatches: {len(failures)} from unrounded means; {len(display_failures)} from displayed values; "
          f"displayed cell rounding mismatches: {len(table_failures)}")
    for item in failures + table_failures:
        print(json.dumps(item, default=str))


if __name__ == "__main__":
    main()
