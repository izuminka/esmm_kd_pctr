"""Extend the matched direct-product baseline to student seeds 42--51.

Pin the original four runs, freeze the working source, and train only missing
seeds. GPU prediction and CPU scoring use the existing paper phase runner.
"""

import argparse
import copy
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from scripts.report_paper_ten_seeds import normalized_config, validate_run
from scripts.run_paper_seed_study import digest, git, verify, write

SEEDS = list(range(42, 52))
MODEL = "esmm_direct_product"
TEACHER = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"
ORIGINAL_STUDY = "artifacts/direct_product_studies/20261015T235815Z"


def read(path):
    return json.loads(Path(path).read_text())


def pinned_original_runs(baseline_study, root):
    """Resolve the completed four-seed study without reading latest pointers."""
    baseline_study, root = Path(baseline_study), Path(root)
    by_seed = {}
    for row in read(baseline_study / "seed_summary.json")["per_seed"]:
        seed = row["seed"]
        run = (root / row["run"]).resolve()
        if seed in by_seed and by_seed[seed] != run:
            raise ValueError(f"original study identifies conflicting runs for seed {seed}")
        by_seed[seed] = run
    if sorted(by_seed) != [42, 43, 44, 45]:
        raise ValueError("original direct study must identify exactly seeds 42--45")
    for seed, run in by_seed.items():
        arm = f"{MODEL}_seed{seed}"
        registry = yaml.safe_load((baseline_study / f"best_models_seed{seed}.yaml").read_text())
        if (not isinstance(registry, dict) or set(registry) != {arm}
                or registry[arm]["run"] != run.name
                or run.parent != (root / "artifacts" / arm).resolve()):
            raise ValueError(f"original private registry disagrees with seed {seed}")
    return by_seed


def _training_run(run, seed, reference, root):
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    if cfg.get("model") != MODEL or cfg.get("data", {}).get("nonclick_sampling") is not None:
        raise ValueError(f"expected full-population direct-product run: {run}")
    validate_run(run, {"id": "direct", "regime": "full", "checkpoint_metric": "logged_ctcvr_ne"},
                 seed, reference, root, training_only=True)


def _snapshot(code, target):
    target.mkdir(parents=True)
    names = set(git(code, "ls-files", "-z").split("\0"))
    names.update(git(code, "ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    hashes = {}
    for name in sorted(names - {""}):
        source = code / name
        if source.is_file():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[name] = digest(destination)
    return hashes


def prepare(study, root=ROOT, code=CODE, python=sys.executable, baseline_study=None):
    root, code, study = [Path(p).resolve() for p in (root, code, study)]
    baseline_study = Path(baseline_study or root / ORIGINAL_STUDY).resolve()
    original = pinned_original_runs(baseline_study, root)
    reference_path = original[42] / "config.yaml"
    reference = yaml.safe_load(reference_path.read_text())
    if (reference.get("model") != MODEL
            or reference["training"].get("checkpoint_metric") != "logged_ctcvr_ne"):
        raise ValueError("original configuration must select the direct serving product")
    for seed, run in original.items():
        _training_run(run, seed, reference, root)

    study.mkdir(parents=True, exist_ok=False)
    (study / "runs").mkdir()
    (study / "registries").mkdir()
    teacher = root / TEACHER
    predictions = {split: str(teacher / "predictions" / f"{split}.parquet")
                   for split in ("train", "val", "test")}
    hashes = {split: digest(path) for split, path in predictions.items()}
    manifest = {"model": MODEL, "variant": "direct", "seeds": SEEDS, "epochs": 5,
                "checkpoint_metric": "logged_ctcvr_ne", "reference_config": str(reference_path),
                "project_root": str(root), "python": str(Path(python).absolute()),
                "baseline_study": str(baseline_study),
                "code_commit": git(code, "rev-parse", "HEAD").strip(),
                "teacher": {"run_dir": str(teacher), "test_predictions": predictions["test"],
                            "pctr_scores_sha256": hashes["test"], "predictions": predictions,
                            "predictions_sha256": hashes}, "runs": [], "jobs": []}
    logged = study / "logged_eval.yaml"
    logged.write_text(yaml.safe_dump({"pctr_model": "dcn_matched_stats_wd3e6_seed42", "logged_pctr": {
        "dcn_run": TEACHER, "predictions": {"test": str(Path(predictions["test"]).relative_to(root))},
        "score_column": "score"}}, sort_keys=False))
    manifest.update(logged_config=str(logged), logged_config_sha256=digest(logged))
    for seed in SEEDS:
        arm = f"{MODEL}_seed{seed}"
        run = original.get(seed)
        if run is None:
            candidates = [p.parent.resolve() for p in (root / "artifacts" / arm).glob("*/run.json")
                          if read(p).get("status") == "FINISHED"]
            if len(candidates) > 1:
                raise ValueError(f"ambiguous finished direct runs for seed {seed}")
            run = candidates[0] if candidates else None
        if run is not None:
            _training_run(run, seed, reference, root)
        item = {"variant": "direct", "seed": seed, "arm": arm,
                "run_dir": str(run) if run else None,
                "run_record": str(study / "runs" / f"direct_{seed}.json")}
        if run is None:
            cfg = copy.deepcopy(reference)
            cfg["training"].update(seed=seed, evaluate_test=False, gpus=1)
            cfg["logged_pctr"]["predictions"].pop("test", None)
            if normalized_config(cfg, root) != normalized_config(reference, root):
                raise ValueError("generated seed configuration changed the training recipe")
            config = study / "configs" / arm / "train.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            item.update(config=str(config), config_sha256=digest(config),
                        registry=str(study / "registries" / f"direct_{seed}.yaml"),
                        task=len(manifest["jobs"]))
            manifest["jobs"].append(item)
        manifest["runs"].append(item)
    snapshot = study / "source/code"
    manifest.update(source_code=str(snapshot), source_sha256=_snapshot(code, snapshot))
    for name in ("artifacts", "artifacts_downsampled", "ali-ccp"):
        (snapshot.parent / name).symlink_to(root / name, target_is_directory=True)
    (study / "source.diff").write_text(git(code, "diff", "HEAD"))
    write(study / "manifest.json", manifest)

    interpreter, location = shlex.quote(manifest["python"]), shlex.quote(str(study))
    phase = shlex.quote(str(snapshot / "scripts/run_paper_seed_phase.py"))
    common = ["#!/bin/bash", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4"]
    for filename, kind, concurrency, memory in (("train.sbatch", "gpu", 4, "64G"),
                                                 ("score.sbatch", "cpu", 8, "24G")):
        if not manifest["jobs"]:
            continue
        lines = [*common, f"#SBATCH --job-name=direct10_{kind}", f"#SBATCH --mem={memory}",
                 "#SBATCH --time=04:00:00", f"#SBATCH --array=0-{len(manifest['jobs']) - 1}%{concurrency}",
                 f"#SBATCH --output={study}/{kind}_%A_%a.out", f"#SBATCH --error={study}/{kind}_%A_%a.err"]
        if kind == "gpu":
            lines.append("#SBATCH --gres=gpu:1")
        lines.extend(["set -euo pipefail", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
                      f'{interpreter} -u {phase} --{kind}-task {location} "$SLURM_ARRAY_TASK_ID"'])
        (study / filename).write_text("\n".join(lines) + "\n")
    reporter = shlex.quote(str(snapshot / "scripts/report_direct_product_ten_seeds.py"))
    lines = [*common, "#SBATCH --job-name=direct10_report", "#SBATCH --mem=32G", "#SBATCH --time=01:00:00",
             f"#SBATCH --output={study}/report_%j.out", f"#SBATCH --error={study}/report_%j.err",
             "set -euo pipefail", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
             f"{interpreter} -u {reporter} --study-dir {location} --project-root {shlex.quote(str(root))}"]
    (study / "report.sbatch").write_text("\n".join(lines) + "\n")
    return manifest


def submit(study):
    study = Path(study).resolve()
    manifest = read(study / "manifest.json")
    verify(manifest)
    for split, expected in manifest["teacher"]["predictions_sha256"].items():
        if digest(manifest["teacher"]["predictions"][split]) != expected:
            raise ValueError(f"fixed teacher {split} predictions changed")
    record = study / "submission_jobs.json"
    with record.open("x") as stream:
        json.dump({"state": "submitting"}, stream)

    def sbatch(*args):
        result = subprocess.check_output(["sbatch", "--parsable", *map(str, args)], text=True).strip().split(";", 1)[0]
        if not result.isdigit():
            raise ValueError(f"unexpected Slurm job ID: {result}")
        return result

    jobs = {}
    if manifest["jobs"]:
        jobs["train_job_id"] = sbatch(study / "train.sbatch")
        write(record, {**jobs, "state": "training_submitted"})
        jobs["score_job_id"] = sbatch(f"--dependency=aftercorr:{jobs['train_job_id']}", study / "score.sbatch")
        write(record, {**jobs, "state": "scoring_submitted"})
        jobs["report_job_id"] = sbatch(f"--dependency=afterok:{jobs['score_job_id']}", study / "report.sbatch")
    else:
        jobs["report_job_id"] = sbatch(study / "report.sbatch")
    write(record, {**jobs, "state": "submitted"})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--submit", type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prepare)
        print(f"Prepared {len(result['jobs'])} missing direct-product seeds in {args.prepare}")
    else:
        print(json.dumps(submit(args.submit)))


if __name__ == "__main__":
    main()
