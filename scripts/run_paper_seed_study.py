"""Extend the main paper's eight variants to matched student seeds 42--51.

Prepare pins existing runs and snapshots working source. GPU tasks train missing
seeds and export predictions; CPU tasks score them. Reporting never uses latest.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
SEEDS = list(range(42, 52))
TEACHER = "artifacts/dcn_matched_stats_wd3e6_seed42/20260917T232959Z"
LEGACY_TEACHER = "artifacts/dcn_matched_stats_wd3e6/20260917T232959Z"
RECIPES = (
    ("esmm", "ESMM", "full", "artifacts", "esmm_ctcvr_ne", "20260918T183118Z"),
    ("esmm_dcn", "ESMM + DCN", "full", "artifacts", "esmm_dcn", "20261016T030003Z"),
    ("kd03", "KD--0.3", "full", "artifacts", "esmm_logged_pctr_kd03", "20260918T204348Z"),
    ("kd05", "KD--0.5", "full", "artifacts", "esmm_logged_pctr_kd05", "20260918T174343Z"),
    ("kd07", "KD--0.7", "full", "artifacts", "esmm_logged_pctr_kd07", "20260918T205311Z"),
    ("kd0", "KD--0", "full", "artifacts", "esmm_logged_pctr_kd_teacher", "20260918T174343Z"),
    ("down_esmm", "ESMM", "downsampled", "artifacts_downsampled", "esmm_retention_5pct", "20260923T014746Z"),
    ("down_kd_tau", "KD tau--0.5", "downsampled", "artifacts_downsampled", "esmm_logged_pctr_kd_tau_retention_5pct", "20260923T235217Z"),
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(code, *args):
    return subprocess.check_output(["git", "-C", str(code), *args], text=True)


def existing_run(root, tree, base, seed, baseline):
    # Seed 42 predates the _seed42 suffix for these original arms.
    name = base if seed == 42 and base not in ("esmm_ctcvr_ne", "esmm_dcn", "esmm_logged_pctr_kd03", "esmm_logged_pctr_kd07") else f"{base}_seed{seed}"
    if seed == 42:
        run = root / tree / name / baseline
        if not (run / "run.json").is_file():
            raise ValueError(f"missing original paper run: {run}")
        if read(run / "run.json").get("status") != "FINISHED":
            raise ValueError(f"original paper run is unfinished: {run}")
        return run.resolve()
    candidates = [p.parent.resolve() for p in (root / tree / name).glob("*/run.json")
                  if read(p).get("status") == "FINISHED"]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous existing runs for {name}: {candidates}")
    return candidates[0] if candidates else None


def training_config(reference, seed):
    cfg = copy.deepcopy(reference)
    cfg["training"].update(seed=seed, evaluate_test=False, gpus=1)
    # Test is scored independently after training, using the same frozen source.
    if cfg["model"] == "esmm":
        cfg.pop("logged_pctr", None)
        cfg.pop("evaluation", None)
    if "logged_pctr" in cfg:
        block = cfg["logged_pctr"]
        block["dcn_run"] = block["dcn_run"].replace(LEGACY_TEACHER, TEACHER)
        block["predictions"] = {key: path.replace(LEGACY_TEACHER, TEACHER)
                                 for key, path in block["predictions"].items() if key != "test"}
    return cfg


def prepare(study, root=ROOT, code=CODE, python=sys.executable):
    root, code, study = map(lambda p: Path(p).resolve(), (root, code, study))
    study.mkdir(parents=True, exist_ok=False)
    teacher = root / TEACHER
    test_scores = teacher / "predictions/test.parquet"
    manifest = {"seeds": SEEDS, "project_root": str(root), "python": str(Path(python).absolute()),
                "teacher": {"run_dir": str(teacher), "test_predictions": str(test_scores),
                            "pctr_scores_sha256": digest(test_scores)},
                "variants": [], "jobs": [], "code_commit": git(code, "rev-parse", "HEAD").strip()}
    logged_config = study / "logged_eval.yaml"
    logged_config.write_text(yaml.safe_dump({"pctr_model": "dcn_matched_stats_wd3e6_seed42", "logged_pctr": {
        "dcn_run": TEACHER, "predictions": {"test": str(test_scores.relative_to(root))}, "score_column": "score"}}, sort_keys=False))
    manifest["logged_config"] = str(logged_config)
    manifest["logged_config_sha256"] = digest(logged_config)
    for variant, label, regime, tree, base, baseline in RECIPES:
        original = existing_run(root, tree, base, 42, baseline)
        reference = yaml.safe_load((original / "config.yaml").read_text())
        entry = {"id": variant, "label": label, "regime": regime,
                 "checkpoint_metric": reference["training"]["checkpoint_metric"],
                 "reference_config": str(original / "config.yaml"), "runs": []}
        for seed in SEEDS:
            run = existing_run(root, tree, base, seed, baseline)
            arm = f"{base}_seed{seed}"
            item = {"variant": variant, "seed": seed, "arm": arm,
                    "run_dir": str(run) if run else None,
                    "run_record": str(study / "runs" / f"{variant}_{seed}.json")}
            if run is None:
                config = study / "configs" / arm / "train.yaml"
                config.parent.mkdir(parents=True)
                cfg = training_config(reference, seed)
                for path in cfg.get("logged_pctr", {}).get("predictions", {}).values():
                    if not (root / path).is_file():
                        raise ValueError(f"missing teacher scores: {path}")
                config.write_text(yaml.safe_dump(cfg, sort_keys=False))
                item.update(config=str(config), config_sha256=digest(config),
                            registry=str(study / "registries" / f"{variant}_{seed}.yaml"),
                            task=len(manifest["jobs"]))
                manifest["jobs"].append(item)
            entry["runs"].append(item)
        manifest["variants"].append(entry)
    snapshot = study / "source/code"
    snapshot.mkdir(parents=True)
    names = set(git(code, "ls-files", "-z").split("\0"))
    names.update(git(code, "ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    hashes = {}
    for name in sorted(names - {""}):
        source = code / name
        if source.is_file():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = digest(target)
    for name in ("artifacts", "artifacts_downsampled", "ali-ccp"):
        (snapshot.parent / name).symlink_to(root / name, target_is_directory=True)
    (study / "source.diff").write_text(git(code, "diff", "HEAD"))
    manifest.update(source_code=str(snapshot), source_sha256=hashes)
    (study / "runs").mkdir()
    (study / "registries").mkdir()
    write(study / "manifest.json", manifest)
    script = shlex.quote(str(snapshot / "scripts/run_paper_seed_phase.py"))
    interpreter = shlex.quote(manifest["python"])
    location = shlex.quote(str(study))
    lines = ["#!/bin/bash", "#SBATCH --job-name=paper_10seeds", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=4", "#SBATCH --mem=64G", "#SBATCH --gres=gpu:1", "#SBATCH --time=04:00:00",
             f"#SBATCH --array=0-{len(manifest['jobs']) - 1}%4", f"#SBATCH --output={study}/slurm_%A_%a.out",
             f"#SBATCH --error={study}/slurm_%A_%a.err", "set -euo pipefail", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
             f'{interpreter} -u {script} --gpu-task {location} "$SLURM_ARRAY_TASK_ID"']
    if manifest["jobs"]:
        (study / "train.sbatch").write_text("\n".join(lines) + "\n")
        scoring = ["#!/bin/bash", "#SBATCH --job-name=paper_10seeds_score", "#SBATCH --nodes=1",
                   "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4", "#SBATCH --mem=24G",
                   "#SBATCH --time=01:00:00", f"#SBATCH --array=0-{len(manifest['jobs']) - 1}%8",
                   f"#SBATCH --output={study}/score_%A_%a.out", f"#SBATCH --error={study}/score_%A_%a.err",
                   "set -euo pipefail", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
                   f'{interpreter} -u {script} --cpu-task {location} "$SLURM_ARRAY_TASK_ID"']
        (study / "score.sbatch").write_text("\n".join(scoring) + "\n")
    report_script = shlex.quote(str(snapshot / "scripts/report_paper_ten_seeds.py"))
    report_lines = ["#!/bin/bash", "#SBATCH --job-name=paper_10seeds_report", "#SBATCH --nodes=1",
                    "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4", "#SBATCH --mem=32G",
                    "#SBATCH --time=01:00:00", f"#SBATCH --output={study}/report_%j.out",
                    f"#SBATCH --error={study}/report_%j.err", "set -euo pipefail",
                    f"{interpreter} -u {report_script} --study-dir {location} --project-root {shlex.quote(str(root))}"]
    (study / "report.sbatch").write_text("\n".join(report_lines) + "\n")
    return manifest


def verify(manifest):
    for name, expected in manifest["source_sha256"].items():
        if digest(Path(manifest["source_code"]) / name) != expected:
            raise ValueError(f"source changed: {name}")
    for job in manifest["jobs"]:
        if digest(job["config"]) != job["config_sha256"]:
            raise ValueError(f"configuration changed: {job['arm']}")
    if digest(manifest["logged_config"]) != manifest["logged_config_sha256"]:
        raise ValueError("logged evaluation config changed")


def run_task(study, task):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each task requires exactly one visible GPU")
    manifest = read(Path(study) / "manifest.json")
    verify(manifest)
    if not 0 <= task < len(manifest["jobs"]):
        raise ValueError(f"invalid task index: {task}")
    job = manifest["jobs"][task]
    record = Path(job["run_record"])
    source = Path(manifest["source_code"])
    if not record.exists():
        registry = Path(job["registry"])
        if not registry.exists():
            subprocess.run([manifest["python"], "-u", str(source / "scripts/train.py"), "--config", job["config"],
                            "--best-models", str(registry)], cwd=source, check=True)
        values = yaml.safe_load(registry.read_text())
        if set(values) != {job["arm"]}:
            raise ValueError("private registry does not identify one run")
        config = yaml.safe_load(Path(job["config"]).read_text())
        run = Path(manifest["project_root"]) / config["training"]["out_dir"] / job["arm"] / values[job["arm"]]["run"]
        if read(run / "run.json")["status"] != "FINISHED":
            raise ValueError(f"training did not finish: {run}")
        write(record, {"seed": job["seed"], "arm": job["arm"], "run_dir": str(run.resolve())})
    saved = read(record)
    if saved.get("seed") != job["seed"] or saved.get("arm") != job["arm"]:
        raise ValueError("run record does not match the requested task")
    run = saved["run_dir"]
    if read(Path(run) / "run.json").get("status") != "FINISHED":
        raise ValueError(f"recorded run is unfinished: {run}")
    if digest(Path(run) / "config.yaml") != job["config_sha256"]:
        raise ValueError("recorded run configuration differs from the task")
    subprocess.run([manifest["python"], "-u", str(source / "scripts/evaluate_run.py"), "--run-dir", run,
                    "--gpus", "1", "--logged-config", manifest["logged_config"]], cwd=source, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path)
    actions.add_argument("--submit", type=Path)
    actions.add_argument("--run-task", nargs=2, metavar=("STUDY", "TASK"))
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prepare)
        print(f"Prepared {len(result['jobs'])} missing runs in {args.prepare}")
    elif args.submit:
        study = args.submit.resolve()
        verify(read(study / "manifest.json"))
        record = study / "submission_jobs.json"
        with record.open("x") as stream:
            json.dump({"state": "submitting"}, stream)
        def sbatch(*arguments):
            result = subprocess.check_output(["sbatch", "--parsable", *map(str, arguments)], text=True).strip()
            job_id = result.split(";", 1)[0]
            if not job_id.isdigit():
                raise ValueError(f"unexpected Slurm job ID: {result}")
            return job_id

        jobs = {}
        if read(study / "manifest.json")["jobs"]:
            jobs["train_job_id"] = sbatch(study / "train.sbatch")
            write(record, {**jobs, "state": "training_submitted"})
            jobs["score_job_id"] = sbatch(f"--dependency=aftercorr:{jobs['train_job_id']}", study / "score.sbatch")
            write(record, {**jobs, "state": "scoring_submitted"})
            jobs["report_job_id"] = sbatch(f"--dependency=afterok:{jobs['score_job_id']}", study / "report.sbatch")
        else:
            jobs["report_job_id"] = sbatch(study / "report.sbatch")
        write(record, {**jobs, "state": "submitted"})
        print(json.dumps(jobs))
    else:
        run_task(Path(args.run_task[0]).resolve(), int(args.run_task[1]))


if __name__ == "__main__":
    main()
