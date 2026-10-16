"""Prepare, optionally submit, and report a four-seed ESMM with DCN study.

The esmm_dcn model replaces only the ESMM CTR head, retaining its shared
embeddings and training recipe.

Preparation snapshots the working source (including untracked files). Jobs run
that snapshot, so edits made while Slurm is queued cannot change the experiment.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
SEEDS = (42, 43, 44, 45)
POPULATIONS = ("CTR", "pCVR", "CTCVR", "CTCVR-logged")
METRICS = ("auc", "pr_auc", "ne", "ce")
MODEL = "esmm_dcn"


def _json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(code, *args):
    return subprocess.run(["git", "-C", str(code), *args], check=True,
                          capture_output=True, text=True).stdout


def _check_config(cfg):
    training = cfg["training"]
    required = {"epochs": 5, "batch_size": 65536,
                "checkpoint_metric": "ctcvr_ne", "keep_checkpoint": "best",
                "evaluate_test": False, "load_test": False}
    if (cfg["model"] != MODEL or training.get("gpus") not in (-1, 1)
            or any(training.get(k) != v for k, v in required.items())):
        raise ValueError("study requires esmm_dcn, five epochs, batch 65536, "
                         "one GPU, best ctcvr_ne checkpoint and separate test evaluation")
    if "dcn_optimizer" in training:
        raise ValueError("esmm_dcn retains the ESMM optimizer; dcn_optimizer is not supported")
    if (training.get("early_stopping", {}).get("enabled", True)
            or "logged_pctr" in cfg or "train_cohort" in cfg["data"]
            or "nonclick_sampling" in cfg["data"]):
        raise ValueError("study requires full-population training without early stopping or logged pCTR")


def prepare(study, config, logged_config, *, code=CODE, root=ROOT,
            python=sys.executable, partition=None, cpus=4):
    """Write a new study without submitting jobs or touching the best-model file."""
    study, config, logged_config, code, root = map(
        lambda p: Path(p).resolve(), (study, config, logged_config, code, root))
    cfg = yaml.safe_load(config.read_text())
    _check_config(cfg)
    variant = cfg["model"]
    logged = yaml.safe_load(logged_config.read_text())
    if (set(logged) != {"pctr_model", "logged_pctr"}
            or set(logged["logged_pctr"]["predictions"]) != {"test"}):
        raise ValueError("logged config must specify exactly one test pCTR source")
    study.mkdir(parents=True, exist_ok=False)
    snapshot = study / "source" / "code"
    snapshot.mkdir(parents=True)
    files = set(_git(code, "ls-files", "-z").split("\0"))
    untracked = _git(code, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    files.update(untracked)
    hashes = {}
    for name in sorted(files - {""}):
        source = code / name
        if not source.is_file():
            continue  # Deleted tracked files remain absent in the snapshot.
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = _hash(target)
    (study / "code.diff").write_text(_git(code, "diff", "HEAD"))
    # The existing scripts resolve data/artifacts relative to the parent of code/.
    # Symlink only these external roots; executable source stays in the snapshot.
    paths = [cfg["data"]["encoded_dir"], cfg["training"]["out_dir"],
             logged["logged_pctr"]["dcn_run"],
             *logged["logged_pctr"]["predictions"].values()]
    if cfg["data"].get("csv_dir"):
        paths.append(cfg["data"]["csv_dir"])
    for path in map(Path, paths):
        if path.is_absolute():
            continue
        if ".." in path.parts or path.parts[0] == "code":
            raise ValueError(f"unsupported external path in snapshot: {path}")
        target = snapshot.parent / path.parts[0]
        if not target.exists() and not target.is_symlink():
            target.symlink_to(root / path.parts[0], target_is_directory=True)
    configs = study / "configs"
    configs.mkdir()
    shutil.copy2(config, configs / "base_train.yaml")
    shutil.copy2(logged_config, configs / "logged_eval.yaml")
    jobs = []
    for seed in SEEDS:
        arm = f"{variant}_seed{seed}"
        seed_config = configs / arm / "train.yaml"
        seed_config.parent.mkdir()
        per_seed = {**cfg, "training": {**cfg["training"], "seed": seed}}
        seed_config.write_text(yaml.safe_dump(per_seed, sort_keys=False))
        jobs.append({"seed": seed, "arm": arm, "config": str(seed_config),
                     "config_sha256": _hash(seed_config),
                     "registry": str(study / f"best_models_seed{seed}.yaml"),
                     "run_record": str(study / f"run_seed{seed}.json")})
    manifest = {"model": variant, "variant": variant,
                "project_root": str(root), "source_code": str(snapshot),
                "python": str(Path(python).absolute()), "seeds": list(SEEDS),
                "epochs": 5, "batch_size": 65536, "checkpoint_metric": "ctcvr_ne",
                "logged_config": str(configs / "logged_eval.yaml"),
                "logged_config_sha256": _hash(configs / "logged_eval.yaml"),
                "code_commit": _git(code, "rev-parse", "HEAD").strip(),
                "code_status": _git(code, "status", "--porcelain"),
                "untracked": sorted(set(untracked) - {""}), "source_sha256": hashes,
                "jobs": jobs}
    _json(study / "submission.json", manifest, exclusive=True)
    interpreter = shlex.quote(manifest["python"])
    script = shlex.quote(str(snapshot / "scripts" / Path(__file__).name))
    location = shlex.quote(str(study))
    common = ["#!/bin/bash", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
              f"#SBATCH --cpus-per-task={cpus}", "#SBATCH --mem=64G",
              "#SBATCH --time=02:00:00"]
    if partition:
        common.append(f"#SBATCH --partition={partition}")
    train_lines = [*common, f"#SBATCH --job-name={variant}",
                   "#SBATCH --array=0-3%4", "#SBATCH --gres=gpu:1",
                   f"#SBATCH --output={study}/slurm_%A_%a.out",
                   f"#SBATCH --error={study}/slurm_%A_%a.err",
                   "set -euo pipefail", f"export OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}",
                   f"{interpreter} -u {script} --run-seed {location} "
                   '"$((42 + SLURM_ARRAY_TASK_ID))"']
    (study / "train.sbatch").write_text("\n".join(train_lines) + "\n")
    # Reporting is CPU-only; it must not request another accelerator.
    report_lines = [*common, f"#SBATCH --job-name={variant}_report",
                    f"#SBATCH --output={study}/report_%j.out", "set -euo pipefail",
                    f"{interpreter} -u {script} --report-study {location}"]
    (study / "report.sbatch").write_text("\n".join(report_lines) + "\n")
    return study


def _manifest(study):
    return json.loads((Path(study) / "submission.json").read_text())


def _verify_inputs(manifest):
    snapshot = Path(manifest["source_code"])
    for name, expected in manifest["source_sha256"].items():
        if _hash(snapshot / name) != expected:
            raise ValueError(f"source snapshot changed: {name}")
    for job in manifest["jobs"]:
        if _hash(job["config"]) != job["config_sha256"]:
            raise ValueError(f"seed config changed: {job['seed']}")
    if _hash(manifest["logged_config"]) != manifest["logged_config_sha256"]:
        raise ValueError("logged evaluation config changed")


def submit(study):
    """Submit a GPU array and an after-success CPU report, recording both IDs."""
    study = Path(study).resolve()
    _verify_inputs(_manifest(study))
    path = study / "submission_jobs.json"
    _json(path, {"state": "submitting"}, exclusive=True)

    def sbatch(*args):
        result = subprocess.run(["sbatch", "--parsable", *map(str, args)],
                                check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdigit():
            raise ValueError(f"sbatch did not return a job ID: {result.stdout!r}")
        return job

    job = sbatch(study / "train.sbatch")
    _json(path, {"train_job_id": job, "state": "training_submitted"})
    report_job = sbatch(f"--dependency=afterok:{job}", study / "report.sbatch")
    _json(path, {"train_job_id": job, "report_job_id": report_job, "state": "submitted"})
    return job, report_job


def run_seed(study, seed):
    """Train once; resolve its private registry and retain that exact run forever."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each study task requires exactly one visible CUDA GPU")
    manifest = _manifest(study)
    _verify_inputs(manifest)
    job, = [job for job in manifest["jobs"] if job["seed"] == seed]
    record_path = Path(job["run_record"])
    snapshot = Path(manifest["source_code"])
    registry_path = Path(job["registry"])
    if not record_path.exists():
        if not registry_path.exists():
            subprocess.run([manifest["python"], "-u", str(snapshot / "scripts/train.py"),
                            "--config", job["config"], "--best-models", str(registry_path)],
                           cwd=snapshot, check=True)
        registry = yaml.safe_load(registry_path.read_text())
        if set(registry) != {job["arm"]}:
            raise ValueError("per-seed registry must contain only this seed's finished run")
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        run_dir = (Path(manifest["project_root"]) / cfg["training"]["out_dir"]
                   / job["arm"] / registry[job["arm"]]["run"]).resolve()
        run = json.loads((run_dir / "run.json").read_text())
        if run["status"] != "FINISHED":
            raise ValueError(f"cannot record unfinished run: {run_dir}")
        _json(record_path, {"seed": seed, "arm": job["arm"], "run_dir": str(run_dir),
                            "config_sha256": _hash(run_dir / "config.yaml")}, exclusive=True)
    record = json.loads(record_path.read_text())
    if record["seed"] != seed or record["arm"] != job["arm"]:
        raise ValueError("run record does not match its seed")
    subprocess.run([manifest["python"], "-u", str(snapshot / "scripts/evaluate_run.py"),
                    "--run-dir", record["run_dir"], "--gpus", "1",
                    "--logged-config", manifest["logged_config"]], cwd=snapshot, check=True)
    return Path(record["run_dir"])


def report(study):
    """Validate four completed runs, then write per-seed metrics and mean ± SEM."""
    sys.path.insert(0, str(CODE))
    from evaluation.reports import study_test_metrics

    study = Path(study).resolve()
    manifest = _manifest(study)
    _verify_inputs(manifest)
    variant = manifest.get("variant")
    if variant != MODEL or manifest.get("model") != MODEL:
        raise ValueError("study model and variant must identify esmm_dcn")
    if sorted(job["seed"] for job in manifest["jobs"]) != list(SEEDS):
        raise ValueError("study must contain exactly seeds 42–45")
    expected_logged = yaml.safe_load(Path(manifest["logged_config"]).read_text())
    teacher_name = expected_logged["pctr_model"]
    rows, sources, census = [], [], {}
    reference_config, teacher_identity = None, None
    for job in sorted(manifest["jobs"], key=lambda j: j["seed"]):
        record = json.loads(Path(job["run_record"]).read_text())
        if record["seed"] != job["seed"] or record["arm"] != job["arm"]:
            raise ValueError("run record seed differs from manifest")
        run_dir = Path(record["run_dir"])
        if _hash(run_dir / "config.yaml") != record["config_sha256"]:
            raise ValueError(f"run config changed: {run_dir}")
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        if cfg != yaml.safe_load(Path(job["config"]).read_text()):
            raise ValueError(f"run config differs from study: {run_dir}")
        _check_config(cfg)
        if cfg["model"] != variant or job["arm"] != f"{variant}_seed{job['seed']}":
            raise ValueError("run model or arm differs from study variant")
        if cfg["training"].pop("seed") != job["seed"]:
            raise ValueError(f"run seed differs from manifest: {run_dir}")
        if reference_config is not None and cfg != reference_config:
            raise ValueError("training configurations differ beyond seed")
        reference_config = cfg
        run = json.loads((run_dir / "run.json").read_text())
        if run["status"] != "FINISHED":
            raise ValueError(f"run is {run['status']}: {run_dir}")
        with (run_dir / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        if [int(r["epoch"]) for r in history] != list(range(1, 6)):
            raise ValueError(f"run must complete all five epochs: {run_dir}")
        scores = [float(r["val_score"]) for r in history]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("nonfinite validation selection score")
        best = history[scores.index(min(scores))]
        selected = run["selected_on"]
        if (selected["metric"] != "ctcvr_ne" or selected["split"] != "val"
                or selected["epoch"] != int(best["epoch"])
                or not math.isclose(selected["value"], float(best["val_score"]), rel_tol=1e-7)
                or any(not math.isclose(float(r["val_score"]), float(r["val_ctcvr_ne"]),
                                        rel_tol=1e-6) for r in history)):
            raise ValueError(f"checkpoint is not minimum validation native CTCVR NE: {run_dir}")
        metrics = study_test_metrics(run_dir)
        if set(metrics) != set(POPULATIONS):
            raise ValueError(f"required native and logged test populations missing: {run_dir}")
        logged_dir = run_dir / "metrics_logged" / teacher_name
        if yaml.safe_load((logged_dir / "config.yaml").read_text()) != expected_logged:
            raise ValueError("logged teacher config differs from study")
        provenance = json.loads((logged_dir / "provenance.json").read_text())
        identity = {key: provenance[key] for key in ("pctr_model", "pctr_scores_sha256", "logged_pctr")}
        if teacher_identity is not None and identity != teacher_identity:
            raise ValueError("logged teacher identity differs across seeds")
        teacher_identity = identity
        if (metrics["CTCVR-logged"]["pctr_model"] != teacher_name
                or identity["pctr_model"] != teacher_name
                or identity["logged_pctr"]["dcn_run"] != expected_logged["logged_pctr"]["dcn_run"]):
            raise ValueError("logged teacher identity differs from study")
        counts = {pop: (metric["rows"], metric["positives"]) for pop, metric in metrics.items()}
        if (counts["CTR"][0] != counts["CTCVR"][0]
                or counts["CTCVR"] != counts["CTCVR-logged"]
                or counts["pCVR"] != (counts["CTR"][1], counts["CTCVR"][1])):
            raise ValueError("native and logged test census inconsistent within run")
        for pop in POPULATIONS:
            if census.setdefault(pop, counts[pop]) != counts[pop]:
                raise ValueError(f"test census inconsistent across seeds: {pop}")
            values = {key: float(metrics[pop][key]) for key in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"nonfinite test metric: {run_dir}, {pop}")
            rows.append({"variant": variant, "seed": job["seed"],
                         "run_dir": str(run_dir), "selected_epoch": selected["epoch"],
                         "population": pop, "rows": counts[pop][0],
                         "positives": counts[pop][1], **values})
        sources.append({"variant": variant, "seed": job["seed"],
                        "run_dir": str(run_dir), "selected_on": selected})
    summary = []
    for pop in POPULATIONS:
        for metric in METRICS:
            values = [row[metric] for row in rows if row["population"] == pop]
            summary.append({"variant": variant, "population": pop,
                            "metric": metric, "n": 4, "mean": statistics.mean(values),
                            "sem": statistics.stdev(values) / math.sqrt(4), "values": values})
    _json(study / "seed_summary.json", {"variant": variant, "sources": sources, "per_seed": rows,
                                        "teacher_identity": teacher_identity, "summary": summary})
    for name, data in (("per_seed.csv", rows), ("seed_summary.csv", summary)):
        with (study / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = [f"{variant}: test mean ± SEM across seeds 42–45 (n=4), after five training epochs.",
             "Each checkpoint minimizes validation native CTCVR (q*v) NE; test data never select checkpoints.",
             "pCVR uses clicked impressions; CTR, native CTCVR and logged CTCVR use all impressions.",
             "Logged CTCVR uses the same frozen DCN source across seeds; native CTCVR uses the jointly trained DCN.",
             "SEM describes training-seed variability on a fixed test split.", "",
             "| Population | AUC | PR-AUC | NE | CE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for pop in POPULATIONS:
        cells = []
        for metric in METRICS:
            item = next(r for r in summary if r["population"] == pop and r["metric"] == metric)
            digits = 8 if metric == "pr_auc" else 6
            cells.append(f"{item['mean']:.{digits}f} ± {item['sem']:.{digits}f}")
        lines.append(f"| {pop} | " + " | ".join(cells) + " |")
    lines += ["", "Exact source runs:"]
    lines.extend(f"- Seed {s['seed']}, epoch {s['selected_on']['epoch']}: {s['run_dir']}" for s in sources)
    path = study / "seed_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run-seed", nargs=2, metavar=("STUDY_DIR", "SEED"))
    action.add_argument("--report-study", type=Path)
    parser.add_argument("--config", type=Path, default=CODE / "configs/esmm_dcn/train.yaml")
    parser.add_argument("--logged-config", type=Path,
                        help="defaults to eval_dcn_matched_stats_wd3e6_seed42.yaml beside --config")
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--submit", action="store_true", help="submit the prepared GPU array and CPU report")
    parser.add_argument("--partition", help="Slurm partition; default uses the cluster default")
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args(argv)
    if args.run_seed:
        print(run_seed(Path(args.run_seed[0]), int(args.run_seed[1])))
    elif args.report_study:
        print(report(args.report_study).read_text())
    else:
        if args.cpus < 1:
            parser.error("--cpus must be positive")
        cfg = yaml.safe_load(args.config.read_text())
        _check_config(cfg)
        study = args.study_dir or ROOT / f"artifacts/{cfg['model']}_studies" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        logged_config = args.logged_config or args.config.parent / "eval_dcn_matched_stats_wd3e6_seed42.yaml"
        print(prepare(study, args.config, logged_config, partition=args.partition, cpus=args.cpus))
        if args.submit:
            print("Training and report job IDs:", *submit(study))


if __name__ == "__main__":
    main()
