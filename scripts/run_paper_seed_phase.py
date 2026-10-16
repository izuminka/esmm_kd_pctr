"""Run GPU prediction and CPU scoring separately against a frozen paper study."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _context(study, task):
    study = Path(study).resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    source = Path(manifest["source_code"])
    spec = importlib.util.spec_from_file_location(
        "frozen_paper_seed_study", source / "scripts/run_paper_seed_study.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.verify(manifest)
    if not 0 <= task < len(manifest["jobs"]):
        raise ValueError(f"invalid task index: {task}")
    return manifest, manifest["jobs"][task], source


def _validate_run(manifest, job, saved):
    if saved.get("seed") != job["seed"] or saved.get("arm") != job["arm"]:
        raise ValueError("run record does not match the requested task")
    run = Path(saved["run_dir"]).resolve()
    if json.loads((run / "run.json").read_text()).get("status") != "FINISHED":
        raise ValueError(f"recorded run is unfinished: {run}")
    if _digest(run / "config.yaml") != job["config_sha256"]:
        raise ValueError("recorded run configuration differs from the task")
    cfg = yaml.safe_load(Path(job["config"]).read_text())
    expected_parent = (Path(manifest["project_root"]) / cfg["training"]["out_dir"] / job["arm"]).resolve()
    if run.parent != expected_parent:
        raise ValueError("recorded run is outside the requested arm")
    return run


def _recorded_run(manifest, job):
    record = Path(job["run_record"])
    if not record.is_file():
        raise ValueError(f"GPU phase has not recorded a finished run: {record}")
    return _validate_run(manifest, job, json.loads(record.read_text()))


def gpu_task(study, task):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU phase requires exactly one visible CUDA GPU")
    manifest, job, source = _context(study, task)
    record = Path(job["run_record"])
    if not record.exists():
        registry = Path(job["registry"])
        if not registry.exists():
            subprocess.run([manifest["python"], "-u", str(source / "scripts/train.py"),
                            "--config", job["config"], "--best-models", str(registry)],
                           cwd=source, check=True)
        values = yaml.safe_load(registry.read_text())
        if not isinstance(values, dict) or set(values) != {job["arm"]}:
            raise ValueError("private registry does not identify one run")
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        run = (Path(manifest["project_root"]) / cfg["training"]["out_dir"]
               / job["arm"] / values[job["arm"]]["run"]).resolve()
        saved = {"seed": job["seed"], "arm": job["arm"], "run_dir": str(run)}
        _validate_run(manifest, job, saved)
        with record.open("x") as stream:
            json.dump(saved, stream, indent=2)
            stream.write("\n")
    run = _recorded_run(manifest, job)
    # A separate interpreter ensures every inference import comes from the
    # original snapshot and releases training allocations before test export.
    prediction = ("import sys; from pathlib import Path; "
                  "sys.path.insert(0, sys.argv[1]); "
                  "from evaluation.predictions import test_predictions; "
                  "test_predictions(Path(sys.argv[2]), Path(sys.argv[3]), gpus=1)")
    subprocess.run([manifest["python"], "-u", "-c", prediction,
                    str(source), str(run), manifest["project_root"]], cwd=source, check=True)
    return run


def cpu_task(study, task):
    manifest, job, source = _context(study, task)
    run = _recorded_run(manifest, job)
    for name in ("test.parquet", "test.provenance.json"):
        if not (run / "predictions" / name).is_file():
            raise ValueError(f"GPU prediction cache is missing {name}; CPU inference is disabled")
    subprocess.run([manifest["python"], "-u", str(source / "scripts/evaluate_run.py"),
                    "--run-dir", str(run), "--gpus", "0",
                    "--logged-config", manifest["logged_config"]], cwd=source, check=True)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--gpu-task", nargs=2, metavar=("STUDY", "TASK"))
    action.add_argument("--cpu-task", nargs=2, metavar=("STUDY", "TASK"))
    args = parser.parse_args(argv)
    if args.gpu_task:
        print(gpu_task(Path(args.gpu_task[0]), int(args.gpu_task[1])))
    else:
        print(cpu_task(Path(args.cpu_task[0]), int(args.cpu_task[1])))


if __name__ == "__main__":
    main()
