"""Queue, run and report the final ten-seed study (main.tex Tables 1 and 2) on Slurm.

Usage (from the project root):
    code/.venv/bin/python code/scripts/run_final_study.py --spec code/studies/final.yaml submit teacher
    code/.venv/bin/python code/scripts/run_final_study.py --spec code/studies/final.yaml submit students esmm down_esmm
    code/.venv/bin/python code/scripts/run_final_study.py --spec code/studies/final.yaml report

Each seed's config is its recipe with only training.seed changed, written under
<study_dir>/configs/<arm>/train.yaml; runs land where the recipe's out_dir says.

`submit teacher` queues one GPU array task per teacher candidate (every recipe
at every seed, training only) and a GPU job, after all of them succeed, that
keeps the run with the lowest validation CTR NE, scores it on test and writes
teacher.json and the logged-pCTR config students are scored with.

`submit students` queues one GPU array task per variant seed (training and
native test scoring) and a CPU job that scores every run's serving product
p*v with the selected teacher, after both the array and teacher selection.
Students whose training reads teacher scores must be submitted once the
teacher is selected, since their recipes name its run.

Every finished step leaves a record in <study_dir>/runs, so a resubmitted
array skips work already done. `task`, `select-teacher` and `logged` are what
the Slurm jobs execute.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
PYTHON = sys.executable
POPULATIONS = ("CTR", "pCVR", "CTCVR", "CTCVR-logged")
METRICS = ("auc", "pr_auc", "ne", "ce")
# Slurm requests per job type; the node has 4 GPUs, 128 CPUs and 900 GB.
GPU_JOB = {"cpus": 16, "mem": "120G", "time": "08:00:00"}
CPU_JOB = {"cpus": 16, "mem": "120G", "time": "08:00:00"}


def digest(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed_config(recipe: Path, seed: int) -> str:
    """The recipe's text with training.seed set, comments kept; checked by parsing."""
    text = recipe.read_text()
    new, count = re.subn(r"(?m)^(  seed: )\d+", rf"\g<1>{seed}", text)
    if count != 1:
        raise ValueError(f"{recipe}: expected exactly one training seed line")
    original, changed = yaml.safe_load(text), yaml.safe_load(new)
    original["training"]["seed"] = seed
    if changed != original:
        raise ValueError(f"{recipe}: seed substitution changed more than training.seed")
    return new


class Study:
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path
        self.spec = yaml.safe_load(spec_path.read_text())
        self.dir = ROOT / self.spec["study_dir"]
        self.seeds = list(self.spec["seeds"])
        for sub in ("configs", "runs", "registries", "logs", "slurm"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    # -- jobs ---------------------------------------------------------------

    def job(self, recipe: str, seed: int, name: str | None = None, **extra) -> dict:
        """One seed of a recipe; `name` (default: the recipe's directory) names its arm,
        which must be unique because runs and configs are filed under it."""
        recipe = ROOT / recipe
        arm = f"{name or recipe.parent.name}_seed{seed}"
        config = self.dir / "configs" / arm / "train.yaml"
        text = seed_config(recipe, seed)
        if config.exists() and config.read_text() != text:
            raise ValueError(f"{config} differs from its recipe; refusing to mix configurations")
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(text)
        return {"arm": arm, "seed": seed, "recipe": str(recipe.relative_to(ROOT)),
                "config": str(config), "record": str(self.dir / "runs" / f"{arm}.json"), **extra}

    def record(self, job: dict) -> dict:
        path = Path(job["record"])
        return json.loads(path.read_text()) if path.exists() else {}

    def run_dir(self, job: dict) -> Path:
        registry = yaml.safe_load((self.dir / "registries" / f"{job['arm']}.yaml").read_text())
        if set(registry) != {job["arm"]}:
            raise ValueError(f"registry for {job['arm']} names {sorted(registry)}")
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        return ROOT / cfg["training"]["out_dir"] / job["arm"] / registry[job["arm"]]["run"]

    def _run(self, job: dict, command: list) -> None:
        env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        with open(self.dir / "logs" / f"{job['arm']}.log", "a") as log:
            subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def execute(self, job: dict, evaluate: bool) -> None:
        """Train (unless a finished run exists) and, if asked, score native test metrics."""
        if self.record(job).get("evaluated", False) >= evaluate and self.record(job):
            return
        registry = self.dir / "registries" / f"{job['arm']}.yaml"
        if not registry.exists():
            self._run(job, [PYTHON, "-u", str(CODE / "scripts/train.py"), "--config", job["config"],
                            "--best-models", str(registry)])
        run = self.run_dir(job)
        if json.loads((run / "run.json").read_text())["status"] != "FINISHED":
            raise RuntimeError(f"training did not finish: {run}")
        if evaluate:
            gpus = "1" if os.environ.get("CUDA_VISIBLE_DEVICES") else "0"
            self._run(job, [PYTHON, "-u", str(CODE / "scripts/evaluate_run.py"), "--run-dir", str(run),
                            "--gpus", gpus])
        Path(job["record"]).write_text(json.dumps(
            {**job, "run_dir": str(run), "config_sha256": digest(job["config"]), "evaluated": evaluate},
            indent=2) + "\n")

    # -- Slurm --------------------------------------------------------------

    def _script(self, name: str, command: str, gpu: bool, array: int | None = None) -> Path:
        res = GPU_JOB if gpu else CPU_JOB
        lines = ["#!/bin/bash", f"#SBATCH --job-name=final_{name}", "#SBATCH --partition=gpu",
                 "#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={res['cpus']}",
                 f"#SBATCH --mem={res['mem']}", f"#SBATCH --time={res['time']}"]
        if gpu:
            lines.append("#SBATCH --gres=gpu:1")
        if array is not None:
            lines.append(f"#SBATCH --array=0-{array - 1}")
            out = self.dir / "slurm" / f"{name}_%A_%a"
        else:
            out = self.dir / "slurm" / f"{name}_%j"
        lines += [f"#SBATCH --output={out}.out", f"#SBATCH --error={out}.err", "set -euo pipefail",
                  f"cd {shlex.quote(str(ROOT))}", command]
        path = self.dir / "slurm" / f"{name}.sbatch"
        path.write_text("\n".join(lines) + "\n")
        return path

    def _submit(self, script: Path, after: list) -> str:
        args = ["sbatch", "--parsable"]
        if after:
            args.append("--dependency=afterok:" + ":".join(after))
        job_id = subprocess.check_output([*args, str(script)], text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch output {job_id!r}")
        ledger = self.dir / "slurm" / "submitted.json"
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        entries.append({"script": script.name, "job_id": job_id, "after": after})
        ledger.write_text(json.dumps(entries, indent=2) + "\n")
        return job_id

    def _command(self, *args, array_index: bool = False) -> str:
        command = " ".join(shlex.quote(str(a)) for a in
                           (PYTHON, "-u", Path(__file__).resolve(), "--spec", self.spec_path, *args))
        return command + ' "$SLURM_ARRAY_TASK_ID"' if array_index else command

    def use_fixed_teacher(self) -> dict:
        """Record an already-trained teacher run (spec teacher.fixed_run) without training."""
        run = ROOT / self.spec["teacher"]["fixed_run"]
        predictions = run / "predictions"
        for split in ("train", "val", "test"):
            if not (predictions / f"{split}.parquet").is_file():
                raise ValueError(f"teacher {run} lacks {split} predictions")
        logged = {"pctr_model": self.spec["teacher"]["pctr_model"],
                  "logged_pctr": {"dcn_run": str(run.relative_to(ROOT)),
                                  "predictions": {"test": str((predictions / "test.parquet").relative_to(ROOT))},
                                  "score_column": "score"}}
        (self.dir / "teacher_eval.yaml").write_text(yaml.safe_dump(logged, sort_keys=False))
        record = {"selected": {"run_dir": str(run)}, "rule": "fixed teacher run named in the spec",
                  "test_predictions_sha256": digest(predictions / "test.parquet")}
        (self.dir / "teacher.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    def submit_teacher(self) -> dict:
        if "fixed_run" in self.spec["teacher"]:
            return {"fixed_teacher": self.use_fixed_teacher()["selected"]["run_dir"]}
        seeds = self.spec["teacher"].get("seeds", self.seeds)
        jobs = [self.job(recipe, seed, evaluate=False)
                for recipe in self.spec["teacher"]["candidates"] for seed in seeds]
        jobs_file = self.dir / "slurm" / "teacher_jobs.json"
        jobs_file.write_text(json.dumps(jobs, indent=2) + "\n")
        array = self._submit(self._script("teacher", self._command("task", jobs_file, array_index=True),
                                          gpu=True, array=len(jobs)), [])
        select = self._submit(self._script("select_teacher", self._command("select-teacher"), gpu=True), [array])
        return {"teacher_array": array, "select_teacher": select}

    def submit_students(self, variant_ids: list) -> dict:
        variants = {v["id"]: v for v in self.spec["variants"]}
        unknown = set(variant_ids) - set(variants)
        if unknown:
            raise SystemExit(f"unknown variants {sorted(unknown)}; the spec lists {sorted(variants)}")
        jobs = [self.job(variants[v]["recipe"], seed, name=v, variant=v, evaluate=True)
                for v in variant_ids for seed in self.seeds]
        name = "students_" + "_".join(variant_ids)
        jobs_file = self.dir / "slurm" / f"{name}_jobs.json"
        jobs_file.write_text(json.dumps(jobs, indent=2) + "\n")
        array = self._submit(self._script(name, self._command("task", jobs_file, array_index=True),
                                          gpu=True, array=len(jobs)), [])
        after = [array]
        if not (self.dir / "teacher.json").exists():
            selections = [e["job_id"] for e in json.loads((self.dir / "slurm" / "submitted.json").read_text())
                          if e["script"] == "select_teacher.sbatch"]
            if not selections:
                raise SystemExit("submit the teacher first")
            after.append(selections[-1])
        logged = self._submit(self._script(f"logged_{name}", self._command("logged", jobs_file), gpu=False), after)
        return {"student_array": array, "logged": logged}

    def submit_rescore(self, variant_ids: list) -> dict:
        """Queue a CPU job scoring finished runs' serving product with the study's teacher."""
        if not (self.dir / "teacher.json").exists():
            raise SystemExit("record the teacher first")
        variants = {v["id"]: v for v in self.spec["variants"]}
        jobs = [self.job(variants[v]["recipe"], seed, name=v, variant=v, evaluate=True)
                for v in variant_ids for seed in self.seeds]
        missing = [j["arm"] for j in jobs if not self.record(j).get("evaluated")]
        if missing:
            raise SystemExit(f"runs not finished: {missing}")
        name = "rescore_" + "_".join(variant_ids)
        jobs_file = self.dir / "slurm" / f"{name}_jobs.json"
        jobs_file.write_text(json.dumps(jobs, indent=2) + "\n")
        return {"rescore": self._submit(self._script(name, self._command("logged", jobs_file), gpu=False), [])}

    # -- what the Slurm jobs run --------------------------------------------

    def task(self, jobs_file: Path, index: int) -> None:
        job = json.loads(jobs_file.read_text())[index]
        self.execute(job, job["evaluate"])

    def select_teacher(self) -> dict:
        jobs = json.loads((self.dir / "slurm" / "teacher_jobs.json").read_text())
        candidates = []
        for job in jobs:
            record = self.record(job)
            if not record:
                raise SystemExit(f"teacher candidate {job['arm']} has not finished")
            run = Path(record["run_dir"])
            selected = json.loads((run / "run.json").read_text())["selected_on"]
            if selected["metric"] != "ctr_ne" or selected["split"] != "val":
                raise ValueError(f"{run}: teacher must be selected on validation CTR NE")
            candidates.append({**job, "run_dir": str(run), "val_ctr_ne": selected["value"],
                               "epoch": selected["epoch"]})
        best = min(candidates, key=lambda c: (c["val_ctr_ne"], c["arm"]))
        # Candidates are compared on validation only; only the selected one meets test.
        self.execute({k: best[k] for k in jobs[0]}, evaluate=True)
        run = Path(best["run_dir"])
        predictions = run / "predictions"
        for split in ("train", "val", "test"):
            if not (predictions / f"{split}.parquet").is_file():
                raise ValueError(f"teacher {run} lacks {split} predictions")
        logged = {"pctr_model": self.spec["teacher"]["pctr_model"],
                  "logged_pctr": {"dcn_run": str(run.relative_to(ROOT)),
                                  "predictions": {"test": str((predictions / "test.parquet").relative_to(ROOT))},
                                  "score_column": "score"}}
        (self.dir / "teacher_eval.yaml").write_text(yaml.safe_dump(logged, sort_keys=False))
        record = {"selected": best, "candidates": sorted(candidates, key=lambda c: c["val_ctr_ne"]),
                  "rule": "lowest validation CTR NE over all candidate recipes and seeds",
                  "test_predictions_sha256": digest(predictions / "test.parquet")}
        (self.dir / "teacher.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    def logged(self, jobs_file: Path) -> None:
        """Score each run's serving product with the study's teacher, in parallel on CPU.

        Scores are filed per teacher name (metrics_logged/<pctr_model>), so a run
        already scored with another teacher is scored again, alongside it.
        """
        config = self.dir / "teacher_eval.yaml"
        name = self.spec["teacher"]["pctr_model"]
        jobs = json.loads(jobs_file.read_text())

        def score(job):
            record = self.record(job)
            if (Path(record["run_dir"]) / "metrics_logged" / name / "test.json").exists():
                return
            self._run(job, [PYTHON, "-u", str(CODE / "scripts/evaluate_run.py"), "--run-dir",
                            record["run_dir"], "--gpus", "0", "--logged-config", str(config)])
            scored = sorted(set(record.get("logged_with", [])) | {name})
            Path(job["record"]).write_text(json.dumps(
                {**record, "logged": True, "logged_with": scored}, indent=2) + "\n")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(score, jobs))

    # -- report -------------------------------------------------------------

    def report(self) -> dict:
        teacher = json.loads((self.dir / "teacher.json").read_text())
        teacher_run = Path(teacher["selected"]["run_dir"])
        pctr_model = self.spec["teacher"]["pctr_model"]
        out = {"teacher": {"run_dir": str(teacher_run),
                           "val": json.loads((teacher_run / "metrics/val.json").read_text())[0],
                           "test": json.loads((teacher_run / "metrics/test.json").read_text())[0]},
               "variants": {}}
        for variant in self.spec["variants"]:
            per_seed = []
            for seed in self.seeds:
                arm = f"{variant['id']}_seed{seed}"
                record = json.loads((self.dir / "runs" / f"{arm}.json").read_text()) \
                    if (self.dir / "runs" / f"{arm}.json").exists() else {}
                if not record or not (Path(record["run_dir"]) / "metrics_logged" / pctr_model
                                      / "test.json").exists():
                    break
                run = Path(record["run_dir"])
                test = {r["population"]: r for r in json.loads((run / "metrics/test.json").read_text())
                        if r["population"] != "CTCVR-logged"}
                val = {r["population"]: r for r in json.loads((run / "metrics/val.json").read_text())}
                provenance = json.loads((run / "metrics_logged" / pctr_model / "provenance.json").read_text())
                if provenance["pctr_scores_sha256"] != teacher["test_predictions_sha256"]:
                    raise ValueError(f"{run} was scored with a different teacher")
                test["CTCVR-logged"] = json.loads((run / "metrics_logged" / pctr_model / "test.json").read_text())[0]
                per_seed.append({"seed": seed, "run_dir": str(run), "test": test, "val": val,
                                 "selected_on": json.loads((run / "run.json").read_text())["selected_on"]})
            if len(per_seed) != len(self.seeds):
                print(f"{variant['id']}: {len(per_seed)} of {len(self.seeds)} seeds finished; not reported")
                continue
            stats = {}
            for split in ("val", "test"):
                for population in POPULATIONS:
                    if population not in per_seed[0][split]:
                        continue
                    for metric in METRICS:
                        values = [s[split][population][metric] for s in per_seed]
                        stats[f"{split}.{population}.{metric}"] = {
                            "mean": statistics.mean(values),
                            "sem": statistics.stdev(values) / len(values) ** 0.5, "values": values}
            out["variants"][variant["id"]] = {"label": variant["label"], "regime": variant["regime"],
                                              "seeds": per_seed, "stats": stats}
        (self.dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
        return out

    def tables(self) -> str:
        """main.tex Tables 1 and 2: test means, four significant figures.

        Rows follow the manuscript: frozen p, then per variant pCVR v and serving
        p*v, with auxiliary q and native q*v kept as commented lines.
        """
        summary = json.loads((self.dir / "summary.json").read_text())
        fmt = lambda x: f"{x:.4g}"
        teacher = summary["teacher"]["test"]
        out = []
        for regime in ("full", "downsampled"):
            q = r"q_i^{\mathrm{full}}" if regime == "downsampled" else "q_i"
            lines = ["DCN-V2 & $p_i$ & " + " & ".join(fmt(teacher[m]) for m in METRICS) + r" \\", r"\hline"]
            for variant in self.spec["variants"]:
                if variant["regime"] != regime or variant["id"] not in summary["variants"]:
                    continue
                stats = summary["variants"][variant["id"]]["stats"]
                cell = lambda population: " & ".join(fmt(stats[f"test.{population}.{m}"]["mean"]) for m in METRICS)
                label = variant["label"]
                has_q = f"test.CTR.auc" in stats
                if has_q:
                    lines.append(f"% \\multirow{{2}}{{*}}{{{label}}} & ${q}$ & {cell('CTR')} \\\\")
                lines.append(f"\\multirow{{2}}{{*}}{{{label}}} & $v_i$ & {cell('pCVR')} \\\\")
                if has_q:
                    lines.append(f"%  & ${q} v_i$ & {cell('CTCVR')} \\\\")
                lines.append(f" & $p_i v_i$ & {cell('CTCVR-logged')} \\\\")
                lines.append(r"\hline")
            out.append(f"% {regime}\n" + "\n".join(lines))
        text = "\n\n".join(out) + "\n"
        (self.dir / "tables.tex").write_text(text)
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("what", choices=["teacher", "students", "rescore"])
    submit.add_argument("variants", nargs="*")
    task = sub.add_parser("task")
    task.add_argument("jobs_file", type=Path)
    task.add_argument("index", type=int)
    sub.add_parser("select-teacher")
    logged = sub.add_parser("logged")
    logged.add_argument("jobs_file", type=Path)
    sub.add_parser("report")
    sub.add_parser("tables")
    args = parser.parse_args()
    study = Study(args.spec.resolve())
    if args.command == "submit":
        result = (study.submit_teacher() if args.what == "teacher" else
                  study.submit_rescore(args.variants) if args.what == "rescore" else
                  study.submit_students(args.variants))
        print(json.dumps(result))
    elif args.command == "task":
        study.task(args.jobs_file, args.index)
    elif args.command == "select-teacher":
        print(json.dumps(study.select_teacher()["selected"], indent=2))
    elif args.command == "logged":
        study.logged(args.jobs_file)
    elif args.command == "tables":
        print(study.tables())
    else:
        summary = study.report()
        print(json.dumps({v: {k: s["mean"] for k, s in d["stats"].items() if k.startswith("test.")}
                          for v, d in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
