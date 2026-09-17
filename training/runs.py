"""Run directories, each model's `latest` pointer, and the best-model record.

Layout under the artifacts root (cfg["training"]["out_dir"]), where <name> is
the config file's name (configs/<name>.yaml):

    <name>/latest.json        the most recent run that finished
    <name>/<run id>/          one folder per run, never rewritten after it ends
        config.yaml           verbatim copy of the config file the run used
        run.json              status, timing, code version, environment, data census,
                              the validation selection and test metrics
        git.diff              uncommitted changes, present only if the tree was dirty
        history.csv           one row per epoch
        best.pt               the checkpoint kept by validation
        predictions/<split>.parquet   with training.save_predictions
        metrics/<split>.json          without it: the split scored in memory
        metrics.json

A run id is the UTC start time in ISO 8601 basic format, so ids sort in start
order and mean the same on every machine.

`code/best_models.yaml` lives in git and names the best finished run per model,
ranked on the validation checkpoint metric. Keeping it in git makes each change
of best model a reviewable diff. Test metrics are recorded there but never
ranked on: selecting a model by test score leaks the test set into selection.

A run that receives SIGTERM (SLURM cancellation or time limit) is recorded as
FAILED. Only one killed outright by SIGKILL -- e.g. still running when SLURM's
KillWait expires -- cannot record its end and stays RUNNING. Neither becomes
latest or best.

Pointers hold run ids, not absolute paths, so the tree can be moved or copied.

    python -m training.runs latest dcn     # prints the latest dcn run folder
"""

import argparse
import csv
import fcntl
import itertools
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone

import yaml

logger = logging.getLogger(__name__)

RUNNING, FINISHED, FAILED = "RUNNING", "FINISHED", "FAILED"

_RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BEST_HEADER = """\
# Best finished run per model, ranked on the validation checkpoint metric.
# scripts/train.py rewrites an entry when a run beats it; review the diff and commit.
# Checkpoint paths are relative to the artifacts root. Test metrics are recorded
# for reference and never used for ranking.
"""


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_atomic(path: str, text: str) -> None:
    """Replace `path` in one step, so no reader ever sees a partial file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _write_atomic(path, json.dumps(obj, indent=2) + "\n")


def new_run_dir(artifacts_root: str, model: str, started_at: datetime) -> str:
    """Create an empty run folder named by UTC start time.

    Runs started in the same second (e.g. a SLURM array) get -2, -3, ... suffixes;
    mkdir is the claim, so two processes can never share a folder.
    """
    base = os.path.join(artifacts_root, model)
    os.makedirs(base, exist_ok=True)
    stamp = started_at.astimezone(timezone.utc).strftime(_RUN_ID_FORMAT)
    for n in itertools.count(1):
        path = os.path.join(base, stamp if n == 1 else f"{stamp}-{n}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue


def git_state(code_dir: str) -> tuple[dict, str]:
    """Commit, dirtiness and untracked files of `code_dir`, plus its uncommitted diff."""
    def git(*args):
        return subprocess.run(["git", "-C", code_dir, *args],
                              capture_output=True, text=True, check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
        untracked = git("ls-files", "--others", "--exclude-standard").split()
        diff = git("diff", "HEAD") if dirty else ""
    except (OSError, subprocess.CalledProcessError):  # not a checkout, or no git binary
        return {"commit": None, "dirty": None, "untracked": []}, ""
    return {"commit": commit, "dirty": dirty, "untracked": untracked}, diff


def environment() -> dict:
    import torch  # deferred: the command-line lookup below shouldn't pay torch's import time

    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def data_census(encoded_dir: str) -> dict:
    """Row, click and conversion counts per split, as preprocess.py recorded them."""
    census = {}
    for split in ("train", "val", "test"):
        path = os.path.join(encoded_dir, split, "split_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            census[split] = {k: v for k, v in meta.items() if k != "split"}
    return census


def read_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)


def run_name(config_path: str) -> str:
    """A run's name is its config file's name: configs/<name>.yaml runs under <name>/.

    Several configs can share one model class (say, loss variants of ESMM); naming
    runs by config keeps their folders, latest pointers and best entries apart.
    """
    return os.path.splitext(os.path.basename(config_path))[0]


def start_run(artifacts_root: str, cfg: dict, config_path: str, code_dir: str,
              encoded_dir: str) -> str:
    """Create the run folder and record everything known before training starts.

    The config is copied byte for byte rather than re-serialised from `cfg`, which
    would drop the comments explaining where each value comes from. The two agree
    because nothing overrides the file at runtime.
    """
    started_at = datetime.now(timezone.utc)
    name = run_name(config_path)
    run_dir = new_run_dir(artifacts_root, name, started_at)
    git, diff = git_state(code_dir)
    if diff:
        _write_atomic(os.path.join(run_dir, "git.diff"), diff)
    shutil.copyfile(config_path, os.path.join(run_dir, "config.yaml"))
    _write_json(os.path.join(run_dir, "run.json"), {
        "run": os.path.basename(run_dir),
        "name": name,
        "model": cfg["model"],
        "status": RUNNING,
        "started_at": utc_iso(started_at),
        "finished_at": None,
        "git": git,
        "environment": environment(),
        "data": data_census(encoded_dir),
    })
    return run_dir


def update_run(run_dir: str, **fields) -> dict:
    run = read_run(run_dir)
    run.update(fields)
    _write_json(os.path.join(run_dir, "run.json"), run)
    return run


def record_logged_pctr(run_dir: str, provenance: dict) -> dict:
    """Add the logged pCTR source (see data.logged_pctr) to run.json, merging per-split files.

    Written as soon as the scores load, so even a run that fails later states
    which scores it used.
    """
    current = read_run(run_dir).get("logged_pctr")
    if current is not None:
        provenance = {**current, "files": {**current["files"], **provenance["files"]}}
    return update_run(run_dir, logged_pctr=provenance)


def finish_run(run_dir: str, status: str, **fields) -> dict:
    return update_run(run_dir, **fields, status=status,
                      finished_at=utc_iso(datetime.now(timezone.utc)))


def selection(run_dir: str, metric: str) -> dict:
    """The epoch validation kept, read back from history.csv.

    The last saved row identifies the exported weights in both best and last modes.
    """
    with open(os.path.join(run_dir, "history.csv"), newline="") as f:
        saved = [row for row in csv.DictReader(f) if row["saved"] == "True"]
    best = saved[-1]
    return {"metric": metric, "split": "val",
            "value": float(best["val_score"]), "epoch": int(best["epoch"])}


def read_test_metrics(run_dir: str) -> dict:
    """Test-split rows of metrics.json, keyed by evaluation population."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        rows = json.load(f)
    return {r["population"]: {k: r[k] for k in ("auc", "pr_auc", "ne", "ce")}
            for r in rows if r["split"] == "test"}


def update_latest(artifacts_root: str, model: str, run_dir: str) -> None:
    run = read_run(run_dir)
    if run["status"] != FINISHED:
        raise ValueError(f"{run['run']} is {run['status']}; only finished runs can be latest")
    _write_json(os.path.join(artifacts_root, model, "latest.json"),
                {"run": run["run"], "finished_at": run["finished_at"]})


def resolve_run(artifacts_root: str, model: str, ref: str = "latest") -> str:
    """Run folder for a run id, or for `latest`."""
    base = os.path.join(artifacts_root, model)
    if ref == "latest":
        with open(os.path.join(base, "latest.json")) as f:
            ref = json.load(f)["run"]
    return os.path.join(base, ref)


def best_entry(run: dict) -> dict:
    entry = {
        "run": run["run"],
        "checkpoint": f"{run['name']}/{run['run']}/best.pt",
        "selected_on": run["selected_on"],
        "test": run.get("test"),
        "trained_from": run["git"],
        "data": run["data"],
        "finished_at": run["finished_at"],
    }
    if "logged_pctr" in run:
        entry["logged_pctr"] = run["logged_pctr"]
    if "train_cohort" in run:
        entry["train_cohort"] = run["train_cohort"]
    return entry


def update_best(best_file: str, model: str, entry: dict) -> bool:
    """Record `entry` as `model`'s best if it beats the current one on validation.

    Only a strictly better value on the same metric replaces an entry (lower
    for pcvr_ne, higher for PR-AUC). When the
    metric names differ the runs aren't comparable, so the entry is kept and a
    warning logged. Returns whether the file changed.

    Runs of different models finish concurrently and all rewrite this one file, so
    the read-compare-write holds an exclusive lock. The lock is on the containing
    directory: the file itself is swapped out by os.replace, so a lock on it would
    be released onto an inode nobody else opens.
    """
    dir_fd = os.open(os.path.dirname(os.path.abspath(best_file)), os.O_RDONLY)
    try:
        fcntl.flock(dir_fd, fcntl.LOCK_EX)
        return _update_best_locked(best_file, model, entry)
    finally:
        os.close(dir_fd)  # closing releases the lock


def _update_best_locked(best_file: str, model: str, entry: dict) -> bool:
    best = {}
    if os.path.exists(best_file):
        with open(best_file) as f:
            best = yaml.safe_load(f) or {}

    current = best.get(model)
    if current is not None:
        have, new = current["selected_on"], entry["selected_on"]
        if have["metric"] != new["metric"]:
            logger.warning("%s: best run %s was selected on %s, run %s on %s; not comparable, "
                           "keeping %s", model, current["run"], have["metric"], entry["run"],
                           new["metric"], current["run"])
            return False
        better = (new["value"] < have["value"] if new["metric"] == "pcvr_ne"
                  else new["value"] > have["value"])
        if not better:
            return False

    best[model] = entry
    _write_atomic(best_file, _BEST_HEADER + yaml.safe_dump(best, sort_keys=False))
    return True


def main():
    ap = argparse.ArgumentParser(description="Print a run folder.")
    ap.add_argument("ref", help="a run id, or 'latest'")
    ap.add_argument("model")
    ap.add_argument("--artifacts", default=os.path.join(_ROOT_DIR, "artifacts"))
    args = ap.parse_args()
    print(resolve_run(args.artifacts, args.model, args.ref))


if __name__ == "__main__":
    main()
