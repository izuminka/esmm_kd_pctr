"""Frozen logged pCTR scores from a finished DCN run, aligned to an encoded split.

A config names the source exactly, with no fallback to `latest` or the
best-model record:

    logged_pctr:
      dcn_run: artifacts/dcn/<run id>
      predictions: {train: <file>, val: <file>, test: <file>}   # each inside dcn_run
      score_column: score

The scores enter only the training loss; on validation and test they are used
only to evaluate CTCVR on the logged product.

Scores are joined to a split by sample_id, never by row position, so a
predictions file written in a different order still lines up. The join runs on
the split's device and fails unless every split row gets exactly one score and
the click labels agree. Each load returns a provenance record naming the run and
file it read, which callers write next to anything scored with it.
"""

import logging
import os
import time
import warnings
from datetime import datetime, timezone

import pyarrow.parquet as pq
import torch

from training.runs import FINISHED, read_run, utc_iso

logger = logging.getLogger(__name__)


def _source(block: dict, root_dir: str, split: str) -> tuple[str, str, dict]:
    """(predictions file as the config names it, its path, the DCN run.json) for one split."""
    dcn_run = block.get("dcn_run")
    if not dcn_run or os.path.basename(os.path.normpath(dcn_run)) == "latest":
        raise ValueError(f"logged_pctr.dcn_run must name a concrete DCN run folder, got {dcn_run!r}")
    run_dir = os.path.realpath(os.path.join(root_dir, dcn_run))
    if not os.path.exists(os.path.join(run_dir, "run.json")):
        raise ValueError(f"logged_pctr.dcn_run {dcn_run!r} is not a run folder (no run.json)")
    run = read_run(run_dir)
    if run.get("model") != "dcn" or run.get("status") != FINISHED:
        raise ValueError(f"logged_pctr.dcn_run {dcn_run!r} must be a finished dcn run, "
                         f"got model={run.get('model')!r} status={run.get('status')!r}")

    file = (block.get("predictions") or {}).get(split)
    if not file:
        raise ValueError(f"logged_pctr.predictions has no file for split {split!r}")
    path = os.path.realpath(os.path.join(root_dir, file))
    if os.path.commonpath([path, run_dir]) != run_dir:
        raise ValueError(f"logged_pctr.predictions.{split} {file!r} is outside dcn_run {dcn_run!r}")
    if not os.path.exists(path):
        raise ValueError(f"logged_pctr.predictions.{split} {file!r} does not exist")
    return file, path, run


def load_logged_pctr(block: dict, root_dir: str, split: str,
                     sample_ids: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Logged pCTR for each row of a split, on `sample_ids.device`, plus its provenance.

    `sample_ids` and `ys` are the split's own rows, in the order the scores are
    returned. Paths in `block` are relative to `root_dir` (or absolute).
    """
    file, path, run = _source(block, root_dir, split)
    column = block.get("score_column")
    if not column:
        raise ValueError("logged_pctr.score_column is required")
    if column not in pq.read_schema(path).names:
        raise ValueError(f"{path} has no score column {column!r}")

    device = sample_ids.device
    t0 = time.perf_counter()
    table = pq.read_table(path, columns=["sample_id", "y", column], memory_map=True)
    with warnings.catch_warnings():
        # Arrow hands back read-only buffers. They are only ever read: indexing
        # below copies the scores out before anyone can write to them.
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        ids, labels, scores = (torch.from_numpy(table.column(c).to_numpy()).to(device)
                               for c in ("sample_id", "y", column))
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = len(sample_ids)
    if n == 0 or len(ids) != n:
        raise ValueError(f"{path} has {len(ids):,} rows; split {split!r} has {n:,}")
    order = torch.argsort(ids)
    sorted_ids = ids[order]
    if not bool((sorted_ids[1:] > sorted_ids[:-1]).all()):
        raise ValueError(f"{path} repeats a sample_id")
    # Equal counts and unique ids: every split id found means a one-to-one match.
    idx = order[torch.searchsorted(sorted_ids, sample_ids).clamp_(max=n - 1)]
    if not torch.equal(ids[idx], sample_ids):
        raise ValueError(f"{path} is missing sample_ids of split {split!r}")
    if not torch.equal(labels[idx], ys.to(labels.dtype)):
        raise ValueError(f"{path} click labels disagree with split {split!r}")
    p = scores[idx].float()
    t_join = time.perf_counter() - t0

    stat = os.stat(path)
    provenance = {
        "dcn_run": block["dcn_run"],
        "run_id": run["run"],
        "commit": run.get("git", {}).get("commit"),
        "selected_on": run.get("selected_on"),
        "score_column": column,
        "files": {split: {
            "file": file,
            "rows": n,
            "bytes": stat.st_size,
            "mtime": utc_iso(datetime.fromtimestamp(stat.st_mtime, timezone.utc)),
        }},
    }
    logger.info("logged pCTR %s: %d rows, %s[%s] from dcn run %s (read %.2fs, join %.2fs)",
                split, n, file, column, run["run"], t_read, t_join)
    return p, provenance


def attach_logged_pctr(block: dict, root_dir: str, splits: dict) -> dict:
    """Load scores onto each loaded split's `logged_pctr`; returns one merged provenance record.

    `splits` maps split name -> SplitData, or None for a split left unloaded.
    """
    record = None
    for name, split in splits.items():
        if split is None:
            continue
        split.logged_pctr, provenance = load_logged_pctr(block, root_dir, name,
                                                         split.sample_ids, split.ys)
        if record is None:
            record = provenance
        else:
            record["files"].update(provenance["files"])
    return record
