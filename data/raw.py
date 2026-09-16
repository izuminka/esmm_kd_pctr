"""Streaming reader/parser for the official Ali-CCP release.

File format (confirmed against the actual `ali-ccp/*.tar.gz` files):
  sample_skeleton_{train,test}.csv:
    sample_id,y,z,common_feature_index,num_features,feature_list
  common_features_{train,test}.csv:
    common_feature_index,num_features,feature_list

`feature_list` is a "\x01"-separated list of "field_id\x02feat_id\x03value"
entries. `field_id` is a string, not always numeric (e.g. "109_14").

Preferred path: extract tarballs once with extract_tar(), then call
iter_impressions_from_csv() which skips gzip overhead and loads
common_features in parallel across multiple workers.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import tarfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

FeatureBag = dict[str, list[tuple[int, float]]]


# ---------------------------------------------------------------------------
# Validation split -- fixed by hashing the example id, not by row position
# ---------------------------------------------------------------------------

_U64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
# Truncation of the binary64 value 0.1 * 2**64; floor(2**64 / 10) would differ.
_VAL_THRESHOLD = 1_844_674_407_370_955_264


def splitmix64(x: int) -> int:
    """SplitMix64 next(): advance the state by the golden gamma, then finalize.

    Verified against the paper's published census — this variant reproduces its
    train/validation row, click and conversion counts exactly, while applying
    the finalizer alone does not.
    """
    z = (x + _GOLDEN_GAMMA) & _U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return z ^ (z >> 31)


def is_validation(sample_id: int) -> bool:
    """Validation membership, fixed per row by example id rather than position.

    Ali-CCP ships no validation split, so one is carved out of the official
    train population.  Hashing the id keeps membership identical across runs,
    orderings and re-materialisations without shipping an index.
    """
    return splitmix64((sample_id & _U64) ^ 42) < _VAL_THRESHOLD


def _parse_feature_list(feature_list: str) -> FeatureBag:
    fields: FeatureBag = {}
    for entry in feature_list.split("\x01"):
        field_id, rest = entry.split("\x02")
        feat_id, value = rest.split("\x03")
        fields.setdefault(field_id, []).append((int(feat_id), float(value)))
    return fields


@dataclass
class Impression:
    sample_id: int
    y: int
    z: int
    common_feature_index: str
    fields: FeatureBag = field(default_factory=dict)


def parse_skeleton_line(line: str) -> Impression | None:
    """Parse one sample_skeleton line. Returns None if z > y (invalid row)."""
    sample_id, y, z, common_feature_index, _, feature_list = line.rstrip("\n").split(
        ",", maxsplit=5
    )
    y, z = int(y), int(z)
    if z > y:
        return None
    return Impression(int(sample_id), y, z, common_feature_index, _parse_feature_list(feature_list))


def parse_common_features_line(line: str) -> tuple[str, FeatureBag]:
    """Parse one common_features line into (common_feature_index, fields)."""
    common_feature_index, _, feature_list = line.rstrip("\n").split(",", maxsplit=2)
    return common_feature_index, _parse_feature_list(feature_list)


# ---------------------------------------------------------------------------
# Tar-based loading (original path, kept for compatibility / tests)
# ---------------------------------------------------------------------------

def iter_impressions(
    tar_path: str,
    split: str,
    limit: int | None = None,
    common_features_limit: int | None = None,
):
    """Yield Impressions from a tar.gz archive (sequential, gzip-decompressed).

    Prefer iter_impressions_from_csv() when the tarball has been extracted.
    """
    logger.info("Opening tarball %s (split=%s)", tar_path, split)
    with tarfile.open(tar_path, "r:gz") as tar:
        common_features: dict[str, FeatureBag] = {}
        logger.info("Loading common_features_%s.csv from tar...", split)
        with tar.extractfile(f"common_features_{split}.csv") as f:
            for i, raw_line in enumerate(f):
                if common_features_limit is not None and i >= common_features_limit:
                    break
                idx, fields = parse_common_features_line(raw_line.decode("utf-8"))
                common_features[idx] = fields
        logger.info("Loaded %d common feature entries", len(common_features))

        logger.info("Streaming sample_skeleton_%s.csv from tar...", split)
        with tar.extractfile(f"sample_skeleton_{split}.csv") as f:
            yielded = 0
            for raw_line in f:
                if limit is not None and yielded >= limit:
                    return
                impression = parse_skeleton_line(raw_line.decode("utf-8"))
                if impression is None:
                    continue
                common = common_features.get(impression.common_feature_index)
                if common is None:
                    continue
                impression.fields.update(common)
                yield impression
                yielded += 1
        logger.info("Yielded %d impressions", yielded)


# ---------------------------------------------------------------------------
# CSV-based loading (fast path after extraction)
# ---------------------------------------------------------------------------

def extract_tar(tar_path: str, out_dir: str) -> None:
    """Extract tar.gz to out_dir using the system tar command (fast).

    No-op if the expected CSV files already exist in out_dir.
    """
    split = "train" if "train" in os.path.basename(tar_path) else "test"
    sentinel = os.path.join(out_dir, f"common_features_{split}.csv")
    if os.path.exists(sentinel):
        logger.info("Already extracted: %s", sentinel)
        return
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Extracting %s -> %s", tar_path, out_dir)
    subprocess.run(["tar", "-xzf", tar_path, "-C", out_dir], check=True)
    logger.info("Extraction complete")


def seek_to_line_start(f, offset: int) -> None:
    """Position binary file `f` at the first line beginning at or after `offset`.

    Byte-range chunking splits mid-line, so a chunk must skip the partial line it
    lands in.  Skipping unconditionally drops a line whenever `offset` happens to
    fall exactly on a line start: the previous chunk stops before it (its end is
    not < end) and this one discards it, so no chunk reads it.  Checking the
    preceding byte distinguishes the two cases.
    """
    if offset <= 0:
        f.seek(0)
        return
    f.seek(offset - 1)
    if f.read(1) != b"\n":
        f.readline()  # we landed mid-line; drop the remainder


def _parse_common_features_chunk(path: str, start: int, end: int) -> dict:
    """Parse lines in byte range [start, end) from a plain common_features CSV.

    Opens in binary mode so that seek()/tell() operate on true byte offsets.
    Skips the partial first line if start > 0 (we may have split mid-line).
    Reads past `end` to complete the line that straddles the boundary, so
    each line is parsed by exactly one worker.
    """
    result = {}
    with open(path, "rb") as f:
        seek_to_line_start(f, start)
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8")
            idx, fields = parse_common_features_line(line)
            result[idx] = fields
    return result


def _load_common_features_parallel(path: str, n_workers: int) -> dict[str, FeatureBag]:
    """Load a common_features CSV into a dict using n_workers parallel processes.

    Uses imap_unordered so each chunk dict is merged and discarded as it
    arrives, keeping peak memory at ~2x the final dict size rather than
    (n_workers + 1)x.
    """
    size = os.path.getsize(path)
    chunk_size = size // n_workers
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size if i < n_workers - 1 else size)
        for i in range(n_workers)
    ]
    logger.info(
        "Loading %s (%.1f GB) with %d workers, ~%.0f MB each",
        os.path.basename(path), size / 1e9, n_workers, chunk_size / 1e6,
    )

    merged: dict[str, FeatureBag] = {}
    args = [(path, s, e) for s, e in ranges]
    with mp.Pool(n_workers) as pool:
        for i, chunk in enumerate(pool.starmap(_parse_common_features_chunk, args)):
            merged.update(chunk)
            logger.debug("Merged chunk %d/%d (%d entries so far)", i + 1, n_workers, len(merged))

    logger.info("Loaded %d common feature entries", len(merged))
    return merged


def iter_impressions_from_csv(
    csv_dir: str,
    split: str,
    limit: int | None = None,
    n_workers: int = 8,
):
    """Yield Impressions from pre-extracted plain CSV files.

    common_features is loaded in parallel across n_workers processes.
    Much faster than iter_impressions() — no gzip decompression, parallel parse.
    """
    cf_path = os.path.join(csv_dir, f"common_features_{split}.csv")
    sk_path = os.path.join(csv_dir, f"sample_skeleton_{split}.csv")

    common_features = _load_common_features_parallel(cf_path, n_workers)

    logger.info("Streaming %s", os.path.basename(sk_path))
    yielded = 0
    with open(sk_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and yielded >= limit:
                return
            impression = parse_skeleton_line(line)
            if impression is None:
                continue
            common = common_features.get(impression.common_feature_index)
            if common is None:
                continue
            impression.fields.update(common)
            yield impression
            yielded += 1
    logger.info("Yielded %d impressions from %s split", yielded, split)
