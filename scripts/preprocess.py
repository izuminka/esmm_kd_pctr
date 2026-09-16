"""One-time preprocessing: CSV → encoded numpy arrays.

Reads the extracted CSV files, builds vocab from the train split,
encodes all three splits (train, val, test) using EmbeddingBag's
(ids, weights, offsets) format, and saves everything to disk.

Ali-CCP ships features in two files: per-impression features in the
skeleton, plus user-level bags in common_features keyed by
common_feature_index.  Each bag is shared by ~58 impressions, so the
encoding keeps that indirection instead of materialising the join:
bags are encoded ONCE into a common table and each impression stores a
row pointer into it.  Expanding the join here would turn ~8 GB of real
data into ~180 GB and make it impossible to hold on a GPU.
training/dataset.py stitches the two halves back together per batch.

Speed strategy:
  - Skeleton CSV is split into N byte-range chunks (N = os.cpu_count()).
  - Workers are forked so they inherit the vocab + row-pointer map via
    copy-on-write (zero pickling cost for those dicts).
  - Train and val are encoded in a SINGLE pass over the train skeleton CSV.

Usage:
    python scripts/preprocess.py --config configs/dcn.yaml

Output (under cfg.data.encoded_dir, relative to project root):
    common_train/{field_id}_ids.npy, {field_id}_weights.npy, {field_id}_offsets.npy
    common_test/  ... (same structure)
    train/sample_id.npy, y.npy, z.npy, cf_row.npy
    train/{field_id}_ids.npy, {field_id}_weights.npy, {field_id}_offsets.npy
    val/  ... (same structure; shares common_train)
    test/ ... (same structure; uses common_test)
    vocab.pkl
    field_ids.json
"""

import argparse
import array
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time

import numpy as np
import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import (
    _load_common_features_parallel,
    is_validation,
    parse_common_features_line,
    parse_skeleton_line,
    seek_to_line_start,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Module-level globals — set in main process BEFORE fork so workers inherit
# them via copy-on-write (no IPC / pickling cost for these large dicts)
# ---------------------------------------------------------------------------
_CF_ROW: dict = {}  # {cf_index: dense row index into the common table}
_CF_BAGS: dict = {}  # {cf_index: FeatureBag} — only for the common-table pool
_VOCAB: dict = {}   # {field_id: {token_str: token_id}}
_FIELD_IDS: list = []
_SPLIT_NAMES: tuple = ("train", "val")  # ("test",) when encoding the test source


# ---------------------------------------------------------------------------
# Validation split — fixed by hashing the example id, not by row position
# ---------------------------------------------------------------------------

def _split_of(sample_id: int) -> str:
    if _SPLIT_NAMES == ("test",):
        return "test"
    return "val" if is_validation(sample_id) else "train"


# ---------------------------------------------------------------------------
# Vocab building (streaming, no Impression objects held)
# ---------------------------------------------------------------------------

def discover_field_ids(csv_dir: str, n_bags: int = 2_000, n_rows: int = 50_000) -> list:
    """Field ids present in the release, read from the head of each file.

    Ali-CCP splits its fields across two files, so both are sampled. Reading
    heads keeps this to seconds; loading the full common-features table just to
    learn 23 names costs minutes.
    """
    found: set = set()
    cf_path = os.path.join(csv_dir, "common_features_train.csv")
    sk_path = os.path.join(csv_dir, "sample_skeleton_train.csv")
    with open(cf_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_bags:
                break
            found.update(parse_common_features_line(line)[1])
    with open(sk_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            imp = parse_skeleton_line(line)
            if imp is not None:
                found.update(imp.fields)
    return sorted(found)


def _vocab_chunk_worker(sk_path: str, byte_start: int, byte_end: int) -> dict:
    """Collect distinct training tokens per field from one byte range.

    Reads the _CF_BAGS / _FIELD_IDS globals inherited through fork.
    """
    tokens: dict[str, set] = {f: set() for f in _FIELD_IDS}
    with open(sk_path, "rb") as f:
        seek_to_line_start(f, byte_start)
        while f.tell() < byte_end:
            raw = f.readline()
            if not raw:
                break
            imp = parse_skeleton_line(raw.decode("utf-8"))
            if imp is None or is_validation(imp.sample_id):
                continue
            bag = _CF_BAGS.get(imp.common_feature_index)
            if bag is None:
                continue
            for field_id, entries in imp.fields.items():
                if field_id in tokens:
                    tokens[field_id].update(str(feat_id) for feat_id, _ in entries)
            for field_id, entries in bag.items():
                if field_id in tokens:
                    tokens[field_id].update(str(feat_id) for feat_id, _ in entries)
    return tokens


def build_vocab_streaming(csv_dir: str, field_ids: list, n_workers: int, cf: dict) -> dict:
    """Fit each field's vocabulary on the training rows of the train split.

    Validation rows are excluded: tokens occurring only there must stay unseen
    (id 0, weight masked), exactly as an unseen token would at serving time.

    Parallel over byte ranges, merging each worker's sets as they arrive so
    peak memory stays near the merged vocabulary rather than the sum of all
    workers' partial copies.
    """
    global _CF_BAGS, _FIELD_IDS
    t0 = time.perf_counter()
    sk_path = os.path.join(csv_dir, "sample_skeleton_train.csv")
    logger.info("Building vocab from training rows with %d workers...", n_workers)

    _CF_BAGS, _FIELD_IDS = cf, field_ids  # set BEFORE fork for COW inheritance
    size = os.path.getsize(sk_path)
    cs = size // n_workers
    ranges = [(sk_path, i * cs, (i + 1) * cs if i < n_workers - 1 else size)
              for i in range(n_workers)]

    tokens_by_field: dict[str, set] = {f: set() for f in field_ids}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for i, part in enumerate(pool.starmap(_vocab_chunk_worker, ranges), 1):
            for field_id, toks in part.items():
                tokens_by_field[field_id] |= toks
            logger.debug("  merged vocab chunk %d/%d", i, len(ranges))
    _CF_BAGS = {}

    vocab = {
        field_id: {token: i + 1 for i, token in enumerate(sorted(tokens))}
        for field_id, tokens in tokens_by_field.items()
    }
    logger.info(
        "Vocab built: %d fields, %d total tokens (%.1fs)",
        len(vocab), sum(len(v) for v in vocab.values()), time.perf_counter() - t0,
    )
    return vocab


# ---------------------------------------------------------------------------
# Common-feature table — each unique bag encoded exactly once
# ---------------------------------------------------------------------------

def _encode_common_field(field_id: str, out_dir: str) -> tuple:
    """Encode one field across every common bag.  Reads _CF_BAGS / _VOCAB globals.

    Tokens accumulate in array.array rather than Python lists, which would box
    every id and weight.
    """
    table = _VOCAB[field_id]
    ids = array.array("i")
    wts = array.array("f")
    offs = array.array("q")
    for bag in _CF_BAGS.values():  # insertion order == the row numbering in _CF_ROW
        offs.append(len(ids))
        for feat_id, weight in bag.get(field_id, ()):
            token_id = table.get(str(feat_id), 0)
            ids.append(token_id)
            wts.append(0.0 if token_id == 0 else weight)
    offs.append(len(ids))  # CSR: n+1 offsets, so row i spans offs[i]:offs[i+1]

    fn = field_id.replace("/", "_")
    np.save(os.path.join(out_dir, f"{fn}_ids.npy"),     np.array(ids,  dtype=np.int32))
    np.save(os.path.join(out_dir, f"{fn}_weights.npy"), np.array(wts,  dtype=np.float32))
    np.save(os.path.join(out_dir, f"{fn}_offsets.npy"), np.array(offs, dtype=np.int64))
    return field_id, len(ids)


def _encode_common_table(cf: dict, vocab: dict, field_ids: list, out_dir: str,
                          n_workers: int) -> dict:
    """Encode every unique common-feature bag once into `out_dir`.

    Returns {common_feature_index: dense row index}, which the skeleton
    workers use to write each impression's cf_row pointer.

    Parallel across fields: workers are forked so they inherit the multi-GB
    bag dict via copy-on-write instead of pickling it.
    """
    global _CF_BAGS, _VOCAB
    os.makedirs(out_dir, exist_ok=True)
    cf_row = {idx: row for row, idx in enumerate(cf)}  # dict preserves insertion order
    logger.info(
        "Encoding common table: %d unique bags × %d fields → %s",
        len(cf_row), len(field_ids), os.path.basename(out_dir),
    )

    t0 = time.perf_counter()
    _CF_BAGS, _VOCAB = cf, vocab  # set BEFORE fork for COW inheritance
    ctx = mp.get_context("fork")
    with ctx.Pool(min(n_workers, len(field_ids))) as pool:
        for field_id, n_tok in pool.starmap(
            _encode_common_field, [(f, out_dir) for f in field_ids]
        ):
            logger.debug("  common field %s: %d tokens", field_id, n_tok)
    _CF_BAGS = {}

    logger.info("Common table encoded (%.1fs)", time.perf_counter() - t0)
    return cf_row


def _assert_fields_disjoint(sk_path: str, cf: dict, n_sample: int = 20_000) -> None:
    """Fail if any impression carries a field that its common bag also carries.

    training/dataset.py concatenates the impression's own tokens with its common
    bag's tokens, whereas data/raw.py's `imp.fields.update(common)` lets the
    common bag *replace* a field wholesale.  Those agree only while the two
    halves are disjoint (they are in Ali-CCP: ad-side vs user-side fields).  If
    that ever stops holding, pooling would silently change, so check rather than
    assume.  Sampled from several offsets — cheap next to a full pass.
    """
    size = os.path.getsize(sk_path)
    offsets = (0.0, 0.25, 0.5, 0.75)
    per_offset = max(1, n_sample // len(offsets))
    checked = 0
    with open(sk_path, "rb") as f:
        for frac in offsets:
            seek_to_line_start(f, int(size * frac))
            taken = 0
            while taken < per_offset:
                raw = f.readline()
                if not raw:
                    break
                imp = parse_skeleton_line(raw.decode("utf-8"))
                if imp is None:
                    continue
                bag = cf.get(imp.common_feature_index)
                if bag is None:
                    continue
                taken += 1
                checked += 1
                overlap = set(imp.fields) & set(bag)
                if overlap:
                    raise RuntimeError(
                        f"Fields {sorted(overlap)} appear in both the skeleton row and its "
                        f"common bag (sample_id={imp.sample_id}).  The encoding assumes the "
                        "two halves are disjoint; see training/dataset.py:get_batch."
                    )
    logger.info("Field-disjointness check passed (%d joined rows sampled)", checked)


# ---------------------------------------------------------------------------
# Parallel encoding — workers inherit _CF_ROW, _VOCAB, _FIELD_IDS via fork
# ---------------------------------------------------------------------------

def _encode_chunk_worker(sk_path: str, byte_start: int, byte_end: int,
                          tmp_dir: str, chunk_id: int) -> tuple:
    """Encode one byte-range chunk of the skeleton CSV.

    Writes per-chunk numpy files to tmp_dir.
    Returns (prefix_path, n_rows_encoded).
    """
    # Rows are routed to their split here rather than sliced afterwards: the
    # split rule is per-row, so train and validation rows interleave and no
    # contiguous row range corresponds to a split.
    out = {
        s: {"sid": [], "y": [], "z": [], "cf": [],
            "acc": {f: {"ids": [], "weights": [], "offsets": []} for f in _FIELD_IDS}}
        for s in _SPLIT_NAMES
    }

    with open(sk_path, "rb") as f:
        seek_to_line_start(f, byte_start)
        while f.tell() < byte_end:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8")
            imp = parse_skeleton_line(line)
            if imp is None:
                continue
            cf_row = _CF_ROW.get(imp.common_feature_index)
            if cf_row is None:
                continue

            o = out[_split_of(imp.sample_id)]
            o["sid"].append(imp.sample_id)
            o["y"].append(imp.y)
            o["z"].append(imp.z)
            o["cf"].append(cf_row)

            for field_id in _FIELD_IDS:
                a = o["acc"][field_id]
                a["offsets"].append(len(a["ids"]))
                for feat_id, weight in imp.fields.get(field_id, []):
                    token_id = _VOCAB[field_id].get(str(feat_id), 0)
                    a["ids"].append(token_id)
                    a["weights"].append(0.0 if token_id == 0 else weight)

    prefix = os.path.join(tmp_dir, f"{chunk_id:05d}")
    counts = {}
    for split, o in out.items():
        np.save(f"{prefix}_{split}_sid.npy", np.array(o["sid"], dtype=np.int64))
        np.save(f"{prefix}_{split}_y.npy",   np.array(o["y"], dtype=np.int8))
        np.save(f"{prefix}_{split}_z.npy",   np.array(o["z"], dtype=np.int8))
        np.save(f"{prefix}_{split}_cf.npy",  np.array(o["cf"], dtype=np.int32))
        for field_id in _FIELD_IDS:
            a = o["acc"][field_id]
            fn = field_id.replace("/", "_")
            np.save(f"{prefix}_{split}_{fn}_ids.npy", np.array(a["ids"],     dtype=np.int32))
            np.save(f"{prefix}_{split}_{fn}_wts.npy", np.array(a["weights"], dtype=np.float32))
            np.save(f"{prefix}_{split}_{fn}_off.npy", np.array(a["offsets"], dtype=np.int64))
        counts[split] = len(o["sid"])

    return prefix, counts


def _parallel_encode(sk_path: str, cf_row: dict, vocab: dict, field_ids: list,
                      n_workers: int, tmp_dir: str, split_names: tuple) -> list:
    """Encode sk_path in parallel.  Returns list of (prefix, counts) by chunk id."""
    global _CF_ROW, _VOCAB, _FIELD_IDS, _SPLIT_NAMES
    # Set globals BEFORE forking so workers inherit via COW (no pickling of large dicts)
    _CF_ROW, _VOCAB, _FIELD_IDS, _SPLIT_NAMES = cf_row, vocab, field_ids, split_names

    size = os.path.getsize(sk_path)
    chunk_size = size // n_workers
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size if i < n_workers - 1 else size)
        for i in range(n_workers)
    ]
    logger.info(
        "Parallel encoding: %s (%.1f GB) with %d workers, ~%.0f MB each",
        os.path.basename(sk_path), size / 1e9, n_workers, chunk_size / 1e6,
    )

    args = [(sk_path, s, e, tmp_dir, i) for i, (s, e) in enumerate(ranges)]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = pool.starmap(_encode_chunk_worker, args)

    results.sort(key=lambda r: r[0])  # sort by prefix (=chunk id) to preserve row order
    totals = {s: sum(r[1][s] for r in results) for s in split_names}
    logger.info("Encoding done: %s", "  ".join(f"{s}={n:,}" for s, n in totals.items()))
    return results


# ---------------------------------------------------------------------------
# Merge chunk files → final split directories
# ---------------------------------------------------------------------------

def _merge_chunks(results: list, field_ids: list, split_dirs: dict) -> None:
    """Concatenate each split's chunk arrays into its output directory.

    Workers already routed every row to its split, so a split's rows are the
    concatenation of its per-chunk arrays in chunk order -- no positional
    slicing, which the hash-based validation rule would make meaningless.
    """
    t0 = time.perf_counter()
    prefixes = [r[0] for r in results]
    logger.info("Merging %d chunks -> %s", len(results), sorted(split_dirs))

    for split, out_dir in split_dirs.items():
        os.makedirs(out_dir, exist_ok=True)

        sid = np.concatenate([np.load(f"{p}_{split}_sid.npy") for p in prefixes])
        y   = np.concatenate([np.load(f"{p}_{split}_y.npy")   for p in prefixes])
        z   = np.concatenate([np.load(f"{p}_{split}_z.npy")   for p in prefixes])
        cf  = np.concatenate([np.load(f"{p}_{split}_cf.npy")  for p in prefixes])
        n_rows = len(sid)

        with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
            json.dump({"split": split, "n_rows": int(n_rows),
                       "n_clicks": int(y.sum()), "n_conversions": int(z.sum())}, f)
        np.save(os.path.join(out_dir, "sample_id.npy"), sid)
        np.save(os.path.join(out_dir, "y.npy"), y)
        np.save(os.path.join(out_dir, "z.npy"), z)
        np.save(os.path.join(out_dir, "cf_row.npy"), cf)
        logger.info("  [%s] %d rows, %d clicks, %d conversions",
                    split, n_rows, int(y.sum()), int(z.sum()))
        del sid, y, z, cf

        for field_id in field_ids:
            fn = field_id.replace("/", "_")
            ids_parts, wts_parts, off_parts, tok_counts, row_counts = [], [], [], [], []
            for p in prefixes:
                ids_c = np.load(f"{p}_{split}_{fn}_ids.npy")
                off_c = np.load(f"{p}_{split}_{fn}_off.npy")
                ids_parts.append(ids_c)
                wts_parts.append(np.load(f"{p}_{split}_{fn}_wts.npy"))
                off_parts.append(off_c)
                tok_counts.append(len(ids_c))
                row_counts.append(len(off_c))

            ids_all = np.concatenate(ids_parts); del ids_parts
            wts_all = np.concatenate(wts_parts); del wts_parts
            off_all = np.concatenate(off_parts); del off_parts

            # Each chunk's offsets restart at 0; shift by the running token base.
            base = row_ptr = 0
            for tok_n, row_n in zip(tok_counts, row_counts):
                off_all[row_ptr:row_ptr + row_n] += base
                base += tok_n
                row_ptr += row_n

            # CSR: n+1 entries, final one the total token count.
            out_off = np.empty(n_rows + 1, dtype=np.int64)
            out_off[:-1] = off_all
            out_off[-1] = len(ids_all)

            np.save(os.path.join(out_dir, f"{fn}_ids.npy"), ids_all)
            np.save(os.path.join(out_dir, f"{fn}_weights.npy"), wts_all)
            np.save(os.path.join(out_dir, f"{fn}_offsets.npy"), out_off)
            del ids_all, wts_all, off_all

    logger.info("Merge complete (%.1fs)", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Ali-CCP CSVs to numpy arrays.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_dir     = os.path.join(_ROOT_DIR, cfg["data"]["csv_dir"])
    encoded_dir = os.path.join(_ROOT_DIR, cfg["data"]["encoded_dir"])
    n_workers   = os.cpu_count() or 8

    os.makedirs(encoded_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: vocab (build once, reload on subsequent runs)
    # ------------------------------------------------------------------
    vocab_path     = os.path.join(encoded_dir, "vocab.pkl")
    field_ids_path = os.path.join(encoded_dir, "field_ids.json")

    # common_features_train is needed by the vocab pass, the disjointness check
    # and the common-table encoder; it is 8.6 GB of CSV and minutes to parse, so
    # load it at most once per run and hand the same dict to all three.
    cf_train = None

    if os.path.exists(vocab_path):
        logger.info("Loading existing vocab from %s", vocab_path)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(field_ids_path) as f:
            field_ids = json.load(f)
    else:
        field_ids = discover_field_ids(csv_dir)
        logger.info("Found %d fields: %s", len(field_ids), field_ids)

        cf_train = _load_common_features_parallel(
            os.path.join(csv_dir, "common_features_train.csv"), n_workers
        )
        vocab = build_vocab_streaming(csv_dir, field_ids, n_workers, cf_train)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(field_ids_path, "w") as f:
            json.dump(field_ids, f)
        logger.info("Saved vocab + field_ids to %s", encoded_dir)

    # ------------------------------------------------------------------
    # Step 2: encode train + val in a single pass (parallel)
    # ------------------------------------------------------------------
    train_done = os.path.exists(os.path.join(encoded_dir, "train", "y.npy"))
    val_done   = os.path.exists(os.path.join(encoded_dir, "val",   "y.npy"))
    if train_done and val_done:
        logger.info("[train+val] Already encoded, skipping")
    else:
        sk_train = os.path.join(csv_dir, "sample_skeleton_train.csv")
        if cf_train is None:  # vocab was reloaded from disk, so nothing loaded it
            cf_train = _load_common_features_parallel(
                os.path.join(csv_dir, "common_features_train.csv"), n_workers
            )
        _assert_fields_disjoint(sk_train, cf_train)
        cf_row = _encode_common_table(
            cf_train, vocab, field_ids, os.path.join(encoded_dir, "common_train"), n_workers
        )
        del cf_train  # workers only need the row map, not the multi-GB bags
        cf_train = None

        with tempfile.TemporaryDirectory(dir=encoded_dir, prefix="tmp_train_") as tmp:
            results = _parallel_encode(sk_train, cf_row, vocab, field_ids, n_workers, tmp,
                                       split_names=("train", "val"))
            _merge_chunks(results, field_ids, {
                "train": os.path.join(encoded_dir, "train"),
                "val": os.path.join(encoded_dir, "val"),
            })

    # ------------------------------------------------------------------
    # Step 3: encode test (parallel)
    # ------------------------------------------------------------------
    if os.path.exists(os.path.join(encoded_dir, "test", "y.npy")):
        logger.info("[test] Already encoded, skipping")
    else:
        sk_test = os.path.join(csv_dir, "sample_skeleton_test.csv")
        cf_test = _load_common_features_parallel(
            os.path.join(csv_dir, "common_features_test.csv"), n_workers
        )
        _assert_fields_disjoint(sk_test, cf_test)
        cf_row = _encode_common_table(
            cf_test, vocab, field_ids, os.path.join(encoded_dir, "common_test"), n_workers
        )
        del cf_test

        with tempfile.TemporaryDirectory(dir=encoded_dir, prefix="tmp_test_") as tmp:
            results = _parallel_encode(sk_test, cf_row, vocab, field_ids, n_workers, tmp,
                                       split_names=("test",))
            _merge_chunks(results, field_ids, {"test": os.path.join(encoded_dir, "test")})

    logger.info("Preprocessing complete. Encoded data at: %s", encoded_dir)


if __name__ == "__main__":
    main()
