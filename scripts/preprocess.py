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
    iter_impressions_from_csv,
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


# ---------------------------------------------------------------------------
# Vocab building (streaming, no Impression objects held)
# ---------------------------------------------------------------------------

def build_vocab_streaming(csv_dir: str, field_ids: list, n_workers: int) -> dict:
    """Build vocab by streaming train skeleton once; never holds all rows in RAM."""
    logger.info("Building vocab (streaming train split)...")
    tokens_by_field: dict[str, set] = {f: set() for f in field_ids}
    count = 0
    t0 = time.perf_counter()
    for imp in iter_impressions_from_csv(csv_dir, "train", n_workers=n_workers):
        for field_id, entries in imp.fields.items():
            if field_id in tokens_by_field:
                tokens_by_field[field_id].update(str(feat_id) for feat_id, _ in entries)
        count += 1
        if count % 2_000_000 == 0:
            logger.info("  vocab pass: %dM rows (%.1fs)", count // 1_000_000, time.perf_counter() - t0)
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
    sample_ids, ys, zs, cf_rows = [], [], [], []
    acc: dict = {f: {"ids": [], "weights": [], "offsets": []} for f in _FIELD_IDS}

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

            sample_ids.append(imp.sample_id)
            ys.append(imp.y)
            zs.append(imp.z)
            cf_rows.append(cf_row)

            for field_id in _FIELD_IDS:
                a = acc[field_id]
                a["offsets"].append(len(a["ids"]))
                for feat_id, weight in imp.fields.get(field_id, []):
                    token_id = _VOCAB[field_id].get(str(feat_id), 0)
                    a["ids"].append(token_id)
                    a["weights"].append(0.0 if token_id == 0 else weight)

    prefix = os.path.join(tmp_dir, f"{chunk_id:05d}")
    np.save(f"{prefix}_sid.npy", np.array(sample_ids, dtype=np.int64))
    np.save(f"{prefix}_y.npy",   np.array(ys, dtype=np.int8))
    np.save(f"{prefix}_z.npy",   np.array(zs, dtype=np.int8))
    np.save(f"{prefix}_cf.npy",  np.array(cf_rows, dtype=np.int32))
    for field_id in _FIELD_IDS:
        a = acc[field_id]
        fn = field_id.replace("/", "_")
        np.save(f"{prefix}_{fn}_ids.npy", np.array(a["ids"],     dtype=np.int32))
        np.save(f"{prefix}_{fn}_wts.npy", np.array(a["weights"], dtype=np.float32))
        np.save(f"{prefix}_{fn}_off.npy", np.array(a["offsets"], dtype=np.int64))

    return prefix, len(sample_ids)


def _parallel_encode(sk_path: str, cf_row: dict, vocab: dict, field_ids: list,
                      n_workers: int, tmp_dir: str) -> list:
    """Encode sk_path in parallel.  Returns list of (prefix, n_rows) sorted by chunk id."""
    global _CF_ROW, _VOCAB, _FIELD_IDS
    # Set globals BEFORE forking so workers inherit via COW (no pickling of large dicts)
    _CF_ROW, _VOCAB, _FIELD_IDS = cf_row, vocab, field_ids

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
    total_rows = sum(r[1] for r in results)
    logger.info("Encoding done: %d rows total", total_rows)
    return results


# ---------------------------------------------------------------------------
# Merge chunk files → final split directories
# ---------------------------------------------------------------------------

def _merge_chunks(results: list, field_ids: list, out_dirs: list,
                   val_frac: float = 0.0) -> None:
    """Concatenate chunk arrays and write to one or more output directories.

    If two out_dirs are given (train + val), the first 1-val_frac fraction
    goes to out_dirs[0] and the remainder to out_dirs[1].
    If one out_dir is given, all rows go there.

    Since rows are stored in order and each chunk writes tokens sequentially,
    the token data for any contiguous row range [a, b) is guaranteed to
    occupy a contiguous slice of the global flat array.  We exploit this to
    avoid a Python per-row loop (which would take hours at 34 M rows × 23 fields).
    """
    t0 = time.perf_counter()
    logger.info("Merging %d chunks → %s", len(results), [os.path.basename(d) for d in out_dirs])

    prefixes = [r[0] for r in results]

    # ---- Label arrays (small: ~42M × 10 bytes each) ----------------------------
    sid_all = np.concatenate([np.load(f"{p}_sid.npy") for p in prefixes])
    y_all   = np.concatenate([np.load(f"{p}_y.npy")   for p in prefixes])
    z_all   = np.concatenate([np.load(f"{p}_z.npy")   for p in prefixes])
    cf_all  = np.concatenate([np.load(f"{p}_cf.npy")  for p in prefixes])

    n_total = len(sid_all)
    if len(out_dirs) == 2:
        n_train = int(n_total * (1 - val_frac))
        # row ranges: (start_row, end_row) — end is exclusive
        row_ranges = [(0, n_train), (n_train, n_total)]
    else:
        row_ranges = [(0, n_total)]

    for out_dir, (rs, re) in zip(out_dirs, row_ranges):
        os.makedirs(out_dir, exist_ok=True)
        # Record the split actually written; the checkpoint must report this,
        # not whatever val_frac a later training config happens to carry.
        with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
            json.dump({"val_frac": val_frac, "n_rows": re - rs}, f)
        np.save(os.path.join(out_dir, "sample_id.npy"), sid_all[rs:re])
        np.save(os.path.join(out_dir, "y.npy"),         y_all[rs:re])
        np.save(os.path.join(out_dir, "z.npy"),         z_all[rs:re])
        np.save(os.path.join(out_dir, "cf_row.npy"),    cf_all[rs:re])
        logger.info("  [%s] labels saved (%d rows)", os.path.basename(out_dir), re - rs)
    del sid_all, y_all, z_all, cf_all

    # ---- Per-field token arrays -------------------------------------------------
    for field_id in field_ids:
        fn = field_id.replace("/", "_")

        # Load all chunk files; track each chunk's token count to fix offsets.
        ids_parts, wts_parts, off_parts = [], [], []
        tok_counts: list[int] = []      # number of tokens in each chunk
        row_counts: list[int] = []      # number of rows   in each chunk
        for p in prefixes:
            ids_c = np.load(f"{p}_{fn}_ids.npy")
            wts_c = np.load(f"{p}_{fn}_wts.npy")
            off_c = np.load(f"{p}_{fn}_off.npy")
            ids_parts.append(ids_c)
            wts_parts.append(wts_c)
            off_parts.append(off_c)
            tok_counts.append(len(ids_c))
            row_counts.append(len(off_c))

        ids_all = np.concatenate(ids_parts); del ids_parts
        wts_all = np.concatenate(wts_parts); del wts_parts

        # Fix per-chunk offsets: each chunk resets to 0; add cumulative token base.
        off_all = np.concatenate(off_parts); del off_parts
        base = 0
        row_ptr = 0
        for tok_n, row_n in zip(tok_counts, row_counts):
            off_all[row_ptr:row_ptr + row_n] += base
            base    += tok_n
            row_ptr += row_n

        # Vectorized write: token data for a contiguous row range [rs, re) is
        # the contiguous token slice [off_all[rs], off_all[re]) (or end of array
        # when re == n_total).  No Python per-row loop needed.
        for out_dir, (rs, re) in zip(out_dirs, row_ranges):
            # off_all has one entry per row, so an index at n_total (an empty
            # trailing range, e.g. val_frac=0) means "past the last token".
            tok_start = int(off_all[rs]) if rs < n_total else len(ids_all)
            tok_end   = int(off_all[re]) if re < n_total else len(ids_all)
            # Local offsets for this slice — subtract the global token base, then
            # append the total so offsets are CSR (n+1) and row i spans [i]:[i+1].
            out_off = np.empty(re - rs + 1, dtype=np.int64)
            out_off[:-1] = off_all[rs:re] - tok_start
            out_off[-1] = tok_end - tok_start
            np.save(os.path.join(out_dir, f"{fn}_ids.npy"),     ids_all[tok_start:tok_end])
            np.save(os.path.join(out_dir, f"{fn}_weights.npy"), wts_all[tok_start:tok_end])
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
    val_frac    = cfg["training"]["val_frac"]
    n_workers   = os.cpu_count() or 8

    os.makedirs(encoded_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: vocab (build once, reload on subsequent runs)
    # ------------------------------------------------------------------
    vocab_path     = os.path.join(encoded_dir, "vocab.pkl")
    field_ids_path = os.path.join(encoded_dir, "field_ids.json")

    if os.path.exists(vocab_path):
        logger.info("Loading existing vocab from %s", vocab_path)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(field_ids_path) as f:
            field_ids = json.load(f)
    else:
        logger.info("Discovering field IDs from a small sample...")
        field_ids_set: set = set()
        for imp in iter_impressions_from_csv(csv_dir, "train", limit=10_000,
                                              n_workers=min(8, n_workers)):
            field_ids_set.update(imp.fields.keys())
        field_ids = sorted(field_ids_set)
        logger.info("Found %d fields: %s", len(field_ids), field_ids)

        vocab = build_vocab_streaming(csv_dir, field_ids, n_workers)
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
        cf_train = _load_common_features_parallel(
            os.path.join(csv_dir, "common_features_train.csv"), n_workers
        )
        _assert_fields_disjoint(sk_train, cf_train)
        cf_row = _encode_common_table(
            cf_train, vocab, field_ids, os.path.join(encoded_dir, "common_train"), n_workers
        )
        del cf_train  # workers only need the row map, not the multi-GB bags

        with tempfile.TemporaryDirectory(dir=encoded_dir, prefix="tmp_train_") as tmp:
            results = _parallel_encode(sk_train, cf_row, vocab, field_ids, n_workers, tmp)
            _merge_chunks(
                results, field_ids,
                out_dirs=[os.path.join(encoded_dir, "train"), os.path.join(encoded_dir, "val")],
                val_frac=val_frac,
            )

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
            results = _parallel_encode(sk_test, cf_row, vocab, field_ids, n_workers, tmp)
            _merge_chunks(
                results, field_ids,
                out_dirs=[os.path.join(encoded_dir, "test")],
            )

    logger.info("Preprocessing complete. Encoded data at: %s", encoded_dir)


if __name__ == "__main__":
    main()
