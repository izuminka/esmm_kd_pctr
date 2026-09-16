"""Equivalence tests for the deduplicated encoding, against real Ali-CCP data.

preprocess.py stores each common-feature bag once and gives every impression a
cf_row pointer; training/dataset.py stitches the two halves back together per
batch.  The thing that must hold is that the model sees exactly what it saw when
the join was materialised per row, so these tests compare get_batch() against
data.vocab.encode_batch() over the same impressions.

A fixture is carved out of the real extracted CSVs rather than synthesised: the
delimiter nesting, the skeleton/common split and the ragged field layout are
exactly the things a synthetic fixture would paper over.
"""

import os
import pickle
import subprocess
import sys

import pytest
import torch
import yaml

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
_CSV_DIR = os.path.join(_ROOT_DIR, "ali-ccp", "extracted")

if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data.raw import is_validation, iter_impressions_from_csv  # noqa: E402
from data.vocab import encode_batch  # noqa: E402
from training.dataset import NumpyDataset  # noqa: E402

_N_BAGS = 4000       # common_features rows to copy into the fixture
_N_ROWS = 400        # skeleton rows (that reference those bags) to keep
_SCAN_LIMIT = 400_000  # skeleton lines to scan looking for matches

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(_CSV_DIR, "common_features_train.csv")),
    reason="extracted Ali-CCP CSVs not present",
)


def _carve_split(csv_dir: str, split: str, out_dir: str) -> int:
    """Copy a small but *self-consistent* slice of one split into out_dir.

    Head-of-file slices of the two files barely intersect (the skeleton
    references bags scattered through common_features), so take the bags first
    and then keep only skeleton rows pointing at them.
    """
    cf_src = os.path.join(csv_dir, f"common_features_{split}.csv")
    sk_src = os.path.join(csv_dir, f"sample_skeleton_{split}.csv")

    kept_idx = set()
    with open(cf_src, encoding="utf-8") as fin, \
         open(os.path.join(out_dir, f"common_features_{split}.csv"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= _N_BAGS:
                break
            kept_idx.add(line.split(",", 1)[0])
            fout.write(line)

    n = 0
    with open(sk_src, encoding="utf-8") as fin, \
         open(os.path.join(out_dir, f"sample_skeleton_{split}.csv"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if n >= _N_ROWS or i >= _SCAN_LIMIT:
                break
            parts = line.split(",", 5)
            if len(parts) == 6 and parts[3] in kept_idx and int(parts[2]) <= int(parts[1]):
                fout.write(line)
                n += 1
    return n


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    """Run the real preprocess.py over a fixture carved from the real CSVs."""
    base = tmp_path_factory.mktemp("alliccp")
    csv_dir = base / "csv"
    csv_dir.mkdir()
    enc_dir = base / "encoded"

    n_train = _carve_split(_CSV_DIR, "train", str(csv_dir))
    n_test = _carve_split(_CSV_DIR, "test", str(csv_dir))
    if n_train < 50 or n_test < 10:
        pytest.skip(f"too few joinable rows in fixture (train={n_train}, test={n_test})")

    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": "dcn",
        "training": {"epochs": 1, "batch_size": 64,
                     "eval_batch_size": 128, "lr": 1e-3, "grad_clip_norm": 0,
                     "shuffle_block_size": 128,
                     "gpus": 0, "out_dir": str(base / "artifacts")},
        "data": {"csv_dir": str(csv_dir), "encoded_dir": str(enc_dir)},
    }))

    subprocess.run(
        [sys.executable, os.path.join(_CODE_DIR, "scripts", "preprocess.py"),
         "--config", str(cfg_path)],
        check=True, cwd=_CODE_DIR, capture_output=True, text=True, timeout=900,
    )
    return str(enc_dir), str(csv_dir), n_train


def _bags(encoded_field, n_rows):
    """(ids, weights, offsets) -> list of per-row sorted [(id, weight)] pairs."""
    ids, weights, offsets = encoded_field
    ids, weights, offsets = ids.tolist(), weights.tolist(), offsets.tolist()
    out = []
    for i in range(n_rows):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(ids)
        out.append(sorted(zip(ids[start:end], weights[start:end])))
    return out


def test_batch_matches_materialised_join(encoded):
    """get_batch must reproduce encode_batch over the row-wise merged features."""
    enc_dir, csv_dir, n_total = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))

    # Legacy path: data.raw merges the common bag into each impression itself.
    # Select the split the same way preprocess did -- by id hash, not position.
    imps = [i for i in iter_impressions_from_csv(csv_dir, "train", n_workers=1)
            if not is_validation(i.sample_id)]
    n_train = len(imps)
    assert len(ds.train) == n_train, (len(ds.train), n_train)

    with open(os.path.join(enc_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    expected = encode_batch(imps, ds.field_ids, vocab)
    actual, y, z = ds.train.get_batch(list(range(n_train)))

    assert set(actual) == set(expected)
    for field_id in ds.field_ids:
        got = _bags(actual[field_id], n_train)
        want = _bags(expected[field_id], n_train)
        assert got == want, f"field {field_id} differs"

    assert y.tolist() == [float(i.y) for i in imps]
    assert z.tolist() == [float(i.z) for i in imps]


def test_arbitrary_row_order(encoded):
    """Indices are gathered independently, so a shuffled batch must still match."""
    enc_dir, csv_dir, _ = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))
    imps = [i for i in iter_impressions_from_csv(csv_dir, "train", n_workers=1)
            if not is_validation(i.sample_id)]
    n_train = len(imps)

    with open(os.path.join(enc_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    order = list(range(n_train - 1, -1, -3))  # reversed, strided
    expected = encode_batch([imps[i] for i in order], ds.field_ids, vocab)
    actual, _, _ = ds.train.get_batch(order)

    for field_id in ds.field_ids:
        assert _bags(actual[field_id], len(order)) == _bags(expected[field_id], len(order)), field_id


def test_common_bags_are_not_copied_into_rows(encoded):
    """The whole point: a shared bag is stored once, not once per referencing row.

    Reintroducing `imp.fields.update(common)` in preprocess.py would put the
    user-side tokens back into every row's own arrays, which is exactly what
    blows 8 GB up to 180 GB.  Assert they are not there.
    """
    enc_dir, _, _ = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))
    common = ds.train.common

    user_side = [f for f in ds.field_ids if len(common.ids[f]) > 0]
    assert user_side, "fixture has no common-side fields; test would be vacuous"

    for field_id in user_side:
        for split in (ds.train, ds.val):
            assert len(split.ids[field_id]) == 0, (
                f"field {field_id} has per-row tokens; the common bag was "
                "materialised into rows instead of being shared"
            )

    # cf_row indexes the table, and rows genuinely share bags
    cf = ds.train.cf_row
    assert int(cf.min()) >= 0 and int(cf.max()) < len(common)
    n_distinct = len(set(cf.tolist()))
    assert n_distinct < len(ds.train), "fixture has no bag sharing; test is weak"

    # the saving is real: expanding would cost len(train) copies of each bag
    tokens_shared = sum(len(common.ids[f]) for f in user_side) / max(len(common), 1)
    assert tokens_shared * n_distinct < tokens_shared * len(ds.train)


def test_trains_end_to_end(encoded, tmp_path):
    """Fixture → GPU-resident tensors → Trainer.fit → checkpoint + predictions."""
    import pandas as pd

    from models.dcn import DCNv2
    from training.trainer import Trainer

    enc_dir, _, _ = encoded
    out = str(tmp_path / "artifacts")
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))

    cfg = {"model": "dcn", "training": {
        "epochs": 1, "batch_size": 64, "eval_batch_size": 128, "lr": 1e-3,
        "grad_clip_norm": 0, "shuffle_block_size": 128,
        "seed": 42, "gpus": 0, "out_dir": out,
    }}
    model = DCNv2(ds.vocab_sizes, emb_dim=8, cross_rank=4,
                  num_cross_layers=1, deep_hidden_dims=[8])
    trainer = Trainer(cfg, root_dir="/")
    ckpt_path = trainer.fit(model, ds, metric_fn=None)

    assert os.path.exists(ckpt_path)
    for split, n in [("train", len(ds.train)), ("val", len(ds.val)), ("test", len(ds.test))]:
        df = pd.read_parquet(os.path.join(out, f"predictions_{split}.parquet"))
        assert list(df.columns) == ["sample_id", "y", "z", "score"]
        assert len(df) == n
        assert df["score"].between(0.0, 1.0).all()


def test_test_split_can_be_left_unloaded(encoded):
    """load_test=False keeps ~12 GB of VRAM free during the training loop."""
    enc_dir, _, _ = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"), load_test=False)
    assert ds.test is None
    assert len(ds.train) > 0 and len(ds.val) > 0
    ds.train.get_batch([0])  # train path unaffected


def test_split_rule_matches_paper_reference_vectors():
    """Pin the SplitMix64 variant the paper's census implies.

    The bare finalizer (no golden-gamma step) is the other common reading of
    "splitmix64(x)"; it puts ~315 different rows in validation across the full
    train split, which is invisible until the reproduced numbers drift. These
    vectors are the standard SplitMix64 outputs for a zero seed.
    """
    from data.raw import _VAL_THRESHOLD, splitmix64

    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(0xE220A8397B1DCDAF ^ 0) != 0  # sanity: mixes, not identity
    assert _VAL_THRESHOLD == 1_844_674_407_370_955_264


def test_train_and_val_are_disjoint_and_hash_determined(encoded):
    """Every encoded row sits on the side of the split its id hashes to."""
    from data.raw import is_validation

    enc_dir, _, _ = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))

    train_ids = ds.train.sample_ids.tolist()
    val_ids = ds.val.sample_ids.tolist()
    assert not (set(train_ids) & set(val_ids)), "a row landed in both splits"
    assert all(not is_validation(i) for i in train_ids)
    assert all(is_validation(i) for i in val_ids)


def test_offsets_are_csr(encoded):
    """dataset.py derives row ends as offsets[i+1]; that needs n+1 entries."""
    enc_dir, _, _ = encoded
    ds = NumpyDataset.load(enc_dir, device=torch.device("cpu"))

    for split in (ds.train, ds.val):
        for field_id in ds.field_ids:
            off = split.offsets[field_id]
            assert len(off) == len(split) + 1, (field_id, len(off), len(split))
            assert int(off[-1]) == len(split.ids[field_id])
            assert int(off[0]) == 0

    common = ds.train.common
    for field_id in ds.field_ids:
        off = common.offsets[field_id]
        assert len(off) == len(common) + 1
        assert int(off[-1]) == len(common.ids[field_id])
