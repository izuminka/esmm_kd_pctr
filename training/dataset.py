"""Dataset for preprocessed Ali-CCP numpy arrays.

Loaded once after preprocess.py runs and copied in full onto a single
device (GPU if available); Trainer consumes this instead of raw
Impression objects, eliminating the 109GB Python-object overhead.

Ali-CCP features come in two halves: per-impression features, plus a
user-level bag shared by ~58 impressions.  preprocess.py keeps those
halves separate — expanding the join costs ~180 GB, against ~8 GB kept
apart — so a batch is assembled here by gathering both halves and
concatenating them per row.  The model still sees one bag per field and
is unaware of the split.

numpy is used only for np.load. All batch construction and training
are pure PyTorch.
"""

import json
import logging
import os
import pickle

import numpy as np  # only for np.load; not used in the training hot path
import torch

logger = logging.getLogger(__name__)


def _load(path: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).to(device)


def _load_ragged(src_dir: str, field_ids: list, device: torch.device) -> tuple:
    """Load per-field (ids, weights, offsets) arrays onto `device`.

    Offsets are CSR: n+1 entries, so row i spans offsets[i] : offsets[i+1].
    Row ends are read from the next offset rather than stored alongside, which
    would duplicate a large array for a value already one index away.
    """
    ids, weights, offsets = {}, {}, {}
    for field_id in field_ids:
        fname = field_id.replace("/", "_")
        ids[field_id] = _load(os.path.join(src_dir, f"{fname}_ids.npy"), device)       # int32
        weights[field_id] = _load(os.path.join(src_dir, f"{fname}_weights.npy"), device)
        offsets[field_id] = _load(os.path.join(src_dir, f"{fname}_offsets.npy"), device)
    return ids, weights, offsets


def _exclusive_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Start position of each segment when segments are packed back to back."""
    out = torch.zeros_like(lengths)
    if len(lengths) > 1:
        out[1:] = lengths[:-1].cumsum(0)
    return out


def _gather_index(starts: torch.Tensor, lengths: torch.Tensor, total: int,
                  dest_off: torch.Tensor | None = None) -> tuple:
    """Flat (src, dst) index pairs copying ragged segments into a buffer.

    Segment k reads src[starts[k] : starts[k] + lengths[k]] and lands at
    dst[dest_off[k] : dest_off[k] + lengths[k]].  Vectorised via
    repeat_interleave — no Python loop over rows.

    `total` (== lengths.sum()) is passed in, and on to repeat_interleave as
    output_size, because otherwise each call blocks to read the sum back off
    the device.  dest_off=None skips building dst, for callers writing the
    segments straight out rather than into a shared buffer.
    """
    inner = torch.arange(total, dtype=torch.int64, device=lengths.device) - \
        torch.repeat_interleave(_exclusive_cumsum(lengths), lengths, output_size=total)
    src = torch.repeat_interleave(starts, lengths, output_size=total) + inner
    if dest_off is None:
        return src, None
    return src, torch.repeat_interleave(dest_off, lengths, output_size=total) + inner


class CommonTable:
    """The unique common-feature bags, encoded once and shared across rows.

    Written by preprocess.py to ali-ccp/encoded/common_{train,test}/.  Many
    impressions point at the same row here, which is what keeps the whole
    dataset small enough to hold on a single GPU.
    """

    def __init__(self, table_dir: str, field_ids: list, device: torch.device):
        self.field_ids = field_ids
        self.device = device
        self.ids, self.weights, self.offsets = _load_ragged(table_dir, field_ids, device)
        self._n = len(self.offsets[field_ids[0]]) - 1  # CSR offsets hold n+1 entries

    def __len__(self) -> int:
        return self._n


class SplitData:
    """One split (train / val / test), fully loaded onto a single device.

    Arrays were written by preprocess.py in EmbeddingBag format:
        {field}_ids.npy      — flat int32 token ids, impression-own features only
        {field}_offsets.npy  — int64 CSR offsets, n+1 entries (row i: [i]:[i+1])
        {field}_weights.npy  — float32 weights, same shape as ids
        sample_id.npy        — int64
        cf_row.npy           — int32 row pointer into the CommonTable
        y.npy                — int8 click labels
        z.npy                — int8 conversion labels
    """

    def __init__(self, split_dir: str, field_ids: list, common: CommonTable,
                 device: torch.device):
        self.field_ids = field_ids
        self.common = common
        self.device = device

        self.sample_ids = _load(os.path.join(split_dir, "sample_id.npy"), device)  # int64
        self.ys = _load(os.path.join(split_dir, "y.npy"), device)                  # int8
        self.zs = _load(os.path.join(split_dir, "z.npy"), device)                  # int8
        self.cf_row = _load(os.path.join(split_dir, "cf_row.npy"), device).long()  # int32

        self.ids, self.weights, self.offsets = _load_ragged(split_dir, field_ids, device)
        self._n = len(self.sample_ids)
        # Frozen per-row pCTR from a DCN run, aligned to these rows; attached by
        # data.logged_pctr when a config names one, else None.
        self.logged_pctr = None

    def __len__(self) -> int:
        return self._n

    def get_batch(self, indices) -> tuple:
        """Return (encoded, y, z) for the given row indices.

        encoded: {field_id: (ids_int64, weights_float32, offsets_int64)}
        y, z:    float32 tensors, shape [B]

        Each bag is the row's own tokens followed by its common-bag tokens.
        Pooling is order-independent, so the model sees exactly what it would
        have seen had the join been materialised on disk.
        """
        dev = self.device
        idx = torch.as_tensor(indices, dtype=torch.int64, device=dev)
        cf = self.cf_row[idx]
        encoded = {}

        for field_id in self.field_ids:
            own_off = self.offsets[field_id]
            com_off = self.common.offsets[field_id]
            own_start = own_off[idx]
            own_len = own_off[idx + 1] - own_start
            com_start = com_off[cf]
            com_len = com_off[cf + 1] - com_start

            bag_off = _exclusive_cumsum(own_len + com_len)
            # One readback for both totals; two separate .sum() calls would
            # block twice per field.
            own_total, com_total = torch.stack((own_len.sum(), com_len.sum())).tolist()

            if com_total == 0:
                # Field lives entirely on the impression; gather it directly
                # rather than paying for a buffer and a scatter.
                src, _ = _gather_index(own_start, own_len, own_total)
                ids = self.ids[field_id][src].long()
                wts = self.weights[field_id][src]
            elif own_total == 0:
                src, _ = _gather_index(com_start, com_len, com_total)
                ids = self.common.ids[field_id][src].long()
                wts = self.common.weights[field_id][src]
            else:
                total = own_total + com_total
                ids = torch.zeros(total, dtype=torch.int64, device=dev)
                wts = torch.zeros(total, dtype=torch.float32, device=dev)

                src, dst = _gather_index(own_start, own_len, own_total, bag_off)
                ids[dst] = self.ids[field_id][src].long()
                wts[dst] = self.weights[field_id][src]

                src, dst = _gather_index(com_start, com_len, com_total, bag_off + own_len)
                ids[dst] = self.common.ids[field_id][src].long()
                wts[dst] = self.common.weights[field_id][src]

            encoded[field_id] = (ids, wts, bag_off)

        y = self.ys[idx].float()
        z = self.zs[idx].float()
        return encoded, y, z


class NumpyDataset:
    """Encoded Ali-CCP dataset, fully loaded onto one device.

    Usage:
        dataset = NumpyDataset.load(encoded_dir, device=device)
        model = DCNv2(dataset.vocab_sizes)
        trainer.fit(model, dataset, metric_fn)
    """

    def __init__(
        self,
        train: SplitData,
        val: SplitData,
        test: SplitData,
        vocab: dict,
        field_ids: list,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.vocab = vocab
        self.field_ids = field_ids

    @property
    def vocab_sizes(self) -> dict:
        # len(vocab[f]) = number of real tokens; DCNv2 adds +1 internally for OOV slot 0
        return {f: len(v) for f, v in self.vocab.items()}

    @classmethod
    def load(cls, encoded_dir: str, device: torch.device,
             load_test: bool = True) -> "NumpyDataset":
        """Load from a directory produced by preprocess.py, fully onto `device`.

        All tensors are copied to `device` at load time, so training does
        zero-copy indexing with no per-batch host<->device transfers.

        The train/val split is fixed on disk by preprocess.py, which assigns
        each row by hashing its example id; there is no split fraction to pass.

        load_test=False leaves `test` and its common table on disk, freeing the
        VRAM they would occupy; only the prediction export needs them.
        """
        vocab_path = os.path.join(encoded_dir, "vocab.pkl")
        field_ids_path = os.path.join(encoded_dir, "field_ids.json")

        logger.info("Loading vocab from %s", vocab_path)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(field_ids_path) as f:
            field_ids = json.load(f)

        logger.info("Loading arrays onto %s...", device)
        # train and val come from the same source file, so they share one table
        common_train = CommonTable(os.path.join(encoded_dir, "common_train"), field_ids, device)
        train = SplitData(os.path.join(encoded_dir, "train"), field_ids, common_train, device)
        val   = SplitData(os.path.join(encoded_dir, "val"),   field_ids, common_train, device)

        test = None
        if load_test:
            common_test = CommonTable(os.path.join(encoded_dir, "common_test"), field_ids, device)
            test = SplitData(os.path.join(encoded_dir, "test"), field_ids, common_test, device)

        logger.info(
            "Loaded on %s — train=%d  val=%d  test=%s  (%d common bags)",
            device, len(train), len(val), len(test) if test else "not loaded",
            len(common_train),
        )
        return cls(train, val, test, vocab, field_ids)
