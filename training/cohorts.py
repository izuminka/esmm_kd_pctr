"""Fixed, label-independent training cohorts: the ESMM half and quarter pools.

The recipe (paper appendix, "ESMM recipe") forms one stream from the training
blocks and keeps a prefix of each of four balanced contiguous ranges of it:

    1. order the block inventory by identifier, then permute it with `seed`;
    2. concatenate the blocks, keeping row order inside each, into one stream;
    3. split the stream into `ranks` balanced contiguous ranges;
    4. keep floor(n * fraction) rows in total, split as evenly as possible across
       the ranges with any remainder on the lowest ranks, each range keeping that
       many rows from its start.

On the 38,070,359-row training split this gives the paper's 19,035,179-row half
and its 9,517,589-row quarter with rank prefixes 2,379,398, 2,379,397, 2,379,397
and 2,379,397. A smaller fraction keeps a prefix of each range no longer than a
larger one's, so the quarter is nested in the half.

Blocks here are consecutive runs of `block_size` rows of the encoded split,
identified by their zero-padded index, so identifier order is index order. The
paper's own block boundaries are not published: as it notes, re-materialising
with different blocking reproduces the rule and cohort size, not the identical
rows. Membership depends only on row positions and the settings, never on labels,
and is fixed across epochs.
"""

from fractions import Fraction

import numpy as np
import torch


def balanced(total: int, parts: int) -> list[int]:
    """`total` split into `parts` sizes differing by at most one, larger ones first."""
    q, r = divmod(total, parts)
    return [q + (i < r) for i in range(parts)]


def cohort_rows(n: int, fraction, ranks: int, block_size: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Row indices of the cohort in stream order, and the rows kept from each range.

    `fraction` is read through its decimal string, so 0.5 and 0.25 are exact.
    """
    fraction = Fraction(str(fraction))
    if not 0 < fraction <= 1:
        raise ValueError(f"cohort fraction must be in (0, 1], got {fraction}")

    n_blocks = -(-n // block_size)
    order = np.random.default_rng(seed).permutation(n_blocks).astype(np.int64)
    stream = (order[:, None] * block_size + np.arange(block_size, dtype=np.int64)).ravel()
    stream = stream[stream < n]  # the last block is short unless block_size divides n

    range_starts = np.cumsum([0] + balanced(n, ranks)[:-1])
    kept = balanced(int(n * fraction), ranks)
    rows = np.concatenate([stream[s:s + k] for s, k in zip(range_starts, kept)])
    return rows, kept


class CohortSplit:
    """A loaded split restricted to cohort rows, which it presents as rows 0..len-1.

    Only the row index is stored; features stay in the full split. Logged pCTR is
    attached to the full split, whose predictions file it must match row for row,
    and read through here restricted to the cohort.
    """

    def __init__(self, split, rows: np.ndarray, rank_rows: list[int]):
        self.split = split
        self.rank_rows = rank_rows
        self.device = split.device
        self.rows = torch.as_tensor(rows, dtype=torch.int64).to(split.device)
        self.sample_ids = split.sample_ids[self.rows]
        self.ys = split.ys[self.rows]
        self.zs = split.zs[self.rows]
        self._logged_pctr = self._logged_pctr_of = None

    def __len__(self) -> int:
        return len(self.rows)

    def get_batch(self, indices) -> tuple:
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        return self.split.get_batch(self.rows[idx])

    @property
    def logged_pctr(self):
        full = self.split.logged_pctr
        if full is None:
            return None
        if self._logged_pctr_of is not full:  # gather once per attached tensor
            self._logged_pctr, self._logged_pctr_of = full[self.rows], full
        return self._logged_pctr

    def census(self) -> dict:
        """Size and label counts of the cohort, plus the rows each range kept."""
        return {"n_rows": len(self), "n_clicks": int(self.ys.sum()),
                "n_conversions": int(self.zs.sum()), "rank_rows": self.rank_rows}


def restrict(split, spec: dict) -> CohortSplit:
    """`split` restricted to the cohort a config's `train_cohort` block describes."""
    rows, kept = cohort_rows(len(split), spec["fraction"], spec["ranks"],
                             spec["block_size"], spec["seed"])
    return CohortSplit(split, rows, kept)
