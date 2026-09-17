"""The half and quarter training cohorts: the paper's sizes, nesting, and the split view."""

import numpy as np
import pytest
import torch

from training.cohorts import balanced, cohort_rows, restrict

_N_TRAIN = 38_070_359  # the cleaned official-train rows outside validation
_SPEC = {"ranks": 4, "block_size": 65_536, "seed": 42}


@pytest.fixture(scope="module")
def paper_cohorts():
    return {f: cohort_rows(_N_TRAIN, f, **_SPEC) for f in (0.5, 0.25)}


def test_sizes_and_rank_prefixes_match_the_paper(paper_cohorts):
    half, half_kept = paper_cohorts[0.5]
    quarter, quarter_kept = paper_cohorts[0.25]
    assert len(half) == 19_035_179
    assert half_kept == [4_758_795, 4_758_795, 4_758_795, 4_758_794]
    assert len(quarter) == 9_517_589
    assert quarter_kept == [2_379_398, 2_379_397, 2_379_397, 2_379_397]


def test_rows_are_distinct_in_range_and_quarter_is_nested_in_half(paper_cohorts):
    half, _ = paper_cohorts[0.5]
    quarter, _ = paper_cohorts[0.25]
    for rows in (half, quarter):
        assert rows.min() >= 0 and rows.max() < _N_TRAIN
        assert len(np.unique(rows)) == len(rows)
    assert np.isin(quarter, half).all()


def test_exact_one_third_cohort_is_nested_in_half():
    third, kept = cohort_rows(_N_TRAIN, "1/3", **_SPEC)
    half, _ = cohort_rows(_N_TRAIN, 0.5, **_SPEC)
    assert len(third) == 12_690_119
    assert kept == [3_172_530, 3_172_530, 3_172_530, 3_172_529]
    assert np.isin(third, half).all()


def test_each_range_keeps_its_own_prefix_of_the_block_stream():
    n, block_size = 1_003, 10  # short last block
    rows, kept = cohort_rows(n, 0.5, ranks=4, block_size=block_size, seed=7)
    full, _ = cohort_rows(n, 1, ranks=4, block_size=block_size, seed=7)

    assert sorted(full) == list(range(n))  # fraction 1 is the whole stream
    starts = np.cumsum([0] + balanced(n, 4)[:-1])
    expected = np.concatenate([full[s:s + k] for s, k in zip(starts, kept)])
    assert np.array_equal(rows, expected)
    # Blocks move whole: rows inside a block stay consecutive in the stream.
    block_ends = {r for r in range(n) if (r + 1) % block_size == 0} | {n - 1}
    assert all(full[i + 1] == full[i] + 1 for i in range(n - 1) if full[i] not in block_ends)


def test_membership_is_fixed_by_the_settings():
    a, _ = cohort_rows(10_000, 0.25, ranks=4, block_size=64, seed=42)
    b, _ = cohort_rows(10_000, 0.25, ranks=4, block_size=64, seed=42)
    c, _ = cohort_rows(10_000, 0.25, ranks=4, block_size=64, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(np.sort(a), np.sort(c))


class _FakeSplit:
    def __init__(self, n):
        self.device = torch.device("cpu")
        self.sample_ids = torch.arange(100, 100 + n)
        self.ys = (torch.arange(n) % 3 == 0).to(torch.int8)
        self.zs = (torch.arange(n) % 6 == 0).to(torch.int8)
        self.logged_pctr = None
        self.requested = None

    def __len__(self):
        return len(self.sample_ids)

    def get_batch(self, indices):
        self.requested = indices
        return {}, self.ys[indices].float(), self.zs[indices].float()


def test_cohort_split_presents_cohort_rows_as_its_own():
    full = _FakeSplit(60)
    cohort = restrict(full, {"fraction": 0.5, "ranks": 4, "block_size": 4, "seed": 1})
    rows = cohort.rows

    assert len(cohort) == 30
    assert torch.equal(cohort.sample_ids, full.sample_ids[rows])
    cohort.get_batch(torch.tensor([0, 5, 29]))
    assert torch.equal(full.requested, rows[[0, 5, 29]])
    assert cohort.census() == {"n_rows": 30, "n_clicks": int(full.ys[rows].sum()),
                               "n_conversions": int(full.zs[rows].sum()),
                               "rank_rows": [8, 8, 7, 7]}

    assert cohort.logged_pctr is None
    full.logged_pctr = torch.linspace(0, 1, 60)
    assert torch.equal(cohort.logged_pctr, full.logged_pctr[rows])
    assert torch.equal(cohort.logged_pctr[torch.tensor([3])], full.logged_pctr[rows[3:4]])


def test_fraction_outside_zero_one_is_rejected():
    with pytest.raises(ValueError):
        cohort_rows(100, 0, ranks=4, block_size=8, seed=0)
    with pytest.raises(ValueError):
        cohort_rows(100, 1.5, ranks=4, block_size=8, seed=0)
