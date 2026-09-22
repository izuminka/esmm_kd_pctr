"""Unit tests for nonclick retention and inverse CTR posterior recovery."""

import pytest
import torch

from training import downsampling
from test_cohorts import _FakeSplit


@pytest.mark.parametrize("rate", [0.05, 0.5, 1.0])
def test_sampling_preserves_clicks_and_alignment(rate):
    full = _FakeSplit(10000)
    full.logged_pctr = torch.linspace(0, 1, len(full))
    spec = {"rate": rate, "seed": 2026}
    kept = downsampling.restrict(full, spec)
    again = downsampling.restrict(full, spec)
    assert torch.equal(kept.sample_ids, again.sample_ids)
    assert set(full.sample_ids[full.ys == 1].tolist()) <= set(kept.sample_ids.tolist())
    assert kept.ys.sum() == full.ys.sum()
    assert kept.zs.sum() == full.zs.sum()
    assert torch.equal(kept.logged_pctr, full.logged_pctr[kept.rows])
    idx = torch.arange(min(20, len(kept)))
    _, y, z = kept.get_batch(idx)
    assert torch.equal(y, full.ys[kept.rows[idx]].float())
    assert torch.equal(z, full.zs[kept.rows[idx]].float())
    if rate == 1:
        assert torch.equal(kept.sample_ids, full.sample_ids)
    else:
        retained_fraction = (kept.ys == 0).sum() / (full.ys == 0).sum()
        assert float(retained_fraction) == pytest.approx(rate, abs=.02)
        other = downsampling.restrict(full, {**spec, "seed": 2027})
        assert not torch.equal(kept.sample_ids, other.sample_ids)


@pytest.mark.parametrize("rate", [0, -0.5, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_retention_rejected(rate):
    with pytest.raises(ValueError, match="retention rate"):
        downsampling.from_config({"model": "esmm", "data": {
            "nonclick_sampling": {"rate": rate, "seed": 2026}}})


@pytest.mark.parametrize("rate", [.05, .5, 1.])
def test_posterior_recovers_known_population(rate):
    # Derive the retained CTR from class counts, independently of the inverse.
    clicked = torch.tensor([0, 1, 10, 50, 100], dtype=torch.float64)
    nonclicked = 100 - clicked
    retained_ctr = clicked / (clicked + rate * nonclicked)
    recovered = downsampling.recover_ctr(retained_ctr, rate)
    torch.testing.assert_close(recovered, clicked / 100)
    assert recovered[0] == 0 and recovered[-1] == 1
    if rate == 1:
        assert recovered is retained_ctr


def test_config_default_and_model_guard():
    assert downsampling.from_config({"model": "esmm"}) is None
    cfg = {"model": "esmm_logged_pctr", "data": {
        "nonclick_sampling": {"rate": .5, "seed": 2026}}}
    with pytest.raises(ValueError, match="plain ESMM"):
        downsampling.from_config(cfg)
