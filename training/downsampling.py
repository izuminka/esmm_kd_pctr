"""Nonclick retention and inverse CTR posterior shared by train and inference."""

import math

import numpy as np
import torch

from training.cohorts import CohortSplit

# CTR space by split: align the KD-tau teacher for training; recover student
# CTR for full-population validation/test. Callers pass the selected boolean.
is_ctr_downsampled_space = {"train": True, "val": False, "test": False}

DEFAULT_NONCLICK_RATE = 1.0
NONCLICK_SAMPLING_KEY = "nonclick_sampling"


def validate_rate(rate):
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or not 0 < rate <= 1):
        raise ValueError("nonclick retention rate must be finite and in (0, 1]")
    return float(rate)


def _validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"rate", "seed"}:
        raise ValueError("nonclick_sampling requires exactly rate and seed")
    seed = spec["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("nonclick_sampling seed must be a nonnegative integer")
    return {"rate": validate_rate(spec["rate"]), "seed": seed}


def from_config(cfg):
    """One opt-in contract; existing configs have no sampling or correction."""
    spec = cfg.get("data", {}).get(NONCLICK_SAMPLING_KEY)
    if spec is None:
        return None
    if cfg["model"] != "esmm" and not (
            cfg["model"] == "esmm_logged_pctr"
            and cfg.get("model_params", {}).get("objective") == "kd_tau"):
        raise ValueError("nonclick_sampling supports plain ESMM or logged ESMM with objective kd_tau only")
    return _validate_spec(spec)


def training_model_params(cfg):
    """Derive KD-tau's constructor rate from the single sampling config.

    Other objectives keep their parameters unchanged. Checkpoint reconstruction
    uses saved constructor parameters directly, rather than this config adapter.
    """
    params = dict(cfg.get("model_params", {}))
    if cfg["model"] == "esmm_logged_pctr" and params.get("objective") == "kd_tau":
        if "nonclick_rate" in params:
            raise ValueError("configure retention only in data.nonclick_sampling")
        sampling = from_config(cfg)
        params["nonclick_rate"] = sampling["rate"] if sampling else DEFAULT_NONCLICK_RATE
    return params


def restrict(split, spec):
    """Reuse the cohort view; membership is fixed by encoded row order and seed."""
    spec = _validate_spec(spec)
    rate = spec["rate"]
    clicks = split.ys.cpu().numpy() == 1
    keep = clicks | (np.random.default_rng(spec["seed"]).random(len(split)) < rate)
    rows = np.flatnonzero(keep)
    if not len(rows):
        raise ValueError("nonclick sampling retained no training rows")
    return CohortSplit(split, rows, [len(rows)])


def recover_ctr(q, rate=DEFAULT_NONCLICK_RATE):
    """Invert click-based sampling on raw CTR only; pCVR stays unchanged.

    The full-population product is recover_ctr(q_R) * v, never recovery of
    q_R * v. Endpoints are exact; rate=1 leaves existing scores unchanged.
    """
    rate = validate_rate(rate)
    if rate == 1:
        return q
    # Avoid underflow of the rate before division: q=1 would otherwise yield
    # 0/0, and near-one scores can lose representable results. Promote only
    # these extreme rates, then restore the original score dtype/device.
    if (isinstance(q, torch.Tensor) and q.is_floating_point()
            and q.dtype != torch.float64 and rate < torch.finfo(q.dtype).tiny):
        return recover_ctr(q.double(), rate).to(q.dtype)
    if (isinstance(q, np.ndarray) and np.issubdtype(q.dtype, np.floating)
            and q.dtype != np.float64 and rate < np.finfo(q.dtype).tiny):
        return recover_ctr(q.astype(np.float64), rate).astype(q.dtype)
    return rate * q / ((1 - q) + rate * q)


def retained_ctr(p, rate=DEFAULT_NONCLICK_RATE):
    """Map full-population teacher pCTR to the retained population (tau)."""
    rate = validate_rate(rate)
    if rate == 1:
        return p
    if p.dtype != torch.float64 and rate < torch.finfo(p.dtype).tiny:
        return retained_ctr(p.double(), rate).to(p.dtype)
    return p / (p + rate * (1 - p))


def logged_ctr_for_space(p, rate, *, is_ctr_downsampled_space):
    """Original-space logged CTR -> requested space; false is an exact bypass."""
    rate = validate_rate(rate)
    if not is_ctr_downsampled_space or rate == 1:
        return p
    return retained_ctr(p, rate)


def model_ctr_for_space(q, rate, *, is_ctr_downsampled_space):
    """Retained-space model CTR -> requested space; true is an exact bypass."""
    rate = validate_rate(rate)
    if q is None or is_ctr_downsampled_space or rate == 1:
        return q
    return recover_ctr(q, rate)
