"""Nonclick retention and inverse CTR posterior shared by train and inference."""

import math

import numpy as np

from training.cohorts import CohortSplit


def validate_rate(rate):
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or not 0 < rate <= 1):
        raise ValueError("nonclick retention rate must be finite and in (0, 1]")
    return float(rate)


def from_config(cfg):
    """One opt-in contract; existing configs have no sampling or correction."""
    spec = cfg.get("data", {}).get("nonclick_sampling")
    if spec is None:
        return None
    if cfg["model"] != "esmm":
        raise ValueError("nonclick_sampling currently supports plain ESMM only")
    if set(spec) != {"rate", "seed"}:
        raise ValueError("nonclick_sampling requires exactly rate and seed")
    seed = spec["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("nonclick_sampling seed must be a nonnegative integer")
    return {"rate": validate_rate(spec["rate"]), "seed": seed}


def restrict(split, spec):
    """Reuse the cohort view; membership is fixed by encoded row order and seed."""
    rate = validate_rate(spec["rate"])
    clicks = split.ys.cpu().numpy() == 1
    keep = clicks | (np.random.default_rng(spec["seed"]).random(len(split)) < rate)
    rows = np.flatnonzero(keep)
    if not len(rows):
        raise ValueError("nonclick sampling retained no training rows")
    return CohortSplit(split, rows, [len(rows)])


def recover_ctr(q, rate=1.0):
    """Invert click-based sampling on raw CTR only; pCVR stays unchanged.

    The full-population product is recover_ctr(q_R) * v, never recovery of
    q_R * v. Endpoints are exact; rate=1 leaves existing scores unchanged.
    """
    rate = validate_rate(rate)
    if rate == 1:
        return q
    return rate * q / ((1 - q) + rate * q)
