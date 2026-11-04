"""Starting a prediction head at a base rate instead of at 0.5.

A config's model_params.output_bias_init maps each head to its rate on the
training rows: "ctr" is P(click) and "cvr" is P(conversion | click). Only the
final bias is set, which consumes no random draws, so every other weight starts
exactly as it would without it.
"""

import math


def validate_output_bias_init(spec: dict | None, heads: tuple) -> dict | None:
    """Check `spec` maps only names in `heads`, each to a rate in (0, 1)."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or not spec or not set(spec) <= set(heads):
        raise ValueError(f"output_bias_init must map a subset of {heads} to rates, got {spec!r}")
    for head, rate in spec.items():
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate < 1:
            raise ValueError(f"output_bias_init[{head!r}] must lie in (0, 1), got {rate!r}")
    return {head: float(rate) for head, rate in spec.items()}


def log_odds(rate: float) -> float:
    return math.log(rate) - math.log1p(-rate)
