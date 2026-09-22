"""Scores an ESMM model against the papers reported populations.

Evaluation contract (Appendix, tab:aliccp-evaluation-contracts):
  CTR          -- score q,     label y,   all impressions
  CTCVR        -- score q * v, label y*z, all impressions
  clicked pCVR -- score v,     label z,   only impressions where y == 1
  CTCVR logged -- score p * v, label y*z, all impressions (optional)

CTCVR logged scores the product served with the frozen logged pCTR p
(tab:aliccp-quarter-logged-pctr). It is on by default and needs p; plain ESMM
turns it off. A model with no auxiliary CTR head (q is None) omits CTR and CTCVR.

The CTCVR label is y*z (Eq. method-esmm), not z alone. z<=y should already
hold after data.raw's invalid-row filtering, making y*z == z, but computing
y*z here keeps CTCVR correct on its own terms rather than depending on that
invariant having been enforced upstream.
"""

import torch

from evaluation.metrics import compute_metrics
from training.downsampling import recover_ctr, validate_rate


def evaluate_esmm(
    model,                              # an ESMM whose forward returns (q, v)
    batch,                              # encoded features, as from data.vocab.encode_batch
    y: torch.Tensor,                    # click label
    z: torch.Tensor,                    # conversion label; y*z is the CTCVR label
    p: torch.Tensor | None = None,      # frozen logged pCTR, needed for logged CTCVR
    is_logged_ctcvr: bool = True,       # also score the logged product p * v
    nonclick_rate: float = 1.0,         # training retention; evaluation is full-rate
) -> dict:
    nonclick_rate = validate_rate(nonclick_rate)
    if is_logged_ctcvr and p is None:
        raise ValueError("logged CTCVR needs the logged pCTR p; pass p, or is_logged_ctcvr=False")

    model.eval()
    with torch.no_grad():
        q, v = model(batch)
        if q is not None and nonclick_rate < 1.0:
            q = recover_ctr(q, nonclick_rate)

    v_np = v.numpy()
    y_np = y.numpy()
    z_np = z.numpy()
    clicked = y_np == 1

    results = {}
    if q is not None:
        q_np = q.numpy()
        results["ctr"] = compute_metrics(q_np, y_np)
        results["ctcvr"] = compute_metrics(q_np * v_np, y_np * z_np)
    results["pcvr"] = compute_metrics(v_np[clicked], z_np[clicked])
    if is_logged_ctcvr:
        results["ctcvr_logged"] = compute_metrics(p.numpy() * v_np, y_np * z_np)
    return results
