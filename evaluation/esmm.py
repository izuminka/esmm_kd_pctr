"""Scores an ESMM model against the papers three reported populations.

Evaluation contract (Appendix, tab:aliccp-evaluation-contracts):
  CTR         -- score q,     label y,   all impressions
  CTCVR       -- score q * v, label y*z, all impressions
  clicked pCVR -- score v,    label z,   only impressions where y == 1

The CTCVR label is y*z (Eq. method-esmm), not z alone. z<=y should already
hold after data.raw's invalid-row filtering, making y*z == z, but computing
y*z here keeps CTCVR correct on its own terms rather than depending on that
invariant having been enforced upstream.
"""

import torch

from metrics import compute_metrics


def evaluate_esmm(model, batch, y: torch.Tensor, z: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        q, v = model(batch)

    q_np = q.numpy()
    v_np = v.numpy()
    y_np = y.numpy()
    z_np = z.numpy()
    clicked = y_np == 1

    return {
        "ctr": compute_metrics(q_np, y_np),
        "ctcvr": compute_metrics(q_np * v_np, y_np * z_np),
        "pcvr": compute_metrics(v_np[clicked], z_np[clicked]),
    }