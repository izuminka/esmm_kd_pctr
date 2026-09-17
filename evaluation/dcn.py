import torch

from evaluation.metrics import compute_metrics


def evaluate_dcn(
    model,            # a DCN-V2 whose forward returns one logit per row
    batch,            # encoded features, as from data.vocab.encode_batch
    y: torch.Tensor,  # click label
) -> dict:
    """CTR evaluation for the standalone DCN model.

    Returns {"ctr": compute_metrics(score, y)}, matching evaluate_esmm's shape
    so callers can treat both uniformly.  Works with CPU or CUDA tensors.
    """
    model.eval()
    with torch.no_grad():
        logit = model(batch)
    score = torch.sigmoid(logit).cpu().numpy()
    return {"ctr": compute_metrics(score, y.cpu().numpy())}
