import torch

from evaluation.metrics import compute_metrics


def evaluate_dcn(model, batch, y: torch.Tensor) -> dict:
    """CTR evaluation for the standalone DCN model.

    Returns {"ctr": compute_metrics(score, y)}, matching evaluate_esmm's shape
    so callers can treat both uniformly.  Works with CPU or CUDA tensors.
    """
    model.eval()
    with torch.no_grad():
        logit = model(batch)
    score = torch.sigmoid(logit).cpu().numpy()
    return {"ctr": compute_metrics(score, y.cpu().numpy())}
