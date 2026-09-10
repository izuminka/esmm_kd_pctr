"""Ali-CCP evaluation metrics: AUC, PR-AUC, normalized entropy (NE), and
calibration ratio (CE), matching the papers Evaluation recipe.

Population-agnostic: pass whatever score/label subset a population needs
(e.g. all impressions for CTCVR, clicked-only for pCVR) -- these functions
just operate on flat arrays. Callers on torch tensors should
`.detach().cpu().numpy()` first; this module has no torch dependency.
"""

import numpy as np
from scipy.stats import bernoulli
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

EPS = 1e-7


def auc(scores, labels) -> float:
    return roc_auc_score(labels, scores)


def pr_auc(scores, labels) -> float:
    return average_precision_score(labels, scores)


def normalized_entropy(scores, labels) -> float:
    bce = log_loss(labels, scores, labels=[0, 1])
    p = np.clip(np.mean(labels), EPS, 1 - EPS)
    return bce / bernoulli(p).entropy()


def calibration_ratio(scores, labels) -> float:
    return float(np.sum(scores) / np.sum(labels))


def compute_metrics(scores, labels) -> dict:
    return {
        "auc": auc(scores, labels),
        "pr_auc": pr_auc(scores, labels),
        "ne": normalized_entropy(scores, labels),
        "ce": calibration_ratio(scores, labels),
    }
