import math

import numpy as np
import pytest

from evaluation.metrics import auc, calibration_ratio, compute_metrics, normalized_entropy, pr_auc


# --- auc ---


def test_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    assert auc(scores, labels) == pytest.approx(1.0)


def test_auc_tied_scores_gives_half_credit():
    # one positive and one negative tied at the same score: neither ranks above
    # the other, so the paper's "tied pairs get half credit" rule gives AUC=0.5
    assert auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


# --- pr_auc ---


def test_pr_auc_hand_computed():
    # sorted by score desc: 0.9(pos,rank1) 0.5(neg,rank2) 0.3(pos,rank3)
    # precision at each positive's rank: 1/1=1.0, 2/3
    # AP = average of those two = 5/6
    scores = [0.9, 0.5, 0.3]
    labels = [1, 0, 1]

    assert pr_auc(scores, labels) == pytest.approx(5 / 6, rel=1e-6)


# --- normalized_entropy ---


def test_normalized_entropy_hand_computed():
    labels = [1, 0, 1, 0]
    scores = [0.7, 0.2, 0.6, 0.3]

    ne = normalized_entropy(scores, labels)

    def bce(y, s):
        return -(y * math.log(s) + (1 - y) * math.log(1 - s))

    mean_bce = np.mean([bce(y, s) for y, s in zip(labels, scores)])
    p = np.mean(labels)
    entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert ne == pytest.approx(mean_bce / entropy, rel=1e-6)


def test_normalized_entropy_equals_one_for_base_rate_prediction():
    # predicting the constant empirical positive rate for every row makes mean
    # BCE exactly equal to that rate's own entropy, so NE = 1.0 -- independent
    # of the hand-computed arithmetic check above.
    labels = [1, 0, 1, 0, 0]
    base_rate = np.mean(labels)
    scores = [base_rate] * len(labels)

    assert normalized_entropy(scores, labels) == pytest.approx(1.0, rel=1e-6)


# --- calibration_ratio ---


def test_calibration_ratio_hand_computed():
    assert calibration_ratio([0.5, 0.3, 0.8], [1, 0, 1]) == pytest.approx(0.8)


def test_calibration_ratio_is_one_when_predictions_equal_labels():
    labels = [1, 0, 1, 1, 0]

    assert calibration_ratio(labels, labels) == pytest.approx(1.0)


# --- compute_metrics ---


def test_compute_metrics_returns_all_four():
    scores = [0.1, 0.9, 0.4, 0.6]
    labels = [0, 1, 0, 1]

    result = compute_metrics(scores, labels)

    assert set(result.keys()) == {"auc", "pr_auc", "ne", "ce"}
    assert result["auc"] == pytest.approx(auc(scores, labels))
    assert result["pr_auc"] == pytest.approx(pr_auc(scores, labels))
    assert result["ne"] == pytest.approx(normalized_entropy(scores, labels))
    assert result["ce"] == pytest.approx(calibration_ratio(scores, labels))
