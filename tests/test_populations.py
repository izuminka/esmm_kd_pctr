"""Population routing, and parity between the predictions and metrics-only routes."""

import json
import os

import numpy as np
import pandas as pd

from evaluation.metrics import compute_metrics
from scripts.report_metrics import metric_rows
from training.trainer import write_split


def _esmm_split(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    z = (y * (rng.random(n) < 0.3)).astype(np.int8)  # z <= y
    q = rng.random(n).astype(np.float32)
    v = rng.random(n).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    return ids, y, z, {"q": q, "v": v}


def test_esmm_populations_use_contracted_scores_labels_and_rows():
    _, y, z, scores = _esmm_split()
    rows = {r["population"]: r for r in metric_rows("val", "esmm", y, z, scores)}

    assert set(rows) == {"CTR", "CTCVR", "pCVR"}
    clicked = y == 1
    expected = {
        "CTR": (scores["q"], y),
        "CTCVR": (scores["q"] * scores["v"], z),
        "pCVR": (scores["v"][clicked], z[clicked]),
    }
    for population, (score, label) in expected.items():
        row = rows[population]
        assert row["split"] == "val"
        assert row["rows"] == len(label)
        assert row["positives"] == int(label.sum())
        for k, value in compute_metrics(score, label).items():
            assert row[k] == value
    assert rows["pCVR"]["rows"] == int(clicked.sum())


def test_single_head_model_reports_ctr_only():
    _, y, z, scores = _esmm_split()
    rows = metric_rows("test", "dcn", y, z, {"score": scores["q"]})
    assert [r["population"] for r in rows] == ["CTR"]


def test_metrics_only_route_matches_scoring_the_parquet(tmp_path):
    ids, y, z, scores = _esmm_split()
    # Float64 scores and int64 labels, as the Impression path produces: the
    # metrics-only route must cast them exactly as the parquet writer does.
    loose = {c: s.astype(np.float64) for c, s in scores.items()}

    with_preds, without = str(tmp_path / "with"), str(tmp_path / "without")
    parquet = write_split(with_preds, "test", "esmm", ids, y.astype(np.int64),
                          z.astype(np.int64), loose, save_predictions=True)
    scored = write_split(without, "test", "esmm", ids, y.astype(np.int64),
                         z.astype(np.int64), loose, save_predictions=False)

    assert parquet == os.path.join(with_preds, "predictions", "test.parquet")
    assert scored == os.path.join(without, "metrics", "test.json")
    assert not os.path.exists(os.path.join(without, "predictions"))

    df = pd.read_parquet(parquet)
    from_parquet = metric_rows("test", "esmm", df["y"].to_numpy(), df["z"].to_numpy(),
                               {"q": df["q"].to_numpy(), "v": df["v"].to_numpy()})
    with open(scored) as f:
        assert json.load(f) == json.loads(json.dumps(from_parquet))
