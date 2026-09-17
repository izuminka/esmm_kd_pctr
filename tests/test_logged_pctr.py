"""Logged pCTR: loading scores from a DCN run, and the CTCVR-logged population.

The loader tests build a tiny DCN run folder on disk. The real-data test aligns
the pinned DCN run's exports against the encoded splits, when both are present.
"""

import json
import os
import types

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data.logged_pctr import attach_logged_pctr, load_logged_pctr
from evaluation.metrics import compute_metrics
from scripts.report_metrics import _attach_logged_pctr_source, metric_rows

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(CODE_DIR)

IDS = np.array([10, 11, 12, 13, 14], dtype=np.int64)
YS = np.array([1, 0, 0, 1, 0], dtype=np.int8)
SCORES = np.array([0.5, 0.1, 0.2, 0.7, 0.05], dtype=np.float32)


def _dcn_run(root, rows=None, run=None, name="r1"):
    """A finished DCN run folder under root/artifacts/dcn/<name>, exporting `rows` for every split."""
    run_dir = root / "artifacts" / "dcn" / name
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(run or {
        "run": name, "model": "dcn", "status": "FINISHED", "git": {"commit": "abc123"},
        "selected_on": {"metric": "ctr_prauc", "split": "val", "value": 0.08, "epoch": 1}}))
    ids, ys, scores = rows or (IDS, YS, SCORES)
    for split in ("train", "val", "test"):
        pd.DataFrame({"sample_id": ids, "y": ys, "z": np.zeros_like(ys), "score": scores}).to_parquet(
            run_dir / "predictions" / f"{split}.parquet", index=False)
    return {"dcn_run": f"artifacts/dcn/{name}",
            "predictions": {s: f"artifacts/dcn/{name}/predictions/{s}.parquet"
                            for s in ("train", "val", "test")},
            "score_column": "score"}


def _load(block, root, split="train", ids=IDS, ys=YS):
    return load_logged_pctr(block, str(root), split, torch.from_numpy(ids), torch.from_numpy(ys))


# --- alignment ---


def test_scores_are_aligned_by_sample_id_not_row_order(tmp_path):
    order = np.array([3, 0, 4, 1, 2])
    block = _dcn_run(tmp_path, rows=(IDS[order], YS[order], SCORES[order]))

    p, _ = _load(block, tmp_path)

    assert p.dtype == torch.float32
    assert torch.equal(p, torch.from_numpy(SCORES))


def test_provenance_names_the_run_and_file(tmp_path):
    block = _dcn_run(tmp_path)

    _, source = _load(block, tmp_path, split="val")

    path = tmp_path / "artifacts" / "dcn" / "r1" / "predictions" / "val.parquet"
    assert source["dcn_run"] == "artifacts/dcn/r1"
    assert source["run_id"] == "r1" and source["commit"] == "abc123"
    assert source["selected_on"]["metric"] == "ctr_prauc"
    assert source["score_column"] == "score"
    assert source["files"] == {"val": {"file": "artifacts/dcn/r1/predictions/val.parquet",
                                       "rows": 5, "bytes": os.path.getsize(path),
                                       "mtime": source["files"]["val"]["mtime"]}}
    assert source["files"]["val"]["mtime"].endswith("Z")


@pytest.mark.parametrize("rows,match", [
    ((np.array([10, 11, 12, 13, 99]), YS, SCORES), "missing sample_ids"),
    ((np.array([10, 11, 12, 13, 13]), YS, SCORES), "repeats a sample_id"),
    ((IDS, np.array([1, 0, 0, 1, 1], dtype=np.int8), SCORES), "click labels disagree"),
    ((IDS[:4], YS[:4], SCORES[:4]), "has 4 rows"),
])
def test_misaligned_exports_are_refused(tmp_path, rows, match):
    block = _dcn_run(tmp_path, rows=rows)
    with pytest.raises(ValueError, match=match):
        _load(block, tmp_path)


@pytest.mark.parametrize("edit,match", [
    (lambda b: {**b, "dcn_run": "artifacts/dcn/latest"}, "concrete DCN run"),
    (lambda b: {**b, "dcn_run": "artifacts/dcn/nope"}, "not a run folder"),
    (lambda b: {**b, "predictions": {"val": b["predictions"]["val"]}}, "no file for split 'train'"),
    (lambda b: {**b, "predictions": {**b["predictions"], "train": "artifacts/dcn/r2/predictions/train.parquet"}},
     "outside dcn_run"),
    (lambda b: {**b, "score_column": "logit"}, "no score column 'logit'"),
    (lambda b: {k: v for k, v in b.items() if k != "score_column"}, "score_column is required"),
])
def test_ambiguous_or_wrong_sources_are_refused(tmp_path, edit, match):
    block = _dcn_run(tmp_path)
    _dcn_run(tmp_path, name="r2")  # a second, valid run the block must not wander into
    with pytest.raises(ValueError, match=match):
        _load(edit(block), tmp_path)


@pytest.mark.parametrize("run", [
    {"run": "r1", "model": "dcn", "status": "RUNNING"},
    {"run": "r1", "model": "esmm", "status": "FINISHED"},
])
def test_only_a_finished_dcn_run_is_a_source(tmp_path, run):
    block = _dcn_run(tmp_path, run=run)
    with pytest.raises(ValueError, match="finished dcn run"):
        _load(block, tmp_path)


def test_attach_fills_loaded_splits_and_merges_their_files(tmp_path):
    block = _dcn_run(tmp_path)
    split = lambda: types.SimpleNamespace(sample_ids=torch.from_numpy(IDS),
                                          ys=torch.from_numpy(YS), logged_pctr=None)
    splits = {"train": split(), "val": split(), "test": None}

    source = attach_logged_pctr(block, str(tmp_path), splits)

    assert torch.equal(splits["train"].logged_pctr, torch.from_numpy(SCORES))
    assert torch.equal(splits["val"].logged_pctr, torch.from_numpy(SCORES))
    assert set(source["files"]) == {"train", "val"}


# --- the CTCVR-logged population ---


def _esmm_scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    z = (y * (rng.random(n) < 0.3)).astype(np.int8)
    return y, z, {c: rng.random(n).astype(np.float32) for c in ("q", "v", "p")}


@pytest.mark.parametrize("model", ["esmm", "esmm_logged_pctr"])
def test_ctcvr_logged_scores_p_times_v_for_every_esmm_model(model):
    y, z, scores = _esmm_scores()
    rows = {r["population"]: r for r in metric_rows("test", model, y, z, scores)}

    assert list(rows) == ["CTR", "CTCVR", "CTCVR-logged", "pCVR"]
    expected = compute_metrics(scores["p"] * scores["v"], z)
    assert {k: rows["CTCVR-logged"][k] for k in expected} == expected
    assert rows["CTCVR-logged"]["rows"] == len(z)
    assert rows["CTCVR-logged"]["positives"] == int(z.sum())
    assert rows["CTCVR"]["pr_auc"] != rows["CTCVR-logged"]["pr_auc"]


def test_without_p_the_esmm_populations_are_unchanged():
    y, z, scores = _esmm_scores()
    del scores["p"]
    assert [r["population"] for r in metric_rows("val", "esmm", y, z, scores)] == ["CTR", "CTCVR", "pCVR"]


def test_without_q_only_the_q_free_populations_are_reported():
    y, z, scores = _esmm_scores()
    del scores["q"]
    rows = metric_rows("test", "esmm_logged_pctr", y, z, scores)
    assert [r["population"] for r in rows] == ["CTCVR-logged", "pCVR"]


def test_every_ctcvr_logged_row_names_its_source():
    y, z, scores = _esmm_scores()
    rows = metric_rows("val", "esmm", y, z, scores) + metric_rows("test", "esmm", y, z, scores)
    source = {"dcn_run": "artifacts/dcn/r1",
              "files": {"val": {"file": "artifacts/dcn/r1/predictions/val.parquet"},
                        "test": {"file": "artifacts/dcn/r1/predictions/test.parquet"}}}

    _attach_logged_pctr_source(rows, source)

    logged = [r for r in rows if r["population"] == "CTCVR-logged"]
    assert [(r["split"], r["logged_pctr_run"], r["logged_pctr_file"]) for r in logged] == [
        ("val", "artifacts/dcn/r1", "artifacts/dcn/r1/predictions/val.parquet"),
        ("test", "artifacts/dcn/r1", "artifacts/dcn/r1/predictions/test.parquet")]
    assert all("logged_pctr_run" not in r for r in rows if r["population"] != "CTCVR-logged")


def test_a_ctcvr_logged_row_without_a_recorded_source_is_refused():
    y, z, scores = _esmm_scores()
    rows = metric_rows("test", "esmm", y, z, scores)
    with pytest.raises(SystemExit, match="no logged_pctr source"):
        _attach_logged_pctr_source(rows, {"dcn_run": "artifacts/dcn/r1", "files": {"val": {}}})


# --- which score a checkpoint metric selects on ---


def _trainer(model, checkpoint_metric, tmp_path):
    from training.trainer import Trainer
    return Trainer({"model": model,
                    "training": {"epochs": 1, "batch_size": 8, "eval_batch_size": 8, "lr": 1e-3,
                                 "grad_clip_norm": 0, "shuffle_block_size": 1, "gpus": 0,
                                 "checkpoint_metric": checkpoint_metric, "out_dir": str(tmp_path)}},
                   str(tmp_path))


def test_selection_uses_the_native_product(tmp_path):
    scores = {"q": torch.tensor([0.4]), "v": torch.tensor([0.5]), "p": torch.tensor([0.2])}
    y, z = torch.tensor([1.0]), torch.tensor([1.0])

    score, label = _trainer("esmm_logged_pctr", "ctcvr_prauc", tmp_path)._selection_inputs(scores, y, z)

    assert torch.equal(score, scores["q"] * scores["v"]) and torch.equal(label, y * z)


def test_selecting_on_a_score_the_model_does_not_report_says_so(tmp_path):
    """The product objective has no q, so the native product cannot select it."""
    scores = {"v": torch.tensor([0.5]), "p": torch.tensor([0.2])}
    trainer = _trainer("esmm_logged_pctr", "ctcvr_prauc", tmp_path)

    with pytest.raises(ValueError, match="needs the 'q' score"):
        trainer._selection_inputs(scores, torch.tensor([1.0]), torch.tensor([1.0]))


# --- evaluate_esmm's optional logged CTCVR ---


def _tiny_esmm_batch():
    from models.esmm import ESMM
    torch.manual_seed(0)
    model = ESMM({"f1": 8}, emb_dim=4, tower_dims=(4, 3, 2))
    batch = {"f1": (torch.arange(1, 9), torch.ones(8), torch.arange(8))}  # one token per row
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    z = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    p = torch.tensor([0.5, 0.8, 0.7, 0.1, 0.3, 0.02, 0.6, 0.4])
    return model, batch, y, z, p


def test_evaluate_esmm_reports_logged_ctcvr_by_default():
    from evaluation.esmm import evaluate_esmm
    model, batch, y, z, p = _tiny_esmm_batch()

    result = evaluate_esmm(model, batch, y, z, p)

    assert set(result) == {"ctr", "ctcvr", "pcvr", "ctcvr_logged"}
    with torch.no_grad():
        _, v = model.eval()(batch)
    assert result["ctcvr_logged"] == compute_metrics(p.numpy() * v.numpy(), (y * z).numpy())


def test_evaluate_esmm_logged_ctcvr_can_be_turned_off():
    from evaluation.esmm import evaluate_esmm
    model, batch, y, z, _ = _tiny_esmm_batch()

    assert set(evaluate_esmm(model, batch, y, z, is_logged_ctcvr=False)) == {"ctr", "ctcvr", "pcvr"}


def test_evaluate_esmm_refuses_logged_ctcvr_without_p():
    from evaluation.esmm import evaluate_esmm
    model, batch, y, z, _ = _tiny_esmm_batch()

    with pytest.raises(ValueError, match="needs the logged pCTR"):
        evaluate_esmm(model, batch, y, z)


# --- the pinned DCN run against the real encoded splits ---


def _real_block():
    with open(os.path.join(CODE_DIR, "configs", "esmm.yaml")) as f:
        block = yaml.safe_load(f)["logged_pctr"]
    # Exercise the loader independently of which splits the model consumes.
    folder = os.path.dirname(block["predictions"]["test"])
    block["predictions"] = {s: os.path.join(folder, f"{s}.parquet") for s in ("train", "val", "test")}
    return block


def _real_split(split, device):
    enc = os.path.join(ROOT_DIR, "ali-ccp", "encoded", split)
    return (torch.from_numpy(np.load(os.path.join(enc, "sample_id.npy"))).to(device),
            torch.from_numpy(np.load(os.path.join(enc, "y.npy"))).to(device))


def _real_data_present():
    block = _real_block()
    return all(os.path.exists(os.path.join(ROOT_DIR, block["predictions"][s]))
               and os.path.exists(os.path.join(ROOT_DIR, "ali-ccp", "encoded", s, "sample_id.npy"))
               for s in ("train", "val", "test"))


@pytest.mark.skipif(not _real_data_present(), reason="pinned DCN run or encoded splits not present")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_pinned_dcn_scores_align_with_every_encoded_row(split):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block = _real_block()
    sample_ids, ys = _real_split(split, device)

    p, source = load_logged_pctr(block, ROOT_DIR, split, sample_ids, ys)

    assert p.device == sample_ids.device and len(p) == len(sample_ids)
    assert bool(((p >= 0) & (p <= 1)).all())
    assert source["files"][split]["rows"] == len(sample_ids)
    # Independent spot check: look sampled rows up in the parquet by id with pandas.
    df = pd.read_parquet(os.path.join(ROOT_DIR, block["predictions"][split]),
                         columns=["sample_id", "score"]).set_index("sample_id")
    rows = np.random.default_rng(0).choice(len(sample_ids), size=10_000, replace=False)
    ids = sample_ids.cpu().numpy()[rows]
    np.testing.assert_array_equal(p.cpu().numpy()[rows], df.loc[ids, "score"].to_numpy())


@pytest.mark.skipif(not _real_data_present(), reason="pinned DCN run or encoded splits not present")
def test_shuffled_real_export_gives_identical_scores(tmp_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block = _real_block()
    sample_ids, ys = _real_split("val", device)
    in_order, _ = load_logged_pctr(block, ROOT_DIR, "val", sample_ids, ys)

    df = pd.read_parquet(os.path.join(ROOT_DIR, block["predictions"]["val"]))
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert not np.array_equal(shuffled["sample_id"].to_numpy(), df["sample_id"].to_numpy())
    copy = _dcn_run(tmp_path, rows=(shuffled["sample_id"].to_numpy(), shuffled["y"].to_numpy(),
                                    shuffled["score"].to_numpy()))

    from_shuffled, _ = load_logged_pctr(copy, str(tmp_path), "val", sample_ids, ys)

    assert torch.equal(from_shuffled, in_order)
