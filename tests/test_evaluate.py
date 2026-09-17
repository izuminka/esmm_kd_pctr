import torch

from data.raw import Impression
from data.vocab import build_vocabs, encode_batch
from evaluation.esmm import evaluate_esmm
from evaluation.metrics import compute_metrics
from models.esmm import ESMM


def _make_synthetic_batch():
    # 4 clicked rows (mixed z: 1,0,0,1) + 2 unclicked rows (y=0,z=0)
    labels = [(1, 1), (1, 0), (1, 0), (1, 1), (0, 0), (0, 0)]
    rows = [Impression(i, y, z, f"c{i}", {"f": [(i, 1.0)]}) for i, (y, z) in enumerate(labels)]
    vocabs = build_vocabs(rows, ["f"])
    encoded = encode_batch(rows, ["f"], vocabs)
    y = torch.tensor([r.y for r in rows], dtype=torch.float32)
    z = torch.tensor([r.z for r in rows], dtype=torch.float32)
    vocab_sizes = {f: len(v) for f, v in vocabs.items()}
    return encoded, y, z, vocab_sizes


def test_population_routing_is_correct():
    encoded, y, z, vocab_sizes = _make_synthetic_batch()
    torch.manual_seed(0)
    # single field with emb_dim=460 so the concatenated input matches the
    # tower's fixed 460-dim input without needing all 23 real fields
    model = ESMM(vocab_sizes, emb_dim=460)

    result = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False)

    assert set(result.keys()) == {"ctr", "ctcvr", "pcvr"}
    for population in result.values():
        assert set(population.keys()) == {"auc", "pr_auc", "ne", "ce"}

    # Independent cross-check: run the model again ourselves and slice by hand,
    # rather than trusting evaluate_esmm's internal masking.
    model.eval()
    with torch.no_grad():
        q, v = model(encoded)
    q_np, v_np, y_np, z_np = q.numpy(), v.numpy(), y.numpy(), z.numpy()
    clicked = y_np == 1
    assert clicked.sum() == 4

    assert result["ctr"] == compute_metrics(q_np, y_np)
    assert result["ctcvr"] == compute_metrics(q_np * v_np, y_np * z_np)
    assert result["pcvr"] == compute_metrics(v_np[clicked], z_np[clicked])


def test_ctcvr_label_is_y_times_z_not_z_alone():
    # A row with y=0, z=1 violates the papers z<=y constraint and should never
    # occur in cleaned Ali-CCP data (data.raw drops such rows on parse), but
    # evaluate_esmm should compute the CTCVR label correctly on its own terms
    # (y*z) regardless of whether that invariant was enforced upstream.
    labels = [(1, 1), (0, 1), (1, 0)]  # row 2 is the deliberately invalid y=0,z=1 case
    rows = [Impression(i, y, z, f"c{i}", {"f": [(i, 1.0)]}) for i, (y, z) in enumerate(labels)]
    vocabs = build_vocabs(rows, ["f"])
    encoded = encode_batch(rows, ["f"], vocabs)
    y = torch.tensor([r.y for r in rows], dtype=torch.float32)
    z = torch.tensor([r.z for r in rows], dtype=torch.float32)
    vocab_sizes = {f: len(v) for f, v in vocabs.items()}
    torch.manual_seed(0)
    model = ESMM(vocab_sizes, emb_dim=460)

    result = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False)

    model.eval()
    with torch.no_grad():
        q, v = model(encoded)
    q_np, v_np, y_np, z_np = q.numpy(), v.numpy(), y.numpy(), z.numpy()

    # If CTCVR used z alone, its label would be [1, 1, 0]; using y*z it is [1, 0, 0].
    assert (y_np * z_np).tolist() == [1.0, 0.0, 0.0]
    assert result["ctcvr"] == compute_metrics(q_np * v_np, y_np * z_np)
    assert result["ctcvr"] != compute_metrics(q_np * v_np, z_np)


def test_evaluate_esmm_on_real_data(real_batch):
    encoded, y, z, vocab_sizes = real_batch
    model = ESMM(vocab_sizes)

    result = evaluate_esmm(model, encoded, y, z, is_logged_ctcvr=False)

    n_clicked = int((y == 1).sum().item())
    assert n_clicked > 0

    for population in result.values():
        assert 0.0 <= population["auc"] <= 1.0
        assert 0.0 <= population["pr_auc"] <= 1.0
        assert population["ne"] >= 0
        assert population["ce"] >= 0
