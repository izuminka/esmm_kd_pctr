"""Canonical ESMM behavior with DCN replacing only its CTR tower."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from models.dcn import DCNv2, _normalized_weights_and_bounded_stats
from models.esmm import EPS, ESMM, esmm_loss
from models.esmm_dcn import ESMMDCN


VOCAB = {"2": 4, "1": 5}
PARAMS = {
    "emb_dim": 2,
    "tower_dims": (4, 8, 2),
    "aux_weight": 1.0,
    "dcn_params": {
        "emb_dim": 2, "cross_rank": 3, "num_cross_layers": 2,
        "deep_hidden_dims": (8, 4), "dropout": 0.3,
        "pooling": "mean", "bag_stats": True,
    },
}


def _batch():
    return {
        "1": (
            torch.tensor([1, 2, 3, 4, 1, 5]),
            torch.tensor([1.0, 2.0, 0.5, 1.0, 3.0, 1.0]),
            torch.tensor([0, 2, 3]),
        ),
        "2": (
            torch.tensor([2, 1, 3, 4]),
            torch.tensor([1.0, 2.0, 1.5, 0.5]),
            torch.tensor([0, 1, 3]),
        ),
    }


def _model():
    torch.manual_seed(7)
    return ESMMDCN(VOCAB, **PARAMS)


def test_only_ctr_tower_changes_from_vanilla_esmm_and_rng_is_preserved():
    model = _model()
    actual_rng = torch.get_rng_state()
    torch.manual_seed(7)
    vanilla = ESMM(VOCAB, **{k: v for k, v in PARAMS.items() if k != "dcn_params"})
    expected_rng = torch.get_rng_state()

    assert isinstance(model, ESMM)
    assert isinstance(model.ctr_tower, DCNv2)
    assert set(model._modules) == set(vanilla._modules)
    assert model.field_ids == vanilla.field_ids
    assert model.emb_dim == vanilla.emb_dim
    assert model.tower_dims == vanilla.tower_dims
    assert model.aux_weight == vanilla.aux_weight
    assert torch.equal(actual_rng, expected_rng)
    for branch in ("embeddings", "cvr_tower"):
        actual = getattr(model, branch).state_dict()
        expected = getattr(vanilla, branch).state_dict()
        assert actual.keys() == expected.keys()
        for name in actual:
            assert torch.equal(actual[name], expected[name]), name
    model.eval()
    vanilla.eval()
    torch.testing.assert_close(model(_batch())[1], vanilla(_batch())[1], rtol=0, atol=0)


def test_dcn_initialization_matches_standalone_under_same_seed():
    model = _model()
    torch.manual_seed(7)
    standalone = DCNv2(VOCAB, **PARAMS["dcn_params"])

    assert model.ctr_tower.state_dict().keys() == standalone.state_dict().keys()
    for name, value in model.ctr_tower.state_dict().items():
        assert torch.equal(value, standalone.state_dict()[name]), name


def test_shared_tables_supply_distinct_existing_pooling_views_once_per_optimizer():
    model = _model().eval()
    assert model.ctr_tower.embeddings is model.embeddings
    for field in model.field_ids:
        assert model.ctr_tower.embeddings[field].weight is model.embeddings[field].weight

    batch = _batch()
    expected_sum, expected_mean_stats = [], []
    for field in model.field_ids:
        ids, weights, offsets = batch[field]
        embedding = model.embeddings[field]
        expected_sum.append(embedding(ids, offsets, per_sample_weights=weights))
        normalized, stats = _normalized_weights_and_bounded_stats(weights, offsets)
        expected_mean_stats.append(torch.cat(
            (embedding(ids, offsets, per_sample_weights=normalized), stats), dim=1))
    torch.testing.assert_close(model._pool(batch), torch.cat(expected_sum, dim=1))
    captured = []
    handle = model.ctr_tower.cross_layers[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0]))
    try:
        model(batch)
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0], torch.cat(expected_mean_stats, dim=1))

    # nn.Module parameter traversal deduplicates the shared tables: ordinary
    # vanilla-ESMM Adam(model.parameters()) must update each table only once.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    expected_ids = {id(p) for p in model.embeddings.parameters()}
    expected_ids |= {id(p) for p in model.cvr_tower.parameters()}
    expected_ids |= {id(p) for p in model.ctr_tower.parameters()}
    assert {id(p) for p in parameters} == expected_ids


def test_click_loss_supervises_shared_embeddings_but_not_cvr_dense_weights():
    model = _model().eval()
    y = torch.tensor([0.0, 1.0, 1.0])
    q, _ = model(_batch())
    F.binary_cross_entropy(q, y).backward()

    for embedding in model.embeddings.values():
        assert embedding.weight.grad is not None
        assert torch.any(embedding.weight.grad != 0)
    assert all(p.grad is None for p in model.cvr_tower.parameters())
    dense = [p for name, p in model.ctr_tower.named_parameters()
             if not name.startswith("embeddings.")]
    assert all(p.grad is not None for p in dense)
    assert sum(p.grad.abs().sum() for p in dense) > 0


def test_product_loss_reaches_shared_embeddings_and_both_dense_towers():
    model = _model().eval()
    y = torch.tensor([0.0, 1.0, 1.0])
    z = torch.tensor([0.0, 0.0, 1.0])
    q, v = model(_batch())
    F.binary_cross_entropy((q * v).clamp(EPS, 1 - EPS), y * z).backward()

    for embedding in model.embeddings.values():
        assert embedding.weight.grad is not None
        assert torch.any(embedding.weight.grad != 0)
    for branch in (model.ctr_tower, model.cvr_tower):
        dense = [p for name, p in branch.named_parameters()
                 if not name.startswith("embeddings.")]
        assert all(p.grad is not None for p in dense)
        assert sum(p.grad.abs().sum() for p in dense) > 0


def test_unchanged_canonical_esmm_loss_applies_with_no_custom_training_loss():
    model = _model().eval()
    y = torch.tensor([0.0, 1.0, 1.0])
    z = torch.tensor([0.0, 0.0, 1.0])
    q, v = model(_batch())

    assert not hasattr(model, "training_loss")
    expected = (
        F.binary_cross_entropy(q.clamp(EPS, 1 - EPS), y)
        + F.binary_cross_entropy((q.clamp(EPS, 1 - EPS) * v).clamp(EPS, 1 - EPS), y * z)
    )
    torch.testing.assert_close(esmm_loss(y, z, q, v, model.aux_weight), expected)


def test_complete_checkpoint_restoration_preserves_shared_identity_and_eval(tmp_path):
    model = _model().eval()
    assert set(model.model_params["dcn_params"]) == (
        set(inspect.signature(DCNv2).parameters) - {"field_vocab_sizes"})
    path = tmp_path / "esmm_dcn.pt"
    torch.save({"params": model.model_params, "state": model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = ESMMDCN(VOCAB, **checkpoint["params"])
    restored.load_state_dict(checkpoint["state"])
    restored.eval()

    assert restored.ctr_tower.embeddings is restored.embeddings
    for expected, actual, repeated in zip(model(_batch()), restored(_batch()), restored(_batch())):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


def test_explicit_embedding_width_mismatch_is_rejected():
    params = {**PARAMS, "dcn_params": {**PARAMS["dcn_params"], "emb_dim": 3}}
    with pytest.raises(ValueError, match="emb_dim.*share embeddings"):
        ESMMDCN(VOCAB, **params)


def test_omitted_dcn_embedding_width_uses_shared_esmm_width():
    params = {k: v for k, v in PARAMS["dcn_params"].items() if k != "emb_dim"}
    model = ESMMDCN(VOCAB, emb_dim=2, tower_dims=(4, 8, 2), dcn_params=params)

    assert model.model_params["dcn_params"]["emb_dim"] == 2
    assert model.ctr_tower.embeddings is model.embeddings
