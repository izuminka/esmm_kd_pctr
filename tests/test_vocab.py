import torch
from torch import nn

from data.raw import Impression
from data.vocab import build_vocabs, encode_batch


def test_build_vocabs_sorts_lexicographically_not_numerically():
    rows = [Impression(1, 1, 0, "c1", {"f": [(9, 1.0), (10, 1.0), (2, 1.0)]})]

    vocabs = build_vocabs(rows, ["f"])

    # lexicographic string order: "10" < "2" < "9", not numeric order
    assert vocabs["f"] == {"10": 1, "2": 2, "9": 3}


def test_build_vocabs_only_uses_requested_fields():
    rows = [Impression(1, 1, 0, "c1", {"f": [(1, 1.0)], "g": [(2, 1.0)]})]

    vocabs = build_vocabs(rows, ["f"])

    assert list(vocabs.keys()) == ["f"]


def test_unseen_token_forced_to_zero_weight():
    train_rows = [Impression(1, 1, 0, "c1", {"f": [(10, 1.0), (20, 2.0)]})]
    vocabs = build_vocabs(train_rows, ["f"])
    test_rows = [Impression(2, 0, 0, "c2", {"f": [(99, 5.0)]})]  # 99 never seen in training

    encoded = encode_batch(test_rows, ["f"], vocabs)
    ids, weights, _ = encoded["f"]

    assert ids.tolist() == [0]
    assert weights.tolist() == [0.0]


def test_missing_field_yields_empty_bag():
    train_rows = [Impression(1, 1, 0, "c1", {"f": [(10, 1.0)]})]
    vocabs = build_vocabs(train_rows, ["f"])
    test_rows = [Impression(2, 0, 0, "c2", {})]  # row has no entry for field "f" at all

    encoded = encode_batch(test_rows, ["f"], vocabs)
    ids, _, offsets = encoded["f"]

    assert ids.numel() == 0
    assert offsets.tolist() == [0]


def test_encode_batch_known_token_keeps_its_weight():
    rows = [Impression(1, 1, 0, "c1", {"f": [(10, 3.5)]})]
    vocabs = build_vocabs(rows, ["f"])

    encoded = encode_batch(rows, ["f"], vocabs)
    ids, weights, offsets = encoded["f"]

    assert ids.tolist() == [1]
    assert weights.tolist() == [3.5]
    assert offsets.tolist() == [0]


def test_embedding_bag_weighted_sum_pooling():
    # Documents the EmbeddingBag contract encode_batch's output must satisfy: raw
    # weighted sum (no normalization), and a zero vector for an empty bag with no
    # extra padding needed.
    bag = nn.EmbeddingBag(3, 2, mode="sum")
    with torch.no_grad():
        bag.weight.copy_(torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    ids = torch.tensor([1, 2], dtype=torch.long)
    weights = torch.tensor([0.5, 2.0])
    offsets = torch.tensor([0, 2], dtype=torch.long)  # bag 0: tokens [1,2]; bag 1: empty

    out = bag(ids, offsets, per_sample_weights=weights)

    assert torch.allclose(out, torch.tensor([[7.0, 70.0], [0.0, 0.0]]))
