"""Per-field vocabularies for Ali-CCP categorical features, and batch encoding
for `torch.nn.EmbeddingBag`.

Per the paper: fit each field's vocabulary from training tokens only, sort
distinct token strings lexicographically (not numerically), assign IDs from 1,
and reserve ID 0 for unseen tokens (whose weight is then masked to 0).
"""

import torch

from data.raw import Impression


def build_vocabs(impressions: list[Impression], field_ids: list[str]) -> dict[str, dict[str, int]]:
    """Build one `{token_string: id}` vocab per field, ID 0 reserved for unseen."""
    tokens_by_field: dict[str, set[str]] = {f: set() for f in field_ids}
    for impression in impressions:
        for field_id, entries in impression.fields.items():
            if field_id in tokens_by_field:
                tokens_by_field[field_id].update(str(feat_id) for feat_id, _ in entries)
    return {
        field_id: {token: i + 1 for i, token in enumerate(sorted(tokens))}
        for field_id, tokens in tokens_by_field.items()
    }


def encode_batch(
    impressions: list[Impression],
    field_ids: list[str],
    vocabs: dict[str, dict[str, int]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Encode a batch into per-field `(ids, weights, offsets)` for `EmbeddingBag`.

    Unseen tokens (not in the field's vocab) map to ID 0 with weight forced to 0,
    matching the paper. Offsets follow `EmbeddingBag`'s convention: the start
    index of each bag within the flat `ids`/`weights` tensors.
    """
    encoded = {}
    for field_id in field_ids:
        vocab = vocabs[field_id]
        ids, weights, offsets = [], [], []
        for impression in impressions:
            offsets.append(len(ids))
            for feat_id, weight in impression.fields.get(field_id, []):
                token = str(feat_id)
                token_id = vocab.get(token, 0)
                ids.append(token_id)
                weights.append(0.0 if token_id == 0 else weight)
        encoded[field_id] = (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(offsets, dtype=torch.long),
        )
    return encoded
