import os

import pytest
import torch

from data.raw import iter_impressions
from data.vocab import build_vocabs, encode_batch

FIELD_IDS = [
    "101", "109_14", "110_14", "121", "122", "124", "125", "126", "127", "127_14",
    "128", "129", "150_14", "205", "206", "207", "210", "216", "301", "508", "509",
    "702", "853",
]
TAR_PATH = os.path.join(os.path.dirname(__file__), "..", "ali-ccp", "sample_train.tar.gz")


@pytest.fixture(scope="session")
def real_batch():
    """~2,000 real Ali-CCP rows, encoded for ESMM. Shared across test modules
    since it is slow (~5 min, dominated by scanning the common-features file)."""
    impressions = list(iter_impressions(TAR_PATH, "train", limit=2000, common_features_limit=200_000))
    assert len(impressions) > 100  # sanity: join actually produced usable rows
    vocabs = build_vocabs(impressions, FIELD_IDS)
    encoded = encode_batch(impressions, FIELD_IDS, vocabs)
    y = torch.tensor([imp.y for imp in impressions], dtype=torch.float32)
    z = torch.tensor([imp.z for imp in impressions], dtype=torch.float32)
    vocab_sizes = {f: len(v) for f, v in vocabs.items()}
    return encoded, y, z, vocab_sizes
