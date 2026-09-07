"""Streaming reader/parser for the official Ali-CCP release.

File format (confirmed against the actual `ali-cpp/*.tar.gz` files):
  sample_skeleton_{train,test}.csv:
    sample_id,y,z,common_feature_index,num_features,feature_list
  common_features_{train,test}.csv:
    common_feature_index,num_features,feature_list

`feature_list` is a "\x01"-separated list of "field_id\x02feat_id\x03value"
entries. `field_id` is a string, not always numeric (e.g. "109_14").
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass, field

FeatureBag = dict[str, list[tuple[int, float]]]


def _parse_feature_list(feature_list: str) -> FeatureBag:
    fields: FeatureBag = {}
    for entry in feature_list.split("\x01"):
        field_id, rest = entry.split("\x02")
        feat_id, value = rest.split("\x03")
        fields.setdefault(field_id, []).append((int(feat_id), float(value)))
    return fields


@dataclass
class Impression:
    sample_id: int
    y: int
    z: int
    common_feature_index: str
    fields: FeatureBag = field(default_factory=dict)


def parse_skeleton_line(line: str) -> Impression | None:
    """Parse one sample_skeleton line. Returns None if z > y (invalid row)."""
    sample_id, y, z, common_feature_index, _, feature_list = line.rstrip("\n").split(
        ",", maxsplit=5
    )
    y, z = int(y), int(z)
    if z > y:
        return None
    return Impression(int(sample_id), y, z, common_feature_index, _parse_feature_list(feature_list))


def parse_common_features_line(line: str) -> tuple[str, FeatureBag]:
    """Parse one common_features line into (common_feature_index, fields)."""
    common_feature_index, _, feature_list = line.rstrip("\n").split(",", maxsplit=2)
    return common_feature_index, _parse_feature_list(feature_list)


def iter_impressions(
    tar_path: str,
    split: str,
    limit: int | None = None,
    common_features_limit: int | None = None,
):
    """Yield `Impression`s from `tar_path` with common features merged in.

    `split` is "train" or "test", selecting `{sample_skeleton,common_features}_{split}.csv`
    inside the tarball. `limit` bounds the number of yielded impressions.
    `common_features_limit` bounds how many common-features lines are loaded into
    the join dict (the full file is tens of GB); an impression whose
    `common_feature_index` wasn't loaded is skipped. Leave both `None` for a
    complete, correct pass over a split.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        common_features: dict[str, FeatureBag] = {}
        with tar.extractfile(f"common_features_{split}.csv") as f:
            for i, raw_line in enumerate(f):
                if common_features_limit is not None and i >= common_features_limit:
                    break
                idx, fields = parse_common_features_line(raw_line.decode("utf-8"))
                common_features[idx] = fields

        with tar.extractfile(f"sample_skeleton_{split}.csv") as f:
            yielded = 0
            for raw_line in f:
                if limit is not None and yielded >= limit:
                    return
                impression = parse_skeleton_line(raw_line.decode("utf-8"))
                if impression is None:
                    continue
                common = common_features.get(impression.common_feature_index)
                if common is None:
                    continue
                impression.fields.update(common)
                yield impression
                yielded += 1
