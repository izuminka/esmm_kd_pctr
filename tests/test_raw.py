import io
import tarfile

from data.raw import iter_impressions, parse_common_features_line, parse_skeleton_line


def test_parse_skeleton_line_valid_row():
    line = "1,1,0,bacff91692951881,2,210\x029052218\x031.0\x01216\x029154780\x031.0\n"

    impression = parse_skeleton_line(line)

    assert impression.sample_id == 1
    assert impression.y == 1
    assert impression.z == 0
    assert impression.common_feature_index == "bacff91692951881"
    assert impression.fields == {"210": [(9052218, 1.0)], "216": [(9154780, 1.0)]}


def test_parse_skeleton_line_rejects_conversion_without_click():
    line = "2,0,1,bacff91692951881,1,210\x029052218\x031.0\n"  # z=1 > y=0

    assert parse_skeleton_line(line) is None


def test_parse_skeleton_line_allows_click_and_conversion():
    line = "3,1,1,bacff91692951881,1,210\x029052218\x031.0\n"

    impression = parse_skeleton_line(line)

    assert impression is not None
    assert impression.y == 1 and impression.z == 1


def test_parse_common_features_line():
    line = "bacff91692951881,2,101\x02313191\x031.0\x01125\x023438774\x031.0\n"

    idx, fields = parse_common_features_line(line)

    assert idx == "bacff91692951881"
    assert fields == {"101": [(313191, 1.0)], "125": [(3438774, 1.0)]}


def test_parse_handles_non_numeric_field_id():
    line = "abc,1,109_14\x021176401\x032.56495\n"

    _, fields = parse_common_features_line(line)

    assert fields == {"109_14": [(1176401, 2.56495)]}


def _make_tar(tmp_path, skeleton_lines, common_lines):
    tar_path = tmp_path / "fake.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        members = {
            "sample_skeleton_train.csv": skeleton_lines,
            "common_features_train.csv": common_lines,
        }
        for name, lines in members.items():
            data = "".join(lines).encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(tar_path)


def test_iter_impressions_joins_and_filters(tmp_path):
    skeleton = [
        "1,1,0,c1,1,210\x029052218\x031.0\n",
        "2,0,1,c1,1,210\x029052218\x031.0\n",  # invalid: z>y, dropped
        "3,1,0,c2,1,210\x029052218\x031.0\n",  # c2 has no common-features row, skipped
    ]
    common = ["c1,1,101\x02313191\x031.0\n"]
    tar_path = _make_tar(tmp_path, skeleton, common)

    impressions = list(iter_impressions(tar_path, "train"))

    assert len(impressions) == 1
    assert impressions[0].sample_id == 1
    assert impressions[0].fields["101"] == [(313191, 1.0)]  # merged in from common features
    assert impressions[0].fields["210"] == [(9052218, 1.0)]  # from the skeleton row itself


def test_iter_impressions_respects_limit(tmp_path):
    skeleton = [f"{i},1,0,c1,1,210\x029052218\x031.0\n" for i in range(5)]
    common = ["c1,1,101\x02313191\x031.0\n"]
    tar_path = _make_tar(tmp_path, skeleton, common)

    impressions = list(iter_impressions(tar_path, "train", limit=2))

    assert len(impressions) == 2


def test_iter_impressions_common_features_limit_bounds_join(tmp_path):
    skeleton = ["1,1,0,c2,1,210\x029052218\x031.0\n"]
    common = ["c1,1,101\x02313191\x031.0\n", "c2,1,101\x02313191\x031.0\n"]
    tar_path = _make_tar(tmp_path, skeleton, common)

    # only the first common-features line (c1) gets loaded, so c2's join misses
    impressions = list(iter_impressions(tar_path, "train", common_features_limit=1))

    assert impressions == []
