import copy

import pytest

from xmprobe.common import GateFailed, write_jsonl
from xmprobe.data import SchemaError, run, validate_rows
from xmprobe.llmail import convert


@pytest.fixture
def pairs(raw_rows, static, llmail_cfg):
    return convert(raw_rows, static, llmail_cfg)[0]


def test_valid_rows_pass(pairs):
    validate_rows(pairs)


@pytest.mark.parametrize("field", ["target_phrase", "team_id", "group", "clean_content"])
def test_missing_field_named(pairs, field):
    bad = copy.deepcopy(pairs)
    del bad[2][field]
    with pytest.raises(SchemaError, match=f"{bad[2]['pair_id']}.*'{field}'"):
        validate_rows(bad)


def test_wrong_type_rejected(pairs):
    bad = copy.deepcopy(pairs)
    bad[0]["attack_success_phi3"] = "yes"
    with pytest.raises(SchemaError, match="attack_success_phi3"):
        validate_rows(bad)


def test_bad_group_and_duplicate_id(pairs):
    bad = copy.deepcopy(pairs)
    bad[0]["group"] = "train"
    with pytest.raises(SchemaError, match="group"):
        validate_rows(bad)
    dup = copy.deepcopy(pairs) + [copy.deepcopy(pairs[0])]
    with pytest.raises(SchemaError, match="duplicate pair_id"):
        validate_rows(dup)


def test_injection_must_be_in_content(pairs):
    bad = copy.deepcopy(pairs)
    bad[0]["injection_text"] = "not there"
    with pytest.raises(SchemaError, match="injection_text"):
        validate_rows(bad)


def test_stage_reports_counts_and_fails_as_gate(pairs, tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, pairs)
    cfg = {"data": {"pairs_path": str(path)}}
    res = run(cfg)
    assert res["by_group"] == {"main": 5, "probe_evading": 4}
    rows = copy.deepcopy(pairs)
    del rows[0]["level"]
    write_jsonl(path, rows)
    with pytest.raises(GateFailed, match="missing field 'level'"):
        run(cfg)
