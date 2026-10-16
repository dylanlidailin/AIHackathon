import json

from xmprobe.data import validate_rows
from xmprobe.llmail import (closest_benign, convert, format_email, normalize_body, route_level)

from conftest import EMAIL_A


def test_level_routing(llmail_cfg):
    assert route_level("level1a", llmail_cfg) == ("a", "main")
    assert route_level("level1c", llmail_cfg) == ("c", "probe_evading")
    assert route_level("level1b", llmail_cfg) is None
    assert route_level("level2a", llmail_cfg) is None


def test_every_drop_reason_counted(raw_rows, static, llmail_cfg):
    pairs, rep = convert(raw_rows, static, llmail_cfg)
    d = rep.to_dict()["dropped"]
    assert d["other_scenario_or_level"] == 3
    assert d["unparsable_objectives"] == 1
    assert d["partial_success"] == 2
    assert d["too_long"] == 1
    assert d["duplicate"] == 2
    assert d["team_cap"] == 1
    assert rep.seen == len(raw_rows)


def test_groups_and_main_wins_duplicates(raw_rows, static, llmail_cfg):
    pairs, _ = convert(raw_rows, static, llmail_cfg)
    main = [p for p in pairs if p["group"] == "main"]
    evading = [p for p in pairs if p["group"] == "probe_evading"]
    assert {p["level"] for p in main} <= {"a", "e", "g"}
    assert {p["level"] for p in evading} == {"c", "d", "i", "j"}
    texts_main = {normalize_body(p["injection_text"]) for p in main}
    assert not any(normalize_body(p["injection_text"]) in texts_main for p in evading)
    # per-team cap of 2 holds
    from collections import Counter
    assert max(Counter(p["team_id"] for p in pairs).values()) <= 2


def test_shortfall_reported(raw_rows, static, llmail_cfg):
    _, rep = convert(raw_rows, static, llmail_cfg)
    assert rep.shortfall["main"] == 200 - 5
    assert rep.shortfall["probe_evading"] == 100 - 4


def test_pairs_differ_only_in_second_email(raw_rows, static, llmail_cfg):
    pairs, _ = convert(raw_rows, static, llmail_cfg)
    for p in pairs:
        clean, inj = p["clean_content"].split("\n\n"), p["injected_content"].split("\n\n")
        assert len(clean) == len(inj) == 2
        a_pos = clean.index(EMAIL_A)
        assert inj[a_pos] == EMAIL_A
        assert inj[1 - a_pos] == p["injection_text"]
        assert clean[1 - a_pos] in static["benign_pool"]
    assert {p["email_order"] for p in pairs} == {"a_first", "second_first"}


def test_length_matching():
    pool = ["x" * 10, "x" * 100, "x" * 300]
    assert closest_benign(290, pool) == "x" * 300
    assert closest_benign(55, pool) == "x" * 10  # tie 45 vs 45 -> earliest


def test_same_seed_byte_identical(raw_rows, static, llmail_cfg):
    a, _ = convert(raw_rows, static, llmail_cfg)
    b, _ = convert(list(reversed(raw_rows)), static, llmail_cfg)
    dump = lambda ps: "\n".join(json.dumps(p, sort_keys=True) for p in ps)
    assert dump(a) == dump(b)


def test_output_passes_schema(raw_rows, static, llmail_cfg):
    pairs, _ = convert(raw_rows, static, llmail_cfg)
    validate_rows(pairs)
    assert all(p["attack_success_phi3"] for p in pairs if p["level"] in "aeg")
    assert format_email("s", "b") == "Subject of the email: s.   Body: b"
