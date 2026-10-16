import pytest

from xmprobe.llmail import convert
from xmprobe.prompts import CONDITIONS, build_messages, masked


@pytest.fixture
def pair(raw_rows, static, llmail_cfg):
    return convert(raw_rows, static, llmail_cfg)[0][0]


def test_four_prompts_equal_with_slot_masked(pair):
    flat = {masked(build_messages(pair, c), pair).replace("<SLOT>", "") for c in CONDITIONS}
    assert len(flat) == 1


def test_only_image_conditions_have_an_image(pair):
    for c in CONDITIONS:
        n_images = sum(p["type"] == "image" for m in build_messages(pair, c) for p in m["content"])
        assert n_images == (1 if c.startswith("image") else 0)


def test_text_condition_contains_content(pair):
    user = build_messages(pair, "text_injected")[-1]["content"][0]["text"]
    assert pair["injected_content"] in user and pair["user_task"] in user
    assert build_messages(pair, "text_clean")[0]["content"][0]["text"] == pair["system_prompt"]


def test_unknown_condition(pair):
    with pytest.raises(ValueError):
        build_messages(pair, "audio_clean")
