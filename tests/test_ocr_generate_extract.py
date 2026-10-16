import csv
import json

import numpy as np
import pytest

from xmprobe import extract, generate, ocr_check, render
from xmprobe.common import GateFailed, read_jsonl
from xmprobe.config import run_dir, with_overrides
from xmprobe.ocr_check import cer, summarize


def test_cer_ignores_wrapping_whitespace():
    assert cer("hello world", "hello\n  world") == 0.0
    assert cer("abcd", "abce") == 0.25
    assert cer("", "") == 0.0


def test_summary_gate_rules():
    assert summarize([0.01] * 40, 30, 0.10)["pass"] is True
    assert summarize([0.5] * 40, 30, 0.10) == {"n": 40, "median_cer": 0.5, "pass": False,
                                               "reason": "cer_above_gate"}
    s = summarize([0.0] * 10, 30, 0.10)
    assert s["pass"] is False and s["reason"] == "n_below_minimum"


def test_ocr_stage_fails_gate_on_small_sample(run_cfg, fake_runner):
    render.run(run_cfg)
    fake_runner(run_cfg)
    with pytest.raises(GateFailed, match="n_below_minimum"):
        ocr_check.run(run_cfg)
    rows = list(read_jsonl(run_dir(run_cfg) / "ocr.jsonl"))
    assert rows and all(r["cer"] == 0.0 for r in rows)  # fake echoes perfectly


def test_ocr_stage_passes_with_lower_minimum(run_cfg, fake_runner):
    cfg = with_overrides(run_cfg, **{"ocr.min_items": 3})
    render.run(cfg)
    fake_runner(cfg)
    assert ocr_check.run(cfg)["pass"] is True


def test_generate_rows_and_summary(run_cfg, fake_runner):
    render.run(run_cfg)
    fake_runner(run_cfg)
    res = generate.run(run_cfg)
    out = run_dir(run_cfg)
    rows = list(read_jsonl(out / "generations.jsonl"))
    n_pairs = len(render.rendered_pair_ids(run_cfg))
    assert len(rows) == 4 * n_pairs and res["think_leaks"] == 0
    for r in rows:
        assert {"target_hit", "destination_hit", "content_hit", "repetition", "hit_limit"} <= r.keys()
        if r["condition"] in ("text_clean", "image_clean"):
            assert not r["target_hit"]
    assert all(r["target_hit"] for r in rows if r["condition"] == "text_injected")
    table = list(csv.DictReader((out / "behavior_summary.csv").open()))
    assert {(t["group"], t["condition"]) for t in table} == {
        (g, c) for g in ("main", "probe_evading")
        for c in ("text_clean", "text_injected", "image_clean", "image_injected")}
    assert {"target_rate", "rep_rate", "wilson_lo", "wilson_hi", "mcnemar_p_text_vs_image"} <= table[0].keys()
    assert generate.run(run_cfg) == {"skipped": True}


def test_extract_shape_index_and_reuse(run_cfg, fake_runner):
    render.run(run_cfg)
    fr = fake_runner(run_cfg)
    res = extract.run(run_cfg)
    out = run_dir(run_cfg) / "activations"
    n = len(render.rendered_pair_ids(run_cfg))
    for c in ("text_clean", "text_injected", "image_clean", "image_injected"):
        assert np.load(out / f"{c}.npy").shape == (n, fr.n_layers + 1, fr.hidden)
    index = json.loads((out / "index.json").read_text())
    assert len(index["pair_ids"]) == n and index["model_revision"] == "fake-rev" and index["config_hash"]
    assert extract.run(run_cfg) == {"skipped": True}
    # a render change invalidates the image activations' cache key
    changed = with_overrides(run_cfg, **{"render.font_size": 28})
    assert extract.SECTIONS and "render" in extract.SECTIONS
    from xmprobe.common import is_done, stamp_key
    assert not is_done(run_dir(changed), "extract", stamp_key(changed, extract.SECTIONS, None))
