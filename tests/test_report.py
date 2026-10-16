import json

import pytest

from xmprobe import cli, render
from xmprobe.config import run_dir
from xmprobe.report import classify

RCFG = {"works_auroc": 0.70, "low_acc": 0.60}


def row(probe, method, auroc, lo, hi, acc=0.8, variant="raw"):
    return {"target": "present", "probe": probe, "method": method, "eval_group": "main",
            "variant": variant, "is_headline": "True", "auroc": str(auroc), "ci_lo": str(lo),
            "ci_hi": str(hi), "acc": str(acc)}


def table(lr, dom):
    return [row(p, "lr", *v) for p, v in lr.items()] + [row(p, "dom", *v) for p, v in dom.items()]


def test_per_method_p1_gate():
    rows = table({"P1": (0.92, .88, .95), "P2": (0.55, .45, .65), "P3": (0.90, .85, .94)},
                 {"P1": (0.71, .65, .77), "P2": (0.52, .44, .60), "P3": (0.60, .5, .7)})
    v = classify(rows, ["lr", "dom"], True, False, RCFG, 0.80)
    assert v["labels"]["dom"] == "not interpretable (P1)" and v["labels"]["lr"] == "ok"
    assert v["outcome"] == "p2_fails_p3_works" and list(v["per_method"]) == ["lr"]


def test_method_dependent_when_p2_cis_disjoint():
    rows = table({"P1": (0.95, .9, .98), "P2": (0.90, .85, .95), "P3": (0.9, .85, .95)},
                 {"P1": (0.93, .9, .96), "P2": (0.62, .55, .70), "P3": (0.9, .85, .95)})
    v = classify(rows, ["lr", "dom"], True, False, RCFG, 0.80)
    assert v["outcome"] is None and v["reason"] == "method-dependent"


@pytest.mark.parametrize("p2,p3,acc,expected", [
    ((0.55, .45, .65), (0.90, .85, .95), 0.8, "p2_fails_p3_works"),
    ((0.88, .80, .93), (0.90, .85, .95), 0.8, "p2_works"),
    ((0.88, .80, .93), (0.90, .85, .95), 0.52, "shifted"),
    ((0.55, .45, .65), (0.58, .48, .66), 0.8, "both_fail"),
])
def test_outcome_rows(p2, p3, acc, expected):
    lr = {"P1": (0.95, .9, .98), "P2": p2, "P3": p3}
    rows = [row(p, "lr", *v, acc=acc) for p, v in lr.items()]
    assert classify(rows, ["lr"], True, False, RCFG, 0.80)["outcome"] == expected


def test_ocr_fail_and_demo_block_classification():
    rows = table({"P1": (0.95, .9, .98), "P2": (0.9, .85, .95), "P3": (0.9, .85, .95)},
                 {"P1": (0.95, .9, .98), "P2": (0.9, .85, .95), "P3": (0.9, .85, .95)})
    v = classify(rows, ["lr", "dom"], False, False, RCFG, 0.80)
    assert v["outcome"] is None and v["labels"]["lr"] == "not interpretable (OCR)"
    d = classify(rows, ["lr", "dom"], False, True, RCFG, 0.80)
    assert d["outcome"] is None and d["labels"]["lr"] == "OCR not verified" and "demo" in d["reason"]


def test_end_to_end_and_byte_identical_regeneration(big_run_cfg, fake_runner, tmp_path):
    cfg = big_run_cfg
    render.run(cfg)
    fake_runner(cfg)
    for stage in ("validate_data", "ocr_check", "generate", "extract", "probe", "report"):
        cli.run_stage(stage, cfg)
    reports = run_dir(cfg) / "reports"
    md = (reports / "headline.md").read_text()
    assert "## Outcome" in md and "Length gap" in md and "Qwen/Qwen3.5-0.8B" in md
    assert "P2 works" in md  # fake runner plants the same signal in both modalities
    assert (reports / "layer_sweep_present.png").stat().st_size > 10_000
    first = {p.name: p.read_bytes() for p in reports.iterdir()}
    for p in reports.iterdir():
        p.unlink()
    cli.run_stage("report", cfg)
    second = {p.name: p.read_bytes() for p in reports.iterdir()}
    assert first.keys() == second.keys()
    for name in first:
        if not name.endswith(".png"):
            assert first[name] == second[name], name
    m = json.loads(cli.manifest_path(cfg).read_text())
    assert all(m["stages"][s]["status"] == "ok" for s in ("generate", "extract", "probe", "report"))
