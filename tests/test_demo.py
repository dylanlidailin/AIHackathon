from collections import Counter

from xmprobe import cli
from xmprobe.config import run_dir, with_overrides
from xmprobe.data import load_pairs
from xmprobe.demo import stratified_sample
from xmprobe.render import render_pairs


def test_stratified_proportions_within_one_pair():
    pairs = [{"pair_id": str(i), "level": lvl} for i, lvl in enumerate("a" * 94 + "e" * 70 + "g" * 9)]
    sample = stratified_sample(pairs, 100, "level", 42)
    assert len(sample) == 100
    got = Counter(p["level"] for p in sample)
    for lvl, total in Counter(p["level"] for p in pairs).items():
        assert abs(got[lvl] - 100 * total / len(pairs)) <= 1
    assert stratified_sample(pairs, 100, "level", 42) == sample
    assert [int(p["pair_id"]) for p in sample] == sorted(int(p["pair_id"]) for p in sample)


def test_demo_outputs_and_labels(big_run_cfg, fake_runner):
    cfg = with_overrides(big_run_cfg, **{"demo.n_pairs": 30})
    demo_dir = run_dir(cfg) / "demo"
    # pre-render so the fake runner can map the demo images back to their pairs
    render_pairs(load_pairs(cfg, group="main"), demo_dir, cfg["render"])
    fake_runner(cfg, image_dir=demo_dir / "images")

    res = cli.run_stage("demo", cfg)
    md = (demo_dir / "reports" / "headline.md").read_text()
    assert res["n_pairs"] == 30
    assert "demo – signal check, not a result" in md and "OCR not verified" in md
    assert "No outcome classified" in md
    assert (demo_dir / "reports" / "layer_sweep_present.png").exists()
    assert not (demo_dir / "probes" / "obey.csv").exists()
    m = __import__("json").loads(cli.manifest_path(cfg).read_text())
    assert m["stages"]["demo"]["wall_seconds"] >= 0
    assert cli.run_stage("demo", cfg) == {"skipped": True}
