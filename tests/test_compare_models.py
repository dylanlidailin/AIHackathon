import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_models", ROOT / "scripts" / "compare_models.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

CONFIG = {"run_name": "x", "data": {"pairs_path": "data/pairs.jsonl", "llmail": {}}, "render": {"font_size": 22},
          "probe": {"methods": ["lr", "dom"]}, "generate": {"max_new_tokens": 256}, "ocr": {"cer_gate": 0.1}}


def write_run(d: Path, name: str, p2_by_layer, cos_by_layer, *, acc=0.5, obey=(0.45, 0.30, 0.60), n=12,
              config=None):
    """A minimal run directory. p2_by_layer(layer) -> AUROC, cos_by_layer(layer) -> cosine."""
    (d / "probes").mkdir(parents=True)
    (d / "reports").mkdir()
    cfg = copy.deepcopy(config or CONFIG)
    cfg["run_name"] = name
    (d / "run_manifest.json").write_text(json.dumps({"config": cfg, "model_id": f"org/{name}"}))
    (d / ".extract.done.json").write_text(json.dumps({"key": {"pairs": "abc"}, "outputs": []}))
    head = n - 1
    rows = []
    for m in ("lr", "dom"):
        for probe in ("P1", "P2", "P3"):
            for layer in range(n + 1):
                a = 0.5 if layer == 0 else (p2_by_layer(layer) if probe == "P2" else 1.0)
                rows.append({"target": "present", "probe": probe, "method": m, "variant": "raw",
                             "eval_group": "main", "layer": layer, "auroc": a, "ci_lo": a - 0.05, "ci_hi": min(1, a + 0.05),
                             "acc": acc if probe == "P2" else 0.99, "is_headline": layer == head})
    with (d / "probes" / "present.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a, lo, hi = obey
    orows = [{"target": "obey", "probe": "P2", "method": m, "variant": "raw", "eval_group": "main", "layer": head,
              "auroc": a, "ci_lo": lo, "ci_hi": hi, "acc": 0.5, "is_headline": True} for m in ("lr", "dom")]
    with (d / "probes" / "obey.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orows[0]))
        w.writeheader()
        w.writerows(orows)
    with (d / "behavior_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "condition", "target_rate", "wilson_lo", "wilson_hi",
                                          "mcnemar_p_text_vs_image"])
        w.writeheader()
        for c in ("text_injected", "image_injected"):
            w.writerow({"group": "main", "condition": c, "target_rate": 0.15, "wilson_lo": 0.1, "wilson_hi": 0.2,
                        "mcnemar_p_text_vs_image": 1.0})
    with (d / "reports" / "direction_cosine.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cosine"])
        w.writerows([(layer, cos_by_layer(layer)) for layer in range(1, n + 1)])
    return d


def shared_like(layer):  # dip in the middle third, ~1.0 late
    return 0.4 if 5 <= layer <= 7 else (0.99 if layer > 8 else 0.9)


def test_pattern_and_failure(tmp_path):
    good = write_run(tmp_path / "a", "good", shared_like, lambda l: 0.6 if l > 8 else 0.0)
    flat = write_run(tmp_path / "b", "flat", lambda l: 0.6, lambda l: 0.1, acc=0.9)
    runs = [cm.load_run(good), cm.load_run(flat)]
    cm.check_inputs(runs)
    rg, rf = (cm.criteria(r) for r in runs)
    assert [rg[f"C{i}"]["pass"] for i in range(1, 6)] == [True, True, True, True, True]
    assert rf["C1"]["pass"] is False and rf["C2"]["pass"] is None  # C2 needs C1
    assert rf["C3"]["pass"] is False and rf["C4"]["pass"] is False
    out = tmp_path / "out"
    out.mkdir()
    text = cm.report(runs, [rg, rf], out)
    assert "C1: not shared: good pass, flat fail" in text
    assert "**C5: shared** by all models." in text
    cm.plots(runs, out)
    assert (out / "p2_sweep_relative.png").exists() and (out / "direction_cosine_relative.png").exists()


def test_obey_transfer_and_skip(tmp_path):
    r = write_run(tmp_path / "a", "obeys", shared_like, lambda l: 0.6 if l > 8 else 0.0, obey=(0.9, 0.8, 0.97))
    assert cm.criteria(cm.load_run(r))["C5"]["pass"] is False
    (r / "probes" / "obey_meta.json").write_text(json.dumps({"skipped": True}))
    assert cm.criteria(cm.load_run(r))["C5"]["pass"] is None


def test_mismatched_inputs_rejected(tmp_path):
    a = write_run(tmp_path / "a", "a", shared_like, lambda l: 0.5)
    other = copy.deepcopy(CONFIG)
    other["render"]["font_size"] = 28
    b = write_run(tmp_path / "b", "b", shared_like, lambda l: 0.5, config=other)
    with pytest.raises(cm.MismatchError, match="render"):
        cm.check_inputs([cm.load_run(a), cm.load_run(b)])
