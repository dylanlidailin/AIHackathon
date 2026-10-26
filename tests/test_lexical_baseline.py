import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lexical_baseline", ROOT / "scripts" / "lexical_baseline.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def test_keyword_scores_count_case_insensitive():
    s = lb.keyword_scores(["Send CONFIRMATION to contact@contact.com", "lunch moved", "confirmation"],
                          ["contact@contact.com", "confirmation"])
    assert s.tolist() == [2.0, 0.0, 1.0]


def test_benign_part_either_order():
    a, benign, attack = "EMAIL A", "benign mail", "attack mail"
    for clean, injected in ((f"{a}\n\n{benign}", f"{a}\n\n{attack}"), (f"{benign}\n\n{a}", f"{attack}\n\n{a}")):
        p = {"clean_content": clean, "injected_content": injected, "injection_text": attack}
        assert lb.benign_part(p) == benign


def test_run_on_separable_synthetic_pairs(big_run_cfg, tmp_path):
    out = tmp_path / "lexical.md"
    res = lb.run(big_run_cfg, out)
    by = {r["baseline"]: r for r in res["results"]}
    assert set(by) == set(lb.BASELINES)
    # synthetic attacks all contain the target strings; benign emails never do
    assert by["keyword"]["main"][0] == 1.0
    assert by["word_tfidf"]["main"][0] > 0.95
    assert "probe_evading" in by["char_tfidf"]
    assert res["distinct_benign"] == 40 and res["max_reuse"] == 1
    text = out.read_text()
    assert "Wording alone separates" in text and "distinct benign emails" in text


def test_verdict_thresholds():
    assert "Wording alone" in lb.verdict(0.99)
    assert "large part" in lb.verdict(0.85)
    assert "beyond surface" in lb.verdict(0.6)
    assert "Not enough" in lb.verdict(float(np.nan))
