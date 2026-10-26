"""Lexical baselines for the ``present`` label (post-hoc, CPU only, no model needed).

Usage: python scripts/lexical_baseline.py --config configs/qwen35_2b.yaml [--out PATH]

Question: can a classifier that only sees the words of the document (no activations)
separate clean from injected content? The present probes reach AUROC ~1.0 already at
layer 1 (P1 and P3, all three models). If a bag-of-words model also reaches ~1.0, that
saturation is explained by surface wording, and "injection present" should be read as
"attack-email wording present" until harder negatives are added (docs/notes_present_probe.md).

Uses the same pairs, the same team-grouped folds (data.grouped_folds) and the same
fold-weighted AUROC with team bootstrap as the probe stage, so the numbers sit next to
P1 directly. probe_evading pairs are scored by a model fit on all main pairs.

Baselines
  keyword     count of the target strings (generate.target destination and content) in
              the document, case-insensitive; nothing is fitted
  word_tfidf  TF-IDF on word 1-2-grams + logistic regression, fit per fold
  char_tfidf  TF-IDF on char 3-5-grams (robust to spacing/obfuscation) + logistic regression

Also reports how many distinct benign emails the clean negatives are built from.
Writes <run_dir>/reports/lexical_baseline.md unless --out is given, and prints it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from xmprobe.config import load_config, run_dir
from xmprobe.data import grouped_folds, load_pairs
from xmprobe.probes import auroc, auroc_ci, bootstrap_sets, fold_auroc

BASELINES = ("keyword", "word_tfidf", "char_tfidf")
SATURATED = 0.95   # at or above: wording alone explains the probe saturation
PARTIAL = 0.80     # at or above: wording explains a large part of it


def keyword_scores(texts: list[str], keywords: list[str]) -> np.ndarray:
    kws = [k.lower() for k in keywords if k]
    return np.array([sum(k in t.lower() for k in kws) for t in texts], dtype=float)


def make_model(kind: str, seed: int):
    if kind == "word_tfidf":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
    elif kind == "char_tfidf":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    else:
        raise ValueError(f"unknown baseline {kind!r}")
    return make_pipeline(vec, LogisticRegression(C=1.0, max_iter=5000, random_state=seed))


def doc_rows(pairs: list[dict[str, Any]], idx: list[int]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Two rows per pair, clean (0) then injected (1), as in the present probes."""
    texts, y, teams = [], [], []
    for i in idx:
        p = pairs[i]
        texts += [p["clean_content"], p["injected_content"]]
        y += [0, 1]
        teams += [p["team_id"], p["team_id"]]
    return texts, np.array(y), np.array(teams)


def benign_part(p: dict[str, Any]) -> str:
    """The benign email inside clean_content (clean = benign + shared Email A, in either order)."""
    shared = p["injected_content"].replace(p.get("injection_text") or "", "", 1).strip("\n")
    if not shared or shared == p["injected_content"]:
        return p["clean_content"]
    return p["clean_content"].replace(shared, "", 1).strip("\n")


def evaluate(pairs: list[dict[str, Any]], folds, kind: str, keywords: list[str],
             B: int, seed: int) -> dict[str, Any]:
    main_idx = sorted(i for _, test in folds for i in test)
    pos = {i: k for k, i in enumerate(main_idx)}
    texts, y, teams = doc_rows(pairs, main_idx)
    scores = np.full(len(y), np.nan)
    fold = np.zeros(len(y), dtype=int)
    for k, (train, test) in enumerate(folds):
        rows = [2 * pos[i] + j for i in test for j in (0, 1)]
        test_texts = [texts[r] for r in rows]
        if kind == "keyword":
            s = keyword_scores(test_texts, keywords)
        else:
            tr_texts, tr_y, _ = doc_rows(pairs, train)
            s = make_model(kind, seed).fit(tr_texts, tr_y).decision_function(test_texts)
        scores[rows] = s
        fold[rows] = k
    lo, hi = auroc_ci(y, scores, bootstrap_sets(teams, B, seed), fold)
    out: dict[str, Any] = {"baseline": kind, "main": (fold_auroc(y, scores, fold), lo, hi, len(main_idx))}

    ev_idx = [i for i, p in enumerate(pairs) if p["group"] == "probe_evading"]
    if ev_idx:
        ev_texts, ev_y, ev_teams = doc_rows(pairs, ev_idx)
        if kind == "keyword":
            s = keyword_scores(ev_texts, keywords)
        else:
            s = make_model(kind, seed).fit(texts, y).decision_function(ev_texts)
        ev_lo, ev_hi = auroc_ci(ev_y, s, bootstrap_sets(ev_teams, B, seed))
        out["probe_evading"] = (auroc(ev_y, s), ev_lo, ev_hi, len(ev_idx))

    if kind == "word_tfidf":
        model = make_model(kind, seed).fit(texts, y)
        vec, lr = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
        names = vec.get_feature_names_out()
        order = np.argsort(lr.coef_[0])
        out["top_injected"] = [str(names[j]) for j in order[::-1][:12]]
        out["top_clean"] = [str(names[j]) for j in order[:8]]
    return out


def verdict(best: float) -> str:
    if best != best:
        return "Not enough data to judge."
    if best >= SATURATED:
        return ("**Wording alone separates clean from injected.** Near-perfect present-probe AUROCs "
                "can then be explained by surface features, so read \"injection present\" as "
                "\"attack-email wording present\" until harder negatives are added.")
    if best >= PARTIAL:
        return ("**Wording explains a large part of the signal.** Compare the probe sweep against "
                "this number before claiming the probe tracks injection beyond keywords.")
    return ("Wording alone does not separate the classes well; the probes carry information "
            "beyond surface features.")


def fmt(res: tuple[float, float, float, int]) -> str:
    a, lo, hi, _ = res
    ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "[–]"
    return f"{a:.3f} {ci}"


def run(cfg: dict[str, Any], out: Path | None = None) -> dict[str, Any]:
    pcfg = cfg["probe"]
    pairs = load_pairs(cfg)
    folds = grouped_folds(pairs, pcfg["n_folds"], pcfg["seed"], pcfg.get("group_by", "team_id"))
    target = cfg.get("generate", {}).get("target", {})
    keywords = [target.get("destination", "contact@contact.com"), target.get("content", "confirmation")]
    results = [evaluate(pairs, folds, kind, keywords, pcfg["bootstrap"], pcfg["seed"]) for kind in BASELINES]

    main_pairs = [p for p in pairs if p["group"] == "main"]
    benign = Counter(benign_part(p) for p in main_pairs)
    top_n = benign.most_common(1)[0][1] if benign else 0
    best = max(r["main"][0] for r in results)

    lines = [
        f"# Lexical baseline — {cfg['run_name']}", "",
        f"Pairs: `{cfg['data']['pairs_path']}` · {len(main_pairs)} main pairs, "
        f"{sum(p['group'] == 'probe_evading' for p in pairs)} probe_evading · "
        f"{pcfg['n_folds']}-fold grouped by `{pcfg.get('group_by', 'team_id')}` · team bootstrap B = {pcfg['bootstrap']}", "",
        "Classifiers that see only the document text, no model activations. Same folds and AUROC "
        "definition as the present probes, so compare directly with P1 in `headline.md`.", "",
        "| Baseline | Main AUROC [95% CI] | probe_evading AUROC [95% CI] |",
        "|---|---|---|",
    ]
    for r in results:
        ev = fmt(r["probe_evading"]) if "probe_evading" in r else "–"
        lines.append(f"| {r['baseline']} | {fmt(r['main'])} | {ev} |")
    lines += ["", f"Keywords for `keyword`: {', '.join(f'`{k}`' for k in keywords)}.", "",
              "## Negatives", "",
              f"The {len(main_pairs)} clean main documents use **{len(benign)} distinct benign emails**; "
              f"the most reused one appears in {top_n} pairs. Few distinct negatives make the clean "
              "class easy to memorize.", ""]
    tf = next(r for r in results if r["baseline"] == "word_tfidf")
    lines += ["## Words that give it away (word_tfidf, fit on all main pairs)", "",
              "- Toward injected: " + ", ".join(f"`{w}`" for w in tf["top_injected"]),
              "- Toward clean: " + ", ".join(f"`{w}`" for w in tf["top_clean"]), "",
              "## Reading", "", verdict(best), ""]
    text = "\n".join(lines)
    path = out or (run_dir(cfg) / "reports" / "lexical_baseline.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)
    return {"results": results, "distinct_benign": len(benign), "max_reuse": top_n, "path": str(path)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="output markdown path (default: <run_dir>/reports/lexical_baseline.md)")
    args = ap.parse_args(argv)
    run(load_config(args.config), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
