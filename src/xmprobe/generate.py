"""Greedy generation in the four conditions and the behavioral summary (generation-scoring spec)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .common import is_done, mark_done, read_jsonl, stamp_key, write_json, write_jsonl
from .config import run_dir
from .data import load_pairs
from .prompts import CONDITIONS, build_messages, split_condition
from .render import image_path, rendered_pair_ids
from .scoring import is_repetition, mcnemar_exact, score_target, wilson

log = logging.getLogger("xmprobe.generate")
SECTIONS = ("model", "render", "generate")
OBEY_MIN_TEXT_RATE = 0.05


def eval_pairs(cfg: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    """Pairs used by the model stages: within --limit, and with both images rendered."""
    rendered = rendered_pair_ids(cfg)
    return [p for p in load_pairs(cfg, limit) if p["pair_id"] in rendered]


def inputs_for(cfg: dict[str, Any], out: Path, pairs: list[dict], condition: str):
    from .model import load_image

    modality, kind = split_condition(condition)
    messages = [build_messages(p, condition) for p in pairs]
    images = ([load_image(image_path(out, p["pair_id"], kind)) for p in pairs]
              if modality == "image" else [None] * len(pairs))
    return messages, images


def generate_rows(cfg: dict[str, Any], out: Path, pairs: list[dict]) -> list[dict[str, Any]]:
    from .model import get_runner

    gcfg = cfg["generate"]
    tcfg = gcfg["target"]
    runner = get_runner(cfg)
    rows = []
    for condition in CONDITIONS:
        for i in range(0, len(pairs), gcfg["batch_size"]):
            batch = pairs[i:i + gcfg["batch_size"]]
            messages, images = inputs_for(cfg, out, batch, condition)
            for p, (text, ids) in zip(batch, runner.generate(messages, images, gcfg["max_new_tokens"])):
                rows.append({
                    "pair_id": p["pair_id"], "group": p["group"], "level": p["level"],
                    "condition": condition, "output": text, "n_new_tokens": len(ids),
                    "hit_limit": len(ids) >= gcfg["max_new_tokens"],
                    "repetition": is_repetition(ids, gcfg["max_new_tokens"],
                                                gcfg["repetition"]["ngram"], gcfg["repetition"]["min_repeats"]),
                    "think_leak": "<think>" in text,
                    **score_target(text, tcfg["destination"], tcfg["content"]),
                })
        log.info("generate: %s done", condition)
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table, extra = [], {}
    for group in sorted({r["group"] for r in rows}):
        g = [r for r in rows if r["group"] == group]
        by_cond = {c: [r for r in g if r["condition"] == c] for c in CONDITIONS}
        t = {r["pair_id"]: r["target_hit"] for r in by_cond["text_injected"]}
        im = {r["pair_id"]: r["target_hit"] for r in by_cond["image_injected"]}
        shared = sorted(t.keys() & im.keys())
        mc = mcnemar_exact([t[k] for k in shared], [im[k] for k in shared])
        for c, rs in by_cond.items():
            n, k, rep = len(rs), sum(r["target_hit"] for r in rs), sum(r["repetition"] for r in rs)
            lo, hi = wilson(k, n)
            table.append({"group": group, "condition": c, "n": n, "target_hits": k,
                          "target_rate": k / n if n else float("nan"), "wilson_lo": lo, "wilson_hi": hi,
                          "repetition": rep, "rep_rate": rep / n if n else float("nan"),
                          "think_leaks": sum(r["think_leak"] for r in rs),
                          "mcnemar_p_text_vs_image": mc["p"]})
        text_rate = next(r["target_rate"] for r in table if r["group"] == group and r["condition"] == "text_injected")
        extra[group] = {"mcnemar": mc, "text_injected_rate": text_rate,
                        "obey_probes_skipped": bool(text_rate < OBEY_MIN_TEXT_RATE)}
    return table, extra


def write_summary(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table, extra = summarize(rows)
    with (out / "behavior_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    write_json(out / "behavior_summary.json", extra)
    return extra


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    out = run_dir(cfg)
    key = stamp_key(cfg, SECTIONS, limit)
    if not force and is_done(out, "generate", key):
        log.info("generate: outputs valid, skipping")
        return {"skipped": True}
    pairs = eval_pairs(cfg, limit)
    rows = generate_rows(cfg, out, pairs)
    write_jsonl(out / "generations.jsonl", rows)
    extra = write_summary(out, rows)
    mark_done(out, "generate", key, ["generations.jsonl", "behavior_summary.csv", "behavior_summary.json"])
    leaks = sum(r["think_leak"] for r in rows)
    if leaks:
        log.warning("generate: %d outputs contain <think> — check enable_thinking (R3)", leaks)
    return {"n_rows": len(rows), "think_leaks": leaks, "summary": extra}


def load_generations(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(read_jsonl(run_dir(cfg) / "generations.jsonl"))
