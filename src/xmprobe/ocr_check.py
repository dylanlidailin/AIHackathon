"""OCR gate: can the model read the rendered clean content? (ocr-gate spec)

CER is computed on whitespace-collapsed strings, because line breaks introduced by the
renderer's wrapping are not reading errors.
"""

from __future__ import annotations

import logging
import random
import re
import statistics
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from .common import GateFailed, is_done, mark_done, stamp_key, write_json, write_jsonl
from .config import run_dir
from .data import load_pairs
from .prompts import ocr_messages
from .render import image_path, rendered_pair_ids

log = logging.getLogger("xmprobe.ocr")
SECTIONS = ("model", "render", "ocr")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def summarize(cers: list[float], min_items: int, gate: float) -> dict[str, Any]:
    n = len(cers)
    median = statistics.median(cers) if cers else float("nan")
    if n < min_items:
        return {"n": n, "median_cer": median, "pass": False, "reason": "n_below_minimum"}
    passed = median <= gate
    return {"n": n, "median_cer": median, "pass": passed, "reason": None if passed else "cer_above_gate"}


def check(cfg: dict[str, Any], out: Path, n_items: int, *, limit: int | None = None,
          pair_ids: set[str] | None = None) -> dict[str, Any]:
    """Transcribe a seeded sample of clean images under ``out`` and write ocr.jsonl + summary."""
    from .model import get_runner, load_image

    ocfg = cfg["ocr"]
    rendered = rendered_pair_ids(cfg) if out == run_dir(cfg) else {
        p.name[: -len("_clean.png")] for p in (out / "images").glob("*_clean.png")}
    pairs = [p for p in load_pairs(cfg, limit, group="main") if p["pair_id"] in rendered
             and (pair_ids is None or p["pair_id"] in pair_ids)]
    sample = random.Random(ocfg["seed"]).sample(pairs, min(n_items, len(pairs)))
    runner = get_runner(cfg)
    rows = []
    bs = cfg["generate"]["batch_size"]
    for i in range(0, len(sample), bs):
        batch = sample[i:i + bs]
        imgs = [load_image(image_path(out, p["pair_id"], "clean")) for p in batch]
        outs = runner.generate([ocr_messages()] * len(batch), imgs, ocfg["max_new_tokens"])
        for p, (text, _) in zip(batch, outs):
            rows.append({"pair_id": p["pair_id"], "reference": p["clean_content"],
                         "transcription": text, "cer": cer(p["clean_content"], text)})
    write_jsonl(out / "ocr.jsonl", rows)
    summary = summarize([r["cer"] for r in rows], ocfg["min_items"], ocfg["cer_gate"])
    summary.update(gate=ocfg["cer_gate"], min_items=ocfg["min_items"],
                   font_size=cfg["render"]["font_size"])
    write_json(out / "ocr_summary.json", summary)
    return summary


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    out = run_dir(cfg)
    key = stamp_key(cfg, SECTIONS, limit)
    if force or not is_done(out, "ocr_check", key):
        summary = check(cfg, out, cfg["ocr"]["n_items"], limit=limit)
        mark_done(out, "ocr_check", key, ["ocr.jsonl", "ocr_summary.json"])
    else:
        import json
        summary = json.loads((out / "ocr_summary.json").read_text())
        log.info("ocr_check: outputs valid, skipping")
    log.info("ocr: n=%d median_cer=%.3f pass=%s", summary["n"], summary["median_cer"], summary["pass"])
    if not summary["pass"]:
        raise GateFailed("ocr", f"n={summary['n']} median_cer={summary['median_cer']:.3f} "
                                f"reason={summary['reason']}")
    return summary
