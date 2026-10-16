"""Fast sampled signal check (demo-run spec): render -> mini OCR -> extract -> present probes -> report.

Everything is written under ``<run>/demo/`` so the demo never touches the main run's
caches. The mini OCR check has fewer than the gate's minimum items, so image results are
labelled "OCR not verified" and no outcome is classified.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

from .common import is_done, mark_done, stamp_key, write_json
from .config import run_dir
from .data import load_pairs
from .extract import extract_to
from .ocr_check import check as ocr_check
from .probes import run_probes
from .render import render_pairs
from .report import build as build_report

log = logging.getLogger("xmprobe.demo")
SECTIONS = ("model", "render", "ocr", "probe", "demo")


def stratified_sample(pairs: list[dict[str, Any]], n: int, key: str, seed: int) -> list[dict[str, Any]]:
    """Proportional allocation per ``key`` value (largest remainder), seeded draw within strata.
    Returned in input order."""
    if n >= len(pairs):
        return list(pairs)
    strata: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        strata[str(p.get(key))].append(i)
    quotas = {k: n * len(v) / len(pairs) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = sorted(i for k in sorted(strata) for i in rng.sample(strata[k], alloc[k]))
    return [pairs[i] for i in chosen]


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    dcfg = cfg["demo"]
    out = run_dir(cfg) / "demo"
    n = limit or dcfg["n_pairs"]
    key = stamp_key(cfg, SECTIONS, n)
    if not force and is_done(out, "demo", key):
        log.info("demo: outputs valid, skipping")
        return {"skipped": True}

    sample = stratified_sample(load_pairs(cfg, group="main"), n, dcfg["stratify_by"], dcfg["seed"])
    write_json(out / "sample.json", {"pair_ids": [p["pair_id"] for p in sample],
                                     "stratify_by": dcfg["stratify_by"], "seed": dcfg["seed"]})
    written, dropped = render_pairs(sample, out, cfg["render"])
    kept = [p for p in sample if p["pair_id"] not in {d["pair_id"] for d in dropped}]
    log.info("demo: %d pairs (%d dropped oversize)", len(kept), len(dropped))

    ocr = ocr_check(cfg, out, dcfg["ocr_items"], pair_ids={p["pair_id"] for p in kept})
    extract_to(cfg, out, kept, out / "activations")
    run_probes(cfg, out / "activations", out / "probes", ["present"], None)
    verdict = build_report(cfg, out, demo=True)
    mark_done(out, "demo", key, ["reports/headline.md", "reports/layer_sweep_present.png", "probes/present.csv"])
    return {"n_pairs": len(kept), "ocr_median_cer": ocr["median_cer"], "reason": verdict["reason"]}
