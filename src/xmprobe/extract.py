"""Last-prompt-token hidden states at every layer, per condition (activation-extraction spec)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .common import is_done, mark_done, stamp_key, write_json
from .config import config_hash, run_dir
from .generate import eval_pairs, inputs_for
from .prompts import CONDITIONS

log = logging.getLogger("xmprobe.extract")
SECTIONS = ("model", "render")


def extract_to(cfg: dict[str, Any], out: Path, pairs: list[dict[str, Any]], act_dir: Path) -> dict:
    from .model import get_runner

    runner = get_runner(cfg)
    bs = cfg["generate"]["batch_size"]
    act_dir.mkdir(parents=True, exist_ok=True)
    shape = None
    for condition in CONDITIONS:
        chunks = []
        for i in range(0, len(pairs), bs):
            messages, images = inputs_for(cfg, out, pairs[i:i + bs], condition)
            chunks.append(runner.last_token_states(messages, images))
        arr = np.concatenate(chunks, axis=0).astype(np.float16)
        np.save(act_dir / f"{condition}.npy", arr)
        shape = list(arr.shape)
        log.info("extract: %s %s", condition, arr.shape)
    index = {"pair_ids": [p["pair_id"] for p in pairs], "groups": [p["group"] for p in pairs],
             "team_ids": [p["team_id"] for p in pairs], "conditions": list(CONDITIONS),
             "shape": shape, "config_hash": config_hash(cfg, SECTIONS),
             "model_id": cfg["model"]["id"], "model_revision": runner.revision}
    write_json(act_dir / "index.json", index)
    return index


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    out = run_dir(cfg)
    key = stamp_key(cfg, SECTIONS, limit)
    if not force and is_done(out, "extract", key):
        log.info("extract: cache valid for config hash %s, reusing", key["hash"])
        return {"skipped": True}
    index = extract_to(cfg, out, eval_pairs(cfg, limit), out / "activations")
    mark_done(out, "extract", key, [f"activations/{c}.npy" for c in CONDITIONS] + ["activations/index.json"])
    return {"shape": index["shape"], "model_revision": index["model_revision"]}


def load_activations(act_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index = json.loads((act_dir / "index.json").read_text())
    acts = {c: np.load(act_dir / f"{c}.npy").astype(np.float32) for c in index["conditions"]}
    return acts, index
