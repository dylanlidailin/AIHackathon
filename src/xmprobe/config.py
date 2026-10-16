"""YAML config loading and a stable config hash."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("run_name", "model", "data", "render", "generate", "ocr", "probe", "paths")


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    missing = [k for k in REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing sections {missing}")
    cfg["_path"] = str(path)
    cfg["_hash"] = config_hash(cfg)
    return cfg


def _normalized(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop private keys (``_path``, ``_hash``) so the hash depends on content only."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def config_hash(cfg: dict[str, Any], sections: tuple[str, ...] | None = None) -> str:
    """SHA-256 over canonical JSON (sorted keys). Key order in the YAML never matters.

    ``sections`` restricts the hash to a subset, e.g. the sections that affect the
    rendered images or the activations, so unrelated edits do not invalidate caches.
    """
    body = _normalized(cfg)
    if sections is not None:
        body = {k: body.get(k) for k in sections}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["scratch_root"]) / "runs" / cfg["run_name"]


def with_overrides(cfg: dict[str, Any], **updates: Any) -> dict[str, Any]:
    """Deep-copy ``cfg`` and set dotted keys, e.g. ``with_overrides(cfg, **{"render.font_size": 28})``."""
    out = copy.deepcopy(cfg)
    for dotted, value in updates.items():
        node = out
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    out["_hash"] = config_hash(out)
    return out
