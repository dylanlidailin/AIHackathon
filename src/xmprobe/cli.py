"""Stage dispatcher: ``python -m xmprobe.cli <stage> --config <yaml> [--limit N] [--force]``.

Every stage module exposes ``run(cfg, limit=None, force=False) -> dict``. A stage raises
``GateFailed`` when a gate blocks the stages that depend on it; ``all`` stops there.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import GateFailed
from .config import load_config, run_dir

log = logging.getLogger("xmprobe")

STAGES = {
    "validate_data": "xmprobe.data",
    "render": "xmprobe.render",
    "ocr_check": "xmprobe.ocr_check",
    "generate": "xmprobe.generate",
    "extract": "xmprobe.extract",
    "probe": "xmprobe.probes",
    "report": "xmprobe.report",
    "demo": "xmprobe.demo",
}
ALL_ORDER = ["validate_data", "render", "ocr_check", "generate", "extract", "probe", "report"]
TRACKED_PACKAGES = ("torch", "transformers", "scikit-learn", "pillow", "numpy")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _gpu() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[0] or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _font_sha256(cfg: dict[str, Any]) -> str | None:
    font = Path(cfg["render"]["font_path"])
    return hashlib.sha256(font.read_bytes()).hexdigest() if font.exists() else None


def manifest_path(cfg: dict[str, Any]) -> Path:
    return run_dir(cfg) / "run_manifest.json"


def update_manifest(cfg: dict[str, Any], stage: str, record: dict[str, Any]) -> dict[str, Any]:
    """Merge one stage record into ``run_manifest.json`` and refresh the run-level fields."""
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else {"created": _now(), "stages": {}}
    manifest.update(
        git_commit=_git_commit(),
        config={k: v for k, v in cfg.items() if not k.startswith("_")},
        config_hash=cfg["_hash"],
        config_path=cfg.get("_path"),
        packages=_versions(),
        model_id=cfg["model"]["id"],
        model_revision=manifest.get("model_revision") or cfg["model"].get("revision"),
        gpu=_gpu() or manifest.get("gpu"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        font_sha256=_font_sha256(cfg),
        updated=_now(),
    )
    if record.get("model_revision"):
        manifest["model_revision"] = record["model_revision"]
    manifest["stages"][stage] = record
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest


def run_stage(stage: str, cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict:
    module = importlib.import_module(STAGES[stage])
    started = datetime.now(timezone.utc)
    record: dict[str, Any] = {"started": started.isoformat(timespec="seconds"), "limit": limit, "force": force}
    try:
        result = module.run(cfg, limit=limit, force=force) or {}
        record.update(status="ok", **result)
        return result
    except GateFailed as e:
        record.update(status="gate_failed", gate=e.gate, detail=e.detail)
        raise
    except Exception as e:
        record.update(status="error", error=repr(e))
        raise
    finally:
        finished = datetime.now(timezone.utc)
        record.update(finished=finished.isoformat(timespec="seconds"),
                      wall_seconds=round((finished - started).total_seconds(), 2))
        update_manifest(cfg, stage, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xmprobe", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pairs")
    parser.add_argument("--force", action="store_true", help="recompute even if valid outputs exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        log.info("stage %s (config %s, hash %s)", stage, cfg["_path"], cfg["_hash"])
        try:
            run_stage(stage, cfg, limit=args.limit, force=args.force)
        except GateFailed as e:
            log.error("%s — stopping before dependent stages", e)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
