"""Shared helpers: gate errors, skip-if-valid stamps, JSONL I/O."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import config_hash

log = logging.getLogger("xmprobe")


class GateFailed(RuntimeError):
    def __init__(self, gate: str, detail: str):
        super().__init__(f"gate '{gate}' failed: {detail}")
        self.gate = gate
        self.detail = detail


# --- JSONL ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str))


# --- stamps --------------------------------------------------------------------------
# A stage writes ``<dir>/.<stage>.done.json`` after its outputs are complete. The stamp
# records the hash of the config sections the stage depends on, the --limit used, and
# the outputs it produced. A re-run skips when the stamp matches and every output exists.

def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def stamp_key(cfg: dict[str, Any], sections: tuple[str, ...], limit: int | None,
              **extra: Any) -> dict[str, Any]:
    """Identity of a stage's inputs: hashed config sections, --limit, model, the pairs
    file contents, plus any stage-specific ``extra`` values."""
    key = {"hash": config_hash(cfg, sections), "limit": limit, "model": cfg["model"]["id"],
           "pairs": file_sha256(cfg["data"]["pairs_path"])}
    key.update(extra)
    return key


def is_done(out_dir: Path, stage: str, key: dict[str, Any]) -> bool:
    stamp = out_dir / f".{stage}.done.json"
    if not stamp.exists():
        return False
    data = json.loads(stamp.read_text())
    if data.get("key") != key:
        return False
    missing = [p for p in data.get("outputs", []) if not (out_dir / p).exists()]
    if missing:
        log.warning("%s stamp present but outputs missing: %s", stage, missing[:5])
        return False
    return True


def mark_done(out_dir: Path, stage: str, key: dict[str, Any], outputs: list[str]) -> None:
    write_json(out_dir / f".{stage}.done.json", {"key": key, "outputs": sorted(outputs)})


def take(rows: list[Any], limit: int | None) -> list[Any]:
    return rows if limit is None else rows[:limit]
