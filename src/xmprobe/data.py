"""Pairs schema validation (``validate_data`` stage), loading, and grouped folds."""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import GateFailed, read_jsonl, take

log = logging.getLogger("xmprobe.data")

SCHEMA: dict[str, type | tuple[type, ...]] = {
    "pair_id": str,
    "source": str,
    "system_prompt": str,
    "user_task": str,
    "clean_content": str,
    "injected_content": str,
    "injection_text": str,
    "target_phrase": str,
    "group": str,
    "level": str,
    "team_id": str,
    "attack_success_phi3": bool,
}
GROUPS = ("main", "probe_evading")


class SchemaError(ValueError):
    pass


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Raise SchemaError naming the first bad row (pair_id or line number) and field."""
    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        where = f"row {i} (pair_id={row.get('pair_id')!r})" if row.get("pair_id") else f"line {i}"
        for key, typ in SCHEMA.items():
            if key not in row:
                raise SchemaError(f"{where}: missing field '{key}'")
            if not isinstance(row[key], typ):
                raise SchemaError(f"{where}: field '{key}' must be {getattr(typ, '__name__', typ)}, "
                                  f"got {type(row[key]).__name__}")
        if row["group"] not in GROUPS:
            raise SchemaError(f"{where}: field 'group' must be one of {GROUPS}")
        for key in ("clean_content", "injected_content", "user_task", "target_phrase"):
            if not row[key].strip():
                raise SchemaError(f"{where}: field '{key}' is empty")
        if row["injection_text"] not in row["injected_content"]:
            raise SchemaError(f"{where}: field 'injection_text' not found in 'injected_content'")
        if row["pair_id"] in seen:
            raise SchemaError(f"{where}: duplicate pair_id")
        seen.add(row["pair_id"])


def load_pairs(cfg: dict[str, Any], limit: int | None = None,
               group: str | None = None) -> list[dict[str, Any]]:
    """Pairs in file order (deterministic), optionally one group, first ``limit`` rows."""
    rows = list(read_jsonl(cfg["data"]["pairs_path"]))
    if group is not None:
        rows = [r for r in rows if r["group"] == group]
    return take(rows, limit)


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    path = Path(cfg["data"]["pairs_path"])
    if not path.exists():
        raise GateFailed("schema", f"pairs file not found: {path}")
    rows = list(read_jsonl(path))
    try:
        validate_rows(rows)
    except SchemaError as e:
        raise GateFailed("schema", str(e)) from e
    counts = dict(Counter(r["group"] for r in rows))
    log.info("pairs ok: %d rows %s", len(rows), counts)
    return {"n_rows": len(rows), "by_group": counts}


# --- grouped folds -------------------------------------------------------------------

def grouped_folds(pairs: list[dict[str, Any]], n_folds: int, seed: int,
                  group_by: str = "team_id") -> list[tuple[list[int], list[int]]]:
    """K folds over ``main`` pairs, grouped by ``group_by``; ``probe_evading`` never appears.

    Groups are shuffled with ``seed`` and assigned largest-first to the currently
    smallest fold, so folds are balanced in pairs, not just in groups. Returns
    (train_idx, test_idx) as indices into ``pairs``.
    """
    idx_by_group: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        if p["group"] == "main":
            idx_by_group[p[group_by]].append(i)
    groups = sorted(idx_by_group)
    if len(groups) < n_folds:
        raise ValueError(f"only {len(groups)} distinct {group_by} values for {n_folds} folds")
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -len(idx_by_group[g]))  # stable: shuffle breaks size ties
    fold_items: list[list[int]] = [[] for _ in range(n_folds)]
    fold_groups: list[set[str]] = [set() for _ in range(n_folds)]
    for g in groups:
        k = min(range(n_folds), key=lambda j: (len(fold_items[j]), j))
        fold_items[k].extend(idx_by_group[g])
        fold_groups[k].add(g)

    main_idx = sorted(i for items in fold_items for i in items)
    folds = []
    for k in range(n_folds):
        test = sorted(fold_items[k])
        test_set = set(test)
        train = [i for i in main_idx if i not in test_set]
        train_groups = {pairs[i][group_by] for i in train}
        assert not train_groups & fold_groups[k], "group leaked across train/test"
        folds.append((train, test))
    return folds
