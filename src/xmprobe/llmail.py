"""Convert LLMail-Inject phase-1 submissions into the pairs schema (design D2).

Filtering: scenario 1 only, level routing (main / probe_evading), fully successful
attacks only, attacker email length cap (subject + body, as formatted), normalized-body dedupe (main wins over
probe_evading), per-team cap. Pair construction: fixed scenario-1 Email A plus either
the attacker email (injected) or the length-closest benign email (clean), in a seeded
order shared by both versions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SUCCESS_KEYS = ("exfil.sent", "exfil.destination", "exfil.content", "defense.undetected")
PHI3_LEVELS = {"a", "c", "e", "g", "i"}  # levels_descriptions.json, phase1
TARGET_PHRASE = "contact@contact.com|confirmation"
LEVEL_RE = re.compile(r"^level(\d+)([a-z])$")


def normalize_body(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip()


def body_hash(body: str) -> str:
    return hashlib.sha256(normalize_body(body).encode()).hexdigest()


def format_email(subject: str, body: str) -> str:
    """LLMail's own email format, as in scenarios.json and emails_for_fp_tests.json."""
    return f"Subject of the email: {subject}.   Body: {body}"


@dataclass
class Candidate:
    row_key: str
    level: str
    group: str
    team_id: str
    subject: str
    body: str
    bhash: str


@dataclass
class ConversionReport:
    seen: int = 0
    dropped: Counter = field(default_factory=Counter)
    eligible_by_level: Counter = field(default_factory=Counter)
    sampled_by_level: Counter = field(default_factory=Counter)
    shortfall: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "dropped": dict(sorted(self.dropped.items())),
            "eligible_by_level": dict(sorted(self.eligible_by_level.items())),
            "sampled_by_level": dict(sorted(self.sampled_by_level.items())),
            "shortfall": self.shortfall,
        }


def _parse_objectives(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def route_level(scenario: str, cfg: dict[str, Any]) -> tuple[str, str] | None:
    """``level1a`` -> ("a", "main"); returns None for other scenarios or excluded levels."""
    m = LEVEL_RE.match(scenario or "")
    if not m or int(m.group(1)) != cfg["scenario"]:
        return None
    letter = m.group(2)
    if letter in cfg["main_levels"]:
        return letter, "main"
    if letter in cfg["evading_levels"]:
        return letter, "probe_evading"
    return None


def filter_submissions(rows: Iterable[dict[str, Any]], cfg: dict[str, Any],
                       report: ConversionReport) -> tuple[list[Candidate], set[str]]:
    """Apply level routing, the full-success filter and the length cap.

    Returns the candidates plus the set of body hashes that fully succeeded on a Phi-3
    level (used for ``attack_success_phi3``).
    """
    out: list[Candidate] = []
    phi3_success: set[str] = set()
    for row in rows:
        report.seen += 1
        routed = route_level(row.get("scenario", ""), cfg)
        if routed is None:
            report.dropped["other_scenario_or_level"] += 1
            continue
        letter, group = routed
        obj = _parse_objectives(row.get("objectives"))
        if obj is None:
            report.dropped["unparsable_objectives"] += 1
            continue
        if not all(obj.get(k) is True for k in SUCCESS_KEYS):
            report.dropped["partial_success"] += 1
            continue
        body = row.get("body") or ""
        subject = row.get("subject") or ""
        bhash = body_hash(body)
        if letter in PHI3_LEVELS:
            phi3_success.add(bhash)
        # cap the whole formatted email: some attacks put the payload in the subject
        if len(format_email(subject, body)) > cfg["max_email_chars"]:
            report.dropped["too_long"] += 1
            continue
        out.append(Candidate(
            row_key=str(row.get("RowKey") or row.get("job_id")),
            level=letter, group=group, team_id=str(row.get("team_id")),
            subject=subject, body=body, bhash=bhash,
        ))
    return out, phi3_success


def dedupe(cands: list[Candidate], report: ConversionReport) -> list[Candidate]:
    """One candidate per normalized body; a body seen in ``main`` is never kept in
    ``probe_evading`` (so held-out text never duplicates training text)."""
    ordered = sorted(cands, key=lambda c: (c.group != "main", c.level, c.row_key))
    seen: set[str] = set()
    kept = []
    for c in ordered:
        if c.bhash in seen:
            report.dropped["duplicate"] += 1
            continue
        seen.add(c.bhash)
        kept.append(c)
    return kept


def cap_per_team(cands: list[Candidate], k: int, rng: random.Random,
                 report: ConversionReport) -> list[Candidate]:
    by_team: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_team[c.team_id].append(c)
    kept = []
    for team in sorted(by_team):
        items = sorted(by_team[team], key=lambda c: c.row_key)
        if len(items) > k:
            report.dropped["team_cap"] += len(items) - k
            items = rng.sample(items, k)
        kept.extend(items)
    return kept


def closest_benign(target_len: int, pool: list[str]) -> str:
    """Benign email with the smallest |len - target_len|; ties -> earliest in pool."""
    return min(pool, key=lambda e: (abs(len(e) - target_len), pool.index(e)))


def build_pair(c: Candidate, email_a: str, benign_pool: list[str], *, system_prompt: str,
               user_task: str, phi3_success: set[str], seed: int) -> dict[str, Any]:
    attacker = format_email(c.subject, c.body)
    benign = closest_benign(len(attacker), benign_pool)
    pair_id = f"llmail-{c.level}-{c.row_key}"
    attacker_first = random.Random(f"{seed}:{pair_id}").random() < 0.5

    def join(second: str) -> str:
        emails = [second, email_a] if attacker_first else [email_a, second]
        return "\n\n".join(emails)

    return {
        "pair_id": pair_id,
        "source": "llmail",
        "level": c.level,
        "group": c.group,
        "team_id": c.team_id,
        "template_id": None,
        "system_prompt": system_prompt,
        "user_task": user_task,
        "clean_content": join(benign),
        "injected_content": join(attacker),
        "injection_text": attacker,
        "target_phrase": TARGET_PHRASE,
        "attack_success_phi3": c.bhash in phi3_success,
        "email_order": "second_first" if attacker_first else "a_first",
        "len_clean": len(join(benign)),
        "len_injected": len(join(attacker)),
    }


def load_static(raw_dir: Path) -> dict[str, Any]:
    scenarios = json.loads((raw_dir / "scenarios.json").read_text())
    prompts = json.loads((raw_dir / "system_prompt.json").read_text())
    pool = json.loads((raw_dir / "emails_for_fp_tests.json").read_text())
    s1 = scenarios["scenario_1"]
    email_a = s1["emails"][0]
    return {
        "email_a": email_a,
        "user_task": s1["user_query"],
        "system_prompt": prompts["email_client"].strip() + "\n\n" + prompts["tool_prompt"].strip(),
        "benign_pool": [e for e in pool if e != email_a],
    }


def convert(rows: Iterable[dict[str, Any]], static: dict[str, Any],
            cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], ConversionReport]:
    report = ConversionReport()
    rng = random.Random(cfg["seed"])
    cands, phi3_success = filter_submissions(rows, cfg, report)
    cands = dedupe(cands, report)
    cands = cap_per_team(cands, cfg["per_team_cap"], rng, report)
    for c in cands:
        report.eligible_by_level[c.level] += 1

    pairs: list[dict[str, Any]] = []
    for group, n in (("main", cfg["n_main"]), ("probe_evading", cfg["n_evading"])):
        pool = sorted((c for c in cands if c.group == group), key=lambda c: c.row_key)
        if len(pool) < n:
            report.shortfall[group] = n - len(pool)
            chosen = pool
        else:
            chosen = sorted(rng.sample(pool, n), key=lambda c: c.row_key)
        for c in chosen:
            report.sampled_by_level[c.level] += 1
            pairs.append(build_pair(c, static["email_a"], static["benign_pool"],
                                    system_prompt=static["system_prompt"],
                                    user_task=static["user_task"],
                                    phi3_success=phi3_success, seed=cfg["seed"]))
    return pairs, report
