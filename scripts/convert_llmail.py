"""Build data/pairs.jsonl from LLMail-Inject phase 1 (design D2).

Usage:
    python scripts/convert_llmail.py --config configs/qwen35_2b.yaml [--raw-dir DIR] [--out PATH]

``--raw-dir`` holds the downloaded ``data/`` folder of microsoft/llmail-inject-challenge.
Writes the pairs file and ``<out>.report.json`` with candidate counts per level and per
drop reason. Prints a warning when a group has fewer candidates than requested.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from xmprobe.common import read_jsonl, write_json, write_jsonl
from xmprobe.config import load_config
from xmprobe.llmail import convert, load_static

log = logging.getLogger("convert_llmail")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw-dir", help="overrides data.llmail.raw_dir")
    ap.add_argument("--out", help="overrides data.pairs_path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    lcfg = cfg["data"]["llmail"]
    raw_dir = Path(args.raw_dir or lcfg["raw_dir"])
    out = Path(args.out or cfg["data"]["pairs_path"])

    static = load_static(raw_dir)
    pairs, report = convert(read_jsonl(raw_dir / "raw_submissions_phase1.jsonl"), static, lcfg)
    write_jsonl(out, pairs)
    rep = report.to_dict()
    write_json(out.with_suffix(".report.json"), rep)

    log.info("wrote %d pairs to %s", len(pairs), out)
    log.info("eligible by level: %s", rep["eligible_by_level"])
    log.info("dropped: %s", rep["dropped"])
    for group, missing in rep["shortfall"].items():
        log.warning("%s: %d fewer candidates than requested (see design.md Risks for fallbacks)",
                    group, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
