"""Length / image-size confound check for the present probes (post-hoc, CPU only).

Usage: python scripts/confound_check.py --config configs/qwen35_2b.yaml [--run-dir DIR]
  DIR defaults to the config's run dir; pass <run>/demo for the demo.

Reports, for each modality:
  1. Trivial-baseline AUROC of length features alone for clean (0) vs injected (1):
     character length of the content, and rendered image height (image token count grows
     with height at fixed width). A high baseline means length alone separates the classes.
  2. Spearman correlation between each probe's headline-layer score and those features,
     within each class, so it is not driven by the label itself.
Writes <dir>/reports/confound_check.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from xmprobe.common import read_jsonl
from xmprobe.config import load_config, run_dir
from xmprobe.probes import auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = Path(args.run_dir) if args.run_dir else run_dir(cfg)
    pairs = {p["pair_id"]: p for p in read_jsonl(cfg["data"]["pairs_path"])}
    meta = json.loads((base / "probes" / "present_meta.json").read_text())

    def features(pid: str, kind: str) -> dict[str, float]:
        with Image.open(base / "images" / f"{pid}_{kind}.png") as im:
            height = im.height
        return {"chars": len(pairs[pid][f"{kind}_content"]), "image_height": height}

    lines = [f"# Confound check — {base}", "",
             "AUROC of each feature alone for clean (0) vs injected (1); 0.5 = no information, "
             "values far from 0.5 in either direction mean the feature separates the classes.", ""]
    rows = []
    for probe in ("P1", "P2", "P3"):
        for method in cfg["probe"]["methods"]:
            path = base / "probes" / "scores" / f"present_{probe}_{method}_main.csv"
            if not path.exists():
                continue
            layer = meta["headline_layers"][method][probe]
            data = list(csv.DictReader(path.open()))
            y = np.array([int(r["label"]) for r in data])
            s = np.array([float(r[f"L{layer}"]) for r in data])
            feats = [features(r["pair_id"], r["kind"]) for r in data]
            for name in ("chars", "image_height"):
                f = np.array([x[name] for x in feats], dtype=float)
                within = []
                for cls in (0, 1):
                    m = y == cls
                    rho = spearmanr(s[m], f[m]).statistic if m.sum() > 2 else float("nan")
                    within.append(rho)
                rows.append((probe, method, layer, name, auroc(y, f), *within))
    lines += ["| Probe | Method | Layer | Feature | Feature-only AUROC | ρ(score, feature) clean | ρ injected |",
              "|---|---|---|---|---|---|---|"]
    for probe, method, layer, name, a, r0, r1 in rows:
        lines.append(f"| {probe} | {method} | {layer} | {name} | {a:.3f} | {r0:.2f} | {r1:.2f} |")
    out = base / "reports" / "confound_check.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
