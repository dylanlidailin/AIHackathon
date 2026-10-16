"""Illustration of "shared direction, shifted threshold" (post-hoc, CPU only).

Usage: python scripts/threshold_figure.py --config configs/qwen35_2b.yaml [--run-dir DIR]

For each method, fits one probe on ALL main text pairs at that method's headline layer
(one model, so every score shares one scale), scores text and image inputs, and draws the
four score distributions with the text threshold and, for illustration only, the best
cutoff chosen on the image scores themselves. Writes <dir>/reports/threshold_shift.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xmprobe.config import load_config, run_dir
from xmprobe.extract import load_activations
from xmprobe.probes import fit

CLEAN, INJ = "#2a78d6", "#eb6834"  # dataviz reference slots 1 and 2 (validated)
INK, MUTED, GRID = "#1c2230", "#5a6476", "#e8e7e2"


def best_cutoff(clean: np.ndarray, inj: np.ndarray) -> tuple[float, float]:
    """Cutoff with the highest accuracy on these scores (in-sample, illustration only)."""
    cands = np.sort(np.r_[clean, inj])
    mids = (cands[:-1] + cands[1:]) / 2
    accs = [((clean <= t).sum() + (inj > t).sum()) / (len(clean) + len(inj)) for t in mids]
    k = int(np.argmax(accs))
    return float(mids[k]), float(accs[k])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = Path(args.run_dir) if args.run_dir else run_dir(cfg)
    acts, index = load_activations(base / "activations")
    meta = json.loads((base / "probes" / "present_meta.json").read_text())
    groups = np.array(index["groups"])
    teams = np.array(index["team_ids"])
    main_idx = np.flatnonzero(groups == "main")
    methods = cfg["probe"]["methods"]

    fig, axes = plt.subplots(len(methods), 1, figsize=(8, 2.9 * len(methods)), dpi=150)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    rows = [("text", "clean"), ("text", "injected"), ("image", "clean"), ("image", "injected")]
    for ax, m in zip(axes, methods):
        layer = meta["headline_layers"][m]["P1"]
        X = np.concatenate([acts["text_clean"][main_idx, layer], acts["text_injected"][main_idx, layer]])
        y = np.r_[np.zeros(len(main_idx), int), np.ones(len(main_idx), int)]
        model = fit(m, X, y, np.r_[teams[main_idx], teams[main_idx]], cfg["probe"])
        scores = {(mod, k): model.score(acts[f"{mod}_{k}"][main_idx, layer]) for mod, k in rows}
        img_cut, img_acc = best_cutoff(scores[("image", "clean")], scores[("image", "injected")])
        txt_acc_on_img = float(np.mean(np.r_[scores[("image", "clean")] <= model.threshold,
                                             scores[("image", "injected")] > model.threshold]))
        for i, (mod, k) in enumerate(rows):
            s = scores[(mod, k)]
            yy = (3 - i) + rng.uniform(-0.18, 0.18, len(s))
            ax.scatter(s, yy, s=9, color=CLEAN if k == "clean" else INJ, alpha=0.75, linewidths=0)
        ax.axvline(model.threshold, color=INK, linewidth=1.5, linestyle="--")
        ax.axvline(img_cut, color=MUTED, linewidth=1.2, linestyle=":")
        ax.text(model.threshold, 3.98, f" text threshold\n (image accuracy {txt_acc_on_img:.2f})",
                fontsize=7, color=INK, va="top")
        ax.text(img_cut, 3.98, f" best image cutoff, in-sample\n (accuracy {img_acc:.2f}; illustration)",
                fontsize=7, color=MUTED, va="top")
        ax.set_yticks([3, 2, 1, 0], ["text · clean", "text · injected", "image · clean", "image · injected"],
                      fontsize=8)
        ax.set_ylim(-0.5, 4.1)
        ax.axhline(1.5, color=GRID, linewidth=1)
        ax.set_title(f"{'LR' if m == 'lr' else 'DoM'} · layer {layer} · probe fit on all {len(main_idx)} "
                     f"text pairs (images not used)", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("probe score (higher = more injection-like)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=CLEAN, label="clean"),
               Line2D([], [], marker="o", linestyle="", color=INJ, label="injected")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f"{cfg['run_name']} · same direction, shifted threshold", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = base / "reports" / "threshold_shift.png"
    fig.savefig(out, metadata={"Software": None})
    print("wrote", out)


if __name__ == "__main__":
    main()
