"""Cross-model comparison against pre-specified criteria C1-C5 (cross-model-comparison spec).

Usage:
  python scripts/compare_models.py --runs RUN_DIR [RUN_DIR ...] --out results/cross_model

Each RUN_DIR is a pipeline run directory (or a copy of one) containing probes/present.csv,
probes/obey.csv or probes/obey_meta.json, behavior_summary.csv, run_manifest.json and either
activations/ or reports/direction_cosine.csv (written here from activations when present).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

SHARED_SECTIONS = ("data", "render", "probe", "generate", "ocr")
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz reference slots 1-4
METHODS = ("lr", "dom")


class MismatchError(ValueError):
    pass


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


# --- loading ---------------------------------------------------------------------------

def direction_cosines(run: Path) -> list[tuple[int, float]]:
    """Per-layer cosine between text and image DoM directions (main pairs). Cached as CSV."""
    cache = run / "reports" / "direction_cosine.csv"
    if cache.exists():
        return [(int(r["layer"]), _f(r["cosine"])) for r in _read_csv(cache)]
    idx = json.loads((run / "activations" / "index.json").read_text())
    main = np.array(idx["groups"]) == "main"
    acts = {c: np.load(run / "activations" / f"{c}.npy", mmap_mode="r") for c in idx["conditions"]}
    out = []
    for layer in range(1, acts["text_clean"].shape[1]):
        get = lambda c: np.asarray(acts[c][main, layer], dtype=np.float64).mean(axis=0)
        dt, di = get("text_injected") - get("text_clean"), get("image_injected") - get("image_clean")
        out.append((layer, float(dt @ di / (np.linalg.norm(dt) * np.linalg.norm(di) + 1e-12))))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cosine"])
        w.writerows(out)
    return out


def load_run(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "run_manifest.json").read_text())
    present = [r for r in _read_csv(run / "probes" / "present.csv") if r["variant"] == "raw" and r["eval_group"] == "main"]
    if not present:
        raise FileNotFoundError(f"{run}: no probes/present.csv")
    n_layers = max(int(r["layer"]) for r in present)
    obey_meta_p = run / "probes" / "obey_meta.json"
    obey_meta = json.loads(obey_meta_p.read_text()) if obey_meta_p.exists() else {}
    pairs_hash = None
    stamp = run / ".extract.done.json"
    if stamp.exists():
        pairs_hash = json.loads(stamp.read_text())["key"].get("pairs")
    return {
        "dir": run, "name": manifest["config"]["run_name"], "model": manifest["model_id"],
        "config": manifest["config"], "pairs_hash": pairs_hash, "n_layers": n_layers,
        "present": present,
        "obey": [r for r in _read_csv(run / "probes" / "obey.csv") if r["variant"] == "raw" and r["eval_group"] == "main"],
        "obey_skipped": bool(obey_meta.get("skipped")),
        "behavior": _read_csv(run / "behavior_summary.csv"),
        "cosine": direction_cosines(run),
    }


def check_inputs(runs: list[dict[str, Any]]) -> None:
    ref = runs[0]
    for r in runs[1:]:
        for sec in SHARED_SECTIONS:
            a = {k: v for k, v in ref["config"][sec].items() if k != "llmail"} if sec == "data" else ref["config"][sec]
            b = {k: v for k, v in r["config"][sec].items() if k != "llmail"} if sec == "data" else r["config"][sec]
            if a != b:
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                raise MismatchError(f"{r['name']} differs from {ref['name']} in config section '{sec}': {diff}")
        if ref["pairs_hash"] and r["pairs_hash"] and ref["pairs_hash"] != r["pairs_hash"]:
            raise MismatchError(f"{r['name']} differs from {ref['name']} in pairs hash")


# --- criteria ----------------------------------------------------------------------------

def thirds(layer: int, n: int) -> str:
    rel = layer / n
    return "early" if rel <= 1 / 3 else ("middle" if rel <= 2 / 3 else "late")


def headline(rows, probe, method):
    for r in rows:
        if r["probe"] == probe and r["method"] == method and r["is_headline"] == "True":
            return r
    return None


def sweep(rows, probe, method) -> list[tuple[int, float]]:
    return sorted((int(r["layer"]), _f(r["auroc"])) for r in rows if r["probe"] == probe and r["method"] == method)


def criteria(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n, P = run["n_layers"], run["present"]
    res: dict[str, dict[str, Any]] = {}
    heads = {m: headline(P, "P2", m) for m in METHODS}

    c1 = all(h and _f(h["auroc"]) >= 0.90 and _f(h["ci_lo"]) > 0.5 for h in heads.values())
    res["C1"] = {"pass": c1, "detail": ", ".join(f"{m.upper()} L{h['layer']} {_f(h['auroc']):.3f} [{_f(h['ci_lo']):.3f}]"
                                                for m, h in heads.items() if h)}
    if c1:
        accs = {m: _f(h["acc"]) for m, h in heads.items()}
        res["C2"] = {"pass": any(a < 0.70 for a in accs.values()),
                     "detail": ", ".join(f"{m.upper()} acc {a:.2f}" for m, a in accs.items())}
    else:
        res["C2"] = {"pass": None, "detail": "C1 not met"}

    c3, det = False, []
    for m in METHODS:
        s = sweep(P, "P2", m)
        mid = [a for l, a in s if l > 0 and thirds(l, n) == "middle" and a == a]
        late = [a for l, a in s if l > 0 and thirds(l, n) == "late" and a == a]
        if mid and late:
            ok = min(mid) < 0.70 and statistics.median(late) >= 0.90
            c3 = c3 or ok
            det.append(f"{m.upper()} mid-min {min(mid):.2f}, late-median {statistics.median(late):.2f}")
    res["C3"] = {"pass": c3, "detail": "; ".join(det)}

    mid = [c for l, c in run["cosine"] if thirds(l, n) == "middle"]
    late = [c for l, c in run["cosine"] if thirds(l, n) == "late"]
    gap = statistics.median(late) - statistics.median(mid)
    res["C4"] = {"pass": gap >= 0.2, "detail": f"cos median middle {statistics.median(mid):.2f}, late "
                                               f"{statistics.median(late):.2f} (gap {gap:+.2f})"}

    if run["obey_skipped"] or not run["obey"]:
        res["C5"] = {"pass": None, "detail": "obey probes skipped"}
    else:
        oh = {m: headline(run["obey"], "P2", m) for m in METHODS}
        if any(h is None or _f(h["auroc"]) != _f(h["auroc"]) for h in oh.values()):
            res["C5"] = {"pass": None, "detail": "obey AUROC not computed (too few positives)"}
        else:
            inc = {m: _f(h["ci_lo"]) <= 0.5 <= _f(h["ci_hi"]) for m, h in oh.items()}
            res["C5"] = {"pass": all(inc.values()),
                         "detail": ", ".join(f"{m.upper()} {_f(h['auroc']):.2f} [{_f(h['ci_lo']):.2f}, {_f(h['ci_hi']):.2f}]"
                                             for m, h in oh.items())}
    return res


# --- output ------------------------------------------------------------------------------

def mark(v) -> str:
    return "not evaluable" if v is None else ("pass" if v else "fail")


def plots(runs, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    def style(ax, ylabel):
        for x in (1 / 3, 2 / 3):
            ax.axvline(x, color="#b5b4ad", linewidth=1, linestyle=":")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel("relative depth (layer / number of layers)")
        ax.set_ylabel(ylabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for color, r in zip(MODEL_COLORS, runs):
        for m, ls in (("lr", "-"), ("dom", "--")):
            pts = [(l / r["n_layers"], a) for l, a in sweep(r["present"], "P2", m) if l > 0]
            ax.plot(*zip(*pts), ls, color=color, linewidth=2, label=f"{r['model'].split('/')[-1]} {m.upper()}")
    ax.axhline(0.5, color="#b5b4ad", linewidth=1, linestyle=":")
    style(ax, "P2 AUROC (text-trained probe on images)")
    ax.set_title("P2 by relative depth · solid = LR, dashed = DoM · dotted lines = thirds", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    fig.savefig(out / "p2_sweep_relative.png", metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for color, r in zip(MODEL_COLORS, runs):
        pts = [(l / r["n_layers"], c) for l, c in r["cosine"]]
        ax.plot(*zip(*pts), "-", color=color, linewidth=2, label=r["model"].split("/")[-1])
    ax.axhline(0, color="#b5b4ad", linewidth=1)
    style(ax, "cosine(text, image) injection direction")
    ax.set_title("Do text and image encode injection along the same direction?", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    fig.savefig(out / "direction_cosine_relative.png", metadata={"Software": None})
    plt.close(fig)


def report(runs, results, out: Path) -> str:
    names = [r["model"].split("/")[-1] for r in runs]
    lines = ["# Cross-model comparison", "",
             "Same pairs, renderer, probe settings and headline-layer rule for every model. Criteria C1–C5 were "
             "fixed before the Qwen3.5-4B and Qwen2.5-VL-3B runs (see the replicate-across-models spec).", "",
             "| Model | Layers | " + " | ".join(f"C{i}" for i in range(1, 6)) + " |", "|---|---|" + "---|" * 5]
    for r, n, res in zip(runs, names, results):
        lines.append(f"| {n} | {r['n_layers']} | " + " | ".join(mark(res[f'C{i}']['pass']) for i in range(1, 6)) + " |")
    lines += ["", "C1 late transfer · C2 threshold shift · C3 mid-depth dip · C4 direction alignment rises late · "
              "C5 obey does not transfer", "", "## Shared pattern", ""]
    for i in range(1, 6):
        vals = [res[f"C{i}"]["pass"] for res in results]
        if all(v is True for v in vals):
            lines.append(f"- **C{i}: shared** by all models.")
        else:
            lines.append(f"- C{i}: not shared: " + ", ".join(f"{n} {mark(v)}" for n, v in zip(names, vals)) + ".")
    lines += ["", "## Numbers behind each decision", ""]
    for n, res in zip(names, results):
        lines.append(f"**{n}**")
        lines += [f"- C{i}: {mark(res[f'C{i}']['pass'])} ({res[f'C{i}']['detail']})" for i in range(1, 6)]
        lines.append("")
    lines += ["## Behavior", "", "| Model | Group | Text target rate [Wilson] | Image target rate [Wilson] | McNemar p |",
              "|---|---|---|---|---|"]
    for r, n in zip(runs, names):
        for g in ("main", "probe_evading"):
            row = {b["condition"]: b for b in r["behavior"] if b["group"] == g}
            if not row:
                continue
            t, i = row["text_injected"], row["image_injected"]
            fmt = lambda b: f"{_f(b['target_rate']):.3f} [{_f(b['wilson_lo']):.3f}, {_f(b['wilson_hi']):.3f}]"
            lines.append(f"| {n} | {g} | {fmt(t)} | {fmt(i)} | {_f(t['mcnemar_p_text_vs_image']):.3f} |")
    lines += ["", "Figures: `p2_sweep_relative.png`, `direction_cosine_relative.png`."]
    text = "\n".join(lines) + "\n"
    (out / "comparison.md").write_text(text)
    (out / "comparison.json").write_text(json.dumps(
        {n: {k: v for k, v in res.items()} for n, res in zip(names, results)}, indent=2, sort_keys=True))
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs = [load_run(Path(p)) for p in args.runs]
    check_inputs(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [criteria(r) for r in runs]
    plots(runs, out)
    print(report(runs, results, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
