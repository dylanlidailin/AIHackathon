"""Linear probes P1/P2/P3 with LR and DoM (probing spec, design D6).

Rows: for the ``present`` target, every pair contributes a clean (label 0) and an
injected (label 1) row per modality. For ``obey``, only injected rows, labelled by
``target_hit``, with repetition-only rows excluded. Outer folds are grouped by
``team_id`` over ``main`` pairs; ``probe_evading`` pairs are only ever scored.

Per outer fold and layer:
  P1  fit on text train rows   -> score text test rows
  P2  the same text fit        -> score image test rows
  P3  fit on image train rows  -> score image test rows
Out-of-fold scores are pooled so each main row is scored once per probe/method/layer.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .common import GateFailed, is_done, mark_done, read_jsonl, stamp_key, write_json
from .config import run_dir
from .data import grouped_folds
from .extract import load_activations

log = logging.getLogger("xmprobe.probes")
SECTIONS = ("model", "render", "probe")
PROBES = ("P1", "P2", "P3")
VARIANTS = ("raw", "centered")
CENTER_NOTE = "centering shifts all scores in a modality by a constant; AUROC unchanged"


# --- metrics -------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fold_auroc(y: np.ndarray, s: np.ndarray, fold: np.ndarray | None) -> float:
    """Test-size-weighted mean of per-fold AUROCs. Pooling raw scores across fold models
    would mix each model's own offset (e.g. w_k . image_offset for P2) into the ranking
    and bias cross-modal AUROC downward. ``fold=None`` ranks all rows together."""
    if fold is None:
        return auroc(y, s)
    num = den = 0.0
    for k in np.unique(fold):
        m = fold == k
        a = auroc(y[m], s[m])
        if a == a:
            num += a * m.sum()
            den += m.sum()
    return num / den if den else float("nan")


def bootstrap_sets(groups: np.ndarray, B: int, seed: int) -> list[np.ndarray]:
    """B resamples of whole groups (teams) with replacement, as row-index arrays."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_by = {g: np.flatnonzero(groups == g) for g in uniq}
    return [np.concatenate([rows_by[g] for g in rng.choice(uniq, size=len(uniq))]) for _ in range(B)]


def auroc_ci(y: np.ndarray, s: np.ndarray, sets: list[np.ndarray],
             fold: np.ndarray | None = None) -> tuple[float, float]:
    vals = [fold_auroc(y[ix], s[ix], None if fold is None else fold[ix]) for ix in sets]
    vals = [v for v in vals if v == v]
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --- methods -------------------------------------------------------------------------

@dataclass
class Fitted:
    method: str
    model: Any            # sklearn pipeline (lr) or direction vector (dom)
    threshold: float      # chosen on training-modality data
    inner_auroc: float    # inner grouped-CV AUROC on the training modality
    C: float | None = None

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.method == "lr":
            return self.model.decision_function(X)
        return X @ self.model


def _inner_splits(groups: np.ndarray, n_splits: int):
    from sklearn.model_selection import GroupKFold

    n = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)) if n >= 2 else []


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def fit(method: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, pcfg: dict[str, Any]) -> Fitted:
    """Fit one method on training rows only. ``groups`` drive the inner CV."""
    splits = _inner_splits(groups, pcfg["inner_folds"])
    if method == "dom":
        d = dom_direction(X, y)
        proj = X @ d
        thr = float((proj[y == 1].mean() + proj[y == 0].mean()) / 2)
        inner = [auroc(y[te], X[te] @ dom_direction(X[tr], y[tr])) for tr, te in splits
                 if len(np.unique(y[tr])) == 2]
        inner = [a for a in inner if a == a]
        return Fitted("dom", d, thr, float(np.mean(inner)) if inner else float("nan"))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    usable = [(tr, te) for tr, te in splits
              if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2]
    if usable:
        gs = GridSearchCV(pipe, {"logisticregression__C": pcfg["C_grid"]}, cv=usable,
                          scoring="roc_auc", n_jobs=pcfg.get("n_jobs", 1), refit=True)
        gs.fit(X, y)
        return Fitted("lr", gs.best_estimator_, 0.0, float(gs.best_score_),
                      C=gs.best_params_["logisticregression__C"])
    C = pcfg["C_grid"][len(pcfg["C_grid"]) // 2]
    pipe.set_params(logisticregression__C=C).fit(X, y)
    return Fitted("lr", pipe, 0.0, float("nan"), C=C)


# --- row construction ----------------------------------------------------------------

@dataclass
class Rows:
    """One modality's rows for one target: activations [n, L+1, H], labels, metadata."""
    X: np.ndarray
    y: np.ndarray
    pair_idx: np.ndarray   # index into the activation index's pair list
    kind: np.ndarray       # "clean" | "injected"


def present_rows(acts: dict[str, np.ndarray], modality: str, idx: np.ndarray) -> Rows:
    clean, inj = acts[f"{modality}_clean"][idx], acts[f"{modality}_injected"][idx]
    n = len(idx)
    return Rows(np.concatenate([clean, inj]), np.r_[np.zeros(n, int), np.ones(n, int)],
                np.r_[idx, idx], np.array(["clean"] * n + ["injected"] * n))


def obey_labels(gens: list[dict[str, Any]], pair_ids: list[str], modality: str) -> tuple[np.ndarray, int]:
    """Per pair: 1 target_hit, 0 neither, -1 excluded (repetition only or missing)."""
    by = {g["pair_id"]: g for g in gens if g["condition"] == f"{modality}_injected"}
    labels, excluded = [], 0
    for pid in pair_ids:
        g = by.get(pid)
        if g is None:
            labels.append(-1)
        elif g["target_hit"]:
            labels.append(1)
        elif g["repetition"]:
            labels.append(-1)
            excluded += 1
        else:
            labels.append(0)
    return np.array(labels), excluded


def obey_rows(acts: dict[str, np.ndarray], modality: str, idx: np.ndarray, labels: np.ndarray) -> Rows:
    keep = idx[labels[idx] >= 0]
    return Rows(acts[f"{modality}_injected"][keep], labels[keep], keep, np.array(["injected"] * len(keep)))


# --- cross-validation ----------------------------------------------------------------

def _subset(rows: Rows, pair_set: set[int]) -> Rows:
    m = np.array([p in pair_set for p in rows.pair_idx], dtype=bool)
    return Rows(rows.X[m], rows.y[m], rows.pair_idx[m], rows.kind[m])


def _clean_mean(rows: Rows, layer: int) -> np.ndarray:
    m = rows.kind == "clean"
    return rows.X[m, layer].mean(axis=0) if m.any() else rows.X[:, layer].mean(axis=0)


def cross_validate(text: Rows, image: Rows, ev_text: Rows | None, ev_image: Rows | None,
                   folds: list[tuple[set[int], set[int]]], teams: np.ndarray,
                   pcfg: dict[str, Any], methods: list[str], fit_fn=None) -> dict[str, Any]:
    """Pooled out-of-fold scores and fold-wise predictions for every probe/method/layer."""
    fit_fn = fit_fn or fit
    n_layers = text.X.shape[1]
    res: dict[str, Any] = {}

    def slot(key, n):
        res.setdefault(key, {"score": np.full((n_layers, n), np.nan), "pred": np.full((n_layers, n), np.nan),
                             "fold": np.full(n, -1)})
        return res[key]

    index_of = lambda rows: {(p, k): i for i, (p, k) in enumerate(zip(rows.pair_idx, rows.kind))}
    pos_text, pos_image = index_of(text), index_of(image)
    inner = {m: np.full((len(folds), n_layers), np.nan) for m in methods}  # P3 layer choice
    ev_scores: dict[str, list] = {}

    for f, (train_pairs, test_pairs) in enumerate(folds):
        tr_t, te_t = _subset(text, train_pairs), _subset(text, test_pairs)
        tr_i, te_i = _subset(image, train_pairs), _subset(image, test_pairs)
        g_t, g_i = teams[tr_t.pair_idx], teams[tr_i.pair_idx]
        rows_t = [pos_text[(p, k)] for p, k in zip(te_t.pair_idx, te_t.kind)]
        rows_i = [pos_image[(p, k)] for p, k in zip(te_i.pair_idx, te_i.kind)]
        for layer in range(n_layers):
            mu_t, mu_i = _clean_mean(tr_t, layer), _clean_mean(tr_i, layer)
            for method in methods:
                for probe, tr, g, te, rows, mu_tr, mu_te, n in (
                    ("P1", tr_t, g_t, te_t, rows_t, mu_t, mu_t, len(text.y)),
                    ("P3", tr_i, g_i, te_i, rows_i, mu_i, mu_i, len(image.y)),
                ):
                    if len(np.unique(tr.y)) < 2 or len(te.y) == 0:
                        continue
                    model = fit_fn(method, tr.X[:, layer], tr.y, g, pcfg)
                    if probe == "P3":
                        inner[method][f, layer] = model.inner_auroc
                    targets = [(probe, te, rows, mu_te, n)]
                    if probe == "P1":
                        targets.append(("P2", te_i, rows_i, mu_i, len(image.y)))
                    for name, tset, trows, mu_mod, nrows in targets:
                        s_raw = model.score(tset.X[:, layer])
                        s_cen = model.score(tset.X[:, layer] - mu_mod + mu_tr)
                        for variant, s in (("raw", s_raw), ("centered", s_cen)):
                            sl = slot((name, method, variant), nrows)
                            sl["score"][layer, trows] = s
                            sl["pred"][layer, trows] = (s > model.threshold).astype(float)
                            sl["fold"][trows] = f
                    if probe == "P1":
                        for ev_name, ev in (("P1", ev_text), ("P2", ev_image)):
                            if ev is not None and len(ev.y):
                                ev_scores.setdefault((ev_name, method), []).append(
                                    (layer, model.score(ev.X[:, layer])))
    # average the fold models' scores on the probe_evading rows
    ev_out = {}
    for (name, method), items in ev_scores.items():
        n = len((ev_text if name == "P1" else ev_image).y)
        acc = np.zeros((n_layers, n))
        cnt = np.zeros(n_layers)
        for layer, s in items:
            acc[layer] += s
            cnt[layer] += 1
        ev_out[(name, method)] = acc / np.maximum(cnt, 1)[:, None]
    return {"oof": res, "inner_p3": {m: np.nanmean(v, axis=0) for m, v in inner.items()}, "evading": ev_out}


# --- stage ---------------------------------------------------------------------------

def _rows_out(target, probe, method, variant, eval_group, rows: Rows, scores, preds, sets, teams,
              pcfg, headline_layer, fold=None):
    out = []
    n_pos, n_neg = int(rows.y.sum()), int((rows.y == 0).sum())
    low_power = min(n_pos, n_neg) < pcfg["low_power_min"]
    too_few = min(n_pos, n_neg) < pcfg["min_class_for_auroc"]
    for layer in range(scores.shape[0]):
        s = scores[layer]
        if variant == "centered" or too_few:
            a, lo, hi = (float("nan"),) * 3
        else:
            a = fold_auroc(rows.y, s, fold)
            lo, hi = auroc_ci(rows.y, s, sets, fold)
        acc = float(np.nanmean(preds[layer] == rows.y)) if preds is not None and np.isfinite(preds[layer]).any() else float("nan")
        out.append({"target": target, "probe": probe, "method": method, "variant": variant,
                    "eval_group": eval_group, "layer": layer, "auroc": a, "ci_lo": lo, "ci_hi": hi,
                    "acc": acc, "n_pos": n_pos, "n_neg": n_neg, "low_power": low_power,
                    "headline_layer": headline_layer, "is_headline": layer == headline_layer,
                    "note": CENTER_NOTE if variant == "centered" else ("auroc omitted: class < "
                            f"{pcfg['min_class_for_auroc']}" if too_few else "")})
    return out


def d_prime(y: np.ndarray, s: np.ndarray) -> float:
    a, b = s[y == 1], s[y == 0]
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))


def select_layer(aucs: list[float], scores: np.ndarray, y: np.ndarray, tie_tol: float) -> int | None:
    """Headline layer from training-modality (text) data only: the best P1 AUROC; layers
    within ``tie_tol`` of it count as tied (P1 saturates at 1.0 in most layers) and the tie
    is broken by the largest text d' of the out-of-fold P1 scores."""
    a = np.asarray(aucs, dtype=float)
    if not np.isfinite(a).any():
        return None
    tied = np.flatnonzero(a >= np.nanmax(a) - tie_tol)
    return int(max(tied, key=lambda l: d_prime(y, scores[l])))


def _write_scores(path: Path, rows: Rows, pair_ids: list[str], scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "kind", "label", *[f"L{l}" for l in range(scores.shape[0])]])
        for j in range(len(rows.y)):
            w.writerow([pair_ids[rows.pair_idx[j]], rows.kind[j], int(rows.y[j]),
                        *[f"{v:.6g}" for v in scores[:, j]]])


def run_target(target: str, acts, index, gens, cfg, out_dir: Path, fit_fn=None) -> dict[str, Any]:
    pcfg = cfg["probe"]
    pair_ids, groups = index["pair_ids"], np.array(index["groups"])
    teams = np.array(index["team_ids"])
    main_idx, ev_idx = np.flatnonzero(groups == "main"), np.flatnonzero(groups == "probe_evading")
    if target == "present":
        build = lambda mod, idx: present_rows(acts, mod, idx)
        counts = {}
    else:
        labels = {m: obey_labels(gens, pair_ids, m) for m in ("text", "image")}
        build = lambda mod, idx: obey_rows(acts, mod, idx, labels[mod][0])
        counts = {f"n_excluded_repetition_{m}": labels[m][1] for m in labels}

    text, image = build("text", main_idx), build("image", main_idx)
    ev_text = build("text", ev_idx) if len(ev_idx) else None
    ev_image = build("image", ev_idx) if len(ev_idx) else None

    pair_dicts = [{"group": g, "team_id": t} for g, t in zip(groups, teams)]
    folds = [(set(tr), set(te)) for tr, te in grouped_folds(pair_dicts, pcfg["n_folds"], pcfg["seed"])]
    cv = cross_validate(text, image, ev_text, ev_image, folds, teams, pcfg, pcfg["methods"], fit_fn)

    out_rows, headline = [], {}
    sets = {name: bootstrap_sets(teams[r.pair_idx], pcfg["bootstrap"], pcfg["seed"])
            for name, r in (("text", text), ("image", image), ("ev_text", ev_text), ("ev_image", ev_image))
            if r is not None and len(r.y)}
    for method in pcfg["methods"]:
        p1 = cv["oof"].get(("P1", method, "raw"))
        p1_auc = ([fold_auroc(text.y, p1["score"][l], p1["fold"]) for l in range(p1["score"].shape[0])]
                  if p1 else [])
        l_star = select_layer(p1_auc, p1["score"], text.y, pcfg.get("tie_tol", 0.001)) if p1 else None
        inner = cv["inner_p3"].get(method)
        l_p3 = int(np.nanargmax(inner)) if inner is not None and np.isfinite(inner).any() else None
        headline[method] = {"P1": l_star, "P2": l_star, "P3": l_p3}
        for probe in PROBES:
            rows = text if probe == "P1" else image
            for variant in VARIANTS:
                sl = cv["oof"].get((probe, method, variant))
                if sl is None:
                    continue
                out_rows += _rows_out(target, probe, method, variant, "main", rows, sl["score"], sl["pred"],
                                      sets["text" if probe == "P1" else "image"], teams, pcfg,
                                      headline[method][probe], fold=sl["fold"])
                if variant == "raw":
                    _write_scores(out_dir / "scores" / f"{target}_{probe}_{method}_main.csv",
                                  rows, pair_ids, sl["score"])
        for probe, rows, key in (("P1", ev_text, "ev_text"), ("P2", ev_image, "ev_image")):
            s = cv["evading"].get((probe, method))
            if s is None or rows is None:
                continue
            out_rows += _rows_out(target, probe, method, "raw", "probe_evading", rows, s, None,
                                  sets[key], teams, pcfg, headline[method][probe])
            _write_scores(out_dir / "scores" / f"{target}_{probe}_{method}_probe_evading.csv",
                          rows, pair_ids, s)

    path = out_dir / f"{target}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    class_counts = {m: {"pos": int(r.y.sum()), "neg": int((r.y == 0).sum())}
                    for m, r in (("text", text), ("image", image))}
    meta = {"target": target, "headline_layers": headline, "class_counts": class_counts,
            "n_folds": len(folds), **counts}
    write_json(out_dir / f"{target}_meta.json", meta)
    return meta


def run_probes(cfg: dict[str, Any], act_dir: Path, out_dir: Path, targets: list[str],
               gens_path: Path | None) -> dict[str, Any]:
    acts, index = load_activations(act_dir)
    gens = list(read_jsonl(gens_path)) if gens_path and gens_path.exists() else []
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for target in targets:
        if target == "obey":
            skipped = _obey_skipped(gens, act_dir.parent)
            if skipped:
                write_json(out_dir / "obey_meta.json", {"target": "obey", "skipped": True, "reason": skipped})
                results["obey"] = {"skipped": True, "reason": skipped}
                continue
        results[target] = run_target(target, acts, index, gens, cfg, out_dir)
        log.info("probe %s done: headline layers %s", target, results[target]["headline_layers"])
    return results


def _obey_skipped(gens: list[dict[str, Any]], run: Path) -> str | None:
    if not gens:
        return "no generations"
    import json
    summary = run / "behavior_summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("main", {}).get("obey_probes_skipped"):
        return "text_injected target rate below 5%"
    return None


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    out = run_dir(cfg)
    act_dir = out / "activations"
    if not (act_dir / "index.json").exists():
        raise GateFailed("activations", f"no activation cache at {act_dir}")
    key = stamp_key(cfg, SECTIONS, limit)
    if not force and is_done(out, "probe", key):
        log.info("probe: outputs valid, skipping")
        return {"skipped": True}
    results = run_probes(cfg, act_dir, out / "probes", ["present", "obey"], out / "generations.jsonl")
    outputs = [f"probes/{t}_meta.json" for t in results] + \
              [f"probes/{t}.csv" for t, r in results.items() if not r.get("skipped")]
    mark_done(out, "probe", key, outputs)
    return {"targets": {t: r.get("headline_layers", r) for t, r in results.items()}}
