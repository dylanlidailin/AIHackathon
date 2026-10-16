import csv

import numpy as np
import pytest

from xmprobe import probes
from xmprobe.common import write_json
from xmprobe.config import with_overrides
from xmprobe.probes import (Fitted, auroc, bootstrap_sets, obey_labels, run_target)

CONDS = ("text_clean", "text_injected", "image_clean", "image_injected")


def synthetic(n_main=60, n_ev=10, n_teams=15, layers=4, hidden=12, signal=2.5, image_signal=None, seed=0):
    """Activations with an injection signal along one direction at layers >= 1 and a
    constant image offset. Returns (acts, index)."""
    rng = np.random.default_rng(seed)
    n = n_main + n_ev
    d = rng.normal(size=hidden)
    d /= np.linalg.norm(d)
    offset = rng.normal(size=hidden) * 4
    acts = {}
    for c in CONDS:
        x = rng.normal(size=(n, layers + 1, hidden))
        modality, kind = c.split("_")
        if kind == "injected":
            s = signal if modality == "text" else (signal if image_signal is None else image_signal)
            x[:, 1:, :] += s * d
        if modality == "image":
            x += offset
        acts[c] = x
    groups = ["main"] * n_main + ["probe_evading"] * n_ev
    teams = [f"t{i % n_teams}" for i in range(n_main)] + [f"e{i % 3}" for i in range(n_ev)]
    index = {"pair_ids": [f"p{i}" for i in range(n)], "groups": groups, "team_ids": teams,
             "conditions": list(CONDS)}
    return acts, index


@pytest.fixture
def pcfg_cfg(run_cfg):
    return with_overrides(run_cfg, **{"probe.bootstrap": 50, "probe.n_jobs": 1})


def read(path):
    return list(csv.DictReader(path.open()))


def test_auroc_basics():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2])))
    sets = bootstrap_sets(np.array(["a", "a", "b", "c"]), 5, seed=0)
    assert len(sets) == 5 and all(len(s) >= 3 for s in sets)


def test_planted_signal_transfers(pcfg_cfg, tmp_path):
    acts, index = synthetic()
    meta = run_target("present", acts, index, [], pcfg_cfg, tmp_path)
    rows = read(tmp_path / "present.csv")
    for method in ("lr", "dom"):
        for probe in ("P1", "P2", "P3"):
            head = [r for r in rows if r["method"] == method and r["probe"] == probe and r["variant"] == "raw"
                    and r["eval_group"] == "main" and r["is_headline"] == "True"]
            assert len(head) == 1, (method, probe)
            assert float(head[0]["auroc"]) > 0.95, (method, probe, head[0]["auroc"])
            assert float(head[0]["ci_lo"]) <= float(head[0]["auroc"]) <= float(head[0]["ci_hi"])
        assert meta["headline_layers"][method]["P1"] != 0  # embedding layer carries no signal
    # both methods, all layers, evading rows for P1/P2 only
    assert {r["method"] for r in rows} == {"lr", "dom"}
    assert {int(r["layer"]) for r in rows} == set(range(5))
    assert {r["probe"] for r in rows if r["eval_group"] == "probe_evading"} == {"P1", "P2"}
    assert (tmp_path / "scores" / "present_P2_dom_main.csv").exists()


def test_centered_rows_report_accuracy_only(pcfg_cfg, tmp_path):
    acts, index = synthetic()
    run_target("present", acts, index, [], pcfg_cfg, tmp_path)
    rows = read(tmp_path / "present.csv")
    cen = [r for r in rows if r["variant"] == "centered"]
    assert cen and all(r["auroc"] == "nan" and "AUROC unchanged" in r["note"] for r in cen)
    # the image offset breaks the raw DoM threshold on P2; centering repairs accuracy
    pick = lambda v: next(float(r["acc"]) for r in rows if r["probe"] == "P2" and r["method"] == "dom"
                          and r["variant"] == v and r["is_headline"] == "True" and r["eval_group"] == "main")
    assert pick("centered") > pick("raw")


def test_no_transfer_when_image_signal_absent(pcfg_cfg, tmp_path):
    acts, index = synthetic(image_signal=0.0)
    run_target("present", acts, index, [], pcfg_cfg, tmp_path)
    rows = read(tmp_path / "present.csv")
    head = lambda p: next(float(r["auroc"]) for r in rows if r["probe"] == p and r["method"] == "lr"
                          and r["variant"] == "raw" and r["is_headline"] == "True" and r["eval_group"] == "main")
    assert head("P1") > 0.95 and head("P2") < 0.75 and head("P3") < 0.75


def test_fits_only_see_training_teams(pcfg_cfg, tmp_path):
    """DoM (and LR) directions are computed only from the fold's training team_ids, and
    probe_evading rows never reach a fit."""
    acts, index = synthetic()
    teams = np.array(index["team_ids"])
    seen = []
    real_fit = probes.fit

    def recording_fit(method, X, y, groups, pcfg):
        seen.append((method, set(groups)))
        return real_fit(method, X, y, groups, pcfg)

    from xmprobe.data import grouped_folds
    pair_dicts = [{"group": g, "team_id": t} for g, t in zip(index["groups"], index["team_ids"])]
    folds = grouped_folds(pair_dicts, 5, 42)
    train_team_sets = [{teams[i] for i in tr} for tr, _ in folds]
    test_team_sets = [{teams[i] for i in te} for _, te in folds]
    run_target("present", acts, index, [], pcfg_cfg, tmp_path, fit_fn=recording_fit)
    assert {m for m, _ in seen} == {"lr", "dom"}
    for _, g in seen:
        assert not any(t.startswith("e") for t in g)  # evading teams
        assert any(g == tr for tr in train_team_sets)
        assert not any(g & te for te, tr in zip(test_team_sets, train_team_sets) if g == tr)


def test_p2_layer_choice_ignores_images(pcfg_cfg, tmp_path):
    acts, index = synthetic()
    meta_a = run_target("present", acts, index, [], pcfg_cfg, tmp_path / "a")
    scrambled = dict(acts)
    rng = np.random.default_rng(9)
    scrambled["image_injected"] = rng.normal(size=acts["image_injected"].shape) * 10
    scrambled["image_clean"] = rng.normal(size=acts["image_clean"].shape) * 10
    meta_b = run_target("present", scrambled, index, [], pcfg_cfg, tmp_path / "b")
    for m in ("lr", "dom"):
        assert meta_a["headline_layers"][m]["P2"] == meta_b["headline_layers"][m]["P2"]


def test_obey_labels_exclude_repetition():
    gens = [
        {"pair_id": "a", "condition": "text_injected", "target_hit": True, "repetition": True},
        {"pair_id": "b", "condition": "text_injected", "target_hit": False, "repetition": True},
        {"pair_id": "c", "condition": "text_injected", "target_hit": False, "repetition": False},
        {"pair_id": "a", "condition": "image_injected", "target_hit": False, "repetition": False},
    ]
    labels, excluded = obey_labels(gens, ["a", "b", "c", "d"], "text")
    assert labels.tolist() == [1, -1, 0, -1] and excluded == 1


def test_obey_low_power_and_omitted_auroc(pcfg_cfg, tmp_path):
    acts, index = synthetic()
    gens = []
    for i, pid in enumerate(index["pair_ids"]):
        gens.append({"pair_id": pid, "condition": "text_injected", "target_hit": i % 2 == 0, "repetition": False})
        gens.append({"pair_id": pid, "condition": "image_injected", "target_hit": i in (0, 2, 4), "repetition": i == 1})
    meta = run_target("obey", acts, index, gens, pcfg_cfg, tmp_path)
    rows = read(tmp_path / "obey.csv")
    img = [r for r in rows if r["probe"] in ("P2", "P3") and r["eval_group"] == "main"]
    assert img and all(r["low_power"] == "True" for r in img)
    assert all(r["auroc"] == "nan" and "omitted" in r["note"] for r in img if r["variant"] == "raw")
    assert meta["n_excluded_repetition_image"] == 1
    assert meta["class_counts"]["image"]["pos"] == 3
