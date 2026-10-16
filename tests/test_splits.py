import random

import pytest

from xmprobe.data import grouped_folds


def make_pairs(n_main=60, n_evading=15, n_teams=17, seed=0):
    rng = random.Random(seed)
    pairs = [{"pair_id": f"m{i}", "group": "main", "team_id": f"t{rng.randrange(n_teams)}"}
             for i in range(n_main)]
    pairs += [{"pair_id": f"e{i}", "group": "probe_evading", "team_id": f"t{rng.randrange(n_teams)}"}
              for i in range(n_evading)]
    rng.shuffle(pairs)
    return pairs


def test_no_team_on_both_sides():
    pairs = make_pairs()
    for train, test in grouped_folds(pairs, 5, seed=42):
        assert not {pairs[i]["team_id"] for i in train} & {pairs[i]["team_id"] for i in test}


def test_evading_never_in_any_fold():
    pairs = make_pairs()
    for train, test in grouped_folds(pairs, 5, seed=42):
        assert all(pairs[i]["group"] == "main" for i in train + test)


def test_each_main_pair_tested_exactly_once():
    pairs = make_pairs()
    tested = [i for _, test in grouped_folds(pairs, 5, seed=42) for i in test]
    assert sorted(tested) == sorted(i for i, p in enumerate(pairs) if p["group"] == "main")


def test_deterministic_and_balanced():
    pairs = make_pairs()
    assert grouped_folds(pairs, 5, seed=42) == grouped_folds(pairs, 5, seed=42)
    sizes = [len(t) for _, t in grouped_folds(pairs, 5, seed=42)]
    assert max(sizes) - min(sizes) <= max(len([p for p in pairs if p["team_id"] == t])
                                          for t in {p["team_id"] for p in pairs})


def test_too_few_groups():
    pairs = [{"pair_id": str(i), "group": "main", "team_id": "t0" if i < 5 else "t1"} for i in range(10)]
    with pytest.raises(ValueError, match="distinct team_id"):
        grouped_folds(pairs, 5, seed=42)
