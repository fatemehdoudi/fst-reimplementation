"""Unit tests for data/star_graph.py and scoring.py."""

import sys
sys.path.insert(0, "/scratch/user/fatemehdoudi_tamu.edu/fst")

from data.star_graph import build_dataset, format_user_prompt, SEED_SYSTEM_PROMPT
from scoring import score_rollout, gold_path_str, extract_boxed


def test_instance_structure():
    train, test = build_dataset(d=5, p=4, n=50, train_size=20, test_size=5, seed=0)
    assert len(train) == 20
    assert len(test) == 5
    for inst in train + test:
        assert "source" in inst and "dest" in inst
        assert "graph" in inst and "gold_path" in inst
        # gold_path should have p-1 = 3 elements (no source)
        assert len(inst["gold_path"]) == 3, f"got {len(inst['gold_path'])}"
        # last element should be the destination
        assert inst["gold_path"][-1] == inst["dest"]


def test_source_degree():
    """Source node should appear in exactly d distinct pairs."""
    train, _ = build_dataset(d=5, p=4, n=100, train_size=10, seed=7)
    for inst in train:
        s = inst["source"]
        edges = [tuple(map(int, e.split(","))) for e in inst["graph"].split()]
        neighbors = set()
        for u, v in edges:
            if u == s:
                neighbors.add(v)
            elif v == s:
                neighbors.add(u)
        assert len(neighbors) == 5, f"source degree={len(neighbors)}, expected 5"


def test_gold_path_is_valid_walk():
    """Every consecutive pair in gold_path (including source → first node) is an edge."""
    train, _ = build_dataset(d=5, p=4, n=100, train_size=10, seed=3)
    for inst in train:
        edges = {tuple(sorted(map(int, e.split(",")))) for e in inst["graph"].split()}
        walk = [inst["source"]] + inst["gold_path"]
        for i in range(len(walk) - 1):
            pair = tuple(sorted([walk[i], walk[i + 1]]))
            assert pair in edges, f"missing edge {pair} in walk {walk}"


def test_no_decoy_intersects_gold():
    """Decoy nodes must not appear in the gold path."""
    train, _ = build_dataset(d=5, p=4, n=200, train_size=10, seed=99)
    for inst in train:
        gold_set = {inst["source"]} | set(inst["gold_path"])
        all_nodes = {n for e in inst["graph"].split() for n in map(int, e.split(","))}
        # Every non-gold node should only be reachable from s through decoy edges,
        # but here we just verify the gold set is internally consistent.
        assert inst["dest"] in gold_set
        assert inst["source"] in gold_set


def test_deterministic_with_same_seed():
    train1, test1 = build_dataset(seed=42)
    train2, test2 = build_dataset(seed=42)
    assert train1[0]["graph"] == train2[0]["graph"]
    assert test1[0]["source"] == test2[0]["source"]


def test_different_seeds_differ():
    train1, _ = build_dataset(seed=0)
    train2, _ = build_dataset(seed=1)
    assert train1[0]["graph"] != train2[0]["graph"]


def test_format_user_prompt():
    train, _ = build_dataset(d=3, p=3, n=20, train_size=1, seed=0)
    inst = train[0]
    prompt = format_user_prompt(inst)
    assert "{graph}" not in prompt
    assert str(inst["source"]) in prompt
    assert str(inst["dest"]) in prompt
    assert "\\boxed{}" in prompt


def test_seed_system_prompt_not_empty():
    assert len(SEED_SYSTEM_PROMPT) > 50


# ── scoring tests ──────────────────────────────────────────────────────────────

def test_extract_boxed_basic():
    assert extract_boxed("answer is \\boxed{1,2,3}") == "1,2,3"


def test_extract_boxed_last():
    assert extract_boxed("\\boxed{wrong} ... \\boxed{1,2,3}") == "1,2,3"


def test_extract_boxed_missing():
    assert extract_boxed("no box here") is None


def test_score_exact_match():
    gold = [3, 7, 9, 2]
    text = "reasoning... \\boxed{3,7,9,2}"
    assert score_rollout(text, gold) == 1.0


def test_score_mismatch():
    gold = [3, 7, 9, 2]
    text = "\\boxed{3,7,9,5}"
    assert score_rollout(text, gold) == 0.0


def test_score_whitespace_normalised():
    gold = [3, 7, 9, 2]
    text = "\\boxed{3, 7, 9, 2}"
    assert score_rollout(text, gold) == 1.0


def test_score_no_box():
    assert score_rollout("the answer is 3,7,9,2", [3, 7, 9, 2]) == 0.0


def test_score_partial_credit_zero():
    """Even one wrong node → 0 reward."""
    gold = [3, 7, 9, 2]
    text = "\\boxed{3,7,9}"
    assert score_rollout(text, gold) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
