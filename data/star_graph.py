"""
Star-graph dataset construction (Appendix C).

Each instance has source degree d, path length p (nodes), node pool size n.
Headline config: (d, p, n) = (25, 20, 500), train=10 000, test=200.
"""

from __future__ import annotations

import random
from typing import TypedDict

# ── Verbatim prompt template (Appendix C) ─────────────────────────────────────
USER_TEMPLATE = (
    "Given a bi-directional graph in the form of space separated edges, "
    "output a path from source node to the destination node in the form of "
    "comma separated integers.\n"
    "For this question the graph is {graph}\n"
    "The source node is {source}\n"
    "The destination node is {destination}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

# Verbatim seed system prompt (Appendix C); also used as GEPA seed candidate
SEED_SYSTEM_PROMPT = (
    "You are solving a graph path-finding task. You will be given a list of edges "
    "and a source and destination node. Output one valid path from source to "
    "destination. Inspect the source node's neighbors first, identify which neighbor "
    "leads to the destination via a sequence of valid edges, then commit to that "
    "branch. Each consecutive pair in your output path must be a valid edge in the "
    "graph. Put your final answer comma-separated inside boxed braces."
)


class StarGraphInstance(TypedDict):
    source: int
    dest: int
    graph: str          # space-separated "u,v" edge pairs (shuffled)
    gold_path: list[int]  # [v1, …, v_{p-2}, goal] — what the model must output


def _make_instance(d: int, p: int, n: int, rng: random.Random) -> StarGraphInstance:
    """
    Build one star-graph instance.

    Gold path: s → v1 → … → v_{p-2} → g  (p nodes, p-1 edges).
    The model must output v1, …, v_{p-2}, g (p-1 values, NOT the source).

    Decoys: d-1 chains of length p rooted at s; nodes drawn from unused pool
    so no decoy intersects the gold path or another decoy.
    """
    pool = list(range(n))
    rng.shuffle(pool)

    # Draw source and goal (distinct)
    s = pool.pop()
    g = pool.pop()

    # Draw p-2 intermediate gold-path nodes
    intermediates = [pool.pop() for _ in range(p - 2)]
    gold_nodes = [s] + intermediates + [g]  # length p
    gold_edges = [(gold_nodes[i], gold_nodes[i + 1]) for i in range(p - 1)]

    # Draw d-1 decoy chains of length p (p-1 new nodes each, rooted at s)
    decoy_edges: list[tuple[int, int]] = []
    for _ in range(d - 1):
        chain_nodes = [s] + [pool.pop() for _ in range(p - 1)]
        decoy_edges.extend(
            (chain_nodes[i], chain_nodes[i + 1]) for i in range(p - 1)
        )

    # Merge, shuffle, serialize
    all_edges = gold_edges + decoy_edges
    rng.shuffle(all_edges)
    graph_str = " ".join(f"{u},{v}" for u, v in all_edges)

    return StarGraphInstance(
        source=s,
        dest=g,
        graph=graph_str,
        gold_path=intermediates + [g],  # v1, …, v_{p-2}, g
    )


def build_dataset(
    d: int = 25,
    p: int = 20,
    n: int = 500,
    train_size: int = 10_000,
    test_size: int = 200,
    seed: int = 42,
) -> tuple[list[StarGraphInstance], list[StarGraphInstance]]:
    """Return (train_split, test_split) generated with a fixed seed."""
    rng = random.Random(seed)
    train = [_make_instance(d, p, n, rng) for _ in range(train_size)]
    test = [_make_instance(d, p, n, rng) for _ in range(test_size)]
    return train, test


def format_user_prompt(inst: StarGraphInstance) -> str:
    return USER_TEMPLATE.format(
        graph=inst["graph"],
        source=inst["source"],
        destination=inst["dest"],
    )
