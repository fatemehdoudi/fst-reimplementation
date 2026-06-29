from __future__ import annotations

import os
from typing import Callable

from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    GEPAResult,
    ReflectionConfig,
    optimize_anything,
)

from config import FSTConfig

_STR_KEY = "current_candidate"

_OBJECTIVE = (
    "Find a system prompt that maximizes the star-graph path-finding reward. "
    "The prompt should instruct the model to inspect the source node's neighbors first, "
    "identify the neighbor leading toward the destination via a sequence of valid edges, "
    "and commit to that branch. Output comma-separated nodes inside \\boxed{}."
)


def _build_frontier_state(
    candidates: list[str],
    val_subscores: list[dict],
    run_dir: str,
    frontier_type: str = "hybrid",
) -> None:
    """Write a fresh gepa_state.bin pre-populated with all K candidates.

    Candidates and their per-instance scores come from the previous cycle's GEPAResult.
    total_num_evals is set to 0 so the next optimize_anything call receives a completely
    fresh budget — this is the core of Option C: population injected, budget reset.

    DataIds in val_subscores are integer indices (0..len(anchor)-1) and match the
    current anchor regardless of which examples are at those positions, so GEPA will
    treat the injected scores as a valid prior for candidate selection.
    """
    assert candidates and len(candidates) == len(val_subscores)

    # objective_scores_by_val_id must be a non-empty dict (even with empty inner dicts)
    # to satisfy GEPAState.__init__'s frontier_type='hybrid' guard. This mirrors what
    # the optimize_anything adapter does for scalar evaluators: each example gets {}.
    def _make_obj_scores(subscores: dict) -> dict:
        return {k: {} for k in subscores}

    seed_eval = ValsetEvaluation(
        outputs_by_val_id={},
        scores_by_val_id=dict(val_subscores[0]),
        objective_scores_by_val_id=_make_obj_scores(val_subscores[0]),
    )
    state = GEPAState(
        seed_candidate={_STR_KEY: candidates[0]},
        base_evaluation=seed_eval,
        frontier_type=frontier_type,
    )
    # Set budget counters that GEPAState.__init__ leaves unset.
    state.num_full_ds_evals = 1
    state.total_num_evals = 0  # fresh budget: remaining = max_metric_calls - 0

    for cand_text, subscores in zip(candidates[1:], val_subscores[1:]):
        cand_eval = ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id=dict(subscores),
            objective_scores_by_val_id=_make_obj_scores(subscores),
        )
        state.update_state_with_new_program(
            parent_program_idx=[0],
            new_program={_STR_KEY: cand_text},
            valset_evaluation=cand_eval,
            run_dir=None,
            num_metric_calls_by_discovery_of_new_program=0,
        )

    os.makedirs(run_dir, exist_ok=True)
    state.save(run_dir)


def _extract_top_k(result: GEPAResult, K: int) -> list[str]:
    """Return up to K best-scoring candidates; pad by repeating the best if fewer exist."""
    n = len(result.candidates)
    ranked = sorted(range(n), key=lambda i: -result.val_aggregate_scores[i])
    texts = [result.candidates[i][_STR_KEY] for i in ranked[:K]]
    while len(texts) < K:
        texts.append(texts[0])
    return texts


def run(
    policy_fn: Callable[[str, dict], float],
    anchor: list[dict],
    seed_prompts: list[str],
    K: int,
    run_dir: str,
    cfg: FSTConfig,
    prev_result: GEPAResult | None = None,
) -> tuple[list[str], GEPAResult]:
    """Run one GEPA fast-loop cycle and return (new_Phi, GEPAResult).

    Option C warm-start: when prev_result is provided, the full ranked frontier from the
    previous cycle (up to K candidates) is injected into a fresh run_dir before
    optimize_anything is called. GEPA loads the pre-populated state, starts with the
    stale scores as a selection prior, and uses a completely fresh metric-call budget to
    explore and score new proposals under the current policy.

    Args:
        policy_fn: evaluator(candidate_str, example_dict) -> float wrapping the current
            vLLM policy. Called by GEPA for each (candidate, example) pair it evaluates.
        anchor: fixed dataset slice for this cycle (cfg.gepa_eval_examples examples).
        seed_prompts: current population Phi, ordered best-first (K elements).
        K: population size (must match len(seed_prompts) on first cycle).
        run_dir: per-cycle directory; receives gepa_state.bin.
        cfg: FSTConfig supplying reflection_lm and gepa_max_metric_calls.
        prev_result: GEPAResult from the previous cycle, or None for cycle 0.

    Returns:
        (new_Phi, result) where new_Phi is a K-element list sorted best-first.
    """
    engine_cfg = EngineConfig(
        run_dir=run_dir,
        max_metric_calls=cfg.gepa_max_metric_calls,
        display_progress_bar=False,
    )

    # Option C injection: pre-populate run_dir with the full ranked frontier from the
    # previous cycle so GEPA starts with population diversity, not just one seed.
    if prev_result is not None and prev_result.candidates:
        n = len(prev_result.candidates)
        ranked = sorted(range(n), key=lambda i: -prev_result.val_aggregate_scores[i])
        top_k_idx = ranked[: min(K, n)]
        cand_texts = [prev_result.candidates[i][_STR_KEY] for i in top_k_idx]
        cand_subscores = [prev_result.val_subscores[i] for i in top_k_idx]
        _build_frontier_state(cand_texts, cand_subscores, run_dir, frontier_type=engine_cfg.frontier_type)

    config = GEPAConfig(
        engine=engine_cfg,
        reflection=ReflectionConfig(
            reflection_lm=cfg.reflection_lm,
        ),
    )

    result = optimize_anything(
        seed_candidate=seed_prompts[0],
        evaluator=policy_fn,
        dataset=anchor,
        objective=_OBJECTIVE,
        config=config,
    )

    new_phi = _extract_top_k(result, K)
    return new_phi, result
