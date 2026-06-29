"""
Q3 warm-start test: call gepa.optimize_anything twice on the same run_dir and verify
(a) the Pareto frontier from run 1 persists into run 2, and
(b) candidates are re-evaluated (not served from stale cache) under a "new policy."

We simulate a policy change by toggling a global that flips which prompt keyword wins.
The reflection LM proposes mutations, so run 2 may propose new candidates building on
the frontier it loads from disk. We check:
  - gepa_state.bin exists after run 1
  - run 2 loads more candidates than run 1 generated (frontier persisted + extended)
  - the best candidate changes (run 2 re-scored under the new evaluator)
"""

import os
import tempfile

import gepa.optimize_anything as oa
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

POLICY_VERSION = 0   # mutated between runs to simulate a changing policy

DATASET = [{"id": i, "text": f"example {i}"} for i in range(10)]


def evaluator(candidate: str, example: dict) -> float:
    """
    "alpha" prompts win in policy 0; "beta" prompts win in policy 1.
    """
    if POLICY_VERSION == 0:
        score = 1.0 if "alpha" in candidate else 0.1
    else:
        score = 0.1 if "alpha" in candidate else 1.0
    oa.log(f"policy={POLICY_VERSION} score={score:.1f} snippet={candidate[:30]!r}")
    return score


SEED = "alpha strategy: inspect neighbors first"


def run_gepa(run_dir: str, seed: str, budget: int = 30) -> object:
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            max_metric_calls=budget,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openai/gpt-5.2",
            reflection_minibatch_size=3,
        ),
    )
    return optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        dataset=DATASET,
        objective="Find a system prompt that maximizes the evaluation score.",
        config=config,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as run_dir:
        state_path = os.path.join(run_dir, "gepa_state.bin")

        # ── RUN 1 ─────────────────────────────────────────────────────────────
        print("=== RUN 1 (policy 0: 'alpha' prompt wins) ===")
        POLICY_VERSION = 0
        r1 = run_gepa(run_dir, SEED)
        n1 = r1.num_candidates
        best1 = r1.best_candidate
        score1 = r1.val_aggregate_scores[r1.best_idx]
        print(f"  candidates: {n1}")
        print(f"  best score: {score1:.3f}")
        print(f"  best prompt: {best1[:80]!r}")
        print(f"  gepa_state.bin present: {os.path.exists(state_path)}")

        print()

        # ── RUN 2 (same run_dir → should load frontier from disk) ─────────────
        print("=== RUN 2 (policy 1: 'beta' prompt wins, same run_dir) ===")
        POLICY_VERSION = 1
        r2 = run_gepa(run_dir, SEED)
        n2 = r2.num_candidates
        best2 = r2.best_candidate
        score2 = r2.val_aggregate_scores[r2.best_idx]
        print(f"  candidates: {n2}")
        print(f"  best score: {score2:.3f}")
        print(f"  best prompt: {best2[:80]!r}")

        print()

        # ── VERDICT ───────────────────────────────────────────────────────────
        print("=== VERDICT ===")
        frontier_grew = n2 > n1
        best_changed = best1 != best2
        print(f"  candidates run1={n1}  run2={n2}  (grew: {frontier_grew})")
        print(f"  best candidate changed: {best_changed}")

        if not os.path.exists(state_path):
            print("FAIL: gepa_state.bin was not written — no warm-start possible")
        elif frontier_grew:
            print("PASS (frontier): run 2 loaded run 1's frontier and extended it")
        else:
            print("WARN (frontier): candidate count did not grow — state may not have loaded")

        if best_changed:
            print("PASS (re-eval): best candidate shifted under new policy — stale scores NOT cached")
        else:
            print("WARN (re-eval): best candidate is same as run 1 — may be serving stale scores")

        # Summary for the plan decision
        print()
        if frontier_grew and best_changed:
            print("→ Option B (disk warm-start) works correctly. Use it.")
        elif frontier_grew and not best_changed:
            print("→ Frontier loads but scores may be stale. Verify cache_evaluation=False (default).")
        else:
            print("→ Option B unreliable. Fall back to Option C (manual frontier injection).")
