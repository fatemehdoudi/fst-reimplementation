"""
CPU unit-tests for rl/advantages.py.

Key invariant under test (Algorithm 1, lines 7-11):
  All G rollouts for a problem share ONE (r̄, σ) statistic, regardless of which
  prompt index (0..K-1) produced them.

All expected values are derived analytically and noted inline.
No GPU required.
"""

from __future__ import annotations

import math

import pytest

from rl.rollout import Rollout
from rl.advantages import compute


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_rollout(problem_idx: int, prompt_idx: int, reward: float) -> Rollout:
    """Minimal Rollout for advantages tests — tensor fields left None."""
    return Rollout(
        problem_idx=problem_idx,
        prompt_idx=prompt_idx,
        system_prompt="",
        response_text="",
        reward=reward,
    )


def flat(advantages: list[list[float]]) -> list[float]:
    return [a for grp in advantages for a in grp]


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOutputShape:
    def test_single_group(self):
        rollouts = [[make_rollout(0, k, float(k % 2)) for k in range(4)]]
        adv = compute(rollouts)
        assert len(adv) == 1
        assert len(adv[0]) == 4

    def test_multi_group(self):
        rollouts = [
            [make_rollout(p, k, 0.5) for k in range(3)]
            for p in range(5)
        ]
        adv = compute(rollouts)
        assert len(adv) == 5
        assert all(len(grp) == 3 for grp in adv)

    def test_unequal_group_sizes(self):
        rollouts = [
            [make_rollout(0, k, float(k)) for k in range(2)],
            [make_rollout(1, k, float(k)) for k in range(6)],
        ]
        adv = compute(rollouts)
        assert len(adv[0]) == 2
        assert len(adv[1]) == 6


class TestPerProblemGroupingInvariant:
    """
    THE KEY TEST: all G rollouts for a problem share one (r̄, σ) — not per-prompt.

    If rollouts were split by prompt, a group of size G/K = 1 would always have
    σ = 0 and every advantage would be 0.  The correct behaviour is to group all
    G rollouts together and compute one shared statistic.
    """

    def test_single_problem_two_prompts(self):
        """
        1 problem, K=2 prompts, 1 rollout each → G=2 total.
        rewards:  prompt-0 → 1.0,  prompt-1 → 0.0
        Group stats:  r̄ = 0.5,  σ = 0.5
        raw_adv:  [ (1-0.5)/0.5, (0-0.5)/0.5 ] = [ 1.0, -1.0 ]
        Only one group → batch norm is a no-op (mean=0, std=1 of [1,-1]).
        """
        rollouts = [[
            make_rollout(0, 0, 1.0),
            make_rollout(0, 1, 0.0),
        ]]
        adv = compute(rollouts)

        # Advantages must be opposite in sign (same magnitude)
        assert_close(adv[0][0], -adv[0][1], msg="opposite rewards → opposite advantages")
        # Specifically: raw [1, -1], batch std = 1 → final = [1, -1]
        assert_close(abs(adv[0][0]), 1.0, tol=1e-5)

    def test_four_rollouts_across_four_prompts(self):
        """
        1 problem, K=4 prompts, 1 rollout each → G=4.
        rewards: [1, 0, 1, 0] (alternating by prompt)
        Group stats over ALL 4:  r̄ = 0.5,  σ = 0.5
        raw_adv:  [1, -1, 1, -1]

        If incorrectly split per-prompt (G/K = 1 each), every single-element
        group has σ=0 and every advantage would be 0.  The correct result is
        non-zero.
        """
        rollouts = [[
            make_rollout(0, 0, 1.0),
            make_rollout(0, 1, 0.0),
            make_rollout(0, 2, 1.0),
            make_rollout(0, 3, 0.0),
        ]]
        adv = compute(rollouts)

        # Non-zero: correct grouping produces signal
        assert all(abs(a) > 0.5 for a in adv[0]), \
            "per-prompt grouping would give 0; correct grouping gives ±1"

        # Rollouts with the same reward get the same advantage
        assert_close(adv[0][0], adv[0][2], msg="same reward → same advantage")
        assert_close(adv[0][1], adv[0][3], msg="same reward → same advantage")

        # Opposite rewards → opposite advantages
        assert_close(adv[0][0], -adv[0][1], tol=1e-5)

    def test_group_sum_is_near_zero(self):
        """After step-1, each group's raw advantages sum to 0 (mean-subtracted)."""
        rollouts = [[
            make_rollout(0, k, float(k) / 3)
            for k in range(4)
        ]]
        adv = compute(rollouts)
        # Batch norm on a single group rescales but keeps sum≈0 (mean=0 → sum=0)
        assert_close(sum(adv[0]), 0.0, tol=1e-5, msg="group advantages sum ≈ 0")


class TestUniformRewards:
    def test_all_ones_single_group(self):
        """When all rewards are 1.0, σ = 0 → all advantages are 0."""
        rollouts = [[make_rollout(0, k, 1.0) for k in range(8)]]
        adv = compute(rollouts)
        assert all(abs(a) < 1e-6 for a in adv[0])

    def test_all_zeros_single_group(self):
        rollouts = [[make_rollout(0, k, 0.0) for k in range(8)]]
        adv = compute(rollouts)
        assert all(abs(a) < 1e-6 for a in adv[0])

    def test_all_equal_multi_group(self):
        """If every problem has uniform rewards, all advantages should be 0."""
        rollouts = [
            [make_rollout(p, k, 1.0) for k in range(4)]
            for p in range(4)
        ]
        adv = compute(rollouts)
        assert all(abs(a) < 1e-6 for a in flat(adv))


class TestBatchNormalisation:
    """
    Verify step-2: after both normalisation steps, the full set of advantages
    has mean ≈ 0 and (when non-trivial) std ≈ 1.
    """

    def test_batch_mean_near_zero(self):
        rollouts = [
            [make_rollout(0, 0, 1.0), make_rollout(0, 1, 0.0)],
            [make_rollout(1, 0, 0.5), make_rollout(1, 1, 0.5)],
            [make_rollout(2, 0, 1.0), make_rollout(2, 1, 0.0)],
        ]
        adv = compute(rollouts)
        mean = sum(flat(adv)) / len(flat(adv))
        assert_close(mean, 0.0, tol=1e-5, msg="batch mean of final advantages")

    def test_two_problems_concrete_values(self):
        """
        2 problems, G=2 each.

        Problem 0: rewards [1, 0]  → r̄=0.5, σ=0.5 → raw=[1, -1]
        Problem 1: rewards [0.5, 0.5] → r̄=0.5, σ=0   → raw=[0, 0]

        Batch raw = [1, -1, 0, 0]
        batch_mean = 0.0
        batch_std  = sqrt((1 + 1 + 0 + 0) / 4) = sqrt(0.5) ≈ 0.70711

        Final:
          adv[0] = [1/0.70711, -1/0.70711] ≈ [1.41421, -1.41421]
          adv[1] = [0, 0]
        """
        rollouts = [
            [make_rollout(0, 0, 1.0), make_rollout(0, 1, 0.0)],
            [make_rollout(1, 0, 0.5), make_rollout(1, 1, 0.5)],
        ]
        adv = compute(rollouts)

        expected_scale = 1.0 / math.sqrt(0.5)   # ≈ 1.41421

        assert_close(adv[0][0],  expected_scale, tol=1e-4, msg="problem0, rollout0")
        assert_close(adv[0][1], -expected_scale, tol=1e-4, msg="problem0, rollout1")
        assert_close(adv[1][0], 0.0, tol=1e-6, msg="problem1, rollout0")
        assert_close(adv[1][1], 0.0, tol=1e-6, msg="problem1, rollout1")

    def test_step2_does_not_alter_zero_groups(self):
        """Zero-advantage groups stay at 0 after batch normalisation."""
        rollouts = [
            [make_rollout(0, 0, 1.0), make_rollout(0, 1, 0.0)],   # non-zero raw
            [make_rollout(1, 0, 1.0), make_rollout(1, 1, 1.0)],   # zero raw
        ]
        adv = compute(rollouts)
        assert_close(adv[1][0], 0.0, tol=1e-6)
        assert_close(adv[1][1], 0.0, tol=1e-6)


class TestStarGraphScale:
    """Smoke-tests with the actual FST batch dimensions (32 problems × G=8)."""

    def test_full_batch_shape(self):
        batch_size = 32
        G = 8
        K = 8
        rollouts = [
            [make_rollout(p, k, float((p + k) % 2)) for k in range(G)]
            for p in range(batch_size)
        ]
        adv = compute(rollouts)
        assert len(adv) == batch_size
        assert all(len(grp) == G for grp in adv)

    def test_full_batch_mean_near_zero(self):
        batch_size = 32
        G = 8
        import random
        rng = random.Random(42)
        rollouts = [
            [make_rollout(p, k, float(rng.randint(0, 1))) for k in range(G)]
            for p in range(batch_size)
        ]
        adv = compute(rollouts)
        all_adv = flat(adv)
        mean = sum(all_adv) / len(all_adv)
        assert_close(mean, 0.0, tol=1e-4, msg="batch mean at FST scale")
