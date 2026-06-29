"""
Per-rollout advantage computation for the FST training loop.

Two-step normalisation (Algorithm 1, lines 7–11):

  Step 1 — Problem baseline (per-problem group):
      All G rollouts for problem p share ONE (r̄_p, σ_p) statistic, regardless
      of which of the K prompts produced them.
          A_raw_i = (r_i - r̄_p) / (σ_p + ε)

  Step 2 — Batch normalisation (across all B×G rollouts in the minibatch):
          A_i = (A_raw_i - μ_batch) / (σ_batch + ε)

When all rewards in a group are equal, σ_p = 0 and all raw advantages for that
problem are 0.  If every group has uniform rewards, σ_batch = 0 and all final
advantages are 0.
"""

from __future__ import annotations

import torch

from rl.rollout import Rollout

_EPS = 1e-8


def compute(rollouts: list[list[Rollout]]) -> list[list[float]]:
    """
    Compute normalised advantages.

    Args:
        rollouts: rollouts[p] is a list of G Rollout objects for problem p.
                  All G entries share one (r̄_p, σ_p) — do NOT split by prompt.

    Returns:
        advantages: same nested shape; advantages[p][g] is a float scalar.
    """
    # ── Step 1: per-problem group normalisation ───────────────────────────────
    raw_advantages: list[list[float]] = []
    for group in rollouts:
        rewards = torch.tensor([r.reward for r in group], dtype=torch.float64)
        r_mean = rewards.mean()
        r_std = rewards.std(unbiased=False)
        raw_adv = ((rewards - r_mean) / (r_std + _EPS)).tolist()
        raw_advantages.append(raw_adv)

    # ── Step 2: batch normalisation across all B×G raw advantages ────────────
    flat = torch.tensor(
        [a for group in raw_advantages for a in group], dtype=torch.float64
    )
    batch_mean = flat.mean()
    batch_std = flat.std(unbiased=False)
    flat_normed = ((flat - batch_mean) / (batch_std + _EPS)).tolist()

    # Reshape back to the original nested structure.
    advantages: list[list[float]] = []
    idx = 0
    for group in raw_advantages:
        G = len(group)
        advantages.append(flat_normed[idx : idx + G])
        idx += G

    return advantages
