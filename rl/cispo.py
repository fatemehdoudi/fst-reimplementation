"""
CISPO surrogate loss + optimizer step (Eq. 4, Appendix D).

Loss formula per token t:
    ρt     = π_θ(yt | x, ϕ, y<t) / π_θ_old(yt | x, ϕ, y<t)
    weight = sg( clip(ρt, clip_low, clip_high) )
    L_t    = -weight * A * log π_θ(yt | x, ϕ, y<t)

Aggregated at prompt level: mean over response tokens per rollout,
then mean over the group.

KL-to-reference penalty (per token, one-directional):
    kl_t = log π_θ(yt | …) - log π_ref(yt | …)

Total loss = mean(L_cispo) + kl_coef * mean(kl)

Q1 answer (user): use clip(ρt, clip_low, clip_high) — both bounds are applied.
clip_low=1.0 (default) means ratios below 1.0 are floored, effectively
preventing negative IS weights. clip_high=3.0 caps the upward boost.
Both are exposed as config fields for A/B testing.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RolloutBatch:
    """
    One RL minibatch ready for a CISPO gradient step.

    Shapes: (B, L) where B = total rollouts in the batch (= problems × G),
    L = max response length after padding.
    """
    input_ids: Tensor          # (B, L_prompt + L_resp) — full sequence
    labels: Tensor             # (B, L_resp) — response tokens; -100 for padding
    old_log_probs: Tensor      # (B, L_resp) — π_θ_old per token; 0.0 for padding
    advantages: Tensor         # (B,) — per-rollout advantage (already normalised)
    attention_mask: Tensor     # (B, L_prompt + L_resp)
    prompt_len: Tensor         # (B,) — number of prompt tokens per row


def _response_log_probs(
    model: nn.Module,
    batch: RolloutBatch,
) -> Tensor:
    """
    Compute per-token log-probs for the response tokens under `model`.

    Returns shape (B, L_resp); padding positions hold 0.0.
    """
    device = batch.input_ids.device.type
    amp_ctx = (
        torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with amp_ctx:
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).logits  # (B, L_prompt + L_resp, V)

    B, L_total, V = logits.shape
    L_resp = batch.labels.shape[1]

    # Logits for response tokens are at positions [L_prompt : L_prompt + L_resp - 1],
    # predicting labels[:, 1:] ... actually we need logit at position t to predict
    # label at t, so we take logits[:, L_prompt-1 : L_prompt-1+L_resp, :].
    # Simpler: the full-sequence logit at position i predicts token i+1.
    # Response token k (0-indexed) lives at input_ids[:, prompt_len + k].
    # The logit that predicts it is at position prompt_len + k - 1.
    # We do this in a vectorised way using the gather trick.

    # Shift: logits[:, :-1, :] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()  # (B, L_total-1, V)
    shift_ids = batch.input_ids[:, 1:].contiguous()  # (B, L_total-1)

    log_probs_all = torch.log_softmax(shift_logits, dim=-1)  # (B, L_total-1, V)
    # Gather the log-prob of the actual token
    token_log_probs = log_probs_all.gather(
        2, shift_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, L_total-1)

    # Extract the slice corresponding to response tokens.
    # Row i: response starts at prompt_len[i] in input_ids, so in token_log_probs
    # (which is shifted by 1) the response starts at prompt_len[i] - 1.
    # We build a mask and extract L_resp columns.
    resp_log_probs = torch.zeros(B, L_resp, device=logits.device, dtype=logits.dtype)
    for i in range(B):
        start = batch.prompt_len[i].item() - 1
        end = start + L_resp
        end = min(end, token_log_probs.shape[1])
        length = end - start
        resp_log_probs[i, :length] = token_log_probs[i, start:end]

    # Zero out padding positions (where labels == -100)
    resp_log_probs = resp_log_probs * (batch.labels != -100).float()
    return resp_log_probs


def cispo_loss(
    actor: nn.Module,
    ref_model: nn.Module,
    batch: RolloutBatch,
    clip_low: float = 1.0,
    clip_high: float = 3.0,
    kl_coef: float = 1e-3,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute CISPO loss + KL-to-reference penalty.

    Returns (scalar loss, metrics dict).
    """
    valid_mask = (batch.labels != -100).float()  # (B, L_resp)
    response_lengths = valid_mask.sum(dim=1).clamp(min=1)  # (B,)

    # Current policy log-probs (requires grad)
    curr_log_probs = _response_log_probs(actor, batch)  # (B, L_resp)

    # IS ratio per token (detached for the weight; current log_prob stays in graph)
    with torch.no_grad():
        log_ratio = curr_log_probs - batch.old_log_probs  # (B, L_resp)
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(min=clip_low, max=clip_high)

    # Per-token CISPO loss:  -clipped_ratio * A * log π_θ
    # advantage is per-rollout (B,); broadcast to (B, L_resp)
    adv = batch.advantages.unsqueeze(1)  # (B, 1)
    token_loss = -(clipped_ratio * adv * curr_log_probs) * valid_mask  # (B, L_resp)

    # Prompt-level aggregation: mean over response tokens
    rollout_loss = token_loss.sum(dim=1) / response_lengths  # (B,)
    policy_loss = rollout_loss.mean()

    # KL penalty: log π_θ - log π_ref (one-directional, per sampled token)
    with torch.no_grad():
        ref_log_probs = _response_log_probs(ref_model, batch)  # (B, L_resp)
    kl_per_token = (curr_log_probs - ref_log_probs) * valid_mask  # (B, L_resp)
    kl_per_rollout = kl_per_token.sum(dim=1) / response_lengths  # (B,)
    kl_loss = kl_per_rollout.mean()

    loss = policy_loss + kl_coef * kl_loss

    valid = valid_mask.bool()
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl": kl_loss.item(),
        "mean_ratio": clipped_ratio[valid].mean().item(),   # post-clip
        "mean_raw_ratio": ratio[valid].mean().item(),       # pre-clip (diagnostic)
        "frac_clipped_high": (ratio > clip_high)[valid].float().mean().item(),
        "frac_clipped_low": (ratio < clip_low)[valid].float().mean().item(),
    }
    return loss, metrics


def build_optimizer(
    actor: nn.Module,
    lr: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    warmup_steps: int = 10,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warm-up (no decay after warm-up), per Appendix D."""
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0  # constant after warm-up

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
