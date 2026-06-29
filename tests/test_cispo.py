"""Unit tests for rl/cispo.py (no GPU required — uses small fake tensors on CPU)."""

import sys
sys.path.insert(0, "/scratch/user/fatemehdoudi_tamu.edu/fst")

import math
import torch
import torch.nn as nn
from rl.cispo import cispo_loss, build_optimizer, RolloutBatch, _response_log_probs


# ── Tiny fake model that returns fixed logits ──────────────────────────────────

class ConstantLogitModel(nn.Module):
    """Outputs a fixed logit for each token position; ignores input."""
    def __init__(self, vocab_size: int, logit_value: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        # Learnable scalar so autograd works
        self.offset = nn.Parameter(torch.tensor(logit_value))

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        # Uniform logits across vocab: all log-probs = -log(V)
        logits = torch.zeros(B, L, self.vocab_size) + self.offset
        return type("Out", (), {"logits": logits})()


def make_batch(B: int = 4, L_prompt: int = 3, L_resp: int = 5, V: int = 8):
    """Build a RolloutBatch with small random tensors on CPU."""
    L_total = L_prompt + L_resp
    input_ids = torch.randint(0, V, (B, L_total))
    labels = torch.randint(0, V, (B, L_resp))
    # Mask last 2 positions as padding
    labels[:, -2:] = -100
    old_log_probs = torch.full((B, L_resp), fill_value=-math.log(V))
    old_log_probs[:, -2:] = 0.0  # zero at padding
    advantages = torch.randn(B)
    attention_mask = torch.ones(B, L_total, dtype=torch.long)
    prompt_len = torch.full((B,), L_prompt, dtype=torch.long)
    return RolloutBatch(
        input_ids=input_ids,
        labels=labels,
        old_log_probs=old_log_probs,
        advantages=advantages,
        attention_mask=attention_mask,
        prompt_len=prompt_len,
    )


V = 8
L_PROMPT = 3
L_RESP = 5


def test_response_log_probs_shape():
    model = ConstantLogitModel(V)
    batch = make_batch(B=4, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    lp = _response_log_probs(model, batch)
    assert lp.shape == (4, L_RESP), f"got {lp.shape}"


def test_response_log_probs_uniform():
    """With uniform logits, each token's log-prob should be -log(V)."""
    model = ConstantLogitModel(V, logit_value=0.0)
    batch = make_batch(B=2, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    lp = _response_log_probs(model, batch)
    valid = batch.labels != -100
    expected = -math.log(V)
    assert torch.allclose(lp[valid], torch.full_like(lp[valid], expected), atol=1e-5), \
        f"expected {expected}, got {lp[valid]}"


def test_response_log_probs_zero_at_padding():
    model = ConstantLogitModel(V)
    batch = make_batch(B=2, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    lp = _response_log_probs(model, batch)
    padding = batch.labels == -100
    assert (lp[padding] == 0.0).all(), "padding positions should be 0.0"


def test_cispo_loss_shape_and_backward():
    actor = ConstantLogitModel(V)
    ref = ConstantLogitModel(V)
    batch = make_batch(B=4, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    loss, metrics = cispo_loss(actor, ref, batch, clip_low=1.0, clip_high=3.0, kl_coef=1e-3)
    assert loss.shape == (), "loss should be a scalar"
    assert not torch.isnan(loss), "loss is NaN"
    loss.backward()  # should not raise
    assert actor.offset.grad is not None, "no gradient flowed to actor"


def test_cispo_zero_advantage_zero_policy_loss():
    """With all advantages = 0, policy_loss should be ~0."""
    actor = ConstantLogitModel(V)
    ref = ConstantLogitModel(V)
    batch = make_batch(B=4, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    batch.advantages.zero_()
    _, metrics = cispo_loss(actor, ref, batch, kl_coef=0.0)
    assert abs(metrics["policy_loss"]) < 1e-6, f"policy_loss={metrics['policy_loss']}"


def test_cispo_clip_low_floors_ratio():
    """
    When curr_log_probs > old_log_probs (ratio > 1), clip_low=1.0 is a no-op.
    When curr_log_probs < old_log_probs (ratio < 1), clip_low=1.0 floors ratio to 1.

    We test: if old_log_probs are very large (so ratio << 1), the clipped ratio
    should be exactly clip_low=1.0 for all valid tokens.
    """
    actor = ConstantLogitModel(V, logit_value=0.0)   # log_prob = -log(V)
    ref = ConstantLogitModel(V)
    batch = make_batch(B=2, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    # Force old_log_probs to be very high → ratio = exp(curr - old) << 1
    batch.old_log_probs = torch.full_like(batch.old_log_probs, -0.1)  # near 0
    batch.old_log_probs[batch.labels == -100] = 0.0  # preserve padding zeros
    _, metrics = cispo_loss(actor, ref, batch, clip_low=1.0, clip_high=3.0, kl_coef=0.0)
    # Mean ratio should be clipped to 1.0 since all raw ratios < 1
    assert abs(metrics["mean_ratio"] - 1.0) < 0.05, \
        f"expected mean_ratio ≈ 1.0, got {metrics['mean_ratio']}"
    assert metrics["frac_clipped_low"] > 0.9, \
        f"expected most tokens clipped at low, got frac={metrics['frac_clipped_low']}"


def test_cispo_clip_high_caps_ratio():
    """When old_log_probs are very small (ratio >> 1), ratio should be capped at clip_high."""
    actor = ConstantLogitModel(V, logit_value=0.0)  # log_prob = -log(V) ≈ -2.08
    ref = ConstantLogitModel(V)
    batch = make_batch(B=2, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    # Force old_log_probs very negative → ratio = exp(curr - old) >> 1
    batch.old_log_probs = torch.full_like(batch.old_log_probs, -20.0)
    batch.old_log_probs[batch.labels == -100] = 0.0
    _, metrics = cispo_loss(actor, ref, batch, clip_low=1.0, clip_high=3.0, kl_coef=0.0)
    assert abs(metrics["mean_ratio"] - 3.0) < 0.05, \
        f"expected mean_ratio ≈ 3.0, got {metrics['mean_ratio']}"
    assert metrics["frac_clipped_high"] > 0.9


def test_kl_zero_when_same_model():
    """KL should be ~0 when actor and ref are identical."""
    actor = ConstantLogitModel(V, logit_value=0.0)
    ref = ConstantLogitModel(V, logit_value=0.0)
    batch = make_batch(B=4, L_prompt=L_PROMPT, L_resp=L_RESP, V=V)
    batch.advantages.zero_()
    _, metrics = cispo_loss(actor, ref, batch, kl_coef=1.0)
    assert abs(metrics["kl"]) < 1e-5, f"KL should be ~0, got {metrics['kl']}"


def test_build_optimizer_warmup():
    model = ConstantLogitModel(V)
    opt, sched = build_optimizer(model, lr=1e-3, warmup_steps=4)
    # At step 0 (before first step call), lr should be lr * 1/4
    assert abs(sched.get_last_lr()[0] - 1e-3 * (1 / 4)) < 1e-9
    for _ in range(4):
        sched.step()
    # After warmup, lr should equal the full lr
    assert abs(sched.get_last_lr()[0] - 1e-3) < 1e-9


def test_metrics_keys():
    actor = ConstantLogitModel(V)
    ref = ConstantLogitModel(V)
    batch = make_batch()
    _, metrics = cispo_loss(actor, ref, batch)
    for key in ("loss", "policy_loss", "kl", "mean_ratio", "frac_clipped_high", "frac_clipped_low"):
        assert key in metrics, f"missing metric: {key}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
