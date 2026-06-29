from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FSTConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # ── Star-graph dataset (Appendix C) ────────────────────────────────────────
    d: int = 25        # source node degree
    p: int = 20        # path length (nodes); p-1 edges on gold path
    n: int = 500       # node pool size {0 … n-1}
    train_size: int = 10_000
    test_size: int = 200
    data_seed: int = 42

    # ── GRPO rollout group (Table 2) ───────────────────────────────────────────
    train_batch_size: int = 32   # problems per RL step
    G: int = 8                   # rollouts per problem
    K: int = 8                   # GEPA population size; must divide G
    T: int = 6                   # RL steps per FST cycle

    # ── Sequence lengths (Table 1, star-graph row) ─────────────────────────────
    max_ctx_len: int = 8192
    max_prompt_len: int = 4096
    max_response_len: int = 4096

    # ── CISPO surrogate (Eq. 4 + Appendix D) ──────────────────────────────────
    # clip(ρt, clip_low, clip_high): Q1 answer = B, both bounds exposed for A/B
    clip_low: float = 1.0
    clip_high: float = 3.0
    kl_coef: float = 1e-3        # KL-to-reference penalty coefficient

    # ── Optimizer (Appendix D) ─────────────────────────────────────────────────
    lr: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 10       # linear warm-up; no decay after

    # ── GEPA / fast loop (Table 2, star-graph row) ────────────────────────────
    gepa_eval_examples: int = 200
    gepa_max_metric_calls: int = 960
    # Make reflection_lm a config field so cheap models can be swapped for debug
    reflection_lm: str = "openai/gpt-5.2"

    # ── Evaluation (Appendix D shared config) ─────────────────────────────────
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_n: int = 4              # mean@4
    eval_every: int = 50         # steps between validation runs

    # ── Checkpointing ──────────────────────────────────────────────────────────
    ckpt_every: int = 50

    # ── vLLM (Appendix D, TP=1 for star-graph) ────────────────────────────────
    vllm_tp: int = 1
    # Actor + ref model occupy ~16 GiB before vLLM starts; 0.85 keeps vLLM's
    # pool under the remaining free memory on a single H200 (139.8 GiB).
    vllm_gpu_mem_util: float = 0.75
    # Rollout sampling knobs (temperature / top-p during training rollouts)
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0

    # ── Paths ──────────────────────────────────────────────────────────────────
    run_dir: str = "runs/star_graph"


# ── Debug preset (single GPU, tiny batch, cheap GEPA model) ───────────────────
DEBUG_CONFIG = FSTConfig(
    train_batch_size=2,
    G=4,
    K=4,
    T=2,
    gepa_eval_examples=4,
    gepa_max_metric_calls=8,
    reflection_lm="openai/gpt-4o-mini",  # cheap stand-in for debugging
    eval_every=5,
    ckpt_every=5,
    run_dir="runs/debug",
)

# ── Smoke-test preset — one end-to-end cycle on a GPU node, minimal compute ───
# Designed so the full cycle (1× GEPA + T=2 RL steps) finishes in a few minutes
# on a single H100 without burning real API budget or wall time.
#
# Key limits:
#   2 problems/step × T=2 steps  → 4 training problems total per cycle
#   G=4, K=2, G/K=2              → 4 vLLM rollouts per problem per step (8 total/step)
#   gepa_eval_examples=2          → GEPA evaluates candidates on 2 problems
#   gepa_max_metric_calls=20      → ≈10 GEPA proposals maximum
#   eval_every / ckpt_every = 99  → no mid-cycle eval or checkpoint (avoids 200-problem eval)
#   test_size=4                   → tiny test set if eval is triggered manually
SMOKE_CONFIG = FSTConfig(
    train_batch_size=2,
    G=4,
    K=2,
    T=2,
    train_size=500,           # small dataset; enough for shuffled stream
    test_size=4,
    gepa_eval_examples=2,
    gepa_max_metric_calls=20,
    reflection_lm="openai/gpt-4o-mini",
    eval_every=99,            # don't trigger eval inside the single cycle
    ckpt_every=99,
    run_dir="runs/smoke",
    vllm_gpu_mem_util=0.3,    # leave room for actor gradients on shared GPU
)
