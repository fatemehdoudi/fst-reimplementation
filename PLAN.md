# FST Implementation Plan

## What we are building

Fast-Slow Training (Algorithm 1, Appendix B) for the star-graph task.
- **Slow loop**: GRPO with CISPO surrogate (Eq. 4), HuggingFace for gradient updates, vLLM for rollout generation.
- **Fast loop**: `gepa.optimize` wrapping GPT-5.x as the reflection LM; we define a custom `evaluator` closure that calls vLLM with the current policy weights.
- **Target**: Qwen3-4B-Instruct, no thinking mode, `(d,p,n)=(25,20,500)`, 10 000 train / 200 test examples.

---

## Module layout

```
fst/
├── PLAN.md
├── config.py               # FSTConfig dataclass — all hypers in one place
├── data/
│   ├── star_graph.py       # procedural graph generation (Appendix C)
│   └── stream.py           # infinite shuffled data stream + lookahead-batch sampler
├── rl/
│   ├── rollout.py          # vLLM rollout generation; owns the LLM handle
│   ├── advantages.py       # per-problem group-relative advantage (Eq. 3)
│   └── cispo.py            # CISPO loss (Eq. 4) + AdamW optimizer step
├── fast/
│   └── gepa_wrapper.py     # GEPA cycle: evaluator closure → gepa.optimize → top-K prompts
├── scoring.py              # star-graph reward: extract last \boxed{}, exact-match
├── eval.py                 # mean@4 validation loop (no GEPA prompt)
├── trainer.py              # Algorithm 1 outer loop; owns the cycle counter
└── slurm/
    └── job.sh              # SLURM submission script (8×H100, single node)
```

---

## Algorithm 1 as implemented

```
Φ ← [seed_prompt]                          # gepa_wrapper initial population

for cycle c = 0, 1, 2, … :
    Lc ← stream.prefetch(T * batch_size)   # 6×32 = 192 problems
    gepa_anchor ← Lc[:num_eval_examples]   # first 200 (star-graph budget)

    # --- FAST LOOP ---
    Φ ← gepa_wrapper.run(
            policy_fn=rollout.score_fn,    # evaluator closure over current vLLM
            anchor=gepa_anchor,
            seed_prompts=Φ,                # warm-start from previous population
            K=K,                           # K=8
            run_dir=f"gepa_state/cycle_{c}",
            max_metric_calls=960,
        )

    # --- SLOW LOOP (T steps) ---
    for t in 0 … T-1:
        Bt ← Lc[t*batch_size : (t+1)*batch_size]   # 32 problems

        # Generate G=8 rollouts per problem (G/K=1 per prompt)
        rollouts ← rollout.generate(Bt, Φ, G=G, K=K)

        # Per-problem advantages (Problem baseline, Eq. 3)
        advantages ← advantages.compute(rollouts)

        # CISPO gradient step (Eq. 4)
        cispo.step(actor, ref_model, rollouts, advantages)

        # Sync updated weights into vLLM handle
        rollout.sync_weights(actor)
```

---

## Key algorithm details

### GRPO group construction (per problem p, per cycle)
- Population Φ = {ϕ(1), …, ϕ(K)}, K=8 for star-graph.
- For each p in the minibatch, generate G=8 rollouts: one rollout per prompt (G/K=1).
- All G=8 rollouts for p share a single (r̄g, σg) statistic — this is the **Problem baseline** (advantage_grouping="question" in GEPA's terminology).
- Advantages are then batch-normalized across all problems in Bt before the CISPO step.

### CISPO surrogate (Eq. 4)
```
ρt = π_θ(yt | x, ϕ, y<t) / π_θ_old(yt | x, ϕ, y<t)   # per-token IS ratio
L_cispo = -E[ sg(clip(ρt, clip_low, clip_high)) · A · ∇ log π_θ ]
```
Appendix D: `clip_low=1.0`, `clip_high=3.0`.  
KL-to-reference penalty: `coef=1e-3`, reference = frozen copy of the initial θ0 (base model).  
Optimizer: AdamW, lr=1e-6, β=(0.9, 0.999), wd=0, 10-step linear warm-up, no decay.  
Loss aggregated at **prompt level**: mean over tokens within each response, then mean over the group.

### GEPA wrapper (fast/gepa_wrapper.py)
- Uses `gepa.optimize` (the flat API, not `optimize_anything`).
- **Evaluator closure**: captures the current vLLM handle; for each (candidate_prompt, example) generates 1 rollout and returns the reward.
- **Seed across cycles**: pass `run_dir=f"gepa_state/cycle_{c}"`. GEPA saves `gepa_state.bin` there. On cycle c+1 we pass `run_dir=f"gepa_state/cycle_{c+1}"` as a fresh directory — GEPA starts from `seed_candidate=Φ[0]` (best of previous population) each cycle rather than warm-starting through disk state. The previous population's remaining K-1 prompts are submitted as additional evaluation candidates at cycle start. (**See open question Q3 below.**)
- **Extracting top-K**: after `gepa.optimize` returns, sort all discovered candidates by `result.val_aggregate_scores[i]`, take indices of top-K, return those candidates as strings. If fewer than K unique candidates exist, repeat the best.

### Star-graph reward (scoring.py)
```python
import re

def extract_boxed(text: str) -> str | None:
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def score(rollout_text: str, gold_path: list[int]) -> float:
    pred = extract_boxed(rollout_text)
    gold = ",".join(str(v) for v in gold_path)
    return 1.0 if pred == gold else 0.0
```
Gold path = v1, v2, …, v_{p-2}, goal (intermediate nodes + goal, **not** source).

### Star-graph dataset (data/star_graph.py)
Parameters: d=25, p=20, n=500, train=10 000, test=200, fixed seed for reproducibility.  
Graph generation per Appendix C:
1. Sample source s, goal g from {0…n-1} without replacement.
2. Sample p-2 intermediate nodes → gold path s→v1→…→v_{p-2}→g.
3. Attach d-1 decoy chains of length p from s, nodes drawn fresh from unused pool.
4. Shuffle all edges uniformly; serialize as space-separated "u,v" pairs.

Prompt template (verbatim from Appendix C):
```
Given a bi-directional graph in the form of space separated edges, output a path from source
node to the destination node in the form of comma separated integers.
For this question the graph is {graph}
The source node is {source}
The destination node is {destination}
Please reason step by step, and put your final answer within \boxed{}.
```
System prompt (seed for GEPA, also used by RL at cycle 0):
```
You are solving a graph path-finding task. You will be given a list of edges and a source and
destination node. Output one valid path from source to destination. Inspect the source node's
neighbors first, identify which neighbor leads to the destination via a sequence of valid edges,
then commit to that branch. Each consecutive pair in your output path must be a valid edge in the
graph. Put your final answer comma-separated inside boxed braces.
```

### Hyperparameters (Appendix D, Tables 1–2, star-graph)
| Parameter | Value |
|---|---|
| Base model | Qwen3/Qwen3-4B-Instruct |
| Context / prompt / response (tokens) | 8192 / 4096 / 4096 |
| Train batch size (problems/step) | 32 |
| G (rollouts/problem) | 8 |
| ppo_mini_batch_size | 32 |
| K (population size) | 8 |
| G/K (rollouts per prompt per problem) | 1 |
| Cycle length T | 6 |
| GEPA eval examples | 200 |
| GEPA max_metric_calls | 960 |
| Reflection LM | openai/gpt-5.2 (via `reflection_lm` arg) |
| clip_low / clip_high | 1.0 / 3.0 |
| KL coef | 1e-3 |
| LR | 1e-6 |
| Warm-up steps | 10 |
| vLLM tensor-parallel | 1 |
| Eval temperature | 0.6, top_p=0.95, n=4 (mean@4) |
| Checkpoint every | 50 steps |

---

## Open questions — STOP, need your answers before coding

### Q1 · CISPO clipping formula (blocks `rl/cispo.py`)

The paper writes `min(ρt, τ)` in Eq. 4, but Appendix D gives `clip_low=1.0, clip_high=3.0`. Two interpretations:

**A (asymmetric)**: `clip(ρt, 1.0, 3.0)` = `max(1.0, min(ρt, 3.0))`  
Ratio is always ≥ 1 regardless of direction; never down-weights a token, only up-weights (capped at 3×). This makes the update purely additive for positive-advantage rollouts.

**B (upper-only)**: `min(ρt, 3.0)` with clip_low used only as a validation check or for something else  
Standard truncated IS: clip only from above at τ=3.0; ratios below 1 are not clipped.

Which interpretation is correct? upper only.

---

### Q2 · vLLM weight sync strategy (blocks `rl/rollout.py` ↔ `rl/cispo.py` interface)

After each AdamW step, the HF model parameters change. vLLM needs to see the updated weights for the next rollout. Options:

**A (in-process GPU memcpy)**: Keep vLLM's `LLM` object alive; after each gradient step call a utility that iterates `actor.state_dict()` and pushes tensors into vLLM's internal worker model via `worker.model_runner.model.load_weights()`. No checkpoint I/O. Complex to set up but fastest.

**B (checkpoint per cycle)**: Save `actor.state_dict()` to disk at the end of each T-step cycle; re-create the `vllm.LLM` object pointing to the new checkpoint. Slow (disk + vLLM init ~30–60 s), but dead-simple. Works fine if GEPA dominates wall-clock anyway.

**C (HF generate for GEPA, vLLM for RL)**: Use `actor.generate()` (HF) inside the GEPA evaluator closure — weights are always current, no sync needed. Use `vllm.LLM` only for the T RL rollout steps per cycle (re-created each cycle from checkpoint). Simplest to reason about; slower than A.

**D (vllm v0 weight update API)**: Use `llm.collective_rpc("update_weights", args=(...))` (vLLM ≥ 0.6 feature). Clean but version-dependent.

Which approach do you prefer? C to start, plan for A later.

---

### Q3 · GEPA warm-start / population seeding across cycles (blocks `fast/gepa_wrapper.py`)

`gepa.optimize` takes a single `seed_candidate`. Each FST cycle should start GEPA from the previous population Φc. Options:

**A (fresh run each cycle, best-prompt seed)**: Each cycle starts `gepa.optimize` from scratch with `seed_candidate=Φ[0]` (the best-scoring prompt from the previous cycle). The other K-1 prompts in Φ are lost as initialization points; GEPA may rediscover similar ones organically.

**B (disk warm-start via run_dir)**: Keep a persistent GEPA `run_dir`. Each new cycle re-calls `gepa.optimize` pointing to the same dir; GEPA loads `gepa_state.bin` and picks up its full candidate frontier. The evaluator uses the new policy, so re-evaluations happen. The old per-instance scores become stale but GEPA's frontier structure (which prompts exist, lineage) is preserved.

**C (manual frontier injection)**: Between cycles, manually add the K-1 non-best prompts from Φc into the GEPA state as additional candidates before calling `gepa.optimize`. Requires digging into `GEPAState` internals.

Option A is simplest. Option B is closest to what the paper describes ("previous population Φc as the seed"). Which do you want? Answer: B, but verify

---

### Q4 · Distributed training setup (blocks `slurm/job.sh` and `trainer.py`)

8×H100 available. Qwen3-4B in BF16 fits on 1 GPU (~8 GB). Options:

**A (1 GPU)**: Single GPU for everything — simple, leave 7 GPUs unused. Fine for debugging.

**B (DDP, 8 GPU)**: Torch `DistributedDataParallel` for the actor; vLLM on GPU 0 only (TP=1 as paper says). Gradient steps scale to 8× effective batch. Needs careful device allocation so vLLM and the DDP process on GPU 0 don't OOM.

**C (FSDP, 8 GPU)**: Fully sharded; more memory-efficient but more complex.

Paper says single-node, GPU util ~0.6 for star-graph. Likely DDP with 8 replicas. Which do you want to start with? Answer: A (single GPU) first

---

### Q5 · Rollout reuse (Appendix F) — implement now or later?

Appendix F describes reusing GEPA's evaluation rollouts as RL training trajectories (~1/3 of rollouts served from cache, ~29% faster). It requires caching `(problem, prompt, response, reward)` tuples during GEPA and injecting them into the RL minibatch.

Start with rollout reuse, or leave it for a later pass? Answer: later

---

### Q6 · `gpt-5.2` vs `gpt-5.1` — model name for reflection LM

The paper uses `gpt-5.2` (Appendix D). The installed `gepa` package defaults to `openai/gpt-5.1`. Which model name should we pass to `reflection_lm`? (Exact string matters for the OpenAI API call.) gpt-5.2
