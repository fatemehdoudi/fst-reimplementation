"""
Rollout generation for the FST training loop.

Q2 decision (from PLAN.md): Option C.
  - GEPA evaluator (score_fn): uses HF actor.generate() — weights are always current,
    no sync needed.
  - RL rollouts (generate): uses vLLM, re-created each cycle from a saved checkpoint
    via sync_weights(). No in-process GPU memcpy required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from config import FSTConfig
from data.star_graph import StarGraphInstance, format_user_prompt
from rl.cispo import RolloutBatch
from scoring import score_rollout

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass
class Rollout:
    """One model completion for a (problem, system-prompt) pair."""

    problem_idx: int       # index in the RL minibatch (0..batch_size-1)
    prompt_idx: int        # index into Phi (0..K-1)
    system_prompt: str
    response_text: str
    reward: float          # binary star-graph reward (0.0 or 1.0)

    # None only during CPU unit-tests where tensor fields are unnecessary.
    input_ids: Tensor | None = field(default=None, repr=False)
    response_ids: Tensor | None = field(default=None, repr=False)
    # Per-token log π_θ_old over response_ids; used by CISPO for ρt = exp(new - old).
    old_log_probs: Tensor | None = field(default=None, repr=False)


# ── Batch collation ───────────────────────────────────────────────────────────


def collate_batch(
    rollouts: list[list[Rollout]],
    advantages: list[list[float]],
    pad_id: int,
    device: torch.device,
) -> RolloutBatch:
    """
    Pack nested rollouts + computed advantages into a RolloutBatch for cispo_loss().

    rollouts[p][g] = Rollout for problem p, rollout g (G per problem).
    advantages[p][g] = float advantage scalar (already two-step normalised).

    Returns a RolloutBatch with shapes (B, *) where B = batch_size * G.
    """
    flat_rollouts = [r for group in rollouts for r in group]
    flat_adv = [a for grp in advantages for a in grp]

    B = len(flat_rollouts)
    max_prompt = max(r.input_ids.shape[0] for r in flat_rollouts)       # type: ignore[union-attr]
    max_resp = max(r.response_ids.shape[0] for r in flat_rollouts)      # type: ignore[union-attr]
    L = max_prompt + max_resp

    full_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, max_resp), -100, dtype=torch.long)
    old_lp = torch.zeros(B, max_resp, dtype=torch.float32)
    attn = torch.zeros(B, L, dtype=torch.long)
    plen = torch.zeros(B, dtype=torch.long)

    for i, r in enumerate(flat_rollouts):
        p_ids = r.input_ids                           # type: ignore[union-attr]
        r_ids = r.response_ids                        # type: ignore[union-attr]
        o_lp = r.old_log_probs                        # type: ignore[union-attr]
        P = p_ids.shape[0]
        R = r_ids.shape[0]

        full_ids[i, :P] = p_ids
        full_ids[i, P : P + R] = r_ids
        labels[i, :R] = r_ids
        old_lp[i, :R] = o_lp
        attn[i, : P + R] = 1
        plen[i] = P

    return RolloutBatch(
        input_ids=full_ids.to(device),
        labels=labels.to(device),
        old_log_probs=old_lp.to(device),
        advantages=torch.tensor(flat_adv, dtype=torch.float32, device=device),
        attention_mask=attn.to(device),
        prompt_len=plen.to(device),
    )


# ── Engine ────────────────────────────────────────────────────────────────────


class RolloutEngine:
    """
    Owns the HF actor (always live) and the vLLM handle (re-created each cycle).

    Usage pattern per FST cycle c:
        engine.init_vllm(ckpt_dir_c)       # start of RL phase
        for t in range(T):
            rollouts = engine.generate(Bt, phi, G, K)
            ...
        engine.sync_weights(actor, ckpt_dir_{c+1})   # save + recreate vLLM
    """

    def __init__(
        self,
        cfg: FSTConfig,
        actor: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.cfg = cfg
        self.actor = actor
        self.tokenizer = tokenizer
        self._llm = None  # vLLM LLM; created lazily via init_vllm()

    # ── vLLM lifecycle ────────────────────────────────────────────────────────

    def init_vllm(self, model_path: str) -> None:
        """Create (or recreate) the vLLM engine pointing to model_path."""
        if self._llm is not None:
            del self._llm
            torch.cuda.empty_cache()

        from vllm import LLM  # lazy: vLLM is GPU-only

        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=self.cfg.vllm_tp,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=self.cfg.vllm_gpu_mem_util,
            max_model_len=self.cfg.max_ctx_len,
        )

    def teardown_vllm(self) -> None:
        """Release the vLLM engine and free its GPU memory pool."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            torch.cuda.empty_cache()

    def sync_weights(self, actor: AutoModelForCausalLM, ckpt_dir: str) -> None:
        """Save HF actor checkpoint then rebuild vLLM from it (Q2=C)."""
        actor.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        self.init_vllm(ckpt_dir)

    # ── RL rollout generation (vLLM) ─────────────────────────────────────────

    def generate(
        self,
        batch: list[StarGraphInstance],
        phi: list[str],
        G: int,
        K: int,
    ) -> list[list[Rollout]]:
        """
        Generate G rollouts per problem using vLLM.

        With G/K = 1 (star-graph config), each (problem, prompt) pair yields
        one completion. All 256 pairs (32 problems × 8 prompts) are submitted
        to vLLM in a single batched call.

        Returns rollouts[p] = list of G Rollout objects for problem p, ordered
        by prompt index (0 .. K-1), repeated G//K times.
        """
        assert self._llm is not None, "call init_vllm() before generate()"
        assert G % K == 0
        n_per_prompt = G // K  # = 1 for star-graph

        from vllm import SamplingParams  # lazy

        sampling = SamplingParams(
            temperature=self.cfg.rollout_temperature,
            top_p=self.cfg.rollout_top_p,
            max_tokens=self.cfg.max_response_len,
            logprobs=1,          # log-prob of the sampled token at each position
            n=n_per_prompt,
        )

        # Build text prompts; let vLLM tokenize internally.
        # transformers 5.x apply_chat_template(tokenize=True) returns BatchEncoding
        # (not list[int]), which breaks vLLM's prompt_token_ids path.
        all_prompts: list[str] = []
        metadata: list[tuple[int, int]] = []  # (problem_idx, prompt_idx)

        for prob_idx, problem in enumerate(batch):
            for k, sys_prompt in enumerate(phi):
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_user_prompt(problem)},
                ]
                text: str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                all_prompts.append(text)
                metadata.append((prob_idx, k))

        outputs = self._llm.generate(all_prompts, sampling)

        # Collect into rollouts[problem_idx]
        rollouts: list[list[Rollout]] = [[] for _ in range(len(batch))]

        for (prob_idx, prompt_idx), out in zip(metadata, outputs):
            problem = batch[prob_idx]
            sys_prompt = phi[prompt_idx]
            # Use the token IDs vLLM actually consumed — authoritative source.
            prompt_tensor = torch.tensor(list(out.prompt_token_ids), dtype=torch.long)

            for completion in out.outputs:
                response_text = completion.text
                resp_ids = list(completion.token_ids)

                # Extract per-token log π_θ_old for the sampled tokens.
                lp_list: list[float] = []
                for tok_id, lp_dict in zip(
                    resp_ids, completion.logprobs or []
                ):
                    entry = (lp_dict or {}).get(tok_id)
                    lp_list.append(entry.logprob if entry is not None else 0.0)

                reward = score_rollout(response_text, problem["gold_path"])

                rollouts[prob_idx].append(
                    Rollout(
                        problem_idx=prob_idx,
                        prompt_idx=prompt_idx,
                        system_prompt=sys_prompt,
                        response_text=response_text,
                        reward=reward,
                        input_ids=prompt_tensor,
                        response_ids=torch.tensor(resp_ids, dtype=torch.long),
                        old_log_probs=torch.tensor(lp_list, dtype=torch.float32),
                    )
                )

        return rollouts

    # ── GEPA evaluator closure (HF actor) ────────────────────────────────────

    def score_fn(self, system_prompt: str, example: StarGraphInstance) -> float:
        """
        Evaluate one (system_prompt, problem) pair using HF actor.generate().

        Used as the GEPA evaluator: weights are always current, no sync needed.
        Signature matches gepa.optimize_anything evaluator: (str, dict) -> float.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_user_prompt(example)},
        ]
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers 5.x returns BatchEncoding (UserDict, not dict) instead of a plain tensor
        input_ids: Tensor = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(self.actor.device)

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out_ids = self.actor.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.max_response_len,
                temperature=self.cfg.eval_temperature,
                top_p=self.cfg.eval_top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = out_ids[0, input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return score_rollout(response_text, example["gold_path"])
