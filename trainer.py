"""
FST Algorithm 1 outer loop.

Cycle structure
  FAST  — GEPA optimises the system-prompt population Φ using HF actor.generate()
           (Q2=C: weights are always current, no sync needed).
  SLOW  — T CISPO gradient steps using vLLM-generated rollouts.
           After every RL step, actor is saved to disk and vLLM is recreated from
           that checkpoint so the next step sees the updated policy (Q2=C).
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import FSTConfig
from data.star_graph import SEED_SYSTEM_PROMPT, StarGraphInstance, build_dataset
import fast.gepa_wrapper as gepa_wrapper
from rl.advantages import compute as compute_advantages
from rl.cispo import build_optimizer, cispo_loss
from rl.rollout import RolloutEngine, collate_batch


# ── Data helpers ──────────────────────────────────────────────────────────────


def _stream(data: list[StarGraphInstance], seed: int) -> Iterator[StarGraphInstance]:
    """Infinite shuffled iterator; re-shuffles after every full pass."""
    import random
    rng = random.Random(seed)
    while True:
        epoch = list(data)
        rng.shuffle(epoch)
        yield from epoch


def _prefetch(it: Iterator[StarGraphInstance], n: int) -> list[StarGraphInstance]:
    return [next(it) for _ in range(n)]


# ── Trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    """
    Owns all mutable state: actor, ref_model, rollout engine, optimizer,
    data stream, prompt population Φ, and cycle/step counters.
    """

    def __init__(self, cfg: FSTConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.run_dir, exist_ok=True)
        self._metrics_path = os.path.join(cfg.run_dir, "metrics.jsonl")
        self._ckpt_dir = os.path.join(cfg.run_dir, "ckpts")
        # Rolling checkpoint used for every per-step vLLM sync; never accumulates.
        self._sync_ckpt = os.path.join(self._ckpt_dir, "sync_latest")

        # ── Model + tokenizer ─────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.actor: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.actor.train()

        # Frozen reference model θ0 — loaded once, never updated.
        self.ref_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # ── Rollout engine ────────────────────────────────────────────────────
        self.engine = RolloutEngine(cfg, self.actor, self.tokenizer)

        # ── Optimizer + LR schedule ───────────────────────────────────────────
        self.optimizer, self.scheduler = build_optimizer(
            self.actor,
            lr=cfg.lr,
            beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
        )

        # ── Dataset ───────────────────────────────────────────────────────────
        train, test = build_dataset(
            d=cfg.d, p=cfg.p, n=cfg.n,
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            seed=cfg.data_seed,
        )
        self.test_data = test
        self._data_iter = _stream(train, seed=cfg.data_seed)

        # ── Training state ────────────────────────────────────────────────────
        # Φ starts with the seed prompt; gepa_wrapper pads to K after the first
        # GEPA call, so the RL slow loop always sees K elements.
        self.phi: list[str] = [SEED_SYSTEM_PROMPT]
        self._prev_gepa_result = None
        self.global_step: int = 0
        self.cycle: int = 0

        # ── Bootstrap vLLM from the initial (pre-training) weights ────────────
        initial_ckpt = os.path.join(self._ckpt_dir, "initial")
        self._save_ckpt(initial_ckpt)
        self.engine.init_vllm(initial_ckpt)

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self, num_cycles: int) -> None:
        """Run num_cycles FST cycles; each = one GEPA pass + T CISPO steps."""
        for c in range(num_cycles):
            self.cycle = c
            t_cycle = time.time()

            # Prefetch GEPA anchor and T RL minibatches from separate draws so
            # GEPA and RL are evaluated on independent examples.
            anchor = _prefetch(self._data_iter, self.cfg.gepa_eval_examples)
            rl_batches = [
                _prefetch(self._data_iter, self.cfg.train_batch_size)
                for _ in range(self.cfg.T)
            ]

            self._fast_loop(c, anchor)
            self._slow_loop(c, rl_batches)

            self._log({
                "event": "cycle_done",
                "cycle": c,
                "global_step": self.global_step,
                "elapsed_s": round(time.time() - t_cycle, 1),
            })

        self._save_state()

    # ── Fast loop (GEPA) ──────────────────────────────────────────────────────

    def _fast_loop(self, cycle: int, anchor: list[StarGraphInstance]) -> None:
        """Optimise Φ with GEPA, injecting the previous cycle's frontier (Option C)."""
        self.actor.eval()  # inference only; score_fn already uses no_grad
        gepa_dir = os.path.join(self.cfg.run_dir, "gepa_state", f"cycle_{cycle}")
        t0 = time.time()

        self.phi, self._prev_gepa_result = gepa_wrapper.run(
            policy_fn=self.engine.score_fn,
            anchor=anchor,
            seed_prompts=self.phi,
            K=self.cfg.K,
            run_dir=gepa_dir,
            cfg=self.cfg,
            prev_result=self._prev_gepa_result,
        )

        r = self._prev_gepa_result
        self._log({
            "event": "gepa_done",
            "cycle": cycle,
            "gepa_best_score": max(r.val_aggregate_scores) if r else 0.0,
            "gepa_num_candidates": r.num_candidates if r else 0,
            "elapsed_s": round(time.time() - t0, 1),
        })

    # ── Slow loop (T CISPO steps) ─────────────────────────────────────────────

    def _slow_loop(
        self,
        cycle: int,
        rl_batches: list[list[StarGraphInstance]],
    ) -> None:
        self.actor.train()

        for t, batch in enumerate(rl_batches):
            t0 = time.time()
            metrics = self._rl_step(batch)
            self.global_step += 1

            self._log({
                "event": "rl_step",
                "cycle": cycle,
                "t": t,
                "global_step": self.global_step,
                "elapsed_s": round(time.time() - t0, 1),
                **metrics,
            })

            if self.global_step % self.cfg.eval_every == 0:
                val = self._evaluate()
                self._log({"event": "eval",
                           "global_step": self.global_step,
                           "val_mean4": round(val, 4)})
                self.actor.train()  # _evaluate sets eval mode; restore here

            if self.global_step % self.cfg.ckpt_every == 0:
                path = os.path.join(self._ckpt_dir, f"step_{self.global_step}")
                self._save_ckpt(path)

            # Re-sync vLLM to the just-updated actor weights (Q2=C).
            # Uses a rolling directory so only one copy lives on disk at a time.
            self.engine.sync_weights(self.actor, self._sync_ckpt)

    # ── Single RL gradient step ───────────────────────────────────────────────

    def _rl_step(self, batch: list[StarGraphInstance]) -> dict[str, float]:
        """Generate rollouts → compute advantages → CISPO backward → step."""
        # --- Rollout generation (vLLM) ---------------------------------------
        rollouts = self.engine.generate(
            batch, self.phi, G=self.cfg.G, K=self.cfg.K,
        )

        # Free vLLM's GPU pool before the forward/backward pass — actor and
        # ref_model need the memory, and sync_weights will rebuild vLLM after.
        self.engine.teardown_vllm()

        # --- Advantage computation (two-step: per-problem then batch) --------
        adv = compute_advantages(rollouts)

        # --- CISPO gradient step ---------------------------------------------
        device = next(self.actor.parameters()).device
        rb = collate_batch(
            rollouts, adv,
            pad_id=self.tokenizer.pad_token_id,
            device=device,
        )

        self.optimizer.zero_grad()
        loss, metrics = cispo_loss(
            self.actor, self.ref_model, rb,
            clip_low=self.cfg.clip_low,
            clip_high=self.cfg.clip_high,
            kl_coef=self.cfg.kl_coef,
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        flat_rewards = [r.reward for grp in rollouts for r in grp]
        metrics["mean_reward"] = sum(flat_rewards) / len(flat_rewards)
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        """Mean@eval_n over the full test set using the best prompt in Φ."""
        self.actor.eval()
        best_prompt = self.phi[0]
        total = 0.0

        for problem in self.test_data:
            scores = [
                self.engine.score_fn(best_prompt, problem)
                for _ in range(self.cfg.eval_n)
            ]
            total += sum(scores) / len(scores)

        return total / len(self.test_data)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_ckpt(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.actor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _save_state(self) -> None:
        """Write Φ, cycle, and step counter so runs can be inspected or resumed."""
        state = {
            "cycle": self.cycle,
            "global_step": self.global_step,
            "phi": self.phi,
        }
        with open(os.path.join(self.cfg.run_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)

    def _log(self, metrics: dict) -> None:
        line = json.dumps(metrics)
        with open(self._metrics_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FST training (Algorithm 1)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Use DEBUG_CONFIG: tiny batch, cheap reflection LM, fast iteration",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use SMOKE_CONFIG: 1-cycle end-to-end test, minimal GPU/API cost",
    )
    parser.add_argument("--num_cycles", type=int, default=100)
    args = parser.parse_args()

    from config import DEBUG_CONFIG, SMOKE_CONFIG, FSTConfig
    if args.smoke:
        cfg = SMOKE_CONFIG
        args.num_cycles = 1
    elif args.debug:
        cfg = DEBUG_CONFIG
    else:
        cfg = FSTConfig()

    Trainer(cfg).train(num_cycles=args.num_cycles)


if __name__ == "__main__":
    main()
