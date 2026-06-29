"""
Star-graph reward function (Appendix C).

Extracts the last \\boxed{...} from the rollout text and applies exact-match
against the gold path rendered as a comma-separated string.

No partial credit: reward is 1.0 on exact match, 0.0 otherwise.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} in text, stripped of whitespace."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def gold_path_str(gold_path: list[int]) -> str:
    """Render gold path [v1, …, v_{p-2}, goal] as the comma-separated string
    the model must output.  Source node is NOT included."""
    return ",".join(str(v) for v in gold_path)


def score_rollout(rollout_text: str, gold_path: list[int]) -> float:
    """Return 1.0 if the last \\boxed{} matches the gold path exactly, else 0.0.

    For Qwen3-4B-Instruct (no thinking mode) there is no </think> block to
    strip — we search the full response.
    """
    pred = extract_boxed(rollout_text)
    if pred is None:
        return 0.0
    # Normalise whitespace inside the prediction (e.g. "1, 2, 3" → "1,2,3")
    pred_normalised = re.sub(r"\s*,\s*", ",", pred)
    return 1.0 if pred_normalised == gold_path_str(gold_path) else 0.0
