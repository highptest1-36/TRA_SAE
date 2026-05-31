"""
TRA-SAE GRPO Reward Functions (unit-aware, v2)
===============================================
All reward components used by GRPOTrainer.

Reward breakdown
----------------
  format_reward       weight 0.30   — structural correctness of output format
  correctness_reward  weight 0.60   — answer correctness (symbolic_verifier v2)
  unit_reward         weight 0.10   — unit scale match bonus
  length_penalty      weight −0.10  — applied only when reasoning > 800 tokens

Total score: sum of all components, clipped to [−0.10, 1.0]

TRL GRPOTrainer compatibility
------------------------------
The trainer calls reward_funcs as:
    score = reward_fn(completions, **kwargs)

where completions is a list[str] of generated texts, and kwargs contains
fields that were forwarded from the dataset (ground_truth, subject, question).
Each function here accepts those kwargs to stay compatible.
"""
from __future__ import annotations

import re
import logging
from typing import Sequence

logger = logging.getLogger("tra-sae.reward")

# ── Config (mirrors src/config.py — avoiding circular import) ─────────────────
_REWARD_FORMAT       = 0.30
_REWARD_CORRECTNESS  = 0.60
_REWARD_UNIT         = 0.10
_REWARD_LENGTH_PEN   = -0.10
_REWARD_LENGTH_MAX   = 800   # reasoning tokens above this → penalty


# ── Tag patterns ──────────────────────────────────────────────────────────────
_RE_REASONING    = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",       re.DOTALL | re.IGNORECASE)
_RE_EXPLANATION  = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL | re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Format reward (0.0 – 0.30)
# ─────────────────────────────────────────────────────────────────────────────

def format_reward(completion: str) -> float:
    """Score the structural quality of the generated output.

    Full credit (0.30) requires all three XML tags to be present and
    non-empty.  Partial credit awarded per tag.

    Tags checked:
      <reasoning>…</reasoning>   → 1/3
      <answer>…</answer>         → 1/3
      <explanation>…</explanation> → 1/3
    """
    score = 0.0

    if _RE_REASONING.search(completion):
        r_match = _RE_REASONING.search(completion)
        if r_match and r_match.group(1).strip():
            score += 1 / 3

    if _RE_ANSWER.search(completion):
        a_match = _RE_ANSWER.search(completion)
        if a_match and a_match.group(1).strip():
            score += 1 / 3

    if _RE_EXPLANATION.search(completion):
        e_match = _RE_EXPLANATION.search(completion)
        if e_match and e_match.group(1).strip():
            score += 1 / 3

    return round(_REWARD_FORMAT * score, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correctness reward (0.0 or 0.60)
# ─────────────────────────────────────────────────────────────────────────────

def correctness_reward(
    completion: str,
    ground_truth: str,
    subject: str = "",
    question_text: str = "",
) -> float:
    """Binary correctness via symbolic_verifier v2.

    Returns _REWARD_CORRECTNESS (0.60) if verify_answer() is True, else 0.0.
    Z3 engine invoked when subject=='logic' and verify_answer supports it.
    """
    try:
        from src.symbolic_verifier import verify_answer
        correct = verify_answer(
            completion,
            ground_truth,
            subject=subject,
            question_text=question_text,
            use_z3=True,
        )
        return _REWARD_CORRECTNESS if correct else 0.0
    except Exception as exc:
        logger.warning(f"[reward] correctness_reward exception: {exc}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unit scale reward (0.0 or 0.10)
# ─────────────────────────────────────────────────────────────────────────────

def unit_reward(completion: str, ground_truth: str) -> float:
    """Bonus reward when the predicted and ground truth values share the same
    SI unit scale (e.g., both in kΩ, or both in µJ).

    Only meaningful for physics questions.  Returns 0 for logic/MCQ.
    """
    try:
        from src.symbolic_verifier import same_unit_scale
        return _REWARD_UNIT if same_unit_scale(completion, ground_truth) else 0.0
    except Exception as exc:
        logger.debug(f"[reward] unit_reward exception: {exc}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Length penalty (0.0 or −0.10)
# ─────────────────────────────────────────────────────────────────────────────

def length_penalty(completion: str) -> float:
    """Apply −0.10 if the <reasoning> section exceeds REWARD_LENGTH_MAX tokens.

    Token count is approximated as whitespace-split word count (fast; no
    tokenizer needed at reward-computation time).
    """
    m = _RE_REASONING.search(completion)
    if m:
        words = len(m.group(1).split())
        if words > _REWARD_LENGTH_MAX:
            return _REWARD_LENGTH_PEN
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    completion: str,
    ground_truth: str,
    subject: str = "",
    question_text: str = "",
) -> float:
    """Compute total GRPO reward for one (completion, ground_truth) pair.

    Components:
      fmt   = format_reward(completion)              — 0.00 to 0.30
      corr  = correctness_reward(…)                  — 0.00 or 0.60
      unit  = unit_reward(completion, ground_truth)  — 0.00 or 0.10
      lpen  = length_penalty(completion)             — 0.00 or −0.10

    Total = clip(fmt + corr + unit + lpen, −0.10, 1.0)

    Args:
        completion:    Full model output string.
        ground_truth:  Reference answer from the dataset.
        subject:       'physics' | 'logic' | '' — used by correctness_reward.
        question_text: Original question — used by Z3 engine for logic.

    Returns:
        float in [−0.10, 1.0]
    """
    fmt  = format_reward(completion)
    corr = correctness_reward(completion, ground_truth, subject, question_text)
    unit = unit_reward(completion, ground_truth)
    lpen = length_penalty(completion)

    total = fmt + corr + unit + lpen
    clipped = max(-0.10, min(1.0, total))

    logger.debug(
        f"[reward] fmt={fmt:.3f}  corr={corr:.3f}  unit={unit:.3f}  "
        f"lpen={lpen:.3f}  total={total:.3f}  clipped={clipped:.3f}"
    )
    return clipped


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRL GRPOTrainer-compatible batch reward functions
# ─────────────────────────────────────────────────────────────────────────────
# TRL calls: scores = reward_fn(completions, **kwargs_from_dataset_columns)
# Each function returns list[float] of length == len(completions).
#
# TRL ≥ 1.4 may pass completions as list[list[dict]] (chat-message format)
# instead of list[str].  _extract_text() normalises both cases.

def _extract_text(comp) -> str:
    """Extract plain string from a completion that may be str or list[dict].

    TRL ≥ 1.4 wraps generated text in message dicts:
        [{"role": "assistant", "content": "..."}, ...]
    Earlier versions pass plain strings.  This helper handles both.
    """
    if isinstance(comp, list):
        # Take the last message's content (the assistant turn)
        for msg in reversed(comp):
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        return ""
    return str(comp) if comp is not None else ""


def batch_format_reward(
    completions,
    **kwargs,
) -> list[float]:
    return [format_reward(_extract_text(c)) for c in completions]


def batch_correctness_reward(
    completions,
    ground_truth: Sequence[str] | None = None,
    subject: Sequence[str] | None = None,
    question: Sequence[str] | None = None,
    **kwargs,
) -> list[float]:
    """Batch correctness reward compatible with TRL's GRPOTrainer.

    Expects dataset to include columns 'ground_truth', 'subject', 'question'.
    """
    scores = []
    for i, comp in enumerate(completions):
        gt      = (ground_truth[i] if ground_truth else "")
        subj    = (subject[i]      if subject      else "")
        qtext   = (question[i]     if question     else "")
        scores.append(correctness_reward(_extract_text(comp), gt, subj, qtext))
    return scores


def batch_compute_reward(
    completions,
    ground_truth: Sequence[str] | None = None,
    subject: Sequence[str] | None = None,
    question: Sequence[str] | None = None,
    **kwargs,
) -> list[float]:
    """Composite reward — main reward function passed to GRPOTrainer."""
    scores = []
    for i, comp in enumerate(completions):
        gt      = (ground_truth[i] if ground_truth else "")
        subj    = (subject[i]      if subject      else "")
        qtext   = (question[i]     if question     else "")
        scores.append(compute_reward(_extract_text(comp), gt, subj, qtext))
    return scores
