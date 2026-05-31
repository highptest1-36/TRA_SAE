"""
TRA-SAE Agent Nodes v2
========================
Node functions for the LangGraph StateGraph v2 (dual-LoRA + self-consistency).

Changes vs v1:
  - MAX_NEW_TOKENS = 1024  (was 512)
  - AGENT_MAX_RETRIES = 2  (was 3)
  - model.set_adapter(subject) called before every generate()
  - retrieve_context_node: passes subject= filter to retriever
  - generate_answer_node:
      attempt 0 → single generation, temp=0.1 (greedy-ish)
      attempt 1 → self-consistency: 5 samples, temp=0.7, majority vote
  - verify_answer_node: calls verify_answer(..., subject=, question_text=)
  - add stop_strings=["</explanation>"] to generation kwargs
  - router_node: new first node, classifies subject via SubjectRouter

Five core nodes + one router node:
  0. router_node               — SubjectRouter → state["subject"]
  1. retrieve_context_node     — TF-IDF few-shot, subject-filtered
  2. generate_answer_node      — model inference (single or self-consistency)
  3. verify_answer_node        — symbolic correctness check (Z3 hook for logic)
  4. generate_explanation_node — explanation selection
  5. format_output_node        — assemble final XML
"""
from __future__ import annotations

import re
import time
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger("tra-sae.agent")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

MAX_NEW_TOKENS       = 1024
AGENT_MAX_RETRIES    = 2
SELF_CONSISTENCY_N   = 5
SELF_CONSISTENCY_TEMP = 0.7

# Temperature schedule: attempt 0 = greedy, attempt 1 = self-consistency
_TEMP_SCHEDULE = [0.1, SELF_CONSISTENCY_TEMP]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["\n--- Reference examples (similar problems) ---"]
    for i, ex in enumerate(examples, 1):
        q = ex["question"][:400] + "..." if len(ex["question"]) > 400 else ex["question"]
        a = ex["answer"]
        e = ex.get("explanation", "")[:200]
        lines.append(
            f"\nExample {i}:\n{q}\n"
            f"<answer>\n{a}\n</answer>\n"
            f"<explanation>\n{e}\n</explanation>"
        )
    lines.append("\n--- Now answer the following ---\n")
    return "\n".join(lines)


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_explanation(text: str) -> str:
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _format_score(text: str) -> float:
    score = 0.0
    if re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL | re.IGNORECASE):
        score += 0.33
    if re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE):
        score += 0.34
    if re.search(r"<explanation>.*?</explanation>", text, re.DOTALL | re.IGNORECASE):
        score += 0.33
    return round(score, 3)


def _generate_single(
    model: Any,
    tokenizer: Any,
    question: str,
    few_shot_context: str,
    temperature: float,
    device: str = "cuda",
    subject: str = "",
) -> str:
    """Single generation pass with optional adapter switching."""
    import torch

    # Switch LoRA adapter for the appropriate subject
    if subject and hasattr(model, "set_adapter"):
        try:
            model.set_adapter(subject)
        except Exception:
            pass  # adapter not loaded for this subject → use current

    user_content = few_shot_context + question if few_shot_context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    # Use tokenize=False to get a string, then tokenize separately.
    # apply_chat_template with return_tensors returns BatchEncoding in transformers 5.x,
    # which causes issues; the two-step approach is safer and explicit.
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_NEW_TOKENS * 2).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    do_sample = temperature > 0.05
    gen_kwargs: dict[str, Any] = dict(
        input_ids          = input_ids,
        attention_mask     = attention_mask,
        max_new_tokens     = MAX_NEW_TOKENS,
        do_sample          = do_sample,
        pad_token_id       = tokenizer.eos_token_id,
        eos_token_id       = tokenizer.eos_token_id,
        stop_strings       = ["</explanation>"],
        tokenizer          = tokenizer,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"]       = 0.9

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _self_consistency_vote(answers: list[str]) -> tuple[str, float]:
    """Majority vote over a list of raw answer strings.

    Returns (best_raw_output, vote_fraction).
    """
    extracted = [_extract_answer(a) for a in answers]
    if not extracted:
        return answers[0] if answers else "", 0.0

    counts = Counter(extracted)
    majority_ans, majority_count = counts.most_common(1)[0]
    vote_fraction = majority_count / len(extracted)

    # Return the raw output that produced the majority answer
    for raw in answers:
        if _extract_answer(raw) == majority_ans:
            return raw, vote_fraction

    return answers[0], vote_fraction


# ──────────────────────────────────────────────────────────────────────────────
# Node 0 — router  (new in v2)
# ──────────────────────────────────────────────────────────────────────────────

def router_node(state: dict) -> dict:
    """Classify question as 'physics' or 'logic' using SubjectRouter.

    If the state already has a subject (set by the dataset), use it directly.
    Otherwise call the router.
    """
    # If subject already known from dataset, skip routing
    subject = state.get("subject", "")
    if subject in ("physics", "logic"):
        logger.debug(f"[router] subject already set: {subject}")
        return {"subject": subject, "adapter_name": subject}

    router = state.get("_router")
    if router is None:
        logger.warning("[router] No router available — defaulting to 'physics'")
        return {"subject": "physics", "adapter_name": "physics"}

    question = state.get("question", "")
    try:
        subject, confidence = router.predict(question)
        logger.info(f"[router] subject='{subject}' confidence={confidence:.3f}")
    except Exception as exc:
        logger.warning(f"[router] prediction failed ({exc}) — defaulting to 'physics'")
        subject, confidence = "physics", 0.5

    return {
        "subject":      subject,
        "adapter_name": subject,
        "confidence":   round(confidence, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 — retrieve_context
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_context_node(state: dict) -> dict:
    """Retrieve top-3 subject-filtered training examples for few-shot context."""
    retriever = state["_retriever"]
    question  = state["question"]
    subject   = state.get("subject", "")   # v2: pass subject filter

    try:
        examples = retriever.retrieve(question, top_k=3, subject=subject or None)
        logger.debug(
            "[retrieve] subject=%s  top scores: %s",
            subject, [f"{e['score']:.3f}" for e in examples],
        )
    except Exception as exc:
        logger.warning("[retrieve] failed: %s — zero-shot", exc)
        examples = []

    return {"retrieved_examples": examples}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 — generate_answer
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_node(state: dict) -> dict:
    """Generate an answer with adapter switching + self-consistency on retry.

    Attempt 0: single generation, temperature=0.1
    Attempt 1: self-consistency voting (SELF_CONSISTENCY_N=5, temp=0.7)
    """
    model     = state["_model"]
    tokenizer = state["_tokenizer"]
    device    = state.get("_device", "cuda")
    question  = state["question"]
    subject   = state.get("subject", "")
    retry     = state.get("retry_count", 0)
    examples  = state.get("retrieved_examples", [])

    # Few-shot only on attempt 0
    few_shot = _format_few_shot_block(examples) if retry == 0 else ""

    t0 = time.time()

    if retry == 0:
        # Attempt 0: greedy single generation
        temperature = _TEMP_SCHEDULE[0]
        raw_output = _generate_single(
            model, tokenizer, question, few_shot,
            temperature, device, subject,
        )
        confidence  = None   # will be set in generate_explanation_node
        vote_frac   = None

        logger.info(
            "[generate] attempt=0  temp=%.2f  %.1fs  fmt=%.2f",
            temperature, time.time() - t0, _format_score(raw_output),
        )

    else:
        # Attempt 1: self-consistency voting
        temperature = SELF_CONSISTENCY_TEMP
        candidates  = []
        for _i in range(SELF_CONSISTENCY_N):
            c = _generate_single(
                model, tokenizer, question, "",
                temperature, device, subject,
            )
            candidates.append(c)

        raw_output, vote_frac = _self_consistency_vote(candidates)
        logger.info(
            "[generate] attempt=1  self_consistency N=%d  vote_frac=%.2f  %.1fs",
            SELF_CONSISTENCY_N, vote_frac, time.time() - t0,
        )

    all_attempts = list(state.get("all_attempts", []))
    all_attempts.append(raw_output)

    update: dict = {
        "raw_output":   raw_output,
        "all_attempts": all_attempts,
        "retry_count":  retry + 1,
    }
    if vote_frac is not None:
        update["confidence"] = round(vote_frac, 4)

    return update


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 — verify_answer
# ──────────────────────────────────────────────────────────────────────────────

def verify_answer_node(state: dict) -> dict:
    """Check generated answer against ground truth with Z3 hook for logic."""
    from src.symbolic_verifier import verify_answer, extract_answer_from_text

    raw_output    = state.get("raw_output", "")
    ground_truth  = state.get("ground_truth", "")
    subject       = state.get("subject", "")
    question_text = state.get("question", "")

    predicted = extract_answer_from_text(raw_output)

    if ground_truth:
        # Evaluation / training mode: full symbolic check with Z3 hook
        correct = verify_answer(
            predicted,
            ground_truth,
            subject=subject,
            question_text=question_text,
            use_z3=True,
        )
        logger.debug(
            "[verify] pred='%s'  gt='%s'  subject=%s  correct=%s",
            predicted[:60], ground_truth[:40], subject, correct,
        )
    else:
        # Inference-only mode: accept if XML format is valid
        correct = bool(
            re.search(r"<answer>.*?</answer>", raw_output, re.DOTALL | re.IGNORECASE)
        )
        logger.debug("[verify] inference-only  format_ok=%s", correct)

    return {
        "generated_answer": predicted,
        "verified":         correct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 4 — generate_explanation
# ──────────────────────────────────────────────────────────────────────────────

def generate_explanation_node(state: dict) -> dict:
    """Extract or select the best explanation from all attempts."""
    verified     = state.get("verified", False)
    raw_output   = state.get("raw_output", "")
    all_attempts = state.get("all_attempts", [])
    vote_frac    = state.get("confidence")   # set by self-consistency, may be None

    if verified:
        explanation = _extract_explanation(raw_output)
        selected    = raw_output
        confidence  = vote_frac if vote_frac is not None else (0.85 + 0.15 * _format_score(raw_output))
    else:
        # Best-of-N fallback by format score
        selected = max(all_attempts, key=_format_score) if all_attempts else raw_output
        explanation = _extract_explanation(selected)
        from src.symbolic_verifier import extract_answer_from_text
        best_answer = extract_answer_from_text(selected)
        logger.info(
            "[explain] fallback best-of-N  fmt=%.2f  ans='%s'",
            _format_score(selected), best_answer[:40],
        )
        confidence = vote_frac if vote_frac is not None else (0.30 * _format_score(selected))

    if not explanation:
        explanation = "No explanation could be extracted from the model output."

    return {
        "explanation": explanation,
        "confidence":  round(confidence, 4),
        "raw_output":  selected,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 5 — format_output
# ──────────────────────────────────────────────────────────────────────────────

def format_output_node(state: dict) -> dict:
    """Assemble final XML submission."""
    answer      = state.get("generated_answer", "").strip()
    explanation = state.get("explanation", "").strip()

    if not answer:
        answer = "unknown"
    if not explanation:
        explanation = "No explanation available."

    final_output = (
        f"<answer>\n{answer}\n</answer>\n"
        f"<explanation>\n{explanation}\n</explanation>"
    )

    logger.info(
        "[format] verified=%s  confidence=%.3f  subject=%s  answer='%s'",
        state.get("verified", False),
        state.get("confidence", 0.0),
        state.get("subject", ""),
        answer[:60],
    )

    return {"final_output": final_output}

from __future__ import annotations

import re
import time
import logging
from typing import Any

logger = logging.getLogger("tra-sae.agent")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

# Temperature schedule for retries: start low, allow more exploration on retry
_TEMP_SCHEDULE = [0.1, 0.4, 0.7]
MAX_NEW_TOKENS = 512


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_few_shot_block(examples: list[dict]) -> str:
    """Format retrieved examples as few-shot context appended to the query."""
    if not examples:
        return ""
    lines = ["\n--- Reference examples (similar problems) ---"]
    for i, ex in enumerate(examples, 1):
        # Truncate long questions to keep prompt short
        q = ex["question"][:400] + "..." if len(ex["question"]) > 400 else ex["question"]
        a = ex["answer"]
        e = ex.get("explanation", "")[:200] + "..." if len(ex.get("explanation", "")) > 200 else ex.get("explanation", "")
        lines.append(
            f"\nExample {i}:\n{q}\n"
            f"<answer>\n{a}\n</answer>\n"
            f"<explanation>\n{e}\n</explanation>"
        )
    lines.append("\n--- Now answer the following ---\n")
    return "\n".join(lines)


def _extract_answer(text: str) -> str:
    """Extract content between <answer>…</answer> tags."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_explanation(text: str) -> str:
    """Extract content between <explanation>…</explanation> tags."""
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _format_score(text: str) -> float:
    """Score the format quality of a generation (0–1).

    Rewards presence of required XML tags.
    """
    score = 0.0
    if re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL | re.IGNORECASE):
        score += 0.33
    if re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE):
        score += 0.34
    if re.search(r"<explanation>.*?</explanation>", text, re.DOTALL | re.IGNORECASE):
        score += 0.33
    return round(score, 3)


def _generate(
    model: Any,
    tokenizer: Any,
    question: str,
    few_shot_context: str,
    temperature: float,
    device: str = "cuda",
) -> str:
    """Run a single forward pass and return the decoded generation."""
    import torch

    user_content = few_shot_context + question if few_shot_context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # enable_thinking=False is critical for Qwen3: suppresses <think> mode so the
    # model uses our custom <reasoning>/<answer>/<explanation> XML format instead.
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    input_ids = tokenized.to(device)
    attention_mask = torch.ones_like(input_ids)

    do_sample = temperature > 0.05
    gen_kwargs: dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=None,       # suppress "both max_new_tokens and max_length set" warning
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    # Decode only the new tokens
    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 — retrieve_context
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_context_node(state: dict) -> dict:
    """Retrieve top-3 similar training examples for few-shot context."""
    retriever = state["_retriever"]
    question  = state["question"]

    try:
        examples = retriever.retrieve(question, top_k=3)
        logger.debug(
            "[retrieve] top scores: %s",
            [f"{e['score']:.3f}" for e in examples],
        )
    except Exception as exc:
        logger.warning("[retrieve] retrieval failed: %s — proceeding zero-shot", exc)
        examples = []

    return {"retrieved_examples": examples}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 — generate_answer
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_node(state: dict) -> dict:
    """Generate an answer using the Phase 3 GRPO model.

    Uses the temperature schedule: 0.1 on first attempt, 0.4 / 0.7 on retries.
    Stores each attempt in `all_attempts` for best-of-N fallback.
    """
    model     = state["_model"]
    tokenizer = state["_tokenizer"]
    device    = state.get("_device", "cuda")
    question  = state["question"]
    retry     = state.get("retry_count", 0)
    examples  = state.get("retrieved_examples", [])

    temperature = _TEMP_SCHEDULE[min(retry, len(_TEMP_SCHEDULE) - 1)]

    # Only include few-shot context on first attempt; strip on retries to
    # avoid repeating the same incorrect examples.
    few_shot = _format_few_shot_block(examples) if retry == 0 else ""

    t0 = time.time()
    raw_output = _generate(model, tokenizer, question, few_shot, temperature, device)
    elapsed = time.time() - t0

    logger.info(
        "[generate] retry=%d  temp=%.2f  %.1fs  format=%.2f",
        retry, temperature, elapsed, _format_score(raw_output),
    )

    all_attempts = list(state.get("all_attempts", []))
    all_attempts.append(raw_output)

    return {
        "raw_output":    raw_output,
        "all_attempts":  all_attempts,
        "retry_count":   retry + 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 — verify_answer
# ──────────────────────────────────────────────────────────────────────────────

def verify_answer_node(state: dict) -> dict:
    """Check the generated answer against ground truth (eval mode)
    or validate format only (inference-only mode).
    """
    from src.symbolic_verifier import verify_answer, extract_answer_from_text

    raw_output   = state.get("raw_output", "")
    ground_truth = state.get("ground_truth", "")  # empty in inference-only mode

    predicted = extract_answer_from_text(raw_output)

    if ground_truth:
        # Evaluation mode: symbolic correctness check
        correct = verify_answer(predicted, ground_truth)
        logger.debug(
            "[verify] pred='%s'  gt='%s'  correct=%s",
            predicted[:60], ground_truth[:40], correct,
        )
    else:
        # Inference-only mode: accept if format is valid
        correct = bool(re.search(r"<answer>.*?</answer>", raw_output, re.DOTALL | re.IGNORECASE))
        logger.debug("[verify] inference-only mode  format_ok=%s", correct)

    return {
        "generated_answer": predicted,
        "verified":         correct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 4 — generate_explanation
# ──────────────────────────────────────────────────────────────────────────────

def generate_explanation_node(state: dict) -> dict:
    """Extract or select the best explanation from generated outputs.

    If the answer is verified correct, use the explanation from the raw_output.
    If all retries failed, pick the best-formatted attempt as a fallback,
    then extract its explanation.
    """
    verified     = state.get("verified", False)
    raw_output   = state.get("raw_output", "")
    all_attempts = state.get("all_attempts", [])

    if verified:
        # Happy path: use explanation from the correct attempt
        explanation = _extract_explanation(raw_output)
        selected    = raw_output
        confidence  = 0.85 + 0.15 * _format_score(raw_output)
    else:
        # Best-of-N fallback: pick attempt with highest format score
        if all_attempts:
            selected = max(all_attempts, key=_format_score)
        else:
            selected = raw_output

        explanation = _extract_explanation(selected)
        # Recompute the answer from the best attempt
        from src.symbolic_verifier import extract_answer_from_text
        best_answer = extract_answer_from_text(selected)

        logger.info(
            "[explain] fallback best-of-N — format=%.2f  answer='%s'",
            _format_score(selected), best_answer[:40],
        )

        # Update generated_answer to the best fallback
        state = dict(state)          # shallow copy to allow mutation
        state["generated_answer"] = best_answer
        confidence = 0.30 * _format_score(selected)  # low confidence

    if not explanation:
        explanation = "No explanation could be extracted from the model output."

    return {
        "explanation": explanation,
        "confidence":  round(confidence, 4),
        # Pass back the best selected raw output so format_output can use it
        "raw_output":  selected,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 5 — format_output
# ──────────────────────────────────────────────────────────────────────────────

def format_output_node(state: dict) -> dict:
    """Assemble the final XML output for EXACT competition submission.

    Output format:
        <answer>
        [answer text]
        </answer>
        <explanation>
        [explanation text]
        </explanation>
    """
    answer      = state.get("generated_answer", "").strip()
    explanation = state.get("explanation", "").strip()

    if not answer:
        answer = "unknown"
    if not explanation:
        explanation = "No explanation available."

    final_output = (
        f"<answer>\n{answer}\n</answer>\n"
        f"<explanation>\n{explanation}\n</explanation>"
    )

    logger.info(
        "[format] verified=%s  confidence=%.3f  answer='%s'",
        state.get("verified", False),
        state.get("confidence", 0.0),
        answer[:60],
    )

    return {"final_output": final_output}
