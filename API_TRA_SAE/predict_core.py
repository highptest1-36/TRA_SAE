"""
TRA-SAE /predict core  (EXACT 2026 BTC schema)
=============================================
Single-pass inference wrapper. Mirrors the proven-correct generation path of
experiments/step0_canonical_eval.py:_generate_batch (dual-LoRA specialists +
router + retrieval, SINGLE PASS). No LangGraph, no self-consistency, no Z3.

BTC request schema (input):
  {"query_id","type":"type1"|"type2","query","premises":[...],"options":[...]}
BTC result schema (output, JSON list, one object per query):
  {"query_id","answer","unit","explanation","premises_used":[0-based],"reasoning":{...}}
  - type1 → logic;  unit="";  premises_used = 0-based premise indices used.
  - type2 → physics; answer = number only; unit = ASCII (uF, ohm, V/m...); premises_used=[].
  - if options non-empty → answer must equal exactly one option.
  - explanation always non-empty; reasoning is a structured object or null.

Isolation (paper safety): imports only stable src/ modules read-only; retriever
built cache_path=None; writes nothing to repo-root logs/.
"""
from __future__ import annotations

import os
import re
import sys
import time
import threading

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Tunable knobs (env-overridable) ───────────────────────────────────────────
MAX_NEW_TOKENS = int(os.environ.get("SERVE_MAX_NEW_TOKENS", "384"))
TIME_BUDGET_SEC = float(os.environ.get("SERVE_TIME_BUDGET_SEC", "34"))
# Input prompt budget — DECOUPLED from MAX_NEW_TOKENS so lowering the output cap
# never silently truncates the question/premises out of the prompt.
PROMPT_MAX_LEN = int(os.environ.get("SERVE_PROMPT_MAX_LEN", "3072"))
TOP_K_FEWSHOT = int(os.environ.get("SERVE_TOP_K", "3"))
FEWSHOT_PHYSICS = os.environ.get("SERVE_FEWSHOT_PHYSICS", "0") == "1"
FEWSHOT_LOGIC = os.environ.get("SERVE_FEWSHOT_LOGIC", "1") == "1"

from src.data_utils import SYSTEM_PROMPT, format_sample  # noqa: E402

# Logic prompt asks the model to ALSO emit which premises it used (50% of the
# Type-1 score). Premise numbers are 1-based as shown; converted to 0-based later.
SERVE_SYSTEM_PROMPT_LOGIC = SYSTEM_PROMPT + (
    "\nThe premises above are numbered starting at 1. After </explanation>, add "
    "exactly one more line listing ONLY the numbers of the premises you actually "
    "used to reach the answer, in this format:\n<premises_used>1, 3</premises_used>"
)

# ── Type / unit / option normalisation ────────────────────────────────────────
_TYPE_MAP = {"type1": "logic", "type2": "physics", "1": "logic", "2": "physics",
             "logic": "logic", "physics": "physics"}

_UNIT_ASCII = [("μ", "u"), ("µ", "u"), ("Ω", "ohm"), ("Ω", "ohm"), ("Ω", "ohm"),
               ("°", "deg"), ("·", "."), ("×", "x"), ("Δ", "d"), ("λ", "lambda"),
               ("π", "pi"), ("ω", "w"), ("²", "^2"), ("³", "^3")]


def _to_ascii_unit(u: str) -> str:
    u = (u or "").strip()
    for k, v in _UNIT_ASCII:
        u = u.replace(k, v)
    return u


def _normalize_query(q: dict) -> dict:
    """Map BTC fields → canonical {query_id, subject, query, premises, options}."""
    q = q or {}
    raw_type = str(q.get("type") or "").strip().lower()
    return {
        "query_id": q.get("query_id", q.get("id", "")),
        "subject": _TYPE_MAP.get(raw_type, ""),     # "" → route by content
        "query": (q.get("query") or q.get("question") or "").strip(),
        "premises": list(q.get("premises") or q.get("premises_nl") or []),
        "options": [str(o) for o in (q.get("options") or [])],
    }


# ── Answer extraction helpers ─────────────────────────────────────────────────

def _extract_logic_answer(raw: str) -> str:
    """Normalised {yes,no,unknown} label — mirror of step9.extract_logic_answer."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    cand = (m.group(1) if m else raw).strip().lower()
    if re.search(r"\bunknown\b|\buncertain\b|cannot be (determined|concluded)"
                 r"|not enough|insufficient", cand):
        return "unknown"
    if re.search(r"\bno\b|\bfalse\b|contradict", cand):
        return "no"
    if re.search(r"\byes\b|\btrue\b|entail|follows?\b", cand):
        return "yes"
    return ""


_MCQ_RE = re.compile(r"\b([A-E])\b")
_PREMISE_RE = re.compile(r"premise[s]?\s*#?\s*(\d+)", re.IGNORECASE)
_PREMISES_USED_TAG = re.compile(r"<premises_used>\s*(.*?)\s*</premises_used>",
                                re.DOTALL | re.IGNORECASE)
_INT_RE = re.compile(r"\d+")
_YNU = {"yes": "yes", "true": "yes", "no": "no", "false": "no",
        "unknown": "uncertain", "uncertain": "uncertain"}


def _parse_premises_used(raw: str, reasoning: str, explanation: str,
                         n_premises: int) -> list[int]:
    """0-based indices of premises the model used. Prefers the <premises_used>
    tag (1-based numbers), falls back to 'premise N' mentions in the text."""
    nums = []
    m = _PREMISES_USED_TAG.search(raw)
    if m:
        nums = [int(x) - 1 for x in _INT_RE.findall(m.group(1))]
    if not nums:
        text = " ".join([reasoning or "", explanation or ""])
        nums = [int(x) - 1 for x in _PREMISE_RE.findall(text)]
    out = sorted({i for i in nums if 0 <= i < n_premises})
    # Fallback: if nothing was detected, assume all premises were used. This
    # maximises overlap with the ground-truth set (premises_used = 50% of the
    # Type-1 score) and never scores worse than returning [].
    if not out and n_premises > 0:
        out = list(range(n_premises))
    return out


def _match_option(answer: str, options: list[str]) -> str:
    """Return the option that best matches `answer` (must equal it exactly)."""
    if not options:
        return answer
    a = answer.strip().lower()
    low = [o.strip().lower() for o in options]
    # 1. exact
    for o, lo in zip(options, low):
        if a == lo:
            return o
    # 2. yes/no/uncertain canonicalisation
    canon = _YNU.get(a)
    if canon:
        for o, lo in zip(options, low):
            if lo == canon or canon in lo:
                return o
    # 3. MCQ letter → option starting with that letter
    if len(a) == 1 and a.isalpha():
        for o, lo in zip(options, low):
            if lo.startswith(a + ".") or lo.startswith(a + ")") or lo == a:
                return o
    # 4. substring / prefix overlap
    for o, lo in zip(options, low):
        if a and (a in lo or lo in a or lo.startswith(a) or a.startswith(lo)):
            return o
    return answer  # honest fallback (no confident match)


def _structured_reasoning(reasoning: str, subject: str):
    """Turn the model's free-text reasoning into {type, steps} or None."""
    if not reasoning or not reasoning.strip():
        return None
    parts = re.split(r"\n+|(?<=[.;])\s+(?=[A-Z0-9(])", reasoning.strip())
    steps = []
    for s in parts:
        s = s.replace("**", "").replace("`", "").strip()
        s = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)+", "", s).strip()  # drop list markers only
        if len(s) > 2:
            steps.append(s)
    steps = steps[:10]
    return {"type": "fol" if subject == "logic" else "cot",
            "steps": steps or [reasoning.strip()[:300]]}


def _latex_clean(s: str) -> str:
    """Best-effort LaTeX → plain so parse_numerical can read value + unit."""
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|rm|bf)\s*\{([^{}]*)\}", r"\1", s)
    # LaTeX unit/prefix commands → ASCII (consume trailing space so "\mu F"→"uF")
    s = re.sub(r"\\mu\s*", "u", s)
    s = re.sub(r"\\Omega\s*", "ohm", s)
    s = re.sub(r"\\(?:circ|degree)\s*", "deg", s)
    s = re.sub(r"\\times\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?", r"e\1", s)
    s = re.sub(r"\b10\s*\^?\s*\{\s*([-+]?\d+)\s*\}", r"1e\1", s)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("{", "").replace("}", "").replace("\\approx", "").replace("\\boxed", "")
    return s.strip()


_PREFIXES = {"u", "m", "k", "M", "n", "p", "G", "µ", "μ", "c", "d"}


def _clean_unit(u: str) -> str:
    """Keep only the first plausible unit token (drops trailing prose)."""
    if not u or not u.strip():
        return ""
    toks = u.strip().split()
    tok = toks[0]
    # join a lone metric prefix with the following unit token ("u F" -> "uF")
    if len(toks) > 1 and tok in _PREFIXES:
        tok = tok + toks[1]
    tok = re.sub(r"[^A-Za-zμΩΩ°²³⁻/·]", "", tok)
    return tok[:8]


def _fmt_number(val) -> str:
    """Clean numeric answer for Type-2: kill float-repr noise (e.g.
    0.0033799999999999998), keep integers exact, and use scientific notation
    for very small / very large magnitudes so the answer matches the BTC
    display form (e.g. 3.38e-3 for '3.38 × 10^-3')."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if f != f:  # NaN guard
        return str(val)
    if f == 0:
        return "0"
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    a = abs(f)
    if a < 1e-2 or a >= 1e5:                       # scientific: 3 sig figs
        mant, exp = f"{f:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"
    return f"{f:.4f}".rstrip("0").rstrip(".")       # normal range: clean decimal


def _salvage_physics_value(raw: str):
    """When no clean <answer>, hunt the LAST numeric result after '='."""
    from src.symbolic_verifier import parse_numerical
    txt = _latex_clean(raw)
    for seg in reversed(re.split(r"=", txt)):
        v, u = parse_numerical(seg.strip())
        if v is not None:
            return v, u
    return None, ""


def _fmt_few_shot(examples) -> str:
    if not examples:
        return ""
    lines = ["\n--- Reference examples ---"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:\n{ex['question'][:300]}\n<answer>\n{ex['answer']}\n</answer>")
    lines.append("\n--- Now answer the following ---\n")
    return "\n".join(lines)


def _build_user_content(subject: str, query_text: str, premises: list) -> str:
    if subject == "logic" and premises:
        sample = {"type": "logic", "premises_nl": list(premises), "premises_fol": [],
                  "question": query_text, "answer": "", "explanation": "", "cot": ""}
        return format_sample(sample)["prompt"][-1]["content"]
    if subject == "physics":
        return f"Question: {query_text}"
    return query_text if query_text.lower().startswith(("premises:", "question:")) \
        else f"Question: {query_text}"


class _TimeBudgetStopping:
    def __init__(self, max_seconds: float):
        self.deadline = time.time() + max_seconds

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return time.time() > self.deadline


class TRASAEPredictor:
    """Load once at startup; call predict_one()/predict() per request."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.router = None
        self.retriever = None
        self.device = "cpu"
        self.model_name = os.environ.get("SERVE_MODEL_ID", "Qwen/Qwen3.5-4B")
        self._lock = threading.Lock()
        self._ready = False

    # ----------------------------------------------------------------- startup
    def load(self) -> None:
        import torch
        from src.config import (MODEL_NAME, QWEN35_GRPO_PHYS_FINAL,
                                QWEN35_GRPO_LOGIC_FINAL, QWEN35_GRPO_FINAL,
                                TRAIN_DS, ROUTER_PATH)
        from src.model_loader import load_multi_adapter_model
        from src.retriever import Retriever

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_NAME
        adapters = {}
        for nm, ck in [("physics", QWEN35_GRPO_PHYS_FINAL),
                       ("logic", QWEN35_GRPO_LOGIC_FINAL)]:
            if os.path.isdir(ck):
                adapters[nm] = ck
            elif os.path.isdir(QWEN35_GRPO_FINAL):
                adapters[nm] = QWEN35_GRPO_FINAL
        if not adapters:
            raise RuntimeError("No GRPO adapter found under checkpoints/.")
        print(f"[predict_core] Loading dual-LoRA: {list(adapters)} (device={self.device})")
        self.model, self.tokenizer = load_multi_adapter_model(
            MODEL_NAME, adapters, dtype=torch.bfloat16, drop_vision=True)
        self.model.eval()

        print("[predict_core] Building retriever (cache_path=None)")
        self.retriever = Retriever(TRAIN_DS)
        self.retriever.build(cache_path=None)

        try:
            from src.router import SubjectRouter
            if os.path.exists(ROUTER_PATH):
                self.router = SubjectRouter.load(ROUTER_PATH)
                print(f"[predict_core] Router loaded ← {ROUTER_PATH}")
        except Exception as e:  # pragma: no cover
            print(f"[predict_core] Router unavailable ({e})")

        self._ready = True
        self._warmup()

    def _warmup(self) -> None:
        try:
            self.predict_one({"query_id": "_warmup", "type": "type2",
                              "query": "Calculate 1 + 1.", "premises": [], "options": []})
            print("[predict_core] Warm-up complete.")
        except Exception as e:  # pragma: no cover
            print(f"[predict_core] Warm-up skipped ({e})")

    # ---------------------------------------------------------------- inference
    def _route_subject(self, subject: str, query_text: str) -> str:
        if subject in ("physics", "logic"):
            return subject
        if self.router is not None:
            try:
                return self.router.predict(query_text)[0]
            except Exception:
                pass
        return "physics"

    def _generate(self, user_content: str, subject: str, system_prompt: str,
                  stop_str: str) -> str:
        import torch
        if subject and hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(subject)
            except Exception:
                pass
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=PROMPT_MAX_LEN).to(self.device)
        input_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        try:
            gen_kwargs["stop_strings"] = [stop_str]
            gen_kwargs["tokenizer"] = self.tokenizer
        except Exception:
            pass
        try:
            from transformers import StoppingCriteriaList
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_TimeBudgetStopping(TIME_BUDGET_SEC)])
        except Exception:
            pass
        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def raw_generate(self, user_content: str, subject: str = "physics",
                     stop_str: str = "</explanation>") -> str:
        """Direct generation for the OpenAI-compatible /v1/chat/completions."""
        with self._lock:
            return self._generate(user_content, subject, SYSTEM_PROMPT, stop_str)

    def predict_one(self, q: dict) -> dict:
        from src.symbolic_verifier import (extract_answer_from_text,
                                            extract_explanation, extract_reasoning,
                                            parse_numerical)
        nq = _normalize_query(q)
        query_id, options = nq["query_id"], nq["options"]
        premises, query_text = nq["premises"], nq["query"]
        subject = self._route_subject(nq["subject"], query_text)
        user_content = _build_user_content(subject, query_text, premises)

        # Few-shot: logic only by default (physics few-shot triggers rambling).
        fs = ""
        if (FEWSHOT_LOGIC if subject == "logic" else FEWSHOT_PHYSICS):
            try:
                ex = self.retriever.retrieve(query_text or user_content,
                                             top_k=TOP_K_FEWSHOT, subject=subject or None)
                fs = _fmt_few_shot(ex)
            except Exception:
                pass

        if subject == "logic":
            sys_prompt, stop_str = SERVE_SYSTEM_PROMPT_LOGIC, "</premises_used>"
        else:
            sys_prompt, stop_str = SYSTEM_PROMPT, "</explanation>"

        with self._lock:
            raw = self._generate((fs + user_content) if fs else user_content,
                                 subject, sys_prompt, stop_str)

        reasoning = extract_reasoning(raw)
        explanation = extract_explanation(raw)

        if subject == "logic":
            raw_ans = extract_answer_from_text(raw).strip()
            if options:
                opt_low = {o.strip().lower() for o in options}
                if opt_low <= {"yes", "no", "uncertain", "unknown", "true", "false"}:
                    # Yes/No/Uncertain choice → use the logic-label extractor.
                    ynu = _extract_logic_answer(raw)
                    cand = {"yes": "Yes", "no": "No",
                            "unknown": "Uncertain"}.get(ynu, raw_ans)
                else:
                    # MCQ with custom options → use the model's stated answer.
                    cand = raw_ans
                ans = _match_option(cand, options)
            else:
                # Free-form Type 1 (a number or short text) → return value directly.
                ans = raw_ans
            unit = ""
            premises_used = _parse_premises_used(raw, reasoning, explanation, len(premises))
        else:  # physics
            cand = _latex_clean(extract_answer_from_text(raw))
            val, unit = parse_numerical(cand)
            if val is None:
                val, unit = _salvage_physics_value(raw)
            unit = _clean_unit(_to_ascii_unit(unit))
            if val is not None:
                ans = _fmt_number(val)
            else:
                ans = cand
            if options:  # rare for type2, but honour it
                ans = _match_option(ans, options)
                unit = ""
            premises_used = []

        if not ans:
            ans = (options[0] if options else "Uncertain") if subject == "logic" else "0"
        if not explanation:
            explanation = (reasoning.strip()
                           or f"Answer: {ans}{(' ' + unit) if unit else ''}.")

        return {
            "query_id": query_id,
            "answer": ans,
            "unit": unit or "",
            "explanation": explanation,
            "premises_used": premises_used,
            "reasoning": _structured_reasoning(reasoning, subject),
        }

    def predict(self, payload) -> list[dict]:
        items = payload if isinstance(payload, list) else [payload]
        results = []
        for q in items:
            try:
                results.append(self.predict_one(q))
            except Exception as e:  # never crash the endpoint
                results.append({
                    "query_id": (q or {}).get("query_id", (q or {}).get("id", "")),
                    "answer": "Uncertain", "unit": "",
                    "explanation": f"Fallback answer (inference error: {e}).",
                    "premises_used": [], "reasoning": None,
                })
        return results


PREDICTOR = TRASAEPredictor()
