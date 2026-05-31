"""
TRA-SAE Symbolic Verifier v2
=============================
Verifies model answers against ground-truth using rule-based matching.

v2 improvements over v1:
  - Variable-prefix stripping    ("I_total = 1.5 A"  → "1.5 A")
  - Scientific notation parsing  (9.71 × 10^7 / 9.71e7 / 9.71*10^7)
  - SI unit conversion table     (μJ ↔ J,  mA ↔ A,  kΩ ↔ Ω …)
  - Yes/No word-boundary search  ("The answer is no." → "No")
  - Z3 SMT hook for logic Yes/No/Unknown (calls src.z3_engine when available)

Answer types handled:
  - MCQ:              A / B / C / D  (letter comparison)
  - Yes/No/Unknown:   mapped synonym comparison + word-boundary search
  - Numerical:        float comparison with ±2% relative tolerance + SI norm
  - Free text:        normalised exact match (fallback)
"""
from __future__ import annotations

import re
import math
import logging
from typing import Optional

logger = logging.getLogger("tra-sae.verifier")


# ══════════════════════════════════════════════════════════════════════════════
#  SI Unit Conversion Table
#  All keys are lowercase strings as they appear in answers.
#  Values are the scale factor to multiply by to reach the SI base unit.
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
UNIT_TABLE: dict[str, float] = {
    # ── Energy (base: J) ────────────────────────────────────────────────────
    "pj": 1e-12,  "nj": 1e-9,   "μj": 1e-6,  "uj": 1e-6,  "mj": 1e-3,
    "j":  1.0,    "kj": 1e3,    "mj_mega": 1e6,
    # ── Charge (base: C) ─────────────────────────────────────────────────────
    "pc": 1e-12,  "nc": 1e-9,   "μc": 1e-6,  "uc": 1e-6,  "mc": 1e-3,
    "c":  1.0,
    # ── Current (base: A) ─────────────────────────────────────────────────────
    "pa": 1e-12,  "na": 1e-9,   "μa": 1e-6,  "ua": 1e-6,  "ma": 1e-3,
    "a":  1.0,    "ka": 1e3,
    # ── Voltage (base: V) ─────────────────────────────────────────────────────
    "μv": 1e-6,   "uv": 1e-6,   "mv": 1e-3,
    "v":  1.0,    "kv": 1e3,    "mv_mega": 1e6,
    # ── Capacitance (base: F) ─────────────────────────────────────────────────
    "pf": 1e-12,  "nf": 1e-9,   "μf": 1e-6,  "uf": 1e-6,  "mf": 1e-3,
    "f":  1.0,
    # ── Resistance (base: Ω) ──────────────────────────────────────────────────
    "mω": 1e-3,   "ω":  1.0,    "kω": 1e3,   "mω_mega": 1e6,
    "mohm": 1e-3, "ohm": 1.0,   "kohm": 1e3,
    # ── Inductance (base: H) ──────────────────────────────────────────────────
    "nh": 1e-9,   "μh": 1e-6,   "uh": 1e-6,  "mh": 1e-3,
    "h":  1.0,
    # ── Frequency (base: Hz) ─────────────────────────────────────────────────
    "hz": 1.0,    "khz": 1e3,   "mhz": 1e6,  "ghz": 1e9,
    # ── Time (base: s) ────────────────────────────────────────────────────────
    "ps": 1e-12,  "ns": 1e-9,   "μs": 1e-6,  "us": 1e-6,  "ms": 1e-3,
    "s":  1.0,    "min": 60.0,
    # ── Length (base: m) ──────────────────────────────────────────────────────
    "pm": 1e-12,  "nm": 1e-9,   "μm": 1e-6,  "um": 1e-6,  "mm": 1e-3,
    "cm": 1e-2,   "dm": 1e-1,   "m":  1.0,   "km": 1e3,
    # ── Force (base: N) ───────────────────────────────────────────────────────
    "mn": 1e-3,   "n":  1.0,    "kn": 1e3,
    # ── Power (base: W) ───────────────────────────────────────────────────────
    "pw": 1e-12,  "nw": 1e-9,   "μw": 1e-6,  "uw": 1e-6,  "mw": 1e-3,
    "w":  1.0,    "kw": 1e3,    "mw_mega": 1e6,
    # ── Magnetic flux density (base: T) ───────────────────────────────────────
    "μt": 1e-6,   "ut": 1e-6,   "mt": 1e-3,  "t": 1.0,
    # ── Electric field (base: V/m) ────────────────────────────────────────────
    "v/m": 1.0,   "kv/m": 1e3,  "mv/m": 1e-3,
    # ── Dimensionless / special ───────────────────────────────────────────────
    "":   1.0,    "%": 0.01,    "rad": 1.0,
}
# fmt: on

_OMEGA_RE = re.compile(r"[ΩΩ\u03a9\u2126]")

# ── Variable-prefix stripping ─────────────────────────────────────────────────
_VAR_PREFIX_RE = re.compile(
    r"^[A-Za-z_][A-Za-z_0-9]*(?:\s*\([^)]*\))?\s*=\s*",
    re.UNICODE,
)


def _strip_var_prefix(s: str) -> str:
    """Strip leading variable-assignment prefix.  'I_total = 1.5 A' → '1.5 A'."""
    return _VAR_PREFIX_RE.sub("", s).strip()


# ── Scientific notation ───────────────────────────────────────────────────────
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SCI_RE = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)"
    r"\s*"
    r"(?:"
    r"[×xX\*·]\s*10\s*[\^＾]?\s*(?:\(([+-]?\d+)\)|([+-]?\d+))"
    r"|"
    r"[eE]([+-]?\d+)"
    r")",
    re.UNICODE,
)


def _parse_scientific(s: str) -> tuple[float | None, str]:
    """Parse scientific notation; return (value, remaining_unit_string)."""
    s2 = s.translate(_SUPERSCRIPT_MAP)
    m  = _SCI_RE.search(s2)
    if m:
        mantissa = float(m.group(1).replace(",", "."))
        exp_str  = m.group(2) or m.group(3) or m.group(4)
        value    = mantissa * (10.0 ** int(exp_str))
        remaining = s2[m.end():].strip().strip(".,;")
        return value, remaining
    return None, s


# ══════════════════════════════════════════════════════════════════════════════
#  Numerical parser  (public — also used by reward.py)
# ══════════════════════════════════════════════════════════════════════════════

def parse_numerical(ans: str) -> tuple[float | None, str]:
    """Parse numerical value + optional unit.  Returns (raw_value, unit_str)."""
    s = _strip_var_prefix(ans.strip())
    sci_val, remaining = _parse_scientific(s)
    if sci_val is not None:
        return sci_val, remaining.strip().strip(".,;")
    m = re.match(
        r"^([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"([a-zA-ZμΩ°²³⁻/·\s\u03a9\u2126]*)",
        s,
        re.UNICODE,
    )
    if m:
        try:
            return float(m.group(1).replace(",", ".")), m.group(2).strip()
        except ValueError:
            pass
    return None, ""


def _to_si(value: float, unit: str) -> float:
    """Convert value+unit to SI base.  Unknown units → unchanged value."""
    key   = _OMEGA_RE.sub("ω", unit.strip()).replace(" ", "").lower()
    scale = UNIT_TABLE.get(key)
    if scale is not None:
        return value * scale
    scale = UNIT_TABLE.get(re.sub(r"[23]$", "", key))
    if scale is not None:
        return value * scale
    logger.debug(f"[verifier] Unknown unit '{unit}' — no SI normalisation")
    return value


def same_unit_scale(predicted: str, ground_truth: str) -> bool:
    """True when predicted and GT have the same SI scale factor (for reward bonus)."""
    _, pu = parse_numerical(_strip_var_prefix(predicted))
    _, gu = parse_numerical(_strip_var_prefix(ground_truth))
    if not pu and not gu:
        return True
    pk = _OMEGA_RE.sub("ω", pu.strip()).replace(" ", "").lower()
    gk = _OMEGA_RE.sub("ω", gu.strip()).replace(" ", "").lower()
    ps, gs = UNIT_TABLE.get(pk), UNIT_TABLE.get(gk)
    if ps is None or gs is None:
        return pk == gk
    return ps == gs


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_boxed_content(text: str):
    """Extract last \\boxed{...} handling nested braces. Returns str or None."""
    marker = r"\boxed{"
    last_pos, search_start = -1, 0
    while True:
        pos = text.find(marker, search_start)
        if pos == -1:
            break
        last_pos, search_start = pos, pos + 1
    if last_pos == -1:
        return None
    start = last_pos + len(marker)
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def _simplify_latex(expr: str) -> str:
    """Best-effort LaTeX → plain text for numeric/logic answers."""
    # \frac{a}{b} → a/b  (iterate for nested)
    for _ in range(5):
        m = re.search(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", expr)
        if not m:
            break
        expr = expr[: m.start()] + f"{m.group(1)}/{m.group(2)}" + expr[m.end() :]
    # \text{...}, \mathrm{...} etc.
    expr = re.sub(r"\\(?:text|mathrm|rm|bf|it|mbox)\{([^{}]*)\}", r"\1", expr)
    # \sqrt{x} → just x
    expr = re.sub(r"\\sqrt\{([^{}]*)\}", r"\1", expr)
    # drop remaining LaTeX commands and stray chars
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = re.sub(r"[{}$^_]", "", expr)
    return expr.strip()


def extract_answer_from_text(text: str) -> str:
    """Content inside <answer>…</answer>; then \\boxed{}; fallback to last non-empty line."""
    # 1. Our own <answer> tag
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text,
                  flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2. LaTeX \boxed{} — used by DeepSeek-Math, Llemma, Qwen, etc.
    boxed = _extract_boxed_content(text)
    if boxed is not None:
        return _simplify_latex(boxed)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_explanation(text: str) -> str:
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text,
                  flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_reasoning(text: str) -> str:
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text,
                  flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def normalize(ans: str) -> str:
    """Lowercase, strip, remove trailing punctuation."""
    ans = ans.strip().lower()
    return re.sub(r"[.,;:!?]+$", "", ans)


# ── MCQ ───────────────────────────────────────────────────────────────────────

def verify_mcq(predicted: str, ground_truth: str) -> bool:
    _strip = lambda s: re.sub(r"^(option\s*|answer\s*|choice\s*)", "", normalize(s))
    return _strip(predicted)[:1].upper() == _strip(ground_truth)[:1].upper()


# ── Yes / No / Unknown ────────────────────────────────────────────────────────

_YES_NO_MAP: dict[str, str] = {
    "yes": "yes",  "true": "yes",   "correct": "yes",  "affirmative": "yes",
    "no":  "no",   "false": "no",   "incorrect": "no", "negative": "no",
    "unknown": "unknown", "uncertain": "unknown", "maybe": "unknown",
    "cannot be determined": "unknown", "indeterminate": "unknown",
}


def verify_yes_no(predicted: str, ground_truth: str) -> bool:
    """Yes/No/Unknown with word-boundary search across full text."""
    gt_norm = _YES_NO_MAP.get(normalize(ground_truth), normalize(ground_truth))
    pred_direct = _YES_NO_MAP.get(normalize(_strip_var_prefix(predicted)))
    if pred_direct is not None:
        return pred_direct == gt_norm
    pred_lower = predicted.lower()
    for kw in sorted(_YES_NO_MAP, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"\b", pred_lower):
            return _YES_NO_MAP[kw] == gt_norm
    return normalize(predicted) == gt_norm


# ── Numerical ─────────────────────────────────────────────────────────────────

def verify_numerical(
    predicted: str,
    ground_truth: str,
    rel_tol: float = 0.02,
) -> bool:
    """Float comparison with SI normalisation and ±2% tolerance."""
    pred_raw, pred_unit = parse_numerical(_strip_var_prefix(predicted))
    gt_raw,   gt_unit   = parse_numerical(_strip_var_prefix(ground_truth))
    if pred_raw is None or gt_raw is None:
        return normalize(predicted) == normalize(ground_truth)
    pred_si = _to_si(pred_raw, pred_unit)
    gt_si   = _to_si(gt_raw,   gt_unit)
    if gt_si == 0:
        return abs(pred_si) < 1e-9
    return abs(pred_si - gt_si) / abs(gt_si) <= rel_tol


# ── Unified entry point ───────────────────────────────────────────────────────

def verify_answer(
    predicted: str,
    ground_truth: str,
    subject: str = "",
    question_text: str = "",
    use_z3: bool = True,
) -> bool:
    """Auto-detect answer type and verify.

    v2 detection order:
      1. MCQ              (GT is a single letter a–d)
      2. Yes/No/Unknown   (GT in synonym set) — Z3 hook for logic subject
      3. Numerical        (both parse to float)
      4. Exact string     (normalised fallback)
    """
    if "<answer>" in predicted.lower():
        predicted = extract_answer_from_text(predicted)

    gt_norm = normalize(ground_truth)

    # 1. MCQ
    if gt_norm in ("a", "b", "c", "d"):
        return verify_mcq(predicted, ground_truth)

    # 2. Yes / No / Unknown  (+ optional Z3 hook for logic)
    if gt_norm in ("yes", "no", "unknown", "true", "false", "uncertain"):
        if use_z3 and subject == "logic" and question_text:
            try:
                from src.z3_engine import Z3Engine
                z3_result = Z3Engine.verify(question_text, ground_truth)
                if z3_result is not None:
                    return z3_result
            except Exception as exc:
                logger.debug(f"[verifier] Z3 failed ({exc}) — text fallback")
        return verify_yes_no(predicted, ground_truth)

    # 3. Numerical
    pred_val, _ = parse_numerical(_strip_var_prefix(predicted))
    gt_val,   _ = parse_numerical(_strip_var_prefix(ground_truth))
    if pred_val is not None and gt_val is not None:
        return verify_numerical(predicted, ground_truth)

    # 4. Fallback exact match
    return (normalize(_strip_var_prefix(predicted)) ==
            normalize(_strip_var_prefix(ground_truth)))

