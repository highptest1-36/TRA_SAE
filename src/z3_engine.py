"""
TRA-SAE Z3 SMT Engine (FOL Verification)
==========================================
Verifies logical deductions in the Logic dataset using the Z3 SMT solver.

Data layout (raw JSON):
  Each record has:
    premises-FOL  : list[str]   — all FOL premises in the problem set
    questions     : list[str]   — individual questions (MCQ or yes/no)
    answers       : list[str]   — corresponding answers ('Yes'/'No'/'Unknown'/letter)
    idx           : list[list[int]] — 1-based premise indices relevant per question

FOL syntax in the dataset (two styles):
  Style A (Unicode):  ∀x (P(x) → Q(x)),  ∃x (P(x)),  ∀x (¬P(x) → ¬Q(x))
  Style B (Python):   ForAll(x, Implies(P(x), Q(x)))

Z3 verification strategy for yes/no questions:
  "Yes"     → premises ⊨ conclusion  ⟺  SAT(premises ∧ ¬conclusion) is UNSAT
  "No"      → premises ⊭ conclusion  ⟺  SAT(premises ∧ ¬conclusion) is SAT
  "Unknown" → no firm verdict either way

Fallback: returns None when parsing fails → caller uses text matching.
"""
from __future__ import annotations

import re
import json
import logging
import os
from typing import Optional

logger = logging.getLogger("tra-sae.z3_engine")

# ── Try to import z3 ──────────────────────────────────────────────────────────
try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False
    logger.warning("[z3_engine] z3-solver not installed — Z3Engine.verify() always returns None")

_LOGIC_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "Logic_Based_Educational_Queries_Text_Only",
    "Logic_Based_Educational_Queries.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# FOL parser helpers
# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZE_WS = re.compile(r"\s+")

def _norm(text: str) -> str:
    return _NORMALIZE_WS.sub(" ", text.strip().lower())


def _extract_predicates(fol: str) -> set[str]:
    """Extract predicate names (upper-case single token or snake_case) from FOL string."""
    # Style A: single uppercase letters like P, Q, WT, PEP8
    style_a = set(re.findall(r"\b([A-Z][A-Z0-9_]{0,15})\s*\(", fol))
    # Style B: snake_case predicates like completed_core_curriculum
    style_b = set(re.findall(r"\b([a-z][a-z0-9_]{3,})\s*\(", fol))
    return style_a | style_b


class _FOLParser:
    """Convert a single FOL string to a Z3 expression.

    Handles:
      1. ∀x (P(x) → Q(x))
      2. ∀x (¬P(x))
      3. ∀x ((A(x) ∧ B(x)) → C(x))
      4. ∃x (P(x))
      5. ForAll(x, P(x) → Q(x))
      6. Simple ground facts: P(c) where c is a constant

    Returns z3.ExprRef | None
    """

    def __init__(self, all_preds: set[str]) -> None:
        """
        all_preds: set of all predicate names seen across all premises in
                   a problem instance — used to pre-create z3 FuncDecls.
        """
        if not _Z3_AVAILABLE:
            return

        # All predicates are unary (P: Entity → Bool)
        # We use a single entity sort
        self._Entity = z3.DeclareSort("Entity")
        self._x      = z3.Const("x", self._Entity)

        # Build FuncDecl dict
        self._funcs: dict[str, z3.FuncDeclRef] = {}
        for p in all_preds:
            self._funcs[p] = z3.Function(p, self._Entity, z3.BoolSort())

    def _call(self, name: str, var: "z3.ExprRef") -> "z3.ExprRef | None":
        """Return P(var) for predicate name, or None if unknown."""
        func = self._funcs.get(name)
        if func is None:
            return None
        return func(var)

    def _parse_atom(self, text: str, var: "z3.ExprRef") -> "z3.ExprRef | None":
        """Parse P(x) or ¬P(x)."""
        text = text.strip()
        # Negation
        negated = False
        neg_patterns = ("¬", "~", "NOT ", "not ")
        for np in neg_patterns:
            if text.startswith(np):
                negated = True
                text = text[len(np):].strip()
                break

        # Extract predicate name
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*\(\s*\w+\s*\)$", text)
        if not m:
            return None
        pred = m.group(1)
        atom = self._call(pred, var)
        if atom is None:
            return None
        return z3.Not(atom) if negated else atom

    def _parse_conjunction(self, text: str, var: "z3.ExprRef") -> "z3.ExprRef | None":
        """Parse A(x) ∧ B(x) [∧ C(x) …] → z3.And(…)."""
        parts = re.split(r"\s*[∧&]\s*", text)
        atoms = [self._parse_atom(p, var) for p in parts]
        if any(a is None for a in atoms):
            return None
        return z3.And(*atoms) if len(atoms) > 1 else atoms[0]

    def _parse_formula(self, text: str, var: "z3.ExprRef") -> "z3.ExprRef | None":
        """Parse antecedent or consequent of an implication.

        Handles: atom, negated atom, conjunction.
        """
        text = text.strip().lstrip("(").rstrip(")")
        # Try conjunction first
        if re.search(r"[∧&]", text):
            return self._parse_conjunction(text, var)
        return self._parse_atom(text, var)

    def parse(self, fol: str) -> "z3.ExprRef | None":
        """Top-level parse of a single FOL premise string."""
        if not _Z3_AVAILABLE:
            return None

        fol = fol.strip()
        var = self._x

        # ─── Style B: ForAll(x, ...) / Exists(x, ...) ─────────────────
        # Normalize ForAll / Exists
        style_b_all = re.match(r"^ForAll\s*\(\s*\w+\s*,\s*(.*)\)$", fol, re.DOTALL)
        if style_b_all:
            inner = style_b_all.group(1).strip()
            parsed = self._parse_inner_formula(inner, var)
            if parsed is not None:
                return z3.ForAll([var], parsed)

        style_b_ex = re.match(r"^Exists\s*\(\s*\w+\s*,\s*(.*)\)$", fol, re.DOTALL)
        if style_b_ex:
            inner = style_b_ex.group(1).strip()
            parsed = self._parse_inner_formula(inner, var)
            if parsed is not None:
                return z3.Exists([var], parsed)

        # ─── Style A: ∀x (…) ──────────────────────────────────────────
        style_a_all = re.match(r"^[∀∃]?\s*[Ff]or[Aa]ll\s*\w+\s*[,:]?\s*(.*)$", fol, re.DOTALL)
        if re.match(r"^∀\s*\w+\s*\(.*\)$", fol, re.DOTALL):
            inner_m = re.match(r"^∀\s*\w+\s*\((.*)\)\s*$", fol, re.DOTALL)
            if inner_m:
                inner = inner_m.group(1).strip()
                parsed = self._parse_inner_formula(inner, var)
                if parsed is not None:
                    return z3.ForAll([var], parsed)

        # ∃x (P(x))
        if re.match(r"^∃\s*\w+\s*\(.*\)$", fol, re.DOTALL):
            inner_m = re.match(r"^∃\s*\w+\s*\((.*)\)\s*$", fol, re.DOTALL)
            if inner_m:
                inner = inner_m.group(1).strip()
                parsed = self._parse_inner_formula(inner, var)
                if parsed is not None:
                    return z3.Exists([var], parsed)

        # Ground fact: P(const) — treat as assertion on x
        gf = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*\(\s*\w+\s*\)$", fol)
        if gf:
            atom = self._call(gf.group(1), var)
            if atom is not None:
                return z3.ForAll([var], atom)  # treat as universally true

        return None

    def _parse_inner_formula(self, inner: str, var: "z3.ExprRef") -> "z3.ExprRef | None":
        """Parse the body inside ∀x(…) or ForAll(x, …)."""
        # Implication: A → B  (→, ->, ⟹)
        for arrow in ("→", "->", "⟹", "=>", "Implies"):
            if arrow in inner:
                # Split on first occurrence
                idx = inner.index(arrow)
                lhs = inner[:idx].strip().strip("(").strip()
                rhs = inner[idx + len(arrow):].strip().strip(")").strip()
                # Remove wrapping outer parens from rhs if present
                if rhs.startswith("(") and rhs.endswith(")"):
                    rhs = rhs[1:-1].strip()
                antecedent = self._parse_formula(lhs, var)
                consequent = self._parse_formula(rhs, var)
                if antecedent is not None and consequent is not None:
                    return z3.Implies(antecedent, consequent)
                return None

        # No implication — try plain formula (atom or conjunction)
        return self._parse_formula(inner, var)


# ─────────────────────────────────────────────────────────────────────────────
# Z3Engine
# ─────────────────────────────────────────────────────────────────────────────

class Z3Engine:
    """Static engine — loaded once, used throughout inference.

    Lookup table: question_text_normalized → {
        premises_fol: list[str],   — full set of FOL premises for this problem
        relevant_fol: list[str],   — only premises indexed by idx field
        answer: str,               — ground truth ('Yes' | 'No' | 'Unknown' | letter)
    }
    """

    # Class-level state (lazy-loaded)
    _lookup:  dict[str, dict] = {}
    _loaded:  bool = False
    _timeout: int  = 5000   # ms

    @classmethod
    def _load_data(cls) -> None:
        if cls._loaded:
            return
        try:
            with open(_LOGIC_JSON_PATH, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            logger.warning(f"[z3_engine] Data file not found: {_LOGIC_JSON_PATH}")
            cls._loaded = True
            return

        for record in raw:
            premises_fol: list[str] = record.get("premises-FOL", [])
            questions:    list[str] = record.get("questions", [])
            answers:      list[str] = record.get("answers", [])
            idx_lists:    list      = record.get("idx", [])   # list[list[int]] — 1-based

            for q_i, (q_text, ans) in enumerate(zip(questions, answers)):
                # Determine which premises are relevant for this question
                if q_i < len(idx_lists) and idx_lists[q_i]:
                    # 1-based indices
                    relevant = [
                        premises_fol[i - 1]
                        for i in idx_lists[q_i]
                        if isinstance(i, int) and 1 <= i <= len(premises_fol)
                    ]
                else:
                    relevant = list(premises_fol)  # use all if no specific idx

                key = _norm(q_text)
                cls._lookup[key] = {
                    "premises_fol": premises_fol,
                    "relevant_fol": relevant,
                    "answer": str(ans).strip(),
                }

        logger.info(f"[z3_engine] Loaded {len(cls._lookup)} logic question entries")
        cls._loaded = True

    @classmethod
    def verify(
        cls,
        question_text: str,
        ground_truth: str,
    ) -> Optional[bool]:
        """Attempt Z3 verification for a logic question.

        Args:
            question_text:  The question as presented to the model
                            (may include "Premises: ... Question: ..." wrapping).
            ground_truth:   Expected answer from dataset ('Yes' / 'No' / 'Unknown').

        Returns:
            True   — Z3 confirms the ground_truth answer is correct
            False  — Z3 contradicts the ground_truth answer
            None   — Z3 can't determine (parse failure / timeout / Unknown)
        """
        if not _Z3_AVAILABLE:
            return None

        cls._load_data()
        if not cls._lookup:
            return None

        # Extract the core question text (strip premises block if present)
        core_q = cls._extract_question(question_text)
        key    = _norm(core_q)

        # Lookup with exact match, then try substring match
        entry = cls._lookup.get(key)
        if entry is None:
            entry = cls._fuzzy_lookup(key)
        if entry is None:
            logger.debug(f"[z3_engine] No entry found for: '{core_q[:60]}...'")
            return None

        gt_norm = ground_truth.strip().lower()
        if gt_norm not in ("yes", "no"):
            # Can't easily verify 'Unknown' or MCQ with Z3
            return None

        relevant_fol = entry["relevant_fol"]
        if not relevant_fol:
            return None

        # Collect all predicates for FuncDecl setup
        all_preds: set[str] = set()
        for f in relevant_fol:
            all_preds |= _extract_predicates(f)

        # Also parse conclusion from the question
        conclusion_fol = cls._extract_conclusion_fol(core_q, all_preds)
        if conclusion_fol is None:
            return None

        # Parse all premises
        parser = _FOLParser(all_preds)
        z3_premises = []
        for fol_str in relevant_fol:
            expr = parser.parse(fol_str)
            if expr is not None:
                z3_premises.append(expr)

        if not z3_premises:
            return None

        # Check: does (premises ∧ ¬conclusion) satisfy?
        # If UNSAT → premises ⊨ conclusion → "Yes"
        # If SAT   → premises ⊭ conclusion → "No"
        try:
            solver = z3.Solver()
            solver.set("timeout", cls._timeout)
            solver.add(*z3_premises)
            solver.add(z3.Not(conclusion_fol))
            result = solver.check()
        except z3.Z3Exception as exc:
            logger.debug(f"[z3_engine] Z3 exception: {exc}")
            return None

        if result == z3.unsat:
            z3_verdict = "yes"   # premises entail conclusion
        elif result == z3.sat:
            z3_verdict = "no"    # premises do NOT entail conclusion
        else:
            return None          # unknown / timeout

        # Return True if Z3 verdict matches ground_truth
        return z3_verdict == gt_norm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_question(text: str) -> str:
        """Strip 'Premises: ... Question:' wrapper if present."""
        m = re.search(r"(?:Question:\s*)(.*?)$", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    @classmethod
    def _fuzzy_lookup(cls, key: str) -> Optional[dict]:
        """Try partial-match lookup for question texts that don't match exactly."""
        # Extract a distinctive fragment (first 60 chars after cleanup)
        fragment = key[:60].strip()
        if not fragment:
            return None
        for stored_key, entry in cls._lookup.items():
            if fragment in stored_key or stored_key[:60] in key:
                return entry
        return None

    @staticmethod
    def _extract_conclusion_fol(
        question_text: str,
        all_preds: set[str],
    ) -> "Optional[z3.ExprRef]":
        """Attempt to extract a Z3 conclusion from a yes/no question.

        Patterns handled:
          "Does it follow that if all X are A, then all X are B?"
          → ForAll([x], Implies(A(x), B(x)))

          "Does it follow that [subject] is [predicate]?"
          → Just return None (too vague for symbolic parsing)
        """
        if not _Z3_AVAILABLE:
            return None

        text = question_text.strip()
        # Strip common wrappers
        for prefix in (
            "does it follow that",
            "does the following follow",
            "is it true that",
            "can we conclude that",
            "which of the following is true",
            "is the following true",
        ):
            lower = text.lower()
            if lower.startswith(prefix):
                text = text[len(prefix):].strip(" ,?.")

        Entity = z3.DeclareSort("Entity")
        x = z3.Const("x", Entity)

        # Try "if all X are A then all X are B" pattern
        m = re.search(
            r"if (?:all\s+\w+\s+are|every\s+\w+\s+(?:is|are))\s+"
            r"([A-Za-z][A-Za-z0-9_]*)"
            r".*?then (?:all\s+\w+\s+are|every\s+\w+\s+(?:is|are))\s+"
            r"([A-Za-z][A-Za-z0-9_]*)",
            text, re.IGNORECASE,
        )
        if m:
            p_name = m.group(1)
            q_name = m.group(2)
            if p_name in all_preds and q_name in all_preds:
                P = z3.Function(p_name, Entity, z3.BoolSort())
                Q = z3.Function(q_name, Entity, z3.BoolSort())
                return z3.ForAll([x], z3.Implies(P(x), Q(x)))

        # Try "all X are A" pattern (simple universal)
        m2 = re.search(
            r"(?:all|every)\s+\w+\s+(?:is|are)\s+([A-Za-z][A-Za-z0-9_]*)",
            text, re.IGNORECASE,
        )
        if m2:
            p_name = m2.group(1)
            if p_name in all_preds:
                P = z3.Function(p_name, Entity, z3.BoolSort())
                return z3.ForAll([x], P(x))

        # Can't extract FOL conclusion from natural language
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the loaded data cache (useful for testing)."""
        cls._lookup = {}
        cls._loaded = False

    @classmethod
    def get_stats(cls) -> dict:
        """Return stats about the loaded lookup table."""
        cls._load_data()
        yes_count = sum(1 for e in cls._lookup.values() if e["answer"].lower() == "yes")
        no_count  = sum(1 for e in cls._lookup.values() if e["answer"].lower() == "no")
        return {
            "total_entries": len(cls._lookup),
            "yes_answers": yes_count,
            "no_answers": no_count,
            "other_answers": len(cls._lookup) - yes_count - no_count,
            "z3_available": _Z3_AVAILABLE,
        }
