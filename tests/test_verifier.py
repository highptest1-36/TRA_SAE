"""
TRA-SAE Verifier Unit Tests (v2)
==================================
Run with:
    cd /content/drive/MyDrive/TRA-SAE
    python -m pytest tests/test_verifier.py -v

or from the Colab notebook:
    !cd /content/drive/MyDrive/TRA-SAE && python -m pytest tests/test_verifier.py -v
"""
import sys
import os

# Ensure src/ is importable regardless of where pytest is invoked
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from src.symbolic_verifier import (
    parse_numerical,
    verify_numerical,
    verify_yes_no,
    verify_mcq,
    verify_answer,
    same_unit_scale,
    _strip_var_prefix,
    _parse_scientific,
    _to_si,
)


# ─────────────────────────────────────────────────────────────────────────────
# _strip_var_prefix
# ─────────────────────────────────────────────────────────────────────────────

class TestStripVarPrefix:
    def test_variable_equals(self):
        assert _strip_var_prefix("I_total = 1.5 A") == "1.5 A"

    def test_power_var(self):
        assert _strip_var_prefix("P = 9.0 W") == "9.0 W"

    def test_no_prefix(self):
        assert _strip_var_prefix("1.5 A") == "1.5 A"

    def test_long_var(self):
        assert _strip_var_prefix("V_out = 12.0 V") == "12.0 V"

    def test_no_match_leaves_unchanged(self):
        assert _strip_var_prefix("unknown text") == "unknown text"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_scientific
# ─────────────────────────────────────────────────────────────────────────────

class TestParseScientific:
    def test_unicode_times(self):
        val, unit = _parse_scientific("9.71 × 10^7 V/m")
        assert abs(val - 9.71e7) < 1e3
        assert "V" in unit

    def test_asterisk_notation(self):
        val, unit = _parse_scientific("1.5*10^-3 A")
        assert abs(val - 1.5e-3) < 1e-9

    def test_e_notation(self):
        val, unit = _parse_scientific("3.2e4 Hz")
        assert abs(val - 3.2e4) < 1

    def test_no_scientific(self):
        val, unit = _parse_scientific("9.81")
        assert val is None or abs(val - 9.81) < 1e-9

    def test_negative_exponent(self):
        val, unit = _parse_scientific("6.67 × 10^-11 N·m²/kg²")
        assert abs(val - 6.67e-11) < 1e-20


# ─────────────────────────────────────────────────────────────────────────────
# _to_si  (unit conversion)
# ─────────────────────────────────────────────────────────────────────────────

class TestToSI:
    def test_micro_joule(self):
        # 100 μJ → 100 × 1e-6 = 1e-4 J
        from src.symbolic_verifier import _to_si, UNIT_TABLE
        result = _to_si(100.0, "μj")
        assert abs(result - 1e-4) < 1e-15

    def test_kilo_ohm(self):
        from src.symbolic_verifier import _to_si
        result = _to_si(2.0, "kω")
        assert abs(result - 2000.0) < 1e-9

    def test_milli_amp(self):
        from src.symbolic_verifier import _to_si
        result = _to_si(500.0, "ma")
        assert abs(result - 0.5) < 1e-9

    def test_no_unit(self):
        from src.symbolic_verifier import _to_si
        # Unknown unit → return value unchanged
        result = _to_si(42.0, "xyz")
        assert abs(result - 42.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# parse_numerical
# ─────────────────────────────────────────────────────────────────────────────

class TestParseNumerical:
    def test_plain_int(self):
        val, unit = parse_numerical("42")
        assert val == 42.0

    def test_plain_float(self):
        val, unit = parse_numerical("3.14")
        assert abs(val - 3.14) < 1e-6

    def test_value_with_unit(self):
        val, unit = parse_numerical("9.81 m/s²")
        assert abs(val - 9.81) < 1e-6

    def test_none_for_text(self):
        val, unit = parse_numerical("yes")
        assert val is None


# ─────────────────────────────────────────────────────────────────────────────
# verify_numerical
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyNumerical:
    # Variable prefix stripping
    def test_var_prefix_current(self):
        assert verify_numerical("I_total = 1.5 A", "1.5 A") is True

    def test_var_prefix_power(self):
        assert verify_numerical("P = 9.0 W", "9 W") is True

    # SI unit conversion
    def test_micro_joule_vs_plain(self):
        assert verify_numerical("100 μJ", "0.0001") is True

    def test_micro_joule_vs_joule(self):
        assert verify_numerical("100 μJ", "1e-4 J") is True

    def test_kilo_ohm_vs_ohm(self):
        assert verify_numerical("2 kΩ", "2000") is True

    def test_kilo_ohm_vs_ohm_alt(self):
        assert verify_numerical("2 kΩ", "2000 Ω") is True

    # Scientific notation
    def test_unicode_sci_notation(self):
        assert verify_numerical("9.71 × 10^7 V/m", "9.71e7") is True

    def test_asterisk_sci_notation(self):
        assert verify_numerical("1.5*10^-3 A", "0.0015 A") is True

    # Tolerance
    def test_close_values(self):
        assert verify_numerical("9.81", "9.8") is True   # within 1% tolerance

    def test_far_values(self):
        assert verify_numerical("9.81", "100.0") is False

    # Negative
    def test_mismatch(self):
        assert verify_numerical("1.5 A", "2.5 A") is False


# ─────────────────────────────────────────────────────────────────────────────
# verify_yes_no
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyYesNo:
    def test_yes_simple(self):
        assert verify_yes_no("Yes", "Yes") is True

    def test_yes_sentence(self):
        assert verify_yes_no("The answer is yes.", "Yes") is True

    def test_no_sentence(self):
        assert verify_yes_no("The answer is no.", "No") is True

    def test_unknown(self):
        assert verify_yes_no("I believe the answer is unknown.", "Unknown") is True

    def test_case_insensitive(self):
        assert verify_yes_no("YES", "yes") is True

    def test_no_mismatch(self):
        assert verify_yes_no("Yes", "No") is False

    def test_word_boundary(self):
        # "know" contains "no" as substring but should NOT match "No"
        assert verify_yes_no("I know this is true.", "No") is False

    def test_false_synonym(self):
        assert verify_yes_no("False", "No") is True

    def test_true_synonym(self):
        assert verify_yes_no("True", "Yes") is True


# ─────────────────────────────────────────────────────────────────────────────
# verify_mcq
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyMCQ:
    def test_letter_match(self):
        assert verify_mcq("C", "C") is True

    def test_sentence_answer(self):
        assert verify_mcq("The answer is A", "A") is True

    def test_lowercase(self):
        assert verify_mcq("a", "A") is True

    def test_mismatch(self):
        assert verify_mcq("B", "A") is False

    def test_xml_wrapped(self):
        assert verify_mcq("<answer>C</answer>", "C") is True


# ─────────────────────────────────────────────────────────────────────────────
# same_unit_scale
# ─────────────────────────────────────────────────────────────────────────────

class TestSameUnitScale:
    def test_both_kilo(self):
        assert same_unit_scale("2 kΩ", "1.5 kΩ") is True

    def test_kilo_vs_base(self):
        # 2 kΩ vs 2000 Ω — different prefix scale
        assert same_unit_scale("2 kΩ", "2000 Ω") is False

    def test_both_no_unit(self):
        assert same_unit_scale("9.81", "9.81") is True


# ─────────────────────────────────────────────────────────────────────────────
# verify_answer (unified entry point)
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyAnswer:
    def test_mcq_dispatch(self):
        assert verify_answer("A", "A") is True

    def test_yes_no_dispatch(self):
        assert verify_answer("Yes", "Yes") is True
        assert verify_answer("No",  "Yes") is False

    def test_numerical_dispatch(self):
        assert verify_answer("9.81 m/s²", "9.81") is True

    def test_unit_conversion_dispatch(self):
        assert verify_answer("100 μJ", "1e-4 J") is True

    def test_var_prefix_dispatch(self):
        assert verify_answer("I = 1.5 A", "1.5 A") is True

    def test_logic_subject_yes(self):
        # Z3 disabled — should fall back to text matching
        assert verify_answer("Yes", "Yes", subject="logic", use_z3=False) is True

    def test_exact_fallback(self):
        assert verify_answer("Paris", "Paris") is True
        assert verify_answer("Paris", "London") is False


# ─────────────────────────────────────────────────────────────────────────────
# reward.py smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReward:
    def test_format_reward_full(self):
        from src.reward import format_reward
        text = "<reasoning>step by step</reasoning><answer>A</answer><explanation>because...</explanation>"
        score = format_reward(text)
        assert abs(score - 0.30) < 0.01

    def test_format_reward_partial(self):
        from src.reward import format_reward
        text = "<answer>A</answer>"
        score = format_reward(text)
        assert abs(score - 0.10) < 0.01

    def test_format_reward_empty(self):
        from src.reward import format_reward
        assert format_reward("some text no tags") == 0.0

    def test_correctness_reward_correct(self):
        from src.reward import correctness_reward
        score = correctness_reward("<answer>A</answer>", "A")
        assert abs(score - 0.60) < 0.01

    def test_correctness_reward_wrong(self):
        from src.reward import correctness_reward
        score = correctness_reward("<answer>B</answer>", "A")
        assert score == 0.0

    def test_length_penalty_not_triggered(self):
        from src.reward import length_penalty
        short = "<reasoning>" + "word " * 100 + "</reasoning>"
        assert length_penalty(short) == 0.0

    def test_length_penalty_triggered(self):
        from src.reward import length_penalty
        long_text = "<reasoning>" + "word " * 900 + "</reasoning>"
        assert length_penalty(long_text) == -0.10

    def test_compute_reward_clip(self):
        from src.reward import compute_reward
        # Wrong answer + no format + length penalty → negative
        score = compute_reward("garbage", "A")
        assert score >= -0.10
        assert score <= 1.0

    def test_compute_reward_full_credit(self):
        from src.reward import compute_reward
        text = (
            "<reasoning>The answer is A because ...</reasoning>"
            "<answer>A</answer>"
            "<explanation>A is correct</explanation>"
        )
        score = compute_reward(text, "A")
        assert score >= 0.80   # format(0.30) + correctness(0.60) = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# router.py smoke tests (no actual training needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_keyword_fallback_physics(self):
        from src.router import SubjectRouter
        router = SubjectRouter()  # not fitted → keyword fallback
        subject, conf = router.predict("Calculate the voltage across a 10 Ω resistor with 2 A current.")
        assert subject == "physics"
        assert 0.0 <= conf <= 1.0

    def test_keyword_fallback_logic(self):
        from src.router import SubjectRouter
        router = SubjectRouter()
        subject, conf = router.predict(
            "Premises: All mammals are animals. Does it follow that all dogs are animals?"
        )
        assert subject == "logic"

    def test_predict_returns_tuple(self):
        from src.router import SubjectRouter
        router = SubjectRouter()
        result = router.predict("Some question about physics circuits")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)


if __name__ == "__main__":
    # Allow running directly: python tests/test_verifier.py
    pytest.main([__file__, "-v", "--tb=short"])
