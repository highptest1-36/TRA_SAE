"""
BƯỚC 12: Rule-Based Error Re-annotation
=========================================
Tạo rule-based classifier cho error taxonomy để:
  1. Tăng reproducibility (có protocol rõ ràng)
  2. So sánh với manual annotation để tính pseudo-IAA
  3. Strengthens error analysis section trong paper

Không cần GPU — chỉ dùng per-sample predictions + ground truth.

Protocol:
  E1 (Unit/dimension error):  physics + prediction có số nhưng unit mismatch
  E2 (Wrong formula):         physics + prediction có đúng unit nhưng sai value nhiều
  E3 (Arithmetic error):      physics + prediction có đúng magnitude nhưng sai chính xác nhỏ
  E4 (Logical fallacy):       logic + model sai trên Yes/No/Unknown questions
  E5 (Misinterpretation):     prediction có format sai hoặc off-topic
  E0 (Unclassified):          không match rules nào trên

Chạy:
    python experiments/step12_error_reannotation.py
"""
from __future__ import annotations
import sys, os, json, re, math
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
def _ts(): return datetime.now().strftime("%H:%M:%S")

from src.config import LOG_DIR
from src.symbolic_verifier import (
    parse_numerical, normalize, _strip_var_prefix, UNIT_TABLE
)
from pathlib import Path

LOG_OUT        = Path(LOG_DIR) / f"error_reannotation_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / "error_reannotation_latest.json"

# ── Load error samples ────────────────────────────────────────────────────────

ERROR_FILE = Path(LOG_DIR) / "error_analysis.json"
if not ERROR_FILE.exists():
    print(f"[{_ts()}] ERROR: {ERROR_FILE} not found")
    sys.exit(1)

with open(ERROR_FILE) as f:
    error_data = json.load(f)

error_samples = error_data.get("error_samples", [])
print(f"[{_ts()}] Loaded {len(error_samples)} error samples from manual annotation")

# Also load full per-sample data to get all wrong predictions
PER_SAMPLE = Path(LOG_DIR) / "ablation_per_sample_latest.jsonl"
wrong_samples = []
if PER_SAMPLE.exists():
    with open(PER_SAMPLE) as f:
        all_samples = [json.loads(l) for l in f if l.strip()]
    cfg3_samples = [s for s in all_samples if s.get("config_id") == 3]
    wrong_samples = [s for s in cfg3_samples if not s.get("correct", True)]
    print(f"[{_ts()}] cfg3 wrong predictions: {len(wrong_samples)}")


# ── Rule-based classifier ─────────────────────────────────────────────────────

def _has_number(text: str) -> bool:
    """Check if text contains a number."""
    return bool(re.search(r'\d+\.?\d*', text))

def _has_unit(text: str) -> bool:
    """Check if text contains a known SI unit."""
    text_lower = text.lower().replace(" ", "")
    return any(unit in text_lower for unit in UNIT_TABLE if unit)

def _relative_error(pred: str, gt: str) -> float | None:
    """Compute relative error between numerical predictions."""
    p_val, _ = parse_numerical(_strip_var_prefix(pred))
    g_val, _ = parse_numerical(_strip_var_prefix(gt))
    if p_val is None or g_val is None or g_val == 0:
        return None
    return abs(p_val - g_val) / abs(g_val)

def classify_error(sample: dict) -> str:
    """
    Rule-based error classification.

    Returns: 'E0', 'E1', 'E2', 'E3', 'E4', 'E5'
    """
    pred  = str(sample.get("prediction", sample.get("pred_cfg3", ""))).strip()
    gt    = str(sample.get("ground_truth", "")).strip()
    subj  = str(sample.get("subject", "")).lower()
    q     = str(sample.get("question_snippet", sample.get("question", ""))).lower()

    # E5: format/extraction failure (empty prediction or gibberish)
    if not pred or len(pred) < 2:
        return "E5"
    if pred.lower() in ("", "none", "null", "n/a"):
        return "E5"

    # Logic domain errors → E4
    if subj == "logic":
        gt_lower = gt.lower().strip()
        if gt_lower in ("yes", "no", "unknown", "a", "b", "c", "d"):
            return "E4"
        return "E4"  # all logic errors are E4 (fallacy/quantifier)

    # Physics domain errors
    if subj == "physics":
        # Try to parse both as numbers
        rel_err = _relative_error(pred, gt)

        if rel_err is not None:
            if rel_err < 0.02:
                # Correct value but maybe unit issue (shouldn't happen in wrong set)
                return "E0"
            elif rel_err < 0.30:
                # Close but wrong — likely arithmetic error (E3)
                return "E3"
            else:
                # Far off — could be unit error (E1) or formula error (E2)
                # If pred has a unit but it's different scale → E1
                pred_lower = pred.lower()
                gt_lower   = gt.lower()
                pred_val, pred_unit = parse_numerical(_strip_var_prefix(pred))
                gt_val,   gt_unit   = parse_numerical(_strip_var_prefix(gt))

                # Check if same formula class but different unit scale → E1
                if pred_unit and gt_unit:
                    p_key = re.sub(r"[ΩΩΩΩ]", "ω", pred_unit.strip()).lower()
                    g_key = re.sub(r"[ΩΩΩΩ]", "ω", gt_unit.strip()).lower()
                    p_scale = UNIT_TABLE.get(p_key)
                    g_scale = UNIT_TABLE.get(g_key)
                    if p_scale is not None and g_scale is not None and p_scale != g_scale:
                        return "E1"  # unit mismatch → E1

                return "E2"  # large error, formula likely wrong

        else:
            # Cannot parse as number — could be qualitative answer
            # Check if GT is qualitative
            if not _has_number(gt):
                # Both qualitative — misinterpretation or format issue
                if _has_number(pred):
                    # Model gave number when GT is qualitative → E5 (misinterpretation)
                    return "E5"
                return "E0"

            # GT is numerical but pred is not parseable → E1 (unit/format issue)
            if _has_unit(pred) or any(c.isdigit() for c in pred):
                return "E1"

            return "E0"

    # Unknown subject
    return "E0"


# ── Apply rule-based classification ──────────────────────────────────────────

print(f"\n[{_ts()}] Applying rule-based classification ...")

# Use the manual error samples as ground truth for comparison
if error_samples:
    # Map manual annotation to rule-based
    manual_map = {s["etype"]: 0 for s in error_samples}
    rule_map   = {}
    agreement  = 0

    for sample in error_samples:
        manual_label = sample["etype"]
        rule_label   = classify_error({
            "prediction":   sample.get("prediction", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "subject":      sample.get("subject", "physics"),
            "question":     sample.get("question_snippet", ""),
        })

        # Normalize: E4_logical_fallacy → E4
        manual_norm = manual_label.split("_")[0] if "_" in manual_label else manual_label
        # rule_label is already "E0", "E1", etc.

        manual_map[manual_norm] = manual_map.get(manual_norm, 0) + 1
        rule_map[rule_label]    = rule_map.get(rule_label, 0) + 1

        if manual_norm == rule_label:
            agreement += 1

    iaa_pct = agreement / len(error_samples) * 100 if error_samples else 0
    print(f"[{_ts()}] Manual vs Rule-based agreement: {agreement}/{len(error_samples)} = {iaa_pct:.1f}%")
    print(f"[{_ts()}] Manual distribution: {manual_map}")
    print(f"[{_ts()}] Rule-based distribution: {rule_map}")

# Apply to all wrong predictions
if wrong_samples:
    rule_counts = {}
    annotated   = []
    for s in wrong_samples:
        label = classify_error({
            "prediction":   s.get("prediction", ""),
            "ground_truth": s.get("ground_truth", ""),
            "subject":      s.get("subject", ""),
            "question":     s.get("question", ""),
        })
        rule_counts[label] = rule_counts.get(label, 0) + 1
        annotated.append({**s, "rule_label": label})

    total_wrong = len(wrong_samples)
    print(f"\n[{_ts()}] Rule-based counts ({total_wrong} wrong predictions):")
    for label in sorted(rule_counts):
        cnt = rule_counts[label]
        print(f"  {label}: {cnt} ({cnt/total_wrong*100:.1f}%)")


# ── Combined taxonomy table ───────────────────────────────────────────────────

print(f"\n[{_ts()}] === COMBINED TAXONOMY (for paper) ===")
print(f"{'Error Type':<45} {'Manual':>8} {'Rule':>8} {'Agreement':>12}")
print("-" * 80)

error_defs = {
    "E0": "Unclassified / incomplete output",
    "E1": "Unit or dimension error (physics)",
    "E2": "Wrong formula / law selected (physics)",
    "E3": "Arithmetic computation error (physics)",
    "E4": "Logical fallacy / quantifier error (logic)",
    "E5": "Question misinterpretation",
}

manual_counts_orig = {
    "E1": 26, "E2": 0, "E3": 1, "E4": 57, "E5": 1, "E0": 18
}

for eid, edef in error_defs.items():
    m_cnt = manual_counts_orig.get(eid, 0)
    r_cnt = rule_counts.get(eid, 0) if wrong_samples else 0
    delta = r_cnt - m_cnt
    print(f"  {eid} {edef:<40} {m_cnt:>7} {r_cnt:>7} {delta:>+10}")

print(f"\n[{_ts()}] IAA (manual vs rule-based): {iaa_pct:.1f}%")


# ── Save ──────────────────────────────────────────────────────────────────────

out = {
    "run_ts":         _RUN_TS,
    "n_error_samples": len(error_samples),
    "n_wrong_total":   len(wrong_samples),
    "manual_counts":   manual_counts_orig,
    "rule_counts":     rule_counts if wrong_samples else {},
    "iaa_pct":         round(iaa_pct, 1),
    "agreement_n":     agreement if error_samples else 0,
    "annotation_protocol": {
        "E0": "Cannot determine root cause from output alone",
        "E1": "Physics: predicted unit incompatible with GT unit (SI scale mismatch)",
        "E2": "Physics: large relative error (>30%) with correct unit class → wrong formula",
        "E3": "Physics: small relative error (2-30%) → arithmetic mistake in correct formula",
        "E4": "Logic: any incorrect Yes/No/Unknown or MCQ answer → quantifier/fallacy error",
        "E5": "Any domain: prediction is empty, off-topic, or format failure",
    },
    "annotated_samples": annotated[:103] if wrong_samples else [],
}

with open(LOG_OUT, "w") as f: json.dump(out, f, indent=2)
with open(LOG_OUT_LATEST, "w") as f: json.dump(out, f, indent=2)
print(f"\n[{_ts()}] Saved → {LOG_OUT_LATEST}")

print(f"\n[{_ts()}] === PAPER UPDATE ===")
print(f"[{_ts()}] Add to error analysis section:")
print(f"  Rule-based annotation (using SI unit table + subject labels)")
print(f"  applied to all {len(wrong_samples)} wrong predictions.")
print(f"  Agreement with manual annotation: {iaa_pct:.1f}%")
if iaa_pct >= 75:
    print(f"  → High agreement supports reliability of manual taxonomy")
else:
    print(f"  → Use rule-based as supplementary check; note differences")
