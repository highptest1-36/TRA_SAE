"""Part B2 — reclassify cfg3 SINGLE-PASS wrong predictions with the SAME
rule-based taxonomy used in step4_stats_and_errors.py (E1..E6, E0).
Reads logs/ablation_per_sample_canonical_partB.jsonl (cfg3 single-pass)."""
import json, re
from pathlib import Path

ERROR_TYPES = {
    "E1_unit_dimension": "Unit or dimension error (physics)",
    "E2_wrong_formula":  "Wrong formula selection (physics)",
    "E3_arithmetic":     "Arithmetic error (physics)",
    "E4_logical_fallacy":"Logical fallacy or quantifier error (logic)",
    "E5_misinterpretation":"Question misinterpretation",
    "E6_extraction":     "Answer-extraction / incomplete output",
    "E0_unclassified":   "Unclassified",
}

def _classify_error(question, prediction, ground_truth, subject, raw_output=""):
    q_low = question.lower(); pred_low = (prediction or "").lower(); gt_low = (ground_truth or "").lower()
    if raw_output and "<reasoning>" in raw_output.lower() and "<answer>" not in raw_output.lower():
        return "E6_extraction"
    if subject == "logic":
        kw = ["forall","∀","exists","∃","premise","conclusion","therefore","implies","contrapositive","syllogism"]
        if any(k in q_low for k in kw):
            return "E4_logical_fallacy"
    if subject == "physics":
        has_gt = bool(re.search(r'\b(kg|m/s|j|n|w|pa|k|°c|v|a|ω)\b', gt_low))
        has_pr = bool(re.search(r'\b(kg|m/s|j|n|w|pa|k|°c|v|a|ω)\b', pred_low))
        if has_gt and not has_pr:
            return "E1_unit_dimension"
        fkw = ["f=ma","e=mc","v=ir","p=iv","e=qv","kinetic","potential","newton","ohm","coulomb","faraday"]
        if any(k in pred_low for k in fkw):
            return "E2_wrong_formula"
        ng = re.findall(r'\d+\.?\d*', gt_low); npr = re.findall(r'\d+\.?\d*', pred_low)
        if ng and npr:
            try:
                d = abs(float(npr[0]) - float(ng[0]))
                if 0 < d < float(ng[0]) * 0.5:
                    return "E3_arithmetic"
            except (ValueError, ZeroDivisionError):
                pass
    if len((prediction or "").strip()) < 5:
        return "E5_misinterpretation"
    return "E0_unclassified"

PS = Path("logs/ablation_per_sample_canonical_partB.jsonl")
rows = [json.loads(l) for l in open(PS) if l.strip()]
cfg3 = [r for r in rows if int(r.get("config_id", -1)) == 3]
wrong = [r for r in cfg3 if not r.get("correct")]
total = len(wrong)
counts = {k: 0 for k in ERROR_TYPES}
for r in wrong:
    et = _classify_error(r.get("question",""), r.get("prediction",""),
                         r.get("ground_truth",""), r.get("subject",""), r.get("raw_output",""))
    counts[et] += 1
print(f"cfg3 SINGLE-PASS: n_cfg3={len(cfg3)}  wrong={total}")
out = {"source": "cfg3_single_pass_partB", "total_evaluated": len(cfg3), "total_wrong": total,
       "error_counts": counts,
       "error_pct": {k: round(v/total*100, 1) if total else 0 for k, v in counts.items()}}
for k in sorted(counts, key=lambda x: -counts[x]):
    if counts[k]:
        print(f"  {k}: {counts[k]} ({100*counts[k]/total:.1f}%)")
json.dump(out, open("logs/error_analysis_single_pass.json", "w"), indent=2)
print("saved logs/error_analysis_single_pass.json")
