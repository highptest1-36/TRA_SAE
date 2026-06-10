"""
BƯỚC 14: Manual Error Annotation (reviewer "missing #3" + error analysis)
=========================================================================
Tạo template cho việc gán nhãn lỗi THỦ CÔNG 50 mẫu sai của best config, theo
taxonomy 6 nhóm phong phú hơn (gồm nhóm 'answer_extraction' tách riêng cho
trường hợp "reasoning đúng nhưng trích xuất đáp án sai").

Quy trình 2 bước:
  1) Generate template (chạy lần đầu):
        python experiments/step14_manual_error_annotation.py
     → tạo logs/manual_error_template.csv (cột manual_label để trống cho bạn điền)

  2) Sau khi điền cột manual_label trong CSV và lưu thành
     logs/manual_error_filled.csv, chạy lại để tính agreement vs rule-based:
        python experiments/step14_manual_error_annotation.py --score

Taxonomy (điền đúng 1 mã vào cột manual_label):
  FS  = formula_selection      (chọn sai công thức/luật)
  UC  = unit_conversion        (sai đổi đơn vị / scale)
  AR  = arithmetic             (công thức đúng, tính sai)
  PU  = problem_understanding  (hiểu sai đề / chọn sai biến)
  LE  = logic_entailment       (sai phủ định/lượng từ/kéo theo)
  AX  = answer_extraction      (reasoning đúng, đáp án sai format/không trích được)
"""
from __future__ import annotations
import sys, json, csv, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
from pathlib import Path
from collections import Counter
from src.config import LOG_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, default=-1, help="config id to analyse (-1=auto best)")
parser.add_argument("--n", type=int, default=50, help="how many errors to sample")
parser.add_argument("--score", action="store_true", help="compute agreement from filled CSV")
args, _ = parser.parse_known_args()

LP = Path(LOG_DIR)
TEMPLATE = LP / "manual_error_template.csv"
FILLED = LP / "manual_error_filled.csv"
TAXONOMY = {"FS": "formula_selection", "UC": "unit_conversion", "AR": "arithmetic",
            "PU": "problem_understanding", "LE": "logic_entailment", "AX": "answer_extraction"}

import re


def _rule_label(q, pred, gt, subj, raw):
    """Self-contained rule-based suggestion (mirrors step4 taxonomy, mapped to
    the 6-class manual codes). Kept inline to avoid importing step4 (which runs
    its full pipeline on import)."""
    q_low = (q or "").lower()
    pred_low = (pred or "").lower()
    gt_low = (gt or "").lower()
    raw_low = (raw or "").lower()
    # AX: reasoning present but no answer tag → extraction failure
    if raw_low and "<reasoning>" in raw_low and "<answer>" not in raw_low:
        return "AX"
    if subj == "logic":
        if any(k in q_low for k in ["forall", "∀", "exists", "∃", "premise",
                                    "conclusion", "therefore", "implies",
                                    "contrapositive", "syllogism"]):
            return "LE"
    if subj == "physics":
        unit_re = r'\b(kg|m/s|j|n|w|pa|k|°c|v|a|ω)\b'
        if re.search(unit_re, gt_low) and not re.search(unit_re, pred_low):
            return "UC"
        if any(k in pred_low for k in ["f=ma", "e=mc", "v=ir", "p=iv", "e=qv",
                                       "kinetic", "potential", "newton", "ohm",
                                       "coulomb", "faraday"]):
            return "FS"
        ng = re.findall(r'\d+\.?\d*', gt_low)
        npd = re.findall(r'\d+\.?\d*', pred_low)
        if ng and npd:
            try:
                d = abs(float(npd[0]) - float(ng[0]))
                if 0 < d < float(ng[0]) * 0.5:
                    return "AR"
            except (ValueError, ZeroDivisionError):
                pass
    if len((pred or "").strip()) < 5:
        return "AX"
    return "PU"


def _load_rows():
    p = LP / "ablation_per_sample_canonical.jsonl"
    if not p.exists():
        p = LP / "ablation_per_sample_latest.jsonl"
    if not p.exists():
        sys.exit("Missing per-sample JSONL — run step0_canonical_eval.py first.")
    return [json.loads(l) for l in open(p) if l.strip()]


if args.score:
    if not FILLED.exists():
        sys.exit(f"Fill {TEMPLATE} and save as {FILLED} first.")
    rows = list(csv.DictReader(open(FILLED)))
    agree = total = 0
    manual_counts, rule_counts = Counter(), Counter()
    for r in rows:
        m = (r.get("manual_label") or "").strip().upper()
        rb = (r.get("rule_label") or "").strip().upper()
        if m not in TAXONOMY:
            continue
        total += 1
        manual_counts[m] += 1
        rule_counts[rb] += 1
        agree += int(m == rb)
    if total == 0:
        sys.exit("No valid manual_label values found (use codes: FS UC AR PU LE AX).")
    pct = round(agree / total * 100, 1)
    # Cohen's kappa
    labels = list(TAXONOMY)
    po = agree / total
    pe = sum((manual_counts[l] / total) * (rule_counts[l] / total) for l in labels)
    kappa = round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0
    out = {"n_annotated": total, "agreement_pct": pct, "cohens_kappa": kappa,
           "manual_distribution": dict(manual_counts), "rule_distribution": dict(rule_counts)}
    json.dump(out, open(LP / "manual_error_agreement.json", "w"), indent=2)
    print(f"Annotated: {total}  Agreement: {pct}%  Cohen's κ: {kappa}")
    print(f"Manual dist: {dict(manual_counts)}")
    print(f"Saved → {LP/'manual_error_agreement.json'}")
    sys.exit(0)

# ── Generate template ─────────────────────────────────────────────────────────
rows = _load_rows()
accs = {}
by_cfg = {}
for r in rows:
    by_cfg.setdefault(r["config_id"], []).append(r)
for cid, rs in by_cfg.items():
    accs[cid] = sum(x["correct"] for x in rs) / len(rs)
best = args.config if args.config >= 0 else max(
    (c for c in accs if isinstance(c, int) and c >= 1), key=lambda c: accs[c])
errors = [r for r in by_cfg[best] if not r["correct"]]
errors.sort(key=lambda r: r["question"])           # deterministic order
sample = errors[:args.n]

with open(TEMPLATE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "subject", "question", "prediction", "ground_truth",
                "rule_label", "manual_label", "notes", "raw_snippet"])
    for i, r in enumerate(sample):
        rl = _rule_label(r["question"], r["prediction"], r["ground_truth"],
                         r["subject"], r.get("raw_output", ""))
        w.writerow([i, r["subject"], r["question"][:200], r["prediction"][:80],
                    r["ground_truth"][:60], rl, "", "", (r.get("raw_output", "") or "")[:400]])

print(f"Best config = cfg{best} (acc={accs[best]*100:.2f}%), {len(errors)} errors total.")
print(f"Wrote {len(sample)} rows → {TEMPLATE}")
print("Fill the 'manual_label' column (FS/UC/AR/PU/LE/AX), save as "
      f"{FILLED.name}, then re-run with --score.")
print("Taxonomy:", {k: v for k, v in TAXONOMY.items()})
