"""Deterministic re-check of the cfg3 single-pass error taxonomy.

Why: experiments/_partB_error_classify.py assigns E6 ("extraction / incomplete
output") whenever "<answer>" is missing from `raw_output`. In the per-sample log
that field is truncated to 1,500 characters (step0_canonical_eval.py, line 363),
so any long response whose answer tag appears after character 1,500 is counted
as E6 even if it ends with a well-formed answer.

The `prediction` field, however, was extracted from the FULL response
(answer tag -> \\boxed{} -> last non-empty line). This script therefore
re-classifies every incorrect prediction from the form of that extracted
string, with fixed, documented rules (no model, no human judgement):

  NF  no final answer: no <answer>...</answer> tag is visible in the stored
      output AND the extracted string is a reasoning fragment
      (markdown bullet, LaTeX fragment, unfinished sentence, closing tag),
      i.e. the extractor fell back to the last line of a response that
      stopped before a final answer.
  WF  well-formed final answer that is scored wrong, split into
      L-mismatch  logic item, letter option (A-D) given for a Yes/No/Unknown gold label
      L-wrong     logic item, wrong Yes/No/Unknown or wrong option letter
      P-numeric   physics item, numerical answer outside the 2% tolerance
      P-notation  physics item, numerically equal to the gold after
                  normalising "a . 10^k" notation (verifier false negative)
      P-other     physics item, well-formed non-numerical answer
                  (e.g. categorical or symbolic gold label)

Usage: python error_recheck.py <repo> <out-json> <out-csv>
"""
import csv
import json
import re
import sys
from collections import Counter

REPO, OUT_JSON, OUT_CSV = sys.argv[1], sys.argv[2], sys.argv[3]

rows = [json.loads(l) for l in open(f"{REPO}/logs/ablation_per_sample_canonical_partB.jsonl", encoding="utf-8")]
cfg3 = [r for r in rows if str(r["config_id"]) == "3"]
assert len(cfg3) == 217

LETTER = re.compile(r"^\(?([A-Da-d])\)?[.):]?$")
YNU = {"yes", "no", "unknown", "true", "false", "uncertain"}
NUM = re.compile(
    r"^[-+]?\d+(?:[.,]\d+)?\s*(?:(?:[x×*·.]|\\times)\s*10\s*\^?\s*\{?[-−⁻]?\d+\}?|e[-+]?\d+)?"
    r"\s*[A-Za-zΩμµ°/·²³⁻⁰¹⁴⁵⁶⁷⁸⁹\-^0-9 ]{0,12}$")
FRAGMENT_START = ("*", "-", "$", "\\", "</", "<", "let", "so ", "the ", "this ", "but ", "from ",
                  "option", "therefore", "is", "f_", "e_", "i_", "note", "example", "\"")

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def to_float(s):
    s = s.translate(SUP).replace("−", "-").replace(",", "")
    fr = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if fr:
        return float(fr.group(1)) / float(fr.group(2))
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:(?:[x×*·.]|\\times)\s*10\s*\^?\s*\{?([-+]?\d+)\}?|e([-+]?\d+))?", s)
    if not m:
        return None
    v = float(m.group(1))
    exp = m.group(2) or m.group(3)
    return v * (10 ** int(exp)) if exp else v


def well_formed(pred, subj):
    p = pred.strip()
    if not p or len(p) > 40:
        return False
    low = p.lower().rstrip(".")
    if LETTER.match(p) or low in YNU:
        return True
    if subj == "physics" and NUM.match(p) and not p.endswith("="):
        return True
    if subj == "physics" and not low.startswith(FRAGMENT_START) and not re.search(r"[$\\{}=]|^\*", p) \
            and len(p.split()) <= 6 and not p.endswith((":", ",", "(")):
        return True  # short verbal answer such as "Quadrupled" or "upward parabola"
    return False


out, table = [], Counter()
for i, r in enumerate(cfg3):
    if r["correct"]:
        continue
    pred, gt, subj = (r["prediction"] or "").strip(), (r["ground_truth"] or "").strip(), r["subject"]
    old = "E6" if ("<reasoning>" in r["raw_output"].lower() and "<answer>" not in r["raw_output"].lower()) else "other"
    tag_visible = "<answer>" in r["raw_output"].lower() and "</answer>" in r["raw_output"].lower()
    if not (tag_visible or well_formed(pred, subj)):
        lab = "NF"
    elif subj == "logic":
        gl = gt.lower().rstrip(".")
        lab = "L-mismatch" if (LETTER.match(pred) and gl in YNU) else "L-wrong"
    else:
        pv, gv = to_float(pred), to_float(gt)
        if pv is not None and gv is not None:
            # notation false negative: equal values once "a . 10^k" is read as a x 10^k
            lab = "P-notation" if gv != 0 and abs(pv - gv) / abs(gv) <= 0.02 else "P-numeric"
        else:
            lab = "P-other"
    table[(old, lab)] += 1
    out.append({"idx": i, "subject": subj, "prediction": pred, "ground_truth": gt,
                "old_rule_label": old, "recheck_label": lab,
                "stored_output_truncated": len(r["raw_output"]) >= 1500})

summary = Counter(o["recheck_label"] for o in out)
res = {"n_errors": len(out), "recheck_counts": dict(summary),
       "old_E6_breakdown": {k[1]: v for k, v in table.items() if k[0] == "E6"},
       "old_other_breakdown": {k[1]: v for k, v in table.items() if k[0] == "other"},
       "n_outputs_truncated_in_log": sum(len(r["raw_output"]) >= 1500 for r in cfg3)}
json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(json.dumps(res, indent=2, ensure_ascii=False))
for o in out:
    print(o["idx"], o["subject"][0], o["old_rule_label"], o["recheck_label"], "|", o["prediction"][:50], "|", o["ground_truth"][:30])
