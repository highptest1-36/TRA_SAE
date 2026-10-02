"""Audit of logic answer formats in the EXACT 2026 split (no GPU, no model runs).

Counts multiple-choice logic items (lettered options A., B., C., ...) and their
gold labels, and reports per-configuration accuracy on the multiple-choice items
labelled "Unknown" versus the remaining logic items (first-pass predictions).

Usage: python logic_label_audit.py <path-to-TRA_SAE-repo> <out-json>
"""
import json
import re
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO, OUT = sys.argv[1], sys.argv[2]


def load_arrow(p):
    with pa.memory_map(p) as src:
        return ipc.open_stream(src).read_all().to_pylist()


def user_content(prompt):
    return [m for m in prompt if m["role"] == "user"][0]["content"]


def is_mcq(q):
    qpart = q.split("Question:")[-1]
    return all(re.search(rf"(?m)^\s*{L}[.)]\s+\S", qpart) for L in "ABC")


res = {}
for name in ("train", "val"):
    ds = load_arrow(f"{REPO}/processed_data/exact_{name}/data-00000-of-00001.arrow")
    logic = [s for s in ds if s["type"] == "logic"]
    mcq = [s for s in logic if is_mcq(user_content(s["prompt"]))]
    res[name] = {"logic_items": len(logic), "mcq_items": len(mcq),
                 "mcq_gold_labels": dict(Counter(s["answer"] for s in mcq)),
                 "non_mcq_gold_labels": dict(Counter(s["answer"] for s in logic
                                                     if not is_mcq(user_content(s["prompt"]))))}
    if name == "val":
        val = ds

mask = np.array([s["type"] == "logic" and is_mcq(user_content(s["prompt"])) and s["answer"] == "Unknown"
                 for s in val])
logic = np.array([s["type"] == "logic" for s in val])


def first_pass(path, cfg_id):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if str(r["config_id"]) == str(cfg_id)]
    assert len(rows) == len(val)
    return np.array([bool(r["correct"]) and int(r["retry_count"]) == 0 for r in rows])


CAN = f"{REPO}/logs/ablation_per_sample_canonical.jsonl"
PB = f"{REPO}/logs/ablation_per_sample_canonical_partB.jsonl"
cfgs = {"cfg0": first_pass(CAN, 0), "cfg0-R": first_pass(PB, 6)}
for k in (1, 2, 3, 4):
    cfgs[f"cfg{k}"] = first_pass(CAN, k)

res["val_accuracy_split"] = {
    k: {"mcq_unknown_correct": int(v[mask].sum()), "mcq_unknown_n": int(mask.sum()),
        "other_logic_correct": int(v[logic & ~mask].sum()), "other_logic_n": int((logic & ~mask).sum()),
        "other_logic_acc": round(100 * v[logic & ~mask].mean(), 1)}
    for k, v in cfgs.items()}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
