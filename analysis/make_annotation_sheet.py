"""Build the human annotation sheet for the cfg3 single-pass errors.

The paper reports a deterministic re-classification (error_recheck.py). This
sheet lets the authors validate it by hand. All 117 errors are included; the
column `priority` marks a stratified subset of 64 items (all small categories
in full, the larger ones sampled with a fixed seed) for a quicker pass.

Usage: python make_annotation_sheet.py <repo> <error_recheck_cfg3.csv> <out.csv>
Columns to fill: human_label (one code below) and human_notes.
  RE  reasoning error (wrong formula, wrong inference, wrong arithmetic)
  TR  truncated / unfinished response, no final answer
  EX  extraction or verifier error (answer present and correct, but scored wrong)
  LM  label or answer-space mismatch, ambiguous gold label
"""
import csv
import json
import random
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

REPO, RECHECK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
with pa.memory_map(f"{REPO}/processed_data/exact_val/data-00000-of-00001.arrow") as src:
    val = ipc.open_stream(src).read_all().to_pylist()
rows = [json.loads(l) for l in open(f"{REPO}/logs/ablation_per_sample_canonical_partB.jsonl", encoding="utf-8")]
cfg3 = [r for r in rows if str(r["config_id"]) == "3"]
labels = {int(r["idx"]): r for r in csv.DictReader(open(RECHECK, encoding="utf-8-sig"))}

by_cat = {}
for idx, r in labels.items():
    by_cat.setdefault(r["recheck_label"], []).append(idx)
quota = {"NF": 20, "L-wrong": 15, "L-mismatch": 10}
rng = random.Random(2026)
priority = set()
for cat, idxs in by_cat.items():
    k = quota.get(cat, len(idxs))
    priority.update(rng.sample(sorted(idxs), min(k, len(idxs))))

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "priority", "subject", "question", "gold_answer", "extracted_prediction",
                "auto_label", "logged_output_first_1500_chars", "human_label", "human_notes"])
    for idx in sorted(labels):
        q = [m for m in val[idx]["prompt"] if m["role"] == "user"][0]["content"]
        r = cfg3[idx]
        assert r["question"] == q[:200]
        w.writerow([idx, "yes" if idx in priority else "", r["subject"], q, r["ground_truth"],
                    r["prediction"], labels[idx]["recheck_label"], r["raw_output"], "", ""])
print("wrote", OUT, "| items:", len(labels), "| priority:", len(priority))
