"""Verify the manuscript numbers from the per-sample logs and run the
duplicate-exclusion sensitivity analysis (no GPU, no new model runs).

Usage: python verify_and_sensitivity.py <path-to-TRA_SAE-repo> <out-json>
"""
import json
import math
import re
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.stats import binom, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO = sys.argv[1]
OUT = sys.argv[2]


def load_arrow(p):
    with pa.memory_map(p) as src:
        try:
            t = ipc.open_stream(src).read_all()
        except Exception:
            t = ipc.open_file(src).read_all()
    return t.to_pylist()


def user_content(prompt):
    for m in prompt:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def char_ngrams(t, n=8):
    s = re.sub(r"\s+", " ", (t or "").strip().lower())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jacc(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_cc(a, b):
    """b = B correct & A wrong, c = A correct & B wrong (A -> B)."""
    bb = int(np.sum(~a & b))
    cc = int(np.sum(a & ~b))
    if bb + cc == 0:
        return bb, cc, 1.0
    stat = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return bb, cc, float(chi2.sf(stat, 1))


def mcnemar_exact(a, b):
    bb = int(np.sum(~a & b))
    cc = int(np.sum(a & ~b))
    n = bb + cc
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(bb, cc), n, 0.5)))


train = load_arrow(f"{REPO}/processed_data/exact_train/data-00000-of-00001.arrow")
val = load_arrow(f"{REPO}/processed_data/exact_val/data-00000-of-00001.arrow")
tr_q = [user_content(s["prompt"]) for s in train]
va_q = [user_content(s["prompt"]) for s in val]
tr_a = [s["answer"] for s in train]
va_a = [s["answer"] for s in val]
va_t = [s["type"] for s in val]
N = len(val)
subj = np.array(va_t)

# ---- duplicate sets (same definitions as experiments/step10_leakage_check.py)
tr_index = {}
for i, q in enumerate(tr_q):
    tr_index.setdefault(norm(q), []).append(i)
exact = [j for j, q in enumerate(va_q) if norm(q) in tr_index]
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
Xtr = vec.fit_transform(tr_q)
sims = cosine_similarity(vec.transform(va_q), Xtr)
max_sim = sims.max(axis=1)
argmax = sims.argmax(axis=1)
char_j = np.array([jacc(char_ngrams(va_q[j]), char_ngrams(tr_q[int(argmax[j])])) for j in range(N)])
near = [j for j in range(N) if char_j[j] >= 0.95]
near_same = [j for j in near if norm(va_a[j]) == norm(tr_a[int(argmax[j])])]
hi90 = [j for j in range(N) if max_sim[j] >= 0.90]
hi95 = [j for j in range(N) if max_sim[j] >= 0.95]
print("exact", exact, "near", near, "near_same", near_same, "hi90", len(hi90))

# ---- per-sample first-pass correctness, aligned to the validation order
def first_pass(path, cfg_id, retries_run=True):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if str(r["config_id"]) == str(cfg_id)]
    assert len(rows) == N, (path, cfg_id, len(rows))
    vec_ = np.zeros(N, dtype=bool)
    for j, r in enumerate(rows):
        assert r["question"] == va_q[j][:200], (cfg_id, j)
        ok = bool(r["correct"]) if not isinstance(r["correct"], str) else r["correct"] == "True"
        vec_[j] = ok and int(r["retry_count"]) == 0
    return vec_


CAN = f"{REPO}/logs/ablation_per_sample_canonical.jsonl"
PB = f"{REPO}/logs/ablation_per_sample_canonical_partB.jsonl"
cfg = {
    "cfg0": first_pass(CAN, 0),
    "cfg0-R": first_pass(PB, 6),
    "cfg1": first_pass(CAN, 1),
    "cfg2": first_pass(CAN, 2),
    "cfg3": first_pass(CAN, 3),
    "cfg4": first_pass(CAN, 4),
}

# baselines: best-of-three strategy per model (as in the paper)
fb = json.load(open(f"{REPO}/logs/fair_baselines_results_latest.json", encoding="utf-8"))
names = {"B5": "Qwen2-Math-7B-Instruct", "B2": "Qwen2.5-Math-7B-Instruct", "B4": "Mistral-7B-Instruct-v0.3",
         "B6": "DeepSeek-Math-7B-Instruct", "B3": "Llemma-7B"}
best = {}
for r in fb["results"]:
    if r["id"] not in best or r["accuracy_overall"] > best[r["id"]]["accuracy_overall"]:
        best[r["id"]] = r
base = {}
for bid, r in best.items():
    v = np.zeros(N, dtype=bool)
    for s in r["per_sample"]:
        v[int(s["question_id"])] = bool(s["correct"])
    base[names[bid]] = v

report = {"n": N, "dup_sets": {"exact": exact, "near_char_ge_0.95": near,
                               "near_same_answer": near_same, "tfidf_ge_0.90_count": len(hi90), "tfidf_ge_0.95_count": len(hi95)}}

# ---- (1) verification of Table 4 / Table 6
ver = {}
for k, v in {**cfg, **base}.items():
    ph = v[subj == "physics"]
    lo = v[subj == "logic"]
    ver[k] = {"overall": round(100 * v.mean(), 2), "physics": round(100 * ph.mean(), 2),
              "logic": round(100 * lo.mean(), 2), "k": int(v.sum()),
              "wilson95": [round(x, 1) for x in wilson(int(v.sum()), N)]}
pairs = [("cfg0", "cfg0-R"), ("cfg0-R", "cfg1"), ("cfg0", "cfg3"), ("cfg1", "cfg2"), ("cfg2", "cfg3"),
         ("cfg3", "cfg4"), ("cfg0", "cfg1")]
mc = {}
for a, b in pairs:
    bb, cc, p = mcnemar_cc(cfg[a], cfg[b])
    mc[f"{a}->{b}"] = {"delta": round(100 * (cfg[b].mean() - cfg[a].mean()), 2), "b": bb, "c": cc,
                       "p_cc": p, "p_exact": mcnemar_exact(cfg[a], cfg[b])}
report["verify"] = {"accuracy": ver, "mcnemar": mc}

# ---- (2) sensitivity: exclude duplicate subsets
def subset_stats(mask_keep, label):
    n = int(mask_keep.sum())
    out = {"n": n, "acc": {}, "mcnemar": {}}
    for k, v in {**cfg, **base}.items():
        out["acc"][k] = round(100 * v[mask_keep].mean(), 2)
    for a, b in pairs[:5]:
        bb, cc, p = mcnemar_cc(cfg[a][mask_keep], cfg[b][mask_keep])
        out["mcnemar"][f"{a}->{b}"] = {"b": bb, "c": cc, "p_cc": p}
    order = sorted(cfg, key=lambda k: -out["acc"][k])
    out["ordering"] = order
    return out


keep_all = np.ones(N, dtype=bool)
sens = {"full": subset_stats(keep_all, "full")}
for lab, idx in [("minus_exact3", exact), ("minus_near6", near), ("minus_tfidf_ge0.95", hi95), ("minus_tfidf_ge0.90", hi90)]:
    m = np.ones(N, dtype=bool)
    m[idx] = False
    sens[lab] = subset_stats(m, lab)
report["sensitivity"] = sens

# correctness of each config on the duplicate items themselves
report["dup_item_correct"] = {k: {"exact": [bool(v[j]) for j in exact], "near": [bool(v[j]) for j in near]}
                              for k, v in cfg.items()}

json.dump(report, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps(report["verify"], indent=1))
for lab, s in sens.items():
    print(lab, s["n"], s["acc"], s["ordering"])
    print("   ", {k: (v["b"], v["c"], round(v["p_cc"], 4)) for k, v in s["mcnemar"].items()})
