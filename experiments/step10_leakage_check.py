#!/usr/bin/env python3
"""
step10_leakage_check.py -- Train/validation contamination & answer-leakage check.

NO GPU REQUIRED (pure TF-IDF / sklearn + set-based n-gram Jaccard).

Purpose
-------
The paper reports results on a 90/10 stratified split of the released EXACT
training data (1945 train / 217 val). A reviewer correctly noted that no
explicit train<->val leakage check is documented. This script supplies it:

  (i)  TEXT OVERLAP train<->val
       - exact (normalised) duplicate question detection
       - per-val maximum TF-IDF cosine vs the full training set
       - word 3-gram Jaccard against the nearest TF-IDF neighbours
       - near-duplicate-AND-same-answer count (the genuinely worrying case)

  (ii) RETRIEVAL EXEMPLAR LEAKAGE (cfg0-R protocol)
       - for each val item, retrieve the same top-3 exemplars cfg0-R uses and
         check whether any retrieved exemplar is a verbatim copy of the val
         question (which would hand the model the answer).

Outputs
-------
  logs/leakage_check_results.json   (machine-readable)
  stdout summary + top offending pairs + a LaTeX-ready mini table.

Usage
-----
  python3 experiments/step10_leakage_check.py
"""
from __future__ import annotations

import os
import re
import sys
import json
from collections import Counter

import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "processed_data", "exact_train")
VAL = os.path.join(ROOT, "processed_data", "exact_val")
OUT = os.path.join(ROOT, "logs", "leakage_check_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def user_content(prompt) -> str:
    """Return the 'user' turn from a chat-formatted prompt list."""
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def norm(t: str) -> str:
    """Whitespace-collapsed, lower-cased form for exact-match comparison."""
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def word_ngrams(t: str, n: int = 3) -> set:
    toks = re.findall(r"\w+", (t or "").lower())
    if n == 1:
        return set(toks)
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def char_ngrams(t: str, n: int = 8) -> set:
    """Character n-grams over whitespace-normalised text.

    Unlike word n-grams / TF-IDF (which drop short numeric tokens such as the
    exponent in '3e-7'), character n-grams preserve digits, so they separate a
    genuine near-duplicate from the same problem template re-instantiated with
    different numerical values.
    """
    s = re.sub(r"\s+", " ", (t or "").strip().lower())
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def pct(arr, p) -> float:
    return float(np.percentile(arr, p)) if len(arr) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[leakage] loading datasets from {TRAIN} and {VAL}")
    train = load_from_disk(TRAIN)
    val = load_from_disk(VAL)

    tr_q = [user_content(s["prompt"]) for s in train]
    va_q = [user_content(s["prompt"]) for s in val]
    tr_a = [s.get("answer", "") for s in train]
    va_a = [s.get("answer", "") for s in val]
    tr_t = [s.get("type", "unknown") for s in train]
    va_t = [s.get("type", "unknown") for s in val]
    n_tr, n_va = len(tr_q), len(va_q)
    print(f"[leakage] train={n_tr}  val={n_va}  val-subjects={dict(Counter(va_t))}")

    # ---- (i.a) exact (normalised) duplicate questions ----
    tr_norm_index: dict[str, list[int]] = {}
    for i, q in enumerate(tr_q):
        tr_norm_index.setdefault(norm(q), []).append(i)
    exact_dups = [j for j, q in enumerate(va_q) if norm(q) in tr_norm_index]
    exact_dup_same_answer, exact_dup_diff_answer = 0, 0
    for j in exact_dups:
        train_answers = {norm(tr_a[t]) for t in tr_norm_index[norm(va_q[j])]}
        if norm(va_a[j]) in train_answers:
            exact_dup_same_answer += 1
        else:
            exact_dup_diff_answer += 1

    # ---- (i.b) TF-IDF cosine: each val vs ALL train (no subject filter = worst case) ----
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform(tr_q)
    Xva = vec.transform(va_q)
    sims = cosine_similarity(Xva, Xtr)          # (n_va, n_tr)
    max_sim = sims.max(axis=1)
    argmax = sims.argmax(axis=1)

    # ---- (i.c) word 3-gram Jaccard against the 10 nearest TF-IDF neighbours ----
    top10 = np.argsort(-sims, axis=1)[:, :10]
    va_sets = [word_ngrams(q, 3) for q in va_q]
    tr_set_cache: dict[int, set] = {}
    max_jac = np.zeros(n_va)
    jac_arg = np.zeros(n_va, dtype=int)
    for j in range(n_va):
        best, bi = 0.0, -1
        for ti in top10[j]:
            ti = int(ti)
            if ti not in tr_set_cache:
                tr_set_cache[ti] = word_ngrams(tr_q[ti], 3)
            v = jaccard(va_sets[j], tr_set_cache[ti])
            if v > best:
                best, bi = v, ti
        max_jac[j] = best
        jac_arg[j] = bi

    # ---- (i.d) char 8-gram Jaccard vs the nearest TF-IDF neighbour ----
    # Separates TRUE near-duplicates (digits preserved) from template collisions
    # (same wording, different numbers -> different answer).
    char_jac = np.zeros(n_va)
    for j in range(n_va):
        char_jac[j] = jaccard(char_ngrams(va_q[j], 8),
                              char_ngrams(tr_q[int(argmax[j])], 8))

    # ---- (i.e) classify the high-cosine (>=0.95) pairs ----
    hi = [j for j in range(n_va) if max_sim[j] >= 0.95]
    same_answer = lambda j: norm(va_a[j]) == norm(tr_a[int(argmax[j])])
    true_near_identical = [j for j in range(n_va) if char_jac[j] >= 0.95]
    template_collisions = [j for j in hi if char_jac[j] < 0.95]
    # the only genuinely trivialising case: near-identical text AND same answer
    real_leakage = [j for j in true_near_identical if same_answer(j)]
    nd_same_ans = [j for j in hi if same_answer(j)]

    # ---- (ii) retrieval exemplar leakage (cfg0-R protocol) ----
    retr_result: dict = {"available": False}
    try:
        sys.path.insert(0, ROOT)
        from src.retriever import Retriever
        r = Retriever(train_path=TRAIN)
        r.build(cache_path=None)   # in-memory rebuild; avoids stale-cache load bug
        top1_scores, verbatim = [], 0
        for j in range(n_va):
            subj = va_t[j] if va_t[j] in ("physics", "logic") else None
            ex = r.retrieve(va_q[j], top_k=3, subject=subj)
            best = ex[0]["score"] if ex else 0.0
            top1_scores.append(best)
            if ex and best >= 0.95 and norm(ex[0]["question"]) == norm(va_q[j]):
                verbatim += 1
        retr_result = {
            "available": True,
            "protocol": "cfg0-R top-3, subject-filtered",
            "top1_score_mean": float(np.mean(top1_scores)),
            "top1_score_p90": pct(top1_scores, 90),
            "top1_score_p99": pct(top1_scores, 99),
            "top1_score_max": float(np.max(top1_scores)),
            "verbatim_exemplar_count": int(verbatim),
        }
    except Exception as e:  # retriever optional; (i) is the primary check
        retr_result = {"available": False, "error": repr(e)}

    results = {
        "n_train": n_tr,
        "n_val": n_va,
        "val_subject_dist": dict(Counter(va_t)),
        "exact_duplicate_questions": len(exact_dups),
        "exact_duplicate_same_answer": exact_dup_same_answer,
        "exact_duplicate_diff_answer": exact_dup_diff_answer,
        "tfidf_cosine_question": {
            "mean_of_per_val_max": float(np.mean(max_sim)),
            "p50": pct(max_sim, 50),
            "p90": pct(max_sim, 90),
            "p95": pct(max_sim, 95),
            "p99": pct(max_sim, 99),
            "max": float(np.max(max_sim)),
            "count_ge_0.90": int((max_sim >= 0.90).sum()),
            "count_ge_0.95": int((max_sim >= 0.95).sum()),
            "count_ge_0.99": int((max_sim >= 0.99).sum()),
        },
        "word3gram_jaccard": {
            "mean_of_per_val_max": float(np.mean(max_jac)),
            "p90": pct(max_jac, 90),
            "p99": pct(max_jac, 99),
            "max": float(np.max(max_jac)),
            "count_ge_0.80": int((max_jac >= 0.80).sum()),
        },
        "char8gram_jaccard_top1": {
            "mean": float(np.mean(char_jac)),
            "p90": pct(char_jac, 90),
            "p99": pct(char_jac, 99),
            "max": float(np.max(char_jac)),
        },
        "high_cosine_pairs_ge_0.95": len(hi),
        "true_near_identical_char_ge_0.95": len(true_near_identical),
        "template_collisions_cos_ge_0.95_char_lt_0.95": len(template_collisions),
        "near_duplicate_same_answer_count": len(nd_same_ans),
        "real_leakage_near_identical_and_same_answer": len(real_leakage),
        "real_leakage_fraction_pct": round(100.0 * len(real_leakage) / n_va, 2),
        "retriever_exemplar_check": retr_result,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # ---- human-readable summary ----
    print("\n" + "=" * 64)
    print("LEAKAGE CHECK SUMMARY")
    print("=" * 64)
    print(f"  exact duplicate questions (val in train) : {len(exact_dups)}"
          f"  (same-answer={exact_dup_same_answer}, diff-answer={exact_dup_diff_answer})")
    tc = results["tfidf_cosine_question"]
    print(f"  TF-IDF cosine (per-val max vs full train):")
    print(f"     mean={tc['mean_of_per_val_max']:.3f}  p50={tc['p50']:.3f}  "
          f"p90={tc['p90']:.3f}  p99={tc['p99']:.3f}  max={tc['max']:.3f}")
    print(f"     #val >=0.90: {tc['count_ge_0.90']}   "
          f">=0.95: {tc['count_ge_0.95']}   >=0.99: {tc['count_ge_0.99']}")
    jc = results["word3gram_jaccard"]
    print(f"  word-3gram Jaccard (per-val max): mean={jc['mean_of_per_val_max']:.3f}  "
          f"p99={jc['p99']:.3f}  max={jc['max']:.3f}  #>=0.80: {jc['count_ge_0.80']}")
    print(f"  high-cosine pairs (cos>=0.95): {len(hi)}  -> "
          f"true near-identical (char>=0.95): {len(true_near_identical)}, "
          f"template collisions: {len(template_collisions)}")
    print(f"  near-duplicate (cos>=0.95) AND same answer: "
          f"{results['near_duplicate_same_answer_count']}")
    print(f"  REAL leakage (char>=0.95 AND same answer): "
          f"{len(real_leakage)}  ({results['real_leakage_fraction_pct']}% of val)")
    if retr_result.get("available"):
        print(f"  retriever exemplar top-1 score: "
              f"mean={retr_result['top1_score_mean']:.3f}  "
              f"max={retr_result['top1_score_max']:.3f}  "
              f"verbatim exemplars: {retr_result['verbatim_exemplar_count']}")
    else:
        print(f"  retriever check: SKIPPED ({retr_result.get('error')})")

    # top offending pairs (for manual eyeball)
    order = np.argsort(-max_sim)[:5]
    print("\n  top-5 most similar val<->train question pairs:")
    for j in order:
        ti = int(argmax[j])
        print(f"   cos={max_sim[j]:.3f} jac={max_jac[j]:.3f} "
              f"[{va_t[j]}] ans_val={va_a[j]!r} ans_train={tr_a[ti]!r}")
        print(f"      VAL  : {norm(va_q[j])[:140]}")
        print(f"      TRAIN: {norm(tr_q[ti])[:140]}")

    # LaTeX-ready mini table
    print("\n  ---- LaTeX snippet (per-val max question-level similarity) ----")
    print(f"  P50 & {tc['p50']:.2f} \\\\")
    print(f"  P90 & {tc['p90']:.2f} \\\\")
    print(f"  P99 & {tc['p99']:.2f} \\\\")
    print(f"  Max & {tc['max']:.2f} \\\\")
    print(f"  #ge0.90 & {tc['count_ge_0.90']} \\\\")
    print(f"\n[leakage] wrote {OUT}")


if __name__ == "__main__":
    main()
