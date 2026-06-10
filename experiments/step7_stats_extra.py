#!/usr/bin/env python3
"""
step7_stats_extra.py
====================
Phase B of the paper revision plan.

Computes:
  1. Bootstrap 95% CI (10 000 resamples) for every config / baseline.
  2. Extended pairwise McNemar tests (continuity-corrected, two-sided)
     between all pairs that have per-sample data (cfg0-cfg3 from v2
     repeat eval) plus aggregate-level binomial CI for cfg4, cfg5,
     and baselines.
  3. A waterfall figure showing accuracy decomposition (v1 canonical
     numbers):  zero-shot → +SFT → ±logic SFT → +GRPO → cfg3

Outputs
-------
  logs/stats_extra.json          -- all CI and McNemar results
  paper/fig_decomposition.pdf    -- waterfall figure for the paper
  paper/fig_decomposition.png    -- raster copy for README

Uses
----
  logs/ablation_per_sample_latest.jsonl  (v2 repeat eval, cfg0-cfg3)
  logs/qwen35_ablation.json              (v1 canonical, all cfgs)
  logs/baselines_results_latest.json     (baseline aggregate)
  logs/cfg3_multiseed_results_latest.json
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # /…/TRA-SAE
LOGS = ROOT / "logs"
PAPER = ROOT / "paper"

PER_SAMPLE_FILE = LOGS / "ablation_per_sample_latest.jsonl"
ABLATION_V1_FILE = LOGS / "qwen35_ablation.json"
BASELINES_FILE = LOGS / "baselines_results_latest.json"
MULTISEED_FILE = LOGS / "cfg3_multiseed_results_latest.json"
STATS_RESULTS_FILE = LOGS / "stats_results.json"  # existing partial stats

OUTPUT_JSON = LOGS / "stats_extra.json"
OUTPUT_PDF = PAPER / "fig_decomposition.pdf"
OUTPUT_PNG = PAPER / "fig_decomposition.png"

SEED = 42
N_BOOTSTRAP = 10_000

# ── Helpers ───────────────────────────────────────────────────────────

def bootstrap_ci(correct_vec, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """Return (mean_acc, lo, hi) as percentages via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    arr = np.asarray(correct_vec, dtype=float)
    n = len(arr)
    boot = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1) * 100
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(arr.mean() * 100), lo, hi


def binomial_ci(n_correct, n_total, alpha=0.05):
    """Wilson score interval (closed-form, reliable for any n)."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_total
    z = 1.959964  # z_{0.975}
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    lo = max(0.0, (centre - margin) * 100)
    hi = min(100.0, (centre + margin) * 100)
    return p * 100, lo, hi


def mcnemar_test(vec_a, vec_b):
    """
    McNemar's test (continuity-corrected, two-sided).
    Returns (chi2, p_value, b, c) where
      b = A-wrong AND B-correct
      c = A-correct AND B-wrong
    """
    from scipy.stats import chi2 as chi2_dist

    b = sum(1 for a, bb in zip(vec_a, vec_b) if not a and bb)
    c = sum(1 for a, bb in zip(vec_a, vec_b) if a and not bb)
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0, 0, 0
    stat = max(0.0, (abs(b - c) - 1) ** 2 / (b + c))
    p_val = float(1 - chi2_dist.cdf(stat, df=1))
    return float(stat), p_val, int(b), int(c)


def significance_label(p):
    if p < 0.001:
        return "p<0.001 ***"
    elif p < 0.01:
        return "p<0.01 **"
    elif p < 0.05:
        return "p<0.05 *"
    else:
        return f"p={p:.3f} n.s."


# ── Load per-sample data (v2 repeat eval, cfg0-cfg3) ──────────────────
print("Loading per-sample predictions …")
per_sample = {}   # config_id -> list[bool]
with open(PER_SAMPLE_FILE) as f:
    for line in f:
        r = json.loads(line)
        cid = r["config_id"]
        per_sample.setdefault(cid, []).append(bool(r["correct"]))

print(f"  Configs in per-sample file: { {k: len(v) for k, v in per_sample.items()} }")

# ── Load v1 canonical aggregate data ─────────────────────────────────
with open(ABLATION_V1_FILE) as f:
    ablation_v1 = json.load(f)

configs_v1 = {}
for entry in ablation_v1["ablation"]:
    cid = entry["config_id"]
    configs_v1[cid] = {
        "name": entry["config_name"],
        "description": entry["description"],
        "acc": entry["accuracy_overall"],
        "phys": entry["accuracy_physics"],
        "logic": entry["accuracy_logic"],
        "n_correct": entry["n_correct"],
        "n_total": entry["n_total"],
        "n_physics": 141,  # corrected; v1 metadata had 146 (bug)
        "n_logic": 76,
    }

# ── Load baselines ────────────────────────────────────────────────────
with open(BASELINES_FILE) as f:
    baselines_raw = json.load(f)["baselines"]

baselines = {}
for b in baselines_raw:
    baselines[b["id"]] = b

# ── Bootstrap CI for configs with per-sample data (v2 eval) ───────────
print("Computing bootstrap CIs from per-sample data (v2 eval) …")
rng = np.random.default_rng(SEED)

ci_v2 = {}
for cid in sorted(per_sample.keys()):
    vec = per_sample[cid]
    acc, lo, hi = bootstrap_ci(vec, rng=rng)
    ci_v2[cid] = {"acc": acc, "ci_lo": lo, "ci_hi": hi, "n": len(vec)}
    print(f"  cfg{cid}: {acc:.2f}% [{lo:.2f}, {hi:.2f}]")

# ── Wilson CI for v1 aggregate (main table configs) ────────────────────
print("Computing Wilson CIs from v1 aggregate data …")
ci_v1 = {}
for cid, info in configs_v1.items():
    acc, lo, hi = binomial_ci(info["n_correct"], info["n_total"])
    ci_v1[cid] = {"acc": acc, "ci_lo": lo, "ci_hi": hi,
                  "n_correct": info["n_correct"], "n_total": info["n_total"],
                  "name": info["name"], "description": info["description"]}
    print(f"  cfg{cid} ({info['name']}): {acc:.2f}% [{lo:.2f}, {hi:.2f}]")

# ── Wilson CI for baselines ────────────────────────────────────────────
ci_baselines = {}
for bid, info in baselines.items():
    acc, lo, hi = binomial_ci(info["n_correct"], info["n_total"])
    ci_baselines[bid] = {
        "model": info["description"],
        "acc": acc, "ci_lo": lo, "ci_hi": hi,
        "acc_physics": info["accuracy_physics"],
        "acc_logic": info["accuracy_logic"],
        "n_correct": info["n_correct"], "n_total": info["n_total"],
    }
    print(f"  {bid}: {acc:.2f}% [{lo:.2f}, {hi:.2f}]")

# ── Pairwise McNemar (cfg0-cfg3, v2 data) ─────────────────────────────
print("Computing McNemar tests …")
mcnemar_pairs = [
    (0, 1, "cfg0 vs cfg1"),
    (1, 2, "cfg1 vs cfg2"),
    (1, 3, "cfg1 vs cfg3"),
    (2, 3, "cfg2 vs cfg3"),
    (0, 3, "cfg0 vs cfg3"),
]

mcnemar_results = []
for cid_a, cid_b, label in mcnemar_pairs:
    if cid_a not in per_sample or cid_b not in per_sample:
        print(f"  Skipping {label} – no per-sample data")
        continue
    va, vb = per_sample[cid_a], per_sample[cid_b]
    stat, p_val, b, c = mcnemar_test(va, vb)
    # delta accuracy using v2 data
    delta_v2 = ci_v2[cid_b]["acc"] - ci_v2[cid_a]["acc"]
    # delta accuracy using v1 canonical numbers (for paper narrative)
    delta_v1 = configs_v1[cid_b]["acc"] - configs_v1[cid_a]["acc"]
    sig = significance_label(p_val)
    interp = ("statistically significant" if p_val < 0.05
              else "not statistically significant — interpret cautiously")
    mcnemar_results.append({
        "comparison": label,
        "cid_a": cid_a,
        "cid_b": cid_b,
        "delta_acc_v1": round(delta_v1, 2),
        "delta_acc_v2": round(delta_v2, 2),
        "b": b,  # B wrong, A correct
        "c": c,  # A wrong, B correct → net = b – c
        "chi2_statistic": round(stat, 4),
        "p_value": round(p_val, 5),
        "significance": sig,
        "interpretation": interp,
    })
    print(f"  {label}: Δ={delta_v1:+.2f}pp (v1) / {delta_v2:+.2f}pp (v2)"
          f"  b={b} c={c}  {sig}")

# Also load existing stats_results for cfg3 vs cfg0 cross-check
existing_mcnemar = {}
try:
    with open(STATS_RESULTS_FILE) as f:
        existing = json.load(f)
    for m in existing.get("mcnemar_tests", []):
        existing_mcnemar[m["comparison"]] = m
    print(f"  Loaded {len(existing_mcnemar)} existing McNemar results from stats_results.json")
except Exception as e:
    print(f"  Could not load existing stats_results.json: {e}")

# ── Multi-seed summary ────────────────────────────────────────────────
with open(MULTISEED_FILE) as f:
    multiseed = json.load(f)

# ── Save all results ──────────────────────────────────────────────────
output = {
    "note": (
        "CI values for cfg0-cfg3 are computed from ablation_per_sample_latest.jsonl "
        "(v2 repeat evaluation with consistent XML prompting, n=217). "
        "CI values for cfg4, cfg5, and baselines are computed via Wilson score interval "
        "from aggregate n_correct/n_total (v1 canonical evaluation). "
        "McNemar tests use v2 per-sample data. "
        "Main paper results table uses v1 canonical numbers."
    ),
    "n_bootstrap": N_BOOTSTRAP,
    "ci_v2_per_sample": ci_v2,
    "ci_v1_aggregate": ci_v1,
    "ci_baselines": ci_baselines,
    "mcnemar_tests": mcnemar_results,
    "existing_mcnemar_from_stats_results": existing_mcnemar,
    "multiseed_cfg3": {
        "seeds": multiseed["seeds"],
        "per_seed": multiseed["per_seed"],
        "mean_overall": multiseed["mean_overall"],
        "std_overall": multiseed["std_overall"],
        "mean_physics": multiseed["mean_physics"],
        "std_physics": multiseed["std_physics"],
        "mean_logic": multiseed["mean_logic"],
        "std_logic": multiseed["std_logic"],
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved stats to {OUTPUT_JSON}")

# ── Pretty-print LaTeX snippet for CI table ───────────────────────────
print("\n--- LaTeX CI table rows (v1 Wilson CI) ---")
name_map = {0: "cfg0 (zero-shot)", 1: "cfg1 (+SFT Ph.1)", 2: "cfg2 (+Logic SFT)",
            3: "cfg3 (+GRPO)", 4: "cfg4 (Dual-LoRA)", 5: "cfg5 (Full Agent)"}
for cid in sorted(ci_v1.keys()):
    ci = ci_v1[cid]
    row_overall = f"{ci['acc']:.2f}\\% [{ci['ci_lo']:.1f}, {ci['ci_hi']:.1f}]"
    print(f"  {name_map.get(cid, f'cfg{cid}'):30s}  {row_overall}")

print("\n--- LaTeX McNemar table rows ---")
for m in mcnemar_results:
    print(f"  {m['comparison']:20s}  Δ={m['delta_acc_v1']:+.2f}pp  "
          f"b={m['b']} c={m['c']}  {m['significance']:20s}  {m['interpretation']}")

# ── Waterfall Figure ──────────────────────────────────────────────────
print("\nGenerating waterfall / decomposition figure …")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # V1 canonical numbers
    stages = [
        ("Zero-shot\n(cfg0)",      35.48, 35.48),  # (label, start, value)
        ("+ Curriculum\nSFT (cfg1)", 35.48, 52.53),
        ("+ Logic SFT\n(cfg2)",      52.53, 51.61),
        ("+ GRPO\n(cfg3)",           51.61, 53.92),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = {
        "base":  "#4C72B0",
        "pos":   "#55A868",
        "neg":   "#C44E52",
        "final": "#DD8452",
    }

    x_pos = range(len(stages))
    bar_width = 0.55

    for i, (label, start, end) in enumerate(stages):
        delta = end - start
        if i == 0:
            # Base bar
            ax.bar(i, end, width=bar_width, color=colors["base"],
                   alpha=0.85, edgecolor="white", linewidth=0.8)
            ax.text(i, end + 0.6, f"{end:.2f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        else:
            prev_end = stages[i - 1][2]
            color = colors["pos"] if delta >= 0 else colors["neg"]
            ax.bar(i, abs(delta), bottom=min(start, end),
                   width=bar_width, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
            # Transparent base to stack
            ax.bar(i, min(start, end), width=bar_width,
                   color="white", alpha=0.0)
            # Connecting line from previous end
            ax.plot([i - 1 + bar_width / 2, i - bar_width / 2],
                    [prev_end, prev_end], color="gray",
                    linestyle="--", linewidth=0.8, alpha=0.6)
            sign = "+" if delta >= 0 else ""
            ax.text(i, max(start, end) + 0.6,
                    f"{sign}{delta:.2f}pp\n({end:.2f}%)",
                    ha="center", va="bottom", fontsize=9,
                    color=color if abs(delta) > 0.5 else "gray",
                    fontweight="bold")

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([s[0] for s in stages], fontsize=10)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy Gain Decomposition — TRA-SAE Pipeline\n"
                 "(Qwen3.5-4B, EXACT 2026 Validation, n=217)",
                 fontsize=11)
    ax.set_ylim(25, 62)
    ax.yaxis.grid(True, alpha=0.35, linestyle=":")
    ax.set_axisbelow(True)

    # Legend
    pos_patch = mpatches.Patch(color=colors["pos"], alpha=0.85, label="Positive gain")
    neg_patch = mpatches.Patch(color=colors["neg"], alpha=0.85, label="Negative / marginal")
    base_patch = mpatches.Patch(color=colors["base"], alpha=0.85, label="Base accuracy")
    ax.legend(handles=[base_patch, pos_patch, neg_patch],
              loc="lower right", fontsize=9, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"  Saved figure to {OUTPUT_PDF} and {OUTPUT_PNG}")

except Exception as e:
    print(f"  WARNING: Could not generate figure: {e}")
    print("  Install matplotlib to generate the figure: pip install matplotlib")

print("\nDone.")
