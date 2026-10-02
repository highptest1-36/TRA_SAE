# EXACT 2026 — Revision Run Guide (Colab Pro A100)

Sequential plan to regenerate ALL paper results consistently and add the four
reviewer-requested experiments. Run top-to-bottom. **Always smoke-test first.**

Estimated total: **~16–18 GPU-hours, ~$30–45** (Lambda A100 @ $3.67/h reference).
Everything writes to `logs/`; the paper rebuilds entirely from those JSON files.

---

## 0. One-time setup (every new Colab session)

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/TRA-SAE
!nvidia-smi --query-gpu=name,memory.total --format=csv   # confirm A100
!pip -q install "transformers>=4.46" peft trl datasets accelerate scipy z3-solver openai
```

```bash
# Set tokens (B7 gated download may need HF; B8 GPT-4o-mini needs OpenAI)
import os
os.environ["HF_TOKEN"] = "hf_xxx"          # for gated HF models
os.environ["OPENAI_API_KEY"] = "sk-xxx"    # only for B8 GPT-4o-mini
```

**Pre-flight check** — confirm the trained checkpoints exist (step0 needs them):

```bash
!ls -d checkpoints/qwen35_sft/final checkpoints/qwen35_sft_logic/final \
       checkpoints/qwen35_grpo/final checkpoints/qwen35_grpo_physics/final \
       checkpoints/qwen35_grpo_logic/final 2>&1
```
If any are missing, retrain that phase first (run_phase1_sft.py / run_phase1_5_logic_sft.py /
run_phase2_grpo*.py) before continuing.

---

## 1. SMOKE TEST (do this first — ~15 min, ~$1)

Validates every script end-to-end on 10 samples before spending GPU hours.

```bash
!python experiments/step0_canonical_eval.py --smoke-test
!python experiments/step13_tool_baselines.py --smoke-test
!python experiments/step3_baselines.py  --smoke-test --only-ids B2
!python experiments/step8_fair_baselines.py --smoke-test --only-ids B2 --only-strategies cot_boxed
!python experiments/step9_external_benchmark.py --smoke-test --configs cfg0
!python experiments/check_consistency.py    # expect: sanity OK (smoke n=10)
```
All five must finish without tracebacks. Then proceed to the real runs.

---

## 2. PHASE A — Canonical evaluation (BLOCKING, do before everything else)

This is the single source of truth. Resumable: if Colab disconnects, just re-run
the same command — completed configs are skipped automatically.

```bash
# ~9-10h (cfg5 self-consistency ~5-6h of it). Safe to re-run after disconnect.
!python experiments/step0_canonical_eval.py
```
Outputs:
- `logs/qwen35_ablation_canonical.json`     → Tables 2, 3 (CI), 7 (compute)
- `logs/ablation_per_sample_canonical.jsonl`→ Tables 5 (McNemar), 8 (errors)
- aliases `qwen35_ablation_v2_latest.json` + `ablation_per_sample_latest.jsonl` (downstream compat)

**Multi-seed** for the best config (read best from the run above; usually cfg2 or cfg3):

```bash
!python experiments/step0_canonical_eval.py --seed 1337 --config 2   # ~75 min
!python experiments/step0_canonical_eval.py --seed 2024 --config 2   # ~75 min
# (also add --config 3 runs if you want both candidates' std) → Table 4
```

---

## 3. PHASE B — Baselines & external experiments (can run in any order)

```bash
# B2-B8 zero-shot (open 7B + Qwen2.5-7B + GPT-4o-mini API). ~3-4h. Resumable per model.
!HF_TOKEN=$HF_TOKEN OPENAI_API_KEY=$OPENAI_API_KEY python experiments/step3_baselines.py
```
→ `logs/baselines_results_latest.json` → Table 2 (baseline rows), Table 3 (CI)

```bash
# Fair re-eval: each weak model with its native prompt (xml/cot_boxed/cot_plain). ~2.5h
!HF_TOKEN=$HF_TOKEN python experiments/step8_fair_baselines.py
```
→ `logs/fair_baselines_results_latest.json` → new "fair baseline" table; reframes
the "outperforms all 7B" claim into "competitive at lower parameter count."

```bash
# MMLU physics generalization: cfg0 vs cfg2 vs cfg3 on 253 MCQ. ~1h
!python experiments/step9_external_benchmark.py
```
→ `logs/external_benchmark_results_latest.json` → new "Generalization" table;
removes the "single-benchmark" limitation.

```bash
# Tool-augmented: LLM-direct vs LLM+Z3/Python on the best checkpoint. ~1.5h
!python experiments/step13_tool_baselines.py
```
→ `logs/tool_baselines_results_latest.json` → new "tool-assisted" table
(reviewer's missing item #2).

---

## 4. PHASE C — Statistics & error analysis (CPU, fast, free)

```bash
# McNemar (dynamic best config + full pairwise set) + rule-based error taxonomy
!python experiments/step4_stats_and_errors.py
```
→ `logs/stats_results.json` (Table 5), `logs/error_analysis.json` (Table 8)

```bash
# Manual error annotation (reviewer missing #3): generate 50-sample template
!python experiments/step14_manual_error_annotation.py
```
→ edit `logs/manual_error_template.csv`, fill the `manual_label` column
(FS/UC/AR/PU/LE/AX), save as `logs/manual_error_filled.csv`, then:

```bash
!python experiments/step14_manual_error_annotation.py --score
```
→ `logs/manual_error_agreement.json` (agreement % + Cohen's κ vs rule-based)

---

## 5. PHASE D — Consistency gate (run before touching the paper)

```bash
!python experiments/check_consistency.py
```
Must print **ALL CONSISTENCY CHECKS PASSED**. The CHECK 4 section lists every
config whose accuracy changed vs the old V1 numbers — that is your exact to-do
list of numbers to update in `paper/TRA_SAE_EXACT2026_paper.tex`.

---

## 6. Output → paper table map

| Output file | Paper element |
|---|---|
| `qwen35_ablation_canonical.json` | Table 2 (main), Table 3 (CI), Table 7 (compute) |
| `cfg{N}` multi-seed canonical files | Table 4 (mean ± std) |
| `baselines_results_latest.json` | Table 2 baseline rows B2–B8 |
| `fair_baselines_results_latest.json` | NEW fair-baseline table; §"Baseline robustness" |
| `external_benchmark_results_latest.json` | NEW generalization table |
| `tool_baselines_results_latest.json` | NEW tool-assisted table |
| `stats_results.json` | Table 5 (McNemar) |
| `error_analysis.json` + `manual_error_agreement.json` | Table 8 + reliability note |

---

## 7. Troubleshooting

- **Disconnect during step0** → re-run the exact same command; it resumes per-config.
- **OOM on 7B baselines** → lower `EVAL_BATCH_SIZE` (4→2) in step3/step8.
- **B7 download fails** → ensure `HF_TOKEN` set; or drop it with `--only-ids B2 B3 B4 B5 B6`.
- **B8 skipped** → `OPENAI_API_KEY` not set (intentional; B8 is optional commercial anchor).
- **Z3 import warning** → `pip install z3-solver`; logic verification falls back to text otherwise.
- **Never** hand-edit `logs/qwen35_ablation.json` (the OLD V1 file) — it is kept only for the
  CHECK-4 drift diff. The paper draws from the `*_canonical.json` files exclusively.
```
