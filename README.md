# TRA-SAE — Thought-Reasoning Agent with Symbolic Analysis and Evaluation

Code, logs, and analysis scripts for the paper

> **A Cost-Aware Empirical Study of Retrieval, Supervised Fine-Tuning, and Reinforcement Learning for Physics–Logic Question Answering with a 4B Language Model**
> C.-P. Ha et al. Submitted to ICCIES 2027.

TRA-SAE is a low-cost pipeline built on **Qwen3.5-4B** with LoRA adapters. It combines TF-IDF exemplar retrieval, a three-tag response format, supervised fine-tuning (SFT), logic-focused SFT, and GRPO with a composite reward. The pipeline serves as a **controlled setting** for measuring what each component contributes on the EXACT 2026 physics–logic data. It is not proposed as a new algorithm.

---

## Results reported in the paper

All numbers are **single-pass** accuracies on the 217-item validation split (141 physics, 76 logic). The TRA-SAE configurations decode once at temperature 0.1. The evaluation script can retry wrong answers, but retries depend on the answer key and are **not** used. Only the first pass (`retry_count == 0`) is counted.

| ID | Configuration | Overall | Physics | Logic |
|----|---------------|--------:|--------:|------:|
| cfg0 | Zero-shot | 29.95 | 38.30 | 14.47 |
| cfg0-R | + Retrieval, no fine-tuning (control) | 39.17 | 50.35 | 18.42 |
| cfg1 | + SFT | 44.24 | 58.16 | 18.42 |
| cfg2 | + Logic SFT | 46.08 | 59.57 | 21.05 |
| **cfg3** | **+ GRPO (best)** | **47.47** | **63.12** | 18.42 |
| cfg4 | Dual-LoRA domain adapters | 43.78 | 55.32 | 22.37 |

The best zero-shot 7B baseline (Qwen2-Math-7B-Instruct, best of three prompt formats) reaches 29.03.

Paired McNemar tests (continuity-corrected): retrieval cfg0→cfg0-R p = 7.8e-4; SFT cfg0-R→cfg1 p = 0.054; logic SFT cfg1→cfg2 p = 0.39; GRPO cfg2→cfg3 p = 0.61; Dual-LoRA cfg3→cfg4 p = 0.14; cfg0→cfg3 p = 9.3e-8.

Training the evaluated configurations takes 293.8 min on one A100-SXM4-80GB, about **US$18** at US$3.67 per GPU-hour. The evaluation runs take a further 750.8 min (US$45.9). Diagnostic experiments (baselines, reward ablation, tools, seeds, transfer) are not included in these figures.

> **Note on older numbers.** Earlier versions of this repository reported higher accuracies (for example 53.92% for cfg3). Those numbers counted answers recovered by the **answer-dependent retry loop** and are not valid single-pass results. The table above supersedes them. The former "cfg5 / self-consistency ×5" row is also removed: self-consistency ran only inside the retry loop, so in the first pass cfg5 is identical to cfg4.

### Where each number comes from

| Paper item | Source in this repository |
|---|---|
| Main results, McNemar tests, Wilson intervals | `logs/ablation_per_sample_canonical.jsonl` (cfg0–cfg4, first pass = `retry_count == 0`) and `logs/ablation_per_sample_canonical_partB.jsonl` (cfg0-R = config 6), recomputed by `analysis/verify_and_sensitivity.py` |
| Baselines with and without retrieval | `logs/fair_baselines_results_latest.json`, `logs/fair_baselines_retrieval_results_latest.json` |
| Seed stability | `logs/cfg3_multiseed_results_latest.json`, `logs/ablation_per_sample_canonical_seed*.jsonl` |
| Overlap audit | `experiments/step10_leakage_check.py` → `logs/leakage_check_results.json` |
| Overlap sensitivity analysis | `analysis/verify_and_sensitivity.py` → `analysis/results/sensitivity_results.json` |
| MMLU physics and FOLIO transfer | `logs/external_benchmark_results_latest.json`, `logs/external_benchmark_logic_results_latest.json` |
| Tool augmentation | `logs/tool_baselines_results_latest.json` |
| Reward ablation | `logs/reward_ablation_results_latest.json` |
| Compute cost | `logs/compute_profile_latest.json` |
| Error re-classification | `analysis/error_recheck.py` → `analysis/results/error_recheck_cfg3.csv` |
| Logic label formats | `analysis/logic_label_audit.py` → `analysis/results/logic_label_audit.json` |

---

## Pipeline

```
Training
  Qwen3.5-4B ─► Stage 1: SFT (1,945 samples) ─► Stage 2: logic SFT (732 samples) ─► Stage 3: GRPO (K = 4)
                                                       reward = 0.30 format + 0.60 correctness
                                                              + 0.10 unit − 0.10 if reasoning > 800 tokens

Inference (single pass)
  query ─► TF-IDF retriever (top-3 same-domain exemplars) ─► fine-tuned 4B model
        ─► <reasoning> <answer> <explanation> ─► extractor + verifier ─► answer
```

* **Retriever** (`src/retriever.py`): TF-IDF cosine similarity, top 3, restricted to the item's domain label.
* **Dual-LoRA** (cfg4): physics and logic GRPO adapters, selected by the **domain label** of each item. `src/router.py` (TF-IDF + logistic regression) is used only for inputs without a label and is not needed for the EXACT data.
* **Verifier** (`src/symbolic_verifier.py`): option letters and Yes/No/Unknown labels after normalisation; Z3 for logic items when a formal form can be parsed; numerical answers after SI conversion with a **2% relative tolerance**.
* **Extractor**: `<answer>` tag, then `\boxed{}`, then the last non-empty line.

| Module | File |
|---|---|
| Configuration and hyper-parameters | `src/config.py` |
| GRPO reward | `src/reward.py` |
| Answer extraction and verification | `src/symbolic_verifier.py` |
| Z3 engine | `src/z3_engine.py` |
| Retriever | `src/retriever.py` |
| Router (unlabelled inputs only) | `src/router.py` |
| Data loading | `src/data_utils.py` |

### Hyper-parameters

| Parameter | Value |
|---|---|
| LoRA r / α / dropout | 32 / 64 / 0.05, all seven projection matrices |
| Stage 1 SFT | lr 2e-4, 3 epochs, effective batch 16 (58.8 min) |
| Stage 2 logic SFT | lr 5e-5, 2 epochs (15.7 min) |
| Stage 3 GRPO | lr 1e-6, 250 steps, K = 4, β = 0.04 (79.8 min) |
| Max new tokens / exemplars | 1,024 / 3 |

---

## Data

The organizers released two training files (snapshot of 2026-05-09):

* `Logic_Based_Educational_Queries.json`: 411 records, each with shared premises and one or more questions, giving **808 logic questions**. 329 of them are multiple-choice questions with lettered options.
* `Physics_Problems_Text_Only.csv`: 1,755 problems. 401 have no gold answer and are dropped, leaving **1,354 physics questions**.

The 2,162 samples are shuffled and split **at random** (not stratified) with seed 42 into **1,945 training** (1,213 physics, 732 logic) and **217 validation** samples (141 physics, 76 logic). See the data cell of `TRA-SAE_Qwen3.5-4B.ipynb` and `src/data_utils.py`.

The EXACT 2026 data are subject to the challenge terms of use.

---

## Reproducing the analyses (no GPU)

```bash
pip install numpy scipy scikit-learn pyarrow
python analysis/verify_and_sensitivity.py . analysis/results/sensitivity_results.json
python analysis/error_recheck.py . analysis/results/error_recheck_results.json analysis/results/error_recheck_cfg3.csv
python analysis/logic_label_audit.py . analysis/results/logic_label_audit.json
```

`analysis/make_annotation_sheet.py` builds a spreadsheet for manual validation of the error categories.

## Training and evaluation

```bash
python run_phase1_sft.py            # Stage 1
python run_phase1_5_logic_sft.py    # Stage 2
python run_phase2_grpo.py           # Stage 3, mixed adapter (cfg3)
python run_phase2_grpo_physics.py   # cfg4 physics adapter
python run_phase2_grpo_logic.py     # cfg4 logic adapter
python experiments/step0_canonical_eval.py --retries 0   # single-pass evaluation (add --config N for one configuration)
```

Environment: A100-SXM4-80GB, Python 3.12, PyTorch 2.10, Transformers 5.10.dev, TRL 1.4.0, PEFT 0.19.1, z3-solver 4.16.0. LoRA checkpoints are not tracked in git; their paths are set in `src/config.py`.

---

## Repository layout

```
src/                 core modules (config, reward, verifier, Z3 engine, retriever, router, data loading)
run_phase*.py        training entry points (SFT, logic SFT, GRPO, specialist adapters)
experiments/         evaluation, baselines, ablations, statistics, overlap audit
analysis/            post-hoc analyses for the paper (no GPU) and their results
logs/                per-sample predictions and result files behind every reported number
data/                EXACT 2026 training files (release of 2026-05-09)
processed_data/      train/validation split (HuggingFace Arrow)
tests/               verifier unit tests
API_TRA_SAE/         serving layer used for the EXACT 2026 competition
docs/project_notes/  historical working notes (superseded numbers, see the README in that folder)
TRA-SAE_Qwen3.5-4B.ipynb   end-to-end Colab notebook
```

---

## Citation

```bibtex
@misc{ha2026trasae,
  author = {Ha, Cao-Phuc and others},
  title  = {A Cost-Aware Empirical Study of Retrieval, Supervised Fine-Tuning, and
            Reinforcement Learning for Physics--Logic Question Answering with a 4B Language Model},
  year   = {2026},
  note   = {Code: https://github.com/highptest1-36/TRA_SAE}
}
```

For academic and research use only.
