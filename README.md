# TRA-SAE — Thought-Reasoning Agent with Symbolic Analysis and Evaluation

**DAT301m Project · EXACT 2026 Competition**

> Qwen3.5-4B fine-tuned via multi-phase SFT + GRPO with unit-aware rewards, dual-LoRA specialist routing, Z3 SMT verification, and self-consistency voting — targeting the EXACT 2026 public benchmark.

---

## Results

| Config | Description | Overall | Physics | Logic |
|--------|-------------|---------|---------|-------|
| 0 | Zero-shot (no LoRA) | 35.48% | 43.97% | 19.74% |
| 1 | + SFT Phase 1 | 52.53% | 65.25% | 28.95% |
| 2 | + Logic SFT Phase 1.5 | 51.61% | 65.25% | 26.32% |
| **3** | **+ GRPO mixed (best)** | **53.92%** | **67.38%** | **28.95%** |
| 4 | + Dual-LoRA router | 49.77% | 60.28% | 30.26% |
| 5 | + Self-consistency ×5 | 50.69% | 64.54% | 25.00% |

**EXACT 2026 public baseline: 38.71%** · Best achieved: **53.92%** (+15.21 pp)

Evaluation set: 217 samples (141 physics, 76 logic).

---

## Architecture

```
Input Question
     │
     ▼
Subject Router ──────────────────────────────────────────┐
     │                                                   │
     ▼ (physics)                         ▼ (logic)       │
Qwen3.5-4B + Physics LoRA     Qwen3.5-4B + Logic LoRA   │
     │                                   │               │
     └──────────────┬────────────────────┘               │
                    ▼                                    │
          Self-Consistency (5×)                          │
                    │                                    │
                    ▼                                    │
           Z3 SMT Verifier ◄────────────────────────────┘
                    │
                    ▼
             Final Answer
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| Config | `src/config.py` | Paths, model settings, LoRA hyperparams |
| Reward | `src/reward.py` | GRPO reward: format (0.30) + correctness (0.60) + unit (0.10) |
| Reward Evaluator | `src/reward_evaluator_keras.py` | TF/Keras self-evaluator for reward signal |
| Agent Graph | `src/agent_graph.py` | LangGraph-style multi-node inference graph |
| Agent Nodes | `src/agent_nodes.py` | Individual reasoning/retry/verification nodes |
| Symbolic Verifier | `src/symbolic_verifier.py` | Answer extraction + exact/fuzzy matching |
| Z3 Engine | `src/z3_engine.py` | SMT constraint solving for logic problems |
| Router | `src/router.py` | Physics vs. logic subject classifier |
| Retriever | `src/retriever.py` | Few-shot example retrieval |
| Data Utils | `src/data_utils.py` | Dataset loading + preprocessing |
| Model Loader | `src/model_loader.py` | HuggingFace + PEFT loader (BF16 on A100) |

---

## Training Pipeline

### Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA A100-SXM4-80GB |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.10.0.dev0 |
| TRL | 1.4.0 |
| PEFT | 0.19.1 |
| Z3-solver | 4.16.0 |

### Phases

```
Phase 1   ── SFT on full EXACT train set (1,945 samples)
              Loss: 0.3085  │  Steps: 366  │  Time: 58.8 min

Phase 1.5 ── Logic-specialist SFT (732 logic samples)
              Loss: 0.1326  │  Steps: 92   │  Time: 15.7 min

Phase 2   ── GRPO mixed training (unit-aware reward)
              Loss: 0.0295  │  Steps: 250  │  Time: 79.8 min

Phase 2P  ── GRPO physics specialist
              Loss: 0.0182  │  Steps: 200  │  Time: 76.6 min

Phase 2L  ── GRPO logic specialist
              Loss: 0.0703  │  Steps: 150  │  Time: 62.9 min

Phase 3   ── GRPO continued refinement
Phase 4   ── Agent evaluation + 6-config ablation
```

### GRPO Reward Breakdown

```
format_reward        0.30  — all three XML tags present and non-empty
correctness_reward   0.60  — symbolic answer match
unit_reward          0.10  — physical unit scale match bonus
length_penalty      −0.10  — applied when reasoning > 800 tokens
─────────────────────────────────────────────────────────
Total score ∈ [−0.10, 1.0]
```

---

## Quick Start

### Prerequisites

```bash
pip install transformers>=4.40.0 peft trl datasets torch z3-solver
# Optional: tensorflow (for reward_evaluator_keras.py)
pip install tensorflow
```

### Run Full Pipeline

```bash
# Run all phases sequentially
python run_all_steps.py

# Or run individual phases:
python run_phase1_sft.py          # Phase 1: SFT
python run_phase1_5_logic_sft.py  # Phase 1.5: Logic SFT
python run_phase2_grpo.py         # Phase 2: GRPO mixed
python run_phase2_grpo_physics.py # Phase 2P: Physics specialist
python run_phase2_grpo_logic.py   # Phase 2L: Logic specialist
python run_phase3_grpo.py         # Phase 3: GRPO refinement
python run_phase4_v2_agent.py     # Phase 4: Evaluation + ablation
```

### Evaluation Only

```bash
# Full ablation (all 6 configs, 217 samples)
python run_phase4_v2_agent.py

# Quick smoke test (10 samples per config)
python run_phase4_v2_agent.py --smoke-test

# Single config
python run_phase4_v2_agent.py --config 3
```

### Output Format

The model generates structured responses:

```xml
<reasoning>
[Step-by-step chain-of-thought]
</reasoning>
<answer>
[Final answer: letter / Yes/No/Unknown / number+unit]
</answer>
<explanation>
[Concise explanation]
</explanation>
```

---

## Repository Structure

```
TRA-SAE/
├── src/                        # Core modules
│   ├── config.py               # Central configuration
│   ├── reward.py               # GRPO reward functions
│   ├── reward_evaluator_keras.py
│   ├── agent_graph.py          # Multi-node agent graph
│   ├── agent_nodes.py
│   ├── symbolic_verifier.py    # Answer extraction + matching
│   ├── z3_engine.py            # SMT solver integration
│   ├── router.py               # Subject classifier
│   ├── retriever.py            # Few-shot retrieval
│   ├── data_utils.py
│   └── model_loader.py
├── experiments/                # Experimental scripts
│   ├── step1_rerun_cfg0_3.py
│   ├── step2_multiseed_cfg3.py
│   ├── step3_baselines.py
│   ├── step4_stats_and_errors.py
│   ├── step5_reward_ablation.py
│   └── step6_latency.py
├── tests/
│   └── test_verifier.py
├── logs/                       # Experiment results (JSON/JSONL)
├── paper/                      # LaTeX paper + docs
│   ├── TRA_SAE_final_paper.pdf
│   ├── TRA_SAE_EXACT2026_paper.tex
│   ├── references.bib
│   ├── PAPER_PLAN.md
│   └── EXACT2026_experiment_record.md
├── data/                       # EXACT 2026 dataset (original)
├── processed_data/             # Tokenized HuggingFace datasets
├── run_phase1_sft.py           # Training entry points
├── run_phase1_5_logic_sft.py
├── run_phase2_grpo.py
├── run_phase2_grpo_physics.py
├── run_phase2_grpo_logic.py
├── run_phase3_grpo.py
├── run_phase4_v2_agent.py
├── run_all_steps.py
├── TRA-SAE_Qwen3.5-4B.ipynb   # Colab notebook
└── kehoach.md                  # Project plan (Vietnamese)
```

---

## Data

- **Logic_Based_Educational_Queries** — 411 records (multiple-choice logic)
- **Physics_Problems_Text_Only** — 1,755 rows (numerical physics, free-response)
- Train split: 1,945 samples · Val/eval split: 217 samples

Dataset source: EXACT 2026 official release (2026-05-09).

---

## Checkpoints

Model checkpoints are **not tracked in git** (multi-GB). Paths defined in `src/config.py`:

| Checkpoint | Path |
|------------|------|
| SFT (mixed) | `checkpoints/qwen35_sft/final` |
| SFT (logic) | `checkpoints/qwen35_sft_logic/final` |
| GRPO (mixed) | `checkpoints/qwen35_grpo/final` |
| GRPO (physics) | `checkpoints/qwen35_grpo_physics/final` |
| GRPO (logic) | `checkpoints/qwen35_grpo_logic/final` |

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{trasae2026,
  title  = {TRA-SAE: Thought-Reasoning Agent with Symbolic Analysis and Evaluation},
  year   = {2026},
  note   = {EXACT 2026 Competition, DAT301m}
}
```

---

## License

For academic/research use only. Dataset subject to EXACT 2026 terms of use.
