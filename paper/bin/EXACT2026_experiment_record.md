# EXACT 2026 - Qwen3.5-4B Pipeline Experiment Record

Date archived: 2026-05-27
Workspace: /content/drive/MyDrive/TRA-SAE

## 1) Objective
- Competition: EXACT 2026
- Public baseline to beat: 38.71%
- Internal target: 65% to 75%
- Final best achieved in this run: 53.92% (Config 3: grpo_mixed)

## 2) Environment snapshot
- OS: Linux
- Python: 3.12.13
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB
- Driver: 580.82.07
- Torch: 2.10.0+cu128
- CUDA runtime in torch: 12.8
- trl: 1.4.0
- transformers: 5.10.0.dev0
- peft: 0.19.1
- torchao: 0.17.0
- z3-solver: 4.16.0

Notes:
- torchao printed a compatibility warning for torch < 2.11.0, but the training/eval run completed successfully.

## 3) Data and splits
Source snapshot in data/README.txt:
- Logic set: 411 records
- Physics set: 1,755 rows

Processed splits used by pipeline:
- Train: 1,945 samples (physics 1,213; logic 732)
- Validation/eval set used in Phase 4: 217 samples (physics 141; logic 76)

## 4) Training pipeline summary
- Phase 1 SFT
  - Result file: logs/phase1_sft_results.json
  - Loss: 0.308529
  - Steps: 366
  - Time: 58.8 min
  - Output checkpoint: checkpoints/qwen35_sft/final

- Phase 1.5 Logic SFT
  - Result file: logs/phase1_5_logic_sft_results.json
  - Loss: 0.132642
  - Steps: 92
  - Time: 15.7 min
  - Output checkpoint: checkpoints/qwen35_sft_logic/final

- Phase 2 GRPO mixed
  - Result file: logs/phase2_grpo_results.json
  - Loss: 0.029465
  - Steps: 250
  - Time: 79.8 min
  - Output checkpoint: checkpoints/qwen35_grpo/final

- Phase 2P GRPO physics specialist
  - Result file: logs/phase2p_grpo_physics_results.json
  - Loss: 0.018171
  - Steps: 200
  - Time: 76.6 min
  - Output checkpoint: checkpoints/qwen35_grpo_physics/final

- Phase 2L GRPO logic specialist
  - Result file: logs/phase2l_grpo_logic_results.json
  - Loss: 0.070274
  - Steps: 150
  - Time: 62.9 min
  - Output checkpoint: checkpoints/qwen35_grpo_logic/final

## 5) Phase 4 ablation protocol
Evaluation script: run_phase4_v2_agent.py

Important settings used in final run:
- enable_thinking = False
- AGENT_MAX_RETRIES = 2
- MAX_NEW_TOKENS = 1024
- SELF_CONSISTENCY_N = 5
- EVAL_BATCH_SIZE = 8
- Batched generation enabled for attempts and for self-consistency internal sampling
- Left padding tokenizer for stable batched generation

## 6) Final ablation results
From logs/qwen35_ablation.json:

| Config | Name              | Overall | Physics | Logic | Expected |
|-------:|-------------------|--------:|--------:|------:|---------:|
| 0 | zero_shot         | 35.48 | 43.97 | 19.74 | 25-30 |
| 1 | sft_phase1        | 52.53 | 65.25 | 28.95 | 40-45 |
| 2 | sft_logic         | 51.61 | 65.25 | 26.32 | 45-50 |
| 3 | grpo_mixed        | 53.92 | 67.38 | 28.95 | 50-55 |
| 4 | dual_lora_router  | 49.77 | 60.28 | 30.26 | 55-60 |
| 5 | full_v2           | 50.69 | 64.54 | 25.00 | 65-75 |

Key outcomes:
- Best config: 3 (grpo_mixed) with 53.92%
- Improvement over competition baseline 38.71%: +15.21 percentage points
- Gap to 65% target: -11.08 percentage points
- Config 5 self-consistency was expensive: 342.8 min, no gain vs config 3

## 7) Paper-facing findings
1. The largest gain comes from Phase 1 SFT (+17.05 pp over zero-shot in this ablation).
2. Logic-only SFT (Phase 1.5) did not improve total score over Phase 1.
3. Mixed GRPO (Phase 2) gave the best total score.
4. Dual specialist routing did not outperform mixed GRPO in this setup.
5. Naive self-consistency (N=5) increased compute cost strongly without accuracy gain.

## 8) Limitations to report
- Logic subset remains low (max 30.26%).
- Confidence fields in exported JSON are 0.0 and are not informative for calibration claims.
- Router contribution needs dedicated error analysis (possible mis-routing on boundary cases).

## 9) Exact files for reproducibility
- Main ablation table: logs/qwen35_ablation.json
- Final summary: logs/qwen35_final_summary.json
- Per-sample config 5 outputs: logs/qwen35_final_results.jsonl
- Phase 4 run logs: logs/phase4_ablation_stdout.log, logs/phase4_ablation_cfg4_5.log
- Evaluation script: run_phase4_v2_agent.py
- Global config: src/config.py
