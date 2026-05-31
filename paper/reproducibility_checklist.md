# Reproducibility Checklist (EXACT 2026 run)

## Code and config
- [x] Main eval script fixed and archived: run_phase4_v2_agent.py
- [x] Global config archived: src/config.py
- [x] Checkpoint paths documented in config

## Data
- [x] Raw dataset snapshot present under data/
- [x] Processed train split present: processed_data/exact_train
- [x] Processed val split present: processed_data/exact_val
- [x] Split sizes recorded in paper notes

## Model artifacts
- [x] SFT checkpoint: checkpoints/qwen35_sft/final
- [x] Logic SFT checkpoint: checkpoints/qwen35_sft_logic/final
- [x] Mixed GRPO checkpoint: checkpoints/qwen35_grpo/final
- [x] Physics specialist checkpoint: checkpoints/qwen35_grpo_physics/final
- [x] Logic specialist checkpoint: checkpoints/qwen35_grpo_logic/final
- [x] Router artifact: checkpoints/router.pkl (if present)

## Results artifacts
- [x] Phase 4 ablation JSON: logs/qwen35_ablation.json
- [x] Phase 4 summary JSON: logs/qwen35_final_summary.json
- [x] Phase 4 per-sample JSONL: logs/qwen35_final_results.jsonl
- [x] Phase 4 stdout logs: logs/phase4_ablation_stdout.log and logs/phase4_ablation_cfg4_5.log
- [x] Phase 1/1.5/2/2P/2L result JSON files present in logs/

## Environment
- [x] Python version captured
- [x] GPU + driver captured
- [x] Package versions captured (torch, trl, transformers, peft, torchao, z3)

## Reporting
- [x] Consolidated experiment record created: paper/EXACT2026_experiment_record.md
- [x] Ablation table CSV created: paper/phase4_ablation_results.csv
- [x] Data quality note created: paper/data_quality_checks.md
- [x] Artifact manifest created: paper/artifact_manifest.json
