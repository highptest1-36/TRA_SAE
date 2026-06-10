# Data Quality Checks for Reported Metrics

Date: 2026-05-27

## Check 1: Split-count consistency inside ablation JSON
Source: logs/qwen35_ablation.json

Rule checked:
- n_physics + n_logic should be less than or equal to n_total

Observed:
- Config 0: n_total=217, n_physics=146, n_logic=76, sum=222 (inconsistent)
- Config 1: n_total=217, n_physics=146, n_logic=76, sum=222 (inconsistent)
- Config 2: n_total=217, n_physics=146, n_logic=76, sum=222 (inconsistent)
- Config 3: n_total=217, n_physics=146, n_logic=76, sum=222 (inconsistent)
- Config 4: n_total=217, n_physics=141, n_logic=76, sum=217 (consistent)
- Config 5: n_total=217, n_physics=141, n_logic=76, sum=217 (consistent)

Implication for paper:
- Use overall accuracy values as primary score.
- Use per-subject percentages with caution for configs 0-3 until denominator provenance is re-validated.
- Config 4 and 5 subset counts align exactly with val split composition (141 physics, 76 logic).

## Check 2: Runtime consistency
- Config 5 elapsed time is much larger (342.8 min), consistent with self-consistency sampling overhead.
- Configs 0-4 elapsed times (71-90 min) are in expected range for batched evaluation.

## Check 3: Confidence fields
- avg_confidence_correct and avg_confidence_wrong are 0.0 across configs.
- Do not use these fields for calibration claims in the paper.
