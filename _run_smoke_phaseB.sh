#!/bin/bash
# Phase B smoke runner — sequential (shared GPU). De-risks step3/8/9 before full Phase B.
cd /content/drive/MyDrive/TRA-SAE
set -o pipefail

echo "######## SMOKE 1/4: step3_baselines (B2 only) ########"
python3 -u experiments/step3_baselines.py --smoke-test --only-ids B2
echo "===STEP3 EXIT=$?==="

echo "######## SMOKE 2/4: step8_fair_baselines (B2, cot_boxed) ########"
python3 -u experiments/step8_fair_baselines.py --smoke-test --only-ids B2 --only-strategies cot_boxed
echo "===STEP8 EXIT=$?==="

echo "######## SMOKE 3/4: step9_external_benchmark (cfg0) ########"
python3 -u experiments/step9_external_benchmark.py --smoke-test --configs cfg0
echo "===STEP9 EXIT=$?==="

echo "######## SMOKE 4/4: check_consistency ########"
python3 -u experiments/check_consistency.py
echo "===CONSISTENCY EXIT=$?==="

echo "######## ALL SMOKE DONE ########"
