#!/bin/bash
# Re-run C2 (FOLIO logic) cfg2 + cfg3, which crashed on a torchao/peft version
# clash (now fixed by removing torchao). cfg0 already succeeded (34.48%).
# Each config runs in its own fresh process; then merge cfg0+cfg2+cfg3.
set -u
cd /content/drive/MyDrive/TRA-SAE

echo "[c2fix] waiting for C1 resume+merge to finish (sentinel)..."
while ! grep -q "\[resume\] ALL DONE" logs/_c1_resume_chain.log 2>/dev/null; do
  sleep 60
done
sleep 10
echo "[c2fix] C1 done; running C2 cfg2 at $(date +%H:%M:%S)"

python3 experiments/step9_external_benchmark.py --benchmark logic --configs cfg2 \
    > logs/_c2_cfg2.log 2>&1
cp logs/external_benchmark_logic_results_latest.json logs/external_benchmark_logic_cfg2.json
echo "[c2fix] cfg2 done at $(date +%H:%M:%S)"

python3 experiments/step9_external_benchmark.py --benchmark logic --configs cfg3 \
    > logs/_c2_cfg3.log 2>&1
cp logs/external_benchmark_logic_results_latest.json logs/external_benchmark_logic_cfg3.json
echo "[c2fix] cfg3 done at $(date +%H:%M:%S)"

# Merge cfg0 (backup) + cfg2 + cfg3 into the canonical logic file
python3 - <<'PY'
import json
base = json.load(open('logs/external_benchmark_logic_cfg0_backup.json'))
runs = {r['config']: r for r in base['results']}     # cfg0
for f in ('logs/external_benchmark_logic_cfg2.json',
          'logs/external_benchmark_logic_cfg3.json'):
    try:
        d = json.load(open(f))
        for r in d['results']:
            runs[r['config']] = r
    except Exception as e:
        print('WARN merge', f, repr(e))
base['results'] = [runs[k] for k in ('cfg0', 'cfg2', 'cfg3') if k in runs]
json.dump(base, open('logs/external_benchmark_logic_results_latest.json', 'w'), indent=2)
print('[c2 merge] configs:', [(r['config'], r['accuracy_overall']) for r in base['results']])
PY
echo "[c2fix] ALL C2 DONE at $(date +%H:%M:%S)"
