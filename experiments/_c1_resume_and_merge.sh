#!/bin/bash
# Resume the 4 C1 runs that were lost when step8 was OOM-killed at run 12,
# then merge with the backed-up 11 runs into the canonical latest file.
# Runs in small fresh processes (<=3 runs each) to avoid the host-RAM
# accumulation that killed the original 15-in-one-process run.
set -u
cd /content/drive/MyDrive/TRA-SAE

echo "[resume] waiting for C2 (step9 logic) to release the GPU..."
while pgrep -f "[s]tep9_external_benchmark.py.*benchmark logic" >/dev/null 2>&1; do
  sleep 60
done
echo "[resume] C2 finished at $(date +%H:%M:%S); resuming missing C1 runs"

# B6 = all 3 strategies (fresh process)
python3 experiments/step8_fair_baselines.py --with-retrieval --only-ids B6 \
    > logs/_c1_resume_B6.log 2>&1
cp logs/fair_baselines_retrieval_results_latest.json logs/fair_baselines_retrieval_B6.json
echo "[resume] B6 done at $(date +%H:%M:%S)"

# B5 cot_plain = the single run that was cut off (fresh process)
python3 experiments/step8_fair_baselines.py --with-retrieval --only-ids B5 \
    --only-strategies cot_plain > logs/_c1_resume_B5cp.log 2>&1
cp logs/fair_baselines_retrieval_results_latest.json logs/fair_baselines_retrieval_B5cp.json
echo "[resume] B5 cot_plain done at $(date +%H:%M:%S)"

# Merge: backup(11) + B6(3) + B5cp(1), dedup by (id, strategy), latest wins.
python3 - <<'PY'
import json
base = json.load(open('logs/fair_baselines_retrieval_DONE11_backup.json'))
runs = {(r['id'], r['strategy']): r for r in base['results']}
for f in ('logs/fair_baselines_retrieval_B6.json',
          'logs/fair_baselines_retrieval_B5cp.json'):
    try:
        d = json.load(open(f))
        for r in d['results']:
            runs[(r['id'], r['strategy'])] = r
    except Exception as e:
        print('WARN merge', f, repr(e))
base['results'] = sorted(runs.values(), key=lambda r: (r['id'], r['strategy']))
base['retrieval'] = True
json.dump(base, open('logs/fair_baselines_retrieval_results_latest.json', 'w'), indent=2)
print(f"[merge] final runs = {len(base['results'])} :",
      [f"{r['id']}-{r['strategy']}" for r in base['results']])
PY
echo "[resume] ALL DONE at $(date +%H:%M:%S)"
