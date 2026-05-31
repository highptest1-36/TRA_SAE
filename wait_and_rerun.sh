#!/bin/bash
cd /content/drive/MyDrive/TRA-SAE
LOG_TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Watching step6 PID=25272 and pipeline9907 PID=9907..."

# Chờ cả step6 lẫn pipeline 9907 xong
while ps -p 25272 > /dev/null 2>&1 || ps -p 9907 > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] step6 + pipeline9907 DONE. Checking status..."
cat /content/drive/MyDrive/TRA-SAE/logs/pipeline_status.json | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {k}: {v[\"status\"]}') for k,v in d['steps'].items()]"

echo "[$(date)] Starting step2, step3, step4, step5 with fixed code..."
python3 -u run_all_steps.py --steps step2 step3 step4 step5 --force \
    > logs/pipeline_rerun_${LOG_TS}.log 2>&1
echo "[$(date)] Done. Exit=$?"
