#!/bin/bash
cd /content/drive/MyDrive/TRA-SAE
LOG_TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Waiting for step6 to finish..."
wait_step6() {
    while ps -p 22484 > /dev/null 2>&1; do
        sleep 30
    done
    echo "[$(date)] step6 done."
}
wait_step6

echo "[$(date)] Starting rerun of step2, step3, step4, step5..."
python3 -u run_all_steps.py --steps step2 step3 step4 step5 --force \
    > logs/pipeline_rerun_${LOG_TS}.log 2>&1

echo "[$(date)] Rerun done. Exit: $?"
