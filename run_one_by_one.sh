#!/bin/bash
cd /content/drive/MyDrive/TRA-SAE

log() { echo "[$(date '+%H:%M:%S')] $*"; }

FILTER='grep -v "Loading weights\|Fetching\|torchao._C\|Warning\|flash\|causal\|unauthenticated\|it/s\|[█▏▎▍▌▋▊▉]"'

# ── Chờ step2 (PID 51047) xong, rồi kill pipeline runner (51046) ──
if ps -p 51047 > /dev/null 2>&1; then
    log "Chờ step2 (PID 51047) hoàn thành..."
    while ps -p 51047 > /dev/null 2>&1; do sleep 15; done
    log "Step2 DONE."
fi

# Kill pipeline runner cũ nếu còn
kill 51046 2>/dev/null && log "Killed pipeline runner 51046" || true
sleep 2

# ── STEP 3 ──
log "====== BẮT ĐẦU STEP 3 ======"
S3LOG="logs/direct_step3_$(date +%Y%m%d_%H%M%S).log"
python3 -u experiments/step3_baselines.py 2>&1 | tee "$S3LOG" | \
    grep -v "Loading weights\|Fetching\|torchao._C\|Warning\|flash\|causal\|unauthenticated\|it/s\|[█▏▎▍▌▋▊▉]"
S3_EXIT=${PIPESTATUS[0]}
log "Step3 exit=$S3_EXIT  log=$S3LOG"

# ── STEP 4 ──
log "====== BẮT ĐẦU STEP 4 ======"
S4LOG="logs/direct_step4_$(date +%Y%m%d_%H%M%S).log"
python3 -u experiments/step4_stats_and_errors.py 2>&1 | tee "$S4LOG"
S4_EXIT=${PIPESTATUS[0]}
log "Step4 exit=$S4_EXIT  log=$S4LOG"

# ── STEP 5 ──
log "====== BẮT ĐẦU STEP 5 ======"
S5LOG="logs/direct_step5_$(date +%Y%m%d_%H%M%S).log"
python3 -u experiments/step5_reward_ablation.py 2>&1 | tee "$S5LOG" | \
    grep -v "Loading weights\|Fetching\|torchao._C\|Warning\|flash\|causal\|unauthenticated\|it/s\|[█▏▎▍▌▋▊▉]\|Generating train"
S5_EXIT=${PIPESTATUS[0]}
log "Step5 exit=$S5_EXIT  log=$S5LOG"

log "====== TẤT CẢ XONG ======"
