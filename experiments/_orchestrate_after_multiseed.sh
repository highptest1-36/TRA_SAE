#!/usr/bin/env bash
# Orchestrator: chạy nốt Phase B + Phase C sau khi multi-seed cfg3 xong.
# Mỗi bước log riêng, lỗi-thì-bỏ-qua (continue-on-error), cập nhật file tiến độ.
set -u
cd /content/drive/MyDrive/TRA-SAE

MULTISEED_LOG="${1:?cần truyền đường dẫn log multi-seed}"
STATUS="logs/_ORCH_STATUS.txt"
TS() { date +%Y%m%d_%H%M%S; }
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

# Nạp key từ .env và set model Gemini
set -a; [ -f .env ] && source .env; set +a
export GEMINI_MODEL="gemini-2.5-flash-lite"

echo "================ ORCHESTRATOR START $(date '+%F %T') ================" > "$STATUS"
mark "Đợi multi-seed xong (marker 'MULTISEED ALL DONE' trong $MULTISEED_LOG)"

# --- Đợi multi-seed (poll marker, tối đa ~5h) ---
WAITED=0; MAXWAIT=$((5*3600))
while ! grep -q "MULTISEED ALL DONE" "$MULTISEED_LOG" 2>/dev/null; do
  sleep 60; WAITED=$((WAITED+60))
  if [ "$WAITED" -ge "$MAXWAIT" ]; then
    mark "CẢNH BÁO: quá $((MAXWAIT/3600))h chưa thấy multi-seed xong — vẫn tiếp tục Phase B."
    break
  fi
done
mark "Multi-seed xong (hoặc timeout). Bắt đầu Phase B."
for f in logs/qwen35_ablation_canonical_seed1337.json logs/qwen35_ablation_canonical_seed2024.json; do
  [ -f "$f" ] && mark "  OK có $f" || mark "  THIẾU $f (kiểm tra log multi-seed)"
done

# --- Preflight deps (im lặng) ---
mark "Preflight: đảm bảo openai + datasets + scipy có"
pip -q install openai datasets scipy >/dev/null 2>&1 || mark "  (pip preflight có lỗi, vẫn tiếp tục)"

run_step() {
  # run_step "<nhãn>" "<logfile>" <command...>
  local label="$1"; local log="$2"; shift 2
  mark ">>> BẮT ĐẦU $label  (log: $log)"
  if "$@" > "$log" 2>&1; then
    mark "<<< XONG    $label"
  else
    mark "<<< LỖI ($?) $label — xem $log (tiếp tục bước sau)"
  fi
}

T=$(TS)
# B8: Gemini 2.5 flash-lite (chỉ B8, tự skip B2-B6 đã xong)
run_step "B8 Gemini(flash-lite)" "logs/orch_B8_${T}.log" \
  python experiments/step3_baselines.py --only-ids B8

# B7: fair baseline (Qwen2.5-7B-Instruct, gated → cần HF_TOKEN)
run_step "B7 fair baseline"      "logs/orch_B7_${T}.log" \
  python experiments/step8_fair_baselines.py

# step9: external benchmark (MMLU physics, cfg0/cfg2/cfg3)
run_step "step9 external bench"  "logs/orch_step9_${T}.log" \
  python experiments/step9_external_benchmark.py

# step13: tool baselines FULL n=217 (không --smoke-test)
run_step "step13 tool baselines" "logs/orch_step13_${T}.log" \
  python experiments/step13_tool_baselines.py

# Phase C: stats (sau B8 để baselines file có B8) + tạo template annotation
run_step "step4 stats+errors"    "logs/orch_step4_${T}.log" \
  python experiments/step4_stats_and_errors.py

run_step "step14 template"       "logs/orch_step14_${T}.log" \
  python experiments/step14_manual_error_annotation.py

mark "================ ORCHESTRATOR DONE $(date '+%F %T') ================"
mark "Tiếp theo (cần người): điền logs/manual_error_template.csv rồi step14 --score; và Phase D cập nhật paper."
