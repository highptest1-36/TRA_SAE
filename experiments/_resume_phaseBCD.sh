#!/usr/bin/env bash
# ============================================================================
# AUTO-RESUME Phase B/C/D — TRA-SAE  (đi kèm TIEP_TUC_PHASE_BCD_20260610.md)
# Chạy nốt: step9 (MMLU) → step13 (tool) → step4 (stats) → step14 (template)
#           → check_consistency.  Mỗi phần tự BỎ QUA nếu đã xong.
#
# DÙNG KHI NÀO: SAU khi Colab disconnect/reconnect, KHÔNG có job nào đang chạy.
#   bash experiments/_resume_phaseBCD.sh
# (Nếu đang có step9/step13 chạy nền thì script sẽ tự dừng để tránh trùng.)
# ============================================================================
set -u
cd /content/drive/MyDrive/TRA-SAE
STATUS="logs/_RESUME_BCD_STATUS.txt"
TS() { date +%Y%m%d_%H%M%S; }
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

echo "================ RESUME B/C/D START $(date '+%F %T') ================" > "$STATUS"

# --- Guard: không chạy nếu đã có tiến trình eval đang sống ---
if ps aux | grep -E "step9_external|step13_tool" | grep -v grep >/dev/null; then
  mark "DỪNG: đang có step9/step13 chạy nền. Đợi nó xong (hoặc kill) rồi chạy lại script."
  exit 1
fi

# --- Setup env ---
mark "Setup: nạp .env + GEMINI_MODEL + đảm bảo z3, gỡ torchao"
set -a; [ -f .env ] && source .env; set +a
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"
pip -q uninstall -y torchao >/dev/null 2>&1 || true
python -c "import z3" 2>/dev/null || pip -q install z3-solver >/dev/null 2>&1
python -c "import z3; print('z3 OK', z3.get_version_string())" || { mark "LỖI: thiếu z3"; exit 1; }

run_step() {  # run_step "<nhãn>" "<logfile>" <command...>
  local label="$1"; local log="$2"; shift 2
  mark ">>> BẮT ĐẦU $label  (log: $log)"
  if "$@" > "$log" 2>&1; then mark "<<< XONG    $label"
  else mark "<<< LỖI ($?) $label — xem $log (vẫn tiếp bước sau)"; fi
}

T=$(TS)

# --- (A) step9 MMLU — bỏ qua nếu đã có full (smoke=False, n=253) ---
S9_DONE=$(python -c "import json;d=json.load(open('logs/external_benchmark_results_latest.json'));print('Y' if (not d.get('smoke') and d.get('n_samples')==253) else 'N')" 2>/dev/null || echo N)
if [ "$S9_DONE" = "Y" ]; then mark "step9 ĐÃ xong (skip)."
else run_step "step9 MMLU"        "logs/resume_step9_${T}.log"  python experiments/step9_external_benchmark.py; fi

# --- (B) step13 tool — bỏ qua nếu đã có file (không smoke) ---
S13_DONE=$(python -c "import json,os;p='logs/tool_baselines_results_latest.json';print('Y' if os.path.exists(p) and not json.load(open(p)).get('smoke_test') else 'N')" 2>/dev/null || echo N)
if [ "$S13_DONE" = "Y" ]; then mark "step13 ĐÃ xong (skip)."
else run_step "step13 tool"       "logs/resume_step13_${T}.log" python experiments/step13_tool_baselines.py; fi

# --- (C) step4 stats+errors (luôn chạy lại — nhanh, ghi đè data cũ) ---
run_step "step4 stats+errors"     "logs/resume_step4_${T}.log"  python experiments/step4_stats_and_errors.py

# --- (D) step14 template (luôn tạo lại template; phần --score cần điền tay) ---
run_step "step14 template"        "logs/resume_step14_${T}.log" python experiments/step14_manual_error_annotation.py

# --- (E) check_consistency ---
run_step "check_consistency"      "logs/resume_consistency_${T}.log" python experiments/check_consistency.py

mark "================ RESUME B/C/D DONE $(date '+%F %T') ================"
mark "CẦN NGƯỜI: điền logs/manual_error_template.csv → step14 --score; rồi cập nhật paper theo CHECK 4."
echo ""; echo "Xem trạng thái: tail -f $STATUS"
