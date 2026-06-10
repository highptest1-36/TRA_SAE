#!/usr/bin/env bash
# compile_paper.sh — Biên dịch bản Expert Systems ra PDF (pdflatex -> bibtex -> pdflatex x2).
# Yêu cầu: đã chạy setup_texlive.sh trước (cần pdflatex/bibtex).
#   bash paper/bin/compile_paper.sh
# Kết quả: paper/TRA_SAE_Wiley.pdf
set -e
DIR="/content/drive/MyDrive/TRA-SAE/paper/format/WileyDesign/Optimal-Design-layout"
OUT="/content/drive/MyDrive/TRA-SAE/paper/TRA_SAE_Wiley.pdf"
cd "$DIR"

if ! command -v pdflatex >/dev/null; then
  echo "!! Chưa có pdflatex. Chạy trước: bash paper/bin/setup_texlive.sh"; exit 1
fi

echo "[1/4] pdflatex pass 1..."; pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex >/tmp/c1.log 2>&1 || true
echo "[2/4] bibtex...";          bibtex TRA_SAE_Wiley            >/tmp/cb.log 2>&1 || true
echo "[3/4] pdflatex pass 2..."; pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex >/tmp/c2.log 2>&1 || true
echo "[4/4] pdflatex pass 3..."; pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex >/tmp/c3.log 2>&1 || true

echo "--- bibtex warnings/errors ---"; grep -iE "warning|error|illegal" /tmp/cb.log || echo "  none"
echo "--- undefined citations ---";    grep -c "Citation .* undefined" /tmp/c3.log || echo 0

if [ -f TRA_SAE_Wiley.pdf ]; then
  cp TRA_SAE_Wiley.pdf "$OUT"
  echo "OK -> $OUT ($(pdfinfo TRA_SAE_Wiley.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
  echo "Xem PDF tại: paper/TRA_SAE_Wiley.pdf  (mở từ Drive)"
else
  echo "!! KHONG tao duoc PDF. Xem log: /tmp/c1.log /tmp/c3.log"; exit 1
fi
