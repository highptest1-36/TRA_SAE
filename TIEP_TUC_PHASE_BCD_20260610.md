# 🔄 TIẾP TỤC — TRA-SAE Phase B/C/D (cập nhật 2026-06-10 ~03:32)

> ✅✅ **PIPELINE TỰ ĐỘNG ĐÃ XONG HẾT (Phase A→D), check_consistency PASSED.**
> Còn lại 2 việc TAY: (1) điền `logs/manual_error_template.csv` → step14 --score;
> (2) cập nhật số vào `paper/TRA_SAE_EXACT2026_paper.tex` (xem mục 6).
>
> Mọi thứ nằm trên Google Drive nên CODE + KẾT QUẢ đều an toàn qua disconnect.
> (Mục 0/3 bên dưới chỉ cần nếu muốn chạy lại phần nào.)

---

## 0️⃣ SETUP SAU KHI DISCONNECT (bắt buộc mỗi session mới)

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/TRA-SAE
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install "transformers>=4.46" peft trl datasets accelerate scipy z3-solver openai
!pip -q uninstall -y torchao          # ⭐ BẮT BUỘC — nếu không, LoRA cfg2/cfg3 chết
!python -c "import z3; print('z3', z3.get_version_string())"   # ⭐ phải có z3 (logic + step13)
```
```python
import os
# Key đã lưu sẵn trong .env (HF_TOKEN, GEMINI_API_KEY). Nạp:
# bash: set -a; source .env; set +a; export GEMINI_MODEL=gemini-2.5-flash-lite
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash-lite")
```
> Khi chạy script qua `!`, nhớ nạp key:
> `!set -a; source .env; set +a; export GEMINI_MODEL=gemini-2.5-flash-lite; python experiments/stepXX.py`

---

## 1️⃣ TRẠNG THÁI TỔNG (✅ xong / 🔄 đang chạy / ⏳ chờ)

| Phần | Trạng thái | File output | Ghi chú |
|---|---|---|---|
| Phase A canonical (6 cfg) | ✅ | `logs/qwen35_ablation_canonical.json` | cfg3 best 54.84% |
| Multi-seed cfg3 (1337, 2024) | ✅ | `logs/qwen35_ablation_canonical_seed{1337,2024}.json` | 3 seed |
| step3 baselines B2–B6 | ✅ | `logs/baselines_results_latest.json` | |
| **B8 Gemini flash-lite** | ✅ | (trong baselines_results_latest) | 24.88% |
| **Fair baseline (step8, đã FIX)** | ✅ | `logs/fair_baselines_results_latest.json` | 15 cặp, xong 00:01 |
| **step9 MMLU (đã FIX)** | ✅ xong 02:03 | `logs/external_benchmark_results_latest.json` | cfg0=42.29 cfg2=47.04 **cfg3=48.62** → GENERALIZE ✅ |
| **step13 tool baselines** | ✅ xong 03:30 | `logs/tool_baselines_results_latest.json` | Arm A 39.63 / Arm B 40.09 (+0.46pp) |
| step4 stats+errors | 🔄 đang chạy (CPU) | `logs/stats_results.json`, `error_analysis.json` | ghi đè file cũ 29/5 |
| step14 manual annotation | ⏳ | `logs/manual_error_*.csv/json` (CHƯA) | cần điền tay 50 mẫu |
| check_consistency | ⏳ | — | cổng trước khi sửa paper |
| Cập nhật số vào paper .tex | ⏳ | `paper/TRA_SAE_EXACT2026_paper.tex` | |

---

## 2️⃣ BUG ĐÃ TÌM & SỬA HÔM NAY (đừng để tái phát)

**Bug chung: `max_new_tokens=512` cắt model trước khi ra đáp số → extract vớ ký tự ngẫu nhiên → điểm dưới random.**

- **step8_fair_baselines.py** (đã sửa, file trên Drive):
  - `max_new_tokens 512 → 1024`; input `max_length 1024 → 2048`.
  - Kết quả: Qwen2.5-Math 15→25%, bảng fair-baseline giờ hợp lệ.
- **step9_external_benchmark.py** (đã sửa, file trên Drive):
  - `max_new_tokens 512 → 1024`.
  - Extraction `<answer>` dùng `\b[ABCD]\b` (chống vớ chữ trong "CORRECT"→C).
  - `BATCH 4 → 16` (tăng tốc 4x, an toàn vì left-padding).
  - Smoke cfg0: 21% → **45%** (đúng kỳ vọng REVISION_PLAN 40-45%).
- **step13**: đã kiểm tra — KHÔNG dính bug (dùng `MAX_NEW_TOKENS` config=1024 + left-pad + verifier chung). Chạy thẳng.
- Backup dữ liệu buggy: `logs/_backup_fair_baselines_partial_buggy512.json`, `logs/_backup_external_buggy512.json`.

---

## 3️⃣ NẾU DISCONNECT — CHẠY LẠI PHẦN DỞ (đúng thứ tự)

> step9/step13 **KHÔNG resume giữa chừng** (ghi đè file từ đầu) → chạy lại nguyên lệnh là được.
> Các phần ✅ ở mục 1 KHÔNG cần chạy lại.

```bash
# (A) step9 MMLU — nếu chưa thấy 3 config trong external_benchmark_results_latest.json (smoke=False, n=253)
python experiments/step9_external_benchmark.py            # ~1h (batch16), cfg0/cfg2/cfg3

# (B) step13 tool baselines (cần z3!) — ~1.5h
python experiments/step13_tool_baselines.py

# (C) Phase C — CPU, nhanh, miễn phí
python experiments/step4_stats_and_errors.py             # ghi đè stats_results.json + error_analysis.json (file 29/5 là CŨ)
python experiments/step14_manual_error_annotation.py     # tạo logs/manual_error_template.csv
#   → điền cột manual_label (FS/UC/AR/PU/LE/AX) → lưu logs/manual_error_filled.csv
python experiments/step14_manual_error_annotation.py --score

# (D) Cổng nhất quán — phải in "ALL CONSISTENCY CHECKS PASSED"
python experiments/check_consistency.py
#   CHECK 4 liệt kê số cần cập nhật vào paper/TRA_SAE_EXACT2026_paper.tex
```

**Kiểm tra nhanh step9 đã xong chưa:**
```bash
python -c "import json;d=json.load(open('logs/external_benchmark_results_latest.json'));print('smoke=',d.get('smoke'),'n=',d.get('n_samples'))"
# Cần: smoke=False, n=253. Nếu smoke=True/n=20 → step9 chưa chạy full, chạy lại lệnh (A).
```

---

## 4️⃣ KẾT QUẢ CHÍNH (lưu lại phòng mất)

**Phase A canonical (seed 42), n=217 (141 physics + 76 logic):**
| cfg | tên | overall | physics | logic |
|---|---|---|---|---|
| 0 | zero-shot | 36.41 | 45.39 | 19.74 |
| 1 | +SFT | 48.85 | 63.12 | 22.37 |
| 2 | +Logic SFT | 52.07 | 65.25 | 27.63 |
| **3** | **+GRPO mixed** | **54.84** ⭐ | 69.50 | 27.63 |
| 4 | Dual LoRA+Router | 50.69 | 62.41 | 28.95 |
| 5 | Self-consistency 5× | 52.53 | 65.25 | 28.95 |

**Multi-seed cfg3:** 54.84 (s42) / 51.15 (s1337) / 52.07 (s2024) → **mean ≈ 52.69, std ≈ 1.93**.

**Baselines zero-shot (step3):** B2 Qwen2.5-Math 28.57 · B3 Llemma 12.90 · B4 Mistral 9.68 · B5 Qwen2-Math 26.73 · B6 DeepSeek-Math 11.98 · **B8 Gemini-2.5-flash-lite 24.88**.

**Fair baseline (step8 FIXED) — best strategy/model:** Qwen2.5-Math 24.88 (xml) · Qwen2-Math **29.03** (xml) · Mistral 16.59 (cot_plain) · DeepSeek-Math 12.44 (cot_boxed) · Llemma 11.52 (xml).
→ Tất cả baseline 7B đều **< cfg3 (54.84%)**. Prompt native KHÔNG giúp đa số model (chỉ Mistral +7pp). Luận điểm "model 4B của ta vượt 7B" vững.

**step9 MMLU (FIXED, n=253) — XONG:** cfg0 **42.29** (hs 41.06 / col 44.12) · cfg2 **47.04** (47.68 / 46.08) · cfg3 **48.62** (46.36 / 51.96).
→ **Fine-tune trên EXACT GENERALIZE sang MMLU physics** (cfg3 > cfg2 > cfg0, +6.33pp so zero-shot). Gỡ giới hạn "single-benchmark". (Bản buggy 512-token ra ~11-21% random → kết luận sai; fix kịp.)

**step13 tool-assisted (cfg3, n=217):** Arm A LLM-direct 39.63% (phys 49.65 / logic 21.05) · Arm B LLM+tools 40.09% (51.06 / 19.74). **Δ=+0.46pp** (tool: calc 62 / none 155; z3 logic hầu hết abstain). → tool-augment gain marginal.

**step4 McNemar (CHECK PASSED):** cfg0→cfg3 p≈0.0 (Δ18.43pp, sig) · cfg0→cfg1 SFT p=3.1e-5 (sig) · cfg1→cfg2 logic-SFT p=0.096 (**KHÔNG sig** — giải thích regression logic SFT) · best=cfg3.
**Error (cfg3, 98 sai/217):** rule-based phân loại yếu (đa số E0_unclassified) → cần annotation tay (step14).

---

## 6️⃣ CẬP NHẬT PAPER (`paper/TRA_SAE_EXACT2026_paper.tex`)

**(a) Sửa 6 số bảng ablation chính (V1 cũ → canonical mới) — từ CHECK 4:**
| cfg | V1 cũ | → canonical mới |
|---|---|---|
| cfg0 | 35.48 | **36.41** |
| cfg1 | 52.53 | **48.85** (Δ-3.68, đổi nhiều) |
| cfg2 | 51.61 | **52.07** |
| cfg3 | 53.92 | **54.84** (best) |
| cfg4 | 49.77 | **50.69** |
| cfg5 | 50.69 | **52.53** (Δ+1.84) |

**(b) Thêm/ghi đè 3 bảng mới:**
- Bảng **fair-baseline** (từ `fair_baselines_results_latest.json`): Qwen2-Math 29.03 / Qwen2.5-Math 24.88 / Mistral 16.59 / DeepSeek-Math 12.44 / Llemma 11.52 — tất cả < cfg3 54.84.
- Bảng **MMLU generalization** (từ `external_benchmark_results_latest.json`): cfg0 42.29 / cfg2 47.04 / cfg3 48.62.
- Bảng **tool-assisted** (từ `tool_baselines_results_latest.json`): Arm A 39.63 / Arm B 40.09.
- Cập nhật B8 baseline = Gemini-2.5-flash-lite 24.88 (thay GPT-4o-mini).

**(c) Narrative:** dùng McNemar cfg1→cfg2 p=0.096 để giải thích logic-SFT regression; nhấn fair-eval + MMLU-generalize là đóng góp mới.

---

## 5️⃣ THAM CHIẾU
- Kế hoạch gốc Phase B/C/D: `EXACT2026_RUN_GUIDE.md`
- Map output→bảng paper: mục 6 của `EXACT2026_RUN_GUIDE.md`
- Plan revision (reviewer): `REVISION_PLAN.md`
- KHÔNG sửa tay `logs/qwen35_ablation.json` (file V1 cũ, chỉ để CHECK 4 so drift).
- Nhắn Claude **"tiếp tục theo TIEP_TUC_PHASE_BCD_20260610.md"** là chạy đúng kế hoạch này.
