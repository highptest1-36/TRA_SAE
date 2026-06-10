# Phase A — trạng thái & cách tiếp tục (cập nhật 2026-06-09 ~04:20)

> Đọc file này khi quay lại. cfg0–cfg4 ĐÃ LƯU AN TOÀN. Chỉ cfg5 có thể cần chạy lại.

## ✅ Trạng thái Phase A (canonical eval, 217 mẫu)
| Config | Overall | Physics | Logic | Trạng thái |
|---|---|---|---|---|
| cfg0 zero-shot | 36.41% | 45.39% | 19.74% | ✅ đã lưu |
| cfg1 +SFT | 48.85% | 63.12% | 22.37% | ✅ đã lưu |
| cfg2 +Logic SFT | 52.07% | 65.25% | 27.63% | ✅ đã lưu |
| **cfg3 +GRPO mixed** | **54.84%** ⭐ | 69.50% | 27.63% | ✅ đã lưu (TỐT NHẤT) |
| cfg4 Dual LoRA+Router | 50.69% | 62.41% | 28.95% | ✅ đã lưu |
| cfg5 Self-consistency 5× | ? | ? | ? | ⏳ đang chạy lúc rời máy (chưa lưu) |

- File chính: `logs/qwen35_ablation_canonical.json` (+ `logs/ablation_per_sample_canonical.jsonl`).
- Backup cfg0–4: `logs/_backup_phaseA/` (phòng hỏng).
- Log chạy: `logs/step0_canonical_PHASEA_20260608_174121.log`.
- **Lưu ý quan trọng:** cfg0–4 được chấm CÓ z3-solver. Nếu chạy lại cfg5 mà THIẾU z3 → số logic cfg5 sẽ lệch (không nhất quán). PHẢI cài z3 trước khi resume.

## 🔁 Khi quay lại — 2 tình huống

### A) Colab session VẪN SỐNG (VM chưa bị thu hồi)
```bash
tail -40 logs/step0_canonical_PHASEA_20260608_174121.log
```
- Thấy `cfg5 DONE ...` + `SANITY CHECK` + `ALL DONE` → **Phase A xong cả 6 config.** Sang mục "Sau Phase A".
- Còn đang chạy (có dòng `attempt ...`) → để chạy nốt (~còn ≤2h từ lúc rời máy).

### B) Session MỚI (VM bị thu hồi — phải setup lại)
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/TRA-SAE
!pip -q install "transformers>=4.46" peft trl datasets accelerate scipy z3-solver openai
!pip -q uninstall -y torchao          # ⭐ BẮT BUỘC, nếu không LoRA chết
!python -c "import z3; print('z3 OK')" # ⭐ phải có z3 để cfg5 nhất quán cfg0-4
```
Rồi resume (tự bỏ qua cfg0–4 đã xong, chỉ chạy cfg5 ~5h):
```bash
!python experiments/step0_canonical_eval.py
```

## ▶️ Sau khi Phase A xong cả 6 config
1. **Multi-seed cho config tốt nhất (cfg3)** — ~3h:
   ```bash
   !python experiments/step0_canonical_eval.py --seed 1337 --config 3
   !python experiments/step0_canonical_eval.py --seed 2024 --config 3
   ```
2. **Phase B** (đặt key trước; B2–B6 đã có sẵn n=217, chỉ chạy thêm B7+B8):
   ```python
   import os
   os.environ["HF_TOKEN"]       = "hf_..."   # cho B7 (Qwen2.5-7B-Instruct, gated)
   os.environ["GEMINI_API_KEY"] = "..."      # cho B8 (Gemini — đã thay GPT-4o-mini)
   # tùy chọn: os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
   ```
   ```bash
   !HF_TOKEN=$HF_TOKEN GEMINI_API_KEY=$GEMINI_API_KEY python experiments/step3_baselines.py
   !HF_TOKEN=$HF_TOKEN python experiments/step8_fair_baselines.py
   !python experiments/step9_external_benchmark.py
   !python experiments/step13_tool_baselines.py
   ```
3. **Phase C** (CPU): `step4_stats_and_errors.py`, `step14_manual_error_annotation.py`.
4. **Phase D**: `check_consistency.py` → phải in "ALL CONSISTENCY CHECKS PASSED". CHECK 4 liệt kê số cần cập nhật vào `paper/TRA_SAE_EXACT2026_paper.tex`.

## Nhắn Claude "tiếp tục theo PHASE_A_STATUS_VA_RESUME.md" là tôi chạy đúng kế hoạch.
