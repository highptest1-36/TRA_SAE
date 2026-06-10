# Kế hoạch tiếp tục — TRA-SAE EXACT2026 (cập nhật 2026-06-07, 20:35)

> Đọc file này đầu tiên khi mở Colab ngày mai. Mọi lệnh đều copy-paste chạy được.
> Tham chiếu kế hoạch gốc: `EXACT2026_RUN_GUIDE.md`.

---

## ✅ ĐÃ LÀM HÔM NAY

1. **Cài deps** (transformers 5.9.0, peft 0.19.1, trl, z3-solver, openai…).
2. **Sửa lỗi môi trường quan trọng**: `pip install` trên Colab kéo peft mới + còn `torchao 0.10.0` cũ → peft báo `ImportError` khi load LoRA (cfg1–cfg5 chết).
   **Đã fix bằng `pip uninstall -y torchao`** (pipeline BF16 không cần torchao). → CẦN LÀM LẠI MỖI SESSION COLAB MỚI (xem Bước 0).
3. **Smoke test step0 (6 config) CHẠY THÀNH CÔNG, EXIT=0.** Kết quả smoke n=10 (chỉ để kiểm tra chạy được, KHÔNG có ý nghĩa số liệu):
   - cfg0=30% · cfg1=40% · cfg2=50% · cfg3=50% · cfg4=50% · cfg5=40%
   - SANITY CHECK 7+3=10 cho cả 6 config: OK.
4. **step13 smoke chạy gần xong** (Arm A xong, Arm B dở thì dừng). step3/step8/step9/check_consistency **CHƯA smoke**.
5. **Đã xóa file smoke** (`qwen35_ablation_canonical*.json`, `ablation_per_sample_canonical.jsonl`, tool baseline smoke) để full run mai không bị resume nhầm dữ liệu n=10.

## ⚠️ VẤN ĐỀ CHƯA GIẢI QUYẾT (làm trước khi chạy full)

**Nghi ngờ lỗi "chồng adapter" trong step0.** cfg2=cfg3=cfg4 ra **đúng 50%** giống hệt nhau (đáng ngờ). Nguyên nhân có thể: step0 dùng chung 1 `base_model`, khi `PeftModel.from_pretrained(base_model, cfg1)` rồi `del model` thì lớp LoRA cfg1 **vẫn dính trong base_model**; cfg2/cfg3 nạp tiếp lên → có cảnh báo:
```
UserWarning: Already found a `peft_config` attribute in the model.
This will lead to having multiple adapters in the model.
```
→ cfg2, cfg3 có thể bị sai số. cfg0 (chạy đầu, base sạch) và cfg4/cfg5 (dùng `load_multi_adapter_model` load base MỚI) thì AN TOÀN.

**Đã viết sẵn test kiểm chứng:** `experiments/_adapter_contamination_test.py` (so output cfg2 trên base-sạch vs base-bẩn). Phải chạy test này TRƯỚC, rồi mới quyết định.

## ⏱️ LƯU Ý THỜI GIAN
Qwen3.5-4B dùng linear-attention, kernel nhanh (FLA) không có → chạy bằng torch thuần, **chậm**. Đo từ smoke: cfg5 self-consistency mất 16.5 phút cho 10 mẫu.
→ **Full Phase A (217 mẫu) ước tính 15–20h, KHÔNG phải 9–10h như guide.** Step0 resume được theo từng config nếu Colab rớt (chạy lại đúng lệnh cũ là tiếp tục).

---

## 🚀 KẾ HOẠCH NGÀY MAI — chạy lần lượt

### BƯỚC 0 — Setup session mới (BẮT BUỘC mỗi lần mở Colab)
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/TRA-SAE
!nvidia-smi --query-gpu=name,memory.total --format=csv   # xác nhận A100
!pip -q install "transformers>=4.46" peft trl datasets accelerate scipy z3-solver openai
!pip -q uninstall -y torchao            # ⭐ FIX BẮT BUỘC — nếu không, LoRA chết
```
```python
import os
os.environ["HF_TOKEN"]       = "hf_xxx"   # cho B5/B7 (model gated). Không có → bỏ B7, dùng --only-ids B2 B3 B4 B5 B6
os.environ["OPENAI_API_KEY"] = "sk-xxx"   # chỉ cho B8 GPT-4o-mini (tùy chọn)
```
Pre-flight (phải thấy đủ 5 đường dẫn):
```bash
!ls -d checkpoints/qwen35_sft/final checkpoints/qwen35_sft_logic/final \
       checkpoints/qwen35_grpo/final checkpoints/qwen35_grpo_physics/final \
       checkpoints/qwen35_grpo_logic/final
```

### BƯỚC 1 — Kiểm chứng lỗi chồng adapter (~5 phút) ⭐ LÀM ĐẦU TIÊN
```bash
!python experiments/_adapter_contamination_test.py
```
- Nếu in **"NO contamination — step0 reuse is SAFE"** → bỏ qua Bước 2, sang Bước 3.
- Nếu in **"CONTAMINATION CONFIRMED"** → làm Bước 2 (sửa step0) rồi mới chạy full.

### BƯỚC 2 — (CHỈ KHI test xác nhận lỗi) Sửa step0
Sửa nhánh single-LoRA trong `experiments/step0_canonical_eval.py` (~dòng 259-260) để mỗi config nạp base MỚI sạch thay vì dùng chung `base_model`. (Claude sẽ sửa giúp khi có kết quả test — báo "có contamination, sửa step0 đi".)

### BƯỚC 3 — Dọn lại file canonical cho chắc (đề phòng smoke còn sót)
```bash
!rm -f logs/qwen35_ablation_canonical*.json logs/ablation_per_sample_canonical.jsonl
```

### BƯỚC 4 — (tùy chọn) Smoke nốt 3 script chưa test (~10 phút)
```bash
!python experiments/step3_baselines.py  --smoke-test --only-ids B2
!python experiments/step8_fair_baselines.py --smoke-test --only-ids B2 --only-strategies cot_boxed
!python experiments/step9_external_benchmark.py --smoke-test --configs cfg0
!python experiments/check_consistency.py   # mong đợi: sanity OK (smoke)
```
Sau đó XÓA lại file smoke trước khi full: `!rm -f logs/qwen35_ablation_canonical*.json logs/ablation_per_sample_canonical.jsonl logs/*_latest.json` (cẩn thận: chỉ xóa nếu là smoke).

### BƯỚC 5 — PHASE A: Canonical eval (BLOCKING, ~15–20h, resume được)
```bash
!python experiments/step0_canonical_eval.py
```
Rớt mạng → chạy lại đúng lệnh trên, nó tự bỏ qua config đã xong.
Multi-seed cho config tốt nhất (đọc config tốt nhất từ kết quả trên, thường cfg2 hoặc cfg3):
```bash
!python experiments/step0_canonical_eval.py --seed 1337 --config 2
!python experiments/step0_canonical_eval.py --seed 2024 --config 2
```

### BƯỚC 6 — PHASE B: Baselines & external (~6–7h, chạy thứ tự nào cũng được)
```bash
!HF_TOKEN=$HF_TOKEN OPENAI_API_KEY=$OPENAI_API_KEY python experiments/step3_baselines.py
!HF_TOKEN=$HF_TOKEN python experiments/step8_fair_baselines.py
!python experiments/step9_external_benchmark.py
!python experiments/step13_tool_baselines.py
```

### BƯỚC 7 — PHASE C: Stats & error (CPU, nhanh, miễn phí)
```bash
!python experiments/step4_stats_and_errors.py
!python experiments/step14_manual_error_annotation.py     # tạo template
# → điền cột manual_label trong logs/manual_error_template.csv, lưu thành logs/manual_error_filled.csv
!python experiments/step14_manual_error_annotation.py --score
```

### BƯỚC 8 — PHASE D: Cổng nhất quán (trước khi sửa paper)
```bash
!python experiments/check_consistency.py    # phải in "ALL CONSISTENCY CHECKS PASSED"
```
CHECK 4 sẽ liệt kê config nào đổi số so với V1 cũ → đó là danh sách số cần cập nhật trong `paper/TRA_SAE_EXACT2026_paper.tex`.

---

## GHI CHÚ
- Mọi output ghi vào `logs/*.json` — paper dựng lại từ đó.
- KHÔNG sửa tay `logs/qwen35_ablation.json` (file V1 cũ, chỉ để so drift ở CHECK 4).
- Khi Claude chạy nền + ghi log file: muốn xem realtime trên terminal Colab thì `!tail -f logs/<tên>.log`.
