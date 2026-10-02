# Kế hoạch Revision — TRA-SAE → CMC Submission
> Cập nhật: 2026-06-03 | Phản hồi reviewer Q1

## Phân tích vấn đề & Giải pháp

| # | Vấn đề reviewer | Giải pháp | Script | Chi phí |
|---|----------------|-----------|--------|---------|
| A | Baseline không công bằng (XML penalty) | Re-eval với model-native prompt + flexible extractor | `step8_fair_baselines.py` | ~$4 |
| B | Không có external benchmark | MMLU physics (253 MCQ) zero-shot và cfg3 | `step9_external_benchmark.py` | ~$2 |
| C | GRPO chưa chứng minh ở budget dài | Extend R1/R4 lên 500 steps, vẽ convergence curve | `step10_grpo_extended.py` | ~$16 |
| D | Logic SFT regression chưa giải thích | Per-sample analysis cfg1→cfg2 + re-run nhẹ hơn | `step11_logic_sft_analysis.py` | ~$4 |
| E | Error taxonomy yếu | Rule-based re-annotation với protocol rõ | `step12_error_reannotation.py` | $0 |

**Tổng chi phí ước tính: ~$26 | Thời gian: ~14 giờ GPU**

---

## Chi tiết từng thực nghiệm

### EXP-A: Fair Baseline Re-evaluation (`step8_fair_baselines.py`)

**Vấn đề gốc:** Mistral (9.68%), Llemma (12.90%), DeepSeek-Math (11.98%) bị điểm thấp
do XML non-compliance, không phải do reasoning yếu. Đây là điểm reviewer sẽ attack mạnh nhất.

**Giải pháp:**
- Chạy 3 biến thể prompt cho mỗi model:
  - `xml`: prompt gốc (yêu cầu XML tags)
  - `cot_boxed`: CoT + `\boxed{answer}` (Qwen-Math, DeepSeek-Math native)
  - `cot_plain`: CoT + "The answer is:" (Mistral, Llemma)
- Dùng unified flexible extractor: XML → \boxed{} → "Answer:" → last line
- Report accuracy theo từng extraction method
- **Kết quả kỳ vọng:** Qwen2.5-Math sẽ tăng lên ~45-55% với prompt phù hợp

**Tác động lên paper:** Thay vì "outperform all 7B baselines", ta có thể nói:
> "Under fair evaluation, Qwen2.5-Math-7B reaches X%; our fine-tuned 4B model achieves Y%,
> demonstrating competitive performance at significantly lower parameter count."

---

### EXP-B: External Benchmark — MMLU Physics (`step9_external_benchmark.py`)

**Vấn đề gốc:** Toàn bộ thực nghiệm trên EXACT 2026 split — không có generalization evidence.

**Dataset:** MMLU (`cais/mmlu`)
- `high_school_physics`: 151 test samples (MCQ, A/B/C/D)
- `college_physics`: 102 test samples (MCQ, A/B/C/D)
- Total: 253 samples

**Thực nghiệm:**
- Eval cfg0 (zero-shot Qwen3.5-4B) on MMLU physics
- Eval cfg3 (best fine-tuned model) on MMLU physics
- Không cần fine-tune lại — chỉ inference
- **Câu hỏi khoa học:** Việc fine-tune trên EXACT 2026 có generalize sang MMLU hay không?

**Kết quả kỳ vọng:**
- cfg0 ~40-45% (MMLU physics MCQ tương đối dễ với LLM)
- cfg3 ~45-55% (nếu generalize) hoặc tương đương cfg0 (nếu overfit EXACT)
- Cả hai kết quả đều có giá trị: generalization → tốt; hoặc domain gap → honest negative finding

**Chi phí:** ~$2 (1h inference trên 253 samples × 2 configs)

---

### EXP-C: Extended GRPO Training (`step10_grpo_extended.py`)

**Vấn đề gốc:** R1 vs R4 chỉ so sánh ở 150 steps. Reviewer hỏi "có thể reward shaping
cần nhiều steps hơn?" — nhưng ta không có data để trả lời.

**Thực nghiệm:**
- Chạy R1 (full reward) và R4 (correctness-only) cho 500 steps
- Checkpoint và evaluate tại: 150, 250, 350, 500 steps
- Plot accuracy curve theo số steps
- **Câu hỏi khoa học:** Tại budget dài hơn, full reward (R1) có vượt correctness-only (R4)?

**Kết quả kỳ vọng (3 scenarios):**
1. R1 vượt R4 ở ~350-500 steps → **Confirms hypothesis**: auxiliary rewards cần budget dài
2. R4 vẫn dẫn ở 500 steps → **Stronger finding**: correctness-only is universally better
3. Cả hai plateau → **Finding**: 250 steps là đủ cho 4B model size

**Tất cả 3 scenario đều publishable!**

**Chi phí:** ~$16 (R1 500 steps ≈ 9h + R4 500 steps ≈ 9h, chạy tuần tự)

---

### EXP-D: Logic SFT Regression Analysis (`step11_logic_sft_analysis.py`)

**Vấn đề gốc:** cfg2 (Logic SFT) giảm 0.92pp so với cfg1 — tại sao lại giữ bước này trong
pipeline chính? Reviewer sẽ hỏi điều này.

**Thực nghiệm:**
- Part 1: Per-sample analysis từ ablation_per_sample_latest.jsonl
  - Tìm samples cfg1 đúng nhưng cfg2 sai (physics regression do logic SFT)
  - Quantify: bao nhiêu physics samples bị ảnh hưởng? Pattern gì?
- Part 2: Re-run Logic SFT với LR=5e-5 (half của 1e-4 hiện tại) — gentle, ít catastrophic forgetting
  - Eval ngay sau training
  - So sánh với cfg2 gốc

**Chi phí:** ~$4 (analysis miễn phí, 1 re-run = ~$1)

---

### EXP-E: Rule-Based Error Re-annotation (`step12_error_reannotation.py`)

**Vấn đề gốc:** Error taxonomy single annotator, không reproducible.

**Giải pháp không cần GPU:**
- Viết rule-based classifier cho error types dựa trên string patterns:
  - E4 (logical fallacy): subject=="logic" và ground_truth ≠ prediction
  - E1 (unit error): physics answer có unit mismatch
  - E2 (formula error): physics answer có giá trị trong range hợp lý nhưng unit sai
  - E0 (unclassified): còn lại
- Chạy trên toàn bộ 103 wrong predictions
- So sánh rule-based vs manual annotation → tính pseudo-IAA
- Nếu agreement > 80% → support reliability của manual annotation

---

## Thứ tự chạy khuyến nghị

```
Ngày 1 (ít GPU):  EXP-E (0h GPU, free) + EXP-D Part1 (0h GPU, free)
Ngày 1 (GPU):     EXP-B (1h, $2)
Ngày 2:           EXP-A (2.5h, $4) + EXP-D Part2 (1h, $1)
Ngày 3-4:         EXP-C (18h, $16) — chạy nền qua đêm
```

**Ghi chú:** EXP-C (extended GRPO) là tốn nhất nhưng KHÔNG blocking — có thể viết paper
trước, thêm kết quả sau khi chạy xong.

---

## Tác động lên paper sau revision

### Thay đổi bắt buộc trong paper:
1. **Table 3 (Main Results):** Thêm cột "Qwen2.5-Math-7B (fair eval)" với số liệu từ EXP-A
2. **Table mới: MMLU Generalization** (EXP-B) — 4 dòng: cfg0/cfg3 × HS/College physics
3. **Figure mới: GRPO Convergence Curve** (EXP-C) — acc vs steps cho R1/R4/500 steps
4. **Section 4.3:** Giải thích Logic SFT regression dựa trên EXP-D analysis
5. **Section 5.6 (Error Analysis):** Thêm rule-based agreement % từ EXP-E

### Thay đổi narrative quan trọng:
- Reframe title/abstract: **"empirical study"** thay vì "proposed method"
- Xóa/soften claim "outperform 7B baselines"
- Thêm contribution mới: **"fair multi-prompt baseline evaluation framework"** từ EXP-A
- Thêm **"first study fine-tuning small LLM on EXACT + evaluating on MMLU"** từ EXP-B
