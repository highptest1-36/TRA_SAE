# EXACT 2026 — Kế hoạch viết paper Q3 (IEEE Access / Applied Intelligence)

Ngày bắt đầu: 2026-05-27
Mục tiêu submit: Applied Intelligence (Springer) hoặc Cognitive Computation
Target tier: Q2-Q3
Deadline tự đặt: 2026-07-07 (6 tuần)

---

## TỔNG QUAN LỖI HIỆN TẠI VÀ CÁCH SỬA

### BUG 1 — Bất nhất đếm mẫu cfg 0-3 [CRITICAL — reject vòng 1]
Vấn đề:
  - logs/qwen35_ablation.json, config 0-3: n_physics=146, n_logic=76, tổng=222 ≠ n_total=217
  - Config 4-5: n_physics=141, n_logic=76, tổng=217 = n_total (đúng)
  - Nguyên nhân: cfg 0-3 và cfg 4-5 chạy ở hai thời điểm khác nhau;
    dataset split thực tế có 141 physics + 76 logic = 217, nhưng lúc chạy cfg 0-3
    biến đếm n_physics được lấy từ label field thay vì đếm thực tế.
Cách sửa:
  - Chạy lại cfg 0-3 bằng script: experiments/step1_rerun_cfg0_3.py
  - Không thay đổi code eval; chỉ isolate bug đếm trong cfg_result builder
  - Kết quả mới ghi vào: logs/qwen35_ablation_v2.json

### BUG 2 — Single seed [CRITICAL — không có ± std]
Vấn đề:
  - Toàn bộ kết quả là 1 lần chạy duy nhất
  - Reviewer sẽ không chấp nhận bảng số liệu không có std
Cách sửa:
  - Chạy 3 seeds (42, 1337, 2024) chỉ cho cfg 3 (best config)
  - Script: experiments/step2_multiseed_cfg3.py
  - Báo mean ± std trong Table 2

### BUG 3 — Không có baseline ngoài [CRITICAL — không thể so sánh]
Vấn đề:
  - Chỉ có zero-shot của chính mình làm baseline
  - Không thể biết 53.92% là tốt hay tệ trong bức tranh chung
Cách sửa:
  - Chạy 6 baselines zero-shot trên cùng val set 217 mẫu
  - Script: experiments/step3_baselines.py
  - Danh sách 6 baselines (xem mục Baselines bên dưới)

### BUG 4 — Không có statistical test [CRITICAL]
Vấn đề:
  - Chưa chứng minh cải thiện có ý nghĩa thống kê
Cách sửa:
  - McNemar test: cfg3 vs cfg0, cfg1 vs cfg0
  - Script tích hợp trong: experiments/step4_stats_and_errors.py

### BUG 5 — Không có error analysis [IMPORTANT]
Vấn đề:
  - Không biết model sai ở đâu → không explain được tại sao
Cách sửa:
  - Phân loại 217 mẫu sai của cfg3 thành 5 nhóm
  - Script: experiments/step4_stats_and_errors.py

### BUG 6 — Không có ablation reward [IMPORTANT]
Vấn đề:
  - Unit-aware reward là novelty chính nhưng chưa đo tác động riêng
Cách sửa:
  - Rerun GRPO phase 2 với 4 biến thể reward
  - Script: experiments/step5_reward_ablation.py

---

## DANH SÁCH 6 BASELINES (mở rộng từ 3 lên 6)

| # | Model | Params | Lý do chọn | API/local | Script |
|---|---|---|---|---|---|
| B1 | Qwen3.5-4B zero-shot (ours) | 4B | Baseline nội bộ, cùng architecture | local | đã có |
| B2 | Qwen2.5-Math-7B zero-shot | 7B | Best open math model ~cùng size | local HF | step3 |
| B3 | Llemma-7B zero-shot | 7B | Open science/math LLM | local HF | step3 |
| B4 | Mistral-7B-Instruct-v0.3 zero-shot | 7B | General instruction model, popular baseline | local HF | step3 |
| B5 | InternLM2.5-Math-7B zero-shot | 7B | Strong Asian math model, cùng domain | local HF | step3 |
| B6 | GPT-4o-mini zero-shot | ~8B | Proprietary SOTA, context anchor | API | step3 |

Lý do chọn: 5 open + 1 proprietary; tất cả ≤ 8B để so sánh công bằng với Qwen3.5-4B.
Tất cả chạy zero-shot (không fine-tune) — để thấy rõ ưu điểm của pipeline training.

---

## 10 BƯỚC THỰC HIỆN CHI TIẾT

### TUẦN 1 (2026-05-27 → 2026-06-02)

#### BƯỚC 1: Fix bug + Rerun cfg 0-3 (1 ngày)
  Script: experiments/step1_rerun_cfg0_3.py
  Input: processed_data/exact_val (217 mẫu, 141 physics, 76 logic đã xác nhận)
  Output: logs/qwen35_ablation_v2.json (chỉ cfg 0-3)
  Kiểm tra: n_physics + n_logic == n_total cho tất cả configs
  Thời gian ước: ~5 giờ (cfg0=71min, cfg1=88min, cfg2=88min)

#### BƯỚC 2: Multi-seed cfg3 (2 ngày)
  Script: experiments/step2_multiseed_cfg3.py
  Seeds: [42, 1337, 2024]
  Output: logs/cfg3_multiseed_results.json
  Kết quả cần: mean ± std cho overall/physics/logic
  Thời gian ước: 3 × 86 min ≈ 4.3 giờ

### TUẦN 2 (2026-06-03 → 2026-06-09)

#### BƯỚC 3: Baseline inference 6 models (2-3 ngày)
  Script: experiments/step3_baselines.py
  Download models qua HuggingFace (cần ~28 GB cache):
    - Qwen/Qwen2.5-Math-7B-Instruct
    - EleutherAI/llemma_7b
    - mistralai/Mistral-7B-Instruct-v0.3
    - internlm/internlm2_5-math-7b-chat
    - GPT-4o-mini qua openai API
  Eval cùng format: prompt SYSTEM_PROMPT, same 217 val samples
  Output: logs/baselines_results.json
  Thời gian ước: ~6-8 giờ tổng (mỗi model ~1-1.5 giờ)

#### BƯỚC 4: Statistical test + Error analysis (1 ngày)
  Script: experiments/step4_stats_and_errors.py
  4a) McNemar test: cfg3 vs cfg0 (expect p < 0.001)
  4b) McNemar test: cfg3 vs mỗi baseline
  4c) Error analysis: đọc logs/qwen35_final_results.jsonl (217 mẫu cfg5)
      Phân loại 5 nhóm:
        E1: Unit/dimension error (physics)
        E2: Wrong formula selection (physics)
        E3: Numerical arithmetic error (physics/logic)
        E4: Logical fallacy / quantifier error (logic)
        E5: Question misinterpretation (both)
  Output: logs/stats_results.json, logs/error_analysis.json

### TUẦN 3 (2026-06-10 → 2026-06-16)

#### BƯỚC 5: Ablation reward components (2 ngày)
  Script: experiments/step5_reward_ablation.py
  Rerun Phase 2 GRPO với 4 biến thể reward:
    R1: full reward (format=0.30, correct=0.60, unit=0.10, len_pen=-0.10)
    R2: no unit reward (format=0.30, correct=0.70, no unit, no len_pen)
    R3: no format reward (correct=0.80, unit=0.10, no format, no len_pen)
    R4: only correctness (correct=1.0, no other components)
  Eval mỗi variant trên val set
  Output: logs/reward_ablation_results.json
  Thời gian ước: 4 × 80 min ≈ 5.3 giờ training + 4 × 87 min eval

#### BƯỚC 6: Latency / Cost analysis (nửa ngày)
  Script: experiments/step6_latency.py
  Đo: samples/sec, VRAM peak, total GPU-hours, USD cost (A100 = $3.67/hr on Lambda)
  Output: logs/compute_profile.json

### TUẦN 4-5 (2026-06-17 → 2026-06-30)

#### BƯỚC 7: Viết draft paper (10 ngày)
  File: paper/TRA_SAE_EXACT2026_paper.tex
  Template: paper/format/ACCESS_latex_template_20240429/access.tex
  Cấu trúc 8 section (xem mục Paper Structure bên dưới)
  Điền số liệu từ các JSON output của bước 1-6

#### BƯỚC 8: Bảng & Hình (song song với viết)
  - Table 1: Dataset statistics
  - Table 2: Main results comparison (6 baselines + 6 configs)
  - Table 3: Multi-seed cfg3 mean ± std
  - Table 4: Reward component ablation
  - Table 5: Compute cost
  - Table 6: Error analysis
  - Figure 1: Pipeline diagram (SFT → GRPO → DualLoRA → SC)
  - Figure 2: Accuracy per config bar chart
  - Figure 3: Training loss curves (Phase 1/1.5/2)
  - Figure 4: Error type pie chart

### TUẦN 6 (2026-07-01 → 2026-07-07)

#### BƯỚC 9: Internal review + polish (3 ngày)
  - Đọc lại toàn bộ, sửa flow
  - Check reference format IEEEtran
  - Verify tất cả số liệu trong bảng khớp với JSON

#### BƯỚC 10: Submit (1 ngày)
  - Chọn venue: Applied Intelligence (Springer) hoặc Cognitive Computation
  - Chuẩn bị cover letter
  - Submit qua Editorial Manager / ScholarOne

---

## CẤU TRÚC PAPER (8 sections, ~10 trang IEEE 2-cột)

Section 1: Introduction (1 trang)
  - Context: LLM reasoning for physics + logic
  - Problem: small model, resource-limited, multi-domain
  - Contributions (3 bullet): (i) pipeline, (ii) unit-aware reward, (iii) empirical findings
  - Paper organization

Section 2: Related Work (1.5 trang)
  Group A: Math/Science LLMs — DeepSeekMath, Qwen2.5-Math, Llemma, MAmmoTH
  Group B: RLHF/GRPO — InstructGPT, PPO, GRPO, Process Supervision
  Group C: Neuro-symbolic — Logic-LM, LINC, PAL, Math-Shepherd
  Group D: Parameter-efficient fine-tuning — LoRA, QLoRA, LoRAHub

Section 3: Problem Formulation (0.5 trang)
  - Formal: dataset D = D_physics ∪ D_logic, model M, evaluation metric Acc

Section 4: Methodology (2 trang)
  4.1 Overview pipeline
  4.2 Phase 1: SFT với 3-tag XML format
  4.3 Phase 1.5: Logic curriculum SFT
  4.4 Phase 2: GRPO với unit-aware reward (công thức toán học đầy đủ)
  4.5 Agent v2: Dual-LoRA + Router + Self-Consistency

Section 5: Experimental Setup (0.75 trang)
  5.1 Datasets (EXACT 2026: train/val split, physics/logic distribution)
  5.2 Baselines (6 models)
  5.3 Evaluation metrics (Accuracy, McNemar p-value)
  5.4 Implementation details (hyperparameters table)

Section 6: Results and Analysis (2.5 trang)
  6.1 Main results — Table 2 (6 baselines + ablation)
  6.2 Multi-seed reliability — Table 3
  6.3 Reward ablation — Table 4
  6.4 Compute cost — Table 5
  6.5 Error analysis — Table 6 + Figure 4
  6.6 Key findings (negative results: SC và DualLoRA không giúp)

Section 7: Discussion (0.75 trang)
  - Khi nào SC không hiệu quả
  - Router accuracy và domain confusion
  - Limitations: small eval set, single competition domain

Section 8: Conclusion (0.5 trang)
References (~27 entries)

---

## VENUES VÀ DEADLINE

| Venue | Quartile | IF | Submission link | Review time |
|---|---|---|---|---|
| Applied Intelligence (Springer) | Q2 | 5.3 | springer.com/journal/10489 | 4-8 tháng |
| Cognitive Computation (Springer) | Q2 | 4.5 | springer.com/journal/12559 | 4-6 tháng |
| Neurocomputing (Elsevier) | Q1-Q2 | 6.0 | editorialmanager.com/neucom | 4-6 tháng |
| IEEE Access | Q1 | 3.9 | ieeeaccess.ieee.org | 1-2 tháng |
| Expert Systems (Wiley) | Q2 | 3.0 | onlinelibrary.wiley.com | 3-5 tháng |

Khuyến nghị: Submit Applied Intelligence trước, nếu reject sau 2 tháng thì resubmit Neurocomputing.
