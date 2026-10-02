# 📋 TRA-SAE — BẢN GHI NHẬN & HƯỚNG DẪN CUỐI (handoff)

> Cập nhật: 2026-06-10. File này ghi lại TOÀN BỘ: project là gì, đã làm gì &
> tại sao, file ở đâu, cách nộp, việc tay còn lại, và lệnh để chạy lại.
> **Đọc Mục 2 + 3 trước nếu bạn chỉ muốn nộp bài.**

---

## 1. TÓM TẮT 1 PHÚT
Nghiên cứu: dùng **1 model 4B + ngân sách <$64** xây pipeline suy luận khoa học
**vật lý + logic**, đánh giá chặt chẽ & trung thực. Paper đã viết xong theo
**format Wiley → nộp tạp chí *Expert Systems***. Đã qua 2 vòng review góp ý + 1
vòng chạy GPU bổ sung + rà soát số/citation toàn diện + sửa lỗi compile.
**Trạng thái: SẴN SÀNG NỘP** sau khi bạn điền tên tác giả.

---

## 2. FILE GIAO NỘP (ở đâu — là gì)

| File | Đường dẫn | Chú thích |
|---|---|---|
| 📦 **Zip Overleaf** | `paper/TRA_SAE_ExpertSystems_overleaf.zip` (6.1MB) | Upload thẳng lên Overleaf. Đã verify hợp lệ, 1 file .tex (tự nhận main), bibliography nhúng sẵn (không cần bibtex) |
| 📄 **Cover letter** | `paper/cover_letter_ExpertSystems.txt` | Nộp riêng trong hệ thống tạp chí (KHÔNG nằm trong zip) |
| 📝 **Source .tex** | `paper/format/WileyDesign/Optimal-Design-layout/TRA_SAE_Wiley.tex` | Bản gốc đang chỉnh (828 dòng, 10 bảng, 3 hình, 27 tài liệu) |
| 🖼️ Hình | cùng thư mục trên: `fig1_pipeline.png`, `fig_stage_accuracy.png`, `fig_vs_baselines.png` | PNG thật (matplotlib), không PDF |

**Dữ liệu nguồn (single source of truth — đừng xóa):**
`logs/qwen35_ablation_canonical.json` (cfg0-5), `logs/qwen35_ablation_canonical_partB.json`
(cfg0-R + cfg3 single-pass), `logs/error_analysis_single_pass.json`,
`logs/fair_baselines_results_latest.json`, `logs/external_benchmark_results_latest.json`,
`logs/tool_baselines_results_latest.json`, `logs/reward_ablation_results_latest.json`.

---

## 3. HƯỚNG DẪN NỘP (chi tiết từng bước)

### Bước 1 — Compile thử trên Overleaf
1. Vào overleaf.com → **New Project → Upload Project** → chọn
   `TRA_SAE_ExpertSystems_overleaf.zip`.
2. Overleaf tự nhận `TRA_SAE_Wiley.tex` là main (chỉ có 1 file .tex).
3. Compiler để **pdfLaTeX** → bấm **Recompile**.
   - ⚠️ Bibliography đã **nhúng tay** (`thebibliography`), KHÔNG chạy bibtex →
     compile nhanh, không lỗi `\ifnum`/empty-stack như bản trước.
4. Ra PDF ~10-12 trang.

### Bước 2 — Điền thông tin tác giả (sửa trực tiếp trên Overleaf)
Mở `TRA_SAE_Wiley.tex`, sửa các dòng (đang để ẩn cho review):
```latex
\author[1]{Anonymous Author(s)}              → tên thật
\address[1]{\orgdiv{Affiliation withheld...}  → khoa/đơn vị thật
\corres{Corresponding author. \email{...}}    → email thật
\authormark{ANONYMOUS \textsc{et al.}}        → HỌ tác giả đầu
```
`\volume{}`, `\articledoi{}` **để trống** = đúng chuẩn submit (tạp chí điền sau).

### Bước 3 — Nộp trên hệ thống Expert Systems
1. Vào https://onlinelibrary.wiley.com/journal/14680394 → **Submit an article**
   (qua Research Exchange / ScholarOne của Wiley).
2. Upload: PDF (export từ Overleaf) + source files (zip).
3. Dán nội dung `cover_letter_ExpertSystems.txt` (đã điền tên/ngày) vào ô cover letter.
4. Chọn loại bài: **Original Article**.

---

## 4. ✅ CHECKLIST VIỆC TAY CÒN LẠI (đánh dấu khi xong)
- [ ] Compile thử Overleaf ra PDF OK.
- [ ] Điền **tên tác giả + đơn vị + email** (Bước 2).
- [ ] Điền `[Author name(s)]`, `[Affiliation]`, `[Date]` trong cover letter.
- [ ] **THU HỒI `GEMINI_API_KEY`** đã lộ trong chat → tạo key mới ở Google AI Studio. ⚠️
- [ ] (Tùy chọn nâng chất) Chuẩn bị **GitHub/Zenodo repo** cho code → đổi "upon
      acceptance" thành link repo nếu có.
- [ ] (Tùy chọn) **Manual error annotation** 50 mẫu (xem Mục 8) → IAA cho §6.8.

---

## 5. KẾT QUẢ CHÍNH (đã verify với dữ liệu gốc — single-pass)

**Bảng ablation (217 mẫu):**
| cfg | Overall | Physics | Logic | Ghi chú |
|---|---|---|---|---|
| cfg0 zero-shot | 29.95 | 38.30 | 14.47 | gốc, không gì |
| **cfg0-R** +retrieval | **39.17** | 50.35 | 18.42 | control tách retrieval |
| cfg1 +SFT | 44.24 | 58.16 | 18.42 | |
| cfg2 +Logic SFT | 46.08 | 59.57 | 21.05 | |
| **cfg3 +GRPO** | **47.47** | **63.12** | 18.42 | **tốt nhất** |
| cfg4 Dual-LoRA | 43.78 | 55.32 | 22.37 | không giúp |
| cfg5 SC×5 | 44.70 | 56.74 | 22.37 | không giúp |

**Phân rã có ý nghĩa thống kê (McNemar):**
- Retrieval: **+9.22pp, p=7.8×10⁻⁴ (significant)**
- SFT: **+5.07pp, p=0.054 (biên)**
- GRPO over logic-SFT: +1.38pp, p=0.61 (không sig)

**Khác:** baseline 7B fair 11.5–29.0% (đều thua cfg3) · MMLU cfg3 48.62 > cfg0
42.29 (generalize) · tool +0.46pp (gần như không) · multi-seed 46.70±0.70 ·
error single-pass: extraction 72.6%, logic 17.9%.

---

## 6. TỔNG HỢP PROJECT (what / why / đóng góp)

**Bài toán:** với model nhỏ (4B) + ngân sách thấp, đẩy suy luận khoa học (lý+logic)
tới đâu, và **thành phần nào thực sự tạo gain**.

**Đã làm:** pipeline 3 giai đoạn (SFT → Logic SFT → GRPO) + retrieval lúc thi;
7 cấu hình; đánh giá single-pass sạch (không rò rỉ nhãn); decomposition
retrieval-vs-finetune; McNemar + Wilson CI + 3 seed; 6 baseline fair multi-prompt;
MMLU ngoài; tool Z3/Python; reward ablation; error analysis; compute profile.

**Đóng góp khoa học (đây là dạng *empirical / reproducibility study*, KHÔNG phải
"method mới"):**
1. Công thức tái lập **siêu rẻ (<$64)** cho QA khoa học đa lĩnh vực trên 4B.
2. **Phân rã nhân-quả** chỉ ra gain thật từ **retrieval > fine-tuning > GRPO** —
   sửa ngộ nhận "gán hết cho SFT/GRPO".
3. **Negative results đặc tả rõ:** SC, Dual-LoRA, tool, reward phụ đều KHÔNG giúp ở 4B.
4. **Protocol sạch** (single-pass, baseline fair, benchmark ngoài).
5. Chẩn đoán **logic là nút thắt** (lý 63% vs logic 18%).

---

## 7. HÀNH TRÌNH & CÁC FIX QUAN TRỌNG (ghi nhận để truy vết)

| Giai đoạn | Việc | Lý do |
|---|---|---|
| Phase A→D | Chạy nốt pipeline overnight (B8 Gemini, fair, MMLU, tool, stats) | hoàn thành dữ liệu |
| Bug #1 | **step8/step9 `max_new_tokens 512→1024`** | 512 cắt output trước `\boxed{}`/`<answer>` → điểm sai giả |
| Bug #2 | **Oracle-retry trong step0** → chuyển **single-pass** | eval cũ dùng đáp án để retry = rò rỉ nhãn; headline 54.84%→47.47% (thật) |
| Dọn | Move IEEE/CMC format cũ → `paper/bin/` | chuyển sang elsarticle rồi Wiley |
| Viết | Paper elsarticle → **Wiley (Expert Systems)** | đổi venue hợp scope hơn |
| Review #1 fix | tách retrieval (cfg0-R), error single-pass, baseline reconcile, soften claims, Data Availability, Appendix prompts | theo reviewer |
| Part B (GPU) | chạy `cfg0-R` (39.17%) + `cfg3 single-pass` (error 117 câu) | tách retrieval/SFT tận gốc + error đúng protocol |
| Review #2 fix | "borderline" cho SFT, GRPO-không-giúp-logic, public-baseline note, dedicated-run note | theo reviewer |
| Rà soát | **sửa 2 lỗi thật: Wilson CI cfg0-R [32.5→32.9,45.8]; citation MMLU (ETHICS→MMLU)** | verify số/citation |
| Fix compile | **Bỏ bibtex → `thebibliography` nhúng tay** | `wileyNJD-Chicago.bst` lỗi empty-stack/ifnum → timeout |

---

## 8. LỆNH THAM KHẢO (chạy lại nếu cần — cần GPU Colab + BƯỚC 0 setup)

**Setup môi trường (sau disconnect):**
```bash
%cd /content/drive/MyDrive/TRA-SAE
!pip -q install "transformers>=4.46" peft trl datasets accelerate scipy z3-solver openai
!pip -q uninstall -y torchao        # ⭐ bắt buộc, nếu không LoRA chết
!python -c "import z3; print('z3 OK')"
```

**Chạy lại 1 phần (ví dụ):**
```bash
# Eval single-pass 1 config, ghi file riêng (KHÔNG đụng canonical):
!python experiments/step0_canonical_eval.py --config 3 --retries 0 --out-tag partB
# cfg0+retrieval:
!python experiments/step0_canonical_eval.py --config 6 --retries 0 --out-tag partB
# Error taxonomy single-pass:
!python experiments/_partB_error_classify.py
```

**Manual error annotation (Mục 4):**
```bash
# 1) mở logs/manual_error_template.csv, điền cột manual_label (FS/UC/AR/PU/LE/AX)
# 2) lưu thành logs/manual_error_filled.csv
!python experiments/step14_manual_error_annotation.py --score   # → agreement + kappa
```

**Sinh lại bibliography nhúng tay (nếu sửa references.bib):**
```bash
# script đã dùng nằm trong lịch sử; có thể tái tạo từ paper/_thebib_block.tex
```

---

## 9. ⚠️ LƯU Ý QUAN TRỌNG
- **Bảo mật:** key Gemini đã lộ → THU HỒI ngay (Mục 4).
- **Header banner** trong PDF hiện chữ "Allergy" (ảnh mẫu template Wiley) — vẫn
  compile; thay `images/allergy.eps` bằng banner Expert Systems nếu muốn.
- **Khung lề** trong PDF do gói `showframe` của template — bình thường; muốn bỏ
  thì comment `\RequirePackage{showframe}` trong `USG.cls`.
- **Đánh giá venue (từ reviewer):** Expert Systems = lựa chọn hợp lý nhất.
  KHÔNG nên nhắm Neurocomputing/Q1 mạnh (novelty vừa phải, kết quả trung-khá).
- Nếu Overleaf vẫn timeout sau khi bỏ bibtex (ít khả năng): tắt `showframe` /
  giản gói nặng — nhắn để được hỗ trợ.

---
*Hết. File này là bản ghi nhận chính thức của project tính đến 2026-06-10.*
