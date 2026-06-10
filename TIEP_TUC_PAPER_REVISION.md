# TIẾP TỤC — Paper Revision (Expert Systems / Wiley)

> Cập nhật: **2026-06-11 (lần 5)** · Trạng thái: **ĐÃ COMPILE OK** — PDF 9 trang, APA chuẩn, 0 lỗi · còn **2 việc của tác giả** · **+ có PLAN phản hồi reviewer Expert Systems**.
> Mục đích file này: nếu Colab/phiên chat **disconnect**, đọc file này là biết **đang ở đâu, làm gì tiếp, chạy lệnh nào**.

> 📋 **PLAN CẢI THIỆN THEO NHẬN XÉT REVIEWER (2026-06-11):** xem
> **`paper/PLAN_REVISION_ExpertSystems.md`** — bản resume đầy đủ: đối chiếu từng ý reviewer ↔ bài,
> before→after chính xác cho từng sửa text (Track A), script no-GPU (Track B), thí nghiệm GPU (Track C),
> facts đã verify + lệnh tái lập số liệu. **Chưa thực thi** — chờ duyệt/làm.
> Tóm tắt: ~1/2 nhận xét reviewer thì bài đã làm rồi; 3 lỗ hổng thật = baseline 7B chưa có retriever (#5),
> manual error annotation 0/50 chưa làm (#6), chưa có official test + thiếu leakage check (#4).

---

## 0. TL;DR (đọc 30 giây)

- **PAPER ĐÃ HOÀN CHỈNH & COMPILE OK.** PDF preview: **`paper/TRA_SAE_Wiley.pdf`** (9 trang, APA thật, 0 lỗi/0 undefined).
- Đã sửa toàn bộ lỗi reviewer Expert Systems (citation APA, abstract ≤250, title, DOI, ngày tháng, back matter) **+ Challenge Background/lineage EXACT 2026 ↔ XAI 2025 + bảng positioning + bảng ví dụ lỗi thật**; đã bỏ badge OPEN ACCESS + masthead "Allergy".
- **Mọi thứ lưu trên Google Drive** → an toàn khi Colab disconnect. (Chỉ TeX Live là tạm, phải cài lại sau restart.)
- **File nộp/compile Overleaf:** `paper/TRA_SAE_ExpertSystems_overleaf.zip`.
- **Còn lại (BẠN làm):** (1) điền thông tin tác giả, (2) điền link repo. Hết.

### ⚡ RESUME trong 2 lệnh (sau khi Colab reconnect)
```bash
cd /content/drive/MyDrive/TRA-SAE
bash paper/bin/setup_texlive.sh    # cài TeX Live (1 lần/session, ~5-15 phút)
bash paper/bin/compile_paper.sh    # ra paper/TRA_SAE_Wiley.pdf
python3 paper/bin/verify_paper.py  # (tùy chọn) kiểm tra toàn vẹn nguồn
```

---

## 1. ⚠️ FILE NÀO LÀ CHÍNH (tránh nhầm lẫn)

Project có **2 bản paper khác nhau** — đừng sửa nhầm:

| File | Template / Journal | Số liệu | Trạng thái |
|---|---|---|---|
| `paper/bin/TRA_SAE_paper.tex` | elsarticle / **Neurocomputing** | 36.41→54.84 (multi-attempt) | **BẢN CŨ — đã chuyển vào bin/ (2026-06-11)** |
| `paper/format/WileyDesign/Optimal-Design-layout/TRA_SAE_Wiley.tex` | Wiley USG.cls / **Expert Systems** | 29.95→47.47 (single-pass) | ✅ **BẢN ĐANG DÙNG** |

- **Master (sửa ở đây):** `paper/format/WileyDesign/Optimal-Design-layout/TRA_SAE_Wiley.tex`
- **Bản sao tiện mở:** `paper/TRA_SAE_Wiley.tex` (giống hệt master)
- **Gói upload Overleaf:** `paper/TRA_SAE_ExpertSystems_overleaf.zip` (đã chứa tex+bib+USG.cls mới + figures + fonts)

> Hai bản số liệu khác nhau vì **giao thức eval khác**: bản Expert Systems dùng **single-pass** (1 lần greedy, không retry — trung thực hơn). Đây là bản gửi tạp chí.

---

## 2. KẾT QUẢ CHÍNH (self-contained, để khỏi mở lại file)

Validation 217 mẫu (141 physics, 76 logic), single greedy pass:

| Cấu hình | Overall | Physics | Logic |
|---|---|---|---|
| cfg0 zero-shot | 29.95 | 38.30 | 14.47 |
| cfg0-R +retrieval (no FT) | 39.17 | 50.35 | 18.42 |
| cfg1 +SFT | 44.24 | 58.16 | 18.42 |
| cfg2 +Logic SFT | 46.08 | 59.57 | 21.05 |
| **cfg3 +GRPO (best)** | **47.47** | **63.12** | 18.42 |
| cfg4 Dual-LoRA+Router | 43.78 | 55.32 | 22.37 |
| cfg5 Full agent (SC×5) | 44.70 | 56.74 | 22.37 |

**Thông điệp chính:** retrieval đóng góp lớn nhất (+9.22pp, p<1e-3); SFT +5.07pp (borderline); GRPO-over-curriculum **không** có ý nghĩa thống kê (p=0.61). Logic là bottleneck. Chi phí train $18, full pipeline <$64.

---

## 3. TIẾN TRÌNH

### ✅ ĐÃ XONG (đã ghi vào file + zip)

**Lỗi bắt buộc (Nhóm A):**
- [x] Citation APA: `[ASNA]`→`[APA]`, bỏ `thebibliography` thủ công → `\bibliographystyle{WileyNJD-APA}` + `\bibliography{references}`; **bundled `WileyNJD-APA.bst` (=apacite)** vào gói; alias `\citep/\citet`→`\cite/\citeA` trong preamble.
- [x] Abstract **277→212 từ** (≤250).
- [x] Title → "TRA-SAE: Low-Cost Retrieval-Augmented Fine-Tuning for Mixed-Domain Scientific Reasoning with a 4B Language Model".
- [x] Running title (`\titlemark`) 32 ký tự (<40).
- [x] Bỏ ngày "00 Month 2026".
- [x] DOI footer: patch `USG.cls` → không in `https://doi.org/` trơ.
- [x] Back matter: "Financial Disclosure"→**Funding**; wording số ít; Conflicts số ít.

**Phần Challenge Background/lineage (bạn brainstorm) — đã verify nguồn rồi viết:**
- [x] Related Work: subsection mới **"Explainable educational QA challenges"** + **bảng positioning** (Table `tabrelated`).
- [x] Intro + Problem Formulation: định danh EXACT 2026 (IEEE IJCNN 2026) + tiền thân XAI 2025 + 3 dimensions (correctness/explanation/reasoning depth).
- [x] Gemini = **non-submission diagnostic** (luật cấm closed-source, trích nguồn website).
- [x] 2 entry .bib mới đã verify: `Nguyen2025XAIChallenge` (CEUR-WS Vol-4152, ITADATA 2025, 16 tác giả) + `EXACT2026` (website).

**Nhóm B/C:**
- [x] Bảng ví dụ lỗi thật (Table `tab11`) — 3 ví dụ cfg3 lấy từ `logs/ablation_per_sample_canonical_partB.jsonl`.
- [x] Câu giải thích baseline 7B thấp (mixed-domain + extraction).
- [x] Framing "empirical study, không phải lời giải hoàn chỉnh".
- [x] Cite SciBench + GPQA ở future work.
- [x] Chính tả UK đồng nhất (DPO→Optimization tên riêng).

**Verify tự động (chạy lại bằng `verify_paper.py`):** abstract 212 từ · 26/26 cite-key có trong .bib · brace + 8 table + 4 table* + 3 figure + 12 tabular* + ... đều cân bằng · không còn thebibliography · không còn "00 Month 2026".

### ⏳ CÒN LẠI — VIỆC CỦA BẠN (3 việc)

1. **Thông tin tác giả** — sửa trong `TRA_SAE_Wiley.tex` (dòng ~44–52), hiện là placeholder:
   - `\author[1]{Anonymous Author(s)}` → tên thật, thêm `[\orcid{...}]` nếu có.
   - `\authormark{ANONYMOUS \textsc{et al.}}` → HỌ tác giả.
   - `\address[1]{...}` → khoa/trường/thành phố/quốc gia.
   - `\corres{Corresponding author. \email{withheld@review.org}}` → email thật.
2. **Link repo** — trong Data Availability, đổi `https://github.com/ANONYMISED-FOR-REVIEW` thành link thật (hoặc anonymous repo cho review).
3. **Compile Overleaf** để xác nhận PDF (xem Mục 4).

### 🔮 TÙY CHỌN (nếu reviewer yêu cầu thêm, chưa làm)

- Manual annotation 30–50 lỗi (hiện error taxonomy là rule-based — đã ghi rõ là diagnostic).
- Extended GRPO 500 steps (đã ghi limitation "budget-conditional").
- Chuyển Appendix prompt sang Supporting Information nếu bài quá dài.

---

## 4. BƯỚC TIẾP THEO NGAY (làm theo thứ tự)

### Bước 1 — Upload & compile trên Overleaf
1. Lên overleaf.com → New Project → **Upload Project** → chọn `paper/TRA_SAE_ExpertSystems_overleaf.zip`.
2. Main document: `TRA_SAE_Wiley.tex`. Menu → Compiler = **pdfLaTeX**.
3. Compile theo thứ tự: **Recompile → (chạy BibTeX) → Recompile → Recompile**.
   (Overleaf thường tự chạy bibtex; nếu không, Logs → "Run BibTeX", hoặc bấm Recompile vài lần.)

### Bước 2 — Kiểm tra PDF (checklist)
- [ ] Citation hiện **author-year**: `(Wei et al., 2022)` và `Wei et al. (2022)`.
- [ ] Reference list xếp **alphabet**, KHÔNG còn `Wei2022` thô hay `[Wei et al.(2022)]`.
- [ ] Abstract gọn, title mới, KHÔNG còn "00 Month 2026", KHÔNG còn "https://doi.org/" trơ.
- [ ] Bảng positioning (Related Work) + bảng ví dụ lỗi (Error analysis) hiện đúng.
- [ ] Figure 1/2/3 hiện rõ.

### Bước 3 — Điền thông tin tác giả + repo (Mục 3 ⏳) rồi compile lại.

### ✅ APA ĐÃ FIX (cập nhật lần 2 — 2026-06-10)
**Nguyên nhân "?" trước đây:** USG.cls gọi `\bibliographystyle{WileyNJD-APA}` (chữ HOA W) nhưng file APA gốc tên `wileyNJD-APA.bst` (chữ thường) → Overleaf phân biệt hoa/thường → không thấy file → BibTeX lỗi → tất cả citation thành `?`.
**Đã sửa:** bundle `WileyNJD-APA.bst` (= apacite.bst, đúng chuẩn Wiley APA) đúng tên HOA vào gói; cấu hình hiện tại = `[APA]` + `\bibliographystyle{WileyNJD-APA}`. Đây là cặp chính thức (NJDapacite.sty + WileyNJD-APA.bst) → ra **APA thật**.

### Fallback cuối cùng (chỉ nếu APA vẫn lỗi): Chicago — LƯU Ý không phải APA thật
```latex
\documentclass[APA,twocolumn]{USG}        →   \documentclass[CHICAGO,twocolumn]{USG}
\bibliographystyle{WileyNJD-APA}          →   \bibliographystyle{wileyNJD-Chicago}
```
Chicago compile chắc chắn nhưng dùng full tên + có thể thiếu năm in-text → **chỉ dùng khi bí**, không nộp được như APA.

---

## 5. RESUME KHI COLAB DISCONNECT (quan trọng)

**Mọi file đã nằm trên Google Drive — KHÔNG mất khi Colab ngắt.** Các bước phục hồi:

```python
# (1) Trong Colab: mount lại Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# (2) Về thư mục dự án
cd /content/drive/MyDrive/TRA-SAE

# (3) XÁC NHẬN paper còn nguyên vẹn (không cần LaTeX)
python3 paper/bin/verify_paper.py
#   -> phải in "=> TAT CA OK."
```

```bash
# (4) Nếu lỡ chỉnh tay master mà muốn đóng gói lại zip:
cd /content/drive/MyDrive/TRA-SAE/paper/format/WileyDesign/Optimal-Design-layout
ZIP=/content/drive/MyDrive/TRA-SAE/paper/TRA_SAE_ExpertSystems_overleaf.zip
zip "$ZIP" TRA_SAE_Wiley.tex references.bib USG.cls
# và đồng bộ bản sao tiện mở:
cp TRA_SAE_Wiley.tex /content/drive/MyDrive/TRA-SAE/paper/TRA_SAE_Wiley.tex
```

> Lưu ý: việc còn lại (compile, điền tác giả) làm trên **Overleaf**, KHÔNG cần Colab. Colab chỉ dùng để verify/đóng gói lại nếu chỉnh thêm.

---

## 6. THAM CHIẾU COMMAND (giải thích từng lệnh)

| Việc | Lệnh | Giải thích |
|---|---|---|
| Verify paper | `python3 paper/bin/verify_paper.py` | Kiểm tra abstract ≤250 từ, cite-key, cân bằng ngoặc/môi trường, không còn thebibliography. In FAIL nếu hỏng. |
| Đếm từ abstract | (nằm trong verify) | — |
| Đóng gói zip | `zip "$ZIP" TRA_SAE_Wiley.tex references.bib USG.cls` | Cập nhật 3 file đã đổi vào zip (giữ nguyên figures/fonts). Chạy TRONG thư mục `Optimal-Design-layout/`. |
| Liệt kê nội dung zip | `unzip -l paper/TRA_SAE_ExpertSystems_overleaf.zip` | Xem zip chứa gì + ngày sửa. |
| Xem 1 file trong zip | `unzip -p paper/TRA_SAE_ExpertSystems_overleaf.zip TRA_SAE_Wiley.tex \| less` | In nội dung file trong zip không cần giải nén. |
| Lấy lại ví dụ lỗi cfg3 | đọc `logs/ablation_per_sample_canonical_partB.jsonl` (config_id=3) | Nguồn 3 ví dụ trong Table tab11. |

**KHÔNG chạy lại** `paper/bin/apply_revision_edits.py` — đây là script một-lần **đã áp dụng rồi**; chạy lại sẽ báo lỗi assertion (vì chuỗi gốc đã bị thay) — đó là tính năng an toàn, nghĩa là "đã làm xong".

---

## 7. BẢN ĐỒ FILE

```
TRA-SAE/
├── TIEP_TUC_PAPER_REVISION.md          ← FILE NÀY (resume/status)
├── paper/
│   ├── TRA_SAE_ExpertSystems_overleaf.zip   ← UPLOAD CÁI NÀY lên Overleaf
│   ├── TRA_SAE_Wiley.tex                     ← bản sao tiện mở (= master)
│   ├── references.bib                        ← bib master (đã thêm 2 entry)
│   ├── cover_letter_ExpertSystems.txt
│   ├── bin/                                  ← script + TOÀN BỘ file cũ (xem _MOVED_MANIFEST.txt)
│   │   ├── verify_paper.py                   ← script kiểm tra (tái dùng)
│   │   ├── apply_revision_edits.py           ← script đã áp dụng (KHÔNG chạy lại)
│   │   ├── compile_paper.sh / setup_texlive.sh
│   │   └── TRA_SAE_paper.tex                 ← BẢN CŨ Neurocomputing (đã dọn vào đây 2026-06-11)
│   └── format/WileyDesign/Optimal-Design-layout/
│       ├── TRA_SAE_Wiley.tex                 ← MASTER (sửa ở đây)
│       ├── references.bib                    ← bib build
│       ├── USG.cls                           ← class Wiley (đã patch DOI)
│       ├── WileyNJD-APA.bst                 ← bst APA (=apacite) — FIX dấu ?
│       ├── wileyNJD-Chicago.bst              ← bst cho fallback Chicago
│       └── fig1_pipeline / fig_stage_accuracy / fig_vs_baselines (.png)
```

---

## 8. NHẬT KÝ THAY ĐỔI (2026-06-10)

- `TRA_SAE_Wiley.tex` (master + bản sao): documentclass, preamble alias, title, titlemark, keywords, abstract, intro lineage, Related Work subsection + tabrelated, Problem Formulation cite + dimensions, Gemini disclaimer, baseline-low note, tab11 ví dụ lỗi, logic-gap framing, conclusion SciBench/GPQA, back matter, bibliography→BibTeX.
- `references.bib` (×2): +`Nguyen2025XAIChallenge`, +`EXACT2026`.
- `USG.cls`: guard DOI footer.
- `TRA_SAE_ExpertSystems_overleaf.zip`: cập nhật 3 file trên.
- Nguồn đã verify: EXACT website (ura.hcmut.edu.vn/exact), CEUR-WS Vol-4152 paper98 (ITADATA 2025).

## 9. CẬP NHẬT LẦN 2 (2026-06-10)
- Sau khi compile Overleaf: `[APA]` ra toàn `?` do **lệch hoa/thường tên .bst** (class gọi `WileyNJD-APA`, file là `wileyNJD-APA`).
- Fix: tải `WileyNJD-APA.bst` (= apacite.bst, từ repo Wiley NJD chính thức), bundle ĐÚNG TÊN HOA + bản thường vào gói; đổi `\bibliographystyle{apacite}`→`{WileyNJD-APA}`. Zip đã cập nhật → tải lại & upload Overleaf.

## 10. CẬP NHẬT LẦN 3 (2026-06-10) — fix lỗi BibTeX khi compile Overleaf
- **Lỗi:** `BibTeX: Illegal, another \bibstyle command : {WileyNJD-APA}` + compile timed out.
- **Nguyên nhân:** USG.cls khi `[APA]` ĐÃ tự gọi `\bibliographystyle{WileyNJD-APA}`. Dòng `\bibliographystyle` mình thêm trong body là lần 2 → BibTeX cấm 2 lệnh bibstyle → lỗi (và đây mới là thủ phạm thật của dấu `?` ban đầu, không chỉ casing).
- **Fix:** XOÁ dòng `\bibliographystyle{...}` trong body, chỉ giữ `\bibliography{references}` (đúng như mẫu Wiley `Optimal-Design-layout.tex`). Zip đã cập nhật.
- **Cách nhanh nhất cho bạn:** trong Overleaf, xoá đúng dòng `\bibliographystyle{WileyNJD-APA}` (giữ `\bibliography{references}`) → Recompile → BibTeX → Recompile.
- **Compile timed out (free plan):** thường do vòng lặp bibtex-lỗi + class nặng + STIX font lần đầu. Sau khi fix bibtex, compile lại thường lọt. Nếu vẫn timeout: bấm **Start free trial** (miễn phí, tăng thời gian compile) HOẶC nhờ Claude compile trong Colab để lấy PDF đã verify.

## 11. CẬP NHẬT LẦN 4 (2026-06-10) — ĐÃ COMPILE THÀNH CÔNG TRONG COLAB
- Cài TeX Live trong Colab → compile `pdflatex→bibtex→pdflatex×2` ⇒ **PDF 9 trang, 0 lỗi, 0 undefined citation, 0 overfull**.
- BibTeX chạy đúng `WileyNJD-APA.bst` (apacite v6.03 APA) ⇒ **APA thật**: in-text `(Yu et al., 2023)` / `Wei et al. (2022)`; reference list `Wei, J., Wang, X., … (2022). … Retrieved from https://…`.
- Dọn thêm 2 artefact demo của template trong `USG.cls`: bỏ **badge OPEN ACCESS** (dòng ~1052) và **masthead "Allergy"** (node allergy.eps ~1582). Giữ logo WILEY.
- Sửa `Bengio2009` `@article`→`@inproceedings` (hết warning "No journal").
- **PDF preview:** `paper/TRA_SAE_Wiley.pdf` (mở từ Drive). Zip Overleaf đã cập nhật USG.cls + references.bib.
- Compile lại trong Colab bất cứ lúc nào:
  ```bash
  cd /content/drive/MyDrive/TRA-SAE/paper/format/WileyDesign/Optimal-Design-layout
  pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex && bibtex TRA_SAE_Wiley && \
  pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex && pdflatex -interaction=nonstopmode TRA_SAE_Wiley.tex
  ```
  (TeX Live đã cài trong session; nếu Colab restart phải cài lại: `apt-get install -y texlive-latex-extra texlive-fonts-extra texlive-bibtex-extra texlive-science texlive-pictures poppler-utils`)

