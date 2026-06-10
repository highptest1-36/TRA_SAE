================================================================================
 TRA-SAE — Expert Systems (Wiley) SUBMISSION PACKAGE
 Gói nộp bài hoàn chỉnh để gửi GVHD / submit
================================================================================

Manuscript: TRA-SAE: A Low-Cost Retrieval-Augmented Expert Reasoning System for
            Mixed Physics-Logic Question Answering with a 4B Language Model
Journal   : Expert Systems (Wiley) — Original Article — single-anonymous review

Tác giả:
  1) Cao-Phuc Ha (first author)
     Department of Artificial Intelligence, FPT University, Ho Chi Minh City, Vietnam
     caophucai@gmail.com | ORCID 0009-0004-7024-5166
  2) Phu-Nguyen Le (corresponding author)
     Faculty of Engineering and Technology, Nguyen Tat Thanh University, HCMC, Vietnam
     lpnguyen@ntt.edu.vn (alt: nguyenlp9@fpt.edu.vn) | ORCID 0000-0001-6773-6336

--------------------------------------------------------------------------------
CẤU TRÚC FOLDER
--------------------------------------------------------------------------------
01_Manuscript/
    TRA_SAE_ExpertSystems_Main.pdf   <- bản PDF đã compile, de-anonymized (peer review)
    TRA_SAE_ExpertSystems_Main.tex   <- source LaTeX (đã sửa tên tác giả/keywords/abstract)
    references.bib                   <- bibliography
    latex_source.zip                 <- toàn bộ source compile được trên Overleaf (pdfLaTeX)
02_Title_Page/
    Title_Page_Author_Info.docx      <- title page đầy đủ tên/affiliation/email/ORCID/đóng góp
03_Cover_Letter/
    Cover_Letter_ExpertSystems.docx  <- cover letter (đã ghi đúng tác giả + journal fit + COI)
04_Figures/
    Figure_1_TRA_SAE_Overview.pdf/.png    (300 dpi)
    Figure_2_Pipeline_Accuracy.pdf/.png   (300 dpi)
    Figure_3_Baseline_Comparison.pdf/.png (300 dpi)
    Figure_Captions.txt
05_Declarations/
    Conflict_of_Interest.txt
    Funding_Statement.txt
    Data_Availability_Statement.txt
    Ethics_Statement.txt             (ethics / patient consent / permission / clinical trial)
    Author_Contributions.docx        (CRediT, 2 tác giả)
06_Reproducibility/
    Repository_Link.txt
    README_reproducibility.pdf
    Result_Artifacts_Summary.xlsx / .pdf
07_Submission_Info/
    Submission_Metadata_for_GVHD.docx  <- toàn bộ metadata để điền vào portal Wiley

--------------------------------------------------------------------------------
ĐÃ HOÀN THÀNH (theo checklist guideline)
--------------------------------------------------------------------------------
[x] De-anonymize: thay "Anonymous Author(s)" -> 2 tác giả thật + affiliation + email
[x] ORCID cho cả hai tác giả (submitting author bắt buộc có - đã có)
[x] Keywords rút còn 7 (large language models; expert reasoning systems; knowledge
    engineering; scientific question answering; retrieval-augmented generation;
    curriculum fine-tuning; neuro-symbolic reasoning)
[x] Abstract ~229 từ (<= 250)  [đã đếm lại]
[x] Author Contributions viết theo từng tác giả (CRediT)
[x] Acknowledgments / Funding / COI / Data Availability cập nhật cho 2 tác giả
[x] Cover letter: contribution + journal fit + code/data availability + COI + emails
[x] Figures 300 dpi (PDF + PNG) + captions
[x] Declarations đầy đủ (ethics, patient consent, permission, clinical trial = N/A)
[x] Reproducibility package (README + bảng map kết quả -> script -> log)
[x] PDF compile sạch (11 trang, 26 references, không lỗi)

--------------------------------------------------------------------------------
*** VIỆC CẦN LÀM TRƯỚC KHI SUBMIT (ACTION ITEMS) ***
--------------------------------------------------------------------------------
1. [REPO URL] Đã đặt URL repo = https://github.com/highptest1-36/TRA_SAE  (kiểm tra repo public + đầy đủ file). Nếu đổi, sửa lại ở:
      - manuscript .tex (đang là placeholder https://github.com/highptest1-36/TRA_SAE)
      - 05_Declarations/Data_Availability_Statement.txt
      - 03_Cover_Letter/Cover_Letter_ExpertSystems.docx ([insert URL])
      - 07_Submission_Info/Submission_Metadata_for_GVHD.docx
   Sau khi sửa .tex, compile lại (hoặc upload latex_source.zip lên Overleaf) để cập nhật PDF.
2. [OPEN ACCESS] Thống nhất với GVHD: chọn STANDARD (subscription) để KHÔNG mất phí.
   Open Access APC ~ US$3,600 / GBP 2,390 / EUR 3,020. Không có phí nộp bài.
3. [WORD COUNT] Xác nhận manuscript <= 15,000 từ (hiện ~7-8k, OK) nếu portal hỏi.
4. [FIGURE 1] Xem lại wording Figure 1: GRPO KHÔNG nên thể hiện như "best/winner"
   vì kết quả cho thấy GRPO không có lợi ích thống kê rõ (xem Figure_Captions.txt).
5. [DATE/SIGNATURE] Điền ngày + ký cover letter.
6. [SUGGESTED REVIEWERS] (tùy chọn) chuẩn bị 3-5 reviewer nếu portal yêu cầu.

Ghi chú: review của Expert Systems là single-anonymous => tên tác giả ĐƯỢC hiển thị
cho reviewer, nên manuscript để tên thật là đúng.
