# PLAN CẢI THIỆN PAPER — Expert Systems (Wiley) — BẢN RESUME ĐẦY ĐỦ

> Lập 2026-06-11. **Mục đích:** sống sót khi Colab disconnect. Đọc file này là biết
> đang ở đâu, cần sửa gì, sửa ở dòng nào, before→after ra sao, lệnh nào để verify.
> **CHƯA sửa gì vào paper** — đây là plan đã duyệt, chờ thực thi.
>
> File chính cần sửa (MASTER):
> `paper/format/WileyDesign/Optimal-Design-layout/TRA_SAE_Wiley.tex`
> Bản sao tiện mở (đồng bộ sau khi sửa): `paper/TRA_SAE_Wiley.tex`

---

## 0. TL;DR (đọc 30 giây)
- Reviewer đúng về mặt sự kiện, NHƯNG ~1/2 fix họ đề xuất **bài đã làm rồi**.
- **3 lỗ hổng thật:** (#5) baseline 7B chưa chạy với retriever · (#6) manual error annotation 0/50 chưa làm · (#4) chưa có official test + thiếu leakage check văn bản.
- **Thứ tự nên làm:** Track A (sửa text, 0 chi phí) → Track B (no GPU) → Track C (GPU) → Track D (thừa nhận).
- Số liệu single-pass đã tái lập KHỚP bản thảo (xem Mục 2).

## RESUME sau khi Colab reconnect
```bash
from google.colab import drive; drive.mount('/content/drive')   # trong Colab
cd /content/drive/MyDrive/TRA-SAE
python3 paper/bin/verify_paper.py        # xác nhận paper còn nguyên (=> TAT CA OK)
# Tái lập số liệu single-pass để chắc chắn log gốc còn đúng:
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open('logs/ablation_per_sample_canonical.jsonl')]
agg=collections.defaultdict(lambda:{'n':0,'sp':0,'np':0,'spp':0,'nl':0,'spl':0}); name={}
for r in rows:
    c=r['config_id']; name[c]=r['config_name']; a=agg[c]; a['n']+=1
    sp=r['correct'] and r.get('retry_count',0)==0; a['sp']+=sp
    if r['subject']=='physics': a['np']+=1; a['spp']+=sp
    else: a['nl']+=1; a['spl']+=sp
for c in sorted(agg):
    a=agg[c]; print(f"cfg{c} {name[c]:<14} overall={100*a['sp']/a['n']:.2f} phys={100*a['spp']/a['np']:.2f} logic={100*a['spl']/a['nl']:.2f}")
PY
# Kỳ vọng: cfg0 29.95/38.30/14.47 · cfg3 47.47/63.12/18.42  (khớp paper)
```

---

## 1. FACTS ĐÃ VERIFY (đừng verify lại từ đầu)
| Điều | Kết luận | Bằng chứng |
|---|---|---|
| Số single-pass của paper | **Tái lập KHỚP** (cfg0 29.95→cfg3 47.47; logic 18.42<21.05) | `logs/ablation_per_sample_canonical.jsonl`, lọc retry_count==0 |
| `qwen35_ablation_canonical.json` | Là số **multi-attempt** (36.41→54.84), KHÔNG dùng cho paper | có field `per_retry` |
| Baseline 7B | Chạy **zero-shot, KHÔNG retrieval** | `experiments/step8_fair_baselines.py` không import retriever; `logs/fair_baselines_results_latest.json` |
| `_contamination_test.log` | Chỉ test **LoRA adapter** (verdict: safe), KHÔNG phải train↔val leakage | đọc đầu file |
| `manual_error_template.csv` | `manual_label` **0/50 điền** (annotation chưa làm); 50 dòng toàn logic AX/LE | đếm cột |
| Official test split | **KHÔNG tồn tại** (chỉ self-split 90/10) | `processed_data/` chỉ có exact_train(1945)+exact_val(217) |
| External benchmark | Chỉ **MMLU physics** (253 q); KHÔNG có logic benchmark | `logs/external_benchmark_results_latest.json` |
| PDF history dates | **KHÔNG in** Received/Revised/Accepted | `pdftotext TRA_SAE_Wiley.pdf` không có chuỗi này |
| Error analysis | Paper dùng nhất quán n=117 (`error_analysis_single_pass.json`, E6=85=72.6%); file n=98 là run cũ | 2 file json |

---

## 2. BẢN ĐỒ DỮ LIỆU (log → nội dung)
- `logs/ablation_per_sample_canonical.jsonl` — per-sample 6 cfg × 217 (single-pass = retry_count==0). **Nguồn số chính của paper.**
- `logs/ablation_per_sample_canonical_partB.jsonl` — run giữ full generation cho error analysis (nguồn tab11).
- `logs/ablation_per_sample_canonical_seed{1337,2024}.jsonl` — multi-seed (tab5).
- `logs/fair_baselines_results_latest.json` — 5 baseline × 3 prompt (tab3, dòng B1–B6). **Zero-shot.**
- `logs/external_benchmark_results_latest.json` — MMLU physics 253q (tab6).
- `logs/error_analysis_single_pass.json` — taxonomy n=117 (tab10). E1=2,E4=21,E6=85,E0=9.
- `logs/manual_error_template.csv` — template annotation (CHƯA điền).
- `logs/cfg3_multiseed_results_latest.json` — seed 42/1337/2024 (tab5).
- `src/retriever.py` — retriever đã có (dùng cho Track C).
- `experiments/step8_fair_baselines.py` — script baseline (sửa cho Track C).
- `experiments/step9_external_benchmark.py` — script MMLU (mở rộng cho logic benchmark).

---

## 3. TRACK A — SỬA TEXT (0 chi phí, làm trước). File: MASTER tex.
> Mỗi mục có: vị trí (dòng ~ + chuỗi neo để search) · TRƯỚC · SAU. Dòng có thể xê dịch khi sửa dần — **search theo chuỗi neo** cho chắc.

### A1. Nâng kết quả âm thành luận điểm (ý #1 novelty)
**(a) Abstract** — neo: `is not statistically significant. A fair multi-prompt`
- TRƯỚC: `...whereas the GRPO increment over the curriculum stage is not statistically significant. A fair multi-prompt re-evaluation places...`
- SAU: `...whereas the GRPO increment over the curriculum stage is not statistically significant. Under this single-pass protocol, retrieval---not weight training---supplies the majority of attainable gains at this scale, while GRPO reward shaping and a bespoke unit-consistency reward add no statistically separable benefit. A fair multi-prompt re-evaluation places...`
- ⚠️ Abstract hiện 212 từ; thêm ~28 từ → ~240, vẫn <250. **Chạy `verify_paper.py` sau khi sửa để chắc.**

**(b) Related Work** — neo: `GRPO reward shaping, tool augmentation, and self-consistency.`
- Chèn NGAY SAU câu đó (trước `Table~\ref{tabrelated} situates`):
  `We do not propose a new training algorithm; the contribution is a controlled attribution of which standard components actually pay off at 4B scale and which do not, isolated by the retrieval-only control (cfg0-R) and a uniform single-pass protocol.`

**(c) (Tùy chọn) Title** — neo: `Mixed-Domain Scientific Reasoning with a 4B Language Model}`
- Có thể thêm phụ đề: `... with a 4B Language Model: A Controlled Attribution Study}` (cân nhắc độ dài running head — `\titlemark` giữ nguyên 'Retrieval-Augmented 4B Reasoning').

### A2. Caveat logic regression trong nhiễu (ý #2)
- Neo: `than for formal logical inference.` (≈ dòng 414)
- Chèn NGAY SAU dấu chấm:
  `This $2.63$~pp difference corresponds to two of 76 logic items and lies within the non-significant cfg2$\rightarrow$cfg3 transition ($p=0.61$, Table~\ref{tab4}); it should therefore be read as GRPO leaving logic accuracy unchanged rather than as a genuine regression.`

### A3. Hạ vai trò GRPO ở keyword + caveat "strongest" (ý #3)
**(a) Keywords** — neo: `group relative policy optimization |`
- TRƯỚC: `retrieval-augmented generation | group relative policy optimization | curriculum fine-tuning |`
- SAU (bỏ GRPO): `retrieval-augmented generation | curriculum fine-tuning |`
**(b) Results** — neo: `The strongest configuration, cfg3 (SFT, logic SFT, and GRPO), reaches 47.47\%`
- Thêm mệnh đề: `...reaches 47.47\% overall accuracy (its increment over the curriculum stage being within validation-set noise), 17.51~pp above...`

### A4. Làm rõ "không có official test" (ý #4, phần text)
**(a) Limitations 'Validation-set size'** — neo: `including the GRPO-over-SFT increment ($+1.38$~pp, $p=0.61$).`
- Chèn NGAY SAU:
  `The 217 samples are a 90/10 stratified split of the released training data; the official competition test labels were not available, so no independent test-set result is reported. Consequently only the large cfg0$\rightarrow$cfg3 gain ($p=9.3\times10^{-8}$) survives the small-sample caveat, whereas the increments among the fine-tuned configurations do not.`
**(b)** Kiểm tra dòng ~107 `and a held-out evaluation set`: giữ (mô tả task EXACT), nhưng nhờ (a) ở trên đã chống hiểu nhầm. Không cần đổi.

### A5. (Tùy chọn) Một câu efficiency finding từ tab9
- Neo: `Phase-1 SFT provides the highest accuracy per training dollar`
- Đã có sẵn câu này — có thể nâng vào Conclusion nếu muốn nhấn. Không bắt buộc.

### Sau khi xong Track A
```bash
python3 paper/bin/verify_paper.py     # abstract <250, ngoặc cân bằng, cite-key đủ
cp paper/format/WileyDesign/Optimal-Design-layout/TRA_SAE_Wiley.tex paper/TRA_SAE_Wiley.tex  # đồng bộ bản sao
# Đóng gói lại zip (trong thư mục layout):
cd paper/format/WileyDesign/Optimal-Design-layout
zip /content/drive/MyDrive/TRA-SAE/paper/TRA_SAE_ExpertSystems_overleaf.zip TRA_SAE_Wiley.tex references.bib USG.cls
```

---

## 4. TRACK B — KHÔNG CẦN GPU

### B1. Leakage check train↔val (ý #4)
- **Mục tiêu:** chứng minh 217 val không trùng/cận trùng 1945 train.
- **Cách:** script mới (gợi ý `experiments/step10_leakage_check.py`):
  - Đọc `processed_data/exact_train` + `processed_data/exact_val`.
  - (i) TF-IDF cosine / n-gram Jaccard mỗi val vs toàn train → báo max-sim distribution, #cặp > 0.9.
  - (ii) Với mỗi val, kiểm tra 3 retrieval-exemplar (qua `src/retriever.py`) có chứa ground-truth answer không (đề phòng answer leakage).
- **Thêm vào paper:** 1 đoạn ở Experimental Setup hoặc Limitations + bảng nhỏ (max-sim percentile, #near-duplicate).

### B2. Manual error annotation (ý #6) — ĐÒN BẨY CAO NHẤT không GPU
- **Nguyên liệu đã đủ:** `logs/ablation_per_sample_canonical_partB.jsonl` (full generation) + `error_analysis_single_pass.json` (n=117).
- **Việc:**
  1. Re-sample ~60–100 lỗi phủ E6 (extraction/incomplete), E4 (logic), physics — KHÔNG dùng 50 dòng logic-only hiện tại.
  2. Điền `manual_label` ∈ {true-extraction, true-incomplete, genuine-reasoning, unit-error, quantifier-error} + notes.
  3. Báo confusion rule_label vs manual_label; con số then chốt = **% trong E6 thực sự là extraction/length** (đóng đinh claim 72.6%).
- **Cần người chốt cuối.** Claude có thể chạy first-pass LLM-assisted để bạn duyệt nhanh.
- **Thêm vào paper:** thay câu "manual validation left to future work" bằng kết quả thật + 1 bảng confusion; cập nhật tab10 caption.

---

## 5. TRACK C — CẦN GPU (~nửa ngày A100)

### C1. Baseline 7B + retriever (ý #5 — ĐIỂM MẠNH NHẤT của reviewer)
- **Sửa `experiments/step8_fair_baselines.py`:** thêm cờ `--with-retrieval`; trong hàm build message, prepend đúng 3 exemplar mà cfg0-R dùng (qua `src/retriever.py`), giữ nguyên 3 prompt strategy + greedy + extractor.
- **Chạy:** ≥ Qwen2-Math-7B, Qwen2.5-Math-7B, Mistral-7B (× 3 strategy, n=217). Runtime tham chiếu ~18–40 phút/model.
- **Thêm vào paper:** hàng B1-R/B2-R... vào Table 3; cột/đoạn retrieval-delta mỗi baseline; phát biểu lại "fine-tuning-over-retrieval margin" thật (sau khi trừ phần retrieval baseline cũng hưởng).

### C2. External LOGIC benchmark (validate điểm nghẽn logic)
- **Mở rộng `experiments/step9_external_benchmark.py`:** thêm loader cho 1 trong FOLIO / LogiQA / ProofWriter / PrOntoQA (HF datasets).
- **Chạy:** cfg0/cfg2/cfg3 zero-shot transfer.
- **Thêm vào paper:** bảng song song tab6 (MMLU physics) cho logic → chứng minh logic-gap không chỉ do format EXACT.

---

## 6. TRACK D — NGOÀI TẦM (chỉ thừa nhận)
- Official held-out test set: cần label ban tổ chức EXACT 2026, không tự tạo. Đã ghi ở A4. Giữ nguyên trong Limitations.

---

## 7. CÁI KHÔNG ĐỘNG VÀO (đã ổn — đừng sửa lại)
Framing empirical · GRPO non-significance (đã báo) · cfg0-R retrieval isolation · multi-prompt baseline best-of-three · Wilson CI + multi-seed · logic-gap disclosure · negative results (SC, Dual-LoRA) · APA citations · history dates (PDF không in) · badge/masthead (đã gỡ OPEN ACCESS + Allergy).

---

## 8. CHECKLIST THỰC THI
- [x] A1 framing (abstract + related work) — abstract giờ 241 từ (<250)  ✅ 2026-06-12
- [x] A2 caveat logic regression  ✅ 2026-06-12
- [x] A3 keyword (bỏ GRPO) + "strongest" caveat  ✅ 2026-06-12
- [x] A4 wording official test (Limitations)  ✅ 2026-06-12
- [x] A1c phụ đề title `--- A Controlled Attribution Study` (em-dash)  ✅ 2026-06-12
- [x] A5 câu efficiency vào Conclusion (Phase-1 SFT best accuracy/dollar, <US$4)  ✅ 2026-06-12
- [x] Đồng bộ bản sao + đóng gói lại zip  ✅ 2026-06-12
- [x] B1 leakage check (script `experiments/step10_leakage_check.py` + đoạn "Train--validation overlap" trong Dataset)  ✅ 2026-06-12
- [x] B2 — **THEO LỰA CHỌN: CHỈ THỪA NHẬN** (user 2026-06-12). KHÔNG chạy annotation. Đã thêm câu rõ "no human or expert annotation" vào bullet Limitations Extraction-bounded.  ✅
- [x] C1 baseline+retriever (step8 `--with-retrieval` + bảng `tabretr` + subsection "Does retrieval explain the baseline gap?")  ✅ 2026-06-12
- [x] C2 external logic benchmark (step9 `--benchmark logic` FOLIO + bảng `tabfolio`)  ✅ 2026-06-12
- [x] D thừa nhận official test (đã gộp vào A4)  ✅
- [x] Compile lại — **PDF 10 trang, 0 undefined, 0 lỗi**  ✅ 2026-06-12

---

## 8c. KẾT QUẢ TRACK C — SỐ THẬT (2026-06-12, A100-40GB)
> C1: `logs/fair_baselines_retrieval_results_latest.json` · C2: `logs/external_benchmark_logic_results_latest.json`

**C1 — baseline 7B + retrieval (best-of-three, n=217). Map theo TÊN model (ID step8 ≠ ID paper):**
| Paper ID | Model | no-R | +R | Δ |
|---|---|---|---|---|
| B1 | Qwen2-Math | 29.03 | **30.88** | +1.85 |
| B2 | Qwen2.5-Math | 24.88 | 27.65 | +2.77 |
| B4 | Mistral | 16.59 | 16.59 | +0.00 |
| B5 | DeepSeek-Math | 12.44 | 21.66 | +9.22 |
| B6 | Llemma | 11.52 | 10.60 | −0.92 |
- Best baseline+retrieval = 30.88% (B1) — VẪN thua cfg0-R 39.17 (−8.29) & cfg3 47.47 (−16.59). Gemini (B3 paper) KHÔNG chạy retrieval (API diagnostic).
- → Kết luận paper: lợi thế KHÔNG do retrieval. Trung bình retrieval-delta +2.58pp.

**C2 — FOLIO logic transfer (203 val, majority ≈35.5%):**
- cfg0 34.48 · **cfg2 48.28 (+13.80)** · cfg3 46.80 (+12.32). Logic curriculum generalize THẬT; GRPO không giúp logic (cfg2→cfg3 −1.48).

**Sự cố đã xử lý:** C1 chết run 12 (host-RAM tích lũy 15 model/1 process) → resume process nhỏ ≤3 run. C2 cfg2 chết do **torchao 0.10.0 xung đột peft** → `pip uninstall torchao` (LoRA bf16 không cần torchao). Cả hai có backup + merge.
**Scripts mới:** `experiments/step8_fair_baselines.py --with-retrieval` · `experiments/step9_external_benchmark.py --benchmark logic` · `_c1_resume_and_merge.sh` · `_c2_fix_cfg23.sh`.

---

## 8b. KẾT QUẢ LEAKAGE CHECK (B1) — SỐ THẬT (2026-06-12)
> Nguồn: `logs/leakage_check_results.json` · script `experiments/step10_leakage_check.py` (no-GPU, TF-IDF + char-8gram + retriever).
> ⚠️ KHÔNG sạch tuyệt đối — đã báo cáo TRUNG THỰC trong paper (không tuyên bố "zero leakage").
- 3 câu trùng verbatim (val∈train): 2 cùng đáp án, 1 khác đáp án (capacitor = ví dụ E6).
- 66/217 cosine ≥0.90; 36 ≥0.95 — NHƯNG 30/36 là *cùng template khác số* (TF-IDF mất chữ số mũ → trùng giả).
- char-8gram ≥0.95 (giữ chữ số): chỉ **6** thật sự gần-giống; **5/217 (2.3%)** vừa gần-giống vừa trùng đáp án = leakage thật.
- Retriever (cfg0-R): top-1 exemplar cosine mean 0.676, 3 verbatim exemplar (= 3 câu trùng).
- Kết luận paper: ~2% overlap quá nhỏ để giải thích gain 17.51pp; bug cache retriever đã né (build cache_path=None).

---

## 9. LỆNH HAY DÙNG
| Việc | Lệnh |
|---|---|
| Verify paper | `python3 paper/bin/verify_paper.py` |
| Tái lập số single-pass | xem block ở Mục 0 (RESUME) |
| Compile PDF | `bash paper/bin/setup_texlive.sh` rồi `bash paper/bin/compile_paper.sh` |
| Đóng gói Overleaf | trong `Optimal-Design-layout/`: `zip $ZIP TRA_SAE_Wiley.tex references.bib USG.cls` |
| Kiểm PDF không in dates | `pdftotext paper/.../TRA_SAE_Wiley.pdf - \| grep -i received` (rỗng = OK) |

> Liên kết: bản đánh giá tổng + đối chiếu ngắn nằm trong git history hội thoại;
> file resume gốc của dự án: `TIEP_TUC_PAPER_REVISION.md`.
