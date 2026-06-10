# API_TRA_SAE — HƯỚNG DẪN CHẠY API (EXACT 2026 / NovaWing)

> **Đây là tài liệu DUY NHẤT để chạy lại API.** Lần sau muốn chạy, chỉ cần nói:
> *"đọc `API_TRA_SAE/RUNBOOK.md` và chạy API"* — làm theo §0 là xong.
>
> - Thư mục code: `/content/drive/MyDrive/TRA-SAE/API_TRA_SAE/`
> - URL nộp (cố định, không đổi): `https://alibi-aneurism-dupe.ngrok-free.dev/predict`
> - Secrets ở `.env` tại **gốc repo** (`/content/drive/MyDrive/TRA-SAE/.env`), KHÔNG nằm trong thư mục này.
> - Mọi config & các bản vá đã được "nướng" sẵn vào script → khởi động lại là ra **đúng trạng thái đã chạy được**.

---

## ⚡ 0. CHẠY NHANH / KHỞI ĐỘNG LẠI

Cần một runtime Colab **A100/H100** (GPU ≥ ~10 GB trống). Có 2 cách chạy — chọn 1.

### Cách A — Chạy qua Claude Code / terminal (KHUYẾN NGHỊ)

Chạy lần lượt từ gốc repo. **Quan trọng:** `start_colab.sh` phải chạy NỀN (background) và **chỉ chạy 1 lần** (chạy 2 lần song song sẽ tạo 2 tunnel → lỗi `ERR_NGROK_6030`).

```bash
cd /content/drive/MyDrive/TRA-SAE

# 1) Cài deps + gỡ torchao cũ + cài ngrok (tự bỏ qua nếu đã có). Bắt buộc khi VM mới.
bash API_TRA_SAE/setup_colab.sh

# 2) Bật API + ngrok (chạy NỀN). Đợi tới khi /health = ok. Lần đầu VM mới phải
#    tải base model ~8 GB nên lâu (vài phút); các lần sau model đã cache → ~30-60s.
nohup bash API_TRA_SAE/start_colab.sh > API_TRA_SAE/logs/start.out 2>&1 &
#    Theo dõi:  tail -f API_TRA_SAE/logs/uvicorn.log   (chờ dòng "API IS LIVE" / health ok)

# 3) Kiểm tra thật qua URL công khai: 6 case kiểu BTC, phải < 60s/câu.
python API_TRA_SAE/_diag_btc6.py https://alibi-aneurism-dupe.ngrok-free.dev/predict

# 4) Bật watchdog GIỮ SỐNG (chạy nền, để nguyên suốt buổi chấm).
nohup bash API_TRA_SAE/keepalive.sh > API_TRA_SAE/logs/keepalive.log 2>&1 &
```

**Kiểm tra "đã sẵn sàng":**
```bash
curl -s https://alibi-aneurism-dupe.ngrok-free.dev/health      # -> {"status":"ok","device":"cuda"}
curl -s https://alibi-aneurism-dupe.ngrok-free.dev/v1/models   # -> id: Qwen/Qwen3.5-4B
ps -e -o pid,cmd | grep -E "uvicorn API_TRA_SAE.app|ngrok http 8000|keepalive.sh" | grep -v grep
# Phải thấy ĐÚNG 1 uvicorn, ĐÚNG 1 ngrok, 1 keepalive.
```

### Cách B — Chạy trong notebook Colab (6 cell)

```python
# Cell 1
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/TRA-SAE
```
```bash
!nvidia-smi --query-gpu=memory.used,memory.free --format=csv   # Cell 2: cần ≥10GB trống
!bash API_TRA_SAE/setup_colab.sh                               # Cell 3: deps + gỡ torchao + ngrok
!bash API_TRA_SAE/start_colab.sh                               # Cell 4: chờ in "API IS LIVE"
!python API_TRA_SAE/_diag_btc6.py https://alibi-aneurism-dupe.ngrok-free.dev/predict  # Cell 5
!bash API_TRA_SAE/keepalive.sh                                 # Cell 6: ĐỂ CHẠY MÃI, giữ tab mở
```
Sẵn sàng khi: Cell 4 in **"API IS LIVE"**, Cell 5 mọi case **< 60s**, Cell 6 in `OK — ... /health ok`.
**Giữ tab Colab mở suốt buổi chấm.**

### Dừng sau khi chấm xong (KHÔNG dùng `pkill python` — sẽ trúng job GPU khác)
```bash
pkill -f 'uvicorn API_TRA_SAE.app'; pkill -f 'ngrok http'
# và dừng watchdog: kill <pid keepalive>  (xem bằng: ps -e -o pid,cmd | grep keepalive.sh | grep -v grep)
```

---

## 📤 1. NỘP GÌ CHO BTC

| Mục | Giá trị |
|---|---|
| **Prediction API URL** | `https://alibi-aneurism-dupe.ngrok-free.dev/predict` |
| **vLLM model URL** | `https://alibi-aneurism-dupe.ngrok-free.dev/v1/models` |
| **Gói nộp (ZIP)** | `API_TRA_SAE/submission/NovaWing.zip` |
| **Test UI (trình duyệt)** | `https://alibi-aneurism-dupe.ngrok-free.dev/docs` (bấm qua cảnh báo ngrok 1 lần) |

URL **cố định** (gắn với tài khoản ngrok) → mỗi lần khởi động lại đều y hệt, **không cần báo lại BTC**.
Build lại gói nộp (tùy chọn): `python API_TRA_SAE/build_package.py NovaWing`.

---

## 🧩 2. HỆ THỐNG NÀY LÀ GÌ (mô tả)

- **Model:** base `Qwen/Qwen3.5-4B` (mã nguồn mở) + **2 LoRA adapter** (physics, logic) + **router** chọn môn.
  Tổng tham số active ~4B (< 8B). Suy luận **1 pass** (greedy), không self-consistency, không Z3.
- **Đường suy luận:** `type1`→adapter logic; `type2`→adapter physics; nếu thiếu `type` thì router tự định tuyến theo nội dung.
- **`POST /predict`** nhận 1 object JSON **hoặc** 1 list; luôn trả về **list** gồm các object:
  `{query_id, answer, unit, explanation, premises_used, reasoning}`.
  - **type1** (logic): `unit=""`; `premises_used` = chỉ số premise dùng (**0-based**).
    options Yes/No/Uncertain hoặc MCQ → `answer` = đúng 1 option; options rỗng → trả số/text trực tiếp.
  - **type2** (physics): `answer` = số (dạng khoa học cho số rất nhỏ/lớn, vd `3.38e-3`); `unit` ASCII (A, V, ohm, uF, V/m, J, W); `premises_used=[]`.
  - `explanation` luôn khác rỗng; `reasoning` = `{type, steps}` (fol/cot) hoặc null.
- **`GET /v1/models`** → `id: Qwen/Qwen3.5-4B` (BTC kiểm định danh tính model).
- **Độ trễ:** 1 worker, greedy 1 pass + giới hạn token + ngân sách thời gian + dừng sớm → **< 60s/câu**
  (đo thật qua ngrok: ~20–34s). Code chỉ đọc `src/` (read-only), không ghi gì ngoài `API_TRA_SAE/logs/`.

**Các knob đã nướng vào script + code mặc định:**
`SERVE_MAX_NEW_TOKENS=384`, `SERVE_TIME_BUDGET_SEC=34`, `SERVE_PROMPT_MAX_LEN=3072`,
`SERVE_FEWSHOT_PHYSICS=0`, `SERVE_FEWSHOT_LOGIC=1`.

---

## 🛡️ 3. WATCHDOG `keepalive.sh` (đã sửa — RẤT QUAN TRỌNG)

Watchdog này khác bản cũ và là lý do API không còn "rớt mạng" giữa chừng:

- **Chỉ restart khi TIẾN TRÌNH uvicorn/ngrok thực sự CHẾT** (crash/OOM/Colab kill).
- Server đang **bận** trả lời 1 câu /predict (20–34s) → `/health` trả lời chậm là **bình thường**;
  watchdog **KHÔNG** restart trong trường hợp này.
- `/health` vẫn được ping mỗi 20s (timeout dài) chỉ để ghi log + chống idle, **không** dùng để quyết định restart.

> ⛔ **Lỗi cũ (đã bỏ):** bản keepalive cũ hiểu "/health chậm = API chết" → cứ ~2 phút lại giết
> uvicorn+ngrok dù server đang chấm dở cho BTC → BTC thấy `ERR_NGROK_3200` (offline) liên tục.
> Bản hiện tại đã loại bỏ hoàn toàn lỗi dương tính giả này.

---

## 🛠️ 4. CONFIG & CÁC BẢN VÁ (vì sao các giá trị hiện tại) — lịch sử để lần sau khỏi vấp lại

1. **Latency cap** — `MAX_NEW_TOKENS 512→384`, `TIME_BUDGET_SEC 48→34`. Worst-case ~32s, dư biên dưới 60s.
   BTC timeout ở 60s; cấu hình cũ (512/48s) chạy MCQ/physics ~44–46s + hop ngrok → vượt 60s → 0 điểm 2 câu.
2. **Tách độ dài input** — thêm `PROMPT_MAX_LEN=3072`, độc lập với `MAX_NEW_TOKENS`. Trước đây input bị buộc
   `max_length = MAX_NEW_TOKENS*2`, hạ token output sẽ **cắt mất câu hỏi/premise** → đáp án rác. **ĐỪNG nối lại 2 thông số này.**
3. **Format số type-2** — `_fmt_number()` trả số sạch / ký hiệu khoa học (vd `3.38e-3`), bỏ rác kiểu `0.0033799999…`.
4. **Đơn vị ASCII** — `µ`→`u`, `Ω`→`ohm` (trước bị nuốt thành `F`/rỗng).
5. **(MỚI) Gỡ torchao cũ** — VM Colab mới có sẵn `torchao 0.10.0`; peft 0.19.1 **raise lỗi** khi nạp LoRA
   (`incompatible version of torchao`, cần >0.16.0) → crash lúc load model. LoRA bf16 **không dùng** torchao,
   nên `setup_colab.sh` đã tự gỡ nó (idempotent). Đừng cài lại torchao.
6. **(MỚI) Watchdog process-aware** — xem §3.

> Khi load model trên VM mới: tự **tải base model ~8 GB** từ HuggingFace (cần `HF_TOKEN` trong `.env`, mạng tới
> huggingface.co). Đường "fast path not available / ~11–12 tok/s" là **bình thường** (eager attention) — các cap đã tính tới.

---

## ✅ 5. KẾT QUẢ ĐÃ VERIFY THẬT (qua URL công khai)

`API_TRA_SAE/_diag_btc6.py` — 6 case kiểu BTC, đo qua ngrok:
- **Latency: max ~34s · mean ~30s · 0 timeout** — tất cả dưới 60s (điều kiện sống còn để BTC chấm được). ✓
- Schema đúng 100% (query_id/answer/unit/explanation/premises_used/reasoning).
- `/health` ok, `/v1/models` → `Qwen/Qwen3.5-4B`.

### Giới hạn đã biết (không chặn chấm, nên biết)
- **Đơn vị physics:** đôi khi trả theo đơn vị có tiền tố (vd `3.384 mJ` = `3.38e-3 J`). Đúng vật lý; pass **nếu** grader quy đổi SI.
- **`premises_used`:** đúng với chuỗi đơn giản (Yes/No, Text); với MCQ/abstention có thể liệt kê cả chuỗi suy luận
  thay vì 1 premise quyết định → mất phần điểm premise của câu đó. (Fix chất lượng model, chưa áp dụng.)
- **Độ chính xác đáp án dao động nhẹ** giữa các lần do time-budget cắt sinh ở mốc khác nhau khi GPU nhanh/chậm.

---

## 🆘 6. TROUBLESHOOTING (tất cả lỗi đã gặp + cách xử)

| Triệu chứng | Nguyên nhân | Cách xử |
|---|---|---|
| Trình duyệt `ERR_NGROK_3200` "endpoint offline" | Tunnel chưa chạy (VM mới/đứt) HOẶC watchdog cũ vừa giết server | Chạy lại §0 (Cách A). Watchdog mới không còn gây lỗi này. |
| `ERR_NGROK_6030` "multiple endpoints… pooling" | Có **2 tiến trình ngrok** cùng ôm 1 domain (chạy `start_colab.sh` 2 lần song song) | Giết hết ngrok rồi bật lại 1 cái: `for p in $(pgrep -f "ngrok ht""tp"); do kill -9 $p; done` rồi chạy lại §0 bước 2. **Chỉ chạy start_colab.sh 1 lần.** |
| Crash lúc load: `incompatible version of torchao` | VM mới có torchao 0.10.0 cũ | `pip uninstall -y torchao` (đã tự động trong `setup_colab.sh`). |
| `NGROK_DOMAIN / NGROK_AUTHTOKEN missing` | `.env` chưa nạp | Kiểm tra `/content/drive/MyDrive/TRA-SAE/.env` có 2 dòng đó, **không có dấu cách quanh `=`**. |
| ngrok "tunnel already online / session failed" | ngrok cũ còn sống (free tier = 1 tunnel) | Giết ngrok cũ (xem ô trên) rồi chạy lại. |
| Lỗi import `qwen3_5` / model load | Thiếu deps trên VM mới | `bash API_TRA_SAE/setup_colab.sh` (cần `transformers==5.10.x`). |
| API chết giữa buổi | crash thật/OOM | Watchdog `keepalive.sh` tự dựng lại trong ~30-60s, cùng URL; hoặc chạy lại §0 bước 2. |
| GPU < ~8 GB trống | job khác đang chiếm | Đợi/đổi runtime. **Đừng** kill job khác. start/keepalive chỉ kill `uvicorn API_TRA_SAE.app` + `ngrok http`. |

**Kiểm tra số tunnel đang mở:** `curl -s http://localhost:4040/api/tunnels` (chỉ nên có 1).

---

## 📁 7. FILE TRONG THƯ MỤC NÀY

| File | Vai trò |
|---|---|
| `app.py` | FastAPI: `/predict`, `/health`, `/v1/models`, `/docs`. Module = `API_TRA_SAE.app`. |
| `predict_core.py` | Lõi suy luận 1-pass (dual-LoRA + router + retrieval). Chứa các knob mặc định. |
| `setup_colab.sh` | Cài deps pinned + **gỡ torchao cũ** + (start sẽ cài ngrok). Idempotent. |
| `start_colab.sh` | Bật uvicorn + mở tunnel ngrok cố định + warm. Chạy NỀN. |
| `keepalive.sh` | Watchdog **process-aware** (chỉ restart khi tiến trình chết). |
| `run_server.sh` | Bản chạy foreground (ít dùng; ưu tiên `start_colab.sh`). |
| `_diag_btc6.py` | Verify 6 case BTC qua URL (latency + đáp án). |
| `smoke_test.py` | Sweep schema/latency rộng hơn (tùy chọn). |
| `urls.txt` | URL nộp. |
| `submission/NovaWing.zip` | Gói nộp BTC. |
| `solution.pdf` / `solution.tex` | Bài giải nộp kèm. |
| `notation_mapping.csv` + `gen_notation_mapping.py` | Bảng ký hiệu BTC. |
| `logs/` | Log runtime (`uvicorn.log`, `ngrok.log`, `keepalive.log`). |

---

## 🧹 8. GHI CHÚ DỌN DẸP

- Thư mục cũ `serve/` (cùng nội dung) có thể vẫn còn trên Drive nếu lúc đổi tên API đang **chạy live** cho BTC.
  Sau khi xong buổi chấm, có thể xóa an toàn: `rm -rf /content/drive/MyDrive/TRA-SAE/serve`.
  Từ nay chỉ dùng **`API_TRA_SAE/`** là thư mục chính thức.
