# TRA-SAE — EXACT 2026 `/predict` API

Self-contained serving layer for the EXACT 2026 competition. **Does not modify
any paper / experiment code.** Lives entirely under `API_TRA_SAE/`.

## What it does
- Loads the production model **once**: dual-LoRA specialists
  (`checkpoints/qwen35_grpo_physics/final` + `qwen35_grpo_logic/final`) on
  base `Qwen/Qwen3.5-4B`, with a TF-IDF subject router + few-shot retriever.
- Per query: route subject → switch LoRA adapter → retrieve 3 few-shots →
  **single greedy pass** → extract the answer/explanation/reasoning.
- Mirrors the proven-correct path of `experiments/step0_canonical_eval.py`
  (config 4). **No** LangGraph, **no** self-consistency, **no** Z3 (all either
  buggy, too slow for 60s, or need ground truth).

## API contract
`POST /predict` — body is a single query object **or** a JSON list of them:
```json
[{"query_id": "TD401", "type": "physics", "question": "...", "premises": []}]
```
Returns a JSON **list**, one object per query:
```json
[{"query_id":"TD401","answer":"45","unit":"J",
  "explanation":"...","premises_used":[],"reasoning":"..."}]
```
- `explanation` is always non-empty. `reasoning` may be `null`.
- `type`/`premises` are optional — subject is inferred by the router if absent.

`GET /health` → `{"status":"ok"|"loading","device":"cuda"}`.

## Latency (<= 60s/query)
- `SERVE_MAX_NEW_TOKENS=512` (env), `stop_strings=["</explanation>"]`, greedy,
  hard wall-clock watchdog `SERVE_TIME_BUDGET_SEC=50`, warm-up on startup.
- Baseline measured single-pass @1024 tok = 45s; @512 tok ≈ 20–25s.

## GPU / paper safety
- Run the server **only when the GPU is free** of paper experiments
  (model needs ~8 GB; co-residence with a 32 GB run risks OOM + >60s latency).
- Process pins `CUDA_VISIBLE_DEVICES=0` and `expandable_segments:True`.
- Retriever is built `cache_path=None` (in-memory; never repickles
  `logs/retriever_cache.pkl`). Nothing is written to the repo-root `logs/`.

## Run (FIXED ngrok URL — same address every run)
One-time on the [ngrok dashboard](https://dashboard.ngrok.com): create a free
account, copy your **authtoken**, and **reserve a free static domain** (e.g.
`trasae.ngrok-free.app`). Put both in `.env`:
```bash
NGROK_AUTHTOKEN=2x...your_token
NGROK_DOMAIN=trasae.ngrok-free.app
```
Then (only when the GPU is free of paper experiments):
```bash
bash API_TRA_SAE/run_server.sh          # loads model, opens the FIXED tunnel
# submit to the EXACT portal:  https://$NGROK_DOMAIN/predict
python API_TRA_SAE/smoke_test.py --url https://$NGROK_DOMAIN/predict --n 10
```
The URL is bound to your ngrok account, so it is **identical on every restart** —
if Colab disconnects, just re-run and the same URL works (no need to re-submit).
The server must be **up during your judging slot**. ngrok's browser-warning page
does not affect programmatic JSON API requests.

## Files
| File | Purpose |
|---|---|
| `predict_core.py` | model load + `predict_one()` (the inference logic) |
| `app.py` | FastAPI `/predict` + `/health` |
| `notation_mapping.csv` | canonical_latex → meaning → your_notation |
| `gen_notation_mapping.py` | regenerates the CSV from `data/` |
| `smoke_test.py` | BTC-style API check (schema + <60s latency + scoring) |
| `run_server.sh` | uvicorn + cloudflared launcher |
| `urls.txt` | URLs file for the submission package |
