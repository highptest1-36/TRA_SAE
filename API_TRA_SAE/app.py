"""
TRA-SAE EXACT 2026 prediction API (FastAPI)
==========================================
Endpoints
  GET  /health   → readiness probe.
  POST /predict  → body = a single query object OR a JSON list of them.
                   Always returns a JSON list with one result object per query:
                   {query_id, answer, unit, explanation, premises_used, reasoning}.
                   `explanation` is always non-empty; `reasoning` may be null.

Run:
  CUDA_VISIBLE_DEVICES=0 uvicorn API_TRA_SAE.app:app --host 0.0.0.0 --port 8000
  (or use API_TRA_SAE/run_server.sh which also opens a cloudflared tunnel)
"""
from __future__ import annotations

import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from API_TRA_SAE.predict_core import PREDICTOR  # noqa: E402

app = FastAPI(
    title="TRA-SAE EXACT 2026 — /predict API",
    version="1.0",
    description=(
        "Interactive tester for the TRA-SAE prediction API.\n\n"
        "**How to test:** expand **POST /predict** below → click **Try it out** "
        "→ edit the example questions → click **Execute**. The response is a JSON "
        "list with one object per query: `query_id, answer, unit, explanation, "
        "premises_used, reasoning`.\n\n"
        "Each query accepts: `query_id`, `question`, `type` "
        "(`physics`|`logic`, optional), `premises` (list, for logic)."
    ),
)

# Pre-filled example shown in the Swagger /docs "Try it out" body box (BTC schema).
_EXAMPLE_BODY = [
    {"query_id": "T1_0001", "type": "type1",
     "query": "Is Student A eligible for graduation?",
     "premises": ["A student who has completed at least 120 credits is eligible for graduation.",
                  "Student A has completed 118 credits."],
     "options": ["Yes", "No", "Uncertain"]},
    {"query_id": "T2_0001", "type": "type2",
     "query": "Calculate the energy stored in capacitor C when C = 100 uF and U = 30 V.",
     "premises": [], "options": []},
]


@app.on_event("startup")
def _startup() -> None:
    if not PREDICTOR._ready:
        print("[app] Loading model on startup...", flush=True)
        PREDICTOR.load()
        print("[app] Ready.", flush=True)


@app.get("/")
def root():
    return {"service": "TRA-SAE EXACT 2026 /predict",
            "test_ui": "/docs  (open this in a browser to send test questions)",
            "predict": "POST /predict  (JSON list; the BTC endpoint)",
            "health": "/health"}


@app.get("/predict")
def predict_help():
    # Browsers do GET when you visit a URL; the real endpoint is POST.
    return JSONResponse(status_code=405, content={
        "detail": "Use POST for /predict. To test in a browser, open /docs "
                  "→ POST /predict → Try it out → Execute."})


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if PREDICTOR._ready else "loading",
            "device": PREDICTOR.device}


@app.post("/predict",
          summary="Answer one or more queries (BTC endpoint)",
          openapi_extra={"requestBody": {"content": {"application/json": {
              "example": _EXAMPLE_BODY}}}})
async def predict(request: Request):
    payload = await request.json()
    if not PREDICTOR._ready:  # cold path safety
        PREDICTOR.load()
    t0 = time.time()
    results = PREDICTOR.predict(payload)
    n = len(results)
    print(f"[app] /predict served {n} query(ies) in {time.time() - t0:.1f}s", flush=True)
    return JSONResponse(content=results)


# ── OpenAI-compatible endpoints (BTC verifies model identity via /v1/models) ──
_MODEL_ID = os.environ.get("SERVE_MODEL_ID", "Qwen/Qwen3.5-4B")


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible model list — BTC checks this to verify model identity."""
    served = getattr(PREDICTOR, "model_name", None) or _MODEL_ID
    return {"object": "list", "data": [{
        "id": served, "object": "model", "created": 0, "owned_by": "trasae",
        "permission": [],
        "metadata": {"base": served, "adapters": ["physics", "logic"],
                     "active_params": "~4B", "framework": "transformers+peft (OpenAI-compatible)"},
    }]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Minimal OpenAI-compatible chat endpoint backed by the served model."""
    if not PREDICTOR._ready:
        PREDICTOR.load()
    body = await request.json()
    msgs = body.get("messages", [])
    user = next((m.get("content", "") for m in reversed(msgs)
                 if m.get("role") == "user"), "")
    subject = "logic" if PREDICTOR._route_subject("", user) == "logic" else "physics"
    text = PREDICTOR.raw_generate(user, subject=subject)
    return {
        "id": "chatcmpl-trasae", "object": "chat.completion", "created": 0,
        "model": getattr(PREDICTOR, "model_name", None) or _MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
