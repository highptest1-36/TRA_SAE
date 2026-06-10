#!/usr/bin/env bash
# Start the TRA-SAE /predict API behind a FIXED ngrok URL.
# Run ONLY when the GPU is free of paper experiments (see API_TRA_SAE/README_API.md).
#
# Requires (provide once):
#   NGROK_AUTHTOKEN   your ngrok authtoken      (kept in .env, never committed)
#   NGROK_DOMAIN      your reserved static domain, e.g. trasae.ngrok-free.app
# The public endpoint to submit to BTC will be:  https://$NGROK_DOMAIN/predict
set -euo pipefail
cd "$(dirname "$0")/.."

# Load secrets from .env if present (NGROK_AUTHTOKEN, NGROK_DOMAIN, HF_TOKEN...)
[ -f .env ] && set -a && . ./.env && set +a

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SERVE_MAX_NEW_TOKENS="${SERVE_MAX_NEW_TOKENS:-384}"
export SERVE_TIME_BUDGET_SEC="${SERVE_TIME_BUDGET_SEC:-34}"
export SERVE_PROMPT_MAX_LEN="${SERVE_PROMPT_MAX_LEN:-3072}"
PORT="${PORT:-8000}"

: "${NGROK_DOMAIN:?Set NGROK_DOMAIN (your reserved ngrok static domain) in .env}"
: "${NGROK_AUTHTOKEN:?Set NGROK_AUTHTOKEN in .env}"

echo "[run_server] FIXED public endpoint: https://$NGROK_DOMAIN/predict"
echo "[run_server] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  port=$PORT"

# 1) Start the model server (loads on startup, warm-up included).
uvicorn API_TRA_SAE.app:app --host 0.0.0.0 --port "$PORT" --workers 1 &
UV_PID=$!
trap 'kill $UV_PID $NGROK_PID 2>/dev/null || true' EXIT

# 2) Wait for /health = ok.
echo "[run_server] loading model (this is the slow part)..."
for i in $(seq 1 180); do
  if curl -fsS "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[run_server] model ready."; break
  fi
  sleep 5
done

# 3) Install ngrok agent if missing.
if ! command -v ngrok >/dev/null 2>&1; then
  echo "[run_server] installing ngrok agent..."
  ARCH=$(uname -m); [ "$ARCH" = "x86_64" ] && ARCH=amd64
  wget -q "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-${ARCH}.tgz" -O /tmp/ngrok.tgz
  tar -xzf /tmp/ngrok.tgz -C /usr/local/bin && chmod +x /usr/local/bin/ngrok
fi

ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true

# 4) Open the FIXED tunnel (same URL every run).
echo "[run_server] opening ngrok tunnel on https://$NGROK_DOMAIN ..."
ngrok http "$PORT" --domain="$NGROK_DOMAIN" --log=stdout &
NGROK_PID=$!

sleep 6
# Warm the tunnel: the FIRST request through a fresh ngrok tunnel has ~20s
# cold-start overhead. Fire one throwaway request so the BTC's first real
# sample isn't penalised.
echo "[run_server] warming tunnel (first request is slow by design)..."
curl -sS -m 90 -X POST "https://$NGROK_DOMAIN/predict" -H 'Content-Type: application/json' \
  -d '[{"query_id":"_warm","type":"physics","question":"Calculate 1 + 1."}]' >/dev/null 2>&1 || true
echo ""
echo "============================================================"
echo "  SUBMIT THIS to the EXACT portal (Prediction API URL):"
echo "     https://$NGROK_DOMAIN/predict"
echo "  Verify:  python API_TRA_SAE/smoke_test.py --url https://$NGROK_DOMAIN/predict --n 10"
echo "  Keep this process running through your judging slot."
echo "============================================================"
wait $UV_PID
