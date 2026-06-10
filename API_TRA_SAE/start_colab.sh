#!/usr/bin/env bash
# ============================================================================
# RECONNECT SCRIPT — run this in a Colab cell after a disconnect to bring the
# EXACT 2026 API back up at the SAME fixed URL. Non-blocking: starts uvicorn +
# ngrok in the background (they survive after the cell finishes) and returns.
#
#   !bash API_TRA_SAE/start_colab.sh
#
# Needs .env with NGROK_AUTHTOKEN + NGROK_DOMAIN (already on Drive).
# Run API_TRA_SAE/setup_colab.sh FIRST only if the runtime is fresh (deps reset).
# ============================================================================
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SERVE_MAX_NEW_TOKENS="${SERVE_MAX_NEW_TOKENS:-384}"
export SERVE_TIME_BUDGET_SEC="${SERVE_TIME_BUDGET_SEC:-34}"
export SERVE_PROMPT_MAX_LEN="${SERVE_PROMPT_MAX_LEN:-3072}"
PORT="${PORT:-8000}"
mkdir -p API_TRA_SAE/logs

if [ -z "${NGROK_DOMAIN:-}" ] || [ -z "${NGROK_AUTHTOKEN:-}" ]; then
  echo "ERROR: NGROK_DOMAIN / NGROK_AUTHTOKEN missing in .env"; exit 1
fi

echo "[start] stopping any old instances..."
pkill -9 -f "uvicorn API_TRA_SAE.app" 2>/dev/null || true
pkill -9 -f "ngrok http"        2>/dev/null || true
sleep 3

echo "[start] launching uvicorn (model loads on startup; ~30-60s)..."
nohup uvicorn API_TRA_SAE.app:app --host 0.0.0.0 --port "$PORT" --workers 1 \
  > API_TRA_SAE/logs/uvicorn.log 2>&1 &
echo "[start] uvicorn pid $!"

for i in $(seq 1 180); do
  if curl -fsS "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[start] model READY"; break
  fi
  sleep 5
done

if ! command -v ngrok >/dev/null 2>&1; then
  echo "[start] installing ngrok agent..."
  wget -q "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz" -O /tmp/ngrok.tgz
  tar -xzf /tmp/ngrok.tgz -C /usr/local/bin && chmod +x /usr/local/bin/ngrok
fi
ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true

echo "[start] opening FIXED ngrok tunnel on $NGROK_DOMAIN ..."
nohup ngrok http "$PORT" --domain="$NGROK_DOMAIN" --log=stdout \
  > API_TRA_SAE/logs/ngrok.log 2>&1 &
echo "[start] ngrok pid $!"
sleep 6

echo "[start] warming tunnel (first request is slow by design)..."
curl -sS -m 90 -X POST "https://$NGROK_DOMAIN/predict" -H 'Content-Type: application/json' \
  -d '[{"query_id":"_warm","type":"type2","query":"Calculate 1+1.","premises":[],"options":[]}]' \
  >/dev/null 2>&1 || true

echo ""
echo "============================================================"
echo "  API IS LIVE — submit / keep alive during the BTC slot:"
echo "    Prediction API URL : https://$NGROK_DOMAIN/predict"
echo "    vLLM model URL     : https://$NGROK_DOMAIN/v1/models"
echo "  uvicorn + ngrok run in the BACKGROUND (survive this cell)."
echo "  Verify:  !python API_TRA_SAE/smoke_test.py --url https://$NGROK_DOMAIN/predict --n 6 --gap 4"
echo "  Stop:    !pkill -f 'uvicorn API_TRA_SAE.app'; pkill -f 'ngrok http'"
echo "============================================================"
