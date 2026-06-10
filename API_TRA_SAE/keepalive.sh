#!/usr/bin/env bash
# ============================================================================
# KEEPALIVE WATCHDOG (process-aware — NO false positives).
#
# WHY THIS VERSION: the API runs a single uvicorn worker and each /predict
# blocks for 20-34s (greedy generation). While a query is being served, the
# event loop is busy and /health cannot answer quickly. The OLD watchdog read
# that as "API DOWN" and restarted uvicorn+ngrok — i.e. it KILLED a perfectly
# healthy server in the middle of grading, causing ERR_NGROK_3200 flapping.
#
# This version restarts ONLY when the uvicorn OR ngrok *process* is actually
# gone (a real crash / OOM / Colab kill). A busy-but-alive server is never
# touched. /health is still pinged each cycle (best-effort, long timeout) as a
# liveness log + anti-idle, but its result NEVER triggers a restart.
#
# It only ever (re)starts uvicorn/ngrok via start_colab.sh — never other jobs.
#
# Usage:  !bash API_TRA_SAE/keepalive.sh    (leave running; stop with the ⏹ button)
# ============================================================================
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${NGROK_DOMAIN:?Set NGROK_DOMAIN in .env}"
URL="https://${NGROK_DOMAIN}/health"
INTERVAL="${KEEPALIVE_INTERVAL:-20}"

echo "[keepalive] process-aware watchdog on $URL (every ${INTERVAL}s)."
echo "[keepalive] restarts ONLY if uvicorn/ngrok process dies; a busy server is"
echo "[keepalive] never killed. Leave this running. Stop with the square button."

while true; do
  uv=$(pgrep -f "uvicorn API_TRA_SAE.app" | head -1)
  ng=$(pgrep -f "ngrok http"        | head -1)

  if [ -n "$uv" ] && [ -n "$ng" ]; then
    # Both processes alive => server is up (possibly busy). Best-effort health
    # ping with a generous timeout so a long in-flight /predict doesn't look bad.
    resp=$(curl -fsS -m 50 "$URL" 2>/dev/null || echo "")
    if echo "$resp" | grep -q '"status":"ok"'; then
      echo "[$(date +%H:%M:%S)] OK   — uvicorn=$uv ngrok=$ng /health ok"
    elif echo "$resp" | grep -q '"status":"loading"'; then
      echo "[$(date +%H:%M:%S)] LOAD — uvicorn=$uv ngrok=$ng model still loading"
    else
      echo "[$(date +%H:%M:%S)] BUSY — uvicorn=$uv ngrok=$ng alive, /health slow (NOT restarting)"
    fi
  else
    echo "[$(date +%H:%M:%S)] DOWN — uvicorn='${uv:-gone}' ngrok='${ng:-gone}' -> restarting (same URL)..."
    bash API_TRA_SAE/start_colab.sh >/dev/null 2>&1 || true
    echo "[$(date +%H:%M:%S)] restart done"
  fi
  sleep "$INTERVAL"
done
