#!/usr/bin/env bash
# ============================================================================
# ONE-TIME SETUP per fresh Colab runtime. Idempotent: if the model libraries
# already support Qwen3.5 it does nothing; otherwise it installs the pinned
# versions that were validated for this project.  Run AFTER mounting Drive:
#
#   !bash API_TRA_SAE/setup_colab.sh
#
# Then start the API with:  !bash API_TRA_SAE/start_colab.sh
# ============================================================================
cd "$(dirname "$0")/.."

# Some fresh Colab base images ship an old torchao (0.10.0). peft 0.19.1's
# adapter dispatcher RAISES on any torchao < 0.16.0 instead of skipping it,
# which crashes model load ("incompatible version of torchao"). The bf16 LoRA
# adapters never use torchao, so just remove the incompatible copy — peft then
# treats it as unavailable and loads normally. Idempotent (no-op if absent).
if python - <<'PY' 2>/dev/null
import importlib.util, sys
spec = importlib.util.find_spec("torchao")
if spec is None:
    sys.exit(1)  # not installed -> nothing to do
from importlib.metadata import version
from packaging.version import parse
sys.exit(0 if parse(version("torchao")) < parse("0.16.0") else 1)
PY
then
  echo "[setup] removing incompatible torchao (<0.16.0) so peft can load LoRA..."
  pip -q uninstall -y torchao 2>&1 | tail -1
fi

if python -c "import transformers.models.qwen3_5" 2>/dev/null; then
  echo "[setup] transformers already supports qwen3_5 — deps OK, skipping install."
  exit 0
fi

echo "[setup] fresh runtime detected — installing pinned dependencies..."
pip -q install \
  "transformers==5.10.1" "peft==0.19.1" "datasets==4.0.0" \
  "accelerate==1.13.0" "scikit-learn==1.6.1" "scipy==1.16.3" \
  "fastapi==0.136.3" "uvicorn==0.49.0" 2>&1 | tail -3

if python -c "import transformers.models.qwen3_5" 2>/dev/null; then
  echo "[setup] OK — transformers now supports qwen3_5."
else
  echo "[setup] WARNING: qwen3_5 still not importable. The base torch may be too"
  echo "        old for Qwen3.5; if model load fails, also run:"
  echo "        pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128"
fi
