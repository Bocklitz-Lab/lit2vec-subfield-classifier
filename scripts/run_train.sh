#!/usr/bin/env bash
# scripts/run_train.sh
# Run training for Lit2Vec Subfield Classifier (MLP)

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (can be overridden by arg 1)
CONFIG_FILE_DEFAULT="${PROJECT_ROOT}/configs/train.yaml"
CONFIG_FILE="${1:-$CONFIG_FILE_DEFAULT}"

PYTHON_SCRIPT="${PROJECT_ROOT}/src/train.py"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# ---- Sanity checks ----
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ Config not found: $CONFIG_FILE"
  echo "   Pass a valid path: bash scripts/run_train.sh path/to/train.yaml"
  exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "❌ train.py not found at: $PYTHON_SCRIPT"
  exit 1
fi

# ---- Logging ----
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# Optional: record git commit (if in a git repo)
GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"

echo "🚀 Starting training"
echo "🗂  Project root : $PROJECT_ROOT"
echo "⚙️  Config       : $CONFIG_FILE"
echo "🔢 Git commit   : $GIT_HASH"
echo "📝 Log file     : $LOG_FILE"
echo "🐍 Python       : $(python3 -V 2>&1)"

# Optional: set CUDA device (uncomment to pin GPU)
# export CUDA_VISIBLE_DEVICES=0

# Optional: W&B env (prefer configuring in YAML; these will be used if YAML leaves them null)
# export WANDB_PROJECT="mlp-multilabel-example"
# export WANDB_ENTITY="your_entity"

# ---- Run ----
set -x
python3 "$PYTHON_SCRIPT" \
  --config "$CONFIG_FILE" \
  2>&1 | tee "$LOG_FILE"
set +x

echo "✅ Training finished. Logs saved to: $LOG_FILE"
