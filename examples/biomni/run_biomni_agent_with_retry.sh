#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# Wrapper around run_biomni_agent_qwen32b_dapo_drgrpo_rope_all.sh
# 
# Usage:
#   ./run_biomni_agent_with_retry.sh fresh [extra_args...]     # Start from scratch
#   ./run_biomni_agent_with_retry.sh auto [extra_args...]      # Auto-find latest checkpoint
#   ./run_biomni_agent_with_retry.sh /path/to/ckpt [extra_args...]  # Resume from specific checkpoint
#   ./run_biomni_agent_with_retry.sh [extra_args...]           # Use default checkpoint
#
# On failure:
# - Always retries with resume_mode=auto regardless of initial mode
# ------------------------------------------------------------------------------

set -uo pipefail

# Default checkpoint path (edit if you move checkpoints)
DEFAULT_CKPT=/dfs/scratch1/lansong/models/qwen/biomni-training-qwen3-32b-grpo/biomni-training-qwen3-32b-32bsz-temp0.6-clip-0.28-32turn-grpo/global_step_88

# Parse first argument to determine mode
RESUME_MODE=""
RESUME_PATH=""

if [[ $# -ge 1 ]]; then
  case "$1" in
    fresh)
      RESUME_MODE="disable"
      shift
      ;;
    auto)
      RESUME_MODE="auto"
      shift
      ;;
    -*)
      # First arg is a hydra override, use default checkpoint
      RESUME_MODE="resume_path"
      RESUME_PATH="$DEFAULT_CKPT"
      ;;
    *)
      # First arg is a checkpoint path
      RESUME_MODE="resume_path"
      RESUME_PATH="$1"
      shift
      ;;
  esac
else
  # No args, use default checkpoint
  RESUME_MODE="resume_path"
  RESUME_PATH="$DEFAULT_CKPT"
fi

BIOMNI_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
BIOMNI_SCRIPT="${BIOMNI_SCRIPT_DIR}/run_biomni_agent_qwen32b_dapo_drgrpo_rope_all.sh"

if [[ ! -f "${BIOMNI_SCRIPT}" ]]; then
  echo "❌ Cannot find runner script at ${BIOMNI_SCRIPT}" >&2
  exit 1
fi

# -----------------------
# Run with retries until success
# -----------------------

echo "🚀 Starting training with resume_mode=${RESUME_MODE}${RESUME_PATH:+ from ${RESUME_PATH}}"

RETRY_COUNT=0

while true; do
  if [[ ${RETRY_COUNT} -eq 0 ]]; then
    # First attempt with the specified mode
    if [[ "$RESUME_MODE" == "resume_path" ]]; then
      bash "${BIOMNI_SCRIPT}" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${RESUME_PATH}" \
        "$@"
    else
      bash "${BIOMNI_SCRIPT}" \
        trainer.resume_mode="${RESUME_MODE}" \
        "$@"
    fi
  else
    # All subsequent attempts use auto mode
    echo "⚠️  Training run exited with code ${EXIT_CODE}. Retry #${RETRY_COUNT} with resume_mode=auto..." >&2
    bash "${BIOMNI_SCRIPT}" \
      trainer.resume_mode=auto \
      "$@"
  fi
  
  EXIT_CODE=$?
  
  if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "✅ Training completed successfully!"
    break
  fi
  
  ((RETRY_COUNT++))
  
  # Optional: Add a small delay between retries to avoid hammering the system
  echo "💤 Waiting 10 seconds before retry..."
  sleep 10
done

exit ${EXIT_CODE} 