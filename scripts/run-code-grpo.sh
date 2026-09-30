#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
gpu_count="${MERIDIAN_GPUS:-2}"
if [[ "${1:-}" == "--gpu" ]]; then gpu_count="$2"; shift 2; elif [[ "${1:-}" == --gpu=* ]]; then gpu_count="${1#--gpu=}"; shift; fi
[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || { echo "GPU 数量必须是正整数" >&2; exit 2; }
: "${CODE_RL_CHECKPOINT:?请设置 CODE_RL_CHECKPOINT（Stage 2 checkpoint 目录）}"
: "${CODE_RL_INPUT:?请设置 CODE_RL_INPUT（prepared Code JSONL）}"
exec "$VENV_ROOT/bin/torchrun" --standalone --nproc_per_node="$gpu_count" \
  src/meridian_training/code_grpo.py \
  --checkpoint "$CODE_RL_CHECKPOINT" --config "${MERIDIAN_MODEL_CONFIG:-$PROJECT_ROOT/configs/model-500m.yaml}" \
  --input "$CODE_RL_INPUT" --tokenizer-root "$TOKENIZER_ROOT" \
  --output-root "${CODE_RL_OUTPUT:-$MERIDIAN_DATA_ROOT/checkpoints/code-grpo}" "$@"
