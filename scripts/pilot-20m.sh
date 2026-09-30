#!/usr/bin/env bash
set -euo pipefail

# 20M pilot uses the same model, DDP and checkpoint code as the formal run.
source "$(dirname "$0")/common.sh"
: "${DATA_STAGE1_INPUT:?请先设置 DATA_STAGE1_INPUT}"

gpu_count="${MERIDIAN_GPUS:-2}"
if [[ "${1:-}" == "--gpu" ]]; then
  [[ $# -ge 2 ]] || { echo "--gpu 后需要数量" >&2; exit 2; }
  gpu_count="$2"; shift 2
elif [[ "${1:-}" == --gpu=* ]]; then
  gpu_count="${1#--gpu=}"; shift
fi
[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || { echo "GPU 数量必须是正整数：$gpu_count" >&2; exit 2; }

pilot_root="${PILOT_ROOT:-$MERIDIAN_DATA_ROOT/indexed/pilot-20m}"
export MERIDIAN_RUN_ROOT="$pilot_root/indexed"
export DATA_STAGE1_TARGET_TOKENS=20000000
"$PROJECT_ROOT/scripts/encode-data.sh" stage1

exec "$VENV_ROOT/bin/torchrun" --standalone --nproc_per_node="$gpu_count" \
  "$PROJECT_ROOT/src/meridian_training/train.py" \
  --stage stage1 --config "${MERIDIAN_MODEL_CONFIG:-$PROJECT_ROOT/configs/model-500m.yaml}" \
  --dataset-prefix "$pilot_root/indexed/stage1/stage1_text_document" \
  --output-root "$pilot_root/checkpoints" --tokenizer-root "$TOKENIZER_ROOT" \
  --target-tokens 20000000 --checkpoint-interval "${CHECKPOINT_INTERVAL:-100}"
