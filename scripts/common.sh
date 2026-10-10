#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MERIDIAN_DATA_ROOT="${MERIDIAN_DATA_ROOT:-${HOME:-$PROJECT_ROOT}/meridian-data}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-$MERIDIAN_DATA_ROOT/tokenizers/official}"
DATA_ROOT="${DATA_ROOT:-$MERIDIAN_DATA_ROOT/source-datasets}"
MERIDIAN_RUN_ROOT="${MERIDIAN_RUN_ROOT:-$MERIDIAN_DATA_ROOT/training-runs}"
VENV_ROOT="${VENV_ROOT:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"

export PROJECT_ROOT MERIDIAN_DATA_ROOT TOKENIZER_ROOT DATA_ROOT MERIDIAN_RUN_ROOT VENV_ROOT PYTHON_BIN

if [[ -n "${MERIDIAN_DEVICE_PROFILE:-}" ]]; then
  profile="$PROJECT_ROOT/configs/devices/${MERIDIAN_DEVICE_PROFILE}.env"
  [[ -f "$profile" ]] || { echo "未知设备 profile：$MERIDIAN_DEVICE_PROFILE" >&2; exit 2; }
  # Profiles are explicit environment presets; callers may override values after sourcing.
  # shellcheck disable=SC1090
  source "$profile"
  PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"
  export MERIDIAN_DATA_ROOT VENV_ROOT MERIDIAN_GPUS MERIDIAN_GRADIENT_ACCUMULATION PYTHON_BIN \
    MERIDIAN_RECOMPUTE_GRANULARITY MERIDIAN_RECOMPUTE_METHOD MERIDIAN_RECOMPUTE_NUM_LAYERS
fi
if [[ -d "$PROJECT_ROOT/third_party/Megatron-LM/megatron/core/tensor_parallel" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/third_party/Megatron-LM:${PYTHONPATH:-}"
fi

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN (set PYTHON_BIN or VENV_ROOT)" >&2
  exit 1
fi
