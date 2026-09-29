#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/data/tokenizers/official}"
DATA_ROOT="${DATA_ROOT:-/data/source-datasets}"
MERIDIAN_RUN_ROOT="${MERIDIAN_RUN_ROOT:-/data/meridian-training-runs}"
VENV_ROOT="${VENV_ROOT:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"

export PROJECT_ROOT TOKENIZER_ROOT DATA_ROOT MERIDIAN_RUN_ROOT VENV_ROOT PYTHON_BIN
export PYTHONPATH="$PROJECT_ROOT/third_party/Megatron-LM:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN (set PYTHON_BIN or VENV_ROOT)" >&2
  exit 1
fi
