#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

stage="${1:-}"
case "$stage" in stage1|stage2|sft|rl) ;; *) echo "用法：$0 stage1|stage2|sft|rl" >&2; exit 2 ;; esac

input_var="DATA_${stage^^}_INPUT"
input="${!input_var:-}"
if [[ -z "$input" ]]; then
  echo "请设置 $input_var，值可以是 JSON、JSONL、Parquet 文件或 glob" >&2
  exit 2
fi
out="$MERIDIAN_RUN_ROOT/$stage"
mkdir -p "$out"
mode=pretrain
target=0
if [[ "$stage" == stage1 ]]; then target=20000000; fi
if [[ "$stage" == stage2 ]]; then target=80000000; fi
if [[ "$stage" == sft ]]; then mode=sft; target=1000000; fi
if [[ "$stage" == rl ]]; then mode=rl; target=0; fi

exec "$PYTHON_BIN" src/meridian_training/retokenize.py \
  --input "$input" --output-prefix "$out/${stage}_text_document" \
  --tokenizer-root "$TOKENIZER_ROOT" --mode "$mode" --target-tokens "$target" \
  ${DATA_RL_TARGET_SAMPLES:+--target-samples "$DATA_RL_TARGET_SAMPLES"}
