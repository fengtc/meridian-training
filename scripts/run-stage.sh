#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

stage="${1:-}"
case "$stage" in stage1|stage2|sft) ;; *) echo "usage: $0 stage1|stage2|sft [extra runner args...]" >&2; exit 2 ;; esac
shift

if [[ "$stage" == stage1 ]]; then
  dataset="$MERIDIAN_RUN_ROOT/stage1/stage1_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/stage1"
  target=20000000
  resume=()
elif [[ "$stage" == stage2 ]]; then
  dataset="$MERIDIAN_RUN_ROOT/stage2/stage2_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/stage2"
  target=80000000
  resume=(--resume "$MERIDIAN_RUN_ROOT/stage1/checkpoints/stage1-final.pt")
else
  dataset="$MERIDIAN_RUN_ROOT/sft/sft_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/sft"
  target=1000000
  resume=(--resume "$MERIDIAN_RUN_ROOT/stage2/checkpoints/stage2-final.pt")
fi

[[ -f "$dataset.bin" && -f "$dataset.idx" ]] || {
  echo "missing IndexedDataset pair: $dataset.{bin,idx}; run encode-data.sh $stage first" >&2
  exit 1
}
if [[ "$stage" == sft && ( ! -f "$dataset.mask.bin" || ! -f "$dataset.mask.idx" ) ]]; then
  echo "missing SFT assistant mask pair: $dataset.mask.{bin,idx}; rerun encode-data.sh sft" >&2
  exit 1
fi
if [[ "$stage" == stage2 || "$stage" == sft ]]; then
  [[ -f "${resume[1]}" ]] || { echo "missing required resume checkpoint: ${resume[1]}" >&2; exit 1; }
fi

exec "$PYTHON_BIN" src/meridian_training/train.py \
  --stage "$stage" --dataset-prefix "$dataset" --output-root "$run_dir" \
  --tokenizer-root "$TOKENIZER_ROOT" --target-tokens "$target" \
  "${resume[@]}" "$@"
