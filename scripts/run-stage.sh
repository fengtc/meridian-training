#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

stage="${1:-}"
case "$stage" in stage1|stage2|sft) ;; *) echo "用法：$0 stage1|stage2|sft --gpu N [runner 参数]" >&2; exit 2 ;; esac
shift

gpu_count="${MERIDIAN_GPUS:-1}"
runner_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu 后需要数量" >&2; exit 2; }
      gpu_count="$2"; shift 2 ;;
    --gpu=*)
      gpu_count="${1#--gpu=}"; shift ;;
    *) runner_args+=("$1"); shift ;;
  esac
done
[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || { echo "GPU 数量必须是正整数：$gpu_count" >&2; exit 2; }

config="${MERIDIAN_MODEL_CONFIG:-$PWD/configs/model-500m.yaml}"

if [[ "$stage" == stage1 ]]; then
  dataset="$MERIDIAN_RUN_ROOT/stage1/stage1_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/stage1"
  target="${DATA_STAGE1_TARGET_TOKENS:-2000000000}"
  resume=()
elif [[ "$stage" == stage2 ]]; then
  dataset="$MERIDIAN_RUN_ROOT/stage2/stage2_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/stage2"
  target="${DATA_STAGE2_TARGET_TOKENS:-8000000000}"
  resume=(--resume "$MERIDIAN_RUN_ROOT/stage1/checkpoints")
else
  dataset="$MERIDIAN_RUN_ROOT/sft/sft_text_document"
  run_dir="$MERIDIAN_RUN_ROOT/sft"
  target="${DATA_SFT_TARGET_TOKENS:-1000000000}"
  resume=(--resume "$MERIDIAN_RUN_ROOT/stage2/checkpoints")
fi

[[ -f "$dataset.bin" && -f "$dataset.idx" ]] || {
  echo "缺少 IndexedDataset 文件：$dataset.{bin,idx}；请先执行 encode-data.sh $stage" >&2
  exit 1
}
if [[ "$stage" == sft && ( ! -f "$dataset.mask.bin" || ! -f "$dataset.mask.idx" ) ]]; then
  echo "缺少 SFT assistant mask：$dataset.mask.{bin,idx}；请重新执行 encode-data.sh sft" >&2
  exit 1
fi
if [[ "$stage" == stage2 || "$stage" == sft ]]; then
  [[ -f "${resume[1]}/step-latest-rank0.pt" ]] || { echo "缺少续训 checkpoint：${resume[1]}/step-latest-rank0.pt" >&2; exit 1; }
fi

gradient_accumulation="${MERIDIAN_GRADIENT_ACCUMULATION:-8}"
exec "$VENV_ROOT/bin/torchrun" --standalone --nproc_per_node="$gpu_count" \
  src/meridian_training/train.py \
  --stage "$stage" --config "$config" --dataset-prefix "$dataset" --output-root "$run_dir" \
  --tokenizer-root "$TOKENIZER_ROOT" --target-tokens "$target" \
  "${resume[@]}" --gradient-accumulation "$gradient_accumulation" \
  --recompute-granularity "${MERIDIAN_RECOMPUTE_GRANULARITY:-full}" \
  --recompute-method "${MERIDIAN_RECOMPUTE_METHOD:-uniform}" \
  --recompute-num-layers "${MERIDIAN_RECOMPUTE_NUM_LAYERS:-1}" "${runner_args[@]}"
