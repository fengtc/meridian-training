#!/usr/bin/env bash
set -euo pipefail

# 使用 ModelScope 下载 OpenBMB 数据集。大数据集默认下载完整仓库，建议先用
# DATA_ROOT 指向容量充足的磁盘，并在 ModelScope 页面确认许可和可用空间。
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
用法：
  scripts/download-data.sh tokenizer /path/to/tokenizer
  scripts/download-data.sh ultra-fineweb
  scripts/download-data.sh ultra-fineweb-l3
  scripts/download-data.sh math
  scripts/download-data.sh code
  scripts/download-data.sh sft
  scripts/download-data.sh stage1-target
  scripts/download-data.sh stage2-target
  scripts/download-data.sh sft-target
  scripts/download-data.sh rl-target
  scripts/download-data.sh rl
  scripts/download-data.sh rlpr
  scripts/download-data.sh all-target
  scripts/download-data.sh all

也可以只下载指定文件匹配项：
  scripts/download-data.sh sft --include 'data/no_think/Chinese-general/*part-000*.jsonl'

按目标 token 下载（目标用于选择目标分片，实际 token 数以编码 manifest 为准）：
  scripts/download-data.sh stage1-target -2B
  scripts/download-data.sh stage2-target --target-tokens 8B
  scripts/download-data.sh sft-target --target-tokens 1B

也可以直接指定 OpenBMB 数据集：
  scripts/download-data.sh repo OpenBMB/UltraData-RL-2609
EOF
}

command -v modelscope >/dev/null || {
  echo "未找到 modelscope，请先执行：python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple modelscope" >&2
  exit 1
}

download() {
  local repo="$1"; local name="$2"
  shift 2
  local target="$DATA_ROOT/$name"
  mkdir -p "$target"
  echo "下载 $repo -> $target"
  local args=(modelscope download --dataset "$repo" --local_dir "$target")
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --include|--exclude)
        [[ $# -ge 2 ]] || { echo "$1 后面必须提供匹配模式" >&2; exit 2; }
        args+=("$1" "$2")
        shift 2
        ;;
      --target-tokens)
        shift 2
        ;;
      --target-tokens=*|-[0-9]*|--[0-9]*)
        shift
        ;;
      *) echo "不支持的下载参数：$1" >&2; exit 2 ;;
    esac
  done
  "${args[@]}"
}

download_first_shards() {
  local repo="$1"; local name="$2"
  download "$repo" "$name" \
    --include '*part-0*'
}

parse_target_tokens() {
  local value="${1#--target-tokens=}"
  value="${value#-}"
  if [[ "$value" =~ ^([0-9]+([.][0-9]+)?)([BKMGTPE]?)$ ]]; then
    local number="${BASH_REMATCH[1]}" suffix="${BASH_REMATCH[3]}" multiplier=1
    case "$suffix" in K) multiplier=1000 ;; M) multiplier=1000000 ;; G|B) multiplier=1000000000 ;; T) multiplier=1000000000000 ;; P) multiplier=1000000000000000 ;; E) multiplier=1000000000000000000 ;; esac
    TARGET_TOKENS=$(awk -v n="$number" -v m="$multiplier" 'BEGIN { printf "%.0f", n*m }')
    export TARGET_TOKENS
    return 0
  fi
  echo "无效 token 目标：$1（示例：500M、2B、1.5B）" >&2
  exit 2
}

target_from_args() {
  TARGET_TOKENS=""
  local args=("$@") i=0
  while (( i < ${#args[@]} )); do
    local arg="${args[$i]}"
    if [[ "$arg" == --target-tokens ]]; then
      (( i + 1 < ${#args[@]} )) || { echo "--target-tokens 后需要数值" >&2; exit 2; }
      parse_target_tokens "${args[$((i + 1))]}"; i=$((i + 2)); continue
    fi
    if [[ "$arg" == --target-tokens=* || "$arg" == -[0-9]* || "$arg" == --[0-9]* ]]; then parse_target_tokens "$arg"; fi
    i=$((i + 1))
  done
  if [[ -z "$TARGET_TOKENS" ]]; then return 0; fi
  echo "目标下载量：$TARGET_TOKENS tokens（下载分片为估算值，编码后以 manifest 的实际 token 数为准）"
}

copy_tokenizer() {
  local source="$1"
  [[ -s "$source/tokenizer.json" ]] || { echo "缺少 $source/tokenizer.json" >&2; exit 1; }
  [[ -s "$source/chat_template.jinja" ]] || { echo "缺少 $source/chat_template.jinja" >&2; exit 1; }
  mkdir -p "$TOKENIZER_ROOT"
  cp -f "$source/tokenizer.json" "$TOKENIZER_ROOT/tokenizer.json"
  cp -f "$source/chat_template.jinja" "$TOKENIZER_ROOT/chat_template.jinja"
  echo "官方 tokenizer 已复制到 $TOKENIZER_ROOT"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
mkdir -p "$DATA_ROOT"
target_from_args "${@:2}"
case "$1" in
  tokenizer) [[ $# -eq 2 ]] || { usage; exit 2; }; copy_tokenizer "$2" ;;
  ultra-fineweb) download OpenBMB/Ultra-FineWeb ultra-fineweb "${@:2}" ;;
  ultra-fineweb-l3) download OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3 "${@:2}" ;;
  math) download OpenBMB/UltraData-Math ultradata-math "${@:2}" ;;
  code) download OpenBMB/UltraData-Code ultradata-code "${@:2}" ;;
  sft) download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605 "${@:2}" ;;
  sft-small)
    # 每个类别的前三个分片用于本项目 1M assistant-token SFT 目标。
    download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605 \
      --include 'data/no_think/Chinese-general/*part-0*.jsonl' \
      --include 'data/no_think/IF/*part-0*.jsonl' \
      --include 'data/no_think/Knowledge/*part-0*.jsonl' \
      --include 'data/no_think/Code/*part-0*.jsonl' \
      --include 'data/no_think/Math/*part-0*.jsonl'
    ;;
  stage1-target|stage1-small)
    [[ -n "${TARGET_TOKENS:-}" ]] || TARGET_TOKENS=2000000000
    # Ultra-FineWeb 的分片大小不同，先选前置分片，再由编码 manifest 校准。
    if (( TARGET_TOKENS <= 2000000000 )); then
      download OpenBMB/Ultra-FineWeb ultra-fineweb --include '*part-00[0-6]-of-*.parquet'
    else
      download OpenBMB/Ultra-FineWeb ultra-fineweb --include '*part-0*.parquet'
    fi
    ;;
  stage2-target|stage2-small)
    [[ -n "${TARGET_TOKENS:-}" ]] || TARGET_TOKENS=8000000000
    download OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3 --include '*part-0*.parquet'
    download OpenBMB/UltraData-Math ultradata-math --include '*part-0*.parquet'
    download OpenBMB/UltraData-Code ultradata-code --include '*part-0*.parquet'
    ;;
  sft-target)
    [[ -n "${TARGET_TOKENS:-}" ]] || TARGET_TOKENS=1000000000
    download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605 \
      --include 'data/no_think/Chinese-general/*part-0*.jsonl' \
      --include 'data/no_think/Code/*part-0*.jsonl'
    ;;
  rl-target|rl-small) download OpenBMB/UltraData-RL-2609 ultradata-rl-2609 ;;
  rl) download OpenBMB/UltraData-RL-2609 ultradata-rl-2609 "${@:2}" ;;
  rlpr) download OpenBMB/RLPR-Train-Dataset rlpr-train-dataset "${@:2}" ;;
  repo) [[ $# -eq 2 ]] || { usage; exit 2; }; download "$2" "$(basename "$2")" ;;
  all)
    download OpenBMB/Ultra-FineWeb ultra-fineweb
    download OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3
    download OpenBMB/UltraData-Math ultradata-math
    download OpenBMB/UltraData-Code ultradata-code
    download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605
    download OpenBMB/UltraData-RL-2609 ultradata-rl-2609
    ;;
  all-target|all-small)
    download_first_shards OpenBMB/Ultra-FineWeb ultra-fineweb
    download_first_shards OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3
    download_first_shards OpenBMB/UltraData-Math ultradata-math
    download_first_shards OpenBMB/UltraData-Code ultradata-code
    download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605 \
      --include 'data/no_think/Chinese-general/*part-0*.jsonl' \
      --include 'data/no_think/IF/*part-0*.jsonl' \
      --include 'data/no_think/Knowledge/*part-0*.jsonl' \
      --include 'data/no_think/Code/*part-0*.jsonl' \
      --include 'data/no_think/Math/*part-0*.jsonl'
    download_first_shards OpenBMB/UltraData-RL-2609 ultradata-rl-2609
    ;;
  *) usage; exit 2 ;;
esac
