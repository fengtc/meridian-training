#!/usr/bin/env bash
set -euo pipefail

# 使用 ModelScope 下载 OpenBMB 数据集。大数据集默认下载完整仓库，建议先用
# DATA_ROOT 指向容量充足的磁盘，并在 ModelScope 页面确认许可和可用空间。
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
用法：
  scripts/download-data.sh tokenizer /path/to/zgcm-1-official
  scripts/download-data.sh ultra-fineweb
  scripts/download-data.sh ultra-fineweb-l3
  scripts/download-data.sh math
  scripts/download-data.sh code
  scripts/download-data.sh sft
  scripts/download-data.sh rl
  scripts/download-data.sh rlpr
  scripts/download-data.sh all

也可以直接指定 OpenBMB 数据集：
  scripts/download-data.sh repo OpenBMB/UltraData-RL-2609
EOF
}

command -v modelscope >/dev/null || {
  echo "未找到 modelscope，请先执行：python -m pip install modelscope" >&2
  exit 1
}

download() {
  local repo="$1"; local name="$2"
  local target="$DATA_ROOT/$name"
  mkdir -p "$target"
  echo "下载 $repo -> $target"
  modelscope download --dataset "$repo" --local_dir "$target"
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
case "$1" in
  tokenizer) [[ $# -eq 2 ]] || { usage; exit 2; }; copy_tokenizer "$2" ;;
  ultra-fineweb) download OpenBMB/Ultra-FineWeb ultra-fineweb ;;
  ultra-fineweb-l3) download OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3 ;;
  math) download OpenBMB/UltraData-Math ultradata-math ;;
  code) download OpenBMB/UltraData-Code ultradata-code ;;
  sft) download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605 ;;
  rl) download OpenBMB/UltraData-RL-2609 ultradata-rl-2609 ;;
  rlpr) download OpenBMB/RLPR-Train-Dataset rlpr-train-dataset ;;
  repo) [[ $# -eq 2 ]] || { usage; exit 2; }; download "$2" "$(basename "$2")" ;;
  all)
    download OpenBMB/Ultra-FineWeb ultra-fineweb
    download OpenBMB/Ultra-FineWeb-L3 ultra-fineweb-l3
    download OpenBMB/UltraData-Math ultradata-math
    download OpenBMB/UltraData-Code ultradata-code
    download OpenBMB/UltraData-SFT-2605 ultradata-sft-2605
    download OpenBMB/UltraData-RL-2609 ultradata-rl-2609
    ;;
  *) usage; exit 2 ;;
esac
