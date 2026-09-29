#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git submodule update --init --recursive
echo "Megatron-LM 版本：$(git -C third_party/Megatron-LM rev-parse HEAD)"
echo "请先安装适合 DGX Spark GB10 的 CUDA/BF16 PyTorch，再执行：python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e '.[templates]'"
