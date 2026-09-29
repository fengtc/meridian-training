#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git submodule update --init --recursive
echo "Megatron-LM revision: $(git -C third_party/Megatron-LM rev-parse HEAD)"
echo "Install a CUDA-enabled PyTorch build separately, then: python -m pip install -e '.[templates]'"
