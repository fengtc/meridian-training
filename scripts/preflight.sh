#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ -d "$PROJECT_ROOT/third_party/Megatron-LM/megatron" ]] || {
  echo "缺少 Megatron-LM submodule：$PROJECT_ROOT/third_party/Megatron-LM" >&2
  echo "请执行：git submodule update --init --recursive" >&2
  exit 1
}

TOKENIZER_JSON="$TOKENIZER_ROOT/tokenizer.json"
CHAT_TEMPLATE="$TOKENIZER_ROOT/chat_template.jinja"
[[ -s "$TOKENIZER_JSON" ]] || { echo "缺少官方 tokenizer：$TOKENIZER_JSON" >&2; exit 1; }
[[ -s "$CHAT_TEMPLATE" ]] || { echo "缺少官方 chat template：$CHAT_TEMPLATE" >&2; exit 1; }
case "$TOKENIZER_ROOT" in
  *spark-32k*|*MiniCPM5*|*minicpm5*)
    echo "拒绝使用非官方 tokenizer 路径：$TOKENIZER_ROOT" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" - "$TOKENIZER_JSON" "$CHAT_TEMPLATE" <<'PY'
import hashlib, json, pathlib, sys
from tokenizers import Tokenizer

tok_path, template_path = map(pathlib.Path, sys.argv[1:])
tok = Tokenizer.from_file(str(tok_path))
vocab = tok.get_vocab_size(with_added_tokens=True)
template = template_path.read_text(encoding="utf-8")
if not template.strip():
    raise SystemExit("chat_template.jinja 为空")
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({
    "tokenizer": str(tok_path),
    "chat_template": str(template_path),
    "vocab_size": vocab,
    "tokenizer_sha256": sha(tok_path),
    "chat_template_sha256": sha(template_path),
}, sort_keys=True))
PY

"$PYTHON_BIN" - <<'PY'
import json, os, subprocess, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("当前 DGX Spark GB10 环境不支持 BF16")
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "bf16": True,
    "visible_gpus": torch.cuda.device_count(),
}, sort_keys=True))
PY

echo "环境检查通过"
