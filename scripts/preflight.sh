#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ -d "$PROJECT_ROOT/third_party/Megatron-LM/megatron" ]] || {
  echo "missing Megatron-LM submodule: $PROJECT_ROOT/third_party/Megatron-LM" >&2
  echo "run: git submodule update --init --recursive" >&2
  exit 1
}

TOKENIZER_JSON="$TOKENIZER_ROOT/tokenizer.json"
CHAT_TEMPLATE="$TOKENIZER_ROOT/chat_template.jinja"
[[ -s "$TOKENIZER_JSON" ]] || { echo "missing official tokenizer: $TOKENIZER_JSON" >&2; exit 1; }
[[ -s "$CHAT_TEMPLATE" ]] || { echo "missing official chat template: $CHAT_TEMPLATE" >&2; exit 1; }
case "$TOKENIZER_ROOT" in
  *spark-32k*|*MiniCPM5*|*minicpm5*)
    echo "refusing non-official tokenizer path: $TOKENIZER_ROOT" >&2
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
    raise SystemExit("chat_template.jinja is empty")
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
    raise SystemExit("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is unavailable on this Spark runtime")
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "bf16": True,
    "visible_gpus": torch.cuda.device_count(),
}, sort_keys=True))
PY

echo "preflight PASS"
