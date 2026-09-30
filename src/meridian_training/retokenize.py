#!/usr/bin/env python3
"""使用配置的官方 tokenizer 重新编码数据。

输出固定长度的 Megatron IndexedDataset。SFT 另外生成 int8 mask 数据集；1 表示
assistant 内容和 EOS，0 表示上下文。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections.abc import Mapping
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
import sys
MEGATRON_ROOT = ROOT / "third_party" / "Megatron-LM"
if (MEGATRON_ROOT / "megatron" / "core" / "tensor_parallel").is_dir():
    sys.path.insert(0, str(MEGATRON_ROOT))
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

_HF_TOKENIZER_CACHE: dict[str, object] = {}


def hf_tokenizer(root: Path):
    key = str(root.resolve())
    if key not in _HF_TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _HF_TOKENIZER_CACHE[key] = AutoTokenizer.from_pretrained(
            key, local_files_only=True, trust_remote_code=False
        )
    return _HF_TOKENIZER_CACHE[key]


def files(pattern: str) -> list[Path]:
    # 允许用逗号合并多个 ModelScope 数据集目录，例如
    # ``a/**/*.jsonl,b/**/*.parquet``。
    paths: list[Path] = []
    for part in (x.strip() for x in pattern.split(",")):
        if not part:
            continue
        matches = [Path(x) for x in glob.glob(part, recursive=True)]
        if not matches and Path(part).is_file():
            matches = [Path(part)]
        paths.extend(matches)
    return sorted(set(paths))


def records(pattern: str) -> Iterable[dict]:
    for path in files(pattern):
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            with path.open(encoding="utf-8") as stream:
                if suffix == ".json":
                    value = json.load(stream)
                    rows = value if isinstance(value, list) else [value]
                    yield from (row for row in rows if isinstance(row, dict))
                else:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            if isinstance(row, dict):
                                yield row
        elif suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            yield from table.to_pylist()
        else:
            raise ValueError(f"unsupported input format: {path}")


def text_of(row: dict) -> str:
    for key in ("text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def messages_of(row: dict) -> list[dict]:
    value = row.get("messages") or row.get("conversation") or row.get("conversations")
    if not isinstance(value, list):
        raise ValueError("SFT row has no messages/conversation list")
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value") or ""
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "bot", "assistant"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        elif role in {"tool", "function", "observation"}:
            role = "tool"
        else:
            continue
        out.append({"role": role, "content": str(content)})
    if not out or not any(x["role"] == "assistant" for x in out):
        raise ValueError("SFT row has no assistant turn")
    return out


def render(tokenizer_root: Path, tokenizer: Tokenizer, messages: list[dict], eos_id: int) -> tuple[list[int], list[int]]:
    """Render the official template and derive an assistant mask.

    Transformers is preferred because it implements the template contract and
    assistant mask. A strict fallback handles the documented ZGCM format only.
    """
    try:
        hf = hf_tokenizer(tokenizer_root)
        result = hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                        return_assistant_tokens_mask=True, return_dict=True)
        ids = list(result["input_ids"])
        mask = list(result.get("assistant_masks") or result.get("assistant_mask") or [])
        if len(ids) == len(mask) and any(mask):
            return ids, mask
    except Exception:
        pass

    # The official example uses role tokens. Refuse to guess for an unknown
    # template rather than silently training user tokens as assistant targets.
    template = (tokenizer_root / "chat_template.jinja").read_text(encoding="utf-8")
    if "<|user|>" not in template or "<|assistant|>" not in template:
        raise RuntimeError("cannot derive assistant mask; install transformers with the official tokenizer")
    ids: list[int] = []
    mask: list[int] = []
    for message in messages:
        role = message["role"]
        prefix = f"<|{role}>"
        body = message["content"]
        prefix_ids = tokenizer.encode(prefix).ids
        body_ids = tokenizer.encode(body).ids
        if role == "assistant":
            piece = prefix_ids + body_ids + [eos_id]
            ids.extend(piece); mask.extend([0] * len(prefix_ids) + [1] * (len(body_ids) + 1))
        else:
            piece = prefix_ids + body_ids
            ids.extend(piece); mask.extend([0] * len(piece))
    return ids, mask


def render_rl_prompt(tokenizer_root: Path, tokenizer: Tokenizer, query: str | list[dict]) -> list[int]:
    """Render an RL query with the official template and leave generation open."""
    messages = query if isinstance(query, list) else [{"role": "user", "content": query}]
    try:
        hf = hf_tokenizer(tokenizer_root)
        ids = hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, Mapping):
            ids = ids.get("input_ids")
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            return list(ids)
    except Exception:
        pass
    template = (tokenizer_root / "chat_template.jinja").read_text(encoding="utf-8")
    if "<|user|>" not in template or "<|assistant|>" not in template:
        raise RuntimeError("cannot render RL prompt; install transformers with the official tokenizer")
    if isinstance(query, list):
        rendered = "".join(f"<|{m['role']}>{m['content']}" for m in query)
        rendered += "<|assistant|>"
    else:
        rendered = f"<|user|>{query}<|assistant|>"
    return tokenizer.encode(rendered).ids


def rl_query_of(row: dict) -> str | list[dict]:
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if isinstance(prompt, list):
        messages = []
        for item in prompt:
            if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                messages.append({"role": item["role"], "content": str(item["content"])})
        if messages:
            return messages
    return ""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--tokenizer-root", required=True)
    ap.add_argument("--mode", choices=("pretrain", "sft", "rl"), required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--target-samples", type=int, default=0)
    ap.add_argument("--sequence-length", type=int, default=2048)
    args = ap.parse_args()
    root = Path(args.tokenizer_root)
    tok_path = root / "tokenizer.json"
    template_path = root / "chat_template.jinja"
    tokenizer = Tokenizer.from_file(str(tok_path))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise SystemExit("official tokenizer must define <|endoftext|> or <|eos|>")
    target = args.target_tokens
    token_buffer: list[int] = []
    mask_buffer: list[int] = []
    # The Meridian runner consumes two sequences per optimizer step. Keep the
    # packed set even-sized so the requested 4096-token steps are complete.
    target_sequences = (
        math.ceil(math.ceil(target / args.sequence_length) / 2) * 2
        if args.mode == "pretrain" else (args.target_samples or 10**18)
    )
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(str(out) + ".bin", dtype=np.int32)
    mask_builder = IndexedDatasetBuilder(str(out) + ".mask.bin", dtype=np.int8) if args.mode == "sft" else None
    metadata_path = Path(str(out) + ".metadata.jsonl") if args.mode == "rl" else None
    metadata_stream = metadata_path.open("w", encoding="utf-8") if metadata_path else None
    sequences = assistant_tokens = packed_assistant_tokens = documents = skipped_long = 0
    for row in records(args.input):
        if args.mode == "pretrain":
            ids = tokenizer.encode(text_of(row)).ids + [eos]
            masks = [0] * len(ids)
        elif args.mode == "sft":
            ids, masks = render(root, tokenizer, messages_of(row), eos)
        else:
            query = rl_query_of(row)
            if not query:
                continue
            ids = render_rl_prompt(root, tokenizer, query)
            masks = [0] * len(ids)
            if len(ids) > args.sequence_length:
                # RL prompts are standalone requests. Skip overlong requests
                # rather than silently truncating their problem statement.
                skipped_long += 1
                continue
        if not ids:
            continue
        documents += 1
        if args.mode == "rl":
            item = ids + [eos] * (args.sequence_length - len(ids))
            builder.add_item(torch.from_numpy(np.asarray(item, dtype=np.int32))); builder.end_document()
            assert metadata_stream is not None
            metadata_stream.write(json.dumps({
                "uuid": row.get("uuid") or row.get("id") or f"rl-{sequences:08d}",
                "domain": row.get("domain"),
                "source": row.get("source"),
                "query": query,
                "ground_truth": row.get("ground_truth") if row.get("ground_truth") is not None else row.get("reward_model", {}).get("ground_truth") if isinstance(row.get("reward_model"), dict) else None,
                "prompt_tokens": len(ids),
            }, ensure_ascii=False) + "\n")
            sequences += 1
            if sequences >= target_sequences:
                break
            continue
        token_buffer.extend(ids); mask_buffer.extend(masks)
        assistant_tokens += sum(masks)
        while len(token_buffer) >= args.sequence_length and sequences < target_sequences:
            item = token_buffer[:args.sequence_length]
            item_mask = mask_buffer[:args.sequence_length]
            del token_buffer[:args.sequence_length]; del mask_buffer[:args.sequence_length]
            if args.mode == "sft" and not any(item_mask):
                continue
            builder.add_item(torch.from_numpy(np.asarray(item, dtype=np.int32))); builder.end_document()
            if mask_builder is not None:
                mask_builder.add_item(torch.from_numpy(np.asarray(item_mask, dtype=np.int8))); mask_builder.end_document()
            sequences += 1
            packed_assistant_tokens += int(sum(item_mask))
            if sequences >= target_sequences or (args.mode == "sft" and packed_assistant_tokens >= target):
                break
        if sequences >= target_sequences or (args.mode == "sft" and packed_assistant_tokens >= target):
            break
    if token_buffer and sequences < target_sequences and (args.mode == "pretrain" or any(mask_buffer)):
        pad = args.sequence_length - len(token_buffer)
        token_buffer.extend([eos] * pad); mask_buffer.extend([0] * pad)
        builder.add_item(torch.from_numpy(np.asarray(token_buffer, dtype=np.int32))); builder.end_document()
        if mask_builder is not None:
            mask_builder.add_item(torch.from_numpy(np.asarray(mask_buffer, dtype=np.int8))); mask_builder.end_document()
        sequences += 1
        packed_assistant_tokens += int(sum(mask_buffer))
    builder.finalize(str(out) + ".idx")
    if mask_builder is not None:
        mask_builder.finalize(str(out) + ".mask.idx")
    if metadata_stream is not None:
        metadata_stream.close()
    if args.mode == "sft" and packed_assistant_tokens < target:
        raise SystemExit(
            f"SFT source ended at {packed_assistant_tokens} assistant tokens; "
            f"{target} required"
        )
    manifest = {
        "mode": args.mode,
        "input": args.input,
        "output_prefix": str(out),
        "sequence_length": args.sequence_length,
        "target_tokens": target,
        "target_samples": args.target_samples,
        "packed_sequences": sequences,
        "packed_tokens": sequences * args.sequence_length,
        "assistant_tokens": packed_assistant_tokens,
        "metadata": str(metadata_path) if metadata_path else None,
        "documents": documents,
        "skipped_long_prompts": skipped_long,
        "tokenizer": str(tok_path),
        "tokenizer_sha256": sha(tok_path),
        "chat_template": str(template_path),
        "chat_template_sha256": sha(template_path),
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
