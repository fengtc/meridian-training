"""加载 Meridian SFT 检查点并进行最小中文对话测试。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from tokenizers import Tokenizer
from meridian_training.train import build_model
import yaml
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def render_prompt(root: Path, messages: list[dict]) -> str:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for item in messages:
            parts.append(f"<|{item['role']}>{item['content']}")
        return "".join(parts) + "<|assistant|>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--tokenizer-root", required=True, type=Path)
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("需要支持 BF16 的 CUDA 环境")
    tok = Tokenizer.from_file(str(args.tokenizer_root / "tokenizer.json"))
    seq = int(yaml.safe_load(args.config.read_text()).get("sequence_length", 2048))
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        eos = tok.token_to_id("<|eos|>")
    if eos is None:
        raise SystemExit("tokenizer 中没有 <|endoftext|> 或 <|eos|>")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29594")
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)
    model = build_model(tok.get_vocab_size(with_added_tokens=True), yaml.safe_load(args.config.read_text()))
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    def answer(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        ids = tok.encode(render_prompt(args.tokenizer_root, messages)).ids
        if len(ids) >= seq:
            ids = ids[-(seq - 1):]
        generated = list(ids)
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                x = torch.tensor([generated], dtype=torch.long, device="cuda")
                pos = torch.arange(x.shape[1], device="cuda").unsqueeze(0)
                mask = torch.triu(torch.ones((1, 1, x.shape[1], x.shape[1]), device="cuda", dtype=torch.bool), 1)
                logits = model(x, pos, mask)[:, -1, :].float()
                if args.temperature <= 0:
                    next_id = int(logits.argmax(dim=-1).item())
                else:
                    probs = torch.softmax(logits / args.temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())
                generated.append(next_id)
                if next_id == eos or len(generated) >= seq:
                    break
        return tok.decode(generated[len(ids):], skip_special_tokens=False)

    if args.prompt is not None:
        print(f"助手：{answer(args.prompt)}")
    else:
        while True:
            try:
                prompt = input("用户：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt or prompt.lower() in {"exit", "quit"}:
                break
            print(f"助手：{answer(prompt)}")
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
