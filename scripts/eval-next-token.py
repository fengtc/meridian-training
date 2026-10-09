#!/usr/bin/env python3
"""Compare next-token likelihood and predictions across Meridian checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.datasets.indexed_dataset import IndexedDataset
from tokenizers import Tokenizer

from meridian_training.train import build_model


TEXTS = {
    "zh_facts": [
        "北京是中国的首都，也是全国的政治和文化中心。",
        "水在标准大气压下达到一百摄氏度时会沸腾。",
        "计算机程序由指令组成，处理器按照顺序执行这些指令。",
    ],
    "en_facts": [
        "The capital of France is Paris, a city on the Seine river.",
        "Water freezes at zero degrees Celsius under standard conditions.",
        "A compiler translates source code into executable instructions.",
    ],
    "code": [
        "def add(a, b):\n    return a + b\n",
        "for item in items:\n    print(item)\n",
        "def square(x):\n    return x * x\n",
    ],
}

PREFIXES = ["北京是中国的", "水在标准大气压下会", "The capital of France is", "def add(a, b):\n    return"]


def predict(model, ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids], device="cuda", dtype=torch.long)
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    mask = torch.triu(torch.ones((1, 1, len(ids), len(ids)), device="cuda", dtype=torch.bool), 1)
    return model(x, pos, mask)[0].float()


def evaluate(model, tokenizer: Tokenizer) -> dict:
    groups = {}
    with torch.inference_mode():
        for group, texts in TEXTS.items():
            total_loss = 0.0
            total_tokens = 0
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                logits = predict(model, ids)
                labels = torch.tensor(ids[1:], device="cuda")
                total_loss += F.cross_entropy(logits[:-1], labels, reduction="sum").item()
                total_tokens += len(labels)
            mean_loss = total_loss / total_tokens
            groups[group] = {"tokens": total_tokens, "nll": mean_loss, "perplexity": math.exp(mean_loss)}
        next_tokens = {}
        for prefix in PREFIXES:
            ids = tokenizer.encode(prefix, add_special_tokens=False).ids
            logits = predict(model, ids)[-1]
            probs = F.softmax(logits, dim=-1)
            values, indices = probs.topk(5)
            next_tokens[prefix] = [
                {"token": tokenizer.id_to_token(int(index)), "decoded": tokenizer.decode([int(index)], skip_special_tokens=False),
                 "probability": round(float(value), 5)}
                for value, index in zip(values, indices)
            ]
    return {"groups": groups, "next_tokens": next_tokens}


def evaluate_indexed(model, prefix: Path, count: int = 8) -> dict:
    ds = IndexedDataset(str(prefix), mmap=True)
    losses = []
    with torch.inference_mode():
        for index in range(min(count, len(ds))):
            ids = torch.from_numpy(__import__('numpy').asarray(ds[index], dtype='int64')).cuda().unsqueeze(0)
            pos = torch.arange(ids.shape[1], device='cuda').unsqueeze(0)
            attn = torch.triu(torch.ones((1, 1, ids.shape[1], ids.shape[1]), device='cuda', dtype=torch.bool), 1)
            logits = model(ids, pos, attn)[0, :-1].float()
            losses.append(float(F.cross_entropy(logits, ids[0, 1:], reduction='mean')))
    mean = sum(losses) / len(losses)
    return {"sequences": len(losses), "mean_nll": mean, "perplexity": math.exp(mean)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-data", type=Path)
    parser.add_argument("--stage2-data", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29595")
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)
    config = yaml.safe_load(args.config.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer_root / "tokenizer.json"))
    model = build_model(tokenizer.get_vocab_size(with_added_tokens=True), config)
    model.eval()
    results = {}
    for stage, path in (("stage1", args.stage1), ("stage2", args.stage2)):
        state = torch.load(path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        results[stage] = {"checkpoint_step": state["step"], **evaluate(model, tokenizer)}
        if args.stage1_data and args.stage2_data:
            results[stage]["indexed_data"] = {
                "stage1": evaluate_indexed(model, args.stage1_data),
                "stage2": evaluate_indexed(model, args.stage2_data),
            }
        del state
        torch.cuda.empty_cache()
        print(f"{stage} evaluated", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
