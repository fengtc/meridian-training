#!/usr/bin/env python3
"""Meridian 训练入口，支持一个或多个 CUDA 进程。"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
from megatron.core import parallel_state
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


SEQ = 2048
LAYERS = 16
HIDDEN = 576
HEADS = 9
GROUPS = 3
FFN = 1664
LOCAL_PATTERN = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]


def make_model(vocab_size: int) -> GPTModel:
    cfg = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        ffn_hidden_size=FFN,
        kv_channels=HIDDEN // HEADS,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.local,
        window_size=(128, 0),
        window_attn_skip_freq=LOCAL_PATTERN,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        init_method_std=0.02,
        layernorm_epsilon=1e-6,
    )
    return GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=vocab_size,
        max_sequence_length=SEQ,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
    ).cuda().to(dtype=torch.bfloat16)


def load_batch(ds, index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    item = np.asarray(ds[index], dtype=np.int64)
    x = torch.from_numpy(item).to(device=device, dtype=torch.long).unsqueeze(0)
    pos = torch.arange(SEQ, device=device).unsqueeze(0)
    mask = torch.triu(torch.ones((1, 1, SEQ, SEQ), device=device, dtype=torch.bool), 1)
    return x, pos, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stage1", "stage2", "sft"), required=True)
    ap.add_argument("--dataset-prefix", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--tokenizer-root", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()
    if args.micro_batch_size != 1:
        raise SystemExit("Meridian runner requires micro batch size 1")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 is required")
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tokenizer_root) / "tokenizer.json"))
    vocab = tok.get_vocab_size(with_added_tokens=True)
    prefix = Path(args.dataset_prefix)
    if prefix.suffix in {".bin", ".idx"}:
        prefix = prefix.with_suffix("")
    ds = IndexedDataset(str(prefix), mmap=True)
    mask_ds = IndexedDataset(str(prefix) + ".mask", mmap=True) if args.stage == "sft" else None
    if mask_ds is not None and len(mask_ds) != len(ds):
        raise SystemExit("SFT token and mask datasets have different sequence counts")
    world = 1
    tokens_per_step = SEQ * args.micro_batch_size * args.gradient_accumulation * world
    requested_steps = math.ceil(args.target_tokens / tokens_per_step)
    available_steps = max(0, len(ds) // args.gradient_accumulation)
    total_steps = available_steps if args.stage == "sft" else min(requested_steps, available_steps)
    if total_steps <= 0:
        raise SystemExit("dataset has no complete gradient-accumulation batch")
    out = Path(args.output_root)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(args.seed)
    model = make_model(vocab)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"vocab_size": vocab, "parameter_count": parameter_count,
                      "parameter_count_millions": parameter_count / 1e6,
                      "model": {"layers": LAYERS, "hidden_size": HIDDEN,
                                "attention_heads": HEADS, "query_groups": GROUPS,
                                "ffn_hidden_size": FFN}}, sort_keys=True), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.0)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cuda", weights_only=False)
        required = {"model", "optimizer", "scheduler", "rng", "cuda_rng", "python_rng", "numpy_rng", "consumed_tokens"}
        missing = sorted(required.difference(state))
        if missing:
            raise SystemExit(f"resume checkpoint is incomplete; missing: {', '.join(missing)}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        if "rng" in state:
            torch.set_rng_state(state["rng"].cpu())
        if "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
        # A new stage gets its own cosine schedule while retaining Adam moments.
        for group in optimizer.param_groups:
            group["lr"] = args.peak_lr
        print(json.dumps({"resume": str(args.resume), "resume_step": start_step}), flush=True)
    warmup = max(1, round(total_steps * args.warmup_ratio))
    def lr_factor(step: int) -> float:
        if step <= warmup:
            return max(args.min_lr / args.peak_lr, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return (args.min_lr / args.peak_lr) + 0.5 * (1 - args.min_lr / args.peak_lr) * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    log_path = out / "metrics.jsonl"
    device = torch.device("cuda")
    assistant_seen = 0
    for local_step in range(total_steps):
        started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0; valid = 0
        for accum in range(args.gradient_accumulation):
            index = local_step * args.gradient_accumulation + accum
            x, pos, attn = load_batch(ds, index, device)
            logits = model(x, pos, attn)
            labels = x[:, 1:]
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, vocab), labels.reshape(-1), reduction="none")
            if mask_ds is not None:
                mask = torch.from_numpy(np.asarray(mask_ds[index], dtype=np.float32)[1:]).to(device)
                loss = (losses * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)
                valid += int(mask.sum().item())
            else:
                loss = losses.mean(); valid += SEQ - 1
            (loss / args.gradient_accumulation).backward(); loss_sum += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm) or not math.isfinite(loss_sum):
            raise RuntimeError("non-finite loss or gradient")
        optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        assistant_seen += valid if args.stage == "sft" else 0
        step = local_step + 1; elapsed = time.perf_counter() - started
        record = {"stage": args.stage, "step": step, "resumed_from_step": start_step,
                  "consumed_tokens": step * tokens_per_step, "effective_target_tokens": valid,
                  "train_loss": loss_sum / args.gradient_accumulation, "learning_rate": optimizer.param_groups[0]["lr"],
                  "grad_norm": float(grad_norm), "tokens_per_sec": tokens_per_step / elapsed}
        with log_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record) + "\n")
        if step == 1 or step % 100 == 0 or step == total_steps:
            print(json.dumps(record), flush=True)
        if step == total_steps or step % max(1, total_steps // 5) == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "step": step, "consumed_tokens": step * tokens_per_step, "vocab_size": vocab,
                        "parameter_count": parameter_count,
                        "stage": args.stage, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, out / "checkpoints" / f"step-{step}.pt")
        if args.stage == "sft" and assistant_seen >= args.target_tokens:
            break
    final = out / "checkpoints" / f"{args.stage}-final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "consumed_tokens": step * tokens_per_step, "vocab_size": vocab,
                "parameter_count": parameter_count,
                "stage": args.stage, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, final)
    print(f"final_checkpoint={final}", flush=True)
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
