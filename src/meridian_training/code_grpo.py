#!/usr/bin/env python3
"""Small single-host Code GRPO trainer for the 500M pilot.

This intentionally keeps rollout and learning in one DDP process per GPU. It
is suitable for a 2-GPU pilot, while larger jobs should use ZGCM-1/AReaL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from meridian_training.rl import CodeReward
from meridian_training.retokenize import render_rl_prompt
from meridian_training.train import build_model
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--tokenizer-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--batch-groups', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--peak-lr', type=float, default=1e-6)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=Path)
    return p.parse_args()


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((1, 1, length, length), device=device, dtype=torch.bool), 1)


def read_rows(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('domain') in (None, 'Code', 'code') and (row.get('ground_truth') or row.get('answers')):
                    if 'ground_truth' not in row:
                        row['ground_truth'] = row['answers']
                    yield row


def sample_completion(model, prompt, seq, eos, max_new, temperature, device):
    ids = list(prompt[-(seq - 1):])
    with torch.inference_mode():
        for _ in range(max_new):
            x = torch.tensor([ids], device=device, dtype=torch.long)
            pos = torch.arange(len(ids), device=device).unsqueeze(0)
            logits = model(x, pos, causal_mask(len(ids), device))[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1).view(1)
            token = int(nxt.item()); ids.append(token)
            if token == eos or len(ids) >= seq:
                break
    return ids, len(ids) - len(prompt[-(seq - 1):])


def token_logprobs(model, ids, prompt_len, seq, device):
    ids = ids[-seq:]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    pos = torch.arange(len(ids), device=device).unsqueeze(0)
    logits = model(x, pos, causal_mask(len(ids), device))[:, :-1].float()
    labels = x[:, 1:]
    logp = torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_len - 1)
    return logp[:, start:]


def main():
    a = parse_args()
    rank = int(os.environ.get('RANK', 0)); local = int(os.environ.get('LOCAL_RANK', rank)); world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local); device = torch.device(f'cuda:{local}')
    dist.init_process_group('nccl', rank=rank, world_size=world)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(a.seed); torch.manual_seed(a.seed + rank); random.seed(a.seed + rank); np.random.seed(a.seed + rank)
    cfg = yaml.safe_load(a.config.read_text()); seq = int(cfg.get('sequence_length', 2048))
    tok = Tokenizer.from_file(str(a.tokenizer_root / 'tokenizer.json'))
    eos = tok.token_to_id('<|endoftext|>') or tok.token_to_id('<|eos|>')
    if eos is None: raise SystemExit('tokenizer has no EOS token')
    model = build_model(tok.get_vocab_size(with_added_tokens=True), cfg)
    state_path = a.checkpoint / f'step-latest-rank{rank}.pt' if a.checkpoint.is_dir() else a.checkpoint
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model']); model = DDP(model, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.peak_lr, weight_decay=0.01)
    start = 0
    if a.resume:
        rs = torch.load(a.resume / f'step-latest-rank{rank}.pt' if a.resume.is_dir() else a.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(rs['model']); optimizer.load_state_dict(rs['optimizer']); start = int(rs.get('step', 0))
    rows = list(read_rows(a.input)); rows = rows[rank::world]
    if not rows: raise SystemExit('no Code RL rows for this rank')
    reward_fn = CodeReward(); out = a.output_root; (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    for step in range(start, a.steps):
        row = rows[step % len(rows)]
        query = row.get('query') or row.get('messages') or ''
        prompt = render_rl_prompt(a.tokenizer_root, tok, query)
        prompt = prompt[-max(1, seq - a.max_new_tokens - 1):]
        group = []; rewards = []
        for _ in range(a.group_size):
            generated, new_count = sample_completion(model, prompt, seq, eos, a.max_new_tokens, a.temperature, device)
            text = tok.decode(generated[len(prompt):], skip_special_tokens=False)
            result = reward_fn(text, row['ground_truth']); group.append((generated, len(prompt), result.reward)); rewards.append(result.reward)
        mean = float(np.mean(rewards)); std = float(np.std(rewards)); advantages = [(r - mean) / max(std, 1e-6) for r in rewards]
        optimizer.zero_grad(set_to_none=True); losses = []
        for (ids, prompt_len, _), advantage in zip(group, advantages):
            lp = token_logprobs(model, ids, prompt_len, seq, device).mean()
            losses.append(-lp * float(advantage) / a.group_size)
        loss = torch.stack(losses).sum(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        record = {'step': step + 1, 'reward_mean': mean, 'reward_std': std, 'rewards': rewards, 'loss': float(loss.detach()), 'rank': rank}
        if rank == 0: print(json.dumps(record), flush=True); (out / 'metrics.jsonl').open('a').write(json.dumps(record) + '\n')
        if (step + 1) % 10 == 0 or step + 1 == a.steps:
            ckpt = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step + 1, 'stage': 'code_grpo', 'world_size': world}
            torch.save(ckpt, out / 'checkpoints' / f'step-latest-rank{rank}.pt')
            dist.barrier()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
