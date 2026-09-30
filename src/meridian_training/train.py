#!/usr/bin/env python3
"""Config driven BF16 DDP trainer for Meridian stages."""
from __future__ import annotations
import argparse, json, math, os, random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import yaml
from tokenizers import Tokenizer
from megatron.core import parallel_state
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('stage1','stage2','sft'), required=True)
    p.add_argument('--config', required=True); p.add_argument('--dataset-prefix', required=True)
    p.add_argument('--output-root', required=True); p.add_argument('--tokenizer-root', required=True)
    p.add_argument('--target-tokens', type=int, required=True); p.add_argument('--resume', type=Path)
    p.add_argument('--micro-batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--peak-lr', type=float, default=2e-4); p.add_argument('--min-lr', type=float, default=2e-5)
    p.add_argument('--warmup-ratio', type=float, default=0.02); p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0); p.add_argument('--checkpoint-interval', type=int, default=1000)
    return p.parse_args()

def build_model(vocab, cfg):
    seq = int(cfg.get('sequence_length', 2048)); layers = int(cfg['num_layers']); pattern = [1] * layers
    for n in cfg.get('attention', {}).get('global_layers_one_based', []):
        if 1 <= n <= layers: pattern[n - 1] = 0
    tc = TransformerConfig(num_layers=layers, hidden_size=int(cfg['hidden_size']),
        num_attention_heads=int(cfg['num_attention_heads']), num_query_groups=int(cfg['num_query_groups']),
        ffn_hidden_size=int(cfg['ffn_hidden_size']), kv_channels=int(cfg['hidden_size']) // int(cfg['num_attention_heads']),
        normalization='RMSNorm', gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, hidden_dropout=0.0, attention_dropout=0.0, attention_backend=AttnBackend.local,
        window_size=(int(cfg.get('attention', {}).get('local_window', 128)), 0), window_attn_skip_freq=pattern,
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, use_cpu_initialization=False,
        init_method_std=0.02, layernorm_epsilon=1e-6)
    return GPTModel(config=tc, transformer_layer_spec=get_gpt_layer_local_spec(normalization='RMSNorm'),
        vocab_size=vocab, max_sequence_length=seq, position_embedding_type='rope', rotary_percent=1.0,
        rotary_base=int(cfg.get('rotary_base', 10000)), parallel_output=False,
        share_embeddings_and_output_weights=bool(cfg.get('share_embeddings_and_output_weights', True))).cuda().to(dtype=torch.bfloat16)

def get_batch(ds, index, device, seq, mask_ds=None):
    x = torch.from_numpy(np.asarray(ds[index], dtype=np.int64)).to(device).long().unsqueeze(0)
    if x.shape[1] != seq:
        raise RuntimeError(f'dataset item {index} has length {x.shape[1]}, expected {seq}')
    pos = torch.arange(seq, device=device).unsqueeze(0)
    attn = torch.triu(torch.ones((1, 1, seq, seq), device=device, dtype=torch.bool), 1)
    mask = None
    if mask_ds is not None:
        mask = torch.from_numpy(np.asarray(mask_ds[index], dtype=np.float32)).to(device).unsqueeze(0)
    return x, pos, attn, mask

def main():
    a = args(); cfg = yaml.safe_load(Path(a.config).read_text()); seq = int(cfg.get('sequence_length', 2048))
    rank = int(os.environ.get('RANK', 0)); local = int(os.environ.get('LOCAL_RANK', rank)); world = int(os.environ.get('WORLD_SIZE', 1))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA BF16 is required')
    torch.cuda.set_device(local); dist.init_process_group('nccl', rank=rank, world_size=world); parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(a.seed); torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank); random.seed(a.seed + rank)
    tok = Tokenizer.from_file(str(Path(a.tokenizer_root) / 'tokenizer.json')); vocab = tok.get_vocab_size(with_added_tokens=True)
    prefix = Path(a.dataset_prefix).with_suffix(''); ds = IndexedDataset(str(prefix), mmap=True)
    mask_ds = IndexedDataset(str(prefix) + '.mask', mmap=True) if a.stage == 'sft' else None
    if mask_ds is not None and len(mask_ds) != len(ds): raise SystemExit('SFT token/mask counts differ')
    if a.micro_batch_size < 1 or a.gradient_accumulation < 1:
        raise SystemExit('micro-batch-size and gradient-accumulation must be positive')
    global_batch = world * a.gradient_accumulation * a.micro_batch_size
    requested = math.ceil(a.target_tokens / (seq * global_batch)); total = min(requested, len(ds) // global_batch)
    if total <= 0: raise SystemExit('dataset has no complete DDP batch')
    out = Path(a.output_root); (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    model = build_model(vocab, cfg); count = sum(x.numel() for x in model.parameters())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.peak_lr, weight_decay=a.weight_decay); start = 0
    resume_state = None
    if a.resume:
        resume = a.resume / f'step-latest-rank{rank}.pt' if a.resume.is_dir() else a.resume
        resume_state = torch.load(resume, map_location=f'cuda:{local}', weights_only=False)
        if resume_state.get('world_size', world) != world:
            raise SystemExit(f"checkpoint world_size={resume_state['world_size']} differs from current world_size={world}")
        if resume_state.get('sequence_length', seq) != seq:
            raise SystemExit('checkpoint sequence_length differs from model config')
        model.module.load_state_dict(resume_state['model'])
        if resume_state.get('stage') == a.stage:
            opt.load_state_dict(resume_state['optimizer'])
            start = int(resume_state.get('step', 0))
            if 'rng' in resume_state: torch.set_rng_state(resume_state['rng'].cpu())
            if 'cuda_rng' in resume_state: torch.cuda.set_rng_state(resume_state['cuda_rng'].cpu())
    warmup = max(1, round(total * a.warmup_ratio))
    def lr_factor(step):
        if step <= warmup: return max(a.min_lr / a.peak_lr, step / warmup)
        return a.min_lr / a.peak_lr + 0.5 * (1 - a.min_lr / a.peak_lr) * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    if resume_state is not None and resume_state.get('stage') == a.stage and resume_state.get('scheduler'):
        sched.load_state_dict(resume_state['scheduler'])
    if rank == 0: print(json.dumps({'world_size': world, 'parameter_count': count, 'parameters_millions': count / 1e6, 'steps': total, 'tokens_per_step': seq * global_batch}), flush=True)
    if start >= total:
        if rank == 0: print(json.dumps({'stage': a.stage, 'status': 'already_complete', 'step': start}), flush=True)
        dist.barrier(); dist.destroy_process_group(); return
    for local_step in range(start, total):
        step = local_step + 1; opt.zero_grad(set_to_none=True); loss_sum = 0.0
        for acc in range(a.gradient_accumulation):
            batch_indices = [(local_step * a.gradient_accumulation + acc) * world * a.micro_batch_size + rank * a.micro_batch_size + micro for micro in range(a.micro_batch_size)]
            samples = [get_batch(ds, idx, torch.device(f'cuda:{local}'), seq, mask_ds) for idx in batch_indices]
            x = torch.cat([item[0] for item in samples], dim=0)
            pos, attn = samples[0][1], samples[0][2]
            token_mask = torch.cat([item[3] for item in samples], dim=0) if mask_ds is not None else None
            sync_context = nullcontext() if acc == a.gradient_accumulation - 1 else model.no_sync()
            with sync_context:
                logits = model(x, pos, attn); labels = x[:, 1:]
                per_token = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none').view_as(labels)
                if token_mask is not None:
                    weights = token_mask[:, 1:]
                    loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    loss = per_token.mean()
                (loss / a.gradient_accumulation).backward(); loss_sum += float(loss.detach())
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step(); sched.step(); torch.cuda.synchronize()
        if rank == 0 and (step == start + 1 or step % 100 == 0 or step == total):
            record = {'stage': a.stage, 'step': step, 'consumed_tokens': step * seq * global_batch, 'loss': loss_sum / a.gradient_accumulation, 'lr': opt.param_groups[0]['lr'], 'grad_norm': float(grad)}
            print(json.dumps(record), flush=True); (out / 'metrics.jsonl').open('a').write(json.dumps(record) + '\n')
        if step % a.checkpoint_interval == 0 or step == total:
            state = {'model': model.module.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                     'step': step, 'data_cursor': local_step + 1, 'rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state(), 'stage': a.stage, 'parameter_count': count,
                     'target_tokens': a.target_tokens, 'sequence_length': seq, 'world_size': world}
            checkpoint_dir = out / 'checkpoints'
            step_path = checkpoint_dir / f'step-{step}-rank{rank}.pt'
            latest_path = checkpoint_dir / f'step-latest-rank{rank}.pt'
            temp_step = step_path.with_suffix('.pt.tmp')
            temp_latest = latest_path.with_suffix('.pt.tmp')
            torch.save(state, temp_step); os.replace(temp_step, step_path)
            torch.save(state, temp_latest); os.replace(temp_latest, latest_path)
            dist.barrier()
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__': main()
