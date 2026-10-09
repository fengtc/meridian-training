#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch, yaml
from tokenizers import Tokenizer
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from meridian_training.train import build_model

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True, type=Path)
ap.add_argument('--config', required=True, type=Path)
ap.add_argument('--tokenizer-root', required=True, type=Path)
ap.add_argument('--top-k', type=int, default=10)
a = ap.parse_args()
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29597')
torch.distributed.init_process_group('nccl', rank=0, world_size=1)
parallel_state.initialize_model_parallel(1,1); model_parallel_cuda_manual_seed(42)
tok=Tokenizer.from_file(str(a.tokenizer_root/'tokenizer.json'))
model=build_model(tok.get_vocab_size(with_added_tokens=True), yaml.safe_load(a.config.read_text())).eval()
state=torch.load(a.checkpoint,map_location='cuda',weights_only=False); model.load_state_dict(state['model'])
print(f'checkpoint_step={state.get("step")} 输入前缀，查看下一个 token；输入 exit 退出。')
with torch.inference_mode():
  while True:
    try: text=input('\n前缀> ').strip()
    except (EOFError,KeyboardInterrupt): print(); break
    if not text or text.lower() in {'exit','quit'}: break
    ids=tok.encode(text).ids
    x=torch.tensor([ids],device='cuda'); pos=torch.arange(len(ids),device='cuda').unsqueeze(0)
    attn=torch.triu(torch.ones((1,1,len(ids),len(ids)),device='cuda',dtype=torch.bool),1)
    probs=torch.softmax(model(x,pos,attn)[0,-1].float(),dim=-1); vals,inds=probs.topk(a.top_k)
    for rank,(v,i) in enumerate(zip(vals,inds),1): print(f'{rank:2d}. {tok.decode([int(i)],skip_special_tokens=False)!r}  p={float(v):.6f}')
torch.distributed.destroy_process_group()
