# Meridian Training

Meridian is a standalone, scalable language-model training repository. The
project name and model names are independent of hardware, parameter count, and
any upstream dataset provider:

```text
Base model:       Meridian-Base
Instruction model: Meridian-Instruct
Reasoning model:   Meridian-Reasoning
```

The repository is designed to grow from the first small validation run to
larger models, larger corpora, and one or more CUDA GPUs. It does not encode a
GPU model, a parameter count, or a temporary experiment size in its public
names.

## Provenance

The initial model recipe follows the published hybrid-attention design used as
the architecture reference: 16 layers, hidden size 576, 9 attention heads, 3
query groups, FFN size 1664, local attention window 128, global layers 4/9/15,
2048 context, and BF16. The initial training data is sourced from MiniCPM
training datasets and is retokenized with the supplied official tokenizer and
chat template. No upstream model weights are loaded.

These provenance statements are part of the experiment record. They do not
make Meridian a fork of an upstream model project, and the runtime code uses
the neutral Meridian names above.

## Dependency boundary

Megatron-LM is the only model-training framework dependency. It is kept as a
locked git submodule under `third_party/Megatron-LM`; the Meridian repository
does not import code from another project checkout. The data encoder, model
configuration, checkpoint contract, SFT masks, RL prompt metadata, and launch
scripts live in this repository.

```bash
git clone <meridian-repository-url> meridian-training
cd meridian-training
git submodule update --init --recursive
python -m pip install -e '.[templates]'
```

Use a CUDA-enabled PyTorch build appropriate for the host before installing the
Python package. Run `scripts/preflight.sh` before encoding data.

## Data interfaces

The encoder accepts JSON, JSONL, and Parquet. It produces Megatron
`IndexedDataset` files using the configured tokenizer:

```text
pretrain:  <prefix>.bin / <prefix>.idx
sft:       <prefix>.bin / <prefix>.idx plus <prefix>.mask.bin/.idx
rl:        <prefix>.bin / <prefix>.idx plus <prefix>.metadata.jsonl
```

RL prompt metadata keeps `uuid`, `domain`, `source`, `query`, `ground_truth`,
and prompt length outside the model input for reward verification.

## Current launch shape

The first run uses one process, micro-batch 1, gradient accumulation 2, and
4096 tokens per optimizer step. The same scripts can be extended to multiple
processes by changing the distributed launcher and data-parallel configuration;
the repository name and model names remain unchanged.

See [docs/PLAN-cn.md](docs/PLAN-cn.md) for the current staged recipe and data
provenance notes.
