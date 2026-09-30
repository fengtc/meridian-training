# Meridian Training

完整的 DGX Spark GB10 中文实操流程见：[docs/RUNBOOK.md](docs/RUNBOOK.md)。

Meridian 是一个独立、可扩展的语言模型训练项目。项目名称和模型名称不绑定硬件、
参数规模或数据集来源：

```text
基础模型：Meridian-Base
指令模型：Meridian-Instruct
推理模型：Meridian-Reasoning
```

本项目支持 DGX Spark GB10、双卡 RTX 5090 等 CUDA 主机。当前正式配置为
`configs/model-500m.yaml`：约 505M 参数、2048 序列长度、BF16、DDP 数据并行。

## 来源说明

当前基础模型采用 24 层、hidden size 1024、16 个 attention heads、4 个 query
groups、FFN 3840、局部窗口 128、全局层 4/9/15/21、上下文长度 2048、BF16。

训练数据使用新下载的公开数据，使用仓库内 `data/tokenizer/` 的官方
tokenizer 和 `chat_template.jinja` 重新编码。该目录包含 MIT 许可证和来源说明，clone
仓库后不需要再次下载 tokenizer。模型从随机权重开始，不加载任何上游模型权重。架构参考和数据来源会记录
在实验文档中，运行代码和模型名称使用 Meridian 的中性命名。

## Tokenizer

默认直接使用仓库内的 tokenizer：

```bash
export TOKENIZER_ROOT="$PROJECT_ROOT/data/tokenizer"
test -s "$TOKENIZER_ROOT/tokenizer.json"
test -s "$TOKENIZER_ROOT/chat_template.jinja"
```

如果实验需要另一份兼容 tokenizer，可以通过 `TOKENIZER_ROOT` 覆盖，但必须把
`tokenizer.json`、`chat_template.jinja`、来源说明和 SHA256 写入实验 manifest。

## 依赖边界

Megatron-LM 是唯一的训练框架依赖，以锁定版本的 Git submodule 放在
`third_party/Megatron-LM`。数据编码、模型配置、checkpoint 格式、SFT mask、RL
prompt metadata 和训练脚本都属于本项目。

在 DGX Spark GB10 上准备环境：

```bash
git clone --recurse-submodules https://github.com/fengtc/meridian-training.git
cd meridian-training

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -U pip setuptools wheel packaging ninja
```

然后安装已经在 DGX Spark GB10（GB10、CUDA 13.0、Python 3.12）验证过的 PyTorch：

```bash
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple \
  torch torchvision torchaudio
```

该命令应安装 `torch 2.14.0+cu130` 或与当前 Spark 镜像匹配的 CUDA 13 ARM64 版本。
安装后先确认 CUDA 和 BF16，再安装项目依赖：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
```

PyTorch 必须支持 CUDA、BF16，并且要与 Spark 当前镜像和驱动匹配。确认通过后执行：

```bash
./scripts/bootstrap.sh
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e '.[templates]'
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e third_party/Megatron-LM --no-deps
python -m pip check
```

最后运行环境检查：

```bash
export MERIDIAN_DATA_ROOT="$HOME/meridian-data"
export TOKENIZER_ROOT="$MERIDIAN_DATA_ROOT/tokenizers/official"
export DATA_ROOT="$MERIDIAN_DATA_ROOT/source-datasets"
export MERIDIAN_RUN_ROOT="$MERIDIAN_DATA_ROOT/training-runs"
./scripts/preflight.sh
```

## 数据产物

编码器接受 JSON、JSONL 和 Parquet，使用配置的官方 tokenizer 生成 Megatron
`IndexedDataset`：

```text
预训练：<prefix>.bin / <prefix>.idx
SFT：   <prefix>.bin / <prefix>.idx + <prefix>.mask.bin/.idx
RL：    <prefix>.bin / <prefix>.idx + <prefix>.metadata.jsonl
```

RL metadata 保存 `uuid`、`domain`、`source`、`query`、`ground_truth` 和 prompt 长度，
供在线 rollout 和 reward verifier 使用。

## 当前训练入口

训练入口读取 YAML，支持双卡 DDP、rank 分片、梯度累积、Stage 1/Stage 2/SFT、每
rank checkpoint 和断点恢复。正式预训练配额为 Stage 1 2B + Stage 2 8B tokens；
先运行 `scripts/pilot-20m.sh` 完成 20M 双卡验证。下载脚本支持
`scripts/download-data.sh stage1-target -2B` 这样的 token 目标参数。

完整阶段、数据来源和 checkpoint 续训规则见
[docs/PLAN.md](docs/PLAN.md)。
