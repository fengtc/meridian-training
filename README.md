# Meridian Training

完整的 DGX Spark GB10 中文实操流程见：[docs/RUNBOOK.md](docs/RUNBOOK.md)。

Meridian 是一个独立、可扩展的语言模型训练项目。项目名称和模型名称不绑定硬件、
参数规模或数据集来源：

```text
基础模型：Meridian-Base
指令模型：Meridian-Instruct
推理模型：Meridian-Reasoning
```

本项目当前在 **DGX Spark GB10** 上执行训练。后续可以扩展模型层数、hidden size、
训练数据量，并切换到一张或多张 GPU；项目名称和模型名称不需要改变。

## 来源说明

当前基础模型采用公开混合注意力设计作为架构参考：16 层、hidden size 576、9 个
attention heads、3 个 query groups、FFN 1664、局部窗口 128、全局层 4/9/15、
上下文长度 2048、BF16。

训练数据来自 MiniCPM 数据集，使用配置的官方 tokenizer 和 `chat_template.jinja`
重新编码。模型从随机权重开始，不加载任何上游模型权重。架构参考和数据来源会记录
在实验文档中，运行代码和模型名称使用 Meridian 的中性命名。

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

然后安装适配 DGX Spark GB10、CUDA 和 ARM64 的 PyTorch。PyTorch 必须支持 CUDA、
BF16，并且要与 Spark 当前镜像和驱动匹配。PyTorch 安装完成后执行：

```bash
./scripts/bootstrap.sh
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e '.[templates]'
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e third_party/Megatron-LM --no-deps
python -m pip check
```

最后运行环境检查：

```bash
export TOKENIZER_ROOT=/data/tokenizers/official
export DATA_ROOT=/data/source-datasets
export MERIDIAN_RUN_ROOT=/data/meridian-training-runs
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

首轮在 DGX Spark GB10 上使用单进程、micro batch 1、gradient accumulation 2，
每个 optimizer step 处理 4096 tokens。确认单 GPU 流程后，再扩展多 GPU 数据并行。

完整阶段、数据来源和 checkpoint 续训规则见
[docs/PLAN.md](docs/PLAN.md)。
