# Meridian 训练方案

Meridian 是独立项目。代码、checkpoint、日志和模型名称不包含硬件名或参数规模名。
当前训练设备是 **DGX Spark GB10**，后续可以扩展模型规模、数据量和 GPU 数量。

## 当前基线

基础模型从随机初始化开始，使用 BF16 decoder-only Transformer：16 层、hidden size
576、9 个 attention heads、3 个 query groups、FFN 1664、局部窗口 128、全局层 4/9/15、
序列长度 2048。模型名为 `Meridian-Base`，SFT/RL 产物分别命名为
`Meridian-Instruct` 和 `Meridian-Reasoning`。

架构参考来自 ZGCM 项目的公开训练设计；训练数据来自 MiniCPM 数据集。所有数据都由
本项目调用配置的官方 tokenizer 和 `chat_template.jinja` 重新编码。训练不加载上游
模型权重，也不混用其他 tokenizer。

## DGX Spark GB10 环境

```bash
git clone --recurse-submodules https://github.com/fengtc/meridian-training.git
cd meridian-training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -U pip setuptools wheel packaging ninja
```

先安装已经在 DGX Spark GB10 验证过的 ARM64、CUDA 13、BF16 PyTorch。不要把 RTX 5090
的 x86_64 环境或 wheel 复制到 Spark：

```bash
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple \
  torch torchvision torchaudio
```

确认 PyTorch 的 CUDA 和 BF16 可用后：

```bash
./scripts/bootstrap.sh
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e '.[templates]'
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e third_party/Megatron-LM --no-deps
python -m pip check
```

环境门禁：

```bash
export MERIDIAN_DATA_ROOT="$HOME/meridian-data"
export TOKENIZER_ROOT="$MERIDIAN_DATA_ROOT/tokenizers/official"
export DATA_ROOT="$MERIDIAN_DATA_ROOT/source-datasets"
export MERIDIAN_RUN_ROOT="$MERIDIAN_DATA_ROOT/training-runs"
./scripts/preflight.sh
```

门禁会检查 Megatron-LM submodule、官方 tokenizer、chat template、CUDA、BF16 和
当前 GPU 信息。训练数据和 tokenizer 应放在 Spark 的本地 NVMe 或高速挂载路径，避免
训练时从网络文件系统逐 token 读取。

## 阶段

| 阶段 | 数据 | 产物 |
| --- | --- | --- |
| Stage 1 | Ultra-FineWeb、中文通用文本、少量代码/知识 | `stage1_text_document.bin/.idx` |
| Stage 2 | Ultra-FineWeb-L3、UltraData-Math、代码、知识、中英文混合 | `stage2_text_document.bin/.idx` |
| SFT | UltraData-SFT-2605，assistant-only mask | `.bin/.idx` + `.mask.bin/.idx` |
| RL | UltraData-RL-2609 或 RLPR-Train-Dataset | prompt `.bin/.idx` + reward metadata |

Stage 1 使用 20M tokens，Stage 2 使用 80M tokens，SFT 使用 1M assistant tokens。
Stage 2 必须从 Stage 1 checkpoint 继续，SFT 必须从 Stage 2 checkpoint 继续。

RL 的 `ground_truth` 不放进 prompt，而保存在 metadata 中，由 verifier 计算 reward；
Code 样本必须在隔离沙箱中执行测试。

## 数据编码和训练

```bash
export DATA_STAGE1_INPUT=/data/source/stage1/**/*.jsonl
export DATA_STAGE2_INPUT=/data/source/stage2/**/*.jsonl
export DATA_SFT_INPUT=/data/source/sft/**/*.jsonl
export DATA_RL_INPUT=/data/source/rl/**/*.jsonl
export DATA_RL_TARGET_SAMPLES=85995

./scripts/encode-data.sh stage1
./scripts/encode-data.sh stage2
./scripts/encode-data.sh sft
./scripts/encode-data.sh rl

./scripts/run-stage.sh stage1
./scripts/run-stage.sh stage2
./scripts/run-stage.sh sft
```

`third_party/Megatron-LM` 是本项目唯一的外部训练框架依赖。升级它时必须同步更新
submodule gitlink、`dependencies.lock` 和 smoke/resume 验证记录。
