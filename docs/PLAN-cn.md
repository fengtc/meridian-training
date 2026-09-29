# Meridian 训练方案

Meridian 是独立项目。代码、checkpoint、日志和模型名称不包含硬件名、参数规模名、
ZGCM 或 MiniCPM。当前第一阶段只是验证配置，后续可以扩展模型宽度、层数、数据量和
GPU 数量。

## 当前基线

基础模型从随机初始化开始，使用 BF16 decoder-only Transformer：16 层、hidden size
576、9 个 attention heads、3 个 query groups、FFN 1664、局部窗口 128、全局层 4/9/15、
序列长度 2048。模型名为 `Meridian-Base`，SFT/RL 产物分别命名为
`Meridian-Instruct` 和 `Meridian-Reasoning`。

架构参考来自 ZGCM 项目的公开训练设计；训练数据来自 MiniCPM 数据集。所有数据都由
本项目调用配置的官方 tokenizer 和 `chat_template.jinja` 重新编码，不能加载上游模型
权重，也不能混用本项目以外的 tokenizer。

## 阶段

| 阶段 | 数据 | 产物 |
| --- | --- | --- |
| Stage 1 | Ultra-FineWeb、中文通用文本、少量代码/知识 | `stage1_text_document.bin/.idx` |
| Stage 2 | Ultra-FineWeb-L3、UltraData-Math、代码、知识、中英文混合 | `stage2_text_document.bin/.idx` |
| SFT | UltraData-SFT-2605，assistant-only mask | `.bin/.idx` + `.mask.bin/.idx` |
| RL | UltraData-RL-2609 或 RLPR-Train-Dataset | prompt `.bin/.idx` + reward metadata |

Stage 2 从 Stage 1 checkpoint 继续，SFT 从 Stage 2 checkpoint 继续。RL 的
`ground_truth` 不放进 prompt，而保存在 metadata 中，由 verifier 计算 reward；Code
样本必须在隔离沙箱中执行测试。

## 运行

```bash
export TOKENIZER_ROOT=/data/tokenizers/official
export DATA_ROOT=/data/source-datasets
export MERIDIAN_RUN_ROOT=/data/meridian-training-runs

./scripts/bootstrap.sh
./scripts/preflight.sh

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

`third_party/Megatron-LM` 是本项目唯一的外部训练框架依赖。升级它时必须更新
submodule gitlink、`dependencies.lock` 和 smoke/resume 验证记录。
