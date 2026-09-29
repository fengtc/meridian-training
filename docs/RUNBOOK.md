# Meridian 实操运行手册

本文按 DGX Spark GB10、单张可见 GPU 编写。命令从空环境开始，默认使用 `/data` 保存数据和检查点；如果磁盘位置不同，只需要修改第二步的三个路径变量。项目代码参考 ZGCM 的模型设计，训练数据来自 OpenBMB/MiniCPM 数据集，模型从随机权重开始训练。

## 1. 获取代码

```bash
cd /home/paratera
git clone https://github.com/fengtc/meridian-training.git
cd /home/paratera/meridian-training
git submodule update --init --recursive
```

如果代码已经存在，使用下面两条命令更新：

```bash
cd /home/paratera/meridian-training
git pull --ff-only origin main
git submodule update --init --recursive
```

## 2. 创建 Python 环境

DGX Spark 的 PyTorch 必须使用支持 GB10、CUDA 和 BF16 的版本。先确认机器镜像是否已经提供 PyTorch：

```bash
python3 --version
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

创建项目自己的虚拟环境并安装项目依赖。不要在这里盲目覆盖厂商提供的 PyTorch：

```bash
cd /home/paratera/meridian-training
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[templates]'
python -m pip install -e third_party/Megatron-LM --no-deps
python -m pip install modelscope
```

检查环境：

```bash
./scripts/preflight.sh
```

看到 `环境检查通过` 后再继续。若提示 CUDA 或 BF16 不可用，应先修复 DGX Spark 镜像中的 PyTorch，不要用 CPU 运行本项目。

## 3. 设置目录和官方 tokenizer

下面的变量在当前终端会话中有效。官方 tokenizer 必须包含 `tokenizer.json` 和 `chat_template.jinja`，不能使用 MiniCPM5 tokenizer：

```bash
cd /home/paratera/meridian-training
source .venv/bin/activate
export TOKENIZER_ROOT=/data/tokenizers/official
export DATA_ROOT=/data/source-datasets
export MERIDIAN_RUN_ROOT=/data/meridian-training-runs
mkdir -p "$TOKENIZER_ROOT" "$DATA_ROOT" "$MERIDIAN_RUN_ROOT"

# 如果官方文件已经在此目录，直接复制；路径按实际位置修改。
./scripts/download-data.sh tokenizer /home/ubuntu/ZGCM-Training-Lab/data/tokenizer/zgcm-1-official
./scripts/preflight.sh
```

如果 tokenizer 在其他机器或挂载目录，把上面 `tokenizer` 命令最后的路径改成包含这两个文件的目录。

## 4. 下载数据集

以下命令通过 ModelScope 下载 OpenBMB 数据集。预训练数据可能很大，请先确认 `/data` 剩余空间；正式训练只会读取达到目标 token 数所需的部分。

```bash
./scripts/download-data.sh ultra-fineweb
./scripts/download-data.sh ultra-fineweb-l3
./scripts/download-data.sh math
./scripts/download-data.sh code
./scripts/download-data.sh sft
./scripts/download-data.sh rl
```

可选的 RLPR 数据集：

```bash
./scripts/download-data.sh rlpr
```

确认下载结果中确实存在 JSON、JSONL 或 Parquet 文件：

```bash
find "$DATA_ROOT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' \) | head -30
```

## 5. 生成 IndexedDataset

编码工具会用官方 tokenizer 读取原始数据，输出 Megatron 兼容的 `.bin/.idx`。下面的命令自动收集下载目录中的三种支持格式，并用逗号合并多个数据源。

```bash
export DATA_STAGE1_INPUT="$(find "$DATA_ROOT/ultra-fineweb" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' \) -printf '%p,' | sed 's/,$//')"
export DATA_STAGE2_INPUT="$(find "$DATA_ROOT/ultra-fineweb-l3" "$DATA_ROOT/ultradata-math" "$DATA_ROOT/ultradata-code" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' \) -printf '%p,' | sed 's/,$//')"
export DATA_SFT_INPUT="$(find "$DATA_ROOT/ultradata-sft-2605" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' \) -printf '%p,' | sed 's/,$//')"
export DATA_RL_INPUT="$(find "$DATA_ROOT/ultradata-rl-2609" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' \) -printf '%p,' | sed 's/,$//')"
for name in DATA_STAGE1_INPUT DATA_STAGE2_INPUT DATA_SFT_INPUT DATA_RL_INPUT; do
  test -n "${!name}" || { echo "$name 为空，请检查数据下载目录"; exit 1; }
done
```

先编码 RL prompt（它会同时生成同序号的 `metadata.jsonl`，保存 ground truth、domain 和 source）：

```bash
export DATA_RL_TARGET_SAMPLES=1000
./scripts/encode-data.sh rl
ls -lh "$MERIDIAN_RUN_ROOT/rl/rl_text_document".*
```

再编码三个训练阶段：

```bash
./scripts/encode-data.sh stage1
./scripts/encode-data.sh stage2
./scripts/encode-data.sh sft
ls -lh "$MERIDIAN_RUN_ROOT"/{stage1,stage2,sft}/*.{bin,idx}
```

SFT 会额外生成 `sft_text_document.mask.bin` 和 `.mask.idx`，mask 为 1 的位置才计算 assistant loss。RL 的 `.bin/.idx` 只是 prompt，当前项目还没有 rollout、reward、advantage 和 policy update，因此不能把 RL 编码结果当作已经完成 RL 训练。

## 6. 训练 Stage 1、Stage 2 和 SFT

Stage 1 从随机初始化开始：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run-stage.sh stage1 2>&1 | tee "$MERIDIAN_RUN_ROOT/stage1/train.log"
```

Stage 1 完成后，确认 checkpoint 存在，再从它继续 Stage 2：

```bash
test -s "$MERIDIAN_RUN_ROOT/stage1/checkpoints/stage1-final.pt"
CUDA_VISIBLE_DEVICES=0 ./scripts/run-stage.sh stage2 2>&1 | tee "$MERIDIAN_RUN_ROOT/stage2/train.log"
```

最后从 Stage 2 checkpoint 继续 SFT：

```bash
test -s "$MERIDIAN_RUN_ROOT/stage2/checkpoints/stage2-final.pt"
CUDA_VISIBLE_DEVICES=0 ./scripts/run-stage.sh sft 2>&1 | tee "$MERIDIAN_RUN_ROOT/sft/train.log"
```

默认目标分别是 20M、80M 和 1M assistant tokens，micro batch 为 1、梯度累积为 2、每步 4096 tokens、BF16、AdamW、cosine learning rate。训练中断后可以重新执行同一阶段；Stage 2 和 SFT 的续训来源仍由脚本固定指向前一阶段最终 checkpoint。

## 7. 中文对话测试

SFT 完成后，使用最终 checkpoint 做单轮测试：

```bash
python -m meridian_training.chat \
  --checkpoint "$MERIDIAN_RUN_ROOT/sft/checkpoints/sft-final.pt" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --prompt '你好，请介绍一下你自己。' \
  --max-new-tokens 128
```

再测试简单事实和数学问题：

```bash
python -m meridian_training.chat --checkpoint "$MERIDIAN_RUN_ROOT/sft/checkpoints/sft-final.pt" --tokenizer-root "$TOKENIZER_ROOT" --prompt '1+1 等于多少？'
python -m meridian_training.chat --checkpoint "$MERIDIAN_RUN_ROOT/sft/checkpoints/sft-final.pt" --tokenizer-root "$TOKENIZER_ROOT" --prompt '中国的首都是哪里？'
```

也可以进入交互式输入：

```bash
python -m meridian_training.chat --checkpoint "$MERIDIAN_RUN_ROOT/sft/checkpoints/sft-final.pt" --tokenizer-root "$TOKENIZER_ROOT"
```

这是约 150M 参数、总预训练 100M tokens 的快速实验模型，回答不稳定、重复或答错都属于预期现象。对话测试的目的首先是确认 tokenizer、checkpoint、模型结构和推理流程已经连通。

## 8. 结果位置

```text
/data/meridian-training-runs/
├── stage1/checkpoints/stage1-final.pt
├── stage2/checkpoints/stage2-final.pt
├── sft/checkpoints/sft-final.pt
└── rl/rl_text_document.{bin,idx,metadata.jsonl}
```

每个编码阶段还会写出 `*.manifest.json`，其中记录 tokenizer 哈希、chat template 哈希、样本数和实际打包 token 数，提交实验结果时应一并保留。
