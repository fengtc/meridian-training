# 500M / 10B Token 双 RTX 5090 训练方案

## 目标与边界

- 从随机权重训练约 500M 参数的 decoder-only 模型，目标预训练量 10,000,000,000 tokens。
- 机器为单机 2 x RTX 5090 32GB，采用 BF16、CUDA/NCCL、数据并行 DP=2；TP/PP/CP 保持 1。
- 参考 Meridian 的官方 tokenizer、Megatron IndexedDataset、AdamW 和 cosine 学习率流程。
- 先完成数据下载、tokenizer 校验、编码和 20M/100M smoke test，再启动 10B 正式训练。

## 模型配置

`configs/model-500m.yaml`：24 层、hidden size 1024、16 attention heads、4 KV groups、SwiGLU FFN 3840、RMSNorm、RoPE、2048 context，输入输出 embedding tied。按 padded vocab 155,136、无 linear bias 计算约 504.94M 参数；tokenizer 实际词表与 padding 必须分开记录，最终以模型构造后的参数计数为准。

## 数据配比

保持 ZGCM/Meridian 的训练阶段命名，只使用 Stage 1、Stage 2 和 SFT。10B token 只统计预训练阶段；SFT 使用独立的对话数据，不计入 10B。

| 阶段 | token 目标 | 建议内容 | 作用 |
|---|---:|---|---|
| Stage 1 | 2.0B | 高质量 Web/中文/英文混合 | 初始预训练与稳定性建立 |
| Stage 2 | 8.0B | 固定配比的 Web、知识、数学、代码混合数据 | 从 Stage 1 checkpoint 连续学习 |
| SFT | 1.0B assistant tokens（编码后约 1.637B packed tokens） | UltraData-SFT-2605，对话/指令数据，assistant-only mask | 预训练结束后的监督微调 |

建议总配比：中文 Web 35%（3.5B）、英文 Web 30%（3B）、知识/PDF 10%（1B）、数学 15%（1.5B）、代码 10%（1B）。这是本次实验的建议，不是上游已验证的最优配方；各类按统一 tokenizer 编码后的 token 配额执行，不能按文件数分配。中文来自 Ultra-FineWeb，其他类别优先复用本地 full_text 分片。现有分片覆盖不同来源，先做归类和去重再采样；不足类别需继续补齐分片，不用重复样本冒充独立语料。

先规范化正文并去重，用正文哈希和固定 seed 分配约 0.5% 文档到独立验证集，再分别打包；训练集必须另外满足 10B，验证 tokens 不计入 10B。排除 index_only、空正文和不可解析记录。manifest 记录真实 token 数、分片、tokenizer SHA256、去重/过滤计数和 train/validation 边界。打包必须支持 next-token shift，避免跨样本标签错位。

## 双卡训练账本

序列长度 2048、每卡 micro-batch 1、DP=2、gradient accumulation 8 时，每 optimizer step 消耗 32,768 tokens；10B 需要约 305,176 steps。建议有效 batch 先用 accumulation 4（16,384 tokens/step）做 smoke test，正式训练使用 accumulation 8；显存不足时保持 micro-batch 1，只增加 accumulation。

优化器使用 AdamW（weight decay 0.1、grad clip 1.0），peak LR 2e-4，warmup 2%，cosine 到 2e-5。每 10k steps 保存 checkpoint，并在 500M、2B、5B、8.5B、10B token 处保存里程碑；保留最近 3 个和全部里程碑。

Stage 1 到 Stage 2 使用 checkpoint 续训，但不重置模型权重。Stage 2 是一次连续的 8B-token 训练，不拆成子阶段、不要求中途切换数据或学习率。LR warmup 仅指全程最初 2% steps 的学习率爬升，不是独立训练阶段；不要把它命名为 warmup 数据阶段。正式总量按 2B + 8B = 10B 计算；ceil 到完整 batch 后的多余 token 必须在数据准备或训练账本中明确处理。

本次实际实验使用双 RTX 5090 完成 Stage 1、Stage 2 和 SFT。SFT 首次编码使用了简化 role 模板，后续已改为官方 `chat_template.jinja`，补齐 `tokenizer_config.json`/`special_tokens_map.json`，并按完整模板生成 assistant-only mask；正式结果应以官方模板重编码后的 SFT checkpoint 为准。

耗时示例仅用于预算：实测总吞吐 5k/10k/20k tokens/s 时，纯训练分别约 23.1/11.6/5.8 天，额外留 15% 给评估和 checkpoint。两卡显存不能简单视为 64GB 单卡；DDP 每卡保存完整参数、梯度和优化器状态，长词表 logits/activation 是显存重点。建议 fused/chunked CE、activation checkpointing，并实测 SDPA/Flash 与局部 attention 的兼容性。

新增 10B uint32 token 数据约 40GB；原始语料、临时编码文件、验证集和 checkpoint 另计。预留至少 60GB 给 checkpoint/临时产物，下载前检查剩余空间，低于预算时停止扩充原始分片。

## 上游入口改造清单

训练入口已改为读取 YAML，初始化 NCCL DDP，按 rank 分片读取固定长度序列，使用 `no_sync` 做梯度累积，并保存每 rank 的模型、优化器、scheduler、RNG 和数据 cursor。Stage 2/SFT 从前一阶段 checkpoint 加载模型权重；同一阶段重复执行会恢复完整训练状态。正式 2B/8B 训练前必须先完成 20M 双卡 pilot。

## 执行顺序

1. 按设备选择环境：DGX Spark GB10 使用 `configs/devices/dgx-spark-gb10.env`（ARM64、厂商 PyTorch、单卡）；双 RTX 5090 使用 `configs/devices/rtx5090-2.env`（x86_64、CUDA/NCCL、2 卡），两者不共用 Python wheel。
2. 复制官方 tokenizer，核对 vocab、EOS、SHA256；禁止使用 MiniCPM5 或其他 tokenizer。
3. 下载公开数据的目标分片，和已有 `zgcm-data` 预训练 Parquet 合并，生成 `data-manifest.json`。
4. 先编码 20M tokens，检查样本边界、无空文档、无异常超长样本；然后编码 100M tokens 并做双卡短跑。
5. 先做 500M-token pilot，检查 loss 曲线、tokens/s、显存、NCCL、checkpoint 可恢复性；pilot 不计入正式 10B。
6. 重新从随机权重启动正式 Stage 1，训练 2B token；然后从 Stage 1 checkpoint 一次性继续执行 Stage 2 的 8B token，周期性评估 perplexity、中文/英文、代码和数学 held-out 集。
7. 预训练结束后再执行 SFT；SFT 数据和 token 计数独立保存，不混入 10B 预训练账本。

## 当前数据准备状态

数据根目录固定为新挂载盘 `/mnt/meridian-data`，不读取 `/home/ubuntu/zgcm-data`、`mini-pretrain` 或旧 IndexedDataset。新盘当前包含 14 个 Ultra-FineWeb 中文分片、2 个 Ultra-FineWeb-L3 分片、4 个 UltraData-Code Python 分片和 2 个 UltraData-Math 分片；抽样估算约 10.51B token。该估算不是最终训练账本，仍需完整 tokenizer 编码、去重、验证集切分后按 2B/8B 配额裁剪。未完成的 `.incomplete` 文件必须排除。

新盘布局：

```text
/mnt/meridian-data/raw/             # 全新下载的原始 Parquet
/mnt/meridian-data/tokenizer/       # 官方 tokenizer 运行文件
/mnt/meridian-data/manifests/       # SHA256、来源、token 统计
/mnt/meridian-data/indexed/         # 后续生成的 Megatron .bin/.idx
/mnt/meridian-data/checkpoints/     # Stage 1/Stage 2/SFT checkpoint
```
