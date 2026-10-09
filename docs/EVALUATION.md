# Meridian Evaluation

## Next-token evaluation

Compare Stage 1 and Stage 2 on fixed probes and sampled IndexedDataset sequences:

```bash
python scripts/eval-next-token.py \
  --stage1 /path/to/stage1/checkpoints/step-latest-rank0.pt \
  --stage2 /path/to/stage2/checkpoints/step-latest-rank0.pt \
  --stage1-data /path/to/stage1_text_document \
  --stage2-data /path/to/stage2_text_document \
  --config configs/model-500m.yaml \
  --tokenizer-root /path/to/tokenizer \
  --output eval-next-token.json
```

The report includes per-category negative log likelihood, perplexity, and top-k
next-token predictions. The IndexedDataset scores are diagnostic samples, not a
held-out benchmark.

For interactive next-token inspection:

```bash
python scripts/next-token-chat.py \
  --checkpoint /path/to/checkpoint-rank0.pt \
  --config configs/model-500m.yaml \
  --tokenizer-root /path/to/tokenizer \
  --top-k 10
```

## SFT tokenizer requirements

SFT encoding and inference use the tokenizer's `chat_template.jinja` through
`apply_chat_template` when a local Transformers tokenizer configuration is
available. Keep these files together when moving the tokenizer:

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
```

The encoder derives the assistant-only loss mask from the rendered official
template and appends the EOS token to each assistant response.
