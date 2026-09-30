"""Count assistant text tokens in complete fresh SFT JSONL files."""
import json
from collections import defaultdict
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path('/mnt/meridian-data')
tok = Tokenizer.from_file(str(ROOT / 'tokenizer/tokenizer.json'))
counts = defaultdict(lambda: {'files': 0, 'records': 0, 'assistant_tokens': 0})
for path in sorted((ROOT / 'raw/ultradata-sft').rglob('*.jsonl')):
    category = path.parent.name
    item = counts[category]
    item['files'] += 1
    batch = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts = []
            for msg in row.get('messages', []):
                if msg.get('role') == 'assistant':
                    texts.append(str(msg.get('content') or ''))
                    texts.append(str(msg.get('reasoning_content') or ''))
            if texts:
                batch.append(''.join(texts))
            item['records'] += 1
            if len(batch) >= 128:
                item['assistant_tokens'] += sum(len(x.ids) for x in tok.encode_batch(batch, add_special_tokens=False))
                batch.clear()
    if batch:
        item['assistant_tokens'] += sum(len(x.ids) for x in tok.encode_batch(batch, add_special_tokens=False))
total = sum(x['assistant_tokens'] for x in counts.values())
out = {'data_root': str(ROOT), 'tokenizer': str(ROOT / 'tokenizer/tokenizer.json'),
       'complete_files_only': True, 'categories': dict(sorted(counts.items())),
       'assistant_tokens': total, 'target_assistant_tokens': 1_000_000_000,
       'status': 'READY' if total >= 1_000_000_000 else 'INSUFFICIENT'}
(ROOT / 'manifests/sft-assistant-token-count.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(out, ensure_ascii=False))
