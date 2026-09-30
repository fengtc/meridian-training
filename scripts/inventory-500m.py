"""Inventory raw candidates without treating estimates as prepared tokens."""
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path('/home/ubuntu/meridian-data')
OFFICIAL = Path('/home/ubuntu/ZGCM-Training-Lab/data/tokenizer/zgcm-1-official')
dest = ROOT / 'tokenizers/official'
dest.mkdir(parents=True, exist_ok=True)
for name in ['tokenizer.json', 'chat_template.jinja', 'tokenizer_config.json', 'LICENSE', 'README.md']:
    if (OFFICIAL / name).exists():
        shutil.copy2(OFFICIAL / name, dest / name)
tok = Tokenizer.from_file(str(dest / 'tokenizer.json'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

files = sorted(Path('/home/ubuntu/zgcm-data').glob('zgcm-1-pretrain-stage*/*.parquet'))
files += sorted((ROOT / 'source-datasets').rglob('*.parquet'))
records = []
for path in files:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    field = 'text' if 'text' in names else 'content'
    columns = [field] + [x for x in ['release_mode', 'source', 'category'] if x in names]
    # Bounded sampling avoids loading multi-GB string columns into memory.
    rows = next(pf.iter_batches(batch_size=256, columns=columns)).to_pylist()
    texts = [r[field] for r in rows if isinstance(r[field], str) and r[field].strip()]
    counts = [len(x.ids) + 1 for x in tok.encode_batch(texts, add_special_tokens=False)]
    modes = sorted({r.get('release_mode', 'full_text') for r in rows})
    record = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
              'rows': pf.metadata.num_rows, 'text_field': field,
              'sample_rows': len(rows), 'sample_nonempty': len(texts),
              'sample_modes': modes, 'sample_sources': sorted({str(r.get('source', '')) for r in rows}),
              'estimated_tokens_from_first_256_rows': round(sum(counts) / len(rows) * pf.metadata.num_rows),
              'status': 'exclude_index_only' if modes == ['index_only'] else 'candidate_requires_full_encoding'}
    records.append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
manifest = {'status': 'RAW_CANDIDATES_NOT_ENCODED', 'target_train_tokens': 10_000_000_000,
            'tokenizer_sha256': sha(dest / 'tokenizer.json'),
            'chat_template_sha256': sha(dest / 'chat_template.jinja'),
            'tokenizer_vocab_size': tok.get_vocab_size(with_added_tokens=True),
            'warning': 'Prefix samples are biased estimates, not an exact token count. Exclude empty/index-only records, deduplicate and split documents before packing.',
            'files': records}
(ROOT / 'data-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('MANIFEST', ROOT / 'data-manifest.json', flush=True)
