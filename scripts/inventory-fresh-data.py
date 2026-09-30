"""Inventory only the newly mounted /mnt/meridian-data raw corpus."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path('/mnt/meridian-data')
tok_path = ROOT / 'tokenizer/tokenizer.json'
tok = Tokenizer.from_file(str(tok_path))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

files = sorted(ROOT.glob('raw/**/*.parquet'))
records = []
for path in files:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    field = 'text' if 'text' in names else ('content' if 'content' in names else None)
    if field is None:
        continue
    columns = [field]
    for name in ('release_mode', 'source', 'category'):
        if name in names:
            columns.append(name)
    rows = next(pf.iter_batches(batch_size=256, columns=columns)).to_pylist()
    texts = [r[field] for r in rows if isinstance(r[field], str) and r[field].strip()]
    counts = [len(x.ids) + 1 for x in tok.encode_batch(texts, add_special_tokens=False)]
    modes = sorted({r.get('release_mode', 'full_text') for r in rows})
    records.append({
        'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
        'rows': pf.metadata.num_rows, 'text_field': field, 'sample_rows': len(rows),
        'sample_nonempty': len(texts), 'sample_modes': modes,
        'sample_sources': sorted({str(r.get('source', '')) for r in rows}),
        'estimated_tokens_from_first_256_rows': round(sum(counts) / len(rows) * pf.metadata.num_rows),
        'status': 'candidate_requires_full_encoding',
    })

manifest = {
    'status': 'RAW_CANDIDATES_NOT_ENCODED',
    'data_root': str(ROOT),
    'target_train_tokens': 10_000_000_000,
    'tokenizer_sha256': sha(tok_path),
    'tokenizer_vocab_size': tok.get_vocab_size(with_added_tokens=True),
    'source_policy': 'fresh downloads only; no /home/ubuntu/zgcm-data or mini-pretrain inputs',
    'warning': 'Estimates are sampled. Exact token quotas require full encoding, filtering, deduplication and a fixed validation split.',
    'files': records,
}
(ROOT / 'manifests/data-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'files': len(records), 'manifest': str(ROOT / 'manifests/data-manifest.json'),
                  'estimated_tokens': sum(r['estimated_tokens_from_first_256_rows'] for r in records)}, ensure_ascii=False))
