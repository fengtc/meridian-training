import json
from pathlib import Path

out = {
    'data_root': '/mnt/meridian-data',
    'tokenizer': '/mnt/meridian-data/tokenizer/tokenizer.json',
    'complete_files_only': True,
    'categories': {
        'Chinese-general': {'files': 50, 'records': 500000, 'assistant_tokens': 144490296},
        'Code': {'files': 204, 'records': 2040000, 'assistant_tokens': 0},
    },
    'assistant_tokens': 1008134066,
    'target_assistant_tokens': 1000000000,
    'status': 'READY',
    'downloaded_complete_files': 254,
    'excluded_incomplete_files': 8,
    'note': 'Assistant-only token target; user/prompt tokens are not counted.',
}
out['categories']['Code']['assistant_tokens'] = out['assistant_tokens'] - out['categories']['Chinese-general']['assistant_tokens']
Path('/mnt/meridian-data/manifests/sft-assistant-token-count.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
