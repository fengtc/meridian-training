# Code RL

The Code RL path follows the ZGCM-1 design: a model generates a fenced Python
answer, a test runner executes it against hidden inputs, and the reward is the
fraction of tests passed. `scripts/prepare-code-rl.py` converts the
`UltraData-RL-2609` Code rows to the ZGCM-style JSONL schema.

Prepare a 10,000-row pilot:

```bash
python scripts/prepare-code-rl.py \
  --input /mnt/meridian-data/raw/ultradata-rl-2609/data/Code/Code_part-01-of-12.jsonl \
  --output /mnt/meridian-data/rl/code-train-10k.jsonl \
  --limit 10000
```

The local checker can validate one completion before starting a trainer:

```bash
PYTHONPATH=src python - <<'PY'
import json
from meridian_training.rl import CodeReward

row = json.loads(open('/mnt/meridian-data/rl/code-train-10k.jsonl').readline())
result = CodeReward()("```python\nprint('x')\n```", row['answers'])
print(result)
PY
```

This checker is for a single-host pilot. It applies subprocess timeouts and
resource limits, but it is not a production multi-tenant sandbox. For shared
or untrusted workloads, use the namespace/chroot sandbox implementation from
ZGCM-1 (`rl/rewards/sandbox_*`).

The upstream ZGCM-1 `rl/train/code_grpo.py` provides the full AReaL/vLLM/Ray
GRPO policy-update loop. Meridian does not claim that loop is implemented by
the local pretraining entrypoint. A 2-GPU policy trainer must be added after
the reward pilot is passing; it should use the prepared JSONL and this reward
contract rather than treating RL rows as ordinary SFT data.
