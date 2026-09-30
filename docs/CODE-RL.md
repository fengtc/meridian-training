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

The repository also includes a bounded single-host GRPO pilot. It samples a
group of completions per prompt, scores them with `CodeReward`, normalizes the
group rewards, and applies a policy-gradient update:

```bash
export CODE_RL_CHECKPOINT=/mnt/meridian-data/checkpoints/stage2/checkpoints
export CODE_RL_INPUT=/mnt/meridian-data/rl/code-train-10k.jsonl
./scripts/run-code-grpo.sh --gpu 2 --steps 100 --group-size 4 --max-new-tokens 256
```

Rollout runs in the same processes as training and uses the local subprocess
checker, so this is a 2-GPU pilot rather than a production multi-tenant sandbox.
