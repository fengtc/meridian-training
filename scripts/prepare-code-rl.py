#!/usr/bin/env python3
"""Convert UltraData-RL Code rows to the ZGCM-style RL JSONL schema."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if count >= args.limit:
                break
            row = json.loads(line)
            truth = row.get("ground_truth") or {}
            if row.get("domain") != "Code" or not truth.get("inputs"):
                continue
            asset = json.dumps(truth, ensure_ascii=False, sort_keys=True).encode()
            out = {
                "sample_id": row.get("uuid") or f"code-{count:06d}",
                "domain": "code",
                "task_type": "code_stdio" if truth.get("call_type") == "std" else "code",
                "messages": [{"role": "user", "content": row.get("query", "")}],
                "answers": truth,
                "code_asset_hash": hashlib.sha256(asset).hexdigest(),
                "source": row.get("source", "UltraData-RL-2609"),
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"output": str(args.output), "rows": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
