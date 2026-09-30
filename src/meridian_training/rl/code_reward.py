"""Evaluate Python code answers against stdin/stdout test cases.

This is a development pilot checker. It uses a subprocess, timeout and POSIX
resource limits. Production multi-tenant training should use the stronger
namespace/chroot sandbox from ZGCM-1.
"""
from __future__ import annotations

import os
import re
import resource
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    passed: int
    total: int
    valid_code: bool
    truncated: bool = False
    error: str | None = None


def extract_python(text: str) -> str | None:
    blocks = _FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip() or None
    text = (text or "").strip()
    return text or None


def _limits(cpu_seconds: int, memory_bytes: int):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


class CodeReward:
    def __init__(self, timeout_seconds: float = 8.0, memory_mb: int = 512):
        self.timeout_seconds = float(timeout_seconds)
        self.memory_bytes = int(memory_mb) * 1024 * 1024

    def __call__(self, completion: str, ground_truth: dict[str, Any]) -> RewardResult:
        code = extract_python(completion)
        if not code:
            return RewardResult(0.0, 0, len(ground_truth.get("inputs", [])), False, error="empty_code")
        inputs = ground_truth.get("inputs") or []
        outputs = ground_truth.get("outputs") or []
        if len(inputs) != len(outputs) or not inputs:
            return RewardResult(0.0, 0, len(inputs), False, error="invalid_test_asset")
        with tempfile.TemporaryDirectory(prefix="meridian-code-test-") as tmp:
            script = Path(tmp) / "solution.py"
            script.write_text(code, encoding="utf-8")
            passed = 0
            for stdin, expected in zip(inputs, outputs):
                try:
                    result = subprocess.run(
                        ["python", str(script)], input=str(stdin), text=True,
                        capture_output=True, timeout=self.timeout_seconds,
                        cwd=tmp, env={"PATH": os.environ.get("PATH", "")},
                        preexec_fn=lambda: _limits(10, self.memory_bytes),
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode == 0 and result.stdout.strip() == str(expected).strip():
                    passed += 1
            return RewardResult(passed / len(inputs), passed, len(inputs), True)
