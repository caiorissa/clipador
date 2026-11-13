"""Tiny GPU memory probe (nvidia-smi) used to log/prove VRAM release between Whisper and LLM."""

from __future__ import annotations

import shutil
import subprocess


def vram_used_mb() -> int | None:
    """Total used VRAM on GPU 0 in MiB (Windows WDDM does not report per-process usage)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout  # fmt: skip
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
