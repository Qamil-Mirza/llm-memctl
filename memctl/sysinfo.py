"""What a run was made with: code version, machine, packages. Saved as metadata.json."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone


def _run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict:
    commit = _run("git", "rev-parse", "HEAD")
    return {
        "commit": commit or None,
        "short": commit[:7] if commit else "nogit",
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def cpu_info() -> dict:
    model = ""
    try:
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = platform.processor()
    memory_gb = None
    try:
        memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {"model": model, "logical_cores": os.cpu_count(), "memory_gb": memory_gb}


def gpu_info() -> list[dict]:
    output = _run("nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits")
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {"name": parts[0], "memory_total_mb": int(parts[1]), "memory_free_mb": int(parts[2]), "driver": parts[3]}
            )
    return gpus


def packages() -> dict[str, str]:
    return dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))


def collect_metadata() -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_info(),
        "gpu": gpu_info(),
        "packages": packages(),
    }
