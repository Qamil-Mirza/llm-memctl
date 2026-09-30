"""What can run on this machine, right now:  python -m memctl.doctor

Checks the things a component can be blocked by (hardware, credentials, data
files, optional packages, a solver, a model endpoint) and says, for each
component that depends on them, whether it is READY, LIMITED or BLOCKED here
and what would unblock it. This is about the environment, not about whether the
code is correct; tests answer that. `INFRA_REPORT.md` combines both.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

from memctl.sysinfo import cpu_info, git_state, gpu_info

READY, LIMITED, BLOCKED = "READY", "LIMITED", "BLOCKED"
# Measured on this project (docs/restart.md): Qwen3.5-4B needs 8.41 GB at 16-bit and 3.41 GB in
# 4-bit, and 4-bit on a 4 GB card runs out of memory above roughly 3,900 prompt tokens.
GPU_MB_FOR_4B_16BIT = 12_000
GPU_MB_FOR_4B_4BIT = 3_800


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def solver_works() -> bool:
    if not has("pulp"):
        return False
    import pulp

    program = pulp.LpProblem("check", pulp.LpMaximize)
    x = pulp.LpVariable("x", cat="Binary")
    program += x
    program.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[program.status] == "Optimal"


def endpoint_models(base_url: str) -> list[str] | None:
    """Model names served at an OpenAI-compatible endpoint, or None if it does not answer."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as response:
            return [model["id"] for model in json.load(response).get("data", [])]
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return None


def check(base_url: str | None = None) -> dict:
    base_url = base_url or os.environ.get("MEMCTL_LLM_BASE_URL", "http://localhost:11434/v1")
    gpus = gpu_info()
    gpu_mb = max((gpu["memory_total_mb"] for gpu in gpus), default=0)
    cuda = False
    if has("torch"):
        import torch

        cuda = bool(torch.cuda.is_available())
    served = endpoint_models(base_url)
    facts = {
        "git": git_state(),
        "cpu": cpu_info(),
        "gpu": gpus,
        "torch": has("torch"),
        "torch_cuda": cuda,
        "transformers": has("transformers"),
        "sentence_transformers": has("sentence_transformers"),
        "bitsandbytes": has("bitsandbytes"),
        "pulp_cbc_solver": solver_works(),
        "jev_api_key": bool(os.environ.get("JEV_API_KEY")),
        "locomo_file": Path("data/locomo/locomo10.json").exists(),
        "longmemeval_oracle_file": Path("data/longmemeval/longmemeval_oracle.json").exists(),
        "longmemeval_s_file": Path("data/longmemeval/longmemeval_s_cleaned.json").exists(),
        "llm_endpoint": base_url,
        "llm_endpoint_models": served,
    }

    components = []

    def add(name: str, status: str, detail: str, fix: str = "") -> None:
        components.append({"component": name, "status": status, "detail": detail, "to_unblock": fix})

    add("synthetic and workflow environments, heuristic controllers", READY, "pure Python; no GPU or network needed")
    if facts["pulp_cbc_solver"]:
        add("exact oracle (integer program)", READY, "PuLP with the CBC solver works")
    else:
        add("exact oracle (integer program)", BLOCKED, "PuLP or its CBC solver is missing; the oracle falls back to approx",
            "pip install -c constraints.txt pulp")
    if facts["torch"]:
        add("RL controller (training and inference)", READY, "PyTorch is installed; the policy is small and runs on CPU")
    else:
        add("RL controller (training and inference)", BLOCKED, "PyTorch is not installed", "pip install torch (CPU build is enough)")
    if facts["jev_api_key"]:
        add("JEV controller (real model)", READY, "JEV_API_KEY is set; the HTTP client has not been verified against the live API")
    else:
        add("JEV controller (real model)", BLOCKED, "no JEV_API_KEY in the environment; only the labelled FAKE client can run",
            "put JEV_API_KEY in .env")

    if not (facts["torch"] and facts["transformers"]):
        add("local Hugging Face task model", BLOCKED, "torch and transformers are needed",
            "pip install -e '.[models]' or use the Docker image")
    elif not cuda:
        add("local Hugging Face task model", LIMITED, "no CUDA GPU visible to PyTorch: CPU only, very slow",
            "install a CUDA build of torch or use the Docker image with the NVIDIA runtime")
    elif gpu_mb < GPU_MB_FOR_4B_4BIT:
        add("local Hugging Face task model", BLOCKED, f"GPU has {gpu_mb} MB", "a GPU with at least 12 GB")
    elif gpu_mb < GPU_MB_FOR_4B_16BIT:
        add("local Hugging Face task model", LIMITED,
            f"GPU has {gpu_mb} MB: a 4B model fits only in 4-bit and only for prompts up to about 3,900 tokens",
            "a GPU with at least 12 GB (24 GB runs every budget)")
    else:
        add("local Hugging Face task model", READY, f"GPU has {gpu_mb} MB")

    if served is None:
        add("task model / prompted controller via an OpenAI-compatible endpoint", BLOCKED,
            f"nothing answers at {base_url}", "start vLLM or Ollama, or set MEMCTL_LLM_BASE_URL")
    elif not served:
        add("task model / prompted controller via an OpenAI-compatible endpoint", LIMITED,
            f"{base_url} answers but serves no model", "pull a model into the server")
    else:
        add("task model / prompted controller via an OpenAI-compatible endpoint", READY,
            f"{base_url} serves {', '.join(served)}")

    add("dense embeddings (sentence-transformers)", READY if facts["sentence_transformers"] else LIMITED,
        "installed" if facts["sentence_transformers"] else "not installed: only the hashing (word-overlap) embedder is available",
        "" if facts["sentence_transformers"] else "pip install sentence-transformers")
    add("LoCoMo", READY if facts["locomo_file"] else BLOCKED,
        "data/locomo/locomo10.json is present" if facts["locomo_file"] else "data file missing",
        "" if facts["locomo_file"] else "set env.download: true (2.8 MB)")
    if facts["longmemeval_s_file"]:
        add("LongMemEval", READY, "oracle and S splits are present" if facts["longmemeval_oracle_file"] else "S split is present")
    elif facts["longmemeval_oracle_file"]:
        add("LongMemEval", LIMITED, "only the oracle split (evidence sessions only) is present",
            "download longmemeval_s_cleaned.json (277 MB) from the Hugging Face dataset xiaowu0162/longmemeval-cleaned")
    else:
        add("LongMemEval", BLOCKED, "no data file", "download a split into data/longmemeval/")
    add("LLM-as-judge scoring", READY if served else BLOCKED,
        "a served model can act as judge" if served else "no judge model is reachable; QA answers are scored by local token F1",
        "" if served else "serve a model and set env.judge")
    return {"facts": facts, "components": components}


def render(report: dict) -> str:
    facts = report["facts"]
    gpus = ", ".join(f"{gpu['name']} ({gpu['memory_total_mb']} MB)" for gpu in facts["gpu"]) or "none"
    lines = [
        f"code: {facts['git']['short']}{' (uncommitted changes)' if facts['git']['dirty'] else ''} on {facts['git']['branch']}",
        f"cpu:  {facts['cpu']['model']} · {facts['cpu']['logical_cores']} cores · {facts['cpu']['memory_gb']} GB",
        f"gpu:  {gpus}",
        "",
    ]
    width = max(len(row["component"]) for row in report["components"])
    for row in report["components"]:
        lines.append(f"{row['status']:<8} {row['component']:<{width}}  {row['detail']}")
        if row["to_unblock"]:
            lines.append(f"{'':<8} {'':<{width}}  -> {row['to_unblock']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report what can run on this machine.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint to probe")
    args = parser.parse_args()
    report = check(args.base_url)
    print(json.dumps(report, indent=2) if args.json else render(report))


if __name__ == "__main__":
    main()
