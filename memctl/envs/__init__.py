"""Task environments by config name. Add a new one with one line in `REGISTRY`."""

from __future__ import annotations

from memctl.envs.base import TaskEnvironment
from memctl.registry import load

REGISTRY = {
    "synthetic_recall": "memctl.envs.synthetic:SyntheticRecallEnv",
    "workflow": "memctl.envs.workflow:WorkflowEnv",
    "locomo": "memctl.envs.locomo:LoCoMoEnv",
    "longmemeval": "memctl.envs.longmemeval:LongMemEvalEnv",
}


def build_env(config: dict) -> TaskEnvironment:
    return load(REGISTRY, config.get("name"), "environment")(config)
