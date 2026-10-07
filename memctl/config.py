"""Experiment configuration: defaults, validation and a stable hash.

A config is one experiment: one environment, one agent, one controller, one
budget, a number of episodes. `resolve` fills in every default, so the
`config.yaml` saved with a run is complete and does not depend on the code's
defaults at the time it is read back.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from memctl.memory.actions import Operation
from memctl.util import merged

SCHEMA_VERSION = 1

DEFAULTS: dict = {
    "schema_version": SCHEMA_VERSION,
    "name": "experiment",
    "seed": 0,  # episode i uses seed + i, so every controller sees the same episodes
    "episodes": 10,
    "env": {"name": "synthetic_recall"},
    "agent": {"name": "scripted_reader"},
    "controller": {"name": "fifo"},
    "memory": {
        "budget": {"fraction": 0.25},  # or {"tokens": 800}
        "archive_budget": None,  # tokens; None is unlimited
        "allowed_operations": ["KEEP", "EVICT", "NO_OP"],
        "embedder": None,  # None, "hashing", or a sentence-transformers model name
        "compressor": "extractive",  # extractive | truncate | llm
        "consolidator": "dedup",  # dedup | llm
        "compression_model": None,  # a model config, when compressor or consolidator is "llm"
        "compact_ratio": 0.5,
        "store_agent_actions": False,
        # Count each item's "speaker, date: " label (as the reader shows it) in its tokens, so the budget is
        # the prompt the reader pays for (Experiment 13: labels were about 20 tokens a line, uncounted).
        "count_labels": False,
    },
    "reward": {"weights": {"task_reward": 1.0, "forced_fallback": -0.1, "invalid_action": -0.1}},
    "hindsight": {"enabled": True},
    "shadow_controllers": [],
    "pricing": {"task_model_usd_per_mtok": 0.0, "controller_usd_per_mtok": 0.0},
    "logging": {"output_dir": "runs", "detail_episodes": 3},
    "compute_budget": {"max_steps_per_episode": None},
}


class ConfigError(ValueError):
    pass


def resolve(config: dict) -> dict:
    """The config with every default filled in, checked."""
    resolved = merged(copy.deepcopy(DEFAULTS), copy.deepcopy(config))
    if resolved["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"config schema_version {resolved['schema_version']} is not {SCHEMA_VERSION}")
    if isinstance(resolved["controller"], str):
        resolved["controller"] = {"name": resolved["controller"]}
    given = (config.get("memory") or {}).get("budget") or {}
    if "tokens" in given:  # an absolute budget replaces the default fraction
        resolved["memory"]["budget"] = {"tokens": int(given["tokens"])}
    else:
        fraction = float(resolved["memory"]["budget"]["fraction"])
        if not 0 < fraction <= 1:
            raise ConfigError("memory.budget.fraction must be in (0, 1]")
        resolved["memory"]["budget"] = {"fraction": fraction}
    try:
        operations = [Operation(op).value for op in resolved["memory"]["allowed_operations"]]
    except ValueError as error:
        raise ConfigError(f"memory.allowed_operations: {error}") from error
    resolved["memory"]["allowed_operations"] = operations
    if int(resolved["episodes"]) < 1:
        raise ConfigError("episodes must be at least 1")
    return resolved


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text()
    return resolve(yaml.safe_load(text) or {})


def config_hash(config: dict) -> str:
    """A short hash of everything that determines the results (not where they are written)."""
    relevant = {key: value for key, value in config.items() if key != "logging"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:10]


def budget_label(config: dict) -> str:
    budget = config["memory"]["budget"]
    return f"B{budget['tokens']}tok" if "tokens" in budget else f"B{round(float(budget['fraction']) * 100):03d}"
