"""Memory controllers by config name. Add a new one with one line in `REGISTRY`."""

from __future__ import annotations

from memctl.controllers.base import MemoryController
from memctl.registry import load

REGISTRY = {
    "no_controller": "memctl.controllers.heuristics:NoController",
    "full_context": "memctl.controllers.heuristics:FullContext",
    "fifo": "memctl.controllers.heuristics:Fifo",
    "lru": "memctl.controllers.heuristics:Lru",
    "lfu": "memctl.controllers.heuristics:Lfu",
    "random": "memctl.controllers.heuristics:RandomEviction",
    "age_decay": "memctl.controllers.heuristics:AgeDecay",
    "similarity": "memctl.controllers.heuristics:Similarity",
    "salience": "memctl.controllers.heuristics:Salience",
    "archive_everything": "memctl.controllers.heuristics:ArchiveEverything",
    "oracle": "memctl.controllers.oracle:OracleController",
    "prompted_llm": "memctl.controllers.prompted:PromptedLLMController",
    "jev": "memctl.controllers.jev:JEVController",
    "rl": "memctl.controllers.rl:RLController",
}


def build_controller(config: dict, seed: int = 0) -> MemoryController:
    return load(REGISTRY, config.get("name"), "controller")(config, seed)
