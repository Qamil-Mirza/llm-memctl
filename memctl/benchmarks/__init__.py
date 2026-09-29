"""Benchmark loaders. Each one returns a list of `Conversation` objects."""

from __future__ import annotations

from memctl.benchmarks.types import Conversation


def load_benchmark(config: dict, seed: int = 0) -> list[Conversation]:
    """Load the benchmark named in `config["name"]`, keeping the first `limit` conversations."""
    name = config["name"]
    limit = config.get("limit")
    if name == "synthetic":
        from memctl.benchmarks.synthetic import load_synthetic

        conversations = load_synthetic(number=config.get("conversations", 2), seed=seed)
    else:
        raise KeyError(f"unknown benchmark '{name}' (available now: synthetic)")
    return conversations[:limit] if limit else conversations
