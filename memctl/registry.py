"""Name -> class lookup used by the environment, agent and controller registries.

Entries are "module:Class" strings, so a component's dependencies (torch, an API
client) are imported only when that component is asked for.
"""

from __future__ import annotations

import importlib


def load(registry: dict[str, str], name: str | None, kind: str):
    if name not in registry:
        raise KeyError(f"unknown {kind} '{name}'. Known: {sorted(registry)}")
    module_name, _, attribute = registry[name].partition(":")
    return getattr(importlib.import_module(module_name), attribute)
