"""Task models by config name. Add a new one with one line in `REGISTRY`."""

from __future__ import annotations

from memctl.agents.base import Agent
from memctl.registry import load

REGISTRY = {
    "scripted_reader": "memctl.agents.scripted:ScriptedReader",
    "scripted_tool_agent": "memctl.agents.scripted:ScriptedToolAgent",
    "llm": "memctl.agents.llm:LLMAgent",
    "null": "memctl.agents.base:NullAgent",
}


def build_agent(config: dict) -> Agent:
    return load(REGISTRY, config.get("name"), "agent")(config)
