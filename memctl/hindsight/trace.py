"""Evidence sources that need no benchmark labels.

When an environment gives no ground-truth dependencies, these recover an
approximation from the episode's own logs:

- `string_dependencies`: a query's evidence is every earlier item whose text
  contains the correct answer. Finds the item that states the answer; misses
  intermediate hops that do not mention it.
- `tool_value_dependencies`: an agent action's evidence is every earlier item
  that contains a specific value (an identifier, a number) the action used.

Both return the same `Dependency` objects the benchmarks provide, so everything
downstream (oracle, regret, attribution) works unchanged. They over-include:
any item that happens to contain the string counts as an alternative.
"""

from __future__ import annotations

import re

from memctl.task import Dependency, EvidenceRequirement

_VALUE = re.compile(r"[\w-]+")


def _is_specific(token: str) -> bool:
    """Looks like a value rather than a word: has a digit, or is upper case."""
    return any(ch.isdigit() for ch in token) or (token.isupper() and len(token) > 1)


def _originals(items: list[dict]) -> list[dict]:
    """Items that came from the environment (not rewrites, not the agent's own actions)."""
    return [item for item in items if not item["derived_from_ids"] and item["source_type"] != "action"]


def string_dependencies(items: list[dict], queries: list[tuple[str, int, str]]) -> list[Dependency]:
    """`items` are rows of items.jsonl; `queries` are (query id, step, correct answer)."""
    originals = _originals(items)
    dependencies = []
    for query_id, step, gold in queries:
        needle = str(gold).strip()
        carriers = tuple(
            item["id"] for item in originals
            if item["created_at"] < step and needle and needle.lower() in item["content"].lower()
        )
        requirements = (EvidenceRequirement(carriers, needle),) if carriers else ()
        dependencies.append(Dependency(query_id, step, requirements, gold, "string_trace"))
    return dependencies


def tool_value_dependencies(items: list[dict], actions: list[tuple[int, str, str]]) -> list[Dependency]:
    """`actions` are (step, id of the observation answered, the agent's action text)."""
    originals = _originals(items)
    dependencies = []
    for step, query_id, action in actions:
        question = next((item["content"] for item in items if item["id"] == query_id), "")
        requirements = []
        for value in dict.fromkeys(token for token in _VALUE.findall(action or "") if _is_specific(token)):
            if value in question:
                continue  # the instruction itself supplied it
            carriers = tuple(
                item["id"] for item in originals if item["created_at"] < step and value in item["content"]
            )
            if carriers:
                requirements.append(EvidenceRequirement(carriers, value))
        dependencies.append(Dependency(query_id, step, tuple(requirements), None, "tool_value_trace"))
    return dependencies
