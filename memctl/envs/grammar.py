"""The sentence forms of the synthetic environments, shared with their scripted agents.

Every fact carries the step at which it was stated, so that a reader holding two
statements about the same thing can tell which is current.
"""

from __future__ import annotations

import re

FACT = re.compile(r"As of step (\d+), the ([a-z ]+?) of ([\w-]+) is ([\w-]+)\.")
QUERY = re.compile(r"Question: what is the ([a-z ]+?) of (?:the ([a-z ]+?) of )?([\w-]+)\?")
UNKNOWN = "unknown"


def fact_sentence(step: int, attribute: str, entity: str, value: str) -> str:
    return f"As of step {step}, the {attribute} of {entity} is {value}."


def query_sentence(attribute: str, entity: str, via: str | None = None) -> str:
    """`via` makes a two-hop query: the `attribute` of the `via` of `entity`."""
    if via:
        return f"Question: what is the {attribute} of the {via} of {entity}?"
    return f"Question: what is the {attribute} of {entity}?"


def parse_query(text: str) -> tuple[str, str | None, str] | None:
    """(attribute, via, entity), or None if the text is not a query."""
    match = QUERY.search(text)
    return (match.group(1), match.group(2), match.group(3)) if match else None


def find_facts(text: str) -> list[tuple[int, str, str, str]]:
    """Every (step, attribute, entity, value) stated in the text."""
    return [(int(step), attribute, entity, value) for step, attribute, entity, value in FACT.findall(text)]
