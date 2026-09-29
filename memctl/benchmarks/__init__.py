"""Benchmark loaders. Each one returns a list of `Conversation` objects."""

from __future__ import annotations

import random

from memctl.benchmarks.types import Conversation


def load_benchmark(config: dict, seed: int = 0) -> list[Conversation]:
    """Load the benchmark named in `config["name"]`.

    `limit` keeps the first N conversations. `questions_per_conversation` keeps
    a seeded sample of questions, balanced across question categories.
    """
    name = config["name"]
    if name == "synthetic":
        from memctl.benchmarks.synthetic import load_synthetic

        conversations = load_synthetic(number=config.get("conversations", 2), seed=seed)
    elif name == "locomo":
        from memctl.benchmarks.locomo import load_locomo

        conversations = load_locomo(config.get("path", "data/locomo/locomo10.json"))
    else:
        raise KeyError(f"unknown benchmark '{name}' (known: synthetic, locomo)")
    if config.get("limit"):
        conversations = conversations[: config["limit"]]
    if config.get("questions_per_conversation"):
        for conversation in conversations:
            sample_questions(conversation, config["questions_per_conversation"], seed)
    return conversations


def sample_questions(conversation: Conversation, number: int, seed: int) -> None:
    """Keep `number` questions, taking turns between categories so each is represented."""
    rng = random.Random(f"{seed}-{conversation.id}")
    by_category: dict[str, list] = {}
    for question in conversation.questions():
        by_category.setdefault(question.category, []).append(question)
    for questions in by_category.values():
        rng.shuffle(questions)
    kept = set()
    while len(kept) < number and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(kept) < number:
                kept.add(by_category[category].pop().id)
    conversation.events = [e for e in conversation.events if e.kind != "question" or e.question.id in kept]
