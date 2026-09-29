"""A tiny made-up benchmark for tests and smoke runs. Not used for thesis results."""

from __future__ import annotations

import random

from memctl.benchmarks.types import Conversation, Event, Question
from memctl.items import make_item

PEOPLE = ["Alice", "Bruno", "Chen", "Dalia"]
FACTS = [  # (attribute, statement template, possible values)
    ("favourite colour", "My favourite colour is {}.", ["teal", "crimson", "amber", "violet"]),
    ("pet", "My new pet is a {}, adopted last month.", ["parrot", "rabbit", "gecko", "beagle"]),
    ("hometown", "My hometown is {}, I grew up there.", ["Lisbon", "Nairobi", "Osaka", "Quito"]),
    ("job", "My job changed: I now work as a {}.", ["welder", "botanist", "pilot", "cellist"]),
    ("hobby", "My weekend hobby these days is {}.", ["kayaking", "pottery", "fencing", "birding"]),
    ("allergy", "My doctor found an allergy to {}.", ["peanuts", "pollen", "shellfish", "latex"]),
]
FILLER = [
    "How was your week? Mine went by quickly.",
    "The weather has been strange lately, hasn't it?",
    "I should really get more sleep these days.",
    "Did you watch anything good recently?",
    "Traffic was terrible this morning, as usual.",
    "Let's catch up again soon, this was nice.",
    "I had the longest meeting of my life yesterday and nothing was decided in the end.",
]


def _conversation(number: int, rng: random.Random, sessions: int = 3, turns: int = 12) -> Conversation:
    pair = rng.sample(PEOPLE, 2)
    facts = [(speaker, *fact) for speaker in pair for fact in rng.sample(FACTS, 4)]
    rng.shuffle(facts)
    events: list[Event] = []
    known: list[tuple[str, str, str, str]] = []  # (item id, speaker, attribute, value)
    asked = 0

    def ask(evidence: list[tuple[str, str, str, str]]) -> None:
        nonlocal asked
        asked += 1
        if len(evidence) == 1:
            _, speaker, attribute, value = evidence[0]
            text, gold, category = f"What is {speaker}'s {attribute}?", value, "single-hop"
        else:
            (_, s1, a1, v1), (_, s2, a2, v2) = evidence
            text = f"What are {s1}'s {a1} and {s2}'s {a2}?"
            gold, category = f"{v1} and {v2}", "multi-hop"
        ids = tuple(e[0] for e in evidence)
        events.append(Event("question", question=Question(f"c{number}_q{asked}", text, gold, category, ids)))

    for session in range(1, sessions + 1):
        for turn in range(turns):
            speaker = pair[turn % 2]
            item_id = f"c{number}_s{session}_t{turn}"
            mine = [f for f in facts if f[0] == speaker]
            if turn % 3 == 1 and mine:
                fact = mine[0]
                facts.remove(fact)
                _, attribute, template, values = fact
                value = rng.choice(values)
                text = template.format(value)
                known.append((item_id, speaker, attribute, value))
            else:
                text = rng.choice(FILLER)
            source = {"speaker": speaker, "session": session, "date": f"2026-01-{session:02d}"}
            events.append(Event("item", item=make_item(item_id, text, len(events) + 1, source=source)))
        events.append(Event("session_end"))
        if session < sessions and known:
            ask([rng.choice(known)])  # a question in the middle of the conversation
    for fact in known:
        ask([fact])
    for _ in range(2):
        ask(rng.sample(known, 2))
    return Conversation(f"synthetic_{number}", events)


def load_synthetic(number: int = 2, seed: int = 0) -> list[Conversation]:
    rng = random.Random(seed)
    return [_conversation(n, rng) for n in range(number)]
