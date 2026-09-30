"""Phase 0: a controlled long-horizon recall task with exact ground truth.

A stream of facts, distractors and queries. Some facts are asked about much
later, some are asked about again, some are overwritten, some are restated, and
most of the stream is never needed. The simulator knows exactly which items
each query needs, so future usefulness is known and an oracle can be built.

What makes policies differ, and the knob that controls each:
- recency helps when gaps are short                     (`gap`)
- reuse helps a policy that tracks access               (`requery_prob`)
- source type predicts being asked about                (`sources.*.query_prob`)
- overwritten facts are dead weight                     (`update_prob`)
- restated facts are redundant and can be consolidated  (`restate_prob`)
- facts buried in long items reward compaction          (`verbose_prob`)
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from memctl.envs.base import TaskEnvironment
from memctl.envs.grammar import UNKNOWN, fact_sentence, query_sentence
from memctl.memory.items import SourceType
from memctl.task import Dependency, EvidenceRequirement, Observation, StepResult
from memctl.util import merged

_ANSWER_TOKEN = re.compile(r"[\w-]+")

DEFAULTS = {
    "horizon": 300,
    "fact_prob": 0.35,  # share of free steps that state a new fact; the rest are distractors
    "sources": {
        "user": {"share": 0.3, "query_prob": 0.7},
        "tool_output": {"share": 0.4, "query_prob": 0.15},
        "observation": {"share": 0.3, "query_prob": 0.35},
    },
    "gap": {"dist": "loguniform", "min": 10, "max": None},  # steps from a fact to its query; max None = horizon
    "requery_prob": 0.35,
    "update_prob": 0.15,
    "restate_prob": 0.15,
    "two_hop_prob": 0.15,
    "verbose_prob": 0.3,
    "long_distractor_prob": 0.25,
}

ATTRIBUTES = [
    "access code", "serial", "checksum", "location tag", "status code", "port", "batch",
    "priority", "channel", "voucher", "route", "revision",
]
LINK_ATTRIBUTES = ["custodian", "operator", "supervisor", "courier"]
ENTITY_KINDS = ["vault", "server", "order", "ticket", "account", "sensor", "parcel", "node"]
PERSON_KINDS = ["agent", "clerk", "tech"]
FILLER = [
    "The system ran a routine check and nothing changed.",
    "A background job finished without any warnings.",
    "The operator noted that the queue was quiet again.",
    "Nothing unusual was seen during the last sweep.",
    "The connection dropped for a moment and then came back.",
    "Someone asked whether the meeting could be moved to later.",
    "The cache was cleared and the service restarted as planned.",
    "A reminder went out about the weekly review.",
    "The dashboard refreshed and showed the same figures as before.",
    "There was a short delay while the index was rebuilt.",
    "The log rotated and the old file was compressed.",
    "A heartbeat arrived on time from every worker.",
    "The user said thanks and asked for a moment to think.",
    "The retry succeeded on the second attempt with no data lost.",
    "An idle timeout closed a session that nobody was using.",
    "The report was generated and left in the usual folder.",
]


@dataclass
class _Fact:
    attribute: str
    entity: str
    value: str
    sentence: str
    source: SourceType
    carriers: list[str] = field(default_factory=list)  # items stating the current value


class SyntheticRecallEnv(TaskEnvironment):
    name = "synthetic_recall"
    goal = "Answer each question with the current value, or 'unknown' if you cannot tell."

    def __init__(self, config: dict) -> None:
        super().__init__(merged(DEFAULTS, config))
        self.horizon = int(self.config["horizon"])
        self._observations: list[Observation] = []
        self._dependencies: dict[str, Dependency] = {}
        self._index = 0
        self._last_reward = 0.0
        self._scored = 0
        self._correct = 0
        self._stats: dict = {}

    # ---- TaskEnvironment ---------------------------------------------------

    def reset(self, seed: int) -> Observation:
        self._generate(seed)
        self._index = 0
        self._last_reward = 0.0
        self._scored = 0
        self._correct = 0
        return self._observations[0]

    def step(self, agent_action: str | None) -> StepResult:
        current = self._observations[self._index]
        info: dict = {"scored": False, "correct": None}
        self._last_reward = 0.0
        if current.requires_response:
            gold = self._dependencies[current.id].gold
            # The answer is right if it names the value; a language model may wrap it in a sentence.
            correct = gold.lower() in _ANSWER_TOKEN.findall((agent_action or UNKNOWN).lower())
            self._scored += 1
            self._correct += int(correct)
            self._last_reward = float(correct)
            info = {"scored": True, "correct": correct, "gold": gold, "query_id": current.id}
        self._index += 1
        return StepResult(self.get_observation(), self._last_reward, self.is_done(), info)

    def get_observation(self) -> Observation | None:
        return None if self.is_done() else self._observations[self._index]

    def is_done(self) -> bool:
        return self._index >= len(self._observations)

    def get_reward(self) -> float:
        return self._last_reward

    def task_success(self) -> float:
        return self._correct / self._scored if self._scored else 0.0

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        seen = {observation.id for observation in self._observations[: self._index + 1]}
        return [d for query_id, d in self._dependencies.items() if query_id in seen]

    def episode_stats(self) -> dict:
        return dict(self._stats)

    # ---- generation --------------------------------------------------------

    def _generate(self, seed: int) -> None:
        cfg, rng = self.config, random.Random(seed)
        self._observations, self._dependencies = [], {}
        stats = defaultdict(int)
        facts: dict[tuple[str, str], _Fact] = {}
        agenda: dict[int, list[tuple]] = defaultdict(list)
        counter = [100]
        sources = [SourceType(name) for name in cfg["sources"]]
        shares = [cfg["sources"][source.value]["share"] for source in sources]

        def gap() -> int:
            low = max(1, int(cfg["gap"]["min"]))
            high = max(low, int(cfg["gap"]["max"] or self.horizon))
            kind = cfg["gap"].get("dist", "loguniform")
            if kind == "fixed":
                return low
            if kind == "uniform":
                return rng.randint(low, high)
            return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))

        def new_name(kinds: list[str]) -> str:
            counter[0] += rng.randint(1, 9)
            return f"{rng.choice(kinds)}-{counter[0]}"

        def new_value() -> str:
            letters, digits = "ABCDEFGHJKLMNPQRSTUVWXYZ", "0123456789"
            return rng.choice(letters) + rng.choice(digits) + rng.choice(digits) + rng.choice(letters)

        def filler(long: bool) -> list[str]:
            return [rng.choice(FILLER) for _ in range(rng.randint(5, 12) if long else rng.randint(1, 2))]

        def wrap(sentence: str, force_long: bool = False) -> str:
            """The fact alone, or buried at a random position in a long item."""
            if not force_long and rng.random() >= cfg["verbose_prob"]:
                return sentence
            around = filler(long=True)
            around.insert(rng.randint(0, len(around)), sentence)
            return " ".join(around)

        def emit(step: int, content: str, source: SourceType, query: bool = False) -> str:
            observation = Observation(f"o{step:05d}", content, source, requires_response=query)
            self._observations.append(observation)
            return observation.id

        def state_fact(step: int, attribute: str, entity: str, value: str, source: SourceType) -> _Fact:
            sentence = fact_sentence(step, attribute, entity, value)
            item_id = emit(step, wrap(sentence), source)
            fact = _Fact(attribute, entity, value, sentence, source, [item_id])
            facts[attribute, entity] = fact
            return fact

        def requirement(fact: _Fact) -> EvidenceRequirement:
            return EvidenceRequirement(tuple(fact.carriers), fact.sentence)

        for step in range(1, self.horizon + 1):
            pending = agenda.pop(step, [])
            if pending:
                event, rest = pending[0], pending[1:]
                if rest:  # one observation per step: the others wait, ahead of what is already due
                    agenda[step + 1] = rest + agenda[step + 1]
                kind = event[0]
                if kind == "query":
                    _, key, via_key = event
                    fact, via = facts[key], facts.get(via_key)
                    text = query_sentence(fact.attribute, via.entity if via else fact.entity, via.attribute if via else None)
                    query_id = emit(step, text, SourceType.USER, query=True)
                    needs = (requirement(via), requirement(fact)) if via else (requirement(fact),)
                    category = "two_hop" if via else "one_hop"
                    self._dependencies[query_id] = Dependency(query_id, step, needs, fact.value, category)
                    stats["queries"] += 1
                    stats[f"{category}_queries"] += 1
                    if rng.random() < cfg["requery_prob"]:
                        agenda[step + gap()].append(event)
                elif kind == "update":
                    old = facts[event[1]]
                    state_fact(step, old.attribute, old.entity, new_value(), old.source)
                    stats["updates"] += 1
                elif kind == "restate":
                    fact = facts[event[1]]
                    fact.carriers.append(emit(step, wrap(fact.sentence, force_long=True), SourceType.TOOL_OUTPUT))
                    stats["restatements"] += 1
                elif kind == "second_hop":
                    _, person, via_key, source = event
                    fact = state_fact(step, rng.choice(ATTRIBUTES), person, new_value(), source)
                    agenda[step + gap()].append(("query", (fact.attribute, fact.entity), via_key))
                    stats["facts"] += 1
                continue

            if rng.random() < cfg["fact_prob"]:
                source = rng.choices(sources, weights=shares)[0]
                queried = rng.random() < cfg["sources"][source.value]["query_prob"]
                entity = new_name(ENTITY_KINDS)
                stats["facts"] += 1
                if queried and rng.random() < cfg["two_hop_prob"]:
                    person = new_name(PERSON_KINDS)
                    link = state_fact(step, rng.choice(LINK_ATTRIBUTES), entity, person, source)
                    agenda[step + rng.randint(1, 20)].append(
                        ("second_hop", person, (link.attribute, link.entity), source)
                    )
                    continue
                fact = state_fact(step, rng.choice(ATTRIBUTES), entity, new_value(), source)
                key = (fact.attribute, fact.entity)
                if queried:
                    agenda[step + gap()].append(("query", key, None))
                if rng.random() < cfg["update_prob"]:
                    agenda[step + gap()].append(("update", key))
                if rng.random() < cfg["restate_prob"]:
                    agenda[step + gap()].append(("restate", key))
            else:
                long = rng.random() < cfg["long_distractor_prob"]
                source = SourceType.TOOL_OUTPUT if long else rng.choices(sources, weights=shares)[0]
                emit(step, " ".join(filler(long)), source)
                stats["distractors"] += 1

        self._stats = dict(stats)
