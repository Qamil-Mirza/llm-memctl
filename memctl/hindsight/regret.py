"""Decision regret: what a memory decision cost, judged with hindsight.

Two measures, cheap and exact-by-replay:

1. **Requirements destroyed** (cheap, logged for every episode in regret.jsonl).
   A decision is charged with each future evidence requirement it made
   unrecoverable. `summarize_regret` aggregates those rows.

2. **Continuation regret** (by replay). The definition

       regret(i, t, a) = best achievable future utility without forcing a
                       - best achievable future utility given a at (i, t)

   is approximated by letting the hindsight oracle play the rest of the
   episode in both branches: the experiment's controller runs up to step t,
   action a is or is not forced at t, and the oracle takes over after that. It
   costs two episodes per decision, and it is as exact as the oracle is.

Neither is tied to a use. They can feed imitation labels, reward shaping,
diagnostics, prioritised replay or oracle comparisons.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

from memctl.controllers.base import EpisodeInfo, Feedback, MemoryController, StepEvent
from memctl.controllers.oracle import OracleController
from memctl.memory.actions import MemoryAction
from memctl.memory.state import MemoryView
from memctl.task import TaskState

BUCKETS = ((10, "<10"), (50, "10-49"), (200, "50-199"), (10**9, "200+"))


def summarize_regret(rows: list[dict]) -> dict:
    gaps = [row["steps_until_needed"] for row in rows]
    histogram = Counter(next(label for limit, label in BUCKETS if gap < limit) for gap in gaps)
    return {
        "requirements_destroyed": len(rows),
        "queries_affected": len({(row.get("episode_id"), row["query_id"]) for row in rows}),
        "by_operation": dict(Counter(row["operation"] for row in rows)),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_loss": dict(Counter(row["loss"] for row in rows)),
        "steps_until_needed_histogram": dict(histogram),
        "median_steps_until_needed": median(gaps) if gaps else None,
    }


class SwitchController(MemoryController):
    """Runs one controller before `switch_step` and the hindsight oracle from then on."""

    name = "switch"
    uses_hindsight = True

    def __init__(self, first: MemoryController, switch_step: int, oracle_config: dict | None = None) -> None:
        super().__init__({}, first.seed)
        self.first = first
        self.then = OracleController({"eager": False, **(oracle_config or {})}, first.seed)
        self.switch_step = switch_step

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self.first.reset(episode)
        self.then.reset(episode)

    def receive_hindsight(self, hindsight) -> None:
        self.then.receive_hindsight(hindsight)
        if self.first.uses_hindsight:
            self.first.receive_hindsight(hindsight)

    def observe(self, event: StepEvent) -> None:
        self.first.observe(event)

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        return self.first.decide(memory, task) if memory.step < self.switch_step else self.then.decide(memory, task)

    def update(self, feedback: Feedback) -> None:
        self.first.update(feedback)


def continuation_regret(experiment, seed: int, step: int, action: MemoryAction) -> dict:
    """Regret of forcing `action` at `step`, with the oracle playing on from `step` in both branches."""
    original = experiment.controller
    try:
        experiment.controller = SwitchController(original, step)
        free = experiment.run_episode(seed=seed, detail=False).episode
        experiment.controller = SwitchController(original, step)
        forced = experiment.run_episode(seed=seed, detail=False, interventions={step: [action]}).episode
    finally:
        experiment.controller = original
    return {
        "step": step,
        "action": action.log_row(),
        "success_free": free["task_success"],
        "success_forced": forced["task_success"],
        "regret": free["task_success"] - forced["task_success"],
        "queries": free["queries"],
        "hindsight_exact": free["hindsight_exact"],
    }
