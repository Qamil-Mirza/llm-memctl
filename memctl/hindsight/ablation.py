"""Counterfactual ablation: how much does the episode lose without one memory item?

The episode is replayed from the start with one change: the item is deleted at
the step it arrives. The drop in task success is the item's measured value.
This is the most direct evidence source and the most expensive: one replay per
item. It needs a deterministic environment and task model.

It measures value *under the controller and budget of the experiment*. With
`full_context` it is the item's value to a task model that forgets nothing.
"""

from __future__ import annotations

from memctl.memory.actions import MemoryAction, Operation


def ablate(experiment, seed: int, item_ids: list[str]) -> dict[str, dict]:
    """item id -> {baseline_success, ablated_success, success_drop, queries_lost}."""
    baseline = experiment.run_episode(seed=seed, detail=False).episode
    arrival = {item_id: record.step for item_id, record in experiment.hindsight(seed).items.items()}
    effects = {}
    for item_id in item_ids:
        if item_id not in arrival:
            raise KeyError(f"item {item_id} does not occur in the reference pass of seed {seed}")
        lesion = {arrival[item_id]: [MemoryAction(Operation.EVICT, (item_id,))]}
        ablated = experiment.run_episode(seed=seed, detail=False, interventions=lesion).episode
        effects[item_id] = {
            "baseline_success": baseline["task_success"],
            "ablated_success": ablated["task_success"],
            "success_drop": baseline["task_success"] - ablated["task_success"],
            "queries_lost": baseline["correct"] - ablated["correct"],
        }
    return effects
