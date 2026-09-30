"""Hindsight infrastructure: evidence sources, counterfactual ablation and decision regret."""

import pytest

from memctl.config import resolve
from memctl.harness.runner import Experiment, run_one
from memctl.hindsight.ablation import ablate
from memctl.hindsight.regret import continuation_regret, summarize_regret
from memctl.hindsight.trace import string_dependencies, tool_value_dependencies
from memctl.memory.actions import MemoryAction, Operation


def recall_config(controller="fifo", fraction=0.25, horizon=150):
    return resolve(
        {"env": {"name": "synthetic_recall", "horizon": horizon}, "controller": {"name": controller},
         "memory": {"budget": {"fraction": fraction}}}
    )


def workflow_config(controller="full_context"):
    return resolve(
        {"env": {"name": "workflow", "jobs": 6}, "agent": {"name": "scripted_tool_agent"},
         "controller": {"name": controller}, "memory": {"budget": {"fraction": 1.0}}}
    )


def labelled(result):
    """query id -> the set of item ids the benchmark says it needs (all alternatives)."""
    return {d.query_id: {i for r in d.requirements for i in r.item_ids} for d in result.dependencies}


def test_string_tracing_finds_the_labelled_evidence_without_using_the_labels():
    result = run_one(recall_config("full_context"), seed=0)
    queries = [(d.query_id, d.step, d.gold) for d in result.dependencies]
    traced = string_dependencies(result.items, queries)
    truth = labelled(result)
    assert {d.query_id for d in traced} == set(truth)
    exact = 0
    for dependency in traced:
        found = {i for r in dependency.requirements for i in r.item_ids}
        assert found, dependency.query_id
        assert all(int(i[1:]) < dependency.step for i in found)  # only earlier items
        final_hop = set(next(d for d in result.dependencies if d.query_id == dependency.query_id).requirements[-1].item_ids)
        exact += int(found == final_hop)
    assert exact / len(traced) > 0.9  # the item that states the answer; first hops are invisible to it


def test_tool_value_tracing_finds_where_each_used_token_came_from():
    result = run_one(workflow_config(), seed=0)
    actions = [(row["step"], row["observation_id"], row["agent_action"]) for row in result.steps if row["agent_action"]]
    traced = {d.query_id: d for d in tool_value_dependencies(result.items, actions)}
    truth = labelled(result)
    checked = 0
    for query_id, needed in truth.items():
        if not needed:
            continue  # first stages need no token
        found = {i for r in traced[query_id].requirements for i in r.item_ids}
        assert needed <= found, query_id
        checked += 1
    assert checked > 5


def test_ablating_a_needed_item_costs_exactly_the_queries_that_need_it():
    experiment = Experiment(recall_config("full_context"))
    baseline = experiment.run_episode(seed=0)
    hindsight = experiment.hindsight(0)
    needed = next(i for i, steps in hindsight.all_needs.items() if len(steps) == 1
                  and all(len(r.item_ids) == 1 for d in hindsight.dependencies for r in d.requirements if i in r.item_ids))
    never = next(i for i in hindsight.items if not hindsight.ever_needed(i))
    effects = ablate(experiment, seed=0, item_ids=[needed, never])
    queries = baseline.episode["queries"]
    assert effects[needed]["success_drop"] == pytest.approx(1 / queries)
    assert effects[needed]["queries_lost"] == 1
    assert effects[never]["success_drop"] == 0.0
    assert effects[needed]["baseline_success"] == 1.0


def test_continuation_regret_is_the_value_lost_by_forcing_an_action():
    experiment = Experiment(recall_config("fifo", fraction=0.5))
    hindsight = experiment.hindsight(0)
    item = next(i for i, steps in hindsight.needs.items() if len(steps) >= 1
                and hindsight.items[i].step < 60 and steps[0] > hindsight.items[i].step + 20
                and all(len(r.item_ids) == 1 for d in hindsight.dependencies for r in d.requirements if i in r.item_ids))
    step = hindsight.items[item].step + 1
    forced = continuation_regret(experiment, seed=0, step=step, action=MemoryAction(Operation.EVICT, (item,)))
    assert forced["regret"] == pytest.approx(len(hindsight.needs[item]) / forced["queries"])
    assert forced["success_free"] > forced["success_forced"]

    useless = next(i for i in hindsight.items if not hindsight.ever_needed(i) and hindsight.items[i].step < 60)
    harmless = continuation_regret(
        experiment, seed=0, step=hindsight.items[useless].step + 1, action=MemoryAction(Operation.EVICT, (useless,))
    )
    assert harmless["regret"] == 0.0


def test_regret_rows_are_summarised_by_operation_source_and_time_to_need():
    result = run_one(recall_config("no_controller", fraction=0.1), seed=0)
    summary = summarize_regret(result.regrets)
    assert summary["requirements_destroyed"] == len(result.regrets) > 0
    assert summary["by_source"] == {"harness": len(result.regrets)}
    assert summary["by_operation"] == {"EVICT": len(result.regrets)}
    assert sum(summary["steps_until_needed_histogram"].values()) == len(result.regrets)
    assert summary["median_steps_until_needed"] > 0
