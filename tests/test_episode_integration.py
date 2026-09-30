"""Whole episodes on the toy environment: the properties every experiment relies on."""

import pytest

from memctl.config import resolve
from memctl.harness.runner import run_one

ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]
ALL_OPS = ARCHIVE_OPS + ["COMPACT", "COMPACT_AND_ARCHIVE", "CONSOLIDATE"]


def config(controller, fraction=0.25, horizon=200, noise=0.0, operations=None, **memory):
    if isinstance(controller, str):
        controller = {"name": controller}
    settings = {"budget": {"fraction": fraction}, **memory}
    if operations:
        settings["allowed_operations"] = operations
    return resolve(
        {
            "env": {"name": "synthetic_recall", "horizon": horizon},
            "agent": {"name": "scripted_reader", "noise": noise},
            "controller": controller,
            "memory": settings,
        }
    )


def success(controller, seed=0, **kwargs):
    return run_one(config(controller, **kwargs), seed).episode["task_success"]


def test_one_full_toy_episode_produces_every_kind_of_record():
    result = run_one(config("lru"), seed=0)
    episode = result.episode
    assert episode["steps"] == 200 and episode["queries"] > 5
    assert 0.0 <= episode["task_success"] <= 1.0
    assert len(result.steps) == 200 and len(result.rewards) == 200
    assert result.actions and result.items and len(result.items) >= 200
    step = result.steps[50]
    for key in ("episode_id", "step", "seed", "memory", "controller_actions", "forced_actions",
                "agent_action", "reward", "cumulative_reward", "controller_latency_s", "active_tokens_at_read"):
        assert key in step, key
    assert {"task_reward", "forced_fallback", "invalid_action", "active_memory_cost"} <= set(result.rewards[0]["terms"])


def test_the_budget_holds_whenever_the_agent_reads():
    for name in ("no_controller", "fifo", "lru", "random", "salience"):
        result = run_one(config(name, fraction=0.1), seed=1)
        assert all(row["active_tokens_at_read"] <= result.episode["budget"] for row in result.steps), name
        assert result.episode["active_tokens_peak"] <= result.episode["budget"]


def test_the_budget_is_a_fraction_of_the_uncompressed_history():
    full = run_one(config("fifo", fraction=1.0), seed=2).episode
    quarter = run_one(config("fifo", fraction=0.25), seed=2).episode
    assert quarter["budget"] == int(0.25 * full["history_tokens"])
    assert full["budget"] == full["history_tokens"]


def test_at_full_budget_every_controller_is_perfect():
    for name in ("no_controller", "fifo", "lru", "lfu", "random", "age_decay", "salience", "oracle"):
        assert success(name, fraction=1.0) == 1.0, name


def test_the_same_seed_reproduces_the_episode_exactly():
    first, second = run_one(config("random"), seed=5), run_one(config("random"), seed=5)
    assert first.episode["task_success"] == second.episode["task_success"]
    assert [a["target_ids"] for a in first.actions] == [a["target_ids"] for a in second.actions]


def test_no_controller_is_fifo_by_forced_fallback_and_is_reported_as_such():
    lazy = run_one(config("no_controller", fraction=0.1), seed=3)
    fifo = run_one(config("fifo", fraction=0.1), seed=3)
    assert lazy.episode["task_success"] == fifo.episode["task_success"]
    assert lazy.episode["forced_evictions"] > 0 and fifo.episode["forced_evictions"] == 0
    assert lazy.episode["forced_fallback_rate"] > 0
    forced = [a for a in lazy.actions if a["source"] == "harness"]
    assert forced and all(a["operation"] == "EVICT" for a in forced)
    assert not [a for a in fifo.actions if a["source"] == "harness"]
    assert lazy.episode["reward_terms"]["forced_fallback"] == lazy.episode["forced_evictions"]
    assert lazy.episode["reward_total"] < fifo.episode["reward_total"]


def test_the_oracle_beats_the_heuristics_under_pressure_and_exact_is_at_least_approx():
    for seed in range(3):
        exact = success({"name": "oracle", "method": "exact"}, seed, fraction=0.1)
        approx = success("oracle", seed, fraction=0.1)
        assert exact >= approx
        for name in ("fifo", "lru", "random", "salience"):
            assert exact >= success(name, seed, fraction=0.1), (seed, name)
    assert success("oracle", 0, fraction=0.1) > success("fifo", 0, fraction=0.1)


def test_the_exact_oracle_achieves_exactly_what_its_plan_promises():
    result = run_one(config({"name": "oracle", "method": "exact"}, fraction=0.1), seed=4)
    assert result.episode["oracle_method"] == "exact"
    assert result.episode["correct"] == result.episode["oracle_covered"]


def test_with_a_perfect_reader_and_deletion_only_every_failure_is_an_eviction():
    result = run_one(config("fifo", fraction=0.1), seed=0)
    assert result.failures and all(f["label"] == "evicted" for f in result.failures)
    assert all(f["cause"] == "controller" for f in result.failures)
    assert result.episode["failures"] == {"evicted": len(result.failures)}


def test_failures_of_the_fallback_are_marked_as_forced():
    result = run_one(config("no_controller", fraction=0.1), seed=0)
    assert result.failures and all(f["label"] == "evicted" and f["cause"] == "harness" for f in result.failures)


def test_with_full_memory_every_failure_is_the_task_models():
    result = run_one(config("fifo", fraction=1.0, noise=1.0), seed=0)
    assert result.episode["task_success"] == 0.0
    assert {f["label"] for f in result.failures} == {"task_model_reasoning"}


def test_archiving_without_retrieval_fails_as_retrieval_not_as_eviction():
    controller = {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"]}
    result = run_one(config(controller, fraction=0.1, operations=ARCHIVE_OPS), seed=0)
    assert result.failures and {f["label"] for f in result.failures} == {"archived_not_retrieved"}
    assert result.episode["archive_tokens_final"] > 0


def test_archive_with_retrieval_beats_deletion_and_logs_each_retrieval():
    controller = {"name": "lru", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 3}}
    with_archive = run_one(config(controller, fraction=0.1, operations=ARCHIVE_OPS), seed=0)
    assert with_archive.episode["task_success"] > success("lru", 0, fraction=0.1)
    assert with_archive.retrievals and with_archive.episode["retrieved_items"] > 0
    assert 0 < with_archive.episode["retrieval_precision"] <= 1
    assert 0 < with_archive.episode["retrieval_recall"] <= 1
    assert all(row["query"] for row in with_archive.retrievals)


def test_the_oracle_with_archive_retrieves_just_in_time():
    result = run_one(config("oracle", fraction=0.05, operations=ARCHIVE_OPS), seed=0)
    assert result.episode["task_success"] == 1.0
    assert result.episode["retrieval_precision"] == 1.0


def test_a_lossy_compressor_is_blamed_for_the_detail_it_loses():
    controller = {"name": "fifo", "removal": ["COMPACT", "EVICT"], "compact_ratio": 0.3}
    lossy = run_one(config(controller, fraction=0.2, operations=ALL_OPS, compressor="truncate"), seed=0)
    assert "compression_lost_detail" in lossy.episode["failures"]
    careful = run_one(config(controller, fraction=0.2, operations=ALL_OPS, compressor="extractive"), seed=0)
    assert careful.episode["task_success"] > lossy.episode["task_success"]
    assert 0 < careful.episode["compression_ratio"] < 1


def test_an_operation_that_is_not_allowed_is_counted_as_invalid_and_changes_nothing():
    from memctl.controllers.heuristics import NoController
    from memctl.memory.actions import MemoryAction, Operation

    class RogueController(NoController):
        """Archives the newest item every step, in an experiment that only allows deletion."""

        name = "rogue"

        def decide(self, memory, task):
            return [MemoryAction(Operation.MOVE_TO_ARCHIVE, (task.observation.id,))]

    result = run_one(config("no_controller", fraction=0.1), seed=0, controller=RogueController({}, 0))
    assert result.episode["invalid_actions"] == 200
    assert result.episode["archive_tokens_final"] == 0
    assert {f["label"] for f in result.failures} == {"invalid_action"}


def test_the_controller_never_sees_ground_truth():
    from memctl.controllers.heuristics import Fifo

    seen = []

    class Spy(Fifo):
        def decide(self, memory, task):
            seen.append(repr((memory, task)))
            return super().decide(memory, task)

        def update(self, feedback):
            seen.append(repr(feedback.reward_terms))

    result = run_one(config("fifo"), seed=0, controller=Spy({}, 0))
    golds = {d.gold for d in result.dependencies}
    queries = [row for row in result.steps if row["scored"]]
    assert queries and golds
    text = "\n".join(seen)
    assert "Dependency" not in text and "EvidenceRequirement" not in text and "needle" not in text


def test_regret_rows_name_the_decision_that_destroyed_needed_evidence():
    result = run_one(config("fifo", fraction=0.1), seed=0)
    assert result.regrets
    row = result.regrets[0]
    assert row["operation"] == "EVICT" and row["source"] == "controller" and row["steps_until_needed"] >= 0
    lost_queries = {r["query_id"] for r in result.regrets}
    failed_queries = {f["query_id"] for f in result.failures}
    assert failed_queries <= lost_queries
    assert run_one(config("oracle", fraction=1.0), seed=0).regrets == []


def test_unnecessary_retention_is_measured_against_hindsight():
    fifo = run_one(config("fifo", fraction=0.25), seed=0).episode
    oracle = run_one(config("oracle", fraction=0.25), seed=0).episode
    assert 0 < fifo["unnecessary_token_share"] <= 1
    assert oracle["unnecessary_token_share"] < fifo["unnecessary_token_share"]


def test_shadow_controllers_report_agreement_without_acting():
    shadowed = config("lru", fraction=0.1)
    shadowed["shadow_controllers"] = [{"name": "fifo"}, {"name": "lru"}]
    result = run_one(shadowed, seed=0)
    plain = run_one(config("lru", fraction=0.1), seed=0)
    assert result.episode["task_success"] == plain.episode["task_success"]
    assert result.episode["shadow_agreement"]["lru"] == pytest.approx(1.0)
    assert 0 <= result.episode["shadow_agreement"]["fifo"] < 1.0


def test_agent_actions_can_be_stored_as_memory():
    result = run_one(config("fifo", fraction=1.0, store_agent_actions=True), seed=0)
    actions = [item for item in result.items if item["source_type"] == "action"]
    assert len(actions) == result.episode["queries"]
    assert result.episode["task_success"] == 1.0


def test_a_rejected_action_that_did_not_target_the_lost_item_is_not_blamed_for_it():
    from memctl.attribution import failure_label
    from memctl.memory.actions import MemoryAction, Operation
    from memctl.memory.engine import MemoryEngine
    from memctl.memory.state import MemoryState
    from memctl.memory.items import SourceType
    from memctl.hindsight.evidence import EvidenceTracker
    from memctl.task import EvidenceRequirement

    state = MemoryState(budget=6)
    for number, content in enumerate(["needed fact here", "other one here", "third one here"]):
        state.step = number + 1
        state.ingest(f"o{number}", content, SourceType.USER)
    engine = MemoryEngine(["KEEP", "EVICT", "NO_OP"])
    state.step = 4
    engine.apply(state, [MemoryAction(Operation.MOVE_TO_ARCHIVE, ("o1",)), MemoryAction(Operation.KEEP, ("o2",))])
    engine.enforce_budget(state)  # evicts o0 (oldest): the rejected action was about o1, and KEEP was applied
    status = EvidenceTracker(state).status(EvidenceRequirement(("o0",), "needed fact"))
    assert failure_label([status], state) == ("evicted", "harness")
    state2 = MemoryState(budget=6)
    for number, content in enumerate(["needed fact here", "other one here", "third one here"]):
        state2.step = number + 1
        state2.ingest(f"o{number}", content, SourceType.USER)
    state2.step = 4
    engine.apply(state2, [MemoryAction(Operation.MOVE_TO_ARCHIVE, ("o1",))])  # the only action, rejected
    engine.enforce_budget(state2)
    status = EvidenceTracker(state2).status(EvidenceRequirement(("o0",), "needed fact"))
    assert failure_label([status], state2) == ("invalid_action", "harness")


def test_retrieved_but_ignored_means_retrieved_at_this_step():
    from memctl.attribution import failure_label
    from memctl.hindsight.evidence import RequirementStatus
    from memctl.task import EvidenceRequirement

    status = RequirementStatus(EvidenceRequirement(("o0",), "x"), active=["o0"])
    assert failure_label([status], None, retrieved_now=["o0"])[0] == "retrieved_but_ignored"
    assert failure_label([status], None, retrieved_now=[])[0] == "task_model_reasoning"


def test_action_counts_are_per_item():
    result = run_one(config("oracle", fraction=0.1), seed=0)
    counted = sum(result.episode["action_counts"]["controller"].values())
    items = sum(len(row["target_ids"]) for row in result.actions if row["status"] == "applied" and row["source"] == "controller")
    assert counted == items
