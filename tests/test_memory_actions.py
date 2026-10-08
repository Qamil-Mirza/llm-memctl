"""The memory engine: what each operation does to the state, and what it refuses."""

import pytest

from memctl.memory.actions import ActionSource, ActionStatus, MemoryAction, Operation
from memctl.memory.compress import DedupConsolidator, ExtractiveCompressor, TruncateCompressor
from memctl.memory.engine import IMPLEMENTED_OPERATIONS, BudgetError, MemoryEngine
from memctl.memory.items import Fidelity, SourceType, Tier, count_tokens
from memctl.memory.state import MemoryState

FACT = "The access code of vault-317 is K93Q."
FILLER = "The system ran a routine check and nothing changed at all today."


def make_state(budget=100, contents=("one two three", "four five six", "seven eight nine"), **kwargs):
    state = MemoryState(budget=budget, **kwargs)
    for number, content in enumerate(contents):
        state.step = number + 1
        state.ingest(f"o{number}", content, SourceType.OBSERVATION)
    return state


def apply_one(engine, state, operation, *ids, **parameters):
    return engine.apply(state, [MemoryAction(operation, tuple(ids), parameters=parameters)])[0]


def test_ingest_places_items_in_active_and_counts_tokens():
    state = make_state()
    assert [item.id for item in state.active()] == ["o0", "o1", "o2"]
    assert state.active_tokens == 9
    assert state.get("o1").created_at == 2


def test_evict_deletes_the_item_but_keeps_it_for_the_logs():
    state, engine = make_state(), MemoryEngine()
    result = apply_one(engine, state, Operation.EVICT, "o0")
    assert result.status is ActionStatus.APPLIED and result.source is ActionSource.CONTROLLER
    assert state.get("o0").tier is Tier.DELETED
    assert "o0" not in [item.id for item in state.view().active + state.view().archived]
    assert state.active_tokens == 6
    assert result.tokens_freed == 3


def test_archive_then_retrieve_round_trip():
    state, engine = make_state(), MemoryEngine()
    apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o1")
    assert state.get("o1").tier is Tier.ARCHIVE and state.archive_tokens == 3
    state.step = 9
    result = apply_one(engine, state, Operation.RETRIEVE_FROM_ARCHIVE, "o1", query="four")
    assert result.status is ActionStatus.APPLIED
    assert state.get("o1").tier is Tier.ACTIVE and state.get("o1").retrieval_count == 1
    assert state.retrieval_history[-1]["item_ids"] == ["o1"]
    assert state.retrieval_history[-1]["query"] == "four"


def test_retrieving_an_item_that_is_not_archived_is_rejected():
    state, engine = make_state(), MemoryEngine()
    result = apply_one(engine, state, Operation.RETRIEVE_FROM_ARCHIVE, "o1")
    assert result.status is ActionStatus.REJECTED and "not in ARCHIVE" in result.reason
    assert state.get("o1").tier is Tier.ACTIVE


def test_deleted_items_cannot_be_acted_on():
    state, engine = make_state(), MemoryEngine()
    apply_one(engine, state, Operation.EVICT, "o0")
    assert apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o0").status is ActionStatus.REJECTED
    assert apply_one(engine, state, Operation.EVICT, "missing").status is ActionStatus.REJECTED


def test_a_rejected_action_changes_nothing_even_when_some_targets_are_valid():
    state, engine = make_state(), MemoryEngine()
    result = apply_one(engine, state, Operation.EVICT, "o0", "missing")
    assert result.status is ActionStatus.REJECTED
    assert state.get("o0").tier is Tier.ACTIVE


def test_operations_outside_the_allowed_set_are_rejected():
    state = make_state()
    engine = MemoryEngine(allowed_operations=[Operation.KEEP, Operation.EVICT, Operation.NO_OP])
    result = apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o0")
    assert result.status is ActionStatus.REJECTED and "not enabled" in result.reason
    assert apply_one(engine, state, Operation.EVICT, "o0").status is ActionStatus.APPLIED


def test_operations_without_a_handler_are_rejected_not_crashed():
    state, engine = make_state(), MemoryEngine(allowed_operations=list(Operation))
    result = apply_one(engine, state, Operation.PIN, "o0")
    assert result.status is ActionStatus.REJECTED and "no handler" in result.reason
    assert Operation.PIN not in IMPLEMENTED_OPERATIONS


def test_keep_and_no_op_are_applied_without_changing_the_state():
    state, engine = make_state(), MemoryEngine()
    before = state.summary()
    assert apply_one(engine, state, Operation.KEEP, "o0").status is ActionStatus.APPLIED
    assert apply_one(engine, state, Operation.NO_OP).status is ActionStatus.APPLIED
    after = state.summary()
    assert before["active_ids"] == after["active_ids"] and before["active_tokens"] == after["active_tokens"]


def test_compact_replaces_the_source_with_a_shorter_derived_item():
    state = make_state(contents=(f"{FILLER} {FACT} {FILLER}",))
    engine = MemoryEngine(compressor=ExtractiveCompressor())
    result = apply_one(engine, state, Operation.COMPACT, "o0", max_tokens=12)
    assert result.status is ActionStatus.APPLIED
    (new_id,) = result.created_ids
    new = state.get(new_id)
    assert new.tier is Tier.ACTIVE and new.fidelity is Fidelity.COMPACT
    assert new.derived_from_ids == ("o0",) and "K93Q" in new.content
    assert new.token_count <= 12
    old = state.get("o0")
    assert old.tier is Tier.DELETED and old.superseded_by == new_id


def test_truncating_compressor_can_lose_the_detail():
    state = make_state(contents=(f"{FILLER} {FACT}",))
    engine = MemoryEngine(compressor=TruncateCompressor())
    result = apply_one(engine, state, Operation.COMPACT, "o0", max_tokens=6)
    assert "K93Q" not in state.get(result.created_ids[0]).content


def test_compact_is_rejected_when_it_would_not_shrink_the_item():
    state, engine = make_state(), MemoryEngine(compressor=ExtractiveCompressor())
    result = apply_one(engine, state, Operation.COMPACT, "o0", max_tokens=50)
    assert result.status is ActionStatus.REJECTED and state.get("o0").tier is Tier.ACTIVE


def test_compact_and_archive_keeps_the_original_recoverable():
    state = make_state(contents=(f"{FILLER} {FACT} {FILLER}",))
    engine = MemoryEngine(compressor=ExtractiveCompressor())
    result = apply_one(engine, state, Operation.COMPACT_AND_ARCHIVE, "o0", max_tokens=12)
    assert state.get("o0").tier is Tier.ARCHIVE
    assert state.get(result.created_ids[0]).tier is Tier.ACTIVE
    assert state.active_tokens <= 12


def test_consolidate_merges_items_and_removes_duplicates():
    state = make_state(contents=(FACT, f"{FACT} {FILLER}", "The badge of agent-52 is B-8841."))
    engine = MemoryEngine(consolidator=DedupConsolidator())
    before = state.active_tokens
    result = apply_one(engine, state, Operation.CONSOLIDATE, "o0", "o1", "o2")
    assert result.status is ActionStatus.APPLIED
    merged = state.get(result.created_ids[0])
    assert merged.fidelity is Fidelity.CONSOLIDATED and merged.source_type is SourceType.CONSOLIDATED_MEMORY
    assert set(merged.derived_from_ids) == {"o0", "o1", "o2"}
    assert merged.content.count("K93Q") == 1 and "B-8841" in merged.content
    assert state.active_tokens < before
    assert all(state.get(i).tier is Tier.DELETED for i in ("o0", "o1", "o2"))
    assert state.get("o0").reference_count == 1


def test_consolidate_needs_at_least_two_active_items():
    state, engine = make_state(), MemoryEngine()
    assert apply_one(engine, state, Operation.CONSOLIDATE, "o0").status is ActionStatus.REJECTED


def test_consolidate_can_archive_its_sources():
    state, engine = make_state(contents=(FACT, FACT)), MemoryEngine()
    apply_one(engine, state, Operation.CONSOLIDATE, "o0", "o1", archive_sources=True)
    assert state.get("o0").tier is Tier.ARCHIVE and state.get("o1").tier is Tier.ARCHIVE


def test_pinned_items_cannot_be_removed():
    state = MemoryState(budget=100)
    state.ingest("p", "keep me always", SourceType.USER, pinned=True)
    result = apply_one(MemoryEngine(), state, Operation.EVICT, "p")
    assert result.status is ActionStatus.REJECTED and state.get("p").tier is Tier.ACTIVE


def test_archive_budget_is_enforced():
    state, engine = make_state(archive_budget=4), MemoryEngine()
    assert apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o0").status is ActionStatus.APPLIED
    result = apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o1")
    assert result.status is ActionStatus.REJECTED and "archive budget" in result.reason


def test_enforce_budget_evicts_oldest_first_and_marks_the_harness_as_source():
    state, engine = make_state(budget=4), MemoryEngine()
    forced = engine.enforce_budget(state)
    assert [r.action.target_ids for r in forced] == [("o0",), ("o1",)]
    assert all(r.source is ActionSource.HARNESS and r.action.operation is Operation.EVICT for r in forced)
    assert state.active_tokens <= 4 and [i.id for i in state.active()] == ["o2"]


def test_enforce_budget_does_nothing_when_the_state_fits():
    state, engine = make_state(budget=9), MemoryEngine()
    assert engine.enforce_budget(state) == []


def test_enforce_budget_works_even_when_evict_is_not_an_allowed_operation():
    state = make_state(budget=4)
    engine = MemoryEngine(allowed_operations=[Operation.KEEP, Operation.NO_OP])
    assert len(engine.enforce_budget(state)) == 2


def test_enforce_budget_skips_pinned_items_and_fails_loudly_if_they_alone_overflow():
    state = MemoryState(budget=4)
    state.ingest("p", "one two three", SourceType.USER, pinned=True)
    state.ingest("q", "four five six", SourceType.USER)
    MemoryEngine().enforce_budget(state)
    assert [i.id for i in state.active()] == ["p"]
    state.ingest("r", "seven eight nine", SourceType.USER, pinned=True)
    with pytest.raises(BudgetError):
        MemoryEngine().enforce_budget(state)


def test_the_state_records_how_each_item_left_active():
    state, engine = make_state(budget=7), MemoryEngine()
    apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o1")
    state.ingest("o3", "ten eleven twelve", SourceType.OBSERVATION)
    engine.enforce_budget(state)
    assert state.tier_events["o1"][-1]["operation"] == "MOVE_TO_ARCHIVE"
    assert state.tier_events["o1"][-1]["source"] == "controller"
    assert state.tier_events["o0"][-1] == {
        "step": state.step, "operation": "EVICT", "source": "harness", "tier": "DELETED"
    }


def test_access_bookkeeping():
    state = make_state()
    state.step = 7
    state.record_access(["o1", "missing"])
    assert state.get("o1").access_count == 1 and state.get("o1").last_accessed_at == 7


def test_views_do_not_expose_deleted_items_and_items_are_immutable():
    state, engine = make_state(), MemoryEngine()
    apply_one(engine, state, Operation.EVICT, "o0")
    view = state.view()
    assert [i.id for i in view.active] == ["o1", "o2"]
    with pytest.raises(Exception):
        view.active[0].content = "changed"


def test_count_tokens_is_model_free():
    assert count_tokens("Hello, world! It's 5.") == 9


def test_enforce_budget_returns_an_unaffordable_retrieval_to_the_archive_instead_of_deleting_it():
    state, engine = make_state(budget=7), MemoryEngine()
    apply_one(engine, state, Operation.MOVE_TO_ARCHIVE, "o0")  # active: o1, o2 (6 tokens)
    state.step = 5
    apply_one(engine, state, Operation.RETRIEVE_FROM_ARCHIVE, "o0")  # 9 tokens: over budget
    forced = engine.enforce_budget(state)
    assert [(r.action.operation, r.action.target_ids) for r in forced] == [(Operation.MOVE_TO_ARCHIVE, ("o0",))]
    assert state.get("o0").tier is Tier.ARCHIVE and state.get("o1").tier is Tier.ACTIVE  # o1 is older than o0's retrieval, and kept
    assert all(r.source is ActionSource.HARNESS for r in forced)


def test_a_written_note_leaves_its_sources_where_they_were():
    state = make_state(contents=(FACT, "The badge of agent-52 is B-8841.", "four five six"))
    state.move("o1", Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    engine = MemoryEngine(consolidator=DedupConsolidator())
    result = apply_one(engine, state, Operation.CONSOLIDATE, "o0", "o1", write_note=True, note="ranks 9-16")
    assert result.status is ActionStatus.APPLIED
    note = state.get(result.created_ids[0])
    assert note.tier is Tier.ACTIVE and note.metadata["speaker"] == "note" and note.metadata["note"] == "ranks 9-16"
    assert "K93Q" in note.content and "B-8841" in note.content and set(note.derived_from_ids) == {"o0", "o1"}
    assert state.get("o0").tier is Tier.ACTIVE and state.get("o1").tier is Tier.ARCHIVE  # sources untouched
    assert state.get("o0").superseded_by is None
