"""The memory engine: applies MemoryActions to a MemoryState and enforces the budget.

To add an operation: write one function decorated with `@handler(Operation.X)`.
It must check everything before it changes anything and raise `ActionError` to
refuse. Nothing else in the framework needs to change; controllers that never
emit the operation are unaffected.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

from memctl.memory.actions import ActionResult, ActionSource, ActionStatus, MemoryAction, Operation
from memctl.memory.compress import Compressor, Consolidator, DedupConsolidator, ExtractiveCompressor
from memctl.memory.items import Fidelity, MemoryItem, SourceType, Tier, count_tokens
from memctl.memory.state import MemoryState


class ActionError(Exception):
    """The action is not valid in the current state. The state has not been changed."""


class BudgetError(Exception):
    """The budget cannot be met even by the fallback (pinned items alone exceed it)."""


Handler = Callable[["MemoryEngine", MemoryState, MemoryAction, ActionSource], tuple[str, ...]]
HANDLERS: dict[Operation, Handler] = {}


def handler(operation: Operation):
    def register(function: Handler) -> Handler:
        HANDLERS[operation] = function
        return function

    return register


class MemoryEngine:
    def __init__(
        self,
        allowed_operations: Iterable[Operation | str] | None = None,
        compressor: Compressor | None = None,
        consolidator: Consolidator | None = None,
        compact_ratio: float = 0.5,
    ) -> None:
        """`allowed_operations` is the experiment's action set; None allows every implemented one."""
        if allowed_operations is None:
            self.allowed = frozenset(HANDLERS)
        else:
            self.allowed = frozenset(Operation(op) for op in allowed_operations)
        self.compressor = compressor or ExtractiveCompressor()
        self.consolidator = consolidator or DedupConsolidator(self.compressor)
        self.compact_ratio = compact_ratio

    def apply(
        self,
        state: MemoryState,
        actions: Iterable[MemoryAction],
        source: ActionSource = ActionSource.CONTROLLER,
        enforce_allowed: bool = True,
    ) -> list[ActionResult]:
        """Apply actions in order. Each is accepted or rejected on its own.

        `enforce_allowed=False` is for analysis interventions, which are not bound
        by the experiment's action set."""
        return [self._apply_one(state, action, source, enforce_allowed) for action in actions]

    def _apply_one(
        self, state: MemoryState, action: MemoryAction, source: ActionSource, enforce_allowed: bool = True
    ) -> ActionResult:
        before = state.active_tokens
        try:
            if enforce_allowed and action.operation not in self.allowed:
                raise ActionError(f"{action.operation.value} is not enabled in this experiment")
            if action.operation not in HANDLERS:
                raise ActionError(f"{action.operation.value} has no handler yet")
            created = HANDLERS[action.operation](self, state, action, source)
            result = ActionResult(
                action, ActionStatus.APPLIED, source, state.step, created_ids=created,
                tokens_freed=before - state.active_tokens,
            )
        except ActionError as error:
            result = ActionResult(action, ActionStatus.REJECTED, source, state.step, reason=str(error))
        state.action_history.append(result.log_row())
        return result

    def enforce_budget(self, state: MemoryState) -> list[ActionResult]:
        """The fallback every controller shares, logged with source `harness`.

        First, items retrieved from the archive at this very step go back to the
        archive (oldest first) while memory is over budget: a retrieval that does
        not fit is undone, not turned into a deletion. Then the oldest unpinned
        ACTIVE items are evicted until the budget holds. It ignores the allowed
        action set, because something must make the state fit.
        """
        results = []
        if state.active_tokens <= state.budget:
            return results
        for item in state.active():
            if state.active_tokens <= state.budget:
                break
            events = state.tier_events.get(item.id, [])
            just_retrieved = bool(events) and events[-1]["step"] == state.step and events[-1]["operation"] == Operation.RETRIEVE_FROM_ARCHIVE.value
            if not just_retrieved or item.pinned:
                continue
            before = state.active_tokens
            state.move(item.id, Tier.ARCHIVE, Operation.MOVE_TO_ARCHIVE.value, ActionSource.HARNESS.value)
            result = ActionResult(
                MemoryAction(Operation.MOVE_TO_ARCHIVE, (item.id,)), ActionStatus.APPLIED, ActionSource.HARNESS,
                state.step, reason="forced: a retrieval this step did not fit, returned to the archive",
                tokens_freed=before - state.active_tokens,
            )
            state.action_history.append(result.log_row())
            results.append(result)
        for item in state.active():
            if state.active_tokens <= state.budget:
                break
            if item.pinned:
                continue
            before = state.active_tokens
            state.move(item.id, Tier.DELETED, Operation.EVICT.value, ActionSource.HARNESS.value)
            result = ActionResult(
                MemoryAction(Operation.EVICT, (item.id,)), ActionStatus.APPLIED, ActionSource.HARNESS,
                state.step, reason="forced: over budget after the controller acted",
                tokens_freed=before - state.active_tokens,
            )
            state.action_history.append(result.log_row())
            results.append(result)
        if state.active_tokens > state.budget:
            raise BudgetError(
                f"pinned items need {state.active_tokens} tokens but the budget is {state.budget}"
            )
        return results


IMPLEMENTED_OPERATIONS = HANDLERS  # filled in below; `op in IMPLEMENTED_OPERATIONS` reads naturally


# ---- helpers ----------------------------------------------------------------


def _targets(state: MemoryState, action: MemoryAction, tier: Tier, removable: bool = True) -> list[MemoryItem]:
    """The action's target items, all of which must be in `tier`."""
    if not action.target_ids:
        raise ActionError(f"{action.operation.value} needs at least one target")
    if len(set(action.target_ids)) != len(action.target_ids):
        raise ActionError("a target is listed twice")
    items = []
    for item_id in action.target_ids:
        item = state.get(item_id)
        if item is None or item.tier is Tier.DELETED:
            raise ActionError(f"item {item_id} does not exist")
        if item.tier is not tier:
            raise ActionError(f"item {item_id} is not in {tier.value}")
        if removable and item.pinned:
            raise ActionError(f"item {item_id} is pinned")
        items.append(item)
    return items


def _check_archive_room(state: MemoryState, incoming_tokens: int) -> None:
    if state.archive_budget is not None and state.archive_tokens + incoming_tokens > state.archive_budget:
        raise ActionError(
            f"archive budget exceeded ({state.archive_tokens} + {incoming_tokens} > {state.archive_budget})"
        )


def _origin(items: list[MemoryItem]) -> int:
    """The earliest step any of the underlying original content entered memory."""
    return min(item.metadata.get("origin_created_at", item.created_at) for item in items)


def _compact_texts(engine: MemoryEngine, action: MemoryAction, items: list[MemoryItem]) -> list[str]:
    texts = []
    for item in items:
        ratio = float(action.parameters.get("ratio", engine.compact_ratio))
        limit = int(action.parameters.get("max_tokens", math.ceil(ratio * item.token_count)))
        text = engine.compressor.compress(item.content, max(1, limit))
        if not text or count_tokens(text) >= item.token_count:
            raise ActionError(f"compacting item {item.id} would not shrink it")
        texts.append(text)
    return texts


def _add_compact(engine: MemoryEngine, state: MemoryState, item: MemoryItem, text: str) -> str:
    new_id = state.new_id("cmp")
    state.ingest(
        new_id, text, SourceType.GENERATED_SUMMARY, fidelity=Fidelity.COMPACT, derived_from_ids=(item.id,),
        metadata={"compressor": engine.compressor.name, "origin_created_at": _origin([item])},
    )
    return new_id


# ---- handlers ---------------------------------------------------------------


@handler(Operation.NO_OP)
def _no_op(engine, state, action, source):
    return ()


@handler(Operation.KEEP)
def _keep(engine, state, action, source):
    for item_id in action.target_ids:
        item = state.get(item_id)
        if item is None or item.tier is Tier.DELETED:
            raise ActionError(f"item {item_id} does not exist")
    return ()


@handler(Operation.EVICT)
def _evict(engine, state, action, source):
    for item in _targets(state, action, Tier.ACTIVE):
        state.move(item.id, Tier.DELETED, action.operation.value, source.value)
    return ()


@handler(Operation.MOVE_TO_ARCHIVE)
def _move_to_archive(engine, state, action, source):
    items = _targets(state, action, Tier.ACTIVE)
    _check_archive_room(state, sum(item.token_count for item in items))
    for item in items:
        state.move(item.id, Tier.ARCHIVE, action.operation.value, source.value)
    return ()


@handler(Operation.RETRIEVE_FROM_ARCHIVE)
def _retrieve(engine, state, action, source):
    items = _targets(state, action, Tier.ARCHIVE, removable=False)
    for item in items:
        state.move(item.id, Tier.ACTIVE, action.operation.value, source.value)
        state.update(item.id, retrieval_count=item.retrieval_count + 1)
    state.retrieval_history.append(
        {
            "step": state.step,
            "item_ids": [item.id for item in items],
            "tokens": sum(item.token_count for item in items),
            "query": action.parameters.get("query"),
            "scores": action.parameters.get("scores"),
            "method": action.parameters.get("method"),
        }
    )
    return ()


@handler(Operation.COMPACT)
def _compact(engine, state, action, source):
    items = _targets(state, action, Tier.ACTIVE)
    texts = _compact_texts(engine, action, items)
    created = []
    for item, text in zip(items, texts):
        new_id = _add_compact(engine, state, item, text)
        state.move(item.id, Tier.DELETED, action.operation.value, source.value)
        state.update(item.id, superseded_by=new_id, reference_count=item.reference_count + 1)
        created.append(new_id)
    return tuple(created)


@handler(Operation.COMPACT_AND_ARCHIVE)
def _compact_and_archive(engine, state, action, source):
    items = _targets(state, action, Tier.ACTIVE)
    _check_archive_room(state, sum(item.token_count for item in items))
    texts = _compact_texts(engine, action, items)
    created = []
    for item, text in zip(items, texts):
        new_id = _add_compact(engine, state, item, text)
        state.move(item.id, Tier.ARCHIVE, action.operation.value, source.value)
        state.update(item.id, reference_count=item.reference_count + 1)
        created.append(new_id)
    return tuple(created)


@handler(Operation.CONSOLIDATE)
def _consolidate(engine, state, action, source):
    if action.parameters.get("write_note"):
        return _write_note(engine, state, action)
    items = _targets(state, action, Tier.ACTIVE)
    if len(items) < 2:
        raise ActionError("CONSOLIDATE needs at least two ACTIVE items")
    items.sort(key=lambda item: item.created_at)
    archive_sources = bool(action.parameters.get("archive_sources", False))
    total = sum(item.token_count for item in items)
    if archive_sources:
        _check_archive_room(state, total)
    text = engine.consolidator.consolidate([item.content for item in items], action.parameters.get("max_tokens"))
    if not text or count_tokens(text) > total:
        raise ActionError("consolidation produced nothing or grew the content")
    new_id = state.new_id("con")
    state.ingest(
        new_id, text, SourceType.CONSOLIDATED_MEMORY, fidelity=Fidelity.CONSOLIDATED,
        derived_from_ids=tuple(item.id for item in items),
        metadata={"consolidator": engine.consolidator.name, "origin_created_at": _origin(items)},
    )
    for item in items:
        state.move(item.id, Tier.ARCHIVE if archive_sources else Tier.DELETED, action.operation.value, source.value)
        changes = {"reference_count": item.reference_count + 1}
        if not archive_sources:
            changes["superseded_by"] = new_id
        state.update(item.id, **changes)
    return (new_id,)


def _write_note(engine, state, action):
    """CONSOLIDATE with `write_note` (Experiment 20): the consolidator writes one note from the targets, active or
    archived, and the note enters ACTIVE beside them. The sources are not moved or superseded, so the note is an
    addition to memory, not a replacement. With `labelled`, each source is given as "speaker (date): content"."""
    from memctl.retrieval import labelled_text  # memctl.retrieval imports the memory package

    items = []
    for item_id in action.target_ids:
        item = state.get(item_id)
        if item is None or item.tier is Tier.DELETED:
            raise ActionError(f"item {item_id} does not exist")
        items.append(item)
    if not items:
        raise ActionError("a note needs at least one source")
    items.sort(key=lambda item: item.created_at)
    texts, budget = [], int(action.parameters.get("max_input_tokens", 12000))
    for item in items:  # the writer's input is cut at max_input_tokens (§20: 12k)
        line = labelled_text(item) if action.parameters.get("labelled") else item.content
        if count_tokens(line) > budget:
            break
        texts.append(line)
        budget -= count_tokens(line)
    text = engine.consolidator.consolidate(texts or [items[0].content[:2000]], action.parameters.get("max_tokens"))
    if not text:
        raise ActionError("the note is empty")
    new_id = state.new_id("note")
    state.ingest(
        new_id, text, SourceType.CONSOLIDATED_MEMORY, fidelity=Fidelity.CONSOLIDATED,
        derived_from_ids=tuple(item.id for item in items),
        metadata={"speaker": "note", "consolidator": engine.consolidator.name, "note": action.parameters.get("note", "group"),
                  "origin_created_at": _origin(items)},
    )
    return (new_id,)
