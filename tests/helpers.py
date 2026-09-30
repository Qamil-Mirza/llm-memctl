"""Small builders shared by the tests."""

from memctl.controllers.base import EpisodeInfo
from memctl.memory.actions import Operation
from memctl.memory.engine import IMPLEMENTED_OPERATIONS
from memctl.memory.items import SourceType
from memctl.memory.state import MemoryState
from memctl.task import Observation, TaskState

DELETE_ONLY = (Operation.KEEP, Operation.EVICT, Operation.NO_OP)


def make_state(contents, budget=100, sources=None, embedder=None, archive_budget=None):
    """A state with one item per content string, ingested at steps 1, 2, 3, ..."""
    state = MemoryState(budget=budget, embedder=embedder, archive_budget=archive_budget)
    for number, content in enumerate(contents):
        state.step = number + 1
        source = sources[number] if sources else SourceType.OBSERVATION
        state.ingest(f"o{number}", content, source)
    return state


def episode_info(state, allowed=None, embedder=None):
    allowed = IMPLEMENTED_OPERATIONS if allowed is None else allowed
    return EpisodeInfo("test", 0, state.budget, state.archive_budget, frozenset(allowed), embedder=embedder)


def task_for(state, item_id=None, query=False):
    """The TaskState for the newest item (or the named one)."""
    item = state.get(item_id) if item_id else state.active()[-1]
    observation = Observation(item.id, item.content, item.source_type, requires_response=query)
    return TaskState(step=state.step, goal="", observation=observation)


def decide(controller, state, allowed=None, query=False, embedder=None):
    controller.reset(episode_info(state, allowed, embedder))
    return controller.decide(state.view(), task_for(state, query=query))
