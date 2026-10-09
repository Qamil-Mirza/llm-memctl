"""LRE as a memory controller (EXPERIMENTS.md §27): its query-blind scorer (memctl/lre.py) in two pre-declared arms.

    controller: {name: lre, model: runs/lre_exp27_f0/lre.json, mode: native | slot, ...}

- mode "native" (LRE-native): FIFO + floor with LRE's score in place of recency. When ACTIVE is over the limit
  (target_tokens, else the budget) the lowest-scoring unpinned items are moved to the archive until it fits; ties
  are broken oldest-first. Retrieval is FIFO + floor's own: at a question, the `retrieve` block (top_k BM25 hits
  from the archive, fit to the room) is retrieved before any removal and protected from it. The question itself is
  never removed (FIFO never removes it either: it is the newest item). Everything else is the `fifo` controller's
  code path (PriorityController), so LRE-native and FIFO + floor differ only in the priority.
- mode "slot" (LRE-slot): as the §19 head's `keep_none` and §26's OpenJev `select`: everything but the current input
  is archived on arrival; at a question the same `retrieve_candidates` BM25 hits the head sees are ranked by LRE's
  score and the `select_k` best are retrieved (ties keep the BM25 order). A shortlist no longer than select_k is
  retrieved whole.

The score of an item is LRE's logit with its place among the turns seen so far: idx is the item's arrival order
(0-based) and n the number of non-question items seen. The question is never scored and the scorer never sees it.
"""

from __future__ import annotations

import time

from memctl.controllers.base import EpisodeInfo, MemoryController
from memctl.controllers.heuristics import Fifo
from memctl.lre import LREModel, traj_features
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.items import MemoryItem
from memctl.memory.state import MemoryView
from memctl.retrieval import build_retriever
from memctl.task import TaskState

_MODELS: dict[str, LREModel] = {}


def _model(path: str) -> LREModel:
    if path not in _MODELS:
        _MODELS[path] = LREModel.load(path)
    return _MODELS[path]


class _Scores:
    """Arrival order and cached text logits for one episode."""

    def __init__(self, model: LREModel) -> None:
        self.model = model
        self.order: dict[str, int] = {}
        self.text: dict[str, float] = {}

    def see(self, memory: MemoryView, task: TaskState) -> None:
        question = task.observation.id if task.observation.requires_response else None
        new = [item for item in (*memory.active, *memory.archived)
               if item.id not in self.order and item.id != question]
        for item in sorted(new, key=lambda item: (item.created_at, item.id)):
            self.order[item.id] = len(self.order)

    def logit(self, item: MemoryItem) -> float:
        if item.id not in self.text:
            self.text[item.id] = self.model.text_logit(item.content)
        idx, n = self.order[item.id], len(self.order)
        return self.model.intercept + self.text[item.id] + self.model.traj_logit(traj_features(item.content, idx, n))


class LREController(MemoryController):
    name = "lre"
    model_id = "lre-logistic"

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        if not config.get("model"):
            raise ValueError("the lre controller needs `model`: the path of a trained LRE model (memctl.lre_train)")
        self.mode = config.get("mode", "native")
        if self.mode not in ("native", "slot"):
            raise ValueError(f"unknown lre mode '{self.mode}' (known: native, slot)")
        self.model = _model(str(config["model"]))
        self.model_id = f"lre:{config['model']}"
        self.select_k = int(config.get("select_k", 8))
        self.retrieve_candidates = int(config.get("retrieve_candidates", 32))
        self._native = None
        if self.mode == "native":
            self._native = _NativeLRE({k: v for k, v in config.items() if k not in ("name", "model", "mode")}, seed, self)
        self.retriever = None
        self.scores: _Scores | None = None
        self._info: dict = {}

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self.scores = _Scores(self.model)
        if self._native is not None:
            self._native.reset(episode)
        else:
            self.retriever = build_retriever(self.config.get("retrieval_method", "lexical"), episode.embedder)

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        started = time.perf_counter()
        self.scores.see(memory, task)
        actions = self._native.decide(memory, task) if self._native is not None else self._slot(memory, task)
        self._info["lre_seconds"] = round(time.perf_counter() - started, 6)
        return actions

    def _slot(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        self._info = {}
        cleared = [item.id for item in memory.active if not item.pinned and item.id != task.observation.id]
        actions = [MemoryAction(Operation.MOVE_TO_ARCHIVE, tuple(cleared), parameters={"method": "keep_none"})] \
            if cleared and self.allows(Operation.MOVE_TO_ARCHIVE) else []
        if not (self.allows(Operation.RETRIEVE_FROM_ARCHIVE) and task.observation.requires_response and memory.archived):
            return actions
        query = task.observation.content
        shortlist = [item for item, _ in self.retriever.search(query, memory.archived, self.retrieve_candidates)]
        logits = {item.id: self.scores.logit(item) for item in shortlist}
        # The k best by LRE's logit; ties keep the BM25 order (sorted is stable).
        chosen = sorted(shortlist, key=lambda item: -logits[item.id])[: self.select_k]
        if not chosen:
            return actions
        actions.append(MemoryAction(Operation.RETRIEVE_FROM_ARCHIVE, tuple(item.id for item in chosen),
                                    parameters={"method": "lre_slot", "query": query}))
        self._info = {"candidates": len(shortlist), "selected_ids": [item.id for item in chosen],
                      "logits": {k: round(v, 4) for k, v in logits.items()}}
        return actions

    def decision_info(self) -> dict:
        if self._native is not None:
            return {**self._native.decision_info(), **{k: v for k, v in self._info.items() if k == "lre_seconds"}}
        return self._info


class _NativeLRE(Fifo):
    """The `fifo` controller with LRE's logit as the keep-priority; the question is never removed."""

    name = "lre_native"

    def __init__(self, config: dict, seed: int, owner: LREController) -> None:
        super().__init__(config, seed)
        self.owner = owner

    def allows(self, operation: Operation) -> bool:
        return self.owner.allows(operation)

    def priority(self, item, memory, task):
        if item.id == task.observation.id and task.observation.requires_response:
            return float("inf")
        return self.owner.scores.logit(item)
