"""A standard reranker in the §19 head's slot (EXPERIMENTS.md §32).

    controller: {name: rerank, label: lr_pointwise, retrieval_method: lexical, retrieve_floor: 8,
                 retrieve_candidates: 32, floor_head: true, keep_none: true,
                 scorer: {kind: pointwise, model: configs/sweeps/exp32/models/lr_f0.json}}

It is the `rl` controller (`fixed8`) with one change: the k best of the same BM25 shortlist are chosen by a scorer
from memctl/rerank.py instead of the head's retrieve logit. Everything else is the head's code path: `keep_none`
archives every kept item but the current turn at every step, the shortlist is the same `retrieve_candidates` BM25
hits, the features are the head's (built whether or not the scorer uses them), and the picks are retrieved in one
action, so picks too long for the budget are returned by the harness's enforce_budget and counted as forced
removals, exactly as for `fixed8`.

`decision_info` reports `select_ms`: the time to choose the k from the shortlist (the scorer, plus building the
features when the scorer uses them; the BM25 search, common to all arms, is excluded and reported as `search_ms`).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from memctl.controllers.rl import RLController
from memctl.memory.state import MemoryView
from memctl.rerank import Scorer, Stopwatch, top_k
from memctl.rl.policy import RETRIEVE_COLUMN
from memctl.task import TaskState


class RerankController(RLController):
    name = "rerank"

    def __init__(self, config: dict, seed: int = 0) -> None:
        if not (config.get("keep_none") and config.get("floor_head")):
            raise ValueError("the rerank controller is the head's slot: it needs keep_none: true and floor_head: true")
        if not config.get("scorer"):
            raise ValueError("the rerank controller needs a `scorer` block (memctl/rerank.py)")
        if config.get("k_mode", "fixed") != "fixed" or config.get("notes") or config.get("query_rewrite"):
            raise ValueError("the rerank controller supports only a fixed k, without notes or query rewriting")
        if config["scorer"].get("kind") == "head" and not config.get("checkpoint"):
            raise ValueError("scorer kind 'head' needs the head's `checkpoint`")
        super().__init__(config, seed)
        self.scorer = Scorer(config["scorer"])
        self.model_id = f"rerank:{self.scorer.kind}"
        self._memory: MemoryView | None = None
        self._query = ""
        self._select: dict = {}

    def decide(self, memory: MemoryView, task: TaskState):
        self._memory, self._query, self._select = memory, task.observation.content, {}
        features, search = Stopwatch(), Stopwatch()
        self.featurizer.items = features.wrap(type(self.featurizer).items.__get__(self.featurizer))
        self.retriever.search = search.wrap(type(self.retriever).search.__get__(self.retriever))
        try:
            actions = super().decide(memory, task)
        finally:
            del self.featurizer.items
            del self.retriever.search
        if self._select:
            score_s = self._select.pop("_score_s")
            selecting = score_s + (features.seconds if self.scorer.uses_features else 0.0)
            self._select.update(select_ms=round(1000 * selecting, 4), featurize_ms=round(1000 * features.seconds, 4),
                                search_ms=round(1000 * search.seconds, 4))
        return actions

    def _choose(self, decision) -> None:
        decision.value, decision.log_prob = 0.0, 0.0
        if not decision.retrieve_tokens:
            return
        n = decision.n_active  # 0 under keep_none
        started = time.perf_counter()
        items = [self._memory.get(item_id) for item_id in decision.item_ids[n:]]
        head_logits = None
        if self.scorer.kind == "head":
            with torch.no_grad():
                logits, _ = self.policy(torch.from_numpy(decision.items), torch.from_numpy(decision.global_features))
            head_logits = logits[n:, RETRIEVE_COLUMN].numpy()
        scores = self.scorer.score(self._query, items, decision.items[n:], decision.global_features, head_logits)
        k = min(self.retrieve_floor, len(items))
        self._order = top_k(scores, len(items))
        chosen = set(self._order[:k])
        decision.retrieved = [int(j in chosen) for j in range(len(items))]
        self._select = {"_score_s": time.perf_counter() - started,
                        "selected_ids": [items[j].id for j in self._order[:k]],
                        "scores": [round(float(s), 5) for s in np.asarray(scores)]}

    def decision_info(self) -> dict:
        info = super().decision_info()
        return {**info, **self._select} if self._select else info
