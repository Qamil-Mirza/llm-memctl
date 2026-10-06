"""The RL controller: a learned scorer decides what to remove and what to retrieve.

The decision is shaped so that it is learnable and budget-consistent:

- When memory is over budget, the policy picks (item, operation) pairs one at a
  time, without replacement, until memory fits. Each pick is a draw from a
  softmax over every remaining candidate and every allowed removal operation.
- When the agent is about to respond and the archive is not empty, the policy
  also decides, for each of a short list of archive candidates, whether to
  retrieve it.
- Otherwise it does nothing.

This file only *acts*. How the policy is trained (imitation, REINFORCE, PPO, or
anything else) lives in memctl/rl/ and talks to the controller through two
hooks: `recorded` (the decisions of the episode, with everything needed to
recompute their probabilities) and `expert` (an optional function that supplies
the picks to execute or to learn from).

    controller: {name: rl, checkpoint: runs/<id>/checkpoints/policy.pt}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from memctl.controllers.base import EpisodeInfo, Feedback, MemoryController
from memctl.features import FEATURE_VERSION, GLOBAL_DIM, Featurizer
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.items import Fidelity, MemoryItem
from memctl.memory.state import MemoryView
from memctl.retrieval import build_retriever
from memctl.rl.policy import N_COLUMNS, REMOVAL_OPERATIONS, RETRIEVE_COLUMN, ItemPolicy, NeededModel
from memctl.task import TaskState

N_REMOVALS = len(REMOVAL_OPERATIONS)
# The policy is a few thousand parameters scoring a few hundred rows: threads only add overhead,
# and several processes each starting a thread pool slow each other down.
torch.set_num_threads(1)


@dataclass
class Decision:
    """One decision, with everything needed to recompute its probability under another policy."""

    step: int
    items: np.ndarray  # [rows, item_dim]: ACTIVE candidates first, then archive candidates
    global_features: np.ndarray
    item_ids: list[str]
    roots: list[tuple[str, ...]]  # the original observations behind each row (for expert labels)
    n_active: int
    savings: np.ndarray  # [n_active, N_REMOVALS] tokens freed by each operation on each item
    mask: np.ndarray  # [n_active, N_REMOVALS] which (item, operation) pairs are available
    excess: int  # tokens over budget before any retrieval
    retrieve_tokens: list[int] = field(default_factory=list)  # size of each archive candidate
    picks: list[int] = field(default_factory=list)  # flat indices item * N_REMOVALS + operation, in order
    retrieved: list[int] = field(default_factory=list)  # 1 or 0 per archive candidate
    log_prob: float = 0.0
    value: float = 0.0
    return_to_go: float = 0.0  # filled in by the trainer
    # Expert labels, when a trainer supplies them. `expert_rank[i, op]` is the order in which an
    # expert would remove item i by that operation: 0 first, ties allowed, -1 for "not this way".
    expert_rank: np.ndarray | None = None
    expert_retrieved: list[int] | None = None
    # Optional: the expert's cost of removing item i by that operation (NaN where unavailable),
    # for cost-sensitive imitation. Lower is better; `expert_rank` orders the same pairs.
    expert_cost: np.ndarray | None = None


def accepted_picks(rank: np.ndarray, mask: torch.Tensor) -> list[int]:
    """The flat indices the expert would accept next: available pairs with the lowest rank."""
    valid = mask.numpy() & (rank >= 0)
    if not valid.any():
        return []
    best = rank[valid].min()
    return np.flatnonzero((valid & (rank == best)).flatten()).tolist()


def evaluate(policy: ItemPolicy, decision: Decision) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(log-probability of the recorded picks, value, entropy of the first choice) under `policy`."""
    logits, value = policy(torch.from_numpy(decision.items), torch.from_numpy(decision.global_features))
    log_prob = logits.new_zeros(())
    entropy = logits.new_zeros(())
    n = decision.n_active
    if decision.retrieved:
        distribution = torch.distributions.Bernoulli(logits=logits[n:, RETRIEVE_COLUMN])
        log_prob = log_prob + distribution.log_prob(torch.tensor(decision.retrieved, dtype=torch.float32)).sum()
        entropy = entropy + distribution.entropy().sum()
    if decision.picks:
        mask = torch.from_numpy(decision.mask).clone()
        for number, pick in enumerate(decision.picks):
            flat = logits[:n, :N_REMOVALS].masked_fill(~mask, -1e9).flatten()
            log_probs = torch.log_softmax(flat, dim=0)
            log_prob = log_prob + log_probs[pick]
            if number == 0:
                entropy = entropy - (log_probs.exp() * log_probs).sum()
            mask[pick // N_REMOVALS] = False
    return log_prob, value, entropy


class RLController(MemoryController):
    name = "rl"

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.greedy = bool(config.get("greedy", True))
        self.retrieve_candidates = int(config.get("retrieve_candidates", 8))
        self.min_compact_tokens = int(config.get("min_compact_tokens", 30))
        self.compact_ratio = float(config.get("compact_ratio", 0.5))
        self.log_features = bool(config.get("log_features", False))
        self.featurizer = Featurizer(bool(config.get("use_embeddings", False)), int(config.get("embedding_dim", 0)))
        self.policy = ItemPolicy(
            self.featurizer.item_dim, GLOBAL_DIM, int(config.get("hidden", 64)), config.get("architecture", "deepsets")
        )
        self.model_id = f"item-policy-{self.policy.architecture}-h{self.policy.config['hidden']}"
        self.generator = torch.Generator().manual_seed(seed)
        self.retriever = None
        # Hooks for trainers (memctl/rl/). Ordinary runs leave them alone.
        self.record = False
        self.recorded: list[Decision] = []
        self.rewards: list[tuple[int, float]] = []
        self.expert: Callable[[Decision], tuple] | None = None  # (rank, retrieved[, cost])
        self.follow_expert = False
        self._roots: dict[str, tuple[str, ...]] = {}
        self._info: dict = {}
        # Optional plug-in Bayes rule for a priced archive (memctl/rl/needed.py): the policy picks which
        # item to remove; the item is archived iff P(needed again) > archive_price, deleted otherwise.
        self.needed_model = NeededModel.load(config["needed_model"]) if config.get("needed_model") else None
        self.archive_price = float(config.get("archive_price", 0.0))
        if config.get("checkpoint"):
            self.load(config["checkpoint"])

    @property
    def display_name(self) -> str:
        return self.config.get("label") or "rl"

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self.featurizer.reset()
        self.recorded, self.rewards, self._roots = [], [], {}
        self.generator.manual_seed(hash((self.seed, episode.seed)) % (2**31))
        self.retriever = build_retriever(self.config.get("retrieval_method", "lexical"), episode.embedder)

    def update(self, feedback: Feedback) -> None:
        if self.record:
            self.rewards.append((feedback.step, feedback.reward))

    # ---- deciding -------------------------------------------------------------------

    def _root_ids(self, item: MemoryItem) -> tuple[str, ...]:
        if item.id not in self._roots:
            if item.derived_from_ids:
                found: list[str] = []
                for source in item.derived_from_ids:
                    found.extend(self._roots.get(source, (source,)))
                self._roots[item.id] = tuple(dict.fromkeys(found))
            else:
                self._roots[item.id] = (item.id,)
        return self._roots[item.id]

    def _saving(self, item: MemoryItem, operation: Operation) -> int:
        if not self.allows(operation):
            return 0
        if operation in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE):
            if item.fidelity is not Fidelity.FULL or item.token_count < self.min_compact_tokens:
                return 0
            return item.token_count - math.ceil(self.compact_ratio * item.token_count)
        return item.token_count

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        self._info = {}
        for item in memory.active:  # keep lineage up to date even on steps with no decision
            self._root_ids(item)
        shortlist: list[tuple[MemoryItem, float]] = []
        if self.allows(Operation.RETRIEVE_FROM_ARCHIVE) and task.observation.requires_response and memory.archived:
            shortlist = self.retriever.search(task.observation.content, memory.archived, self.retrieve_candidates)
        if not memory.over_budget and not shortlist:
            return []

        active = [item for item in memory.active if not item.pinned] if memory.over_budget or shortlist else []
        candidates = active + [item for item, _ in shortlist]
        top = max((score for _, score in shortlist), default=1.0) or 1.0
        scores = {item.id: score / top for item, score in shortlist}
        savings = np.array([[self._saving(item, op) for op in REMOVAL_OPERATIONS] for item in active], dtype=np.int64)
        savings = savings.reshape(len(active), N_REMOVALS)
        decision = Decision(
            step=memory.step,
            items=self.featurizer.items(candidates, memory, task, scores),
            global_features=self.featurizer.globals(memory, task),
            item_ids=[item.id for item in candidates],
            roots=[self._root_ids(item) for item in candidates],
            n_active=len(active),
            savings=savings,
            mask=savings > 0,
            excess=memory.active_tokens - memory.budget,
            retrieve_tokens=[item.token_count for item, _ in shortlist],
        )
        if self.expert is not None:
            labels = self.expert(decision)
            decision.expert_rank, decision.expert_retrieved = labels[0], labels[1]
            decision.expert_cost = labels[2] if len(labels) > 2 else None
        self._choose(decision)
        if self.record:
            self.recorded.append(decision)

        actions = []
        retrieved = [candidates[decision.n_active + j].id for j, flag in enumerate(decision.retrieved) if flag]
        if retrieved:
            actions.append(
                MemoryAction(
                    Operation.RETRIEVE_FROM_ARCHIVE, tuple(retrieved),
                    parameters={"query": task.observation.content, "method": f"rl+{self.retriever.name}"},
                )
            )
        for pick in decision.picks:
            item, operation = active[pick // N_REMOVALS], REMOVAL_OPERATIONS[pick % N_REMOVALS]
            parameters = {"ratio": self.compact_ratio} if operation in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE) else {}
            actions.append(MemoryAction(operation, (item.id,), parameters=parameters))
        self._info = {
            "candidates": len(candidates),
            "picks": [(decision.item_ids[p // N_REMOVALS], REMOVAL_OPERATIONS[p % N_REMOVALS].value) for p in decision.picks],
            "log_prob": decision.log_prob,
            "value": decision.value,
        }
        if self.log_features:
            self._info["features"] = decision.items.round(4).tolist()
        return actions

    def _choose(self, decision: Decision) -> None:
        """Fill in `picks`, `retrieved`, `log_prob` and `value`, from the policy or from the expert."""
        with torch.no_grad():
            logits, value = self.policy(torch.from_numpy(decision.items), torch.from_numpy(decision.global_features))
        decision.value = float(value)
        n = decision.n_active
        log_prob = 0.0
        excess = decision.excess

        if decision.retrieve_tokens:
            retrieve_logits = logits[n:, RETRIEVE_COLUMN]
            if self.follow_expert and decision.expert_retrieved is not None:
                chosen = torch.tensor(decision.expert_retrieved, dtype=torch.float32)
            elif self.greedy:
                chosen = (retrieve_logits > 0).float()
            else:
                chosen = torch.bernoulli(torch.sigmoid(retrieve_logits), generator=self.generator)
            log_prob += float(torch.distributions.Bernoulli(logits=retrieve_logits).log_prob(chosen).sum())
            decision.retrieved = [int(flag) for flag in chosen.tolist()]
            excess += sum(tokens for tokens, flag in zip(decision.retrieve_tokens, decision.retrieved) if flag)

        mask = torch.from_numpy(decision.mask).clone()
        savings = decision.savings
        needed = None
        if self.needed_model is not None and n:
            rows = np.concatenate([decision.items[:n], np.repeat(decision.global_features[None, :], n, axis=0)], axis=1)
            with torch.no_grad():
                needed = torch.sigmoid(self.needed_model(torch.from_numpy(rows))).numpy()
        while excess > 0 and bool(mask.any()):
            flat = logits[:n, :N_REMOVALS].masked_fill(~mask, -1e9).flatten()
            log_probs = torch.log_softmax(flat, dim=0)
            accepted = []
            if self.follow_expert and decision.expert_rank is not None:
                accepted = accepted_picks(decision.expert_rank, mask)
            if accepted:  # the expert's choice; among equals, the one the policy likes best
                pick = max(accepted, key=lambda p: float(log_probs[p]))
            elif self.greedy:
                pick = int(torch.argmax(flat))
            else:
                pick = int(torch.multinomial(log_probs.exp(), 1, generator=self.generator))
            if needed is not None and not accepted:
                i, evict, archive = pick // N_REMOVALS, REMOVAL_OPERATIONS.index(Operation.EVICT), REMOVAL_OPERATIONS.index(Operation.MOVE_TO_ARCHIVE)
                if pick % N_REMOVALS in (evict, archive) and bool(mask[i, evict]) and bool(mask[i, archive]):
                    pick = i * N_REMOVALS + (archive if needed[i] > self.archive_price else evict)
            log_prob += float(log_probs[pick])
            decision.picks.append(pick)
            mask[pick // N_REMOVALS] = False
            excess -= int(savings[pick // N_REMOVALS, pick % N_REMOVALS])
        decision.log_prob = log_prob

    def decision_info(self) -> dict:
        return self._info

    # ---- checkpoints ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "policy": self.policy.config,
                "feature_version": FEATURE_VERSION,
                "use_embeddings": self.featurizer.use_embeddings,
                "embedding_dim": self.featurizer.embedding_dim,
                "columns": N_COLUMNS,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), weights_only=True)
        if checkpoint["feature_version"] != FEATURE_VERSION:
            raise ValueError(
                f"checkpoint was trained on feature version {checkpoint['feature_version']}, "
                f"this code computes version {FEATURE_VERSION}"
            )
        self.featurizer = Featurizer(checkpoint["use_embeddings"], checkpoint["embedding_dim"])
        settings = checkpoint["policy"]
        self.policy = ItemPolicy(settings["item_dim"], settings["global_dim"], settings["hidden"], settings["architecture"])
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.model_id = f"item-policy-{settings['architecture']}-h{settings['hidden']}"
