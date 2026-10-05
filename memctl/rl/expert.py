"""Expert labels from hindsight, for imitation (behaviour cloning, DAgger).

Two experts, chosen with `training.expert.kind`:

- `oracle` (default): the approximate oracle's rule, expressed on the RL
  controller's own decision: remove what is never needed again first, then what
  is needed furthest in the future; retrieve what is needed right now.
  Never-needed items are deleted; still-needed items are archived when the
  experiment allows archiving and retrieval, and deleted otherwise.

- `regret`: every (item, operation) pair is charged the hindsight regret of
  taking it with the oracle playing on, and the expert accepts *every* pair of
  least regret. With a free archive and just-in-time retrieval, archiving costs
  nothing whether or not the item is needed, and deleting costs the evidence it
  destroys, so for a never-needed item both are accepted. The `oracle` expert
  breaks that tie towards deletion, which is invisible to a learner that cannot
  tell never-needed from needed items: it learns to delete both (Experiment 4b).
  Ties within a regret level are ordered by next need, furthest first, as the
  oracle orders them.

      regret(i, EVICT)   = 1 if i is needed again, else 0
      regret(i, ARCHIVE) = archive_cost + retrieval_risk * (1 if i is needed again, else 0)

  `archive_cost` prices the archive; `retrieval_risk` is the chance the item,
  once archived, is not retrieved when needed (0 with the oracle playing on).
"""

from __future__ import annotations

import numpy as np

from memctl.controllers.rl import Decision
from memctl.hindsight.collect import NEVER, Hindsight
from memctl.memory.actions import Operation
from memctl.rl.policy import REMOVAL_OPERATIONS

EVICT = REMOVAL_OPERATIONS.index(Operation.EVICT)
ARCHIVE = REMOVAL_OPERATIONS.index(Operation.MOVE_TO_ARCHIVE)
KINDS = ("oracle", "regret")


def make_expert(hindsight: Hindsight, kind: str = "oracle", archive_cost: float = 0.0, retrieval_risk: float = 0.0):
    if kind not in KINDS:
        raise ValueError(f"unknown expert kind {kind!r}; expected one of {KINDS}")

    def needs(decision: Decision) -> tuple[list[int], list[int]]:
        step, n = decision.step, decision.n_active
        next_need = [min(hindsight.next_need(root, step) for root in roots) for roots in decision.roots]
        retrieved = [int(next_need[n + j] == step) for j in range(len(decision.retrieve_tokens))]
        return next_need, retrieved

    def oracle(decision: Decision) -> tuple[np.ndarray, list[int]]:
        next_need, retrieved = needs(decision)
        rank = np.full(decision.mask.shape, -1, dtype=np.int64)
        distinct = sorted({need for need in next_need[: decision.n_active] if need != NEVER}, reverse=True)
        position = {need: 1 + order for order, need in enumerate(distinct)}  # furthest need -> rank 1
        for i in range(decision.n_active):
            can_evict, can_archive = bool(decision.mask[i, EVICT]), bool(decision.mask[i, ARCHIVE])
            if next_need[i] == NEVER:
                column, value = (EVICT if can_evict else ARCHIVE), 0
            else:
                column, value = (ARCHIVE if can_archive else EVICT), position[next_need[i]]
            if decision.mask[i, column]:
                rank[i, column] = value
        return rank, retrieved

    def regret(decision: Decision) -> tuple[np.ndarray, list[int]]:
        next_need, retrieved = needs(decision)
        keys: dict[tuple[int, int], tuple[float, int]] = {}
        for i in range(decision.n_active):
            needed = next_need[i] != NEVER
            for column, cost in ((EVICT, float(needed)), (ARCHIVE, archive_cost + retrieval_risk * needed)):
                if decision.mask[i, column]:
                    keys[(i, column)] = (round(cost, 9), -next_need[i])
        order = {key: position for position, key in enumerate(sorted(set(keys.values())))}
        rank = np.full(decision.mask.shape, -1, dtype=np.int64)
        for (i, column), key in keys.items():
            rank[i, column] = order[key]
        return rank, retrieved

    return oracle if kind == "oracle" else regret
