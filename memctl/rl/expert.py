"""Expert labels from hindsight, for imitation (behaviour cloning, DAgger).

The expert is the approximate oracle's rule, expressed on the RL controller's
own decision: remove what is never needed again first, then what is needed
furthest in the future; retrieve what is needed right now. Never-needed items
are deleted; still-needed items are archived when the experiment allows
archiving and retrieval, and deleted otherwise.
"""

from __future__ import annotations

import numpy as np

from memctl.controllers.rl import Decision
from memctl.hindsight.collect import NEVER, Hindsight
from memctl.memory.actions import Operation
from memctl.rl.policy import REMOVAL_OPERATIONS

EVICT = REMOVAL_OPERATIONS.index(Operation.EVICT)
ARCHIVE = REMOVAL_OPERATIONS.index(Operation.MOVE_TO_ARCHIVE)


def make_expert(hindsight: Hindsight):
    def expert(decision: Decision) -> tuple[np.ndarray, list[int]]:
        step, n = decision.step, decision.n_active
        next_need = [min(hindsight.next_need(root, step) for root in roots) for roots in decision.roots]
        retrieved = [int(next_need[n + j] == step) for j in range(len(decision.retrieve_tokens))]

        rank = np.full(decision.mask.shape, -1, dtype=np.int64)
        distinct = sorted({need for need in next_need[:n] if need != NEVER}, reverse=True)
        position = {need: 1 + order for order, need in enumerate(distinct)}  # furthest need -> rank 1
        for i in range(n):
            can_evict, can_archive = bool(decision.mask[i, EVICT]), bool(decision.mask[i, ARCHIVE])
            if next_need[i] == NEVER:
                column, value = (EVICT if can_evict else ARCHIVE), 0
            else:
                column, value = (ARCHIVE if can_archive else EVICT), position[next_need[i]]
            if decision.mask[i, column]:
                rank[i, column] = value
        return rank, retrieved

    return expert
