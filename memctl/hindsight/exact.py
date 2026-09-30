"""Exact hindsight plans, where they can be computed.

`solve_delete_only` answers: with only KEEP and EVICT, which evidence items
should stay in ACTIVE, and until when, so that as many queries as possible
have all their evidence when they are asked? It is an integer program (PuLP /
CBC). It is exact under three conditions, which hold for the delete-only
experiments: a deleted item never returns, items are not rewritten, and the
task model answers correctly whenever the evidence is in ACTIVE.

Other action sets need other solvers; register them in `SOLVERS`. With an
unlimited archive and a perfect retriever the just-in-time oracle in
controllers/oracle.py is already optimal whenever each query's evidence fits
in the budget, so no program is needed there.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from memctl.hindsight.collect import Hindsight


class OracleUnavailable(Exception):
    """No exact plan: the solver is missing, the instance is too large, or time ran out."""


@dataclass
class ExactPlan:
    keep_until: dict[str, int]  # item id -> last step at which it should be in ACTIVE
    covered: int  # queries with all their evidence in ACTIVE
    queries: int
    solver: str = "ilp_delete_only"


def solve_delete_only(
    hindsight: Hindsight, budget: int, time_limit_s: float = 60.0, max_variables: int = 50_000
) -> ExactPlan:
    try:
        import pulp
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise OracleUnavailable("PuLP is not installed") from error

    need_times = {item_id: steps for item_id, steps in hindsight.all_needs.items() if item_id in hindsight.items}
    if sum(len(steps) for steps in need_times.values()) > max_variables:
        raise OracleUnavailable(f"instance has more than {max_variables} variables")

    program = pulp.LpProblem("oracle_delete_only", pulp.LpMaximize)
    # keep[i][k] = 1 if item i is still in ACTIVE at its k-th need time (kept since it arrived).
    keep = {
        item_id: [pulp.LpVariable(f"x_{n}_{k}", cat="Binary") for k in range(len(steps))]
        for n, (item_id, steps) in enumerate(need_times.items())
    }
    for variables in keep.values():
        for earlier, later in zip(variables, variables[1:]):
            program += later <= earlier  # once evicted, never back

    covered = []
    for number, dependency in enumerate(hindsight.dependencies):
        y = pulp.LpVariable(f"y_{number}", cat="Binary")
        covered.append(y)
        for requirement in dependency.requirements:
            carriers = []
            for item_id in requirement.item_ids:
                steps = need_times.get(item_id)
                if steps and hindsight.items[item_id].step <= dependency.step:
                    carriers.append(keep[item_id][steps.index(dependency.step)])
            program += y <= pulp.lpSum(carriers)  # with no carrier this forces y to 0

    # The budget must hold at every step. Occupancy only rises when an evidence item
    # arrives and only falls after a need time, so those steps are the only ones to check.
    check_steps = sorted(
        {hindsight.items[item_id].step for item_id in need_times} | {s for steps in need_times.values() for s in steps}
    )
    for step in check_steps:
        present = []
        for item_id, steps in need_times.items():
            record = hindsight.items[item_id]
            position = bisect_left(steps, step)
            if record.step <= step and position < len(steps):
                present.append(record.tokens * keep[item_id][position])
        if present:
            program += pulp.lpSum(present) <= budget

    weight = sum(hindsight.items[i].tokens * len(v) for i, v in keep.items()) or 1
    tie_break = pulp.lpSum(hindsight.items[i].tokens * x for i, v in keep.items() for x in v)
    program += pulp.lpSum(covered) - tie_break / (10.0 * weight)  # most queries, then least memory

    program.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    if pulp.LpStatus[program.status] != "Optimal":
        raise OracleUnavailable(f"solver status: {pulp.LpStatus[program.status]}")

    keep_until = {}
    for item_id, variables in keep.items():
        kept = [need_times[item_id][k] for k, x in enumerate(variables) if x.value() and x.value() > 0.5]
        if kept:
            keep_until[item_id] = max(kept)
    solved = sum(1 for y in covered if y.value() and y.value() > 0.5)
    return ExactPlan(keep_until, solved, len(covered))


SOLVERS = {"delete_only": solve_delete_only}
