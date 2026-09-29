"""The exact hindsight plan, as a small integer program (solved with PuLP / CBC).

Question: which evidence items should stay in CONTEXT, and until when, so
that as many questions as possible have ALL their evidence in context when
they are asked? Rules: the kept items must fit in the budget at every question,
and an item that has left the context never comes back into it.

The program is small because only evidence items get variables, and questions
asked with no new items in between share one "epoch".
"""

from __future__ import annotations

import pulp

from memctl.benchmarks.types import Conversation


def question_epochs(conversation: Conversation) -> list[list[int]]:
    """Group question steps into epochs: runs of questions with no item arriving in between."""
    epochs: list[list[int]] = []
    previous_kind = None
    for step, event in enumerate(conversation.events, start=1):
        if event.kind == "question":
            if previous_kind == "question":
                epochs[-1].append(step)
            else:
                epochs.append([step])
        if event.kind in ("item", "question"):
            previous_kind = event.kind
    return epochs


def solve_keep_plan(conversation: Conversation, budget: int) -> tuple[dict[str, int], int]:
    """Return (keep_until, covered).

    `keep_until[item_id]` is the last step at which the item should still be in
    CONTEXT. Items not in the dictionary should not be kept. `covered` is the
    number of questions that have all their evidence in context.
    """
    items = {item.id: item for item in conversation.items()}
    epochs = question_epochs(conversation)
    epoch_of = {step: e for e, steps in enumerate(epochs) for step in steps}
    questions = [
        (step, event.question)
        for step, event in enumerate(conversation.events, start=1)
        if event.kind == "question" and event.question.evidence_ids
    ]

    program = pulp.LpProblem("oracle_keep_plan", pulp.LpMaximize)
    keep: dict[tuple[str, int], pulp.LpVariable] = {}  # (item, epoch) -> 1 if in context at that epoch
    last_useful_epoch: dict[str, int] = {}
    for step, question in questions:
        for item_id in question.evidence_ids:
            last_useful_epoch[item_id] = max(last_useful_epoch.get(item_id, -1), epoch_of[step])
    for item_id, last in last_useful_epoch.items():
        for e in range(last + 1):
            if items[item_id].arrival_step < epochs[e][0]:
                keep[item_id, e] = pulp.LpVariable(f"keep_{len(keep)}", cat="Binary")

    covered = {q.id: pulp.LpVariable(f"covered_{n}", cat="Binary") for n, (_, q) in enumerate(questions)}
    partial = []
    for step, question in questions:
        for item_id in question.evidence_ids:
            variable = keep.get((item_id, epoch_of[step]))
            if variable is None:  # evidence arrives after the question: cannot be covered
                program += covered[question.id] == 0
            else:
                program += covered[question.id] <= variable
                partial.append(variable)
    # Main goal: fully covered questions. Tie-break: as much single evidence as possible.
    program += pulp.lpSum(covered.values()) + 0.001 * pulp.lpSum(partial)

    for e in range(len(epochs)):
        in_context = [items[i].tokens * v for (i, epoch), v in keep.items() if epoch == e]
        if in_context:
            program += pulp.lpSum(in_context) <= budget  # the budget holds at every epoch
    for (item_id, e), variable in keep.items():
        if (item_id, e - 1) in keep:
            program += variable <= keep[item_id, e - 1]  # once out of context, never back in

    program.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    keep_until: dict[str, int] = {}
    for (item_id, e), variable in keep.items():
        if variable.value() and variable.value() > 0.5:
            keep_until[item_id] = max(keep_until.get(item_id, 0), epochs[e][-1])
    number_covered = sum(1 for v in covered.values() if v.value() and v.value() > 0.5)
    return keep_until, number_covered
