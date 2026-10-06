"""What a trained RL controller does with needed and never-needed items.

    python -m memctl.rl.probe runs/<training>/checkpoints/policy.pt [--episodes 20]

Runs the greedy policy on the evaluation seeds (0, 1, ...) at each budget, and
classifies every removal it makes by the operation chosen and by whether, in
hindsight, the item was needed again. This is a diagnostic: hindsight is used to
label the picks, never to make them.

If the learner cannot tell needed from never-needed items, imitation makes it
average the expert's action over the two (Weihs et al. 2021, Proposition 1),
so the deletion rate is about the same for both classes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from memctl.config import resolve
from memctl.harness.runner import Experiment
from memctl.hindsight.collect import NEVER
from memctl.rl.policy import REMOVAL_OPERATIONS

ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]


def probe(checkpoint: str, fractions: list[float], episodes: int, horizon: int = 200) -> dict:
    results = {}
    for fraction in fractions:
        config = resolve({
            "env": {"name": "synthetic_recall", "horizon": horizon},
            "agent": {"name": "scripted_reader", "noise": 0.0},
            "controller": {"name": "rl", "checkpoint": checkpoint},
            "memory": {"allowed_operations": ARCHIVE_OPS, "embedder": "hashing", "budget": {"fraction": fraction}},
        })
        experiment = Experiment(config)
        controller = experiment.controller
        controller.record = True
        counts: Counter = Counter()
        retrievals: Counter = Counter()
        success = 0.0
        for seed in range(episodes):
            hindsight = experiment.hindsight(seed)
            success += experiment.run_episode(seed=seed, detail=False).episode["task_success"]
            for decision in controller.recorded:
                need = [min(hindsight.next_need(root, decision.step) for root in roots) for roots in decision.roots]
                for pick in decision.picks:
                    item, op = pick // len(REMOVAL_OPERATIONS), REMOVAL_OPERATIONS[pick % len(REMOVAL_OPERATIONS)]
                    counts[("needed" if need[item] != NEVER else "never", op.value)] += 1
                for j, flag in enumerate(decision.retrieved):
                    due = need[decision.n_active + j] == decision.step
                    retrievals[("due" if due else "not_due", "retrieved" if flag else "skipped")] += 1
        row = {"success": success / episodes}
        for kind in ("needed", "never"):
            total = sum(v for (k, _), v in counts.items() if k == kind)
            row[f"{kind}_removals"] = total
            row[f"{kind}_evict_rate"] = counts[(kind, "EVICT")] / total if total else None
        due = retrievals[("due", "retrieved")] + retrievals[("due", "skipped")]
        hits = retrievals[("due", "retrieved")] + retrievals[("not_due", "retrieved")]
        row["retrieval_recall_of_shortlist"] = retrievals[("due", "retrieved")] / due if due else None
        row["retrieval_precision"] = retrievals[("due", "retrieved")] / hits if hits else None
        row["retrievals_per_episode"] = hits / episodes
        results[f"{fraction:g}"] = row
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25])
    args = parser.parse_args()
    print(json.dumps(probe(args.checkpoint, args.fractions, args.episodes), indent=2))


if __name__ == "__main__":
    main()
