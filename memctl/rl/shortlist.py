"""Where does the evidence a query needs sit when the controller decides what to retrieve?

    python -m memctl.rl.shortlist --config configs/sweeps/d1_regret_expert_eval.yaml \\
        --label bc_regret_s0 --fraction 0.02 --episodes 100 --out runs/diag_shortlist

Replays the episodes of one controller from a sweep config (the scripted reader is
enough: the controller's behaviour does not depend on the reader's answers on the
recall task) and, at every query, records for each piece of evidence the query
needs (one per hop on two-hop queries): whether a carrier is active, archived or
gone, the rank of the best archived carrier in a lexical search of the whole
archive with the question text, and whether the controller retrieved it.

The controller's own shortlist is the top `retrieve_candidates` (8) of that
search, so a hop whose rank is beyond 8 cannot be retrieved whatever the policy
learns. Writes queries.jsonl (one row per query) and summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from memctl.harness.runner import Experiment
from memctl.memory.actions import Operation
from memctl.retrieval import LexicalRetriever
from memctl.sysinfo import collect_metadata
from memctl.util import merged

BUCKETS = ((1, 3, "1-3"), (4, 8, "4-8"), (9, 20, "9-20"), (21, 10**9, ">20"))


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    return next(name for low, high, name in BUCKETS if low <= rank <= high)


def sweep_config(path: str | Path, label: str, fraction: float, episodes: int) -> dict:
    sweep = yaml.safe_load(Path(path).read_text())
    controllers = [c for c in sweep["grid"]["controller"] if c.get("label", c["name"]) == label]
    if not controllers:
        raise ValueError(f"no controller labelled {label!r} in {path}")
    # The scripted reader replaces whatever task model the sweep used: no LLM calls are needed.
    return merged(sweep["base"], {"controller": controllers[0], "episodes": episodes,
                                  "agent": {"name": "scripted_reader", "noise": 0.0},
                                  "memory": {"budget": {"fraction": fraction}}})


def replay(config: dict, episodes: int) -> list[dict]:
    experiment = Experiment(config)
    controller = experiment.controller
    lexical = LexicalRetriever()
    rows: list[dict] = []
    state: dict = {}

    def roots(item) -> tuple[str, ...]:
        if hasattr(controller, "_root_ids"):
            return controller._root_ids(item)
        return tuple(item.derived_from_ids) or (item.id,)

    original_decide = controller.decide

    def decide(memory, task):
        dependency = state["dependencies"].get(task.observation.id)
        actions = original_decide(memory, task)
        if dependency is None:
            return actions
        retrieved = {i for action in actions if action.operation is Operation.RETRIEVE_FROM_ARCHIVE for i in action.target_ids}
        ranking = lexical.search(task.observation.content, memory.archived, len(memory.archived))
        rank_of = {item.id: number + 1 for number, (item, _) in enumerate(ranking)}
        hops = []
        for requirement in dependency.requirements:
            wanted = set(requirement.item_ids)
            active = any(wanted & set(roots(item)) for item in memory.active)
            carriers = [item for item in memory.archived if wanted & set(roots(item))]
            ranks = [rank_of.get(item.id) for item in carriers]
            ranks = [r for r in ranks if r is not None]
            hops.append({
                "where": "active" if active else ("archived" if carriers else "gone"),
                "rank": min(ranks) if ranks else None,
                "retrieved": any(item.id in retrieved for item in carriers),
            })
        rows.append({"seed": state["seed"], "step": memory.step, "query_id": dependency.query_id,
                     "category": dependency.category, "archive_size": len(memory.archived), "hops": hops})
        return actions

    controller.decide = decide
    for index in range(episodes):
        seed = experiment.seed + index
        hindsight = experiment.hindsight(seed)
        state["seed"] = seed
        state["dependencies"] = {d.query_id: d for d in hindsight.dependencies}
        experiment.run_episode(index=index, detail=False)
    return rows


def summarise(rows: list[dict]) -> dict:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for number, hop in enumerate(row["hops"]):
            key = f"{row['category']}/hop{number + 1}"
            if hop["where"] != "archived":
                table[key][hop["where"]] += 1
                continue
            name = bucket(hop["rank"])
            table[key][f"archived:{name}"] += 1
            table[key][f"archived:{name}:retrieved"] += int(hop["retrieved"])
    # The case that dominates Exp 9's failures: hop 1 in view (active or just retrieved), hop 2 archived
    # and not retrieved.
    second: Counter = Counter()
    for row in rows:
        first, last = row["hops"][0], row["hops"][-1]
        if len(row["hops"]) == 2 and (first["where"] == "active" or first["retrieved"]) \
                and last["where"] == "archived" and not last["retrieved"]:
            second[bucket(row["hops"][1]["rank"])] += 1
    return {"queries": len(rows), "by_hop": {k: dict(v) for k, v in sorted(table.items())},
            "hop2_only_missed_by_rank": dict(second)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank of needed evidence in the archive search, by hop.")
    parser.add_argument("--config", required=True, help="sweep config whose grid lists the controller")
    parser.add_argument("--label", required=True)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out) / f"{args.label}__fraction{args.fraction:g}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps({**collect_metadata(), "args": vars(args)}, indent=2))
    rows = replay(sweep_config(args.config, args.label, args.fraction, args.episodes), args.episodes)
    with (out / "queries.jsonl").open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
