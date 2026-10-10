"""Supervised rows for the §32 rerankers, from the TRAINING part of a fold only (EXPERIMENTS.md §32).

    python -m memctl.rerank_data --fold 0 --part train_a --out runs/exp32_rerankers/data
    python -m memctl.rerank_data --fold 0 --part train_b --out runs/exp32_rerankers/data

- `train_a` (the fitting rows): the §19 head A's own training setup (configs/rl/lme_n4/head_f{k}_a.yaml): composed
  4-question episodes from half A of the fold's training part, budget fraction 0.0125, the regret expert
  (retrieval_risk 0.36), seeds 100000 to 100249 (the head's 10 iterations x 25 episodes). DAgger-free: the expert
  drives every episode (the head's first iteration), so the rows do not depend on any learner.
- `train_b` (the inner validation rows): every question of half B of the fold's training part once, as the test
  folds are played: one question per episode, budget fraction 0.05. Head A never trained on half B.

One record per question decision: the 32 candidates' 24 item features and the 6 global features (the head's
inputs, memctl/features.py), the expert's label per candidate (the designated carrier of a requirement needed at
this question: memctl/rl/expert.py and Hindsight.designated), for each requirement the candidates that would
satisfy it (any alternative carrier, as the harness counts evidence), the question and candidate texts (plain, and labelled as the dense search sees them), and token
counts. Nothing here reads the fold's test part: the environment is built with `part: train_a|train_b`.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from memctl.harness.runner import Experiment
from memctl.retrieval import labelled_text
from memctl.rl.expert import make_expert
from memctl.runlog import peak_rss_mb
from memctl.splits import fold_indices
from memctl.envs.longmemeval import _index

HEAD_CONFIG = "configs/rl/lme_n4/head_f{fold}_a.yaml"
PARTS = ("train_a", "train_b")


def part_config(fold: int, part: str, data_path: str | None = None) -> dict:
    config = yaml.safe_load(Path(HEAD_CONFIG.format(fold=fold)).read_text())
    config = copy.deepcopy(config)
    config["env"]["folds"] = {"k": 5, "fold": fold, "part": part}
    if data_path:
        config["env"]["path"] = data_path
    if part == "train_b":
        config["env"]["compose"] = 1
    config.pop("training", None)
    config["controller"]["retrieve_floor"] = 8
    config["memory"]["budget"] = {"fraction": 0.0125 if part == "train_a" else 0.05}
    return config


def collect(fold: int, part: str, episodes: int | None = None, data_path: str | None = None,
            progress: bool = False) -> dict:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS} (never the test part)")
    config = part_config(fold, part, data_path)
    experiment = Experiment(config)
    experiment.config["hindsight"]["enabled"] = False
    controller = experiment.controller
    controller.record, controller.greedy, controller.follow_expert = True, False, True
    if part == "train_a":
        seeds = list(range(100_000, 100_000 + (episodes or 250)))
    else:
        size = len(fold_indices(_index(config["env"]["path"]), 5, fold, part))
        seeds = list(range(episodes or size))
    context: dict = {}
    original = controller.decide

    def decide(memory, task):  # keep the memory view, to read candidate texts
        context["memory"], context["query"] = memory, task.observation.content
        before = len(controller.recorded)
        actions = original(memory, task)
        for decision in controller.recorded[before:]:
            pool = [memory.get(i) for i in decision.item_ids[decision.n_active:]]
            decision.texts = (task.observation.content, [item.content for item in pool],
                              [labelled_text(item) for item in pool])
        return actions

    controller.decide = decide
    records, started = [], time.time()
    for number, seed in enumerate(seeds):
        hindsight = experiment.hindsight(seed)
        controller.expert = make_expert(hindsight, kind="regret", retrieval_risk=0.36)
        episode = experiment.run_episode(seed=seed, detail=False).episode
        diverged = episode.get("hindsight_diverged_at")
        by_step = {d.step: d for d in hindsight.dependencies}
        for decision in controller.recorded:
            if diverged is not None and decision.step >= diverged:
                continue
            n = decision.n_active
            dependency = by_step.get(decision.step)
            roots = decision.roots[n:]
            requirements = []
            for requirement in (dependency.requirements if dependency else ()):
                wanted = set(requirement.item_ids)
                requirements.append([j for j, r in enumerate(roots) if wanted & set(r)])
            evidence = sorted({j for found in requirements for j in found})
            query_item = experiment.env._questions.get(dependency.query_id) if dependency else None
            records.append({
                "fold": fold, "part": part, "seed": seed, "step": decision.step,
                "question_id": getattr(query_item, "id", None) or (dependency.query_id if dependency else None),
                "question_type": getattr(query_item, "category", None),
                "items": decision.items[n:].astype(np.float32), "globals": decision.global_features.astype(np.float32),
                "label": np.asarray(decision.expert_retrieved, dtype=np.int8),
                "requirements": requirements, "n_requirements": len(requirements), "evidence": evidence,
                "tokens": list(decision.retrieve_tokens), "item_ids": list(decision.item_ids[n:]),
                "query": decision.texts[0], "texts": decision.texts[1], "labelled": decision.texts[2],
            })
        controller.recorded = []
        experiment._hindsight.pop(seed, None)
        if progress and (number + 1) % 25 == 0:
            print(f"fold {fold} {part}: {number + 1}/{len(seeds)} episodes, {len(records)} decisions, "
                  f"{time.time() - started:.0f} s", flush=True)
    return {"records": records, "seconds": round(time.time() - started, 1), "peak_rss_mb": peak_rss_mb(),
            "config": config, "seeds": [seeds[0], seeds[-1]]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect §32 supervised rows from a fold's training part.")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="runs/exp32_rerankers/data")
    args = parser.parse_args()
    result = collect(args.fold, args.part, args.episodes, args.data, progress=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"f{args.fold}_{args.part}.pkl"
    with open(path, "wb") as handle:
        pickle.dump(result, handle)
    records = result["records"]
    print(f"{path}: {len(records)} decisions, {sum(len(r['label']) for r in records)} rows, "
          f"{sum(int(r['label'].sum()) for r in records)} positive; {result['seconds']} s; "
          f"peak RSS {result['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
