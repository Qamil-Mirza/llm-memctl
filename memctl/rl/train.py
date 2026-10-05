"""Train the RL controller:  python -m memctl.rl.train --config configs/rl/x.yaml

The config is an ordinary experiment config (environment, agent, memory,
reward, `controller: {name: rl, ...}`) plus a `training` section:

    training:
      seed_offset: 100000            # training episodes use seeds from here on; evaluation uses 0, 1, 2, ...
      budget_fractions: [0.05, 0.1, 0.25, 0.5]   # optional: a budget is drawn per episode
      phases:
        - {algorithm: bc, iterations: 8, episodes: 16, epochs: 4, lr: 0.003, dagger: true}
        - {algorithm: ppo, iterations: 40, episodes: 16, epochs: 4, lr: 0.0003, gamma: 0.995}
      eval: {every: 4, episodes: 20, seed_offset: 50000}
      expert: {kind: regret}         # optional: which hindsight expert labels imitation (memctl/rl/expert.py)

Three disjoint sets of seeds keep the numbers honest: training episodes use
`seed_offset` onward, the evaluations logged during training (and used to pick
`policy_best.pt`) use `eval.seed_offset` onward, and experiments that compare
the trained policy with baselines use 0, 1, 2, ... as every other run does.

Phases run in order on the same policy, so "imitation only", "RL only" and
"imitation then RL" are three configs. The folder it writes looks like a run
folder: config.yaml, metadata.json, train_log.jsonl, summary.json and
checkpoints/policy.pt (the last policy) and checkpoints/policy_best.pt (the one
with the best validation success), which `controller: {name: rl, checkpoint: ...}` loads.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import yaml

from memctl.config import resolve
from memctl.controllers.rl import Decision, RLController
from memctl.harness.runner import Experiment
from memctl.rl.algorithms import ALGORITHMS
from memctl.rl.expert import make_expert
from memctl.sysinfo import collect_metadata

TRAINING_DEFAULTS = {
    "seed_offset": 100_000,
    "budget_fractions": None,
    "phases": [{"algorithm": "bc", "iterations": 8, "episodes": 16}],
    "eval": {"every": 4, "episodes": 20, "seed_offset": 50_000},
    "max_dataset": 40_000,
    "expert": {"kind": "oracle"},
}


def fill_returns(decisions: list[Decision], rewards: list[tuple[int, float]], gamma: float) -> None:
    """Set each decision's `return_to_go`: the discounted sum of rewards from its step onward."""
    by_step = dict(rewards)
    if not by_step:
        return
    last = max(by_step)
    running, value_at = 0.0, {}
    for step in range(last, 0, -1):
        running = by_step.get(step, 0.0) + gamma * running
        value_at[step] = running
    for decision in decisions:
        decision.return_to_go = value_at.get(decision.step, 0.0)


def evaluate_policy(experiment: Experiment, episodes: int, fractions: list[float] | None, seed_offset: int = 50_000) -> dict:
    """Mean task success of the greedy policy on the validation seeds (seed_offset, seed_offset + 1, ...)."""
    controller: RLController = experiment.controller
    saved = (controller.greedy, controller.record, controller.expert, controller.follow_expert)
    saved_budget = copy.deepcopy(experiment.config["memory"]["budget"])
    controller.greedy, controller.record, controller.expert, controller.follow_expert = True, False, None, False
    results = {}
    try:
        for fraction in fractions or [None]:
            if fraction is not None:
                experiment.config["memory"]["budget"] = {"fraction": fraction}
            seeds = range(seed_offset, seed_offset + episodes)
            values = [experiment.run_episode(seed=seed, detail=False).episode["task_success"] for seed in seeds]
            for seed in seeds:
                experiment._hindsight.pop(seed, None)
            results["success" if fraction is None else f"success@{fraction:g}"] = sum(values) / len(values)
    finally:
        controller.greedy, controller.record, controller.expert, controller.follow_expert = saved
        experiment.config["memory"]["budget"] = saved_budget
    return results


def train(config: dict, folder: str | Path | None = None, progress: bool = False) -> Path:
    config = resolve(config)
    if config["controller"]["name"] != "rl":
        raise ValueError("training needs controller.name: rl")
    settings = {**TRAINING_DEFAULTS, **config.get("training", {})}
    settings["eval"] = {**TRAINING_DEFAULTS["eval"], **settings["eval"]}
    config["training"] = settings
    folder = Path(folder) if folder else Path(config["logging"]["output_dir"]) / config["name"]
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(1)
    rng = random.Random(int(config["seed"]))
    experiment = Experiment(config)
    controller: RLController = experiment.controller
    controller.record = True
    metadata = {**collect_metadata(), "models": experiment.models(), "status": "running",
                "parameters": sum(p.numel() for p in controller.policy.parameters())}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log_path = folder / "train_log.jsonl"
    log_path.write_text("")

    next_seed = int(settings["seed_offset"]) + int(config["seed"])
    fractions = settings["budget_fractions"]
    started, episodes_run = time.time(), 0
    eval_episodes, eval_offset = int(settings["eval"]["episodes"]), int(settings["eval"]["seed_offset"])
    best: dict = {"validation": -1.0}
    for phase_number, phase in enumerate(settings["phases"]):
        name = phase["algorithm"]
        algorithm = ALGORITHMS[name](phase)
        optimizer = torch.optim.Adam(controller.policy.parameters(), lr=float(phase.get("lr", 3e-3 if name == "bc" else 3e-4)))
        imitation = name in ("bc", "cost")
        # Regret tracking is only worth its cost when the expert labels need hindsight anyway.
        experiment.config["hindsight"]["enabled"] = False
        dataset: list[Decision] = []
        for iteration in range(int(phase.get("iterations", 8))):
            fresh: list[Decision] = []
            successes, forced = [], 0
            for _ in range(int(phase.get("episodes", 16))):
                seed, next_seed = next_seed, next_seed + 1
                if fractions:
                    experiment.config["memory"]["budget"] = {"fraction": rng.choice(fractions)}
                controller.greedy = False
                if imitation:
                    controller.expert = make_expert(experiment.hindsight(seed), **settings["expert"])
                    # DAgger: the expert drives the first iteration, the learner the rest, the expert labels all.
                    controller.follow_expert = iteration == 0 or not phase.get("dagger", True)
                else:
                    controller.expert, controller.follow_expert = None, False
                episode = experiment.run_episode(seed=seed, detail=False).episode
                experiment._hindsight.pop(seed, None)
                recorded = controller.recorded
                if imitation and episode.get("hindsight_diverged_at") is not None:
                    # From the divergence on, the expert's labels describe another episode.
                    recorded = [d for d in recorded if d.step < episode["hindsight_diverged_at"]]
                if not imitation:
                    fill_returns(controller.recorded, controller.rewards, float(phase.get("gamma", 0.995)))
                fresh += recorded
                successes.append(episode["task_success"])
                forced += episode["forced_evictions"]
                episodes_run += 1
            if imitation:
                dataset = (dataset + fresh)[-int(settings["max_dataset"]):]
                stats = algorithm.update(controller.policy, optimizer, dataset, rng)
            else:
                stats = algorithm.update(controller.policy, optimizer, fresh, rng)
            row = {
                "phase": phase_number, "algorithm": name, "iteration": iteration, "episodes_total": episodes_run,
                "train_success": sum(successes) / len(successes), "forced_evictions": forced,
                "seconds": round(time.time() - started, 1), **stats,
            }
            every = int(settings["eval"]["every"])
            if every and (iteration + 1) % every == 0:
                scores = evaluate_policy(experiment, eval_episodes, fractions, eval_offset)
                row.update(scores)
                validation = sum(scores.values()) / len(scores)
                if validation > best["validation"]:
                    best = {"validation": validation, "episodes_total": episodes_run, "phase": phase_number, **scores}
                    controller.save(folder / "checkpoints" / "policy_best.pt")
                    row["best_so_far"] = True
            with log_path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            if progress:
                shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                print(shown, flush=True)
        controller.save(folder / "checkpoints" / f"policy_phase{phase_number}_{name}.pt")

    controller.save(folder / "checkpoints" / "policy.pt")
    final = evaluate_policy(experiment, eval_episodes, fractions, eval_offset)
    if sum(final.values()) / len(final) > best["validation"]:
        best = {"validation": sum(final.values()) / len(final), "episodes_total": episodes_run, "phase": "final", **final}
        controller.save(folder / "checkpoints" / "policy_best.pt")
    summary = {"episodes_trained": episodes_run, "seconds": round(time.time() - started, 1), "final_eval": final,
               "best_eval": best, "parameters": metadata["parameters"]}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    metadata["status"] = "completed"
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL memory controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="folder to write (default: runs/<name>)")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    folder = train(config, args.output, progress=True)
    print(f"training folder: {folder}")
    print(f"checkpoint:      {folder / 'checkpoints' / 'policy.pt'}")


if __name__ == "__main__":
    main()
