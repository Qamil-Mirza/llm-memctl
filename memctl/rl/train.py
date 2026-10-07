"""Train the RL controller:  python -m memctl.rl.train --config configs/rl/x.yaml

The config is an ordinary experiment config (environment, agent, memory,
reward, `controller: {name: rl, ...}`) plus a `training` section:

    training:
      seed_offset: 100000            # training episodes use seeds from here on; evaluation uses 0, 1, 2, ...
      budget_fractions: [0.05, 0.1, 0.25, 0.5]   # optional: a budget is drawn per episode
      phases:
        - {algorithm: bc, iterations: 8, episodes: 16, epochs: 4, lr: 0.003, dagger: true}
        - {algorithm: ppo, iterations: 40, episodes: 16, epochs: 4, lr: 0.0003, gamma: 0.995}
        - {algorithm: grpo, iterations: 40, episodes: 24, group: 8, epochs: 4, lr: 0.0003}
          # optional `advantage: std | mean | batch | best` (see set_group_advantages)
      eval: {every: 4, episodes: 20, seed_offset: 50000}
      expert: {kind: regret}         # optional: which hindsight expert labels imitation (memctl/rl/expert.py)
      tasks:                         # optional: train one policy on several tasks, one episode each in turn
        - {label: recall, env: {name: synthetic_recall}, budget_fractions: [0.02, 0.05]}
        - {label: workflow, env: {name: workflow}, agent: {name: scripted_tool_agent}}

Each task is the experiment config with the task's keys merged on top (a task
may set its own `budget_fractions`). Validation success is logged per task as
`<label>/success@<fraction>`, and `policy_best.pt` is the best mean over all of
them.

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
from memctl.util import merged

TRAINING_DEFAULTS = {
    "seed_offset": 100_000,
    "budget_fractions": None,
    "phases": [{"algorithm": "bc", "iterations": 8, "episodes": 16}],
    # metric: which validation score picks policy_best.pt: "success", "evidence" (evidence-in-context),
    # or "auto" (evidence when there is no task model, else success).
    "eval": {"every": 4, "episodes": 20, "seed_offset": 50_000, "metric": "auto"},
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


ADVANTAGES = ("std", "mean", "batch", "best")


def set_group_advantages(
    returns: list[tuple], mode: str = "std", strata: dict[int, object] | None = None, question_weight: float = 0.0,
) -> None:
    """GRPO: each decision's advantage is its episode's return relative to its group.

    `returns` holds (group, return, decisions) or (group, return, decisions, rewards by step).

    - `std`   (GRPO): minus the group mean, over the group's standard deviation.
    - `mean`  (Dr. GRPO): minus the group mean, unscaled, so near-tied groups stay small.
    - `batch` (Lite PPO / REINFORCE++): minus the group mean, over the batch's standard deviation,
      or over each stratum's (`strata`: group -> key, e.g. the budget fraction) when given.
    - `best`  (RAFT, filtered imitation): 1 for the group's single best episode, 0 for the rest
      and for groups whose best is shared, so the update raises the likelihood of the best sample.

    `question_weight` adds per-step credit: every sample of a group sees the same question at the
    same step (one seed), so a decision at step t also gets `question_weight` times its episode's
    reward at t minus the group's mean reward at t. It costs no extra episodes.
    """
    if mode not in ADVANTAGES:
        raise ValueError(f"unknown advantage {mode!r}; expected one of {ADVANTAGES}")
    groups: dict[int, list[float]] = {}
    for group, value, *_ in returns:
        groups.setdefault(group, []).append(value)
    mean_of = {group: sum(values) / len(values) for group, values in groups.items()}
    centred = [value - mean_of[group] for group, value, *_ in returns]
    keys = [strata.get(group) if strata else None for group, *_ in returns]
    spread_of: dict[object, float] = {}
    for key in set(keys):
        values = [c for c, k in zip(centred, keys) if k == key]
        spread_of[key] = (sum(c * c for c in values) / len(values)) ** 0.5
    step_means: dict[int, dict[int, float]] = {}
    if question_weight:
        for group in groups:
            members = [row[3] for row in returns if row[0] == group]
            steps = {step for rewards in members for step in rewards}
            step_means[group] = {step: sum(r.get(step, 0.0) for r in members) / len(members) for step in steps}
    for row, offset, key in zip(returns, centred, keys):
        group, value, decisions = row[0], row[1], row[2]
        values = groups[group]
        spread = (sum((v - mean_of[group]) ** 2 for v in values) / len(values)) ** 0.5
        if mode == "std":
            advantage = offset / (spread + 1e-6) if spread > 0 else 0.0
        elif mode == "mean":
            advantage = offset
        elif mode == "batch":
            advantage = offset / (spread_of[key] + 1e-6) if spread_of[key] > 0 else 0.0
        else:
            advantage = float(value == max(values) and values.count(value) == 1)
        for decision in decisions:
            extra = 0.0
            if question_weight:
                extra = question_weight * (row[3].get(decision.step, 0.0) - step_means[group].get(decision.step, 0.0))
            decision.advantage = advantage + extra


def group_spread(member: dict) -> dict:
    """What a GRPO group's return spread is made of: answers, forced fallbacks, or nothing."""
    returns, success = member["returns"], member["success"]
    mean = sum(returns) / len(returns)
    spread = (sum((v - mean) ** 2 for v in returns) / len(returns)) ** 0.5
    if spread == 0:
        source = "tied"
    elif len(set(success)) == 1:
        source = "fallbacks_only"
    else:
        source = "answers"
    return {"spread": round(spread, 6), "source": source}


def evaluate_policy(experiment: Experiment, episodes: int, fractions: list[float] | None, seed_offset: int = 50_000,
                    prefix: str = "") -> dict:
    """Mean task success and evidence-in-context rate of the greedy policy on the validation seeds
    (seed_offset, seed_offset + 1, ...). Evidence-in-context is the share of a question's needed items
    that are active when it is asked (`needed_hit_rate`); it needs no reader, so it is the validation
    signal when the task model is absent and task success is always 0."""
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
            runs = [experiment.run_episode(seed=seed, detail=False).episode for seed in seeds]
            for seed in seeds:
                experiment._hindsight.pop(seed, None)
            suffix = "" if fraction is None else f"@{fraction:g}"
            results[prefix + "success" + suffix] = sum(e["task_success"] for e in runs) / len(runs)
            evidence = [e["needed_hit_rate"] for e in runs if e.get("needed_hit_rate") is not None]
            if evidence:
                results[prefix + "evidence" + suffix] = sum(evidence) / len(evidence)
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
    # (label, experiment, budget fractions) per task; every experiment drives the same controller.
    tasks = [("", experiment, settings["budget_fractions"])]
    if settings.get("tasks"):
        base = {key: value for key, value in config.items() if key != "training"}
        tasks = []
        for task in settings["tasks"]:
            overrides = {key: value for key, value in task.items() if key not in ("label", "budget_fractions")}
            tasks.append((f"{task['label']}/", Experiment(merged(base, overrides), controller),
                          task.get("budget_fractions", settings["budget_fractions"])))

    def evaluate_all() -> dict:
        scores = {}
        for prefix, task_experiment, task_fractions in tasks:
            scores.update(evaluate_policy(task_experiment, eval_episodes, task_fractions, eval_offset, prefix))
        return scores

    metric = settings["eval"]["metric"]
    if metric == "auto":
        metric = "evidence" if (experiment.config.get("agent") or {}).get("name", "null") == "null" else "success"
    if metric not in ("success", "evidence"):
        raise ValueError(f"unknown training.eval.metric {metric!r}")

    def selection(scores: dict) -> float:
        """The validation score: the mean of the chosen metric over tasks and budgets."""
        chosen = [v for k, v in scores.items() if k.split("@")[0].rpartition("/")[2].endswith(metric)]
        return sum(chosen) / len(chosen) if chosen else 0.0

    controller.record = True
    metadata = {**collect_metadata(), "models": experiment.models(), "status": "running",
                "parameters": sum(p.numel() for p in controller.policy.parameters())}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log_path = folder / "train_log.jsonl"
    log_path.write_text("")
    (folder / "groups.jsonl").unlink(missing_ok=True)

    next_seed = int(settings["seed_offset"]) + int(config["seed"])
    started, episodes_run = time.time(), 0
    eval_episodes, eval_offset = int(settings["eval"]["episodes"]), int(settings["eval"]["seed_offset"])
    best: dict = {"validation": -1.0}
    for phase_number, phase in enumerate(settings["phases"]):
        name = phase["algorithm"]
        algorithm = ALGORITHMS[name](phase)
        optimizer = torch.optim.Adam(controller.policy.parameters(), lr=float(phase.get("lr", 3e-3 if name == "bc" else 3e-4)))
        imitation = name in ("bc", "cost")
        # GRPO may keep the expert's labels in its loss (`imitation_weight`); the learner still acts.
        labelled = imitation or bool(phase.get("imitation_weight"))
        # Regret tracking is only worth its cost when the expert labels need hindsight anyway.
        for _, task_experiment, _ in tasks:
            task_experiment.config["hindsight"]["enabled"] = False
        dataset: list[Decision] = []
        group = int(phase.get("group", 1)) if name == "grpo" else 1
        for iteration in range(int(phase.get("iterations", 8))):
            fresh: list[Decision] = []
            successes, forced = [], 0
            returns: list[tuple[int, float, list[Decision]]] = []  # (group, return, decisions), for GRPO
            members: dict[int, dict] = {}  # GRPO: what each group's spread is made of, for groups.jsonl
            for number in range(int(phase.get("episodes", 16))):
                if number % group == 0:  # a new episode: seed, task and budget shared by the whole group
                    seed, next_seed = next_seed, next_seed + 1
                    _, experiment, fractions = tasks[(number // group) % len(tasks)]
                    if fractions:
                        experiment.config["memory"]["budget"] = {"fraction": rng.choice(fractions)}
                controller.sample_offset = number % group if group > 1 else 0
                controller.greedy = False
                if labelled:
                    controller.expert = make_expert(experiment.hindsight(seed), **settings["expert"])
                    # DAgger: the expert drives the first iteration, the learner the rest, the expert labels all.
                    controller.follow_expert = imitation and (iteration == 0 or not phase.get("dagger", True))
                else:
                    controller.expert, controller.follow_expert = None, False
                episode = experiment.run_episode(seed=seed, detail=False).episode
                experiment._hindsight.pop(seed, None)
                recorded = controller.recorded
                if imitation and episode.get("hindsight_diverged_at") is not None:
                    # From the divergence on, the expert's labels describe another episode.
                    recorded = [d for d in recorded if d.step < episode["hindsight_diverged_at"]]
                elif labelled and episode.get("hindsight_diverged_at") is not None:
                    for decision in recorded:  # GRPO keeps every decision but only the labels that still apply
                        if decision.step >= episode["hindsight_diverged_at"]:
                            decision.expert_rank = decision.expert_retrieved = decision.expert_cost = None
                if name == "grpo":
                    total = sum(reward for _, reward in controller.rewards)
                    returns.append((number // group, total, recorded, dict(controller.rewards)))
                    member = members.setdefault(number // group, {
                        "seed": seed, "fraction": experiment.config["memory"]["budget"].get("fraction"),
                        "returns": [], "success": [], "forced": []})
                    member["returns"].append(round(total, 6))
                    member["success"].append(episode["task_success"])
                    member["forced"].append(episode["forced_evictions"])
                elif not imitation:
                    fill_returns(controller.recorded, controller.rewards, float(phase.get("gamma", 0.995)))
                fresh += recorded
                successes.append(episode["task_success"])
                forced += episode["forced_evictions"]
                episodes_run += 1
            controller.sample_offset = 0
            if name == "grpo":
                strata = {n: m["fraction"] for n, m in members.items()} if phase.get("stratify") else None
                set_group_advantages(returns, phase.get("advantage", "std"), strata, float(phase.get("question_weight", 0.0)))
                with (folder / "groups.jsonl").open("a") as file:
                    for number, member in members.items():
                        file.write(json.dumps({"phase": phase_number, "iteration": iteration, "group": number,
                                               **member, **group_spread(member)}) + "\n")
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
                scores = evaluate_all()
                row.update(scores)
                validation = selection(scores)
                row["validation"] = validation
                if validation > best["validation"]:
                    best = {"validation": validation, "metric": metric, "episodes_total": episodes_run,
                            "phase": phase_number, **scores}
                    controller.save(folder / "checkpoints" / "policy_best.pt")
                    row["best_so_far"] = True
            with log_path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            if progress:
                shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                print(shown, flush=True)
        controller.save(folder / "checkpoints" / f"policy_phase{phase_number}_{name}.pt")

    controller.save(folder / "checkpoints" / "policy.pt")
    final = evaluate_all()
    if selection(final) > best["validation"]:
        best = {"validation": selection(final), "metric": metric, "episodes_total": episodes_run, "phase": "final", **final}
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
