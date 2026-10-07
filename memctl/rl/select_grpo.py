"""Set-valued GRPO on the selection head, with the frozen reader's verdict as reward (Experiment 16).

The policy is Experiment 15's floor head: at a question it scores the 16 BM25 candidates and the reader sees
the chosen 5 (keep-none, so nothing else). Here the head is trained on whether the reader then answers
correctly, starting from the `listsum` checkpoint. Design pre-registered in EXPERIMENTS.md §16:

- per question, G subsets of k are sampled without replacement by Gumbel-top-k on the head's logits;
- each subset's probability is the unordered-set probability under Plackett-Luce (Kool et al. 2020), the sum
  over its k! orders, computed exactly;
- reward = judged correct (official LongMemEval judge) + aux_weight x token F1 of the answer against the gold;
- advantage = reward minus the mean of its group (no division by the group's spread);
- loss = PPO-clipped policy gradient on the set log-probability + anchor x the listsum imitation loss;
- questions the reader answers correctly with empty memory are dropped before training (MemAgent's filter);
- a held-out 20% of the train part gates the result: greedy top-k accuracy against the starting checkpoint.

The reader prompt is built by the same LLMAgent as every sweep, so generations share the sweeps' cache.

    python -m memctl.rl.select_grpo --config configs/rl/lme_n3/grpo_f0.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from memctl.agents.llm import LLMAgent
from memctl.config import resolve
from memctl.envs import build_env
from memctl.harness.runner import Experiment
from memctl.judge import Judge
from memctl.memory.items import SourceType
from memctl.memory.state import MemoryState
from memctl.metrics import is_refusal, token_f1
from memctl.rl.algorithms import imitation_loss
from memctl.rl.expert import make_expert
from memctl.rl.policy import RETRIEVE_COLUMN
from memctl.runlog import peak_rss_mb
from memctl.sysinfo import collect_metadata
from memctl.task import Observation, TaskState

DEFAULTS = {"group": 8, "k": 5, "aux_weight": 0.2, "anchor": 0.1, "lr": 3e-4, "epochs": 4, "clip": 0.2,
            "iterations": 6, "questions_per_iteration": 128, "held_out": 0.2, "workers": 32, "seed": 0}


@dataclass
class Case:
    """One question at the moment it is asked: the shortlist the head ranks and everything the reader needs."""

    question_id: str
    category: str
    gold: str
    question: Observation
    goal: str
    decision: object  # memctl.controllers.rl.Decision (features, ids, expert labels)
    turns: dict  # item id -> Observation, for the shortlist
    order: dict  # item id -> position in the conversation (the reader shows memory in arrival order)
    unanswerable: bool = False
    rewards: list = field(default_factory=list)


def gumbel_top_k(logits: torch.Tensor, k: int, generator: torch.Generator) -> list[int]:
    """A sample of k distinct indices from Plackett-Luce(logits), in sampled order."""
    gumbel = -torch.log(-torch.log(torch.rand(logits.shape, generator=generator).clamp_min(1e-12)))
    return torch.topk(logits + gumbel, k).indices.tolist()


def set_log_prob(logits: torch.Tensor, subset: list[int]) -> torch.Tensor:
    """log P(the unordered set) under Plackett-Luce: log of the sum over its k! orders (Kool et al. 2020)."""
    terms = []
    for order in itertools.permutations(subset):
        remaining = torch.ones_like(logits, dtype=torch.bool)
        total = logits.new_zeros(())
        for index in order:
            total = total + logits[index] - torch.logsumexp(logits[remaining], 0)
            remaining = remaining.clone()
            remaining[index] = False
        terms.append(total)
    return torch.logsumexp(torch.stack(terms), 0)


class Reader:
    """The frozen reader and judge of the sweeps, called on a chosen set of turns."""

    def __init__(self, config: dict) -> None:
        self.agent = LLMAgent(config["agent"])
        self.judge = Judge(config["env"]["judge"])

    def answer(self, case: Case, chosen_ids: list[str]) -> tuple[float, str]:
        state = MemoryState(10**9, count_labels=False)
        for item_id in sorted(chosen_ids, key=lambda i: case.order[i]):
            turn = case.turns[item_id]
            state.step = case.order[item_id]
            state.ingest(item_id, turn.content, turn.source_type, metadata=turn.metadata)
        state.step = 10**8
        state.ingest(case.question.id, case.question.content, SourceType.USER)
        task = TaskState(state.step, case.goal, case.question)
        answer = self.agent.act(state.view(), case.question, task).action
        if case.unanswerable:
            return float(is_refusal(answer)), answer
        correct = self.judge.is_correct(case.question.content.removeprefix("Question: "), case.gold, answer, case.category)
        return float(correct), answer


def collect_cases(experiment: Experiment, seeds: list[int]) -> list[Case]:
    """Play each seed's single-question episode with keep-none and record the decision at its question."""
    controller = experiment.controller
    controller.record, controller.greedy, controller.follow_expert = True, True, False
    env = build_env(experiment.config["env"])  # to read the episode's turns and question; never stepped
    cases = []
    for seed in seeds:
        controller.expert = make_expert(experiment.hindsight(seed), kind="regret", retrieval_risk=0.36)
        experiment.run_episode(seed=seed, detail=False)
        experiment._hindsight.pop(seed, None)
        decisions = [d for d in controller.recorded if d.retrieve_tokens]
        if not decisions:
            continue
        episode = env.load_episode(seed)
        question = episode.questions[0]
        observation = Observation("q0000", f"Question: {question.question}", SourceType.USER, requires_response=True)
        cases.append(Case(question.id, question.category, question.gold, observation, env.goal, decisions[-1],
                          {turn.id: turn for turn in episode.turns}, {turn.id: n for n, turn in enumerate(episode.turns)},
                          question.unanswerable))
    controller.expert = None
    return cases


def chosen_ids(case: Case, indices: list[int]) -> list[str]:
    n = case.decision.n_active
    return [case.decision.item_ids[n + i] for i in indices]


def evaluate(policy, cases: list[Case], reader: Reader, k: int, workers: int) -> list[float]:
    """Judged correctness of the greedy top k for each case."""
    picks = []
    with torch.no_grad():
        for case in cases:
            logits = policy(torch.from_numpy(case.decision.items), torch.from_numpy(case.decision.global_features))[0]
            column = logits[case.decision.n_active:, RETRIEVE_COLUMN]
            picks.append(torch.topk(column, min(k, len(column))).indices.tolist())
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda pair: reader.answer(pair[0], chosen_ids(pair[0], pair[1]))[0], zip(cases, picks)))


def train(config: dict) -> Path:
    settings = {**DEFAULTS, **config.get("grpo", {})}
    config = resolve(config)
    folder = Path(config["logging"]["output_dir"]) / config["name"]
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    generator = torch.Generator().manual_seed(int(settings["seed"]))
    # Cases are collected with no reader and no judge (only the shortlist, features and labels are needed);
    # the reader is called only by Reader, on the subsets.
    quiet = {**config, "agent": {"name": "null"}, "env": {k: v for k, v in config["env"].items() if k != "judge"}}
    experiment = Experiment(resolve(quiet))
    reader = Reader(config)
    metadata = {**collect_metadata(), "settings": settings, "status": "running"}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    log = (folder / "train_log.jsonl").open("w")

    pool = list(range(int(settings.get("questions", 400))))
    rng.shuffle(pool)
    split = int(len(pool) * (1 - float(settings["held_out"])))
    train_seeds, held_seeds = pool[:split], pool[split:]
    started = time.time()
    train_cases = collect_cases(experiment, train_seeds)
    held_cases = collect_cases(experiment, held_seeds)

    # MemAgent's filter: drop training questions the reader answers correctly with no memory at all.
    with ThreadPoolExecutor(int(settings["workers"])) as executor:
        empty = list(executor.map(lambda case: reader.answer(case, [])[0], train_cases))
    kept = [case for case, solved in zip(train_cases, empty) if solved < 1.0]
    policy = experiment.controller.policy
    baseline = evaluate(policy, held_cases, reader, int(settings["k"]), int(settings["workers"]))
    log.write(json.dumps({"stage": "start", "train_cases": len(train_cases), "kept_after_filter": len(kept),
                          "held_cases": len(held_cases), "held_accuracy": float(np.mean(baseline))}) + "\n")
    log.flush()
    curve, tie_rates = [{"iteration": 0, "held_accuracy": float(np.mean(baseline))}], []

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["lr"]))
    k, group = int(settings["k"]), int(settings["group"])
    for iteration in range(int(settings["iterations"])):
        batch = rng.sample(kept, min(int(settings["questions_per_iteration"]), len(kept)))
        samples = []  # (case, subset indices, old log-prob)
        with torch.no_grad():
            for case in batch:
                logits = policy(torch.from_numpy(case.decision.items), torch.from_numpy(case.decision.global_features))[0]
                column = logits[case.decision.n_active:, RETRIEVE_COLUMN]
                for _ in range(group):
                    subset = gumbel_top_k(column, min(k, len(column)), generator)
                    samples.append((case, subset, float(set_log_prob(column, subset))))
        with ThreadPoolExecutor(int(settings["workers"])) as executor:
            outcomes = list(executor.map(lambda s: reader.answer(s[0], chosen_ids(s[0], s[1])), samples))
        rewards = [correct + float(settings["aux_weight"]) * token_f1(answer, s[0].gold)
                   for s, (correct, answer) in zip(samples, outcomes)]
        advantages, ties = [], 0
        for start in range(0, len(samples), group):
            chunk = rewards[start:start + group]
            mean = sum(chunk) / len(chunk)
            ties += int(max(chunk) - min(chunk) < 1e-9)
            advantages += [r - mean for r in chunk]
        for _ in range(int(settings["epochs"])):
            losses = []
            for (case, subset, old), advantage in zip(samples, advantages):
                logits = policy(torch.from_numpy(case.decision.items), torch.from_numpy(case.decision.global_features))[0]
                column = logits[case.decision.n_active:, RETRIEVE_COLUMN]
                ratio = torch.exp(set_log_prob(column, subset) - old)
                clipped = torch.clamp(ratio, 1 - float(settings["clip"]), 1 + float(settings["clip"]))
                losses.append(-torch.min(ratio * advantage, clipped * advantage))
            anchor = [imitation_loss(policy, case.decision, "listsum")[0] for case in batch]
            loss = torch.stack(losses).mean() + float(settings["anchor"]) * torch.stack(anchor).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        row = {"iteration": iteration, "questions": len(batch), "mean_reward": float(np.mean(rewards)),
               "correct_rate": float(np.mean([o[0] for o in outcomes])), "tie_groups": ties,
               "groups": len(batch), "loss": float(loss), "seconds": round(time.time() - started)}
        tie_rates.append(ties / max(1, len(batch)))
        if (iteration + 1) % 2 == 0 and iteration + 1 < int(settings["iterations"]):  # learning curve on the held-out slice
            row["held_accuracy"] = float(np.mean(evaluate(policy, held_cases, reader, k, int(settings["workers"]))))
            curve.append({"iteration": iteration + 1, "held_accuracy": row["held_accuracy"]})
        log.write(json.dumps(row) + "\n")
        log.flush()

    final = evaluate(policy, held_cases, reader, k, int(settings["workers"]))
    torch.save({"state_dict": policy.state_dict(), "policy": policy.config, "feature_version": 1,
                "use_embeddings": False, "embedding_dim": 0, "columns": logits.shape[1]}, folder / "policy_grpo.pt")
    differences = [a - b for a, b in zip(final, baseline)]
    curve.append({"iteration": int(settings["iterations"]), "held_accuracy": float(np.mean(final))})
    checkpoint = config["controller"].get("checkpoint")
    summary = {"held_questions": len(held_cases), "start_accuracy": float(np.mean(baseline)),
               "grpo_accuracy": float(np.mean(final)), "paired_differences": differences,
               "start_checkpoint": checkpoint,
               "start_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
               "train_cases": len(train_cases), "kept_after_filter": len(kept), "tie_rate_per_iteration": tie_rates,
               "held_accuracy_curve": curve}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    metadata.update(status="completed", peak_rss_mb=peak_rss_mb(), seconds=round(time.time() - started))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    folder = train(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({k: v for k, v in json.loads((folder / "summary.json").read_text()).items() if k != "paired_differences"}))


if __name__ == "__main__":
    main()
