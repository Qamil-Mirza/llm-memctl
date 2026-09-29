"""Run one experiment:  python -m memctl.run --config configs/x.yaml

Everything comes from the YAML config. The run folder holds the resolved
config, every decision, every answer, the metrics and a report.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from memctl.agent import Agent
from memctl.benchmarks import load_benchmark
from memctl.controllers.base import build_controller
from memctl.controllers.oracle import build_plan
from memctl.embed import DEFAULT_EMBEDDER, build_embedder
from memctl.episode import run_conversation
from memctl.judge import Judge
from memctl.llm import build_llm
from memctl.memory import MemoryState
from memctl.summary import summarize

UNLIMITED = 10**9


def budget_for(config: dict, history_tokens: int) -> int:
    """The token budget B: either a fixed number or a fraction of the full history."""
    budget = config["budget"]
    if "tokens" in budget:
        return int(budget["tokens"])
    return max(1, int(budget["fraction"] * history_tokens))


def budget_label(config: dict) -> str:
    budget = config["budget"]
    return f"B{budget['tokens']}" if "tokens" in budget else f"B{round(budget['fraction'] * 100)}pct"


def run(config: dict, allow_fake_jev: bool = False, progress: bool = False) -> Path:
    """Run the experiment described by `config` and return the run folder."""
    started = time.time()
    seed = int(config.get("seed", 0))
    config["controller"]["allow_fake_jev"] = allow_fake_jev
    memory = config.get("memory", {})
    model = config.get("model", {})
    conversations = load_benchmark(config["benchmark"], seed)
    embedder = build_embedder(memory.get("embedder", DEFAULT_EMBEDDER), memory.get("reference_threshold"))
    llm = build_llm(model)
    agent = Agent(llm, model.get("max_new_tokens", 64), memory.get("max_recall_rounds", 2))
    judge = Judge(build_llm(config["judge"])) if config.get("judge") else None

    decisions, answers, context_tokens, places = [], [], [], Counter()
    totals: dict = Counter()
    budgets, jev_log = {}, []
    display_name = ""
    for conversation in conversations:
        controller = build_controller(config["controller"], seed)  # fresh controller per conversation
        display_name = controller.display_name
        budget = UNLIMITED if controller.ignores_budget else budget_for(config, conversation.total_tokens())
        budgets[conversation.id] = "unlimited" if controller.ignores_budget else budget
        state = MemoryState(
            budget=budget,
            embedder=embedder,
            allow_drop=memory.get("allow_drop", True),
            recall_cost=memory.get("recall_cost_tokens", 50),
            store_top_k=memory.get("store_top_k", 5),
            charge_archive_index=memory.get("charge_archive_index", True),
        )
        # The hindsight plan is built from evidence labels. Only the oracle controller receives it.
        # For every other controller it is used afterwards, to annotate the decisions it already made.
        plan = build_plan(conversation, budget, state.allow_drop)
        if controller.uses_evidence_labels:
            controller.receive_plan(plan)
        result = run_conversation(conversation, controller, agent, state, plan, judge)
        decisions += result.decisions
        answers += result.answers
        context_tokens += result.context_tokens
        places.update(place.value for place in state.place.values())
        totals.update(
            recalls=state.recalls,
            recall_tokens=state.recall_tokens,
            store_searches=state.store_searches,
            controller_calls=result.controller_calls,
        )
        jev_log += [{"conversation_id": conversation.id, **call} for call in getattr(controller, "call_log", [])]
        if progress:
            print(f"  {conversation.id}: {len(result.answers)} questions, {time.time() - started:.0f}s so far", flush=True)

    totals = {**totals, "budgets": budgets}
    metrics = summarize(config, display_name, conversations, answers, decisions, context_tokens, places, totals, jev_log)
    metrics["cost"]["wall_clock_s"] = round(time.time() - started, 2)
    metrics["cost"]["generation_cache"] = {"hits": llm.hits, "misses": llm.misses}
    folder = write_run(config, display_name, decisions, answers, metrics, jev_log)
    from memctl.report import write_report

    write_report(folder)
    return folder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def git_state() -> tuple[str, str]:
    """(full commit hash, short label for folder names). The label ends in -dirty if code is uncommitted."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if not commit:
        return "no git commit yet", "nogit"
    return commit + (" (with uncommitted changes)" if dirty else ""), commit[:7] + ("-dirty" if dirty else "")


def write_run(config, display_name, decisions, answers, metrics, jev_log) -> Path:
    commit, short_commit = git_state()
    name = "_".join(
        [
            date.today().isoformat(),
            display_name,
            config["benchmark"]["name"],
            budget_label(config),
            f"seed{config.get('seed', 0)}",
            short_commit,  # so a rerun after a code change never overwrites an older result
        ]
    )
    folder = Path(config.get("output_dir", "runs")) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (folder / "git_commit.txt").write_text(commit + "\n")
    packages = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True)
    env = f"python {platform.python_version()}\nplatform {platform.platform()}\n\n{packages.stdout}"
    (folder / "env.txt").write_text(env)
    write_jsonl(folder / "decisions.jsonl", decisions)
    write_jsonl(folder / "answers.jsonl", answers)
    if jev_log:
        write_jsonl(folder / "jev_calls.jsonl", jev_log)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one memory-controller experiment.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--limit", type=int, help="only run the first N conversations")
    parser.add_argument("--controller", help="override the controller name in the config")
    parser.add_argument("--budget-fraction", type=float, help="override the budget, e.g. 0.25")
    parser.add_argument("--allow-fake-jev", action="store_true", help="use the fake Jev client (labelled FAKE)")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.limit:
        config["benchmark"]["limit"] = args.limit
    if args.controller:
        config["controller"]["name"] = args.controller
    if args.budget_fraction:
        config["budget"] = {"fraction": args.budget_fraction}
    folder = run(config, allow_fake_jev=args.allow_fake_jev, progress=True)
    print(f"run folder: {folder}")
    print(f"report:     {folder / 'report.md'}")


if __name__ == "__main__":
    main()
