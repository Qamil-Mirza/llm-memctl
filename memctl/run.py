"""Run one experiment:  python -m memctl.run --config configs/x.yaml

Everything comes from the YAML config. The run folder holds the resolved
config, every decision, every answer and the metrics.
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

from memctl.agent import Agent, build_llm
from memctl.benchmarks import load_benchmark
from memctl.controllers.base import build_controller
from memctl.embed import build_embedder
from memctl.episode import run_conversation
from memctl.memory import MemoryState
from memctl.metrics import mean

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


def run(config: dict, allow_fake_jev: bool = False) -> Path:
    """Run the experiment described by `config` and return the run folder."""
    started = time.time()
    seed = int(config.get("seed", 0))
    config["controller"]["allow_fake_jev"] = allow_fake_jev
    memory = config.get("memory", {})
    conversations = load_benchmark(config["benchmark"], seed)
    embedder = build_embedder(memory.get("embedder", "hashing"))
    llm = build_llm(config.get("model", {}))
    agent = Agent(llm, config.get("model", {}).get("max_new_tokens", 64), memory.get("max_recall_rounds", 2))

    decisions, answers, context_tokens, places = [], [], [], Counter()
    totals = Counter()
    jev_log: list[dict] = []
    display_name = ""
    for conversation in conversations:
        controller = build_controller(config["controller"], seed)  # fresh controller per conversation
        display_name = controller.display_name
        budget = UNLIMITED if controller.ignores_budget else budget_for(config, conversation.total_tokens())
        state = MemoryState(
            budget=budget,
            embedder=embedder,
            allow_drop=memory.get("allow_drop", True),
            recall_cost=memory.get("recall_cost_tokens", 50),
            store_top_k=memory.get("store_top_k", 5),
            charge_archive_index=memory.get("charge_archive_index", True),
        )
        result = run_conversation(conversation, controller, agent, state)
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

    metrics = summarize(config, display_name, conversations, answers, context_tokens, places, totals, jev_log)
    metrics["cost"]["wall_clock_s"] = round(time.time() - started, 2)
    metrics["cost"]["generation_cache"] = {"hits": llm.hits, "misses": llm.misses}
    folder = write_run(config, display_name, decisions, answers, metrics, jev_log)
    return folder


def summarize(config, display_name, conversations, answers, context_tokens, places, totals, jev_log) -> dict:
    """Build metrics.json from the per-decision and per-answer rows."""
    by_category: dict[str, list[float]] = {}
    for row in answers:
        by_category.setdefault(row["category"], []).append(row["scores"]["f1"])
    evidence = [e for row in answers for e in row["evidence"]]
    share = lambda test: round(sum(1 for e in evidence if test(e)) / len(evidence), 4) if evidence else None
    return {
        "controller": display_name,
        "fake_jev": display_name.endswith("FAKE"),
        "benchmark": config["benchmark"]["name"],
        "budget": config["budget"],
        "seed": config.get("seed", 0),
        "conversations": len(conversations),
        "questions": len(answers),
        "answer_quality": {
            "f1": mean([row["scores"]["f1"] for row in answers]),
            "f1_by_category": {c: mean(v) for c, v in sorted(by_category.items())},
        },
        "evidence_availability": {
            "in_context": share(lambda e: e["place"] == "CONTEXT"),
            "retrieved_from_store": share(lambda e: e["retrieved_from_store"]),
            "recallable_from_archive": share(lambda e: e["place"] == "ARCHIVE"),
            "in_store_but_not_retrieved": share(lambda e: e["place"] == "STORE" and not e["retrieved_from_store"]),
            "dropped": share(lambda e: e["place"] == "DROPPED"),
        },
        "cost": {
            "avg_context_tokens": mean(context_tokens),
            "peak_context_tokens": max(context_tokens, default=0),
            "avg_prompt_tokens": mean([row["prompt_tokens"] for row in answers]),
            "peak_prompt_tokens": max((row["prompt_tokens"] for row in answers), default=0),
            **dict(totals),
            "jev_calls": len(jev_log),
            "jev_cost_usd": round(sum(call["cost_usd"] for call in jev_log), 6),
        },
        "final_places": dict(places),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_run(config, display_name, decisions, answers, metrics, jev_log) -> Path:
    name = "_".join(
        [
            date.today().isoformat(),
            display_name,
            config["benchmark"]["name"],
            budget_label(config),
            f"seed{config.get('seed', 0)}",
        ]
    )
    folder = Path(config.get("output_dir", "runs")) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    (folder / "git_commit.txt").write_text(commit.stdout.strip() or "no git commit yet\n")
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
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
    parser.add_argument("--allow-fake-jev", action="store_true", help="use the fake Jev client (labelled FAKE)")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.limit:
        config["benchmark"]["limit"] = args.limit
    folder = run(config, allow_fake_jev=args.allow_fake_jev)
    print(f"run folder: {folder}")
    print((folder / "metrics.json").read_text())


if __name__ == "__main__":
    main()
