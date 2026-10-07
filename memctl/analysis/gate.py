"""The Experiment 13 evidence gate: evidence in view at the question, learned controllers against FIFO.

Reads the agent-null gate sweeps (one per LongMemEval fold), pairs every candidate cell with the FIFO cell of
the same budget and target on the same questions (fold, seed), pools the folds, and reports the mean
paired difference with a 95% bootstrap interval. A cell is a go when the interval lies entirely above 0
(the rule pre-registered in EXPERIMENTS.md 13).

    python -m memctl.analysis.gate runs/exp13_gate_f0 ... runs/exp13_gate_f4 [--gate-file runs/_logs/GATE]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

BASELINE = "fifo_top5"
CANDIDATES = ("learned_floor5", "compose_floor5", "synthetic_floor5")


def split_label(cell: str) -> tuple[str, str, str]:
    """(controller, target tag, budget) from a cell name like 'compose_floor5_t2000__fraction0.02'."""
    label, _, fraction = cell.partition("__fraction")
    match = re.fullmatch(r"(.+)_t(\d+)", label)  # not partition("_t"): "fifo_top5" contains it
    return (match.group(1), f"t{match.group(2)}", fraction) if match else (label, "fill", fraction)


# Mean active memory (tokens) per (cell key, question): with no reader there is no prompt, so this stands in for
# prompt size. Filled by load().
MEMORY: dict[tuple[str, str, str], dict[tuple[str, int], float]] = defaultdict(dict)


def load(sweeps: list[Path]) -> dict[tuple[str, str, str], dict[tuple[str, int], float]]:
    values: dict[tuple[str, str, str], dict[tuple[str, int], float]] = defaultdict(dict)
    memory = MEMORY
    for sweep in sweeps:
        for cell in sweep.iterdir():
            path = cell / "episodes.jsonl"
            if not path.exists():
                continue
            key = split_label(cell.name)
            for line in path.open():
                episode = json.loads(line)
                if episode.get("needed_hit_rate") is not None:
                    values[key][(sweep.name, episode["seed"])] = episode["needed_hit_rate"]
                    memory[key][(sweep.name, episode["seed"])] = episode.get("active_tokens_mean") or 0.0
    return values


def interval(differences: list[float], samples: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def sweep_of(controller: str, target: str, fold: int) -> str:
    """The paid sweep that holds a learned cell (configs/sweeps/exp13/make.py)."""
    if controller == "learned_floor5":
        return f"exp13_lme_f{fold}" + ("" if target == "fill" else "_target")
    if controller == "synthetic_floor5" and target == "fill":
        return f"exp13_lme_f{fold}"
    return f"exp13_lme_f{fold}_extra"


def gate(sweeps: list[Path]) -> list[dict]:
    values = load(sweeps)
    rows = []
    for (controller, target, fraction), candidate in sorted(values.items()):
        if controller not in CANDIDATES:
            continue
        baseline = values.get((BASELINE, target, fraction), {})
        shared = sorted(set(candidate) & set(baseline))
        if len(shared) < 2:
            continue
        differences = [candidate[q] - baseline[q] for q in shared]
        low, high = interval(differences)
        rows.append({"controller": controller, "target": target, "budget": fraction, "n": len(shared),
                     "fifo": sum(baseline[q] for q in shared) / len(shared),
                     "candidate": sum(candidate[q] for q in shared) / len(shared),
                     "difference": sum(differences) / len(differences), "low": low, "high": high, "go": low > 0,
                     "fifo_memory": sum(MEMORY[(BASELINE, target, fraction)].get(q, 0.0) for q in shared) / len(shared),
                     "candidate_memory": sum(MEMORY[(controller, target, fraction)].get(q, 0.0) for q in shared) / len(shared)})
        last = rows[-1]
        # Matched by construction (same budget or target); a gap over 10% is flagged, not used to decide.
        last["memory_flag"] = abs(last["candidate_memory"] - last["fifo_memory"]) > 0.1 * max(last["fifo_memory"], 1.0)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", nargs="+", type=Path)
    parser.add_argument("--gate-file", type=Path, help="write the go cells as 'sweep label' lines for the paid run")
    args = parser.parse_args()
    rows = gate(args.sweeps)
    print("| candidate | target | budget | n | FIFO evidence | candidate evidence | difference (95% CI) | "
          "memory tokens FIFO / candidate | go |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = " (>10% apart)" if r["memory_flag"] else ""
        print(f"| {r['controller']} | {r['target']} | {r['budget']} | {r['n']} | {r['fifo']:.3f} | {r['candidate']:.3f} | "
              f"{r['difference']:+.3f} ({r['low']:+.3f}, {r['high']:+.3f}) | {r['fifo_memory']:.0f} / "
              f"{r['candidate_memory']:.0f}{flag} | {'go' if r['go'] else 'no'} |")
    if args.gate_file:
        lines = []
        for r in rows:
            # Targets are run at 2% and 5% only (at 1% the budget is below every target).
            if not r["go"] or (r["target"] != "fill" and r["budget"] == "0.01"):
                continue
            tag = "" if r["target"] == "fill" else "_" + r["target"]
            for fold in range(5):
                lines.append(f"{sweep_of(r['controller'], r['target'], fold)} {r['controller']}{tag}__fraction{r['budget']}")
        args.gate_file.write_text("".join(line + "\n" for line in lines))
        print(f"{len(lines)} go cells written to {args.gate_file}")


if __name__ == "__main__":
    main()
