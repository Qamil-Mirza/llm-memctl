"""Per-category tables for LoCoMo and LongMemEval question-answering sweeps, from the run logs.

For each controller group (seeds `_s0`, `_s1`, ... pooled) and budget fraction it reports:

- the headline judge accuracy, without LoCoMo's adversarial category and without LongMemEval's
  abstention questions, which are scored by refusal and so reward a controller that keeps less;
- a 95% bootstrap interval for the headline, resampling clusters (LoCoMo conversations, whose
  questions share one memory state; LongMemEval questions, one per episode);
- token F1 on the same questions, as a check on a reader that also judges (only where answers
  were logged: the episodes logged in detail);
- accuracy per category, and on the refusal-scored questions separately;
- the reader's prompt size in tokens (mean and median over episodes of each episode's mean prompt),
  from tokens_processed / task_model_calls, so it is available for every episode.

    python -m memctl.analysis.qa_tables runs/exp11b_locomo_qa [more sweeps...] [--per-category]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from memctl.metrics import token_f1
from memctl.runlog import open_trace, trace_exists
from memctl.splits import fold_indices

LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
REFUSAL_SCORED = ("adversarial", "abstention")


@lru_cache(maxsize=4)
def _data(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def question_table(env: dict):
    """A function (seed, question number) -> (cluster, category, gold) for the cell's environment."""
    name, path = env["name"], env.get("path")
    if name == "locomo":
        samples = _data(path or "data/locomo/locomo10.json")
        adversarial = env.get("include_adversarial", True)

        def locomo(seed: int, number: int):
            sample = samples[seed % len(samples)]
            kept = [qa for qa in sample["qa"] if adversarial or qa["category"] != 5]
            qa = kept[number]
            category = LOCOMO_CATEGORIES[qa["category"]]
            return sample["sample_id"], category, "unknown" if category == "adversarial" else str(qa.get("answer", ""))

        return locomo
    if name == "longmemeval":
        instances = _data(path or "data/longmemeval/longmemeval_oracle.json")
        if env.get("folds"):
            folds = env["folds"]
            chosen = fold_indices(instances, int(folds.get("k", 5)), int(folds["fold"]), folds.get("part", "test"))
        else:
            start, end = env.get("subset") or (0, len(instances))
            chosen = list(range(start, end))

        def longmemeval(seed: int, number: int):
            instance = instances[chosen[seed % len(chosen)]]
            qid = str(instance["question_id"])
            category = "abstention" if qid.endswith("_abs") else instance["question_type"]
            return qid, category, str(instance["answer"])

        return longmemeval
    raise ValueError(f"no question table for env {name!r}")


def cell_rows(cell: Path) -> list[dict]:
    """One row per question: cluster, category, judge verdict, F1 and memory tokens.

    Every episode logs its incorrect answers to failures.jsonl, so verdicts come from there for all
    episodes. Answers (for F1) and the memory at read time are in steps.jsonl only for the episodes
    logged in detail; elsewhere F1 is None and memory is the episode's mean active tokens.
    """
    settings = yaml.safe_load((cell / "config.yaml").read_text())
    env = settings["env"]
    lookup = question_table(env)
    # A note that hit the output limit before "Answer:". The agent's own flag (agent_truncated) also counts
    # a bare reply without the marker, mostly "unknown", so where outputs are logged they decide instead.
    limit = int((settings.get("agent") or {}).get("max_new_tokens", 0) or 0)
    failed = set()
    with open(cell / "failures.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            failed.add((row["seed"], row["query_id"]))
    detail = {}
    if trace_exists(cell / "steps.jsonl"):
        with open_trace(cell / "steps.jsonl") as handle:
            for line in handle:
                step = json.loads(line)
                if step.get("scored"):
                    detail[(step["seed"], step["observation_id"])] = step
    rows = []
    with open(cell / "episodes.jsonl") as handle:
        for line in handle:
            episode = json.loads(line)
            seed = episode["seed"]
            # The reader's mean prompt size in this episode: every prompt is counted in tokens_processed,
            # which also holds the controller's own model input when the controller is a language model.
            calls = episode.get("task_model_calls") or 0
            prompt_tokens = (episode["tokens_processed"] / calls
                             if calls and not episode.get("controller_model_calls") else None)
            for number in range(episode["queries"]):
                query_id = f"q{number:04d}"
                cluster, category, gold = lookup(seed, number)
                step = detail.get((seed, query_id))
                if step is not None and bool(step["correct"]) == ((seed, query_id) in failed):
                    raise ValueError(f"{cell}: steps and failures disagree on {seed}/{query_id}")
                rows.append({
                    "cluster": cluster, "question": f"{cluster}:{number}", "category": category,
                    "correct": float((seed, query_id) not in failed),
                    "f1": token_f1(step.get("agent_action") or "", gold) if step else None,
                    "tokens": prompt_tokens,
                    "truncated": (float(is_truncated(step.get("agent_output"), limit)) if step and step.get("agent_output")
                                  is not None else (episode.get("agent_truncated", 0) / calls if calls else 0.0)),
                    # the controller's own job: share of needed items in view when asked (episode level)
                    "evidence": episode.get("needed_hit_rate"),
                })
    return rows


def is_truncated(output: str | None, limit: int) -> bool:
    """No "Answer:" marker and an output near the limit: the note ran out of room. The limit is in the
    reader's tokens (subword pieces, more than words), so 70% of it in words-and-punctuation counts as at it."""
    if not output or "answer:" in output.lower() or not limit:
        return False
    return len(re.findall(r"\w+|[^\w\s]", output)) >= 0.7 * limit


def group_of(label: str) -> str:
    """The controller group of a cell label: seeds (`_s0`) and LongMemEval folds (`fold3__`) pooled."""
    label = re.sub(r"^fold\d+__", "", label)
    stem, _, seed = label.rpartition("_s")
    return stem if stem and seed.isdigit() else label


def bootstrap_interval(rows: list[dict], samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of accuracy, resampling clusters with their questions."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster"]].append(row["correct"])
    groups = list(clusters.values())
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        drawn = rng.choices(groups, k=len(groups))
        total = sum(len(g) for g in drawn)
        means.append(sum(sum(g) for g in drawn) / total if total else 0.0)
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def paired_difference(rows: list[dict], baseline: list[dict], samples: int = 2000, seed: int = 0,
                      unit: str = "cluster") -> dict:
    """Accuracy difference to a baseline on the same questions, with a 95% interval from resampling
    clusters. Each side is averaged over its seeds within a cluster first, so groups with different
    numbers of seeds compare fairly; clusters are weighted by their number of questions."""
    def per_cluster(data):
        sums = defaultdict(lambda: [0.0, 0])
        for row in data:
            sums[row[unit]][0] += row["correct"]
            sums[row[unit]][1] += 1
        return sums

    a, b = per_cluster(rows), per_cluster(baseline)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"mean": None, "ci": (None, None)}
    # Weight = the cluster's question count (one side's; with pooled seeds the counts are multiples).
    weight_of = {c: min(a[c][1], b[c][1]) for c in shared}
    diff_of = {c: a[c][0] / a[c][1] - b[c][0] / b[c][1] for c in shared}

    def estimate(clusters):
        total = sum(weight_of[c] for c in clusters)
        return sum(weight_of[c] * diff_of[c] for c in clusters) / total

    rng = random.Random(seed)
    draws = sorted(estimate(rng.choices(shared, k=len(shared))) for _ in range(samples))
    return {"mean": estimate(shared), "ci": (draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]),
            "clusters": len(shared)}


def summarise(rows: list[dict]) -> dict:
    tokens = [r["tokens"] for r in rows if r["tokens"] is not None]
    headline = [r for r in rows if r["category"] not in REFUSAL_SCORED]
    refusal = [r for r in rows if r["category"] in REFUSAL_SCORED]
    mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
    low, high = bootstrap_interval(headline) if headline else (None, None)
    categories = sorted({r["category"] for r in headline})
    return {
        "n": len(headline), "accuracy": mean([r["correct"] for r in headline]), "ci": (low, high),
        "f1": mean([r["f1"] for r in headline if r["f1"] is not None]),
        "f1_n": sum(r["f1"] is not None for r in headline),
        "refusal_n": len(refusal), "refusal_accuracy": mean([r["correct"] for r in refusal]),
        "tokens": mean(tokens),
        "truncated": mean([r["truncated"] for r in rows]),
        "evidence": mean([r["evidence"] for r in headline if r.get("evidence") is not None]),
        "tokens_median": sorted(tokens)[len(tokens) // 2] if tokens else None,
        "by_category": {c: mean([r["correct"] for r in headline if r["category"] == c]) for c in categories},
    }


def table(sweeps: list[Path], per_category: bool = False, exclude: tuple[str, ...] = (), baseline: str | None = None) -> str:
    cells: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for sweep in sweeps:
        for cell in sorted(p for p in sweep.iterdir() if (p / "episodes.jsonl").exists()):
            if cell.name in exclude:
                continue
            label, _, fraction = cell.name.partition("__fraction")
            cells[(group_of(label), float(fraction or "nan"))] += cell_rows(cell)
    summaries = {key: summarise(rows) for key, rows in cells.items()}
    if baseline:
        for (group, fraction), summary in summaries.items():
            reference = cells.get((baseline, fraction))
            headline = lambda data: [r for r in data if r["category"] not in REFUSAL_SCORED]  # noqa: E731
            pair = (headline(cells[(group, fraction)]), headline(reference)) if reference and group != baseline else None
            summary["paired"] = paired_difference(*pair) if pair else None
            summary["paired_q"] = paired_difference(*pair, unit="question") if pair else None
    categories = sorted({c for s in summaries.values() for c in s["by_category"]})
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    head = ["controller", "budget", "n", "accuracy (95% CI)", "evidence in view", "F1 (n)", "refusal-scored", "prompt tokens (mean / median)", "truncated"]
    if baseline:
        head += [f"Δ vs {baseline} (paired 95% CI, clusters)", "CI resampling questions"]
    if per_category:
        head += categories
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (group, fraction), s in sorted(summaries.items(), key=lambda kv: (kv[0][1], -(kv[1]["accuracy"] or 0))):
        row = [group, f"{fraction:g}", str(s["n"]), f"{fmt(s['accuracy'])} ({fmt(s['ci'][0])}–{fmt(s['ci'][1])})", fmt(s["evidence"]),
               f"{fmt(s['f1'])} ({s['f1_n']})", f"{fmt(s['refusal_accuracy'])} (n={s['refusal_n']})", "-" if s["tokens"] is None else f"{s['tokens']:.0f} / {s['tokens_median']:.0f}",
               f"{s['truncated']:.1%}"]
        if baseline:
            d = s.get("paired")
            q = s.get("paired_q")
            row.append("-" if not d or d["mean"] is None else f"{d['mean']:+.3f} ({d['ci'][0]:+.3f}–{d['ci'][1]:+.3f})")
            row.append("-" if not q or q["mean"] is None else f"({q['ci'][0]:+.3f}–{q['ci'][1]:+.3f})")
        if per_category:
            row += [fmt(s["by_category"].get(c)) for c in categories]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", nargs="+", type=Path)
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=(), help="cell directory names to leave out")
    parser.add_argument("--baseline", help="controller group to compare every row against, paired by question")
    args = parser.parse_args()
    print(table(args.sweeps, args.per_category, tuple(args.exclude), args.baseline))


if __name__ == "__main__":
    main()
