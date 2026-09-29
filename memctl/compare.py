"""Compare the runs in one folder:  python -m memctl.compare runs/locomo_full

Writes one table per budget (accuracy, F1, BLEU-1, average prompt tokens and the
memory / reasoning split for every controller), then the per-question comparison of
`keep_newest` against `oracle` with an exact paired test.

Everything is recomputed from `answers.jsonl` using the same functions that build
`metrics.json` (`memctl.summary`), so the tables cannot drift from the run's own
metrics. Nothing here changes a score, a placement or a failure label: it only reads.

Every number is reported twice, over all questions and over the non-adversarial ones,
because whether adversarial questions belong in the headline is still open
(docs/decisions.md has no entry for it yet).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

from memctl.summary import answer_quality, failure_attribution

BASELINE = "keep_newest"
ORACLE = "oracle"
SUBSETS = [
    ("all questions", lambda row: True),
    ("without adversarial questions", lambda row: row["category"] != "adversarial"),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_runs(folder: Path) -> list[dict]:
    """Every run under `folder`, keeping the newest run per (controller, budget)."""
    runs: dict[tuple[str, str], dict] = {}
    candidates = [f for f in sorted(folder.iterdir()) if (f / "metrics.json").exists()]
    for path in sorted(candidates, key=lambda f: (f / "metrics.json").stat().st_mtime):
        metrics = json.loads((path / "metrics.json").read_text())
        config = yaml.safe_load((path / "config.yaml").read_text())
        runs[metrics["controller"], budget_label(config["budget"])] = {
            "folder": path,
            "controller": metrics["controller"],
            "budget": budget_label(config["budget"]),
            "ignores_budget": metrics["controller"] == "full_context",
            "metrics": metrics,
            "answers": read_jsonl(path / "answers.jsonl"),
        }
    return list(runs.values())


def budget_label(budget: dict) -> str:
    return f"{budget['tokens']} tokens" if "tokens" in budget else f"{round(budget['fraction'] * 100)}%"


def budget_order(label: str) -> float:
    """Sort budgets by size, with full_context's 'unlimited' last."""
    digits = "".join(c for c in label if c.isdigit())
    return float(digits) if digits else math.inf


# ----------------------------------------------------------------- one budget, one table


def controller_row(run: dict, keep: object) -> list:
    """One table row: the scores of this run over the questions `keep` accepts."""
    answers = [row for row in run["answers"] if keep(row)]
    if not answers:
        return [run["controller"], *([None] * 6), 0]
    quality = answer_quality(answers)
    failures = failure_attribution(answers)
    prompt_tokens = sum(row["prompt_tokens"] for row in answers) / len(answers)
    return [
        run["controller"],
        quality["accuracy"],
        quality["f1"],
        quality["bleu1"],
        round(prompt_tokens, 1),
        failures["memory_failures"],
        failures["reasoning_failures"],
        len(answers),
    ]


def table(header: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) and value < 10 else str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def budget_section(runs: list[dict], budget: str, keep: object) -> str:
    """The per-controller table for one budget, baseline first and oracle last."""

    def order(run: dict) -> tuple:
        rank = {BASELINE: 0, ORACLE: 2}.get(run["controller"], 1)
        return (rank, run["controller"])

    here = sorted([r for r in runs if r["budget"] == budget], key=order)
    header = ["Controller", "Accuracy", "F1", "BLEU-1", "Avg prompt tokens",
              "Memory failures", "Reasoning failures", "Questions"]
    return table(header, [controller_row(run, keep) for run in here])


# ----------------------------------------------------------------- the paired test


def binomial_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided binomial test against p = 0.5 (this is McNemar's exact test).

    Doubling the smaller tail is the conventional form and is exact for p = 0.5,
    where the distribution is symmetric.
    """
    if trials == 0:
        return 1.0
    extreme = max(successes, trials - successes)
    tail = sum(math.comb(trials, k) for k in range(extreme, trials + 1)) / 2**trials
    return min(1.0, 2 * tail)


def paired(baseline: dict, oracle: dict, keep: object) -> dict | None:
    """The 2x2 table of right and wrong over the questions both runs answered."""
    mine = {row["question_id"]: row for row in baseline["answers"] if keep(row)}
    theirs = {row["question_id"]: row for row in oracle["answers"] if keep(row)}
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return None
    both_right = only_oracle = only_baseline = both_wrong = 0
    for question_id in shared:
        b, o = mine[question_id]["correct"], theirs[question_id]["correct"]
        both_right += b and o
        only_oracle += o and not b
        only_baseline += b and not o
        both_wrong += not b and not o
    discordant = only_oracle + only_baseline
    return {
        "questions": len(shared),
        "skipped": len(set(mine) ^ set(theirs)),
        "both_right": both_right,
        "only_oracle_right": only_oracle,
        "only_baseline_right": only_baseline,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "baseline_accuracy": both_right + only_baseline,
        "oracle_accuracy": both_right + only_oracle,
        "p_value": binomial_two_sided(only_oracle, discordant),
    }


def paired_section(result: dict | None) -> str:
    if result is None:
        return "No questions are shared by the two runs, so they cannot be compared."
    n = result["questions"]
    lines = [
        table(
            ["Outcome", "Questions", "Share"],
            [
                ["Both right", result["both_right"], result["both_right"] / n],
                [f"Only `{ORACLE}` right", result["only_oracle_right"], result["only_oracle_right"] / n],
                [f"Only `{BASELINE}` right", result["only_baseline_right"], result["only_baseline_right"] / n],
                ["Both wrong", result["both_wrong"], result["both_wrong"] / n],
            ],
        ),
        "",
        f"- Paired over **{n} questions**"
        + (f" ({result['skipped']} appear in only one run and are left out)" if result["skipped"] else "")
        + ".",
        f"- `{BASELINE}` {result['baseline_accuracy'] / n:.3f} against `{ORACLE}` "
        f"{result['oracle_accuracy'] / n:.3f}: a gap of "
        f"{(result['oracle_accuracy'] - result['baseline_accuracy']) / n:+.3f} "
        f"({result['oracle_accuracy'] - result['baseline_accuracy']:+d} questions).",
        f"- The {result['discordant']} questions the two runs disagree on split "
        f"{result['only_oracle_right']} to {result['only_baseline_right']} in favour of "
        f"`{ORACLE}`. McNemar's exact test: **p = {result['p_value']:.4f}**.",
        "",
        verdict(result),
    ]
    return "\n".join(lines)


def verdict(result: dict) -> str:
    """One plain sentence on whether the gap is solid."""
    p, gap = result["p_value"], result["oracle_accuracy"] - result["baseline_accuracy"]
    if result["discordant"] < 10:
        return (f"> **Not solid.** Only {result['discordant']} questions separate the two controllers, "
                "which is too few to conclude anything either way.")
    if p >= 0.05:
        return (f"> **Not solid.** p = {p:.4f} is above 0.05, so this gap is within what chance "
                f"would produce from {result['discordant']} disagreements.")
    direction = ORACLE if gap > 0 else BASELINE
    return (f"> **Solid.** p = {p:.4f}. `{direction}` is ahead by more than chance would explain, "
            f"on {result['discordant']} disagreements.")


# ----------------------------------------------------------------- the document


def write_comparison(folder: Path) -> Path:
    runs = load_runs(folder)
    if not runs:
        raise SystemExit(f"no runs with a metrics.json found in {folder}")
    budgets = sorted({r["budget"] for r in runs}, key=budget_order)
    models = sorted({r["metrics"]["model"] for r in runs})
    quantised = sorted({str(yaml.safe_load((r["folder"] / "config.yaml").read_text())
                            .get("model", {}).get("load_in_4bit", False)) for r in runs})

    out = [
        f"# Comparison of the runs in `{folder}`",
        "",
        f"- Runs: {len(runs)}. Controllers: "
        + ", ".join(f"`{c}`" for c in sorted({r['controller'] for r in runs}))
        + f". Budgets: {', '.join(budgets)}.",
        f"- Answering model: {', '.join(f'`{m}`' for m in models)}."
        + (" **4-bit weights are mixed in here; those numbers are not comparable with 16-bit ones.**"
           if "True" in quantised and len(quantised) > 1 else ""),
        "",
        "Both readings are given throughout: over all questions, and over the non-adversarial",
        "ones only. Adversarial questions are scored by a different rule (right only if the",
        "agent says the information is absent), so they move the averages on their own.",
    ]

    for name, keep in SUBSETS:
        out += ["", f"## Scores, {name}", ""]
        for budget in budgets:
            out += [f"### Budget {budget}", "", budget_section(runs, budget, keep), ""]

        out += [f"## `{BASELINE}` against `{ORACLE}`, {name}", ""]
        for budget in budgets:
            baseline = next((r for r in runs if r["controller"] == BASELINE and r["budget"] == budget), None)
            oracle = next((r for r in runs if r["controller"] == ORACLE and r["budget"] == budget), None)
            out += [f"### Budget {budget}", ""]
            if not baseline or not oracle:
                missing = [n for n, r in ((BASELINE, baseline), (ORACLE, oracle)) if not r]
                out += [f"Not available: no run for {', '.join(f'`{m}`' for m in missing)} at this budget.", ""]
                continue
            out += [paired_section(paired(baseline, oracle, keep)), ""]

    path = folder / "comparison.md"
    path.write_text("\n".join(out) + "\n")
    return path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m memctl.compare runs/<folder>")
    print(f"wrote {write_comparison(Path(sys.argv[1]))}")


if __name__ == "__main__":
    main()
