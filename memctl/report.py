"""Turns a run folder into report.md plus charts:  python -m memctl.report runs/<folder>

The report compares the run with its "siblings": other runs in the same parent
folder with the same benchmark, model and seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from memctl import plots
from memctl.attribution import EXPLANATIONS, LABELS, MEMORY_FAILURES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_run(folder: Path) -> dict:
    config = yaml.safe_load((folder / "config.yaml").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    fraction = config["budget"].get("fraction")
    return {"folder": folder, "config": config, "metrics": metrics, "budget_percent": 100 * fraction if fraction else None}


def siblings(run: dict) -> list[dict]:
    """Runs next to this one that can be compared with it (same benchmark, questions, model, seed)."""
    def key(r: dict) -> tuple:
        return (json.dumps(r["config"]["benchmark"], sort_keys=True), r["config"].get("model", {}).get("name"),
                r["config"].get("seed"), (r["config"].get("judge") or {}).get("name"))

    newest: dict[tuple, dict] = {}  # one run per (controller, budget): the most recent
    folders = [f for f in run["folder"].parent.iterdir() if (f / "metrics.json").exists()]
    for folder in sorted(folders, key=lambda f: (f / "metrics.json").stat().st_mtime):
        other = load_run(folder)
        if key(other) == key(run):
            newest[other["metrics"]["controller"], json.dumps(other["config"]["budget"])] = other
    newest[run["metrics"]["controller"], json.dumps(run["config"]["budget"])] = run
    return sorted(newest.values(), key=lambda r: r["folder"].name.split("_", 1)[1])


def closed_gap(score: float, baseline: float, oracle: float) -> float | None:
    """(method - keep_newest) / (oracle - keep_newest). None when the oracle is not better."""
    if oracle - baseline <= 1e-9:
        return None
    return round((score - baseline) / (oracle - baseline), 3)


def table(header: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def headline(run: dict, same_budget: list[dict]) -> str:
    by_name = {r["metrics"]["controller"]: r["metrics"] for r in same_budget}
    baseline, oracle = by_name.get("keep_newest"), by_name.get("oracle")
    has_judge = "judge" in run["metrics"]["answer_quality"]
    rows = []
    for name, m in by_name.items():
        q, gap = m["answer_quality"], None
        if baseline and oracle:
            gap = closed_gap(q["accuracy"], baseline["answer_quality"]["accuracy"], oracle["answer_quality"]["accuracy"])
        marker = " **(this run)**" if name == run["metrics"]["controller"] else ""
        rows.append([name + marker, q["accuracy"], q["f1"], q["bleu1"], *([q.get("judge")] if has_judge else []),
                     m["evidence_availability"]["questions_with_all_evidence_in_prompt"],
                     m["cost"]["avg_prompt_tokens"], gap])
    header = ["Controller", "Accuracy", "F1", "BLEU-1", *(["Judge"] if has_judge else []),
              "Questions with all evidence in prompt", "Avg prompt tokens", "Closed gap (accuracy)"]
    return table(header, rows)


def attribution_section(metrics: dict) -> str:
    a = metrics["failure_attribution"]
    wrong = a["wrong_answers"]
    if wrong == 0:
        return "No answers were wrong."
    lines = [
        f"Of {metrics['questions']} questions, **{wrong} were answered wrongly**. Of those:",
        "",
        f"- **{a['memory_failures']} ({a['memory_failures'] / wrong:.0%}) were memory failures**: "
        "a needed item was not in front of the model.",
        f"- **{a['reasoning_failures']} ({a['reasoning_failures'] / wrong:.0%}) were reasoning failures**: "
        "every needed item was in the prompt and the model still answered wrongly.",
        f"- {a['unknown']} ({a['unknown'] / wrong:.0%}) cannot be classified (no evidence label).",
        "",
        "![Failure attribution](failure_attribution.png)",
        "",
        table(["Label", "Kind", "Wrong answers", "Meaning"],
              [[f"`{label}`", "memory" if label in MEMORY_FAILURES else "reasoning" if "present" in label else "unknown",
                a["counts"][label], EXPLANATIONS[label]] for label in LABELS]),
    ]
    return "\n".join(lines)


def worked_examples(answers: list[dict], decisions: list[dict], number: int = 3) -> str:
    """Wrong answers traced back to the decision that caused them. One per failure label if possible."""
    wrong = [a for a in answers if not a["correct"]]
    chosen, seen = [], set()
    for label in LABELS:  # memory failures come first in LABELS
        for answer in wrong:
            if answer["failure_label"] == label and label not in seen and len(chosen) < number:
                chosen.append(answer)
                seen.add(label)
    chosen += [a for a in wrong if a not in chosen][: number - len(chosen)]
    if not chosen:
        return "No wrong answers to show."

    moves: dict[tuple, dict] = {}  # (conversation, item) -> the last decision that moved it
    for d in decisions:
        if d["kind"] == "move":
            moves[d["conversation_id"], d["item_id"]] = d
    parts = []
    for n, a in enumerate(chosen, start=1):
        parts += [
            f"### Example {n}: `{a['failure_label']}`",
            "",
            f"- **Question** ({a['category']}, `{a['question_id']}`): {a['question']}",
            f"- **Gold answer:** {a['gold_answer']}",
            f"- **Model answer:** {a['model_answer'] or '(empty)'}",
            f"- **Why it was wrong:** {EXPLANATIONS[a['failure_label']]}",
            "",
        ]
        for e in a["evidence"]:
            status = "in the prompt" if e["in_prompt"] else "**not in the prompt**"
            parts.append(f"Evidence `{e['item_id']}` was in `{e['place']}`, {status}:")
            parts.append(f"> {e['text']}")
            d = moves.get((a["conversation_id"], e["item_id"]))
            if d and d["place"] != "CONTEXT":
                f = d["features"]
                oracle = f" The oracle would have chosen `{d['oracle_place']}`." if d["oracle_place"] else ""
                parts.append(
                    f"\nDecision that moved it: at step {d['step']} ({d['trigger']}), `{d['controller']}` put it in "
                    f"`{d['place']}`. Reason given: \"{d['reason']}\". At that moment the item was {f['age_steps']} "
                    f"steps old, {f['tokens']} tokens, used {f['times_referenced']} times, similarity to the "
                    f"current message {f['similarity_to_query']}.{oracle}"
                )
            else:
                parts.append("\nNo decision moved this item: it stayed in context.")
            parts.append("")
    return "\n".join(parts)


def write_report(folder: Path) -> Path:
    run = load_run(Path(folder))
    folder, metrics = run["folder"], run["metrics"]
    answers, decisions = read_jsonl(folder / "answers.jsonl"), read_jsonl(folder / "decisions.jsonl")
    family = siblings(run)
    same_budget = [r for r in family if r["config"]["budget"] == run["config"]["budget"]]
    name = metrics["controller"]

    plots.failure_attribution_chart(
        metrics["failure_attribution"]["counts"], f"Why answers were wrong ({name})", folder / "failure_attribution.png")
    plots.places_chart({r["metrics"]["controller"]: r["metrics"]["final_places"] for r in same_budget},
                       "Where items were at the end of the conversation", folder / "items_by_place.png")
    # Cost is what the model actually read: average prompt tokens, including retrieved and recalled
    # items. The budget B is only a label on each point, because retrieval is not charged to B.
    curves: dict[str, list[tuple[float, float, str]]] = {}
    for r in family:
        point = (r["metrics"]["cost"]["avg_prompt_tokens"], r["metrics"]["answer_quality"]["accuracy"], budget_tag(r))
        curves.setdefault(r["metrics"]["controller"], []).append(point)
    plots.score_vs_cost_chart(curves, "accuracy", "Accuracy against what the model had to read", folder / "accuracy_vs_cost.png")
    budget_text = "![Accuracy vs prompt tokens](accuracy_vs_cost.png)\n\n" + table(
        ["Controller", "Budget B", "Avg prompt tokens", "Accuracy", "F1"],
        [[r["metrics"]["controller"], budget_tag(r), r["metrics"]["cost"]["avg_prompt_tokens"],
          r["metrics"]["answer_quality"]["accuracy"], r["metrics"]["answer_quality"]["f1"]] for r in family])

    q, c, agreement = metrics["answer_quality"], metrics["cost"], metrics["oracle_agreement"]
    fake = "\n> **FAKE JEV.** This run used the fake Jev client. It says nothing about the real Jev model.\n" if metrics["fake_jev"] else ""
    text = f"""# Report: `{name}` on {metrics['benchmark']} at budget {plots_budget(run)}
{fake}
- Conversations: {metrics['conversations']}, questions: {metrics['questions']}
- Answering model: `{metrics['model']}`. Right or wrong decided by: {answers[0]['correct_decided_by'] if answers else 'n/a'}
- Seed: {metrics['seed']}. Run folder: `{folder.name}`

## 1. Headline

All runs below share the same benchmark, questions, model, seed and budget.
"Closed gap" is `(accuracy - keep_newest) / (oracle - keep_newest)`: 0 means no better than
keep_newest, 1 means as good as the hindsight oracle.

{headline(run, same_budget)}

## 2. Why answers were wrong: memory or reasoning?

{attribution_section(metrics)}

## 3. Accuracy against cost

The cost axis is the **average number of tokens in the prompt**, counting everything the model
read: context, archive index, retrieved and recalled items. The budget `B` limits only the
context, so two controllers with the same `B` can have very different real costs.

{budget_text}

## 4. Where items ended up

![Where items ended up](items_by_place.png)

{table(["Controller", "CONTEXT", "STORE", "ARCHIVE", "DROPPED"], [[r["metrics"]["controller"], *[r["metrics"]["final_places"].get(p, 0) for p in plots.PLACE_COLORS]] for r in same_budget])}

## 5. Agreement with the oracle

For each decision this controller made, the oracle's choice for the same item at the same step
was recorded. Leaving an item in context counts as a decision too ("keep").
Overall agreement: **{agreement['agreement'] if agreement['agreement'] is not None else 'n/a'}** over {agreement['decisions']} graded choices
(moves: {agreement['moves']['agreement']} over {agreement['moves']['decisions']}; keeps: {agreement['keeps']['agreement']} over {agreement['keeps']['decisions']}).

{table(["When the oracle chose", "Decisions", "This controller agreed"], [[place, v["decisions"], v["agreement"]] for place, v in agreement["when_oracle_chose"].items()])}

## 6. Scores by question category

{table(["Category", "Questions", "Accuracy", "F1", "BLEU-1"], [[k, v["questions"], v["accuracy"], v["f1"], v["bleu1"]] for k, v in q["by_category"].items()])}

## 7. Where the evidence was when questions were asked

{table(["Evidence items were...", "Share"], [[k.replace("_", " "), v] for k, v in metrics["evidence_availability"].items() if k != "evidence_items"])}

## 8. Cost

{table(["Measure", "Value"], [[k.replace("_", " "), v] for k, v in c.items() if not isinstance(v, dict)])}

## 9. Worked examples: wrong answers traced to their cause

{worked_examples(answers, decisions)}
"""
    (folder / "report.md").write_text(text)
    return folder / "report.md"


def budget_tag(run: dict) -> str:
    if run["metrics"]["controller"] == "full_context":
        return "unlimited"
    budget = run["config"]["budget"]
    return f"{budget['tokens']} tok" if "tokens" in budget else f"{budget['fraction']:.0%}"


def plots_budget(run: dict) -> str:
    if run["metrics"]["controller"] == "full_context":
        return "unlimited (full_context ignores the budget)"
    budget = run["config"]["budget"]
    return f"{budget['tokens']} tokens" if "tokens" in budget else f"{budget['fraction']:.0%} of the full history"


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        print(write_report(Path(folder)))
