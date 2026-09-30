"""Turn a folder of runs into a report:  python -m memctl.analysis.report runs/<sweep>

Writes `report.md` (tables) and `report.json` (the same numbers) into the folder.
Runs are grouped into *conditions*: sets of runs that differ only in controller
and budget. Controllers are always compared inside one condition, at one budget,
on the same seeded episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from memctl.analysis.load import Run, condition_keys, condition_label, condition_of, condition_sort_key, load_runs
from memctl.analysis.stats import minimum_detectable_effect, oracle_gap_closed, paired_bootstrap, paired_differences
from memctl.analysis.summarize import bootstrap_ci, mean

ORACLES = ("oracle_exact", "oracle_approx", "oracle")
REFERENCES = ORACLES + ("full_context",)
BASELINES = ("no_controller", "fifo")


def budget_key(run: Run) -> str:
    budget = run.config["memory"]["budget"]
    return f"{budget['fraction'] * 100:g}%" if "fraction" in budget else f"{budget['tokens']} tok"


def _sort_budget(label: str) -> float:
    return float(label.rstrip("%").split()[0])


def group(runs: list[Run]) -> dict[tuple, dict[str, dict[str, Run]]]:
    """condition -> budget -> controller -> run."""
    keys = condition_keys(runs)
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        grouped[condition_of(run, keys)][budget_key(run)][run.controller] = run
    return grouped


def pick(cell: dict[str, Run], names: tuple[str, ...]) -> Run | None:
    return next((cell[name] for name in names if name in cell), None)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def controller_order(cells: dict[str, dict[str, Run]]) -> list[str]:
    names = sorted({name for cell in cells.values() for name in cell})
    rank = {name: (2 if name in REFERENCES else 0 if name in BASELINES else 1) for name in names}
    return sorted(names, key=lambda name: (rank[name], name))


def analyse(runs: list[Run], baseline: str | None = None) -> dict:
    """Every number in the report, as nested dictionaries."""
    data: dict = {"conditions": []}
    for condition, cells in sorted(group(runs).items(), key=lambda pair: condition_sort_key(pair[0])):
        budgets = sorted(cells, key=_sort_budget)
        controllers = controller_order(cells)
        entry: dict = {
            "condition": condition_label(condition), "budgets": budgets, "controllers": controllers, "cells": {},
        }
        for budget in budgets:
            cell = cells[budget]
            base = cell.get(baseline) if baseline else pick(cell, BASELINES)
            oracle = pick(cell, ORACLES)
            for name, run in cell.items():
                values = [episode["task_success"] for episode in run.episodes]
                record = {
                    "episodes": len(values),
                    "task_success": mean(values),
                    "ci95": bootstrap_ci(values),
                    "summary": run.summary,
                    "folder": str(run.folder),
                }
                if base is not None and run is not base:
                    differences = paired_differences(run.by_seed(), base.by_seed())
                    record["vs_baseline"] = {
                        "baseline": base.controller, **paired_bootstrap(differences),
                        "mde": minimum_detectable_effect(differences),
                    }
                if oracle is not None and base is not None and name not in REFERENCES:
                    mine, theirs, best = run.by_seed(), base.by_seed(), oracle.by_seed()
                    shared = sorted(set(mine) & set(theirs) & set(best))  # paired: the same episodes for all three
                    record["oracle_gap_closed"] = oracle_gap_closed(
                        mean([mine[s] for s in shared]), mean([theirs[s] for s in shared]), mean([best[s] for s in shared])
                    ) if shared else None
                    record["regret_vs_oracle"] = mean(paired_differences(oracle.by_seed(), run.by_seed()))
                    record["oracle"] = oracle.controller
                entry["cells"].setdefault(budget, {})[name] = record
        data["conditions"].append(entry)
    return data


def render(data: dict, title: str) -> str:
    out = [f"# {title}", ""]
    out.append(
        "Task success is the mean over episodes of the share of queries answered correctly. "
        "Intervals are 95% bootstrap intervals over episodes. Differences are paired by seed."
    )
    for entry in data["conditions"]:
        budgets, controllers, cells = entry["budgets"], entry["controllers"], entry["cells"]
        get = lambda budget, name: cells.get(budget, {}).get(name)  # noqa: E731
        out += ["", f"## Condition: {entry['condition']}", ""]

        out += ["### Task success by memory budget", ""]
        rows = []
        for name in controllers:
            row = [name]
            for budget in budgets:
                record = get(budget, name)
                if record is None:
                    row.append("–")
                elif record["ci95"]:
                    row.append(f"{record['task_success']:.3f} [{record['ci95'][0]:.3f}, {record['ci95'][1]:.3f}]")
                else:
                    row.append(f"{record['task_success']:.3f}")
            rows.append(row)
        out.append(_table(["controller"] + budgets, rows))

        out += ["", "### Difference from the baseline, paired by episode", ""]
        rows = []
        for budget in budgets:
            for name in controllers:
                record = get(budget, name)
                if record and "vs_baseline" in record:
                    versus = record["vs_baseline"]
                    solid = versus["ci_low"] is not None and (versus["ci_low"] > 0 or versus["ci_high"] < 0)
                    rows.append(
                        [budget, name, versus["baseline"], versus["mean"],
                         f"[{_fmt(versus['ci_low'])}, {_fmt(versus['ci_high'])}]", versus["mde"],
                         "yes" if solid else "no", versus["n"]]
                    )
        out.append(
            _table(["budget", "controller", "baseline", "difference", "95% interval",
                    "min. detectable", "interval excludes 0", "episodes"], rows)
        )

        gap_rows = []
        for name in controllers:
            values = [get(budget, name) for budget in budgets]
            if any(record and "oracle_gap_closed" in record for record in values):
                gap_rows.append([name] + [record.get("oracle_gap_closed") if record else None for record in values])
        if gap_rows:
            oracle_name = next(
                record["oracle"] for budget in budgets for record in cells[budget].values() if "oracle" in record
            )
            out += ["", f"### Fraction of the oracle gap closed (oracle: {oracle_name})", ""]
            out.append("(controller − baseline) / (oracle − baseline). Blank where the oracle leaves no gap.")
            out += ["", _table(["controller"] + budgets, gap_rows)]
            regret_rows = [
                [name] + [(get(b, name) or {}).get("regret_vs_oracle") for b in budgets]
                for name in controllers if name not in REFERENCES
            ]
            out += ["", "### Regret relative to the oracle (oracle success − controller success)", ""]
            out.append(_table(["controller"] + budgets, regret_rows))

        out += ["", "### Cost: memory, latency, fallback", ""]
        rows = []
        for budget in budgets:
            for name in controllers:
                record = get(budget, name)
                if record is None:
                    continue
                s = record["summary"]
                latency = s.get("controller_latency_per_step_s")
                rows.append(
                    [budget, name, record["task_success"], s.get("active_tokens_mean"), s.get("active_tokens_peak"),
                     s.get("archive_tokens_peak"), None if latency is None else latency * 1e3,
                     s.get("forced_fallback_rate"), s.get("forced_share_of_removals"), s.get("invalid_actions"),
                     s.get("unnecessary_token_share"), s.get("estimated_cost_usd")]
                )
        out.append(
            _table(["budget", "controller", "success", "active tok (mean)", "active tok (peak)", "archive tok (peak)",
                    "controller ms/step", "steps with forced eviction", "forced share of removals",
                    "invalid actions/ep", "share of active tokens never needed", "est. cost $"], rows)
        )

        out += ["", "### Failure attribution (share of failed queries)", ""]
        labels = sorted({label for b in budgets for r in cells[b].values() for label in r["summary"].get("failures", {})})
        if labels:
            rows = []
            for budget in budgets:
                for name in controllers:
                    record = get(budget, name)
                    if record is None:
                        continue
                    shares = record["summary"].get("failure_shares", {})
                    failed = sum(record["summary"].get("failures", {}).values())
                    rows.append([budget, name, failed] + [shares.get(label, 0.0) if failed else None for label in labels])
            out.append(_table(["budget", "controller", "failed queries"] + labels, rows))
        else:
            out.append("No failed queries.")

        out += ["", "### Items acted on per episode (an action with several targets counts each of them)", ""]
        operations = sorted(
            {op for b in budgets for r in cells[b].values()
             for counts in r["summary"].get("action_counts", {}).values() for op in counts}
        )
        rows = []
        for budget in budgets:
            for name in controllers:
                record = get(budget, name)
                if record is None:
                    continue
                counts, episodes = record["summary"].get("action_counts", {}), record["episodes"]
                chosen, harness = counts.get("controller", {}), counts.get("harness", {})
                rows.append(
                    [budget, name] + [chosen.get(op, 0) / episodes for op in operations]
                    + [harness.get("EVICT", 0) / episodes, harness.get("MOVE_TO_ARCHIVE", 0) / episodes,
                       sum(counts.get("rejected", {}).values()) / episodes]
                )
        out.append(_table(["budget", "controller"] + operations + ["forced EVICT", "forced return to archive", "rejected"], rows))

        retrieval_rows = []
        for budget in budgets:
            for name in controllers:
                record = get(budget, name)
                if record and record["summary"].get("retrieved_items"):
                    s = record["summary"]
                    retrieval_rows.append(
                        [budget, name, s.get("retrievals"), s.get("retrieved_items"), s.get("retrieval_precision"),
                         s.get("retrieval_recall"), s.get("needed_hit_rate")]
                    )
        if retrieval_rows:
            out += ["", "### Retrieval", ""]
            out.append(
                _table(["budget", "controller", "retrieve actions/ep", "items retrieved/ep", "precision", "recall",
                        "needed-memory hit rate"], retrieval_rows)
            )

        shadow_rows = []
        for budget in budgets:
            for name in controllers:
                record = get(budget, name)
                for shadow, value in ((record or {}).get("summary", {}).get("shadow_agreement") or {}).items():
                    shadow_rows.append([budget, name, shadow, value])
        if shadow_rows:
            out += ["", "### Agreement with shadow controllers (overlap of removed items, 1 = identical)", ""]
            out.append(_table(["budget", "controller", "shadow", "agreement"], shadow_rows))
    return "\n".join(out) + "\n"


def misclassified(runs: list[Run], top: int = 10) -> list[dict]:
    """Which kinds of memory are destroyed while still needed, from regret.jsonl and items.jsonl.

    Item details exist only for detail episodes, so source types cover those; the
    time-to-need buckets cover every episode.
    """
    rows = []
    for run in runs:
        regrets = run.rows("regret")
        if not regrets:
            continue
        items = {(row["episode_id"], row["id"]): row for row in run.rows("items")}
        buckets, sources, operations = Counter(), Counter(), Counter()
        for regret in regrets:
            gap = regret["steps_until_needed"]
            buckets["<10" if gap < 10 else "10-49" if gap < 50 else "50-199" if gap < 200 else "200+"] += 1
            operations[f"{regret['source']}:{regret['operation']}"] += 1
            for item_id in regret["target_ids"]:
                item = items.get((regret["episode_id"], item_id))
                if item:
                    sources[item["source_type"]] += 1
        rows.append(
            {
                "controller": run.controller, "budget": budget_key(run), "folder": str(run.folder),
                "destroyed_requirements": len(regrets), "per_episode": len(regrets) / len(run.episodes),
                "steps_until_needed": dict(buckets), "by_source_type": dict(sources), "by_operation": dict(operations),
            }
        )
    rows.sort(key=lambda row: -row["per_episode"])
    return rows[:top] if top else rows


def build_report(root: str | Path, baseline: str | None = None, title: str | None = None) -> tuple[str, dict]:
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no runs found under {root}")
    data = analyse(runs, baseline)
    text = render(data, title or f"Report: {Path(root).name}")
    wrong = misclassified(runs, top=0)
    if wrong:
        text += "\n## Needed memories destroyed, by run\n\n"
        text += _table(
            ["budget", "controller", "requirements destroyed/ep", "steps until needed", "source type (detail episodes)", "by operation"],
            [[w["budget"], w["controller"], w["per_episode"], json.dumps(w["steps_until_needed"]),
              json.dumps(w["by_source_type"]), json.dumps(w["by_operation"])] for w in wrong],
        ) + "\n"
    data["misclassified"] = wrong
    return text, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on a folder of runs.")
    parser.add_argument("root", help="a run folder or a sweep folder")
    parser.add_argument("--baseline", help="controller to compare against (default: no_controller, else fifo)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    text, data = build_report(args.root, args.baseline)
    root = Path(args.root)
    (root / "report.md").write_text(text)
    (root / "report.json").write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {root / 'report.md'}")
    if not args.no_plots:
        from memctl.analysis.plots import plot_all

        for path in plot_all(root):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
