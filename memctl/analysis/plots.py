"""Figures for a folder of runs. Called by `python -m memctl.analysis.report <folder>`.

Every figure is written to `<folder>/plots/` as a PNG; the numbers behind each
one are in `report.md`, which is its table view.

Colour rules (fixed, so figures can be compared with each other):
- a controller keeps its colour in every figure; references (oracles, full
  context) are drawn in grey ink, never in a series colour
- at most eight coloured series per panel; the rest are folded into grey "other" lines
- an operation or a failure label keeps its colour in every figure
The palette is the validated eight-hue categorical order (checked with a
colour-vision-deficiency validator); marker shapes repeat the identity so it
never rests on colour alone.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from memctl.analysis.load import (  # noqa: E402
    Run, condition_keys, condition_label, condition_of, condition_sort_key, load_runs, read_jsonl,
)
from memctl.analysis.report import BASELINES, ORACLES, REFERENCES, _sort_budget, budget_key  # noqa: E402
from memctl.analysis.stats import oracle_gap_closed  # noqa: E402
from memctl.analysis.summarize import mean  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FIXED_CONTROLLERS = ["fifo", "lru", "lfu", "random", "age_decay", "salience", "similarity", "rl"]
REFERENCE_STYLE = {
    "oracle_exact": ("#0b0b0b", "*"), "oracle_approx": ("#52514e", "p"), "oracle": ("#52514e", "p"),
    "full_context": ("#898781", "_"),
}
OPERATION_COLOURS = {
    "EVICT": SLOTS[0], "MOVE_TO_ARCHIVE": SLOTS[1], "RETRIEVE_FROM_ARCHIVE": SLOTS[2], "COMPACT": SLOTS[3],
    "COMPACT_AND_ARCHIVE": SLOTS[4], "CONSOLIDATE": SLOTS[5], "rejected": SLOTS[6], "EVICT (forced)": SLOTS[7],
}
FAILURE_COLOURS = {
    "evicted": SLOTS[0], "archived_not_retrieved": SLOTS[1], "task_model_reasoning": SLOTS[2],
    "compression_lost_detail": SLOTS[3], "retrieved_but_ignored": SLOTS[4], "consolidation_incorrect": SLOTS[5],
    "invalid_action": SLOTS[6], "unknown": MUTED,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif", "font.size": 9, "figure.titleweight": "bold", "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0, "lines.markersize": 6.5,
    }
)


def styles(names: list[str]) -> dict[str, dict]:
    """A colour and a marker for every controller name; the same name always gets the same pair."""
    found: dict[str, dict] = {}
    taken = set()
    ordinary = [name for name in names if name not in REFERENCE_STYLE]
    for name in ordinary:
        if name in FIXED_CONTROLLERS:
            slot = FIXED_CONTROLLERS.index(name)
            found[name] = {"color": SLOTS[slot], "marker": MARKERS[slot]}
            taken.add(slot)
    free = [slot for slot in range(len(SLOTS)) if slot not in taken]
    for name in sorted(name for name in ordinary if name not in found):
        if free:
            slot = free.pop(0)
            found[name] = {"color": SLOTS[slot], "marker": MARKERS[slot]}
        else:  # never a ninth hue: the rest recede
            found[name] = {"color": MUTED, "marker": ".", "other": True}
    for name in names:
        if name in REFERENCE_STYLE:
            colour, marker = REFERENCE_STYLE[name]
            found[name] = {"color": colour, "marker": marker, "reference": True}
    return found


def _line(ax, xs, ys, name, style):
    ax.plot(
        xs, ys, color=style["color"], marker=style["marker"], label=name, markeredgecolor=SURFACE, markeredgewidth=1.4,
        linewidth=1.4 if style.get("reference") or style.get("other") else 2.0,
        markersize=9 if style["marker"] == "*" else 6.5, zorder=2 if style.get("reference") else 3,
    )


def _panels(count: int, width: float = 4.6, height: float = 3.5):
    columns = min(3, max(1, count))
    rows = (count + columns - 1) // columns
    if count == 1:
        width = max(width, 6.4)
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows + 0.5), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[count:]:
        ax.set_visible(False)
    return figure, flat[:count]


def _finish(figure, axes, title: str, path: Path, legend: bool = True) -> Path:
    width, height = figure.get_size_inches()
    legend_rows = 0
    if legend:
        handles: dict = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        if len(handles) >= 2:
            # As many columns as fit the figure width, so long labels wrap instead of running off the edge.
            longest = max(len(label) for label in handles)
            columns = max(1, min(len(handles), 6, int(width * 9.5 / (longest + 7))))
            legend_rows = -(-len(handles) // columns)
            figure.legend(
                handles.values(), handles.keys(), loc="lower center", ncol=columns,
                bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4,
            )
    figure.suptitle(title, x=0.01, ha="left", fontsize=11)
    height += 0.22 * legend_rows  # the legend gets its own room instead of squeezing the panels
    figure.set_size_inches(width, height)
    bottom = (0.22 * legend_rows + 0.12) / height
    figure.tight_layout(rect=(0, bottom, 1, 1 - 0.32 / height))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _percent(value, _position=None) -> str:
    return f"{value * 100:g}%"


def _grouped(runs: list[Run]):
    """condition -> budget label -> controller -> run, plus the list of condition keys."""
    keys = condition_keys(runs)
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        grouped[condition_of(run, keys)][budget_key(run)][run.controller] = run
    return grouped


def _drop_duplicate_baseline(cells: dict) -> dict[str, str]:
    """no_controller and fifo are the same policy; draw one line and say so."""
    rename = {}
    same = all(
        "fifo" in cell and "no_controller" in cell
        and abs(cell["fifo"].summary["task_success"] - cell["no_controller"].summary["task_success"]) < 1e-12
        for cell in cells.values()
    )
    if same and cells:
        rename["fifo"] = "fifo (= no controller)"
    return rename


def plot_success_vs_budget(runs: list[Run], out: Path) -> Path | None:
    grouped = _grouped(runs)
    grouped = {c: cells for c, cells in grouped.items() if len(cells) >= 2 and all("%" in b for b in cells)}
    if not grouped:
        return None
    figure, axes = _panels(len(grouped))
    for ax, (condition, cells) in zip(axes, sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0]))):
        rename = _drop_duplicate_baseline(cells)
        budgets = sorted(cells, key=_sort_budget)
        names = sorted({name for cell in cells.values() for name in cell})
        look = styles(names)
        for name in names:
            if name == "no_controller" and rename:
                continue
            points = [(_sort_budget(b) / 100, cells[b][name].summary["task_success"]) for b in budgets if name in cells[b]]
            _line(ax, [x for x, _ in points], [y for _, y in points], rename.get(name, name), look[name])
        ax.set_xscale("log")
        ax.set_xticks([_sort_budget(b) / 100 for b in budgets])
        ax.xaxis.set_major_formatter(FuncFormatter(_percent))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("active-memory budget (share of uncompressed history)")
        ax.set_ylabel("task success")
        ax.set_title(condition_label(condition), loc="left")
    return _finish(figure, axes, "Task success against memory budget", out / "success_vs_budget.png")


def plot_success_vs_condition(runs: list[Run], out: Path, key: str, label: str, name: str) -> Path | None:
    """Task success against one numeric config key (e.g. env.horizon), one panel per budget."""
    from memctl.analysis.load import flatten

    values = {flatten(run.config).get(key) for run in runs}
    if len(values) < 2 or None in values:
        return None
    by_budget: dict = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        by_budget[budget_key(run)][run.controller][flatten(run.config)[key]] = run.summary["task_success"]
    budgets = sorted(by_budget, key=_sort_budget)[:6]
    figure, axes = _panels(len(budgets))
    for ax, budget in zip(axes, budgets):
        names = sorted(by_budget[budget])
        look = styles(names)
        for controller in names:
            if controller == "no_controller" and "fifo" in names:
                continue
            points = sorted(by_budget[budget][controller].items())
            _line(ax, [x for x, _ in points], [y for _, y in points], controller, look[controller])
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel(label)
        ax.set_ylabel("task success")
        ax.set_title(f"budget {budget}", loc="left")
    return _finish(figure, axes, f"Task success against {label}", out / name)


def plot_oracle_gap(runs: list[Run], out: Path) -> Path | None:
    grouped = {c: cells for c, cells in _grouped(runs).items() if len(cells) >= 2 and all("%" in b for b in cells)}
    panels = []
    for condition, cells in sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0])):
        series: dict[str, list] = defaultdict(list)
        for budget in sorted(cells, key=_sort_budget):
            cell = cells[budget]
            base = next((cell[n] for n in BASELINES if n in cell), None)
            oracle = next((cell[n] for n in ORACLES if n in cell), None)
            if base is None or oracle is None:
                continue
            for name, run in cell.items():
                if name in REFERENCES or run is base or name in BASELINES:
                    continue
                closed = oracle_gap_closed(run.summary["task_success"], base.summary["task_success"], oracle.summary["task_success"])
                if closed is not None:
                    series[name].append((_sort_budget(budget) / 100, closed))
        if series:
            panels.append((condition, series))
    if not panels:
        return None
    figure, axes = _panels(len(panels))
    for ax, (condition, series) in zip(axes, panels):
        look = styles(sorted(series))
        for name in sorted(series):
            _line(ax, [x for x, _ in series[name]], [y for _, y in series[name]], name, look[name])
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_percent))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(sorted({x for points in series.values() for x, _ in points}))
        ax.set_xlabel("active-memory budget")
        ax.set_ylabel("share of oracle gap closed")
        ax.set_title(condition_label(condition), loc="left")
    return _finish(figure, axes, "Oracle gap closed: (controller − baseline) / (oracle − baseline)", out / "oracle_gap_closed.png")


def _focus_cell(cells: dict) -> tuple[str, dict]:
    """The budget to show when a figure has room for one: the tightest with a spread of outcomes."""
    budgets = sorted(cells, key=_sort_budget)
    spread = lambda b: max(r.summary["task_success"] for r in cells[b].values()) - min(  # noqa: E731
        r.summary["task_success"] for r in cells[b].values() if r.controller not in REFERENCES or len(cells[b]) == 1
    )
    best = max(budgets, key=spread)
    return best, cells[best]


def plot_success_vs_latency(runs: list[Run], out: Path) -> Path | None:
    grouped = _grouped(runs)
    if not grouped:
        return None
    figure, axes = _panels(len(grouped), width=5.4, height=3.9)
    for ax, (condition, cells) in zip(axes, sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0]))):
        budget, cell = _focus_cell(cells)
        points = []
        for name, run in sorted(cell.items()):
            latency = (run.summary.get("controller_latency_per_step_s") or 0) * 1e3
            if latency > 0:
                points.append((latency, run.summary["task_success"], name))
        if not points:
            continue
        for latency, success, name in points:
            reference = name in REFERENCES
            ax.scatter([latency], [success], s=46, color=INK_2 if reference else SLOTS[0],
                       marker="D" if reference else "o", edgecolor=SURFACE, linewidth=1.4, zorder=3)
        # Labels go in a column right of the plot, spaced apart, each tied to its point by a leader line.
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        low, high = min(p[0] for p in points), max(p[0] for p in points)
        ax.set_xlim(low / 2, high * 2)
        ordered = sorted(points, key=lambda p: p[1])
        gap = 1.13 / max(len(ordered), 9)
        heights = []
        for _, success, _ in ordered:
            heights.append(max(success, heights[-1] + gap) if heights else max(success, -0.03))
        overflow = max(0.0, heights[-1] - 1.06)
        heights = [h - overflow * (n + 1) / len(heights) for n, h in enumerate(heights)]
        for (latency, success, name), height in zip(ordered, heights):
            ax.annotate(
                name, xy=(latency, success), xytext=(1.03, height), textcoords=ax.get_yaxis_transform(),
                fontsize=7.5, color=INK_2, va="center",
                arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.7, "shrinkB": 3},
            )
        ax.set_xlabel("controller latency (ms per step, log scale)")
        ax.set_ylabel("task success")
        ax.set_title(f"{condition_label(condition)} · budget {budget}", loc="left")
    return _finish(figure, axes, "Task success against controller latency (diamonds: references)",
                   out / "success_vs_latency.png", legend=False)


def plot_failure_attribution(runs: list[Run], out: Path) -> Path | None:
    grouped = _grouped(runs)
    panels = []
    for condition, cells in sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0])):
        budget, cell = _focus_cell(cells)
        rows = [(name, run) for name, run in sorted(cell.items()) if sum(run.summary.get("failures", {}).values())]
        if rows:
            panels.append((condition, budget, rows))
    if not panels:
        return None
    height = 0.34 * max(len(rows) for _, _, rows in panels) + 1.4
    figure, axes = _panels(len(panels), width=5.6, height=height)
    for ax, (condition, budget, rows) in zip(axes, panels):
        for position, (name, run) in enumerate(rows):
            failures = run.summary["failures"]
            queries = sum(e["queries"] for e in run.episodes)
            left = 0.0
            for label, colour in FAILURE_COLOURS.items():
                share = failures.get(label, 0) / queries
                if share:
                    ax.barh(position, share, left=left, height=0.55, color=colour, edgecolor=SURFACE, linewidth=1.6,
                            label=label.replace("_", " "))
                    left += share
        ax.set_yticks(range(len(rows)), [name for name, _ in rows])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FuncFormatter(_percent))
        causes = sorted({label for _, run in rows for label, count in run.summary["failures"].items() if count})
        only = f" (every failure: {causes[0].replace('_', ' ')})" if len(causes) == 1 else ", by cause"
        ax.set_xlabel(f"share of all queries that failed{only}")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{condition_label(condition)} · budget {budget}", loc="left")
    return _finish(figure, axes, "Why queries failed", out / "failure_attribution.png")


def plot_retrieval(runs: list[Run], out: Path) -> Path | None:
    grouped = _grouped(runs)
    panels = []
    for condition, cells in sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0])):
        budget, cell = _focus_cell(cells)
        rows = [(name, run) for name, run in sorted(cell.items()) if run.summary.get("retrieved_items")]
        if rows:
            panels.append((condition, budget, rows))
    if not panels:
        return None
    height = 0.5 * max(len(rows) for _, _, rows in panels) + 1.4
    figure, axes = _panels(len(panels), width=5.6, height=height)
    for ax, (condition, budget, rows) in zip(axes, panels):
        for offset, (metric, colour, label) in enumerate(
            [("retrieval_precision", SLOTS[0], "precision (retrieved items that were needed)"),
             ("retrieval_recall", SLOTS[1], "recall (needed archived evidence that was retrieved)")]
        ):
            values = [run.summary.get(metric) or 0.0 for _, run in rows]
            positions = [position + (offset - 0.5) * 0.36 for position in range(len(rows))]
            ax.barh(positions, values, height=0.32, color=colour, edgecolor=SURFACE, linewidth=1.6, label=label)
            for position, value in zip(positions, values):
                ax.text(value + 0.012, position, f"{value:.2f}", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(range(len(rows)), [name for name, _ in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("rate")
        ax.set_title(f"{condition_label(condition)} · budget {budget}", loc="left")
    return _finish(figure, axes, "Retrieval hit rate", out / "retrieval_hit_rate.png")


def _detail_runs(runs: list[Run]) -> tuple[str, list[Run]]:
    """The runs of one condition and budget that have step-level logs, for the over-time figures."""
    grouped = _grouped(runs)
    for condition, cells in sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0])):
        budget, cell = _focus_cell(cells)
        chosen = [run for _, run in sorted(cell.items()) if (run.folder / "steps.jsonl").exists()]
        if chosen:
            return f"{condition_label(condition)} · budget {budget}", chosen
    return "", []


def plot_occupancy(runs: list[Run], out: Path) -> Path | None:
    title, chosen = _detail_runs(runs)
    data = []
    for run in chosen:
        if run.controller == "full_context":
            continue
        steps = [row for row in read_jsonl(run.folder / "steps.jsonl") if row["episode_id"] == "ep00000"]
        if steps:
            data.append((run.controller, steps))
    if not data:
        return None
    data = data[:9]
    figure, axes = _panels(len(data), width=3.6, height=2.5)
    top = max(max(row["active_tokens_at_read"] for row in steps) for _, steps in data)
    for number, (ax, (name, steps)) in enumerate(zip(axes, data)):
        budget = steps[0]["memory"]["budget"]
        ax.plot([row["step"] for row in steps], [row["active_tokens_at_read"] for row in steps], color=SLOTS[0], linewidth=1.2)
        ax.fill_between([row["step"] for row in steps], [row["active_tokens_at_read"] for row in steps], color=SLOTS[0], alpha=0.10)
        if budget < 10**8:
            ax.axhline(budget, color=INK, linewidth=1)
            top = max(top, budget)
            if number == 0:
                ax.annotate("budget", (steps[-1]["step"], budget), xytext=(0, 3), textcoords="offset points",
                            fontsize=7.5, color=INK_2, ha="right")
        ax.set_title(name, loc="left")
        ax.set_xlabel("step")
        if number % 3 == 0:
            ax.set_ylabel("active tokens at read")
    for ax in axes:
        ax.set_ylim(0, top * 1.12)
    return _finish(figure, axes, f"Memory occupancy over the first episode · {title}", out / "memory_occupancy.png", legend=False)


def plot_actions_over_time(runs: list[Run], out: Path, bins: int = 10) -> Path | None:
    title, chosen = _detail_runs(runs)
    chosen = [run for run in chosen if (run.folder / "memory_actions.jsonl").exists()][:9]
    data = []
    for run in chosen:
        rows = read_jsonl(run.folder / "memory_actions.jsonl")
        if not rows:
            continue
        horizon = max(row["step"] for row in rows)
        episodes = len({row["episode_id"] for row in rows})
        counts: dict[str, list[float]] = defaultdict(lambda: [0.0] * bins)
        for row in rows:
            if row["status"] == "rejected":
                label = "rejected"
            elif row["source"] == "harness":
                label = "EVICT (forced)"
            else:
                label = row["operation"]
            if label in ("KEEP", "NO_OP"):
                continue
            counts[label][min(bins - 1, (row["step"] - 1) * bins // horizon)] += max(1, len(row["target_ids"])) / episodes
        if counts:
            data.append((run.controller, horizon, counts))
    if not data:
        return None
    figure, axes = _panels(len(data), width=4.2, height=2.9)
    for ax, (name, horizon, counts) in zip(axes, data):
        bottom = [0.0] * bins
        centres = [(n + 0.5) * horizon / bins for n in range(bins)]
        for label, colour in OPERATION_COLOURS.items():
            if label in counts:
                ax.bar(centres, counts[label], bottom=bottom, width=0.62 * horizon / bins, color=colour,
                       edgecolor=SURFACE, linewidth=1.2, label=label.replace("_", " ").lower())
                bottom = [a + b for a, b in zip(bottom, counts[label])]
        ax.set_title(name, loc="left")
        ax.set_xlabel("step")
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel("items acted on per episode")
        ax.grid(axis="x", visible=False)
    return _finish(figure, axes, f"Memory actions over time · {title}", out / "actions_over_time.png")


def plot_regret(runs: list[Run], out: Path) -> Path | None:
    title, _ = _detail_runs(runs)
    grouped = _grouped(runs)
    if not grouped:
        return None
    condition, cells = sorted(grouped.items(), key=lambda pair: condition_sort_key(pair[0]))[0]
    budget, cell = _focus_cell(cells)
    order = ["<10", "10-49", "50-199", "200+"]
    data = []
    for name, run in sorted(cell.items()):
        rows = run.rows("regret")
        if not rows:
            continue
        buckets = Counter()
        for row in rows:
            gap = row["steps_until_needed"]
            buckets["<10" if gap < 10 else "10-49" if gap < 50 else "50-199" if gap < 200 else "200+"] += 1
        data.append((name, [buckets[label] / len(run.episodes) for label in order]))
    if not data:
        return None
    data = data[:9]
    figure, axes = _panels(len(data), width=3.4, height=2.7)
    top = max(max(values) for _, values in data) * 1.18
    for ax, (name, values) in zip(axes, data):
        ax.bar(range(len(order)), values, width=0.5, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.6)
        for position, value in enumerate(values):
            ax.text(position, value + top * 0.015, f"{value:.1f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(order)), order)
        ax.set_ylim(0, top)
        ax.set_title(name, loc="left")
        ax.set_xlabel("steps between the decision and the need")
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel("requirements destroyed / episode")
        ax.grid(axis="x", visible=False)
    label = f"{condition_label(condition)} · budget {budget}"
    return _finish(figure, axes, f"Regret: needed evidence destroyed, by how far ahead it was needed · {label}",
                   out / "regret_distribution.png", legend=False)


def plot_training(folder: str | Path) -> Path | None:
    """The learning curve of one training folder (train_log.jsonl)."""
    folder = Path(folder)
    rows = read_jsonl(folder / "train_log.jsonl")
    if not rows:
        return None
    figure, axes = _panels(1, width=7.6, height=3.5)
    ax = axes[0]
    ax.plot([r["episodes_total"] for r in rows], [r["train_success"] for r in rows], color=MUTED, linewidth=1.3,
            label="training episodes (sampled actions)")
    keys = sorted({key for row in rows for key in row if key.startswith("success")}, key=lambda k: float(k.split("@")[1]) if "@" in k else 0)
    for slot, key in enumerate(keys[:8]):
        points = [(r["episodes_total"], r[key]) for r in rows if key in r]
        label = "evaluation (greedy)" if key == "success" else f"evaluation at budget {float(key.split('@')[1]) * 100:g}%"
        ax.plot([x for x, _ in points], [y for _, y in points], color=SLOTS[slot], marker=MARKERS[slot],
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=label)
    for row_before, row_after in zip(rows, rows[1:]):
        if row_before["phase"] != row_after["phase"]:
            ax.axvline(row_before["episodes_total"], color=AXIS, linewidth=1)
            ax.annotate(f"{row_after['algorithm']} starts", (row_before["episodes_total"], 0.02), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("task success")
    ax.set_title(folder.name, loc="left")
    return _finish(figure, axes, "Learning curve", folder / "plots" / "learning_curve.png")


def plot_all(root: str | Path) -> list[Path]:
    root = Path(root)
    runs = [run for run in load_runs(root) if run.summary.get("task_success") is not None]
    if not runs:
        return []
    out = root / "plots"
    made = [
        plot_success_vs_budget(runs, out),
        plot_success_vs_condition(runs, out, "env.horizon", "task horizon (steps)", "success_vs_horizon.png"),
        plot_success_vs_condition(runs, out, "env.gap.min", "minimum dependency distance (steps)", "success_vs_dependency_gap.png"),
        plot_success_vs_latency(runs, out),
        plot_oracle_gap(runs, out),
        plot_occupancy(runs, out),
        plot_actions_over_time(runs, out),
        plot_failure_attribution(runs, out),
        plot_regret(runs, out),
        plot_retrieval(runs, out),
    ]
    written = [path for path in made if path is not None]
    (out / "index.json").write_text(json.dumps([path.name for path in written], indent=2))
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Make the figures for a folder of runs, or a training folder.")
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    if (root / "train_log.jsonl").exists():
        print(plot_training(root))
    else:
        for path in plot_all(root):
            print(path)


if __name__ == "__main__":
    main()
