"""The charts used in reports. One function per chart, all in the same quiet style.

Colours come from a palette checked for colour-blind safety. Places and failure
groups always keep the same colour. Text is never coloured; values are written
next to the marks so no chart depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from memctl.attribution import LABELS, MEMORY_FAILURES, PRESENT_MODEL_WRONG  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PLACE_COLORS = {"CONTEXT": "#2a78d6", "STORE": "#eb6834", "ARCHIVE": "#1baf7a", "DROPPED": "#eda100"}
GROUP_COLORS = {"memory failure": "#2a78d6", "reasoning failure": "#eb6834", "unknown": "#8a8985"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def new_chart(title: str, width: float, height: float):
    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9, length=0)
    axes.set_axisbelow(True)
    return figure, axes


def save(figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def failure_group(label: str) -> str:
    if label in MEMORY_FAILURES:
        return "memory failure"
    return "reasoning failure" if label == PRESENT_MODEL_WRONG else "unknown"


def failure_attribution_chart(counts: dict[str, int], title: str, path: Path) -> None:
    """Horizontal bars: how many wrong answers got each failure label."""
    labels = list(reversed(LABELS))
    figure, axes = new_chart(title, 7.5, 2.9)
    values = [counts.get(label, 0) for label in labels]
    colors = [GROUP_COLORS[failure_group(label)] for label in labels]
    axes.barh(labels, values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    for position, value in enumerate(values):
        axes.text(value + max(values + [1]) * 0.015, position, str(value), va="center", fontsize=9, color=INK)
    axes.set_xlim(0, max(values + [1]) * 1.12)
    axes.set_xlabel("wrong answers", fontsize=9, color=MUTED)
    axes.tick_params(axis="y", labelcolor=INK)
    axes.xaxis.grid(True, color=GRID, linewidth=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in GROUP_COLORS.values()]
    axes.legend(handles, GROUP_COLORS, frameon=False, fontsize=9, loc="lower right", labelcolor=MUTED)
    save(figure, path)


def places_chart(places_by_controller: dict[str, dict[str, int]], title: str, path: Path) -> None:
    """Stacked horizontal bars: where items were at the end, one row per controller."""
    names = list(reversed(places_by_controller))
    figure, axes = new_chart(title, 7.5, 1.1 + 0.42 * len(names))
    left = [0.0] * len(names)
    for place, color in PLACE_COLORS.items():
        totals = [sum(places_by_controller[name].values()) or 1 for name in names]
        shares = [100 * places_by_controller[n].get(place, 0) / t for n, t in zip(names, totals)]
        axes.barh(names, shares, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=place)
        for row, (start, width) in enumerate(zip(left, shares)):
            if width >= 9:  # only label segments wide enough to hold the text
                text_color = "white" if place in ("CONTEXT", "STORE") else INK
                axes.text(start + width / 2, row, f"{width:.0f}%", ha="center", va="center", fontsize=8.5, color=text_color)
        left = [a + b for a, b in zip(left, shares)]
    axes.set_xlim(0, 100)
    axes.set_xlabel("share of items (%)", fontsize=9, color=MUTED)
    axes.tick_params(axis="y", labelcolor=INK)
    axes.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=MUTED)
    axes.set_title(title, loc="left", fontsize=11, color=INK, pad=30)
    save(figure, path)


def score_vs_cost_chart(series: dict[str, list[tuple[float, float, str]]], score_name: str, title: str, path: Path) -> None:
    """A score against real cost. One line per controller; each point is one run.

    Points are (average prompt tokens, score, budget label). The budget label is written
    next to each point.
    """
    figure, axes = new_chart(title, 7.5, 3.8)
    for color, (name, points) in zip(SERIES_COLORS, series.items()):
        points = sorted(points)
        axes.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=2, marker="o",
                  markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name, solid_capstyle="round")
        for tokens, score, tag in points:
            axes.annotate(tag, (tokens, score), xytext=(0, 8), textcoords="offset points",
                          ha="center", fontsize=7.5, color=MUTED)
    axes.set_ylim(0, 1)
    axes.set_xlim(left=0)
    axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes.set_xlabel("average prompt tokens per question (labels: budget B)", fontsize=9, color=MUTED)
    axes.set_ylabel(score_name, fontsize=9, color=MUTED)
    axes.yaxis.grid(True, color=GRID, linewidth=1)
    axes.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=MUTED)
    save(figure, path)
