"""Experiment 13 figure: accuracy against the reader's prompt tokens, LongMemEval and LoCoMo side by side.

One panel per benchmark (same y axis): FIFO with the retrieval floor at its budgets and targets (a line in
prompt-token order), the hindsight oracle, the learned controllers, and, on LongMemEval, the best keep-last
rule. Every series is direct-labelled and has its own marker shape, so identity never rests on colour alone.

    python -m memctl.analysis.frontier_plot .worktrees/exp13/runs --out docs/research/figures/exp13_frontier.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memctl.analysis.qa_tables import REFUSAL_SCORED, cell_rows  # noqa: E402

# Reference categorical palette, slots 1-4 (validated: CVD and normal-vision separation pass; contrast warns,
# so every series is direct-labelled).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"


def point(runs: Path, cells: list[str]) -> tuple[float, float]:
    """(mean prompt tokens, accuracy) pooled over the given cells, refusal-scored questions left out."""
    rows = [r for cell in cells for r in cell_rows(runs / cell) if r["category"] not in REFUSAL_SCORED]
    return sum(r["tokens"] for r in rows) / len(rows), sum(r["correct"] for r in rows) / len(rows)


def folds(template: str) -> list[str]:
    return [template.format(k=k) for k in range(5)]


def series(runs: Path) -> dict:
    lme_fifo = [folds("exp13_lme_f{k}/fifo_top5__fraction0.01"), folds("exp13_lme_f{k}/fifo_top5__fraction0.02"),
                folds("exp13_lme_f{k}_target/fifo_top5_t2000__fraction0.05"),
                folds("exp13_lme_f{k}_target/fifo_top5_t3000__fraction0.05"),
                folds("exp13_lme_f{k}_target/fifo_top5_t4000__fraction0.05"), folds("exp13_lme_f{k}/fifo_top5__fraction0.05")]
    lme_learned = [folds("exp13_lme_f{k}_extra/compose_floor5__fraction0.05"),
                   folds("exp13_lme_f{k}_extra/compose_floor5_t4000__fraction0.05"),
                   folds("exp13_lme_f{k}_extra/compose_floor5_t3000__fraction0.05")]
    lme_keep = [sorted(str(p.relative_to(runs)) for p in (runs / "exp13_lme_keep").glob("fold*__keep_last0_top5__*"))]
    locomo_fifo = [[f"exp13_locomo/fifo_top5_fusion__fraction{f}"] for f in ("0.05", "0.1", "0.25")]
    locomo_oracle = [[f"exp13_locomo/oracle-approx__fraction{f}"] for f in ("0.05", "0.1", "0.25")]
    locomo_learned = [["exp13_locomo_extra/compose_floor5_lme_f0__fraction0.25"]]
    to_points = lambda groups: sorted(point(runs, g) for g in groups)  # noqa: E731
    return {
        "LongMemEval-S (500 questions, 5 folds)": [
            ("FIFO + floor", BLUE, "o", "-", to_points(lme_fifo)),
            ("oracle", ORANGE, "D", "", to_points([folds("exp13_lme_f{k}/oracle-approx__fraction0.05")])),
            ("learned (composed)", AQUA, "s", "", to_points(lme_learned)),
            ("keep-last-0 + top 5", YELLOW, "^", "", to_points(lme_keep)),
        ],
        "LoCoMo (categories 1-4, 10 conversations)": [
            ("FIFO + floor", BLUE, "o", "-", to_points(locomo_fifo)),
            ("oracle", ORANGE, "D", "-", to_points(locomo_oracle)),
            ("learned (composed, transfer)", AQUA, "s", "", to_points(locomo_learned)),
        ],
    }


# Label placement (points): learned labels sit below-left of their last point, so they clear the panel edge
# and the FIFO label beside them.
OFFSETS = {"oracle": (6, -14, "left"), "learned (composed)": (-8, 8, "right"),
           "learned (composed, transfer)": (-8, 10, "right"), "FIFO + floor": (-8, -16, "right")}


def plot(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (title, rows) in zip(axes, data.items()):
        ax.set_facecolor(SURFACE)
        for label, colour, marker, line, points in rows:
            xs, ys = zip(*points)
            ax.plot(xs, ys, linestyle=line or "none", linewidth=2, color=colour, marker=marker, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
            dx, dy, align = OFFSETS.get(label, (6, 6, "left"))
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                        ha=align)
        right = max(x for *_, points in rows for x, _ in points)
        ax.set_xlim(0, right * 1.12)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("reader prompt tokens (mean)", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("judge accuracy", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = series(args.runs)
    for title, rows in data.items():
        print(title)
        for label, _, _, _, points in rows:
            print("  ", label, [(round(x), round(y, 3)) for x, y in points])
    plot(data, args.out)


if __name__ == "__main__":
    main()
