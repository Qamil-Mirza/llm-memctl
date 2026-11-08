"""Draw Figure 2 from tradeoff_data.json (written by tradeoff_data.py). Plotting only; no data are computed here.

Run:  uv run --no-project --with matplotlib python reports/exp51-technical-report/figures/tradeoff_plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "tradeoff_data.json").read_text())

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EDIT, COMP = "#2a78d6", "#52514e"

LABELS = {
    "B0": "B0 unedited",
    "candidate": "Selected edit\n(rank-1, L35)",
    "B1": "B1 logit bias",
    "B2": "B2 instruction",
    "B3": "B3 option\nremoved",
    "B4": "B4 L29 steering",
    "B5": "B5 orthogonalisation",
}
# label offsets in points (dx, dy, ha)
OFFS = {"candidate": (-9, 9, "right"), "B1": (7, 9, "left"), "B2": (8, -6, "left"), "B4": (-8, -11, "right"),
        "B5": (8, -12, "left"), "B3": (0, -24, "center")}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})

fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.3, 3.5), gridspec_kw={"width_ratios": [4.2, 1]})
fig.subplots_adjust(wspace=0.06, left=0.1, right=0.98, top=0.95, bottom=0.14)

for a in (ax, ax2):
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    b0 = D["B0"]["Uns"]["est"]
    a.axvline(b0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for k, v in D.items():
        if k == "B0":
            continue
        x, y = v["Uns"], v["Rel"]
        col = EDIT if k == "candidate" else COMP
        hollow = k == "B0"
        a.errorbar(x["est"], y["est"], xerr=[[x["est"] - x["lo"]], [x["hi"] - x["est"]]],
                   yerr=[[y["est"] - y["lo"]], [y["hi"] - y["est"]]], fmt="none", ecolor=col, elinewidth=1.0,
                   capsize=0, alpha=0.75, zorder=2)
        a.plot(x["est"], y["est"], "o" if k != "candidate" else "D", ms=7 if k == "candidate" else 6,
               mfc="white" if hollow else col, mec=col, mew=1.4, zorder=3)
        dx, dy, ha = OFFS[k]
        a.annotate(LABELS[k], (x["est"], y["est"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                   va="center", fontsize=8, color=INK, fontweight="bold" if k == "candidate" else "normal",
                   clip_on=True, annotation_clip=True)

ax.set_xlim(0.0, 0.125)
ax2.set_xlim(0.62, 0.745)
ax.set_ylim(0.58, 1.03)
ax.annotate("B0 unedited:\nUns 0.035, Rel 0", (D["B0"]["Uns"]["est"], 1.02), xytext=(4, -2), textcoords="offset points",
            ha="left", va="top", fontsize=7.5, color=MUTED)
ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1])
ax2.set_xticks([0.65, 0.70])
ax.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(axis="y", left=False)
for a in (ax, ax2):
    a.spines["top"].set_visible(False)
# axis-break marks
kw = dict(color=MUTED, clip_on=False, lw=0.8)
for a, xs in ((ax, (1, 1)), (ax2, (0, 0))):
    for yb in (0,):
        a.plot([xs[0] - 0.012 * (1 if a is ax else 4.2), xs[0] + 0.012 * (1 if a is ax else 4.2)],
               [yb - 0.015, yb + 0.015], transform=a.transAxes, **kw)

ax.set_ylabel("Recovery of baseline false abstentions (Rel)")
fig.supxlabel("Unsupported answers on unanswerable questions (Uns)", fontsize=9, color=INK, y=0.02)

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig2-tradeoff.{ext}", dpi=200)
print("wrote fig2-tradeoff.pdf/png")
