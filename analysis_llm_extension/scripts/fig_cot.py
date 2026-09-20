#!/usr/bin/env python3
"""Figure for the CoT-vs-Direct discourse-tagging comparison (extension #2)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN = "#2a78d6"; LOSE = "#eb6834"
INK = "#0b0b0b"; SEC = "#52514e"; GRID = "#d8d7d2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.4, "axes.edgecolor": SEC, "axes.linewidth": .6,
    "xtick.color": SEC, "ytick.color": SEC, "text.color": INK,
    "axes.labelcolor": INK, "xtick.major.width": .6, "ytick.major.width": .6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

r = json.load(open("cot_results.json"))
OUT = "/Users/lavanya/debate_analysis/report/figures/fig_cot.pdf"


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


fig, ax = plt.subplots(figsize=(3.4, 2.5))
pairs = ["direct_vs_rule", "cot_vs_rule", "cot_vs_direct"]
plabels = ["Direct vs.\nrule", "CoT vs.\nrule", "CoT vs.\ndirect"]
damsl_vals = [r["damsl"][p]["alpha_masi"] for p in pairs]
beads_vals = [r["beads"][p]["alpha_masi"] for p in pairs]

x = range(len(pairs))
w = 0.32
ax.bar([xi - w / 2 for xi in x], damsl_vals, width=w, color=WIN, zorder=3, label="Dialogue acts (DAMSL)")
ax.bar([xi + w / 2 for xi in x], beads_vals, width=w, color=LOSE, zorder=3, label="Rhetorical categories (Beads)")
ax.axhline(0, color=SEC, lw=.7, zorder=2)
ax.set_xticks(list(x)); ax.set_xticklabels(plabels, fontsize=6.6)
ax.set_ylabel("Krippendorff's $\\alpha$ (MASI)")
ax.grid(axis="y", color=GRID, lw=.5, zorder=0); ax.set_axisbelow(True)
despine(ax)
ax.legend(frameon=False, fontsize=6.0, loc="upper left")
for xi, v in zip(x, damsl_vals):
    ax.text(xi - w / 2, v + (0.02 if v >= 0 else -0.045), f"{v:+.2f}", ha="center", fontsize=5.6)
for xi, v in zip(x, beads_vals):
    ax.text(xi + w / 2, v + (0.02 if v >= 0 else -0.045), f"{v:+.2f}", ha="center", fontsize=5.6)

plt.tight_layout()
plt.savefig(OUT)
print(f"[write] {OUT}")
