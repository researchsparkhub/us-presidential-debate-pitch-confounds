#!/usr/bin/env python3
"""Figure for Method 4 (LLM cross-modal reasoning): debate-level accuracy by
condition against existing baselines, and the contamination probe result
that explains it. Same validated palette/style as the rest of the paper's
figures (figs_v2.py)."""
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

r = json.load(open("crossmodal_results.json"))
OUT = "/Users/lavanya/debate_analysis/report/figures/fig_crossmodal.pdf"


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})

# --- (a) debate-level accuracy: baselines + 3 LLM conditions ---
bars = [
    ("Coin\n(chance)", 0.479, GRID, False),
    ("Majority\nclass", 0.530, GRID, False),
    ("Forest,\nspeakers\nheld out", 0.500, SEC, False),
    ("Forest,\ndebate\nheld out", 0.708, SEC, False),
    ("LLM:\ndescriptors\nonly", r["conditions"]["descriptors"]["debate_majority_accuracy"], WIN, False),
    ("LLM:\ntranscript\nonly", r["conditions"]["transcript"]["debate_majority_accuracy"], LOSE, True),
    ("LLM:\nboth", r["conditions"]["both"]["debate_majority_accuracy"], LOSE, True),
]
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
colors = [b[2] for b in bars]
hatched = [b[3] for b in bars]
x = range(len(bars))
for xi, v, c, h in zip(x, vals, colors, hatched):
    a1.bar(xi, v, color=c, width=.62, zorder=3,
           hatch="///" if h else None, edgecolor="white" if h else "none", linewidth=.5)
a1.axhline(0.5, color=SEC, lw=.7, ls=(0, (3, 2)), zorder=2)
a1.set_xticks(list(x)); a1.set_xticklabels(labels, fontsize=6.0)
a1.set_ylabel("Debate-level accuracy (24 debates)")
a1.set_ylim(0, 1.05)
a1.grid(axis="y", color=GRID, lw=.5, zorder=0); a1.set_axisbelow(True)
despine(a1)
a1.set_title("(a)  Contaminated conditions look best", fontsize=7.4, loc="left", pad=5)
for xi, v in zip(x, vals):
    a1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.0, color=INK)

# --- (b) contamination probe: identification rate ---
probe = r["contamination_probe"]
conds = ["descriptors", "transcript"]
rates = [probe[c]["both_candidates_named"] / probe[c]["n"] for c in conds]
colors2 = [WIN, LOSE]
a2.barh(range(len(conds)), rates, color=colors2, height=.55, zorder=3)
a2.set_yticks(range(len(conds)))
a2.set_yticklabels(["Descriptors\nonly", "Transcript\n(anonymised)"])
a2.set_xlabel("Debates where the model\nnamed both real candidates")
a2.set_xlim(0, 1.05)
a2.grid(axis="x", color=GRID, lw=.5, zorder=0); a2.set_axisbelow(True)
despine(a2)
a2.set_title("(b)  Why: identifiability", fontsize=7.4, loc="left", pad=5)
for yi, v in zip(range(len(conds)), rates):
    a2.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=6.4, color=INK)

plt.tight_layout()
plt.savefig(OUT)
print(f"[write] {OUT}")
