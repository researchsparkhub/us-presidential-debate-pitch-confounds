#!/usr/bin/env python3
"""Corpus construction pipeline, boxes+arrows, same palette/style as the rest
of the paper's figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

WIN = "#2a78d6"
INK = "#0b0b0b"; SEC = "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.6, "text.color": INK,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

STEPS = [
    "Broadcast audio\n+ official transcript",
    "Transcript normalisation\n(two versions: analysis / matching)",
    "Speaker-label resolution\n(candidate, moderator, audience, unattributed)",
    "ASR word timings\n(Whisper)",
    "Transcript\N{EN DASH}ASR matching\n(match rate decides routing)",
    "Sentence-level timing\n(direct or interpolated)",
    "Aligned multimodal corpus\n24 debates, 29,672 sentences",
]

n = len(STEPS)
box_h = 0.62
centers = [n - i - 0.5 for i in range(n)]  # top to bottom, center y per box

fig, ax = plt.subplots(figsize=(3.3, 7.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, n)
ax.axis("off")

for i, (label, cy) in enumerate(zip(STEPS, centers)):
    is_last = i == n - 1
    box = FancyBboxPatch((0.05, cy - box_h / 2), 0.90, box_h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.0, edgecolor=SEC,
                          facecolor=WIN if is_last else "white", zorder=3)
    ax.add_patch(box)
    ax.text(0.5, cy, label, ha="center", va="center", fontsize=7.3,
            color="white" if is_last else INK, zorder=4, linespacing=1.35)

for i in range(n - 1):
    y0 = centers[i] - box_h / 2
    y1 = centers[i + 1] + box_h / 2
    ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                arrowprops=dict(arrowstyle="-|>", color=SEC, lw=1.1))

plt.tight_layout()
OUT = "/Users/lavanya/debate_analysis/report/figures/fig_pipeline.pdf"
plt.savefig(OUT)
print(f"[write] {OUT}")
