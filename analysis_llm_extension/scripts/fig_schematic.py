#!/usr/bin/env python3
"""Conceptual illustration of declination-residual pitch SD: two synthetic
contours sharing the same sentence-level sweep, differing only in how much
local wobble sits on top of it. Purely illustrative (synthetic data), makes
the quantity legible to a non-specialist reader."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

WIN = "#2a78d6"; LOSE = "#eb6834"
INK = "#0b0b0b"; SEC = "#52514e"; GRID = "#d8d7d2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.6, "axes.edgecolor": SEC, "axes.linewidth": .6,
    "xtick.color": SEC, "ytick.color": SEC, "text.color": INK,
    "axes.labelcolor": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

rng = np.random.default_rng(7)
t = np.linspace(0, 1, 200)
sweep = 3.0 - 2.4 * t  # shared linear declination, semitone-like units

def wobble(scale, seed):
    r = np.random.default_rng(seed)
    # smooth-ish local noise: sum of a few low-frequency sinusoids + small jitter
    w = np.zeros_like(t)
    for k, amp in zip([3, 5, 8, 13], [0.6, 0.4, 0.25, 0.15]):
        w += amp * np.sin(2 * np.pi * k * t + r.uniform(0, 2 * np.pi))
    w += r.normal(0, 0.15, size=t.shape)
    w -= w.mean()
    return sweep + scale * w / np.std(w) * 1.0

low = wobble(0.35, 11)   # winning-candidate-like: less residual movement
high = wobble(0.95, 22)  # non-winning-candidate-like: more residual movement

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)

for ax, contour, sweep_line, color, title, resid in (
    (axes[0], low, sweep, WIN, "(a)  Less residual movement", low - sweep),
    (axes[1], high, sweep, LOSE, "(b)  More residual movement", high - sweep),
):
    ax.plot(t, sweep_line, color=SEC, lw=1.1, ls=(0, (4, 2)), zorder=2,
            label="Sentence-level sweep")
    ax.plot(t, contour, color=color, lw=1.6, zorder=3, label="Pitch contour")
    ax.fill_between(t, sweep_line, contour, color=color, alpha=0.15, zorder=1)
    ax.set_title(f"{title}\nresidual SD $\\approx$ {np.std(resid):.2f}",
                 fontsize=7.4, loc="left")
    ax.set_xlabel("Time within sentence")
    ax.set_xticks([]);
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Pitch (semitones, illustrative)")
axes[0].legend(frameon=False, fontsize=6.2, loc="upper right")

plt.tight_layout()
OUT = HERE.parent.parent / "report" / "figures" / "fig_schematic.pdf"
plt.savefig(OUT)
print(f"[write] {OUT}")
