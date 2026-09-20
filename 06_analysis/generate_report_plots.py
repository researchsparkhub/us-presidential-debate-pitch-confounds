#!/usr/bin/env python3
"""
Generate publication-quality figures for the IEEE report.

Outputs PDF (vector) + PNG (raster) versions of:
  fig1_pipeline.{pdf,png}      Pipeline overview diagram
  fig2_lda.{pdf,png}           LDA projection of openSMILE features
  fig3_effects.{pdf,png}       Effect sizes for top discriminating features
  fig4_pauses.{pdf,png}        Pause distribution by winner / loser
  fig5_per_debate.{pdf,png}    Per-debate trends across 1988-2024
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})
sns.set_palette("colorblind")

WINNER_COLOR = "#2E7D32"   # green
LOSER_COLOR = "#C62828"    # red

PROJECT = Path("/Users/lavanya/debate_analysis")
OUT = PROJECT / "outputs"
FIG_DIR = PROJECT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.{{pdf,png}}")


# === Fig 1: Pipeline overview ==============================================

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, color="#E3F2FD"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor="black", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=10,
                            color="black", linewidth=0.8)
        ax.add_patch(a)

    # Row 1: data sources
    box(0.2, 3.6, 2.0, 0.8, "26 debate audio\n(WAV, 1988–2024)", "#F3E5F5")
    box(0.2, 2.4, 2.0, 0.8, "Official transcripts\n(text)", "#F3E5F5")
    box(0.2, 1.2, 2.0, 0.8, "Election metadata\n(winner/loser)", "#F3E5F5")

    # Row 2: alignment stage
    box(3.0, 3.0, 2.2, 1.4,
        "Stage 2: Alignment\n• MFA (20 debates)\n• Text-via-Whisper (4)\n• Whisper-canonical (2)",
        "#E3F2FD")
    arrow(2.2, 4.0, 3.0, 3.9)
    arrow(2.2, 2.8, 3.0, 3.5)

    # Row 3: feature extraction
    box(6.0, 3.6, 2.2, 0.8,
        "Stage 4: Acoustic\nparselmouth + openSMILE",
        "#FFF3E0")
    box(6.0, 2.4, 2.2, 0.8,
        "Stage 5: Linguistic\nspaCy + VADER + Empath",
        "#FFF3E0")
    box(6.0, 1.2, 2.2, 0.8,
        "Stage 5b: Tagging\nDAMSL + Bias",
        "#FFF3E0")
    arrow(5.2, 3.9, 6.0, 4.0)
    arrow(5.2, 3.5, 6.0, 2.8)
    arrow(5.2, 3.2, 6.0, 1.6)
    arrow(2.2, 1.6, 6.0, 1.5)

    # Row 4: master tables
    box(9.0, 2.5, 1.7, 1.5,
        "Master table\n35,459 sentences\n× 270 columns",
        "#E8F5E9")
    arrow(8.2, 4.0, 9.0, 3.5)
    arrow(8.2, 2.8, 9.0, 3.0)
    arrow(8.2, 1.6, 9.0, 2.7)

    # Stage 6
    box(9.0, 0.6, 1.7, 1.4,
        "Stage 6: Analysis\nLDA · effects ·\nembeddings",
        "#FCE4EC")
    arrow(9.85, 2.5, 9.85, 2.0)

    ax.set_title("Pipeline Overview", pad=6)
    save(fig, "fig1_pipeline")


# === Fig 2: LDA projection on openSMILE ====================================

def fig_lda():
    print("[fig2] preparing eGeMAPS feature matrix…")
    supp = pd.read_csv(OUT / "acoustic_features_supplementary.csv")
    master = pd.read_csv(OUT / "master_sentences.csv", low_memory=False)
    j = master[["sentence_id", "speaker_role", "result", "debate_audio_id"]].merge(
        supp, on="sentence_id"
    )
    cand = j[(j["speaker_role"] == "candidate") & (j["result"].isin(["winner", "loser"]))]
    egemaps_cols = [c for c in supp.columns if c.startswith("egemaps_")]
    X = cand[egemaps_cols].copy()
    mask = X.notna().all(axis=1)
    X = X[mask].values
    y = (cand.loc[mask, "result"] == "winner").astype(int).values

    Xs = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1)
    ld1 = lda.fit_transform(Xs, y).flatten()
    # Use PCA1 of the residual space for a 2D scatter
    pca = PCA(n_components=2).fit_transform(Xs)
    pca1 = pca[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel A: 1D KDE / histogram of LD1 by class
    ax = axes[0]
    bins = 60
    ax.hist(ld1[y == 1], bins=bins, alpha=0.55, color=WINNER_COLOR, label="Winners",
            density=True, edgecolor="none")
    ax.hist(ld1[y == 0], bins=bins, alpha=0.55, color=LOSER_COLOR, label="Losers",
            density=True, edgecolor="none")
    ax.axvline(0, color="black", linestyle=":", linewidth=0.6)
    ax.set_xlabel("LD1 (winner direction →)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Distribution along LD1")
    ax.legend(frameon=False, loc="upper right")

    # Panel B: 2D scatter (LD1 vs PCA1)  — subsample for clarity
    ax = axes[1]
    n = len(ld1)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(4000, n), replace=False)
    ax.scatter(ld1[idx][y[idx] == 0], pca1[idx][y[idx] == 0],
               s=2.5, c=LOSER_COLOR, alpha=0.30, label="Losers")
    ax.scatter(ld1[idx][y[idx] == 1], pca1[idx][y[idx] == 1],
               s=2.5, c=WINNER_COLOR, alpha=0.30, label="Winners")
    ax.set_xlabel("LD1 (winner direction →)")
    ax.set_ylabel("PC1 (acoustic variation)")
    ax.set_title("(b) 2D acoustic space")
    ax.legend(frameon=False, loc="upper right", markerscale=4)
    plt.tight_layout()
    save(fig, "fig2_lda")


# === Fig 3: Effect sizes bar chart =========================================

def fig_effects():
    eff = pd.read_csv(OUT / "analysis_effects.csv")
    sig = eff[eff["fixed_eff_significant_bh"] == True].copy()
    sig["abs_d"] = sig["cohens_d"].abs()
    sig = sig.sort_values("abs_d", ascending=True).tail(18)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = [WINNER_COLOR if d > 0 else LOSER_COLOR for d in sig["cohens_d"]]
    ax.barh(sig["label"], sig["cohens_d"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Cohen's d  (positive = higher in winners; negative = higher in losers)")
    ax.set_title("Significant winner-vs-loser differences\n(BH-corrected; per-debate fixed effects)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    # Legend
    win_patch = mpatches.Patch(color=WINNER_COLOR, label="Higher in winners")
    los_patch = mpatches.Patch(color=LOSER_COLOR, label="Higher in losers")
    ax.legend(handles=[win_patch, los_patch], loc="lower right", frameon=False)
    plt.tight_layout()
    save(fig, "fig3_effects")


# === Fig 4: Pause distribution ============================================

def fig_pauses():
    master = pd.read_csv(OUT / "master_sentences.csv", low_memory=False)
    ac = pd.read_csv(OUT / "acoustic_features.csv")
    j = master[["sentence_id", "speaker_role", "result"]].merge(ac, on="sentence_id")
    cand = j[(j["speaker_role"] == "candidate") & (j["result"].isin(["winner", "loser"]))]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel A: pause time per sentence (winsorized)
    pdata = cand[["result", "total_pause_sec"]].dropna()
    pdata = pdata[pdata["total_pause_sec"] < pdata["total_pause_sec"].quantile(0.99)]
    sns.boxplot(data=pdata, x="result", y="total_pause_sec", order=["loser", "winner"],
                palette=[LOSER_COLOR, WINNER_COLOR], showfliers=False, ax=axes[0], width=0.5)
    axes[0].set_title("(a) Pause time per sentence")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total pause (s)")
    axes[0].set_xticklabels(["Loser", "Winner"])

    # Panel B: pauses per sentence count
    n_data = cand[["result", "n_pauses"]].dropna()
    sns.boxplot(data=n_data, x="result", y="n_pauses", order=["loser", "winner"],
                palette=[LOSER_COLOR, WINNER_COLOR], showfliers=False, ax=axes[1], width=0.5)
    axes[1].set_title("(b) Number of pauses per sentence")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Pause count")
    axes[1].set_xticklabels(["Loser", "Winner"])

    plt.tight_layout()
    save(fig, "fig4_pauses")


# === Fig 5: Per-debate trends ==============================================

def fig_per_debate():
    master = pd.read_csv(OUT / "master_sentences.csv", low_memory=False)
    ac = pd.read_csv(OUT / "acoustic_features.csv")
    ling = pd.read_csv(OUT / "linguistic_features.csv")
    j = master[["sentence_id", "year", "debate_audio_id", "speaker", "speaker_role", "result"]] \
        .merge(ac, on="sentence_id").merge(ling, on="sentence_id")
    cand = j[(j["speaker_role"] == "candidate") & (j["result"].isin(["winner", "loser"]))]

    by = cand.groupby(["year", "result"]).agg(
        hnr=("hnr_mean_db", "mean"),
        pause=("total_pause_sec", "mean"),
        n_words=("n_words", "mean"),
        vader=("vader_compound", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), sharex=True)
    metrics = [
        ("hnr",     "Voice clarity (HNR dB)", axes[0][0]),
        ("pause",   "Pause time per sentence (s)", axes[0][1]),
        ("n_words", "Words per sentence", axes[1][0]),
        ("vader",   "Sentiment (VADER compound)", axes[1][1]),
    ]
    for col, ylab, ax in metrics:
        for res, color in [("winner", WINNER_COLOR), ("loser", LOSER_COLOR)]:
            sub = by[by["result"] == res].sort_values("year")
            ax.plot(sub["year"], sub[col], marker="o", color=color,
                    label=res.capitalize(), linewidth=1.5, markersize=4)
        ax.set_ylabel(ylab)
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    axes[0][0].legend(frameon=False, loc="lower right")
    axes[1][0].set_xlabel("Election year")
    axes[1][1].set_xlabel("Election year")
    plt.tight_layout()
    save(fig, "fig5_per_debate")


def main():
    print("[run] generating figures…")
    fig_pipeline()
    fig_lda()
    fig_effects()
    fig_pauses()
    fig_per_debate()
    print(f"\n[done] figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
