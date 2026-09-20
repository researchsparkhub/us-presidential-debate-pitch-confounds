#!/usr/bin/env python3
"""
Stage 6 — winner-vs-loser analysis across all features.

Three analyses, each writing its own output CSV:

  A) Univariate effect sizes (Cohen's d) + per-debate fixed-effects OLS
     for key acoustic + linguistic metrics.
     → outputs/analysis_effects.csv

  B) LDA (Linear Discriminant Analysis) on openSMILE eGeMAPSv02 features
     for winners vs losers, with 5-fold cross-validation.
     → outputs/analysis_lda_coefficients.csv  (top discriminating features)
     → outputs/analysis_lda_summary.txt        (accuracy + projection stats)

  C) Embedding-based semantic comparison: most "winner-like" and
     "loser-like" sentences via centroid distance.
     → outputs/analysis_top_sentences.csv

Run:
    python 06_analysis/winner_loser_analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


PROJECT = Path("/Users/lavanya/debate_analysis")
OUT_DIR = PROJECT / "outputs"
MASTER = OUT_DIR / "master_sentences.csv"
ACOUSTIC = OUT_DIR / "acoustic_features.csv"
SUPP = OUT_DIR / "acoustic_features_supplementary.csv"
LING = OUT_DIR / "linguistic_features.csv"
DAMSL = OUT_DIR / "damsl_bias_tags.csv"
EMB = OUT_DIR / "linguistic_embeddings.parquet"


# === Features to test universally ==========================================

KEY_METRICS = [
    # Acoustic prosody
    ("f0_mean",                  "Pitch (F0 mean Hz)"),
    ("f0_std",                   "Pitch variability (F0 std)"),
    ("intensity_mean_db",        "Loudness (intensity dB)"),
    ("hnr_mean_db",              "Voice clarity (HNR dB)"),
    ("jitter_local",             "Pitch perturbation (jitter)"),
    ("shimmer_local",            "Amplitude perturbation (shimmer)"),
    # Speech timing
    ("speech_rate_wps",          "Speech rate (words/sec)"),
    ("articulation_rate_wps",    "Articulation rate (words/sec)"),
    ("total_pause_sec",          "Total pause time (sec)"),
    ("n_pauses",                 "Pause count"),
    # Syllable rate (from supplementary)
    ("syllable_rate_per_sec",    "Syllable rate (syll/sec)"),
    ("articulation_rate_syll_per_sec", "Articulation rate (syll/sec)"),
    # Linguistic
    ("n_words",                  "Words per sentence"),
    ("type_token_ratio",         "Lexical diversity (TTR)"),
    ("flesch_reading_ease",      "Reading ease (Flesch)"),
    ("flesch_kincaid_grade",     "Grade level (FK)"),
    ("n_clauses",                "Clauses per sentence"),
    ("n_subordinate_clauses",    "Subordinate clauses"),
    ("parse_tree_depth",         "Syntactic depth"),
    ("mean_word_length",         "Mean word length (chars)"),
    # Sentiment
    ("vader_compound",           "Sentiment polarity (VADER)"),
    ("vader_pos",                "Positive sentiment"),
    ("vader_neg",                "Negative sentiment"),
    # Pronouns
    ("pct_pronouns_first",       "1st-person pronoun rate"),
    ("pct_pronouns_second",      "2nd-person pronoun rate"),
    ("pct_pronouns_third",       "3rd-person pronoun rate"),
    # POS
    ("pos_noun_rate",            "Noun rate"),
    ("pos_verb_rate",            "Verb rate"),
    ("pos_adj_rate",             "Adjective rate"),
    ("pos_adv_rate",             "Adverb rate"),
    # Discourse / Bias (key)
    ("damsl_IAFA",               "Imperative (command)"),
    ("damsl_EXPL",               "Explanation"),
    ("damsl_REJ",                "Rejection"),
    ("damsl_AGR",                "Agreement"),
    ("damsl_HS",                 "Historical reference"),
    ("bias_APAT",                "Patriotism appeal"),
    ("bias_AF",                  "Fear appeal"),
    ("bias_AE",                  "Emotional appeal"),
    ("bias_UF",                  "Unity framing"),
    ("bias_IP",                  "Personal stance (I believe...)"),
]


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Standard Cohen's d for two samples, pooled SD."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    return (mx - my) / pooled if pooled > 0 else np.nan


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list[bool] of which tests are significant under BH at alpha."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1)) / n
    sig_in_order = p[order] <= thresholds
    if not sig_in_order.any():
        return [False] * n
    # All hypotheses up to the largest k where p[k] <= alpha*k/n are significant
    max_k = np.where(sig_in_order)[0].max()
    sig_flags = np.zeros(n, dtype=bool)
    sig_flags[order[: max_k + 1]] = True
    return sig_flags.tolist()


def load_joined() -> pd.DataFrame:
    print("[load] master + acoustic + supplementary + linguistic + damsl_bias…")
    master = pd.read_csv(MASTER, low_memory=False)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)
    acoustic = pd.read_csv(ACOUSTIC)
    supp = pd.read_csv(SUPP)
    ling = pd.read_csv(LING)
    damsl = pd.read_csv(DAMSL)

    j = master.merge(acoustic, on="sentence_id", suffixes=("", "_a"))
    j = j.merge(
        supp[["sentence_id", "n_syllables", "syllable_rate_per_sec",
               "articulation_rate_syll_per_sec"]
              + [c for c in supp.columns if c.startswith("egemaps_")]],
        on="sentence_id", how="left",
    )
    j = j.merge(ling, on="sentence_id", suffixes=("", "_l"))
    j = j.merge(damsl, on="sentence_id", how="left")
    print(f"  joined: {len(j):,} sentences × {len(j.columns)} columns")
    return j


# === A) Univariate effects + per-debate fixed-effects ======================

def run_univariate(j: pd.DataFrame) -> pd.DataFrame:
    cand = j[j["speaker_role"] == "candidate"].copy()
    # Drop sentences with no usable result label (UNKNOWN from 2024)
    cand = cand[cand["result"].isin(["winner", "loser"])]
    print(f"  candidate sentences with winner/loser: {len(cand):,}")

    rows = []
    for feat, label in KEY_METRICS:
        if feat not in cand.columns:
            continue
        sub = cand[[feat, "result", "debate_audio_id", "speaker"]].dropna()
        winner = sub[sub["result"] == "winner"][feat]
        loser = sub[sub["result"] == "loser"][feat]
        if len(winner) < 5 or len(loser) < 5:
            continue

        d = cohens_d(winner, loser)

        # Mann-Whitney U (robust to non-normality)
        try:
            u, p_mw = stats.mannwhitneyu(winner, loser, alternative="two-sided")
        except Exception:
            p_mw = np.nan

        # Per-debate fixed effects OLS: feat ~ result + C(debate_audio_id)
        # Captures within-debate winner-vs-loser difference.
        try:
            sub2 = sub.copy()
            sub2["is_winner"] = (sub2["result"] == "winner").astype(int)
            model = smf.ols(f"Q('{feat}') ~ is_winner + C(debate_audio_id)", data=sub2).fit()
            coef = model.params.get("is_winner", np.nan)
            pval_fe = model.pvalues.get("is_winner", np.nan)
        except Exception as e:
            coef, pval_fe = np.nan, np.nan

        rows.append({
            "feature": feat,
            "label": label,
            "n_winner": len(winner),
            "n_loser": len(loser),
            "mean_winner": float(winner.mean()),
            "mean_loser": float(loser.mean()),
            "median_winner": float(winner.median()),
            "median_loser": float(loser.median()),
            "cohens_d": d,
            "mw_pvalue": float(p_mw) if not np.isnan(p_mw) else np.nan,
            "fixed_eff_coef": float(coef) if not np.isnan(coef) else np.nan,
            "fixed_eff_pvalue": float(pval_fe) if not np.isnan(pval_fe) else np.nan,
        })

    df = pd.DataFrame(rows)
    # BH correction across fixed-effects pvalues
    if "fixed_eff_pvalue" in df.columns:
        pvals = df["fixed_eff_pvalue"].fillna(1.0).tolist()
        df["fixed_eff_significant_bh"] = benjamini_hochberg(pvals, alpha=0.05)
    return df


# === B) LDA on openSMILE eGeMAPSv02 ========================================

def run_lda(j: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    print("\n[LDA] preparing eGeMAPSv02 feature matrix…")
    egemaps_cols = [c for c in j.columns if c.startswith("egemaps_")]
    print(f"  using {len(egemaps_cols)} openSMILE features")

    cand = j[(j["speaker_role"] == "candidate") & (j["result"].isin(["winner", "loser"]))]
    feat_df = cand[egemaps_cols].copy()
    # Drop rows with any NaN
    mask = feat_df.notna().all(axis=1)
    feat_df = feat_df[mask]
    y = (cand.loc[mask, "result"] == "winner").astype(int).values
    debate_ids = cand.loc[mask, "debate_audio_id"].values
    speakers = cand.loc[mask, "speaker"].values
    print(f"  N usable: {len(feat_df):,} sentences, "
          f"winners={int(y.sum()):,}, losers={int((1-y).sum()):,}")

    # Standardize
    X = StandardScaler().fit_transform(feat_df.values)

    # Single LDA model on all data → for coefficient inspection
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    train_acc = lda.score(X, y)
    coefs = lda.coef_[0]

    # Cross-validation (5-fold, stratified by class)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(
        LinearDiscriminantAnalysis(n_components=1), X, y, cv=cv, scoring="accuracy"
    )

    # GROUPED-BY-DEBATE CV to control for speaker/era confounds:
    # Each fold leaves out one debate's sentences entirely.
    from sklearn.model_selection import GroupKFold
    n_groups = len(np.unique(debate_ids))
    n_splits_g = min(5, n_groups)
    if n_splits_g >= 2:
        gkf = GroupKFold(n_splits=n_splits_g)
        cv_acc_grouped = cross_val_score(
            LinearDiscriminantAnalysis(n_components=1), X, y, cv=gkf, groups=debate_ids,
            scoring="accuracy",
        )
    else:
        cv_acc_grouped = np.array([np.nan])

    # Top coefficients by absolute weight
    coef_df = pd.DataFrame({
        "feature": [c.replace("egemaps_", "") for c in egemaps_cols],
        "lda_coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    summary_lines = [
        f"openSMILE LDA: winners vs losers",
        f"  N = {len(feat_df):,} sentences "
        f"({int(y.sum()):,} winners, {int((1-y).sum()):,} losers)",
        f"  Features: {len(egemaps_cols)} eGeMAPSv02 functionals",
        f"  Training accuracy:          {train_acc:.3f}",
        f"  5-fold CV accuracy (random): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}",
        f"  Leave-one-debate-out CV acc: {cv_acc_grouped.mean():.3f} ± {cv_acc_grouped.std():.3f}",
        f"    (chance = 0.500; debate-grouped CV is the unbiased estimate)",
        "",
        "Top 15 features by |LDA coefficient| (positive = louder in winners):",
    ]
    for _, r in coef_df.head(15).iterrows():
        summary_lines.append(f"  {r['lda_coefficient']:+.3f}   {r['feature']}")
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    return coef_df, summary


# === C) Embedding-based semantic comparison ================================

def run_embedding_analysis(j: pd.DataFrame) -> pd.DataFrame:
    print("\n[embeddings] computing winner/loser centroid distances…")
    emb = pd.read_parquet(EMB)
    # Join with master to get result label
    cand = j[(j["speaker_role"] == "candidate") & (j["result"].isin(["winner", "loser"]))][
        ["sentence_id", "speaker", "result", "sentence_text", "debate_audio_id", "year"]
    ]
    df = cand.merge(emb, on="sentence_id", how="inner")
    feat_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    X = df[feat_cols].values

    winner_mask = (df["result"] == "winner").values
    centroid_winner = X[winner_mask].mean(axis=0)
    centroid_loser = X[~winner_mask].mean(axis=0)
    direction = centroid_winner - centroid_loser
    direction /= max(1e-9, np.linalg.norm(direction))

    # Project every sentence onto winner-direction axis
    projections = X @ direction
    df["winner_direction_score"] = projections

    # Show top + bottom 10 most discriminative sentences
    top_winner = df.sort_values("winner_direction_score", ascending=False).head(10)
    top_loser = df.sort_values("winner_direction_score", ascending=True).head(10)

    print("\nTop-10 most 'winner-direction' sentences:")
    for _, r in top_winner.iterrows():
        text = str(r["sentence_text"])[:100]
        print(f"  {r['winner_direction_score']:+.3f} [{r['speaker']:18s} {r['result']:6s}] {text}")

    print("\nTop-10 most 'loser-direction' sentences:")
    for _, r in top_loser.iterrows():
        text = str(r["sentence_text"])[:100]
        print(f"  {r['winner_direction_score']:+.3f} [{r['speaker']:18s} {r['result']:6s}] {text}")

    out = pd.concat([top_winner.assign(direction="winner"),
                      top_loser.assign(direction="loser")])
    return out[["sentence_id", "direction", "winner_direction_score", "result",
                "speaker", "year", "debate_audio_id", "sentence_text"]]


# === Main ===================================================================

def main():
    j = load_joined()

    print("\n[A] Univariate effects + per-debate fixed-effects OLS")
    eff = run_univariate(j)
    eff_path = OUT_DIR / "analysis_effects.csv"
    eff.to_csv(eff_path, index=False)
    print(f"\n[write] {eff_path}")

    sig = eff[eff["fixed_eff_significant_bh"] == True].sort_values(
        "cohens_d", key=lambda s: s.abs(), ascending=False
    )
    print(f"\n=== Significant winner-vs-loser effects "
          f"(BH-corrected, with per-debate fixed effects), N={len(sig)} features ===")
    cols_show = ["feature", "label", "mean_winner", "mean_loser", "cohens_d",
                 "fixed_eff_coef", "fixed_eff_pvalue"]
    show = sig[cols_show].copy()
    for c in ["mean_winner", "mean_loser", "cohens_d", "fixed_eff_coef"]:
        show[c] = show[c].round(3)
    show["fixed_eff_pvalue"] = show["fixed_eff_pvalue"].apply(
        lambda p: f"{p:.2e}" if pd.notna(p) else ""
    )
    print(show.to_string(index=False))

    print("\n[B] LDA on openSMILE eGeMAPSv02")
    coefs, summary = run_lda(j)
    coefs.to_csv(OUT_DIR / "analysis_lda_coefficients.csv", index=False)
    (OUT_DIR / "analysis_lda_summary.txt").write_text(summary + "\n")
    print(f"[write] {OUT_DIR / 'analysis_lda_coefficients.csv'}")
    print(f"[write] {OUT_DIR / 'analysis_lda_summary.txt'}")

    print("\n[C] Embedding-based semantic comparison")
    top_sentences = run_embedding_analysis(j)
    top_path = OUT_DIR / "analysis_top_sentences.csv"
    top_sentences.to_csv(top_path, index=False)
    print(f"\n[write] {top_path}")


if __name__ == "__main__":
    main()
