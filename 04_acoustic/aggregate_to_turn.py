#!/usr/bin/env python3
"""
Stage 4c — aggregate per-sentence acoustic features to per-turn metrics.

Why turns? A turn = one continuous speaking opportunity. Some metrics make
more sense at this level:
  - Pauses per turn  : how a candidate paces an extended argument
  - Floor time       : how long they held the floor
  - Speech density   : speech time / floor time
  - Aggregated F0/intensity over the whole turn (more stable than per-sentence)

Inputs:
    outputs/master_sentences.csv               (turn_id + speaker + result)
    outputs/acoustic_features.csv              (pause counts, speech_duration, F0, etc.)
    outputs/acoustic_features_supplementary.csv (syllables, opensmile)

Output:
    outputs/acoustic_features_turn_level.csv

Run:
    python 04_acoustic/aggregate_to_turn.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
MASTER_CSV = PROJECT / "outputs" / "master_sentences.csv"
ACOUSTIC_CSV = PROJECT / "outputs" / "acoustic_features.csv"
SUPP_CSV = PROJECT / "outputs" / "acoustic_features_supplementary.csv"
OUT_CSV = PROJECT / "outputs" / "acoustic_features_turn_level.csv"


def main():
    print(f"[load] {MASTER_CSV}")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)

    print(f"[load] {ACOUSTIC_CSV}")
    acoustic = pd.read_csv(ACOUSTIC_CSV)
    print(f"[load] {SUPP_CSV}")
    supp = pd.read_csv(SUPP_CSV) if SUPP_CSV.exists() else pd.DataFrame()

    # Build the joined per-sentence frame
    j = master.merge(acoustic, on="sentence_id", suffixes=("", "_a"))
    if not supp.empty:
        j = j.merge(supp[["sentence_id", "n_syllables", "syllable_rate_per_sec",
                           "articulation_rate_syll_per_sec"]],
                    on="sentence_id", how="left")
    else:
        for c in ["n_syllables", "syllable_rate_per_sec", "articulation_rate_syll_per_sec"]:
            j[c] = np.nan

    # Filter to rows with a usable turn_id (the 2024 Whisper-canonical rows have empty turn_id)
    has_turn = j["turn_id"].astype(str).str.len() > 0
    j_turns = j[has_turn].copy()
    print(f"  sentences with usable turn_id: {len(j_turns):,} of {len(j):,}")

    # Aggregate per (debate, turn)
    print("[aggregate] computing per-turn metrics…")
    grp = j_turns.groupby(["debate_audio_id", "turn_id"], dropna=False)
    aggregated = grp.agg(
        # Identity / metadata
        speaker=("speaker", "first"),
        party=("party", "first"),
        result=("result", "first"),
        speaker_role=("speaker_role", "first"),
        year=("year", "first"),
        debate_type=("debate_type", "first"),
        date=("date", "first"),
        # Counts
        n_sentences=("sentence_id", "count"),
        n_words=("n_words", "sum"),
        n_syllables=("n_syllables", "sum"),
        # Timing
        turn_start_sec=("start_sec", "min"),
        turn_end_sec=("end_sec", "max"),
        speech_duration_sec=("speech_duration_sec", "sum"),
        total_pause_sec=("total_pause_sec", "sum"),
        n_pauses_within_sentences=("n_pauses", "sum"),
        # F0 (weighted by sentence durations would be ideal but mean is fine for turn summary)
        f0_mean=("f0_mean", "mean"),
        f0_std=("f0_mean", "std"),       # variability across the turn's sentences
        # Intensity
        intensity_mean_db=("intensity_mean_db", "mean"),
        intensity_std_db=("intensity_mean_db", "std"),
        # Voice quality
        jitter_local_mean=("jitter_local", "mean"),
        shimmer_local_mean=("shimmer_local", "mean"),
        hnr_mean_db=("hnr_mean_db", "mean"),
        # Speech rate aggregates
        speech_rate_wps_mean=("speech_rate_wps", "mean"),
        articulation_rate_wps_mean=("articulation_rate_wps", "mean"),
        syllable_rate_per_sec_mean=("syllable_rate_per_sec", "mean"),
        articulation_rate_syll_per_sec_mean=("articulation_rate_syll_per_sec", "mean"),
    ).reset_index()

    # Derived metrics. Use turn_floor_time (end - start) as the denominator
    # for speaking-tempo metrics, NOT summed per-sentence speech_duration_sec
    # (per-sentence pause detection over-subtracts, inflating apparent rate).
    aggregated["turn_floor_time_sec"] = aggregated["turn_end_sec"] - aggregated["turn_start_sec"]
    aggregated["turn_speech_density"] = (
        aggregated["speech_duration_sec"] / aggregated["turn_floor_time_sec"].clip(lower=0.01)
    )
    aggregated["pauses_per_100_words"] = (
        aggregated["n_pauses_within_sentences"] / aggregated["n_words"].clip(lower=1) * 100
    )
    aggregated["mean_pause_sec_per_turn"] = (
        aggregated["total_pause_sec"] / aggregated["n_pauses_within_sentences"].clip(lower=1)
    )
    # Speaking rate over the WHOLE turn (floor time, includes pauses).
    aggregated["turn_speech_rate_wps"] = (
        aggregated["n_words"] / aggregated["turn_floor_time_sec"].clip(lower=0.05)
    )
    aggregated["turn_syllable_rate_per_sec"] = (
        aggregated["n_syllables"] / aggregated["turn_floor_time_sec"].clip(lower=0.05)
    )
    # Articulation rate excludes pause time → uses speech_duration as denominator
    # (sum of within-sentence speech times; gives speaking tempo when actively speaking).
    aggregated["turn_articulation_rate_wps"] = (
        aggregated["n_words"] / aggregated["speech_duration_sec"].clip(lower=0.05)
    )
    aggregated["turn_articulation_rate_syll_per_sec"] = (
        aggregated["n_syllables"] / aggregated["speech_duration_sec"].clip(lower=0.05)
    )

    # Round numeric columns for readability
    for c in aggregated.select_dtypes(include="float").columns:
        aggregated[c] = aggregated[c].round(3)

    aggregated.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  ({len(aggregated):,} turns × {len(aggregated.columns)} columns)")

    print("\n=== Turn count by speaker_role ===")
    print(aggregated.groupby(["speaker_role", "result"]).size().rename("n_turns").to_string())

    cand = aggregated[aggregated["speaker_role"] == "candidate"].copy()
    print(f"\n=== Per-turn metrics by winner vs loser (candidates only) ===")
    print(f"All turns: {len(cand):,}.  Filtering to 'substantial' (≥10 words AND ≥5s floor time)…")
    substantial = cand[(cand["n_words"] >= 10) & (cand["turn_floor_time_sec"] >= 5)]
    print(f"Substantial turns: {len(substantial):,}")

    print("\n--- Mean of substantial-turn metrics ---")
    by_result = substantial.groupby("result").agg(
        n_turns=("turn_id", "count"),
        mean_words_per_turn=("n_words", "mean"),
        mean_floor_time_sec=("turn_floor_time_sec", "mean"),
        mean_speech_density=("turn_speech_density", "mean"),
        mean_pauses_per_100w=("pauses_per_100_words", "mean"),
        mean_pause_sec=("mean_pause_sec_per_turn", "mean"),
        speech_rate_wps=("turn_speech_rate_wps", "mean"),
        articulation_rate_wps=("turn_articulation_rate_wps", "mean"),
        syllable_rate=("turn_syllable_rate_per_sec", "mean"),
        articulation_rate_syll=("turn_articulation_rate_syll_per_sec", "mean"),
        f0_mean=("f0_mean", "mean"),
    ).round(2)
    print(by_result.to_string())

    print("\n--- Median of substantial-turn metrics (robust to outliers) ---")
    by_result_med = substantial.groupby("result").agg(
        n_turns=("turn_id", "count"),
        med_words=("n_words", "median"),
        med_floor_sec=("turn_floor_time_sec", "median"),
        med_speech_density=("turn_speech_density", "median"),
        med_pauses_per_100w=("pauses_per_100_words", "median"),
        med_speech_rate_wps=("turn_speech_rate_wps", "median"),
        med_articulation_rate_wps=("turn_articulation_rate_wps", "median"),
        med_syllable_rate=("turn_syllable_rate_per_sec", "median"),
        med_f0=("f0_mean", "median"),
    ).round(2)
    print(by_result_med.to_string())


if __name__ == "__main__":
    main()
