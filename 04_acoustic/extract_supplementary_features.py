#!/usr/bin/env python3
"""
Stage 4b — supplementary acoustic features (syllables + openSMILE eGeMAPSv02).

For each sentence in outputs/master_sentences.csv, compute:

  Syllable-based rate metrics (faster than the existing word-rate metrics):
    n_syllables                — sum of CMU dict syllable counts across words
    syllable_rate_per_sec      — syllables / sentence duration (speaking tempo)
    articulation_rate_syll_per_sec — syllables / speech-only duration
                                     (excludes pauses; joins to acoustic_features.csv)

  openSMILE eGeMAPSv02 functionals — 88 features per sentence covering:
    F0 (semitone-scaled), loudness, jitter/shimmer/HNR, formant frequencies,
    spectral slope, MFCC means, voiced/unvoiced ratios. The de-facto standard
    feature set for affect / paralinguistic research.

Output:
    outputs/acoustic_features_supplementary.csv
      ~92 numeric columns + sentence_id + status

Join with acoustic_features.csv on sentence_id for downstream analysis.

Run:
    python 04_acoustic/extract_supplementary_features.py
    python 04_acoustic/extract_supplementary_features.py --limit 300
    python 04_acoustic/extract_supplementary_features.py --debates 1003 1010
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd
import syllapy

warnings.filterwarnings("ignore")


PROJECT = Path(__file__).resolve().parent.parent
MASTER_CSV = PROJECT / "outputs" / "master_sentences.csv"
ACOUSTIC_CSV = PROJECT / "outputs" / "acoustic_features.csv"   # for speech_duration
AUDIO_DIR = PROJECT / "data" / "audio"
OUT_CSV = PROJECT / "outputs" / "acoustic_features_supplementary.csv"

TARGET_SR = 16000
MIN_DURATION_SEC = 0.3


# === syllable counting ======================================================

def count_syllables(text: str) -> int:
    """Sum CMU-dictionary syllable counts across all words. Falls back to a
    simple heuristic for OOV words."""
    n = 0
    for w in text.split():
        # Strip punctuation that confuses syllapy
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if not clean:
            continue
        try:
            n += syllapy.count(clean)
        except Exception:
            # Heuristic fallback: count vowel groups, min 1
            vowels = "aeiouy"
            groups = sum(1 for i, c in enumerate(clean.lower())
                         if c in vowels and (i == 0 or clean[i - 1].lower() not in vowels))
            n += max(1, groups)
    return n


# === openSMILE wrapper ======================================================

SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
OPENSMILE_FEATURE_NAMES = list(SMILE.feature_names)


def opensmile_features(clip: np.ndarray, sr: int) -> dict:
    """Run eGeMAPSv02 functionals on a sentence clip. Returns dict of 88 features."""
    try:
        # opensmile expects float32 mono signal
        df = SMILE.process_signal(clip.astype(np.float32), sampling_rate=sr)
        # Returned as 1-row DataFrame indexed by (file, start, end)
        if df.empty:
            return {f"egemaps_{name}": np.nan for name in OPENSMILE_FEATURE_NAMES}
        row = df.iloc[0]
        return {f"egemaps_{name}": float(v) if pd.notna(v) else np.nan
                for name, v in row.items()}
    except Exception:
        return {f"egemaps_{name}": np.nan for name in OPENSMILE_FEATURE_NAMES}


# === audio I/O =============================================================

def load_debate_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32), TARGET_SR


# === per-sentence processing ===============================================

def process_sentence(audio: np.ndarray, sr: int, row: pd.Series,
                     speech_sec_lookup: dict[str, float]) -> dict:
    text = str(row["sentence_text"])
    n_syll = count_syllables(text)
    duration = float(row["duration_sec"])
    sentence_id = row["sentence_id"]
    speech_sec = speech_sec_lookup.get(sentence_id, np.nan)

    out: dict = {
        "sentence_id": sentence_id,
        "debate_audio_id": str(row["debate_audio_id"]),
        "n_syllables": n_syll,
        "syllable_rate_per_sec": n_syll / max(0.05, duration),
        "articulation_rate_syll_per_sec": (
            n_syll / speech_sec if speech_sec and speech_sec > 0.05 else np.nan
        ),
    }

    # Slice + openSMILE
    s = max(0, int(float(row["start_sec"]) * sr))
    e = min(len(audio), int(float(row["end_sec"]) * sr))
    if e - s < int(MIN_DURATION_SEC * sr):
        out.update({f"egemaps_{name}": np.nan for name in OPENSMILE_FEATURE_NAMES})
        out["feature_status"] = "too_short"
        return out

    clip = audio[s:e]
    out.update(opensmile_features(clip, sr))
    out["feature_status"] = "ok"
    return out


# === main ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default=str(MASTER_CSV))
    ap.add_argument("--acoustic-csv", default=str(ACOUSTIC_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--debates", nargs="*")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    print(f"[load] {args.master_csv}")
    master = pd.read_csv(args.master_csv, low_memory=False)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)

    print(f"[load] {args.acoustic_csv}")
    acoustic = pd.read_csv(args.acoustic_csv)
    speech_lookup = dict(zip(acoustic["sentence_id"], acoustic["speech_duration_sec"]))
    print(f"  speech_duration available for {len(speech_lookup):,} sentences")

    if args.debates:
        master = master[master["debate_audio_id"].isin(args.debates)]
    if args.limit:
        master = master.head(args.limit)
    print(f"  processing {len(master):,} sentences across {master['debate_audio_id'].nunique()} debates")

    debates = sorted(master["debate_audio_id"].unique(), key=str)
    out_rows: list[dict] = []

    for di, did in enumerate(debates, 1):
        sub = master[master["debate_audio_id"] == did].reset_index(drop=True)
        wav_path = AUDIO_DIR / f"{did}.wav"
        if not wav_path.exists():
            print(f"[skip] {did}: WAV missing")
            continue
        print(f"[{di}/{len(debates)}] {did}: loading audio + processing {len(sub)} sentences…")
        audio, sr = load_debate_audio(wav_path)

        for idx, row in sub.iterrows():
            try:
                out_rows.append(process_sentence(audio, sr, row, speech_lookup))
            except Exception as e:
                out_rows.append({
                    "sentence_id": row["sentence_id"],
                    "debate_audio_id": did,
                    "n_syllables": np.nan,
                    "syllable_rate_per_sec": np.nan,
                    "articulation_rate_syll_per_sec": np.nan,
                    **{f"egemaps_{n}": np.nan for n in OPENSMILE_FEATURE_NAMES},
                    "feature_status": f"error: {type(e).__name__}",
                })

            if (idx + 1) % 250 == 0:
                print(f"    {idx + 1}/{len(sub)}")

        del audio

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}  ({len(out):,} rows × {len(out.columns)} columns)")

    print("\n=== Status counts ===")
    print(out["feature_status"].value_counts().to_string())
    print("\n=== Sample feature values ===")
    sample_cols = ["n_syllables", "syllable_rate_per_sec", "articulation_rate_syll_per_sec",
                   "egemaps_F0semitoneFrom27.5Hz_sma3nz_amean",
                   "egemaps_loudness_sma3_amean",
                   "egemaps_jitterLocal_sma3nz_amean",
                   "egemaps_HNRdBACF_sma3nz_amean"]
    print(out[sample_cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
