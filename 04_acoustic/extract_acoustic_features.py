#!/usr/bin/env python3
"""
Stage 4 — extract per-sentence acoustic features.

For each sentence in outputs/master_sentences.csv, slice [start_sec, end_sec]
from the source WAV and compute prosody / voice-quality / spectral / timing
features. Features are joined back via sentence_id.

Tools used:
    parselmouth (Praat backend): F0, intensity, jitter, shimmer, HNR
    librosa: spectral centroid, rolloff, ZCR, RMS energy
    direct timing: speech rate, pause statistics

Output:
    outputs/acoustic_features.csv  (one row per sentence, ~25 numeric columns)

Run:
    python 04_acoustic/extract_acoustic_features.py
    python 04_acoustic/extract_acoustic_features.py --debates 1003 1010
    python 04_acoustic/extract_acoustic_features.py --limit 500   # quick sanity run
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import librosa
import soundfile as sf

# Silence the volume of parselmouth/librosa noise about short signals
warnings.filterwarnings("ignore")


PROJECT = Path("/Users/lavanya/debate_analysis")
MASTER_CSV = PROJECT / "outputs" / "master_sentences.csv"
AUDIO_DIR = PROJECT / "data" / "audio"
OUT_CSV = PROJECT / "outputs" / "acoustic_features.csv"

# Audio processing constants
TARGET_SR = 16000           # Whisper + MFA used 16k, keep consistent
MIN_DURATION_SEC = 0.3      # skip sentences shorter than this (unreliable F0)
F0_MIN_HZ = 75              # typical male lower bound
F0_MAX_HZ = 500             # typical female upper bound; covers shouting
SILENCE_THRESHOLD_DB = -25  # below this is treated as a pause
MIN_PAUSE_SEC = 0.15        # silences shorter than this aren't counted as pauses


# === per-sentence feature extractors =======================================

def f0_features(sound: parselmouth.Sound) -> dict:
    """Pitch (F0) statistics. Returns NaNs on signals too short for analysis."""
    try:
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=F0_MIN_HZ, pitch_ceiling=F0_MAX_HZ)
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]
        if len(voiced) < 5:
            return {k: np.nan for k in
                    ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
                     "f0_slope_hz_per_s", "voiced_frac"]}
        # Linear slope (Hz / sec) via simple regression on voiced frames
        times = np.array([pitch.xs()[i] for i, v in enumerate(values) if v > 0])
        if len(times) >= 2 and np.ptp(times) > 0.05:
            slope = float(np.polyfit(times, voiced, 1)[0])
        else:
            slope = 0.0
        return {
            "f0_mean":           float(np.mean(voiced)),
            "f0_std":            float(np.std(voiced)),
            "f0_min":            float(np.min(voiced)),
            "f0_max":            float(np.max(voiced)),
            "f0_range":          float(np.ptp(voiced)),
            "f0_slope_hz_per_s": slope,
            "voiced_frac":       float(len(voiced) / max(1, len(values))),
        }
    except Exception:
        return {k: np.nan for k in
                ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
                 "f0_slope_hz_per_s", "voiced_frac"]}


def intensity_features(sound: parselmouth.Sound) -> dict:
    try:
        intensity = sound.to_intensity(time_step=0.01)
        vals = intensity.values.T.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) < 5:
            return {k: np.nan for k in
                    ["intensity_mean_db", "intensity_std_db", "intensity_range_db"]}
        return {
            "intensity_mean_db":  float(np.mean(vals)),
            "intensity_std_db":   float(np.std(vals)),
            "intensity_range_db": float(np.ptp(vals)),
        }
    except Exception:
        return {k: np.nan for k in
                ["intensity_mean_db", "intensity_std_db", "intensity_range_db"]}


def voice_quality_features(sound: parselmouth.Sound) -> dict:
    """Jitter (pitch period perturbation), shimmer (amplitude perturbation), HNR."""
    try:
        # PointProcess for jitter/shimmer needs pitch info first
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", F0_MIN_HZ, F0_MAX_HZ)
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, pp], "Get shimmer (local)",
                                         0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=F0_MIN_HZ)
        hnr_vals = harmonicity.values.flatten()
        hnr_vals = hnr_vals[(hnr_vals != -200) & ~np.isnan(hnr_vals)]
        hnr_mean = float(np.mean(hnr_vals)) if len(hnr_vals) else np.nan
        return {
            "jitter_local":   float(jitter) if not np.isnan(jitter) else np.nan,
            "shimmer_local":  float(shimmer) if not np.isnan(shimmer) else np.nan,
            "hnr_mean_db":    hnr_mean,
        }
    except Exception:
        return {k: np.nan for k in ["jitter_local", "shimmer_local", "hnr_mean_db"]}


def spectral_features(audio: np.ndarray, sr: int) -> dict:
    """Spectral centroid, rolloff, ZCR, RMS energy (librosa)."""
    if len(audio) < 1024:
        return {k: np.nan for k in
                ["spectral_centroid_mean", "spectral_rolloff_mean",
                 "zero_crossing_rate_mean", "rms_mean", "rms_std"]}
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        return {
            "spectral_centroid_mean":  float(np.mean(centroid)),
            "spectral_rolloff_mean":   float(np.mean(rolloff)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "rms_mean":                float(np.mean(rms)),
            "rms_std":                 float(np.std(rms)),
        }
    except Exception:
        return {k: np.nan for k in
                ["spectral_centroid_mean", "spectral_rolloff_mean",
                 "zero_crossing_rate_mean", "rms_mean", "rms_std"]}


def pause_features(sound: parselmouth.Sound, duration_sec: float, n_words: int) -> dict:
    """Pauses detected as low-intensity gaps. Also speech-rate metrics."""
    try:
        intensity = sound.to_intensity(time_step=0.01)
        times = intensity.xs()
        vals = intensity.values.T.flatten()
        # Threshold relative to the loudest 90th percentile of the sentence
        if len(vals) == 0:
            raise ValueError("empty intensity")
        loud_floor = np.nanpercentile(vals, 90) + SILENCE_THRESHOLD_DB
        is_silent = vals < loud_floor

        # Find contiguous silent runs of duration >= MIN_PAUSE_SEC
        pause_durations: list[float] = []
        in_pause = False
        pause_start_t = 0.0
        for t, s in zip(times, is_silent):
            if s and not in_pause:
                in_pause = True; pause_start_t = t
            elif not s and in_pause:
                in_pause = False
                d = t - pause_start_t
                if d >= MIN_PAUSE_SEC:
                    pause_durations.append(d)
        # Close trailing pause
        if in_pause:
            d = times[-1] - pause_start_t
            if d >= MIN_PAUSE_SEC:
                pause_durations.append(d)

        total_pause = float(sum(pause_durations))
        n_pauses = len(pause_durations)
        speech_sec = max(0.0, duration_sec - total_pause)
        return {
            "total_pause_sec":     total_pause,
            "n_pauses":            n_pauses,
            "mean_pause_sec":      float(np.mean(pause_durations)) if pause_durations else 0.0,
            "speech_duration_sec": speech_sec,
            "speech_rate_wps":     n_words / max(0.05, duration_sec),
            "articulation_rate_wps": n_words / max(0.05, speech_sec),
        }
    except Exception:
        return {
            "total_pause_sec":       np.nan,
            "n_pauses":              np.nan,
            "mean_pause_sec":        np.nan,
            "speech_duration_sec":   np.nan,
            "speech_rate_wps":       n_words / max(0.05, duration_sec) if duration_sec > 0 else np.nan,
            "articulation_rate_wps": np.nan,
        }


# === per-debate processing =================================================

def load_debate_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load full WAV at TARGET_SR, mono."""
    audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32), TARGET_SR


def features_for_sentence(audio: np.ndarray, sr: int,
                          start_sec: float, end_sec: float,
                          n_words: int) -> dict:
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    if e - s < int(MIN_DURATION_SEC * sr):
        return {**{k: np.nan for k in ALL_FEATURE_NAMES},
                "feature_status": "too_short"}

    clip = audio[s:e]
    snd = parselmouth.Sound(clip, sampling_frequency=sr)

    feats = {}
    feats.update(f0_features(snd))
    feats.update(intensity_features(snd))
    feats.update(voice_quality_features(snd))
    feats.update(spectral_features(clip, sr))
    feats.update(pause_features(snd, duration_sec=(end_sec - start_sec), n_words=n_words))
    feats["feature_status"] = "ok"
    return feats


ALL_FEATURE_NAMES = [
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
    "f0_slope_hz_per_s", "voiced_frac",
    "intensity_mean_db", "intensity_std_db", "intensity_range_db",
    "jitter_local", "shimmer_local", "hnr_mean_db",
    "spectral_centroid_mean", "spectral_rolloff_mean",
    "zero_crossing_rate_mean", "rms_mean", "rms_std",
    "total_pause_sec", "n_pauses", "mean_pause_sec",
    "speech_duration_sec", "speech_rate_wps", "articulation_rate_wps",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default=str(MASTER_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--debates", nargs="*", help="Limit to specific debate IDs")
    ap.add_argument("--limit", type=int, help="For dev: only process this many sentences")
    args = ap.parse_args()

    print(f"[load] {args.master_csv}")
    master = pd.read_csv(args.master_csv, low_memory=False)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)
    if args.debates:
        master = master[master["debate_audio_id"].isin(args.debates)]
    if args.limit:
        master = master.head(args.limit)
    print(f"  {len(master):,} sentences across {master['debate_audio_id'].nunique()} debates")

    # Process per-debate so we load each WAV once
    out_rows: list[dict] = []
    debates = sorted(master["debate_audio_id"].unique(), key=str)

    for di, did in enumerate(debates, 1):
        sub = master[master["debate_audio_id"] == did].reset_index(drop=True)
        wav_path = AUDIO_DIR / f"{did}.wav"
        if not wav_path.exists():
            print(f"[skip] {did}: WAV missing at {wav_path}")
            continue

        print(f"[{di}/{len(debates)}] {did}: loading audio + processing {len(sub)} sentences…")
        audio, sr = load_debate_audio(wav_path)

        for idx, row in sub.iterrows():
            try:
                feats = features_for_sentence(
                    audio, sr,
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    n_words=int(row["n_words"]),
                )
            except Exception as e:
                feats = {**{k: np.nan for k in ALL_FEATURE_NAMES},
                         "feature_status": f"error: {type(e).__name__}"}

            out_rows.append({
                "sentence_id": row["sentence_id"],
                "debate_audio_id": did,
                "duration_sec": float(row["duration_sec"]),
                "n_words": int(row["n_words"]),
                **feats,
            })

            if (idx + 1) % 500 == 0:
                print(f"    {idx + 1}/{len(sub)}")

        # Free audio array
        del audio

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}  ({len(out):,} rows, {len(out.columns)} columns)")

    # Quick summary
    print("\n=== Status counts ===")
    print(out["feature_status"].value_counts().to_string())
    print("\n=== Feature presence (non-NaN fraction) ===")
    for col in ALL_FEATURE_NAMES[:8]:
        pct = out[col].notna().sum() / len(out) * 100
        print(f"  {col:30s} {pct:5.1f}%")
    print("  …")
    for col in ALL_FEATURE_NAMES[-4:]:
        pct = out[col].notna().sum() / len(out) * 100
        print(f"  {col:30s} {pct:5.1f}%")


if __name__ == "__main__":
    main()
