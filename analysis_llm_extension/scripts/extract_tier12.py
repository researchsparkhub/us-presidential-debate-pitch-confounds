#!/usr/bin/env python3
"""
Tier 1 (higher moments) + Tier 2 (dynamics) of the semitone F0 contour,
extending the surviving measure of the main paper.

Replicates the ORIGINAL per-sentence-clip openSMILE extraction method exactly
(process each sentence's audio clip independently through the eGeMAPSv02
LLD extractor, matching extract_supplementary_features.py's per-clip calls),
verified to reproduce the existing egemaps_..._stddevNorm / pctlrange0-2
functionals to within float rounding (max abs diff 0.006 over a 39-sentence
spot check).

Frame hop is 10 ms (sma3nz). Voiced frames are those with F0 > 0.

Tier 1 (distributional; order-independent, computed on all voiced frames):
  skewness           - scipy Fisher-Pearson skew
  kurtosis           - scipy Fisher excess kurtosis (0 = normal)
  iqr_range_ratio    - (p75-p25) / (p80-p20), a robustness cross-check on shape

Tier 2 (dynamic; computed only across CONTIGUOUS voiced frame pairs, so a
velocity/turning-point estimate never spans a silent or unvoiced gap):
  velocity_mean_abs  - mean |semitone/10ms| frame-to-frame change
  velocity_sd        - SD of that same signed change
  accel_mean_abs     - mean |second difference| (smoothness of movement)
  turning_pt_rate    - local maxima+minima per second of voiced speech
  total_variation_norm - sum(|consecutive diffs|) / voiced duration (semitones/sec)
  declination_slope  - OLS slope of semitone F0 on time, all voiced frames (semitones/sec)
  declination_resid_sd - SD of residuals after removing that linear trend

Minimum 10 voiced frames (100 ms of voiced speech) required; this is a floor,
not a guarantee of reliability, and reliability is assessed after the fact by
duration stratification exactly as done for HNR in the main paper.
"""
import sys, time, csv, warnings
import numpy as np
import librosa
import opensmile
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = "/Users/lavanya/debate_analysis"
MASTER = f"{PROJECT}/outputs/master_sentences.csv"
AUDIO_DIR = "/Users/lavanya/debate_audio_project/audio"
OUT = "/private/tmp/claude-501/-Users-lavanya-debate-analysis/c2e6601f-99ec-4364-9a03-d6e1384a7237/scratchpad/tier12_features.csv"
EXCLUDE_DEBATES = {"1034", "1036"}
SR = 16000
MIN_VOICED_FRAMES = 10
HOP_SEC = 0.01

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

FIELDS = [
    "sentence_id", "debate_audio_id", "n_voiced_frames", "voiced_duration_sec",
    "skewness", "kurtosis", "iqr_range_ratio",
    "velocity_mean_abs", "velocity_sd", "accel_mean_abs",
    "turning_pt_rate", "total_variation_norm",
    "declination_slope", "declination_resid_sd",
]


def contiguous_runs(voiced_mask):
    """Yield index arrays for maximal runs of consecutive True in voiced_mask."""
    idx = np.flatnonzero(voiced_mask)
    if len(idx) == 0:
        return
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        if len(run) >= 2:
            yield run


def tier_features(f0_semitone, frame_times):
    voiced_mask = f0_semitone > 0
    voiced = f0_semitone[voiced_mask]
    n = len(voiced)
    if n < MIN_VOICED_FRAMES:
        return None

    # Tier 1: distributional, order-independent
    skew = float(stats.skew(voiced, bias=False))
    kurt = float(stats.kurtosis(voiced, fisher=True, bias=False))
    p20, p25, p75, p80 = np.percentile(voiced, [20, 25, 75, 80])
    denom = (p80 - p20)
    iqr_ratio = float((p75 - p25) / denom) if denom > 1e-9 else np.nan

    # Tier 2: dynamics, only within contiguous voiced runs
    vel, acc, tv, turns, voiced_time = [], [], 0.0, 0, 0.0
    for run in contiguous_runs(voiced_mask):
        seg = f0_semitone[run]
        d1 = np.diff(seg)
        vel.append(d1)
        tv += np.sum(np.abs(d1))
        voiced_time += (len(run) - 1) * HOP_SEC
        if len(seg) >= 3:
            d2 = np.diff(d1)
            acc.append(d2)
            sign = np.sign(d1)
            sign = sign[sign != 0]
            if len(sign) >= 2:
                turns += int(np.sum(np.diff(sign) != 0))
    if vel:
        vel_all = np.concatenate(vel)
        vel_mean_abs = float(np.mean(np.abs(vel_all)))
        vel_sd = float(np.std(vel_all, ddof=1)) if len(vel_all) > 1 else np.nan
    else:
        vel_mean_abs = vel_sd = np.nan
    acc_mean_abs = float(np.mean(np.abs(np.concatenate(acc)))) if acc else np.nan
    turning_rate = (turns / voiced_time) if voiced_time > 0 else np.nan
    tv_norm = (tv / voiced_time) if voiced_time > 0 else np.nan

    # declination: OLS on ALL voiced frames (order matters, gaps don't)
    t = frame_times[voiced_mask]
    t0 = t - t.mean()
    if np.ptp(t0) > 1e-6:
        slope, intercept = np.polyfit(t0, voiced, 1)
        resid = voiced - (slope * t0 + intercept)
        resid_sd = float(np.std(resid, ddof=1)) if n > 2 else np.nan
    else:
        slope, resid_sd = np.nan, np.nan

    return {
        "n_voiced_frames": n,
        "voiced_duration_sec": round(voiced_time, 4),
        "skewness": skew, "kurtosis": kurt, "iqr_range_ratio": iqr_ratio,
        "velocity_mean_abs": vel_mean_abs, "velocity_sd": vel_sd,
        "accel_mean_abs": acc_mean_abs,
        "turning_pt_rate": turning_rate, "total_variation_norm": tv_norm,
        "declination_slope": float(slope), "declination_resid_sd": resid_sd,
    }


def main():
    only_debates = set(sys.argv[1:]) or None

    rows_by_debate = {}
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            if r["speaker_role"] != "candidate":
                continue
            d = r["debate_audio_id"]
            if d in EXCLUDE_DEBATES:
                continue
            if only_debates and d not in only_debates:
                continue
            rows_by_debate.setdefault(d, []).append(r)

    debates = sorted(rows_by_debate)
    print(f"[plan] {len(debates)} debates, {sum(len(v) for v in rows_by_debate.values())} candidate sentences",
          flush=True)

    out_f = open(OUT, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    writer.writeheader()

    t_start = time.time()
    n_ok = n_short = n_err = 0
    for di, d in enumerate(debates, 1):
        wav = f"{AUDIO_DIR}/{d}.wav"
        t0 = time.time()
        audio, sr = librosa.load(wav, sr=SR, mono=True)
        audio = audio.astype(np.float32)
        rows = rows_by_debate[d]
        for r in rows:
            sid = r["sentence_id"]
            if not r["start_sec"] or not r["end_sec"]:
                n_short += 1
                continue
            try:
                s, e = float(r["start_sec"]), float(r["end_sec"])
            except ValueError:
                n_short += 1
                continue
            si, ei = max(0, int(s * sr)), min(len(audio), int(e * sr))
            if ei - si < int(0.3 * sr):
                n_short += 1
                continue
            clip = audio[si:ei]
            try:
                df = smile.process_signal(clip, sr)
            except Exception:
                n_err += 1
                continue
            f0 = df["F0semitoneFrom27.5Hz_sma3nz"].values
            frame_times = df.index.get_level_values("start").total_seconds().values
            feats = tier_features(f0, frame_times)
            if feats is None:
                n_short += 1
                continue
            feats["sentence_id"] = sid
            feats["debate_audio_id"] = d
            writer.writerow(feats)
            n_ok += 1
        out_f.flush()
        elapsed = time.time() - t_start
        print(f"[{di:2d}/{len(debates)}] debate {d}: {len(rows)} sentences, "
              f"debate took {time.time()-t0:.0f}s, total elapsed {elapsed/60:.1f} min, "
              f"ok={n_ok} short={n_short} err={n_err}", flush=True)

    out_f.close()
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
