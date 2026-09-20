#!/usr/bin/env python3
"""
Build a cluster→speaker mapping for the 2024 debates (1034, 1036).

The 2024 official transcripts can't be aligned to audio (paraphrased), so we
have no transcript-anchored speaker labels. Instead we use the existing
pyannote diarization (from the old project) — each Whisper sentence will be
assigned to whichever pyannote cluster dominates its time window.

This script PRE-FILLS the mapping with educated guesses based on cluster
duration within the debate window:
  - Top 2 clusters by duration  → candidates (left blank for user to label)
  - Next ~2 clusters             → moderators
  - Rest                         → audience / non-debate

The user can edit `data/metadata/cluster_to_speaker_2024.csv` to assign real
names and re-run build_master_sentences.py.

Run:
    python 03_speakers/build_2024_speaker_mapping.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT = Path("/Users/lavanya/debate_analysis")
OLD_DIARIZATION_CSV = Path("/Users/lavanya/debate_audio_project/Week_4_speaker diarization/output/segments_raw.csv")
ALIGN_DIR = PROJECT / "02_alignment"
META_DIR = PROJECT / "data" / "metadata"
OUT_CSV = META_DIR / "cluster_to_speaker_2024.csv"

TARGETS = ["1034", "1036"]

# Known facts to pre-populate the suggestion column
KNOWN_2024 = {
    "1034": {
        "candidate_1_hint": "Joe Biden",
        "candidate_2_hint": "Donald Trump",
        "moderator_hint":   "Dana Bash / Jake Tapper",
    },
    "1036": {
        "candidate_1_hint": "Kamala Harris",
        "candidate_2_hint": "Donald Trump",
        "moderator_hint":   "Linsey Davis / David Muir",
    },
}


def load_trim_window(did: str) -> tuple[float, float]:
    path = ALIGN_DIR / "mfa_corpus" / did / f"{did}_trim_offset.json"
    if not path.exists():
        return (0.0, float("inf"))
    o = json.loads(path.read_text())
    return (float(o["trim_start"]), float(o["trim_end"]))


def main():
    META_DIR.mkdir(parents=True, exist_ok=True)

    if not OLD_DIARIZATION_CSV.exists():
        raise SystemExit(f"pyannote output not found at {OLD_DIARIZATION_CSV}")

    print(f"[load] {OLD_DIARIZATION_CSV}")
    seg = pd.read_csv(OLD_DIARIZATION_CSV, low_memory=False)
    seg["debate_audio_id"] = seg["debate_audio_id"].astype(str)
    seg["dur"] = seg["end_sec"] - seg["start_sec"]

    rows = []
    for did in TARGETS:
        trim_start, trim_end = load_trim_window(did)
        sub = seg[seg["debate_audio_id"] == did].copy()
        # Restrict to debate window only
        in_window = (sub["start_sec"] >= trim_start) & (sub["end_sec"] <= trim_end)
        sub = sub[in_window]
        if sub.empty:
            print(f"[skip] {did}: no diarized segments inside trim window")
            continue

        by_cluster = (
            sub.groupby("diarized_speaker")["dur"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        print(f"\n=== {did} (debate window {trim_start:.0f}–{trim_end:.0f}s = {(trim_end-trim_start)/60:.1f} min) ===")
        print(f"  {len(by_cluster)} clusters with audio inside the window")

        hints = KNOWN_2024.get(did, {})
        for rank, row in enumerate(by_cluster.itertuples(index=False)):
            cluster, total = row.diarized_speaker, row.dur
            # Suggest role by rank
            if rank == 0:
                role_guess = "candidate"
                speaker_guess = hints.get("candidate_1_hint", "")
            elif rank == 1:
                role_guess = "candidate"
                speaker_guess = hints.get("candidate_2_hint", "")
            elif rank in (2, 3):
                role_guess = "moderator"
                speaker_guess = hints.get("moderator_hint", "")
            elif total >= 30:                     # ≥30 s of audio: likely a real participant
                role_guess = "panelist_or_audience"
                speaker_guess = ""
            else:
                role_guess = "noise_or_skip"
                speaker_guess = ""

            rows.append({
                "debate_audio_id": did,
                "pyannote_cluster": cluster,
                "rank_by_duration": rank + 1,
                "total_sec_in_window": round(total, 1),
                "minutes": round(total / 60, 2),
                "suggested_role": role_guess,
                "suggested_speaker": speaker_guess,
                "verified": "",          # user fills "yes" after confirming
                "final_speaker": "",     # user fills with the actual person's name
                "final_role": "",        # user fills: candidate / moderator / audience / skip
            })

        # Print the suggestion summary
        for r in [x for x in rows if x["debate_audio_id"] == did][:8]:
            print(f"  #{r['rank_by_duration']} {r['pyannote_cluster']:<12s} "
                  f"{r['minutes']:>5.1f}m → {r['suggested_role']:<22s} ({r['suggested_speaker']})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  ({len(out)} cluster rows)")
    print("""
Next steps (manual, optional):
  1. Open the CSV and listen to a clip from each top-ranked cluster to verify
     which one is which candidate. The big two should be the candidates;
     the next biggest are likely moderators.
  2. Fill in `final_speaker` and `final_role` columns and set `verified=yes`.
  3. Re-run: python 03_speakers/build_master_sentences.py
""")


if __name__ == "__main__":
    main()
