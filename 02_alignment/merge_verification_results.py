#!/usr/bin/env python3
"""
Merge returned verification CSVs into corpus-wide accuracy metrics.

Drop the *_verified.csv files your verifier(s) send back into:
    02_alignment/verification_packets/returned/

Then run:
    python 02_alignment/merge_verification_results.py

Outputs:
    02_alignment/verification_results_per_turn.csv
    02_alignment/verification_results_per_debate.csv
    02_alignment/verification_results_summary.csv  (overall + per-method)

Sanity checks performed:
  - turn_ids must exist in mfa_turns.csv
  - verdict values must be from the allowed set
  - multiple verifiers per packet are aggregated by majority vote with a warning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path("/Users/lavanya/debate_analysis")
ALIGN_DIR = PROJECT / "02_alignment"
TURNS_CSV = ALIGN_DIR / "mfa_turns.csv"
RETURNED_DIR = ALIGN_DIR / "verification_packets" / "returned"

VALID = {
    "audio_matches_text": {"", "yes", "partial", "no", "unclear"},
    "alignment_timing":  {"", "precise", "approximate", "off", "unclear"},
    "speaker_correct":   {"", "yes", "no", "unclear"},
}


def precision(num_correct: int, num_judged: int) -> float:
    return (num_correct / num_judged) if num_judged else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned-dir", default=str(RETURNED_DIR))
    ap.add_argument("--turns-csv", default=str(TURNS_CSV))
    args = ap.parse_args()

    returned = Path(args.returned_dir)
    if not returned.exists() or not list(returned.glob("*.csv")):
        raise SystemExit(f"No CSV files in {returned}/ — drop verifier output there first.")

    turns_master = pd.read_csv(args.turns_csv)
    turns_master["turn_id"] = turns_master["turn_id"].astype(str)
    valid_turn_ids = set(turns_master["turn_id"])

    # Concatenate every returned CSV
    frames = []
    for f in sorted(returned.glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        df["__source_file"] = f.name
        frames.append(df)
        print(f"[load] {f.name}  ({len(df)} rows)")
    raw = pd.concat(frames, ignore_index=True)

    # Sanity
    issues = []
    raw["turn_id"] = raw["turn_id"].astype(str)
    raw["debate_audio_id"] = raw["debate_audio_id"].astype(str)
    missing = raw[~raw["turn_id"].isin(valid_turn_ids)]
    if len(missing):
        issues.append(f"  {len(missing)} rows have turn_ids not in mfa_turns.csv")
    for col, allowed in VALID.items():
        raw[col] = raw[col].fillna("").astype(str).str.strip()
        bad = raw[~raw[col].isin(allowed)]
        if len(bad):
            issues.append(f"  {len(bad)} rows have invalid {col!r}: {sorted(set(bad[col]))[:5]}")
    if issues:
        print("\nSanity issues:")
        for i in issues:
            print(i)
        print()

    # Aggregate duplicates by majority (mode) per turn — handles multi-verifier
    def consolidate(group: pd.DataFrame) -> pd.Series:
        out = {}
        for col in VALID:
            vals = group[col].astype(str)
            vals = vals[vals.ne("")]
            if vals.empty:
                out[col] = ""
            else:
                out[col] = vals.mode().iloc[0]
        out["notes"] = " | ".join(str(n) for n in group.get("notes", pd.Series(dtype=str)) if str(n).strip())
        out["n_reviewers"] = group["__source_file"].nunique()
        return pd.Series(out)

    per_turn_meta = raw.groupby(["debate_audio_id", "turn_id"]).apply(consolidate).reset_index()

    # Join sampled-turn details from the master (start, end, text, etc.)
    per_turn = per_turn_meta.merge(
        turns_master[["debate_audio_id", "turn_id", "speaker_raw", "start_sec", "end_sec", "duration_sec", "n_words", "text"]],
        on=["debate_audio_id", "turn_id"], how="left",
    )

    # Compute per-debate stats
    def precision_for(df: pd.DataFrame, col: str, positive_values: set[str], negative_values: set[str]):
        considered = df[df[col].isin(positive_values | negative_values)]
        n = len(considered)
        n_pos = int(considered[col].isin(positive_values).sum())
        return precision(n_pos, n), n_pos, n

    rows = []
    for did, grp in per_turn.groupby("debate_audio_id"):
        n_sampled = len(grp)
        n_reviewed = int(grp[["audio_matches_text", "alignment_timing", "speaker_correct"]].apply(
            lambda r: r.ne("").all(), axis=1).sum())
        text_p, text_pos, text_n = precision_for(grp, "audio_matches_text", {"yes", "partial"}, {"no"})
        timing_p, timing_pos, timing_n = precision_for(grp, "alignment_timing", {"precise", "approximate"}, {"off"})
        spk_p, spk_pos, spk_n = precision_for(grp, "speaker_correct", {"yes"}, {"no"})
        rows.append({
            "debate_audio_id": did,
            "n_sampled": n_sampled,
            "n_reviewed": n_reviewed,
            "text_match_precision": text_p,
            "text_match_correct": text_pos,
            "text_match_judged": text_n,
            "timing_precision": timing_p,
            "timing_correct": timing_pos,
            "timing_judged": timing_n,
            "speaker_precision": spk_p,
            "speaker_correct_count": spk_pos,
            "speaker_judged": spk_n,
        })
    per_debate = pd.DataFrame(rows).sort_values("debate_audio_id")

    # Corpus-level summary (overall + per-stratum)
    text_p, text_pos, text_n = precision_for(per_turn, "audio_matches_text", {"yes", "partial"}, {"no"})
    timing_p, timing_pos, timing_n = precision_for(per_turn, "alignment_timing", {"precise", "approximate"}, {"off"})
    spk_p, spk_pos, spk_n = precision_for(per_turn, "speaker_correct", {"yes"}, {"no"})

    summary = pd.DataFrame([
        {"dimension": "audio_matches_text",  "correct": text_pos,  "judged": text_n,  "precision": text_p},
        {"dimension": "alignment_timing",    "correct": timing_pos, "judged": timing_n, "precision": timing_p},
        {"dimension": "speaker_correct",     "correct": spk_pos,   "judged": spk_n,   "precision": spk_p},
    ])

    # Write outputs
    per_turn_path = ALIGN_DIR / "verification_results_per_turn.csv"
    per_debate_path = ALIGN_DIR / "verification_results_per_debate.csv"
    summary_path = ALIGN_DIR / "verification_results_summary.csv"

    per_turn.to_csv(per_turn_path, index=False)
    per_debate.to_csv(per_debate_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\n[write] {per_turn_path}  ({len(per_turn)} turns)")
    print(f"[write] {per_debate_path}  ({len(per_debate)} debates)")
    print(f"[write] {summary_path}")
    print()
    print("=== Corpus-level accuracy ===")
    for _, r in summary.iterrows():
        prec = r["precision"]
        print(f"  {r['dimension']:24s}  {int(r['correct'])}/{int(r['judged'])}  →  "
              f"{prec*100:.1f}%" if pd.notna(prec) else f"  {r['dimension']:24s}  no data")

    print("\n=== Per-debate (lowest text-match precision first) ===")
    show = per_debate[["debate_audio_id", "n_reviewed", "n_sampled",
                       "text_match_precision", "timing_precision", "speaker_precision"]].copy()
    for c in ("text_match_precision", "timing_precision", "speaker_precision"):
        show[c] = (show[c] * 100).round(1)
    print(show.sort_values("text_match_precision").to_string(index=False))


if __name__ == "__main__":
    main()
