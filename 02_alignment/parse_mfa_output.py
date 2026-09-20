#!/usr/bin/env python3
"""
Convert MFA TextGrid output to clean per-word and per-turn timing CSVs.

For each aligned debate this script reads:
  mfa_output/<debate_id>/<debate_id>.TextGrid     ← MFA word-level alignment
  mfa_corpus/<debate_id>/<debate_id>_turns.json   ← turn boundaries we recorded

and emits:
  mfa_words.csv   one row per word: debate_audio_id, turn_id, speaker_raw, word, start_sec, end_sec
  mfa_turns.csv   one row per turn: debate_audio_id, turn_id, speaker_raw, n_words, start_sec, end_sec, text

Run after MFA has finished:
    python 02_alignment/parse_mfa_output.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path("/Users/lavanya/debate_analysis")
MFA_OUTPUT_DIR = PROJECT / "02_alignment" / "mfa_output"
CORPUS_DIR = PROJECT / "02_alignment" / "mfa_corpus"
WORDS_CSV = PROJECT / "02_alignment" / "mfa_words.csv"
TURNS_CSV = PROJECT / "02_alignment" / "mfa_turns.csv"


def parse_textgrid_words(tg_path: Path) -> list[tuple[float, float, str]]:
    """Minimal TextGrid parser — extracts the `words` IntervalTier.

    Returns list of (start_sec, end_sec, word). Empty intervals (silence
    between words) are preserved with empty word string.
    """
    text = tg_path.read_text(encoding="utf-8")
    # Locate the words tier — MFA emits two tiers: "words" and "phones"
    # Tier blocks look like:
    #   item [n]:
    #       class = "IntervalTier"
    #       name = "words"
    #       intervals: size = N
    #       intervals [k]: xmin = .. xmax = .. text = ".."
    tier_match = re.search(
        r'name\s*=\s*"words".*?intervals:\s*size\s*=\s*\d+\s*(.*?)(?=\s*item\s*\[|\Z)',
        text, re.DOTALL,
    )
    if not tier_match:
        return []
    body = tier_match.group(1)

    intervals = []
    for m in re.finditer(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"([^\"]*)\"",
        body,
    ):
        start, end, word = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        intervals.append((start, end, word))
    return intervals


def main():
    if not MFA_OUTPUT_DIR.exists():
        raise SystemExit(f"MFA output dir missing: {MFA_OUTPUT_DIR}\nRun ./run_mfa.sh first.")

    word_rows = []
    turn_rows = []
    summary = []

    for tg_path in sorted(MFA_OUTPUT_DIR.glob("*/*.TextGrid")):
        did = tg_path.parent.name
        turns_meta_path = CORPUS_DIR / did / f"{did}_turns.json"
        if not turns_meta_path.exists():
            print(f"[skip] {did}: no turns.json next to corpus")
            continue

        meta = json.loads(turns_meta_path.read_text())
        turn_records = meta["turns"]

        intervals = parse_textgrid_words(tg_path)
        # Filter out empty intervals (silences) for the word index
        word_intervals = [(s, e, w) for (s, e, w) in intervals if w]
        n_words_aligned = len(word_intervals)
        n_words_expected = meta.get("n_words", 0)

        if n_words_aligned == 0:
            print(f"[warn] {did}: no aligned words in TextGrid")
            continue

        if n_words_aligned != n_words_expected:
            print(f"[note] {did}: aligned={n_words_aligned}, expected={n_words_expected} "
                  f"(MFA may have collapsed punctuation or split tokens)")

        # Walk turns and slice the word_intervals by recorded indices.
        # If counts diverge, we proportionally scale the slice — better than nothing.
        scale = n_words_aligned / max(1, n_words_expected)
        for t in turn_records:
            wi = int(round(t["word_start_idx"] * scale))
            wj = int(round(t["word_end_idx"] * scale))
            wj = max(wj, wi + 1)
            chunk = word_intervals[wi:wj]
            if not chunk:
                continue
            t_start = chunk[0][0]
            t_end = chunk[-1][1]
            turn_rows.append({
                "debate_audio_id": did,
                "turn_id": t["turn_id"],
                "speaker_raw": t["speaker_raw"],
                "n_words": len(chunk),
                "start_sec": round(t_start, 3),
                "end_sec": round(t_end, 3),
                "duration_sec": round(t_end - t_start, 3),
                "text": t["text"],
            })
            for (ws, we, w) in chunk:
                word_rows.append({
                    "debate_audio_id": did,
                    "turn_id": t["turn_id"],
                    "speaker_raw": t["speaker_raw"],
                    "word": w,
                    "start_sec": round(ws, 3),
                    "end_sec": round(we, 3),
                })

        summary.append({
            "debate_audio_id": did,
            "n_words_expected": n_words_expected,
            "n_words_aligned": n_words_aligned,
            "n_turns_aligned": len([r for r in turn_rows if r["debate_audio_id"] == did]),
            "duration_sec": round(intervals[-1][1] if intervals else 0.0, 1),
        })
        print(f"[{did}] {n_words_aligned} words, "
              f"{summary[-1]['n_turns_aligned']} turns aligned over "
              f"{summary[-1]['duration_sec']}s")

    if not word_rows:
        raise SystemExit("No alignments parsed. Check MFA logs.")

    pd.DataFrame(word_rows).to_csv(WORDS_CSV, index=False)
    pd.DataFrame(turn_rows).to_csv(TURNS_CSV, index=False)
    pd.DataFrame(summary).to_csv(PROJECT / "02_alignment" / "mfa_alignment_summary.csv", index=False)

    print(f"\n[write] {WORDS_CSV}  ({len(word_rows)} words)")
    print(f"[write] {TURNS_CSV}  ({len(turn_rows)} turns)")
    print(f"[write] mfa_alignment_summary.csv ({len(summary)} debates)")


if __name__ == "__main__":
    main()
