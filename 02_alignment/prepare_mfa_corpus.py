#!/usr/bin/env python3
"""
Prepare an MFA corpus from official debate transcripts + audio.

For each debate we emit:
  mfa_corpus/<debate_audio_id>/
    <debate_audio_id>.wav      (symlink to the source audio)
    <debate_audio_id>.lab      (all turn texts concatenated, header stripped)
    <debate_audio_id>_turns.json  (turn boundaries for post-processing,
                                    NOT consumed by MFA itself)

MFA only reads the .wav and .lab; the .json is so `parse_mfa_output.py` can
remap aligned words back to turns + speakers without re-parsing transcripts.

Run:
    python 02_alignment/prepare_mfa_corpus.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DEBATES_CSV = PROJECT / "data" / "metadata" / "debates.csv"
AUDIO_DIR = PROJECT / "data" / "audio"
TRANSCRIPT_DIR = PROJECT / "data" / "transcripts"
OUT_DIR = PROJECT / "02_alignment" / "mfa_corpus"

# Match a leading "SPEAKER:" or "FIRST LAST:" with apostrophes/hyphens allowed.
SPEAKER_PREFIX = re.compile(r"^([A-Z][A-Z.'\-]+(?:\s[A-Z][A-Z.'\-]+)?)\s*:\s*(.*)$")
ANNOTATION_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")


def find_transcript_files(debate_audio_id: str) -> list[Path]:
    """Return transcript file paths for a debate. Some debates split across
    parts (e.g. 1004_1.txt + 1004_2.txt)."""
    direct = TRANSCRIPT_DIR / f"{debate_audio_id}.txt"
    if direct.exists():
        return [direct]
    parts = sorted(TRANSCRIPT_DIR.glob(f"{debate_audio_id}_*.txt"))
    return list(parts)


def parse_turns(text: str) -> list[dict]:
    """Walk the transcript and emit one dict per turn.
    Skips header lines (date / PARTICIPANTS / MODERATORS).
    """
    lines = text.splitlines()

    # Find first line that begins with a SPEAKER: prefix (skips arbitrary header)
    body_start = 0
    for i, ln in enumerate(lines):
        if SPEAKER_PREFIX.match(ln.strip()):
            body_start = i
            break

    turns: list[dict] = []
    current_speaker = None
    current_parts: list[str] = []

    def flush():
        nonlocal current_speaker, current_parts
        if current_speaker is None:
            return
        body = " ".join(p.strip() for p in current_parts if p.strip())
        if body:
            turns.append({"speaker_raw": current_speaker, "text": body})
        current_speaker = None
        current_parts = []

    for raw in lines[body_start:]:
        ln = raw.strip()
        if not ln or ANNOTATION_LINE.match(ln):
            continue
        m = SPEAKER_PREFIX.match(ln)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            rest = m.group(2)
            current_parts = [rest] if rest else []
        else:
            if current_speaker is not None:
                current_parts.append(ln)
    flush()
    return turns


def clean_for_mfa(text: str) -> str:
    """Strip parenthetical stage directions like (Applause) and tidy whitespace."""
    text = re.sub(r"\([^)]*\)", " ", text)   # strip (Applause), (Crosstalk), etc.
    text = re.sub(r"\[[^\]]*\]", " ", text)  # strip [inaudible], etc.
    text = re.sub(r"[‘’]", "'", text)  # smart → straight quotes
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    debates = pd.read_csv(DEBATES_CSV)
    debates["debate_audio_id"] = debates["debate_audio_id"].astype(str)
    audio_ids = sorted(debates["debate_audio_id"].unique())

    rows_summary = []

    for did in audio_ids:
        paths = find_transcript_files(did)
        wav_path = AUDIO_DIR / f"{did}.wav"

        if not paths:
            print(f"[skip] {did}: no transcript file in {TRANSCRIPT_DIR}")
            rows_summary.append({"debate_audio_id": did, "n_turns": 0, "n_words": 0,
                                 "wav_present": wav_path.exists(), "status": "no_transcript"})
            continue
        if not wav_path.exists():
            print(f"[skip] {did}: WAV missing at {wav_path}")
            rows_summary.append({"debate_audio_id": did, "n_turns": 0, "n_words": 0,
                                 "wav_present": False, "status": "no_audio"})
            continue

        # Parse and concatenate turns from all parts of this debate's transcript
        all_turns: list[dict] = []
        for p in paths:
            text = p.read_text(encoding="utf-8")
            all_turns.extend(parse_turns(text))

        if not all_turns:
            print(f"[skip] {did}: no turns parsed")
            rows_summary.append({"debate_audio_id": did, "n_turns": 0, "n_words": 0,
                                 "wav_present": True, "status": "no_turns"})
            continue

        # Build the .lab text and track word-index ranges per turn
        words: list[str] = []
        turn_records: list[dict] = []
        for t_idx, t in enumerate(all_turns, start=1):
            cleaned = clean_for_mfa(t["text"])
            t_words = cleaned.split()
            if not t_words:
                continue
            start_idx = len(words)
            words.extend(t_words)
            end_idx = len(words)  # exclusive
            turn_records.append({
                "turn_index": t_idx,
                "turn_id": f"{did}_t{t_idx:04d}",
                "speaker_raw": t["speaker_raw"],
                "word_start_idx": start_idx,
                "word_end_idx": end_idx,
                "n_words": end_idx - start_idx,
                "text": cleaned,
            })
        all_text = " ".join(words)

        # Write per-debate corpus folder
        debate_dir = OUT_DIR / did
        debate_dir.mkdir(exist_ok=True)
        wav_link = debate_dir / f"{did}.wav"
        if wav_link.exists() or wav_link.is_symlink():
            wav_link.unlink()
        os.symlink(wav_path, wav_link)
        (debate_dir / f"{did}.lab").write_text(all_text + "\n")
        (debate_dir / f"{did}_turns.json").write_text(json.dumps({
            "debate_audio_id": did,
            "n_words": len(words),
            "turns": turn_records,
        }, indent=2))

        rows_summary.append({
            "debate_audio_id": did,
            "n_turns": len(turn_records),
            "n_words": len(words),
            "wav_present": True,
            "status": "ready",
        })
        print(f"[{did}] {len(turn_records)} turns, {len(words)} words → {debate_dir}/")

    summary = pd.DataFrame(rows_summary)
    summary_path = OUT_DIR.parent / "mfa_corpus_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[write] {summary_path}")
    ready = summary[summary["status"] == "ready"]
    print(f"\n  {len(ready)} / {len(summary)} debates ready for MFA")
    if (summary["status"] != "ready").any():
        print(summary[summary["status"] != "ready"][["debate_audio_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
