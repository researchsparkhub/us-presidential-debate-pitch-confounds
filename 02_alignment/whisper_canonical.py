#!/usr/bin/env python3
"""
Produce a Whisper-canonical sentence-level timing file.

For debates where the official transcript can't be aligned to audio
(e.g. the 2024 broadcasts whose "transcripts" are heavily paraphrased
summaries), we treat Whisper's verbatim ASR as the canonical text.

Each Whisper word already carries (start, end). We segment the Whisper
word stream into sentences using punctuation in Whisper's own output,
then emit one row per sentence in the SAME schema as
text_aligned_sentences.csv (minus an official turn_id, which doesn't
apply — Whisper doesn't know about official transcript turns).

Speaker labels are left as "UNKNOWN" here; they'll be filled in stage 3
by joining to diarization output or by a verification pass.

Run:
    python 02_alignment/whisper_canonical.py            # default: 1034 + 1036
    python 02_alignment/whisper_canonical.py --debates 1034 1036 1035
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
ALIGN_DIR = PROJECT / "02_alignment"
WHISPER_DIR = PROJECT / "data" / "whisper_aligned"

DEFAULT_TARGETS = ["1034", "1036"]

OUT_SENTENCES_CSV = ALIGN_DIR / "whisper_canonical_sentences.csv"
OUT_WORDS_CSV = ALIGN_DIR / "whisper_canonical_words.csv"
OUT_SUMMARY_CSV = ALIGN_DIR / "whisper_canonical_summary.csv"

# Cluster words into sentences when one ends with . ! or ? OR a long pause
PUNCT_END = re.compile(r"[.!?]+\s*$")
MIN_SENTENCE_GAP_SEC = 1.0   # treat a >1s pause as a sentence boundary fallback
MIN_SENTENCE_WORDS = 2       # don't emit 1-word "sentences"


def load_whisper_words(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for w in data.get("word_segments", []):
        s, e, t = w.get("start"), w.get("end"), w.get("word")
        if s is None or e is None or not t:
            continue
        out.append({"start": float(s), "end": float(e), "word": str(t).strip()})
    return out


def segment_into_sentences(words: list[dict]) -> list[list[dict]]:
    """Group words into sentences by punctuation + pause boundaries."""
    sentences: list[list[dict]] = []
    cur: list[dict] = []
    prev_end: float | None = None

    for w in words:
        # Long pause closes the previous sentence if it has content
        if prev_end is not None and cur and (w["start"] - prev_end) >= MIN_SENTENCE_GAP_SEC:
            if len(cur) >= MIN_SENTENCE_WORDS:
                sentences.append(cur)
                cur = []
            # if cur was too short, fold the pause-flushed words into the next sentence
        cur.append(w)
        prev_end = w["end"]
        # Punctuation closes the sentence
        if PUNCT_END.search(w["word"]):
            if len(cur) >= MIN_SENTENCE_WORDS:
                sentences.append(cur)
                cur = []

    if cur and len(cur) >= MIN_SENTENCE_WORDS:
        sentences.append(cur)

    return sentences


def process_one(did: str) -> dict | None:
    print(f"\n=== {did} ===")
    path = WHISPER_DIR / f"{did}.json"
    if not path.exists():
        print(f"  no Whisper alignment at {path}")
        return None

    words = load_whisper_words(path)
    if not words:
        print(f"  empty Whisper alignment")
        return None

    sentences = segment_into_sentences(words)
    print(f"  {len(words):,} words → {len(sentences):,} sentences "
          f"(audio span 0 → {words[-1]['end']:.1f}s)")

    word_rows = []
    sent_rows = []
    for s_idx, sent in enumerate(sentences):
        s_start = sent[0]["start"]
        s_end = sent[-1]["end"]
        text = " ".join(w["word"] for w in sent).strip()
        # Tidy up multiple spaces / leading/trailing punctuation artifacts
        text = re.sub(r"\s+", " ", text)

        sent_id = f"{did}_w{s_idx + 1:04d}"
        sent_rows.append({
            "debate_audio_id": did,
            "sentence_id": sent_id,
            "sentence_index": s_idx,
            "speaker_raw": "UNKNOWN",         # to be filled by diarization join later
            "n_words": len(sent),
            "start_sec": round(s_start, 3),
            "end_sec": round(s_end, 3),
            "duration_sec": round(s_end - s_start, 3),
            "sentence_text": text,
            "source": "whisper_canonical",
        })

        for w_i, w in enumerate(sent):
            word_rows.append({
                "debate_audio_id": did,
                "sentence_id": sent_id,
                "word_index_in_sentence": w_i,
                "word": w["word"],
                "start_sec": round(w["start"], 3),
                "end_sec": round(w["end"], 3),
                "source": "whisper_canonical",
            })

    return {
        "word_rows": word_rows,
        "sentence_rows": sent_rows,
        "n_words": len(words),
        "n_sentences": len(sentences),
        "audio_sec": words[-1]["end"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="*", default=DEFAULT_TARGETS,
                    help="Debate IDs (default: 1034 1036)")
    args = ap.parse_args()

    all_words = []
    all_sents = []
    summary = []

    for did in args.debates:
        res = process_one(did)
        if not res:
            summary.append({"debate_audio_id": did, "status": "skipped"})
            continue
        all_words.extend(res["word_rows"])
        all_sents.extend(res["sentence_rows"])
        summary.append({
            "debate_audio_id": did,
            "status": "ok",
            "n_words": res["n_words"],
            "n_sentences": res["n_sentences"],
            "audio_min": round(res["audio_sec"] / 60, 1),
        })

    if all_sents:
        pd.DataFrame(all_words).to_csv(OUT_WORDS_CSV, index=False)
        pd.DataFrame(all_sents).to_csv(OUT_SENTENCES_CSV, index=False)
        print(f"\n[write] {OUT_WORDS_CSV}     ({len(all_words):,} words)")
        print(f"[write] {OUT_SENTENCES_CSV} ({len(all_sents):,} sentences)")
    pd.DataFrame(summary).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"[write] {OUT_SUMMARY_CSV}")

    print("\n=== Per-debate ===")
    for r in summary:
        if r["status"] == "ok":
            print(f"  {r['debate_audio_id']}: {r['n_words']:,} words → {r['n_sentences']:,} sentences "
                  f"({r['audio_min']} min audio)")
        else:
            print(f"  {r['debate_audio_id']}: {r['status']}")


if __name__ == "__main__":
    main()
