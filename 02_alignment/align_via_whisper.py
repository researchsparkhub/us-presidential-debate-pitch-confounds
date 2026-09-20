#!/usr/bin/env python3
"""
Text-based alignment for debates that failed MFA.

Aligns Whisper's verbatim ASR (which already has accurate word-level
timestamps) to the OFFICIAL transcript at the text level. Whisper carries
the timing; the official transcript carries the canonical wording,
punctuation, casing, and turn/speaker structure.

For each Whisper word ↔ original-transcript word that match:
    the original word inherits Whisper's (start_sec, end_sec).
For unmatched stretches in the original transcript (paraphrased,
edited-in, or missed by Whisper), we LINEARLY INTERPOLATE timing
between the nearest matched flanks.

Outputs are emitted in the same schema as the MFA outputs so downstream
joins are identical for Tier A (MFA) and Tier B (this script).

Run:
    python 02_alignment/align_via_whisper.py            # all Tier B debates
    python 02_alignment/align_via_whisper.py --debates 1001 1034
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
ALIGN_DIR = PROJECT / "02_alignment"
WHISPER_DIR = PROJECT / "data" / "whisper_aligned"
TRANSCRIPT_DIR = PROJECT / "data" / "transcripts"

# Default Tier B debates (those where MFA failed across all 3 retries)
DEFAULT_TARGETS = ["1001", "1007", "1008", "1019", "1034", "1036"]

# Output files (paths chosen so they DON'T overwrite the MFA outputs;
# parse_mfa_output.py and downstream stage-3 join can read both)
OUT_WORDS_CSV = ALIGN_DIR / "text_aligned_words.csv"
OUT_TURNS_CSV = ALIGN_DIR / "text_aligned_turns.csv"
OUT_SENTENCES_CSV = ALIGN_DIR / "text_aligned_sentences.csv"
OUT_SUMMARY_CSV = ALIGN_DIR / "text_aligned_summary.csv"

# Transcript parsing: same shape as prepare_mfa_corpus.py expects
SPEAKER_PREFIX = re.compile(r"^([A-Z][A-Z.'\-]+(?:\s[A-Z][A-Z.'\-]+)?)\s*:\s*(.*)$")
ANNOTATION_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
SENTENCE_END = re.compile(r"[.!?]+")


def find_transcript_files(did: str) -> list[Path]:
    direct = TRANSCRIPT_DIR / f"{did}.txt"
    if direct.exists():
        return [direct]
    return sorted(TRANSCRIPT_DIR.glob(f"{did}_*.txt"))


def parse_turns_preserving(text: str) -> list[dict]:
    """Parse transcript into turns. KEEPS original casing + punctuation in
    `turn_text`. Skips header (date/PARTICIPANTS/MODERATORS) and bracketed
    stage directions, but doesn't touch contractions/quotes/etc.
    """
    lines = text.splitlines()
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
        # Drop only bracketed/parenthetical stage directions; keep punctuation
        body = re.sub(r"\([^)]*\)", " ", body)
        body = re.sub(r"\[[^\]]*\]", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
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


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into word tokens, keeping each token's char-offset span
    so we can later locate sentence boundaries in the original string."""
    out = []
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def normalize_for_match(token: str) -> str:
    """Aggressive normalization used ONLY for sequence alignment.
    Lowercase, expand a few common contractions, drop non-alphanumerics
    (so "I'll." matches "i'll" and "1988," matches "1988")."""
    t = token.lower()
    # Common contractions split → match Whisper which writes them out
    t = t.replace("'re", " re").replace("'ve", " ve").replace("'ll", " ll")
    t = t.replace("'d", " d").replace("'s", " s").replace("n't", " nt")
    # Drop non-alphanumerics (keep digit runs)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_whisper_words(path: Path) -> list[tuple[float, float, str]]:
    """Whisper word-level alignment → list of (start, end, raw_word)."""
    data = json.loads(path.read_text())
    out = []
    for w in data.get("word_segments", []):
        s, e, t = w.get("start"), w.get("end"), w.get("word")
        if s is None or e is None or not t:
            continue
        out.append((float(s), float(e), str(t)))
    return out


def expand_to_normalized_atoms(tokens: list[str]) -> tuple[list[str], list[int]]:
    """Apply normalize_for_match to each token, then split on spaces so each
    contraction expansion becomes its own atom. Return (atoms, owner_idx)
    where owner_idx[k] is the index of the original token that produced
    atoms[k]. This lets us run a Needleman-Wunsch on flat atom lists while
    remembering which original token each atom came from."""
    atoms: list[str] = []
    owners: list[int] = []
    for i, tok in enumerate(tokens):
        for atom in normalize_for_match(tok).split():
            atoms.append(atom)
            owners.append(i)
    return atoms, owners


def align_atom_sequences(a_atoms: list[str], b_atoms: list[str]) -> list[tuple[int, int]]:
    """Run difflib's SequenceMatcher over the two atom lists and return all
    (i_in_a, i_in_b) matched pairs. Uses Python stdlib — no extra deps.
    """
    matcher = SequenceMatcher(a=a_atoms, b=b_atoms, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs.append((block.a + k, block.b + k))
    return pairs


def split_into_sentences(text: str) -> list[tuple[int, int]]:
    """Return (char_start, char_end) spans for each sentence in `text`.

    Splits at . / ! / ? followed by whitespace + capital OR end of string.
    Preserves the sentence-ending punctuation inside the span.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        # find next sentence terminator
        m = re.search(r"[.!?]+", text[i:])
        if not m:
            if i < n:
                spans.append((i, n))
            break
        end = i + m.end()  # include the punctuation
        # If there's whitespace after, that's the boundary; otherwise extend
        spans.append((i, end))
        # advance past whitespace
        j = end
        while j < n and text[j].isspace():
            j += 1
        i = j
    return spans


def interpolate_timing(n: int, anchors: dict[int, tuple[float, float]]) -> list[tuple[float, float, str]]:
    """Given an array length n and a dict {idx: (start, end)} of known
    anchored indices (from sequence matching), produce a timing for every
    index by linear interpolation between flanks.

    Returns list of (start, end, source_tag) where source_tag is:
        "matched"       — directly from a sequence match
        "interpolated"  — linearly interpolated between matched flanks
        "extrapolated"  — only one flank known; copies that flank's time
        "unknown"       — no anchors at all (very rare)
    """
    out: list[tuple[float, float, str]] = [(-1.0, -1.0, "unknown")] * n
    if not anchors:
        return out

    anchor_keys = sorted(anchors.keys())
    # Direct anchors
    for k in anchor_keys:
        s, e = anchors[k]
        out[k] = (s, e, "matched")

    # For each non-anchored index, find left + right anchor and interpolate
    for i in range(n):
        if out[i][2] == "matched":
            continue
        # left neighbor
        left_idx = None
        for k in reversed(anchor_keys):
            if k < i:
                left_idx = k
                break
        # right neighbor
        right_idx = None
        for k in anchor_keys:
            if k > i:
                right_idx = k
                break
        if left_idx is not None and right_idx is not None:
            ls = anchors[left_idx][1]      # left's end
            re_ = anchors[right_idx][0]     # right's start
            frac = (i - left_idx) / (right_idx - left_idx)
            t = ls + frac * (re_ - ls)
            out[i] = (t, t, "interpolated")
        elif left_idx is not None:
            ls = anchors[left_idx][1]
            out[i] = (ls, ls, "extrapolated")
        elif right_idx is not None:
            rs = anchors[right_idx][0]
            out[i] = (rs, rs, "extrapolated")
    return out


def process_one(did: str) -> dict | None:
    print(f"\n=== {did} ===")
    transcript_paths = find_transcript_files(did)
    whisper_path = WHISPER_DIR / f"{did}.json"

    if not transcript_paths:
        print(f"  no transcript file at {TRANSCRIPT_DIR}/{did}.txt"); return None
    if not whisper_path.exists():
        print(f"  no Whisper alignment at {whisper_path}"); return None

    # 1. Parse original transcript, keep casing + punctuation
    full_text = "\n".join(p.read_text(encoding="utf-8") for p in transcript_paths)
    turns = parse_turns_preserving(full_text)
    if not turns:
        print(f"  no turns parsed"); return None

    # 2. Build a flat transcript: tokens with char spans into per-turn text,
    #    PLUS a global atom list (normalized) for matching with Whisper.
    all_orig_atoms: list[str] = []
    all_orig_owners: list[tuple[int, int]] = []  # (turn_index, token_index_in_turn)
    turn_tokens: list[list[tuple[str, int, int]]] = []  # per turn

    for t_idx, turn in enumerate(turns):
        toks = tokenize_with_offsets(turn["text"])
        turn_tokens.append(toks)
        for tok_idx, (tok, _s, _e) in enumerate(toks):
            for atom in normalize_for_match(tok).split():
                all_orig_atoms.append(atom)
                all_orig_owners.append((t_idx, tok_idx))

    print(f"  transcript: {len(turns)} turns, {sum(len(x) for x in turn_tokens)} tokens, "
          f"{len(all_orig_atoms)} normalized atoms")

    # 3. Whisper words → atoms (normalized) carrying their (start, end)
    whisper = load_whisper_words(whisper_path)
    w_atoms: list[str] = []
    w_atom_times: list[tuple[float, float]] = []
    for (ws, we, wt) in whisper:
        atoms = normalize_for_match(wt).split()
        if not atoms:
            continue
        # Distribute the whisper word's duration across its atoms (rare; most words → 1 atom)
        n = len(atoms)
        span = (we - ws) / n
        for k, a in enumerate(atoms):
            w_atoms.append(a)
            w_atom_times.append((ws + k * span, ws + (k + 1) * span))

    print(f"  whisper:    {len(whisper)} words, {len(w_atoms)} normalized atoms")

    # 4. Sequence-align Whisper atoms ↔ transcript atoms
    pairs = align_atom_sequences(w_atoms, all_orig_atoms)
    print(f"  matched {len(pairs)} atoms ({len(pairs)/max(1,len(all_orig_atoms))*100:.1f}% of transcript)")

    # 5. Build anchors: transcript_atom_idx → (start, end) from Whisper
    anchors = {orig_i: w_atom_times[w_i] for (w_i, orig_i) in pairs}

    # 6. Interpolate timing for every transcript atom
    timings = interpolate_timing(len(all_orig_atoms), anchors)

    # 7. Aggregate atom-level timings back to TOKEN level (max span per owner token)
    token_timing: dict[tuple[int, int], dict] = {}
    for atom_i, (s, e, src) in enumerate(timings):
        if s < 0:
            continue
        owner = all_orig_owners[atom_i]
        if owner not in token_timing:
            token_timing[owner] = {"start": s, "end": e, "sources": {src}}
        else:
            token_timing[owner]["start"] = min(token_timing[owner]["start"], s)
            token_timing[owner]["end"] = max(token_timing[owner]["end"], e)
            token_timing[owner]["sources"].add(src)

    # 8. Emit per-word, per-turn, per-sentence rows
    word_rows = []
    turn_rows = []
    sentence_rows = []

    for t_idx, turn in enumerate(turns):
        toks = turn_tokens[t_idx]
        turn_id = f"{did}_t{t_idx + 1:04d}"
        speaker_raw = turn["speaker_raw"]

        # Per-word
        tok_word_starts: list[float] = []
        tok_word_ends: list[float] = []
        matched_count = 0
        for tok_i, (tok, char_s, char_e) in enumerate(toks):
            info = token_timing.get((t_idx, tok_i))
            if info:
                src_set = info["sources"]
                src = "matched" if "matched" in src_set else ("interpolated" if "interpolated" in src_set else "extrapolated")
                tok_word_starts.append(info["start"])
                tok_word_ends.append(info["end"])
                if src == "matched":
                    matched_count += 1
                word_rows.append({
                    "debate_audio_id": did,
                    "turn_id": turn_id,
                    "speaker_raw": speaker_raw,
                    "word": tok,
                    "word_index_in_turn": tok_i,
                    "start_sec": round(info["start"], 3),
                    "end_sec": round(info["end"], 3),
                    "alignment_source": src,
                })
            else:
                word_rows.append({
                    "debate_audio_id": did,
                    "turn_id": turn_id,
                    "speaker_raw": speaker_raw,
                    "word": tok,
                    "word_index_in_turn": tok_i,
                    "start_sec": None,
                    "end_sec": None,
                    "alignment_source": "unknown",
                })

        # Per-turn: span = first known start → last known end
        if tok_word_starts and tok_word_ends:
            t_start = min(tok_word_starts)
            t_end = max(tok_word_ends)
        else:
            t_start, t_end = None, None
        turn_rows.append({
            "debate_audio_id": did,
            "turn_id": turn_id,
            "speaker_raw": speaker_raw,
            "n_words": len(toks),
            "n_words_matched": matched_count,
            "start_sec": round(t_start, 3) if t_start is not None else None,
            "end_sec": round(t_end, 3) if t_end is not None else None,
            "duration_sec": round((t_end or 0) - (t_start or 0), 3) if t_start is not None else None,
            "text": turn["text"],
        })

        # Per-sentence: split turn text by sentence punctuation, match tokens by char-offset
        for s_idx, (s_start_char, s_end_char) in enumerate(split_into_sentences(turn["text"])):
            s_text = turn["text"][s_start_char:s_end_char].strip()
            if not s_text:
                continue
            # Tokens whose char-spans fall inside this sentence
            in_sent_starts = []
            in_sent_ends = []
            in_sent_matched = 0
            in_sent_n = 0
            for tok_i, (tok, char_s, char_e) in enumerate(toks):
                if char_s >= s_start_char and char_e <= s_end_char:
                    in_sent_n += 1
                    info = token_timing.get((t_idx, tok_i))
                    if info:
                        in_sent_starts.append(info["start"])
                        in_sent_ends.append(info["end"])
                        if "matched" in info["sources"]:
                            in_sent_matched += 1
            if in_sent_n == 0:
                continue
            if in_sent_starts and in_sent_ends:
                s_start = min(in_sent_starts)
                s_end = max(in_sent_ends)
            else:
                s_start, s_end = None, None
            sentence_rows.append({
                "debate_audio_id": did,
                "turn_id": turn_id,
                "sentence_id": f"{turn_id}_s{s_idx + 1:03d}",
                "sentence_index_in_turn": s_idx,
                "speaker_raw": speaker_raw,
                "n_words": in_sent_n,
                "n_words_matched": in_sent_matched,
                "start_sec": round(s_start, 3) if s_start is not None else None,
                "end_sec": round(s_end, 3) if s_end is not None else None,
                "duration_sec": round((s_end or 0) - (s_start or 0), 3) if s_start is not None else None,
                "sentence_text": s_text,
            })

    matched_total = sum(1 for r in word_rows if r["alignment_source"] == "matched")
    interp_total = sum(1 for r in word_rows if r["alignment_source"] == "interpolated")
    print(f"  {len(word_rows)} words, {matched_total} matched, {interp_total} interpolated, "
          f"{len(turn_rows)} turns, {len(sentence_rows)} sentences")

    return {
        "word_rows": word_rows,
        "turn_rows": turn_rows,
        "sentence_rows": sentence_rows,
        "n_orig_atoms": len(all_orig_atoms),
        "n_matched_atoms": len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="*", default=DEFAULT_TARGETS,
                    help="Debate IDs to process (default: 6 Tier B stragglers)")
    args = ap.parse_args()

    all_words = []
    all_turns = []
    all_sentences = []
    summary = []

    for did in args.debates:
        res = process_one(did)
        if not res:
            summary.append({"debate_audio_id": did, "status": "skipped"})
            continue
        all_words.extend(res["word_rows"])
        all_turns.extend(res["turn_rows"])
        all_sentences.extend(res["sentence_rows"])
        n_words = len(res["word_rows"])
        n_matched = sum(1 for r in res["word_rows"] if r["alignment_source"] == "matched")
        summary.append({
            "debate_audio_id": did,
            "status": "ok",
            "n_words": n_words,
            "n_words_matched": n_matched,
            "match_rate_pct": round(n_matched / max(1, n_words) * 100, 1),
            "n_turns": len(res["turn_rows"]),
            "n_sentences": len(res["sentence_rows"]),
        })

    if all_words:
        pd.DataFrame(all_words).to_csv(OUT_WORDS_CSV, index=False)
        pd.DataFrame(all_turns).to_csv(OUT_TURNS_CSV, index=False)
        pd.DataFrame(all_sentences).to_csv(OUT_SENTENCES_CSV, index=False)
        print(f"\n[write] {OUT_WORDS_CSV}     ({len(all_words)} words)")
        print(f"[write] {OUT_TURNS_CSV}     ({len(all_turns)} turns)")
        print(f"[write] {OUT_SENTENCES_CSV} ({len(all_sentences)} sentences)")

    pd.DataFrame(summary).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"[write] {OUT_SUMMARY_CSV}")
    print("\n=== Per-debate match rates ===")
    for r in summary:
        if r["status"] == "ok":
            print(f"  {r['debate_audio_id']}: {r['n_words_matched']}/{r['n_words']} "
                  f"({r['match_rate_pct']}%) — {r['n_sentences']} sentences")
        else:
            print(f"  {r['debate_audio_id']}: {r['status']}")


if __name__ == "__main__":
    main()
