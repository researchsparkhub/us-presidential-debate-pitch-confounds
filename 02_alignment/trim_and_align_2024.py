#!/usr/bin/env python3
"""
Trim the 2024 debate broadcasts (1034, 1036) to just the debate window, then
realign with MFA.

The 2024 audio files contain hours of pre-show + post-show studio coverage
NOT in the official transcript. MFA's beam search blows up trying to find
the transcript words in non-debate audio. The fix is to slice the WAV to
just the debate portion before alignment.

Strategy:
  1. Read aligned/<id>.json (existing WhisperX word-level alignment of the
     ENTIRE broadcast, including pre/post show).
  2. Read mfa_corpus/<id>/<id>.lab (the actual debate transcript).
  3. Locate where the .lab's FIRST 30 words sit in the Whisper word stream
     → trim_start_sec.
  4. Locate where the .lab's LAST 30 words sit in Whisper → trim_end_sec.
  5. ffmpeg-extract [trim_start, trim_end] into <id>_trimmed.wav.
  6. Build a trimmed corpus and run MFA on it.
  7. Read the resulting TextGrid (in trimmed-audio coords), shift every
     interval by +trim_start_sec, and write it to mfa_output/<id>/<id>.TextGrid
     (so it's in ORIGINAL-audio coords, like all other TextGrids).
  8. Save mfa_corpus/<id>/<id>_trim_offset.json so downstream knows.

Run:
    /Users/lavanya/debate_audio_project/venv_review/bin/python trim_and_align_2024.py

Prerequisites:
  - ffmpeg installed (brew install ffmpeg)
  - conda env 'aligner' with MFA (script invokes mfa via conda run)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rapidfuzz import fuzz

PROJECT = Path("/Users/lavanya/debate_analysis")
ALIGN_DIR = PROJECT / "02_alignment"
WHISPER_DIR = PROJECT / "data" / "whisper_aligned"
CORPUS_DIR = ALIGN_DIR / "mfa_corpus"
OUTPUT_DIR = ALIGN_DIR / "mfa_output"

TARGETS = ["1034", "1036"]
ANCHOR_WORDS = 30          # words to match at start and end
WINDOW_BACKOFF_SEC = 5.0   # add a small pad on each side of the detected window
BEAM = 400
RETRY_BEAM = 1000
TIMEOUT_MIN = 90


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def load_whisper_words(path: Path) -> list[tuple[float, float, str]]:
    """Return [(start, end, word_lowercase)] for all aligned Whisper words."""
    data = json.loads(path.read_text())
    out = []
    for w in data.get("word_segments", []):
        s, e, t = w.get("start"), w.get("end"), w.get("word")
        if s is None or e is None or not t:
            continue
        out.append((float(s), float(e), normalize(t)))
    return out


def locate_window(words: list[tuple[float, float, str]], probe: str,
                  search_from: float = 0.0, search_to: float | None = None) -> tuple[int, float] | None:
    """Find where `probe` (a string of the first/last 30 transcript words) best
    aligns within the Whisper word stream.

    Returns (whisper_word_index, fuzzy_score) for the BEST starting index. We
    slide a window of ~len(probe.split()) Whisper words, fuzzy-match its
    concatenated text against probe, and pick the highest score.
    """
    n_probe_words = len(probe.split())
    n = len(words)
    if search_to is None:
        search_to = words[-1][0]
    best_score = -1.0
    best_idx = -1
    for i in range(n - n_probe_words):
        if words[i][0] < search_from:
            continue
        if words[i][0] > search_to:
            break
        window_text = " ".join(w[2] for w in words[i:i + n_probe_words])
        score = fuzz.partial_ratio(probe, window_text)
        if score > best_score:
            best_score = score
            best_idx = i
            if score >= 95:
                break
    if best_idx < 0:
        return None
    return (best_idx, best_score)


def trim_with_ffmpeg(src: Path, dst: Path, start: float, end: float):
    """Lossless slice via stream copy when possible."""
    if dst.exists():
        dst.unlink()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(src),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def parse_and_shift_textgrid(in_path: Path, out_path: Path, shift: float):
    """Read MFA TextGrid (in trimmed-audio time), add `shift` to every xmin/xmax,
    write to out_path. Also adjusts the file-level xmin/xmax."""
    text = in_path.read_text(encoding="utf-8")

    def shift_value(m):
        return f"{m.group(1)}{float(m.group(2)) + shift:.6f}"

    pattern = re.compile(r"(xmin\s*=\s*|xmax\s*=\s*)([\d.]+)")
    shifted = pattern.sub(shift_value, text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(shifted, encoding="utf-8")


def run_mfa_on_trimmed(corpus_dir: Path, output_dir: Path, log_path: Path) -> int:
    """Invoke MFA via conda run. Returns rc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "conda", "run", "-n", "aligner", "--no-capture-output",
        "mfa", "align",
        str(corpus_dir),
        "english_us_arpa", "english_us_arpa",
        str(output_dir),
        "--clean", "--num_jobs", "1", "--single_speaker",
        "--beam", str(BEAM), "--retry_beam", str(RETRY_BEAM),
    ]
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            return subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                timeout=TIMEOUT_MIN * 60,
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {TIMEOUT_MIN} minutes\n")
            return 124


def process_one(did: str):
    print(f"\n=== {did} ===")
    log_path = ALIGN_DIR / "retry_logs" / f"trim_v2_{did}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write(f"Trim+align for {did}\n")

    whisper_path = WHISPER_DIR / f"{did}.json"
    lab_path = CORPUS_DIR / did / f"{did}.lab"
    full_wav = CORPUS_DIR / did / f"{did}.wav"  # symlink to data/audio/<id>.wav

    if not (whisper_path.exists() and lab_path.exists() and full_wav.exists()):
        msg = f"  missing source files: whisper={whisper_path.exists()} lab={lab_path.exists()} wav={full_wav.exists()}"
        print(msg); log.write(msg + "\n"); return None

    # 1. Get all Whisper words (entire broadcast)
    whisper_words = load_whisper_words(whisper_path)
    if not whisper_words:
        print("  Whisper alignment empty"); log.write("  Whisper alignment empty\n"); return None
    broadcast_end = whisper_words[-1][1]
    print(f"  Whisper covers 0.0–{broadcast_end:.1f}s, {len(whisper_words)} words")

    # 2. First/last N transcript words → probes
    lab_words = normalize(lab_path.read_text()).split()
    if len(lab_words) < 2 * ANCHOR_WORDS:
        print("  transcript too short to anchor"); return None
    head_probe = " ".join(lab_words[:ANCHOR_WORDS])
    tail_probe = " ".join(lab_words[-ANCHOR_WORDS:])

    # 3. Locate head in first 60% of broadcast, tail in last 60%
    head_match = locate_window(whisper_words, head_probe, search_from=0.0, search_to=broadcast_end * 0.6)
    if not head_match:
        print("  could not locate transcript head in Whisper"); return None
    head_idx, head_score = head_match
    trim_start = max(0.0, whisper_words[head_idx][0] - WINDOW_BACKOFF_SEC)
    print(f"  HEAD anchored at Whisper word #{head_idx} ({whisper_words[head_idx][0]:.1f}s, score={head_score})")

    tail_match = locate_window(whisper_words, tail_probe, search_from=broadcast_end * 0.4, search_to=broadcast_end)
    if not tail_match:
        print("  could not locate transcript tail in Whisper"); return None
    tail_idx, tail_score = tail_match
    tail_end_word = min(tail_idx + ANCHOR_WORDS - 1, len(whisper_words) - 1)
    trim_end = min(broadcast_end, whisper_words[tail_end_word][1] + WINDOW_BACKOFF_SEC)
    print(f"  TAIL anchored at Whisper word #{tail_idx} ({whisper_words[tail_idx][0]:.1f}s, score={tail_score})")
    print(f"  Trim window: {trim_start:.1f}s → {trim_end:.1f}s ({(trim_end-trim_start)/60:.1f} min)")

    # Sanity: trimmed window should be at least 60 min
    if trim_end - trim_start < 60 * 60:
        print(f"  WARNING: trimmed window only {(trim_end-trim_start)/60:.1f} min — may be wrong")

    # 4. ffmpeg trim
    trimmed_wav = CORPUS_DIR / did / f"{did}_trimmed.wav"
    trim_with_ffmpeg(full_wav.resolve(), trimmed_wav, trim_start, trim_end)
    print(f"  wrote trimmed audio → {trimmed_wav.name}")

    # Save offset for downstream
    (CORPUS_DIR / did / f"{did}_trim_offset.json").write_text(
        json.dumps({"trim_start": trim_start, "trim_end": trim_end,
                    "head_score": head_score, "tail_score": tail_score}, indent=2)
    )

    # 5. Build a trimmed corpus
    trim_corpus = ALIGN_DIR / "retry_workdir_v2" / did / f"corpus_trim_{did}"
    if trim_corpus.exists():
        shutil.rmtree(trim_corpus)
    (trim_corpus / did).mkdir(parents=True)
    os.symlink(trimmed_wav.resolve(), trim_corpus / did / f"{did}.wav")
    shutil.copy2(lab_path, trim_corpus / did / f"{did}.lab")

    # 6. Run MFA on trimmed corpus, write TextGrid to a temp output dir
    trim_output = ALIGN_DIR / "retry_workdir_v2" / did / "mfa_output_trim"
    rc = run_mfa_on_trimmed(trim_corpus, trim_output, log_path)
    if rc != 0:
        print(f"  MFA failed rc={rc}"); return None

    src_tg = trim_output / did / f"{did}.TextGrid"
    if not src_tg.exists():
        print(f"  MFA finished but no TextGrid at {src_tg}"); return None

    # 7. Shift TextGrid timestamps by +trim_start so they refer to ORIGINAL audio time
    dst_tg = OUTPUT_DIR / did / f"{did}.TextGrid"
    parse_and_shift_textgrid(src_tg, dst_tg, shift=trim_start)
    print(f"  ✓ wrote shifted TextGrid → {dst_tg}")
    return dst_tg


def main():
    for did in TARGETS:
        try:
            process_one(did)
        except Exception as e:
            print(f"  ERROR processing {did}: {e}")

    print("\n=== Summary ===")
    for did in TARGETS:
        tg = OUTPUT_DIR / did / f"{did}.TextGrid"
        print(f"  {'✓' if tg.exists() else '✗'} {did}")


if __name__ == "__main__":
    main()
