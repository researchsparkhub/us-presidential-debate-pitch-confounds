#!/usr/bin/env python3
"""
Stage 3 — produce the canonical sentence-level table that drives stages 4-6.

Inputs:
    02_alignment/text_aligned_sentences.csv       (sentence-level, original transcript text)
    02_alignment/whisper_canonical_sentences.csv  (Whisper-canonical for 1034/1036)
    02_alignment/mfa_corpus/<id>/<id>_trim_offset.json (2024 debate-window bounds)
    data/metadata/debates.csv                     (winner/loser per candidate)
    data/metadata/cluster_to_speaker_2024.csv    (pyannote cluster → candidate, for 1034/1036)
    data/diarization/segments_raw.csv             (pyannote segments)

Output:
    outputs/master_sentences.csv  — one row per sentence, schema documented below.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
ALIGN_DIR = PROJECT / "02_alignment"
META_CSV = PROJECT / "data" / "metadata" / "debates.csv"
CLUSTER_MAP_CSV = PROJECT / "data" / "metadata" / "cluster_to_speaker_2024.csv"
PYANNOTE_SEGS_CSV = PROJECT / "data" / "diarization" / "segments_raw.csv"
OUT_CSV = PROJECT / "outputs" / "master_sentences.csv"

TEXT_ALIGNED = ALIGN_DIR / "text_aligned_sentences.csv"
WHISPER_CANONICAL = ALIGN_DIR / "whisper_canonical_sentences.csv"

# Debates where the official transcript can't be aligned reliably → use
# Whisper-canonical text + pyannote diarization for speaker labels.
USE_WHISPER_CANONICAL = {"1034", "1036"}

# Known moderators + panelists across the 1988-2024 corpus.
# We tag by exact match on UPPER_LAST_NAME first, otherwise fall back to the
# raw label itself.
MODERATORS = {
    # Generic catch-alls
    "MODERATOR", "MODERATORS",
    # 1988-2024 moderators / panelists
    "LEHRER", "JENNINGS", "SHAW", "MASHEK", "GROER", "COMPTON", "WARNER",
    "MITCHELL", "VANOCUR", "RUMSFELD", "SIMPSON", "ROSE", "BERKLEY", "MOYERS",
    "BURNS", "FRANCIS", "WOODRUFF", "BROKAW", "GIBSON", "STAHL", "GREGORY",
    "IFILL", "BLITZER", "SCHIEFFER", "DICKERSON", "RADDATZ", "WALLACE",
    "HARWOOD", "QUIJANO", "QUICK", "STEPHANOPOULOS", "MUIR", "DAVIS", "TAPPER",
    "BASH", "BRENNAN", "ODONNELL", "O'DONNELL", "WELKER", "PAGE", "HOLT",
    "QUINTANILLA", "COOPER", "CRAMER", "MCMAHON", "OBRIEN", "KING",
    "FACENDA", "FLECK", "CROWLEY",
    # 1992 P3 (Larry King Live) panelists
    "ROOK", "THOMAS", "GIBBONS",
}

# Lines that aren't real turns — header artifacts MFA / text-alignment parsers
# mistakenly treat as speakers because they're ALLCAPS+colon.
HEADER_ARTIFACTS = {"SPEAKERS", "PARTICIPANTS", "MODERATORS PANEL"}

# Audience-label patterns. Match against the FULL raw label (case insensitive).
AUDIENCE_PATTERNS = [
    re.compile(r"^(MS|MR|DR|MRS)\.?\s", re.IGNORECASE),
    re.compile(r"^(QUESTION|AUDIENCE QUESTION|UNIDENTIFIED (MALE|FEMALE)|AUDIENCE MEMBER)", re.IGNORECASE),
]

# Manual aliases for candidate-name typos / abbreviations present in transcripts.
# Maps raw transcript label → canonical full name (must exist in debates.csv).
CANDIDATE_ALIASES = {
    "ROMNEHY": "Mitt Romney",
    "OBAM":    "Barack Obama",
}


def is_audience(speaker_raw: str) -> bool:
    raw = speaker_raw.strip()
    return any(p.match(raw) for p in AUDIENCE_PATTERNS)


def is_header_artifact(speaker_raw: str) -> bool:
    return speaker_raw.strip().upper() in HEADER_ARTIFACTS


def build_candidate_lookups(debates_df: pd.DataFrame) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Return:
        - by_debate_lastname: {debate_audio_id: {UPPER_LAST_NAME: {speaker,party,result}}}
        - by_full_name:       {speaker_full_name: {party,result_default}}
    """
    by_debate: dict[str, dict[str, dict]] = {}
    by_full: dict[str, dict] = {}
    for _, r in debates_df.iterrows():
        did = str(r["debate_audio_id"])
        full = str(r["speaker"]).strip()
        key = full.split()[-1].upper()
        by_debate.setdefault(did, {})[key] = {
            "speaker": full,
            "party":   str(r["party"]),
            "result":  str(r["result"]),
        }
        by_full[full] = {"party": str(r["party"]), "result": str(r["result"])}
    return by_debate, by_full


def classify_speaker(
    speaker_raw: str,
    did: str,
    cand_per_debate: dict[str, dict[str, dict]],
    cand_full: dict[str, dict],
    town_hall_audience_lookup: dict[str, set[str]],
) -> dict:
    """Return {speaker_role, speaker, party, result} for any speaker_raw.
    Returns None to signal: skip this row entirely (header artifact)."""
    raw = (speaker_raw or "").strip()

    if not raw or is_header_artifact(raw):
        return None

    raw_upper = raw.upper()
    last_name = raw_upper.split()[-1] if raw_upper else ""

    # 1. Audience pattern → tag as audience
    if is_audience(raw):
        return {"speaker_role": "audience", "speaker": raw.title(),
                "party": "—", "result": "audience"}

    # 2. Town-hall single-turn speakers we've identified as audience
    if last_name in town_hall_audience_lookup.get(did, set()):
        return {"speaker_role": "audience", "speaker": raw.title(),
                "party": "—", "result": "audience"}

    # 3. Manual candidate alias?
    if raw_upper in CANDIDATE_ALIASES:
        full = CANDIDATE_ALIASES[raw_upper]
        info = cand_full.get(full)
        if info:
            return {"speaker_role": "candidate", "speaker": full,
                    "party": info["party"], "result": info["result"]}

    # 4. Candidate from the debate's own roster (by last name)
    cand = cand_per_debate.get(did, {}).get(last_name)
    if cand:
        return {"speaker_role": "candidate", "speaker": cand["speaker"],
                "party": cand["party"], "result": cand["result"]}

    # 5. Known moderators (raw or last-name match)
    if raw_upper in MODERATORS or last_name in MODERATORS:
        return {"speaker_role": "moderator", "speaker": raw.title(),
                "party": "—", "result": "moderator"}

    # 6. Fallback
    return {"speaker_role": "unknown", "speaker": raw,
            "party": "—", "result": "unknown"}


def infer_town_hall_audience(text_df: pd.DataFrame, cand_per_debate: dict[str, dict[str, dict]]) -> dict[str, set[str]]:
    """In town-hall debates (lots of speakers), any non-candidate non-moderator
    speaker with only 1–3 turns is treated as audience. Returns
    {debate_audio_id: {UPPER_LAST_NAMES_of_audience}}.
    """
    out: dict[str, set[str]] = defaultdict(set)
    counts = text_df.groupby(["debate_audio_id", "speaker_raw"])["sentence_id"].count().reset_index(name="n")
    spk_per_debate = counts.groupby("debate_audio_id")["speaker_raw"].nunique()

    # A debate counts as "town-hall-like" if it has ≥7 distinct speakers
    town_hall_debates = set(spk_per_debate[spk_per_debate >= 7].index.astype(str))

    for _, r in counts.iterrows():
        did = str(r["debate_audio_id"])
        if did not in town_hall_debates:
            continue
        raw = str(r["speaker_raw"]).strip()
        if not raw or is_header_artifact(raw) or is_audience(raw):
            continue
        last_name = raw.upper().split()[-1] if raw else ""
        # Skip if it's a known candidate or moderator
        if last_name in cand_per_debate.get(did, {}):
            continue
        if raw.upper() in MODERATORS or last_name in MODERATORS:
            continue
        # Audience heuristic: ≤5 turns and not a candidate/moderator
        if r["n"] <= 5:
            out[did].add(last_name)
    return out


# === 2024 cluster → speaker handling =======================================

def load_2024_cluster_mapping() -> dict[tuple[str, str], dict]:
    """Return {(debate_audio_id, pyannote_cluster): {speaker, party, result, speaker_role}}."""
    if not CLUSTER_MAP_CSV.exists():
        return {}
    df = pd.read_csv(CLUSTER_MAP_CSV)
    df["debate_audio_id"] = df["debate_audio_id"].astype(str)
    out = {}
    for _, r in df.iterrows():
        did = str(r["debate_audio_id"])
        cluster = str(r["pyannote_cluster"])
        # Prefer verified labels; fall back to suggestions
        if str(r.get("verified", "")).strip().lower() == "yes" and r.get("final_speaker"):
            name = str(r["final_speaker"]).strip()
            role = str(r.get("final_role", "candidate")).strip()
        else:
            name = str(r.get("suggested_speaker") or "").strip()
            role = str(r.get("suggested_role", "")).strip()
            if role == "panelist_or_audience":
                role = "audience"
            elif role == "noise_or_skip":
                role = "unknown"

        # Pull party/result from debates.csv if we recognize the speaker name
        party, result = "—", "unknown"
        if role == "candidate" and name:
            # Quick lookup in debates.csv
            debates = pd.read_csv(META_CSV)
            debates["debate_audio_id"] = debates["debate_audio_id"].astype(str)
            match = debates[(debates["debate_audio_id"] == did) & (debates["speaker"] == name)]
            if not match.empty:
                party = str(match.iloc[0]["party"])
                result = str(match.iloc[0]["result"])
        elif role == "moderator":
            result = "moderator"
        elif role == "audience":
            result = "audience"

        out[(did, cluster)] = {
            "speaker": name or f"({cluster})",
            "party": party,
            "result": result,
            "speaker_role": role or "unknown",
        }
    return out


def load_pyannote_segments_for_2024() -> dict[str, list[tuple[float, float, str]]]:
    """Return {did: [(start, end, cluster), ...]} sorted by start, for 1034 and 1036."""
    if not PYANNOTE_SEGS_CSV.exists():
        return {}
    seg = pd.read_csv(PYANNOTE_SEGS_CSV, low_memory=False)
    seg["debate_audio_id"] = seg["debate_audio_id"].astype(str)
    seg = seg[seg["debate_audio_id"].isin(USE_WHISPER_CANONICAL)]
    out: dict[str, list[tuple[float, float, str]]] = {}
    for did, grp in seg.groupby("debate_audio_id"):
        rows = grp[["start_sec", "end_sec", "diarized_speaker"]].sort_values("start_sec")
        out[did] = [(float(s), float(e), str(c)) for s, e, c in rows.itertuples(index=False, name=None)]
    return out


def dominant_cluster(start: float, end: float, segments: list[tuple[float, float, str]]) -> str | None:
    """Find the pyannote cluster with the most overlap inside [start, end].
    Linear scan with early break (segments are sorted by start)."""
    if not segments:
        return None
    totals: dict[str, float] = defaultdict(float)
    for s, e, cluster in segments:
        if e < start:
            continue
        if s > end:
            break
        ov = max(0.0, min(end, e) - max(start, s))
        if ov > 0:
            totals[cluster] += ov
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


# === main ===================================================================

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {META_CSV}")
    debates = pd.read_csv(META_CSV)
    debates["debate_audio_id"] = debates["debate_audio_id"].astype(str)
    cand_per_debate, cand_full = build_candidate_lookups(debates)

    debate_meta = (
        debates[["debate_audio_id", "debate_id", "year", "debate_type", "date"]]
        .drop_duplicates(subset=["debate_audio_id"])
        .set_index("debate_audio_id")
    )

    print(f"[load] {TEXT_ALIGNED}")
    text_df = pd.read_csv(TEXT_ALIGNED, low_memory=False)
    text_df["debate_audio_id"] = text_df["debate_audio_id"].astype(str)
    text_df = text_df[~text_df["debate_audio_id"].isin(USE_WHISPER_CANONICAL)]
    print(f"  text-aligned: {len(text_df):,} sentences, {text_df['debate_audio_id'].nunique()} debates")

    # Build town-hall audience heuristic
    town_hall_audience = infer_town_hall_audience(text_df, cand_per_debate)
    if town_hall_audience:
        for did, names in town_hall_audience.items():
            print(f"  town-hall audience inferred for {did}: {len(names)} speakers")

    # Match-rate lookup for tier assignment
    summary = pd.read_csv(ALIGN_DIR / "text_aligned_summary.csv")
    summary["debate_audio_id"] = summary["debate_audio_id"].astype(str)
    rate_lookup = dict(zip(summary["debate_audio_id"], summary["match_rate_pct"]))

    print(f"[load] {WHISPER_CANONICAL}")
    whisper_df = pd.read_csv(WHISPER_CANONICAL, low_memory=False) if WHISPER_CANONICAL.exists() else pd.DataFrame()
    if not whisper_df.empty:
        whisper_df["debate_audio_id"] = whisper_df["debate_audio_id"].astype(str)
        whisper_df = whisper_df[whisper_df["debate_audio_id"].isin(USE_WHISPER_CANONICAL)]
        print(f"  whisper-canonical: {len(whisper_df):,} sentences, {whisper_df['debate_audio_id'].nunique()} debates")

    # 2024 cluster mapping + pyannote segments for label propagation
    cluster_map = load_2024_cluster_mapping()
    pyannote_segs = load_pyannote_segments_for_2024()
    print(f"  2024 cluster mapping rows: {len(cluster_map)}")

    trim_offsets: dict[str, dict] = {}
    for did in USE_WHISPER_CANONICAL:
        p = ALIGN_DIR / "mfa_corpus" / did / f"{did}_trim_offset.json"
        if p.exists():
            trim_offsets[did] = json.loads(p.read_text())

    # === Build unified rows =================================================
    out_rows: list[dict] = []
    n_skipped_artifacts = 0

    # --- A) text-aligned (24 of 26 Presidential debates) ----------------------
    for _, r in text_df.iterrows():
        did = str(r["debate_audio_id"])
        speaker_raw = str(r["speaker_raw"]).strip()
        clf = classify_speaker(speaker_raw, did, cand_per_debate, cand_full, town_hall_audience)
        if clf is None:
            n_skipped_artifacts += 1
            continue

        match_rate = rate_lookup.get(did, 0)
        tier = "A-equiv" if match_rate >= 85 else ("B-partial" if match_rate >= 60 else "C-lowmatch")

        out_rows.append({
            "debate_audio_id": did,
            "turn_id": r["turn_id"],
            "sentence_id": r["sentence_id"],
            "sentence_index_in_turn": int(r["sentence_index_in_turn"]),
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            "duration_sec": r["duration_sec"],
            "speaker_raw": speaker_raw,
            **clf,
            "sentence_text": r["sentence_text"],
            "n_words": int(r["n_words"]),
            "n_words_matched": int(r["n_words_matched"]),
            "alignment_tier": tier,
            "alignment_source": "text_via_whisper",
            "in_debate_window": True,
        })

    # --- B) whisper-canonical (1034 / 1036) -----------------------------------
    for _, r in whisper_df.iterrows():
        did = str(r["debate_audio_id"])
        s_start, s_end = float(r["start_sec"]), float(r["end_sec"])
        cluster = dominant_cluster(s_start, s_end, pyannote_segs.get(did, []))
        if cluster:
            info = cluster_map.get((did, cluster), {})
            speaker_raw = cluster
            speaker_disp = info.get("speaker", cluster)
            party = info.get("party", "—")
            result = info.get("result", "unknown")
            role = info.get("speaker_role", "unknown")
        else:
            speaker_raw = "UNKNOWN"
            speaker_disp = "UNKNOWN"
            party = "—"
            result = "unknown"
            role = "unknown"

        offsets = trim_offsets.get(did, {})
        trim_start = float(offsets.get("trim_start", 0.0))
        trim_end = float(offsets.get("trim_end", 1e9))
        in_window = (trim_start <= s_start <= trim_end) or (trim_start <= s_end <= trim_end)

        out_rows.append({
            "debate_audio_id": did,
            "turn_id": "",
            "sentence_id": r["sentence_id"],
            "sentence_index_in_turn": int(r.get("sentence_index", 0)),
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": r["duration_sec"],
            "speaker_raw": speaker_raw,
            "speaker": speaker_disp,
            "party": party,
            "result": result,
            "speaker_role": role,
            "sentence_text": r["sentence_text"],
            "n_words": int(r["n_words"]),
            "n_words_matched": int(r["n_words"]),
            "alignment_tier": "B",
            "alignment_source": "whisper_canonical",
            "in_debate_window": bool(in_window),
        })

    # === Join debate-level metadata + sort + write ===========================
    master = pd.DataFrame(out_rows)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)
    master = master.join(debate_meta, on="debate_audio_id")

    col_order = [
        "debate_audio_id", "debate_id", "year", "debate_type", "date",
        "turn_id", "sentence_id", "sentence_index_in_turn",
        "start_sec", "end_sec", "duration_sec",
        "speaker_raw", "speaker", "party", "result", "speaker_role",
        "sentence_text",
        "n_words", "n_words_matched",
        "alignment_tier", "alignment_source", "in_debate_window",
    ]
    master = master[col_order]
    master = master.sort_values(["year", "debate_audio_id", "start_sec"]).reset_index(drop=True)

    master.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  ({len(master):,} sentences, skipped {n_skipped_artifacts} header artifacts)")

    print("\n=== Sentence count by speaker_role ===")
    print(master.groupby(["speaker_role", "result"]).size().rename("n").to_string())

    print("\n=== Candidate words by winner vs loser (all 26 debates) ===")
    cands = master[master["speaker_role"] == "candidate"]
    print(cands.groupby("result").agg(
        n_sentences=("sentence_id", "count"),
        words=("n_words", "sum"),
        debates=("debate_audio_id", "nunique"),
    ).to_string())

    print("\n=== Alignment tier distribution ===")
    print(master.groupby("alignment_tier").size().rename("n_sentences").to_string())

    print("\n=== Unknown-role remaining ===")
    unk = master[master["speaker_role"] == "unknown"]
    if not unk.empty:
        print(unk.groupby("speaker_raw").size().sort_values(ascending=False).head(10).rename("n").to_string())
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
