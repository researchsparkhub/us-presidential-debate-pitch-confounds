#!/usr/bin/env python3
"""
Shared data layer for the LLM cross-modal extension (paper §Combined Text and
Acoustic Analysis, Method 4).

Builds, per debate:
  - an ANONYMISED transcript of candidate turns in original chronological
    order (speaker names, party, and date stripped; candidates relabelled
    Candidate A / B / [C] in a caller-supplied order so repeats can permute it)
  - the same 34 standardised measures used by Methods 1-3 (Eq. 1 of the main
    paper: z-scored within debate, pooled across candidates), averaged to one
    row per candidate

This reconstructs the 34-measure table from the same three source CSVs the
original combined-analysis pipeline used (acoustic_features.csv,
acoustic_features_supplementary.csv, linguistic_features.csv), joined on
sentence_id, restricted to candidate rows in the 24 debates used throughout
the paper (1034, 1036 excluded --- the two 2024 debates dropped for
diarization reasons, Appendix "The Two 2024 Debates").

The 34-name -> raw-column mapping below reproduces Table tab:loadings /
names.json exactly.
"""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

PROJECT = Path("/Users/lavanya/debate_analysis")
MASTER = PROJECT / "outputs" / "master_sentences.csv"
ACOUSTIC = PROJECT / "outputs" / "acoustic_features.csv"
ACOUSTIC_SUPP = PROJECT / "outputs" / "acoustic_features_supplementary.csv"
LINGUISTIC = PROJECT / "outputs" / "linguistic_features.csv"

OUT_DIR = Path(__file__).resolve().parent / "llm_ext_cache"
OUT_DIR.mkdir(exist_ok=True)

EXCLUDE_DEBATES = {"1034", "1036"}

# (raw column, source file key, friendly name) -- 34 measures, matches
# names.json / Table tab:loadings exactly.
MEASURES = [
    ("f0_mean", "ac", "Mean pitch"),
    ("f0_std", "ac", "Pitch SD (Hz)"),
    ("f0_range", "ac", "Pitch range (Hz)"),
    ("intensity_mean_db", "ac", "Loudness"),
    ("hnr_mean_db", "ac", "Voice clarity (Praat)"),
    ("total_pause_sec", "ac", "Pause duration"),
    ("n_pauses", "ac", "Pause count"),
    ("mean_pause_sec", "ac", "Mean pause length"),
    ("speech_rate_wps", "ac", "Speech rate"),
    ("articulation_rate_wps", "ac", "Articulation rate (w/s)"),
    ("syllable_rate_per_sec", "sup", "Syllable rate"),
    ("articulation_rate_syll_per_sec", "sup", "Articulation rate (syll/s)"),
    ("egemaps_F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "sup", "Semitone pitch SD/mean"),
    ("egemaps_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", "sup", "Semitone 20-80 pitch range"),
    ("egemaps_HNRdBACF_sma3nz_amean", "sup", "Voice clarity (openSMILE)"),
    ("n_clauses", "ling", "Clauses"),
    ("n_subordinate_clauses", "ling", "Subordinate clauses"),
    ("parse_tree_depth", "ling", "Parse depth"),
    ("mean_dependency_length", "ling", "Dependency length"),
    ("n_words_spacy", "ling", "Words per sentence"),
    ("mean_word_length", "ling", "Word length"),
    ("pct_function_words", "ling", "Function words"),
    ("pct_pronouns_first", "ling", "1st-person pronouns"),
    ("pct_pronouns_second", "ling", "2nd-person pronouns"),
    ("pct_pronouns_third", "ling", "3rd-person pronouns"),
    ("pos_noun_rate", "ling", "Nouns"),
    ("pos_verb_rate", "ling", "Verbs"),
    ("pos_adj_rate", "ling", "Adjectives"),
    ("pos_adv_rate", "ling", "Adverbs"),
    ("flesch_reading_ease", "ling", "Reading ease"),
    ("flesch_kincaid_grade", "ling", "Grade level"),
    ("vader_compound", "ling", "Sentiment"),
    ("vader_pos", "ling", "Positive sentiment"),
    ("vader_neg", "ling", "Negative sentiment"),
]
assert len(MEASURES) == 34
FRIENDLY_NAMES = [m[2] for m in MEASURES]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    if v is None or v == "" or v == "NA" or v == "nan":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_everything():
    master = _read_csv(MASTER)
    ac_rows = {r["sentence_id"]: r for r in _read_csv(ACOUSTIC)}
    sup_rows = {r["sentence_id"]: r for r in _read_csv(ACOUSTIC_SUPP)}
    ling_rows = {r["sentence_id"]: r for r in _read_csv(LINGUISTIC)}
    src = {"ac": ac_rows, "sup": sup_rows, "ling": ling_rows}

    debates = {}  # debate_id -> {candidates: [name,...], turns: [...], sentences: [...]}
    for row in master:
        if row["speaker_role"] != "candidate":
            continue
        d = row["debate_audio_id"]
        if d in EXCLUDE_DEBATES:
            continue
        rec = debates.setdefault(d, {
            "year": row["year"], "date": row["date"],
            "candidates": {}, "sentences": [],
        })
        speaker, result, sid = row["speaker"], row["result"], row["sentence_id"]
        rec["candidates"].setdefault(speaker, result)
        vals = {}
        for col, key, name in MEASURES:
            r = src[key].get(sid)
            vals[name] = _to_float(r.get(col)) if r else None
        rec["sentences"].append({
            "sentence_id": sid,
            "turn_id": row["turn_id"],
            "sentence_index_in_turn": int(row["sentence_index_in_turn"]),
            "speaker": speaker,
            "result": result,
            "text": row["sentence_text"],
            "measures": vals,
        })
    return debates


def zscore_and_aggregate(rec):
    """Eq. 1 of the main paper: z-score each measure within the debate
    (pooled across candidates), then average to one row per candidate."""
    sents = rec["sentences"]
    per_measure_pool = {name: [] for name in FRIENDLY_NAMES}
    for s in sents:
        for name in FRIENDLY_NAMES:
            v = s["measures"][name]
            if v is not None:
                per_measure_pool[name].append(v)
    stats = {}
    for name in FRIENDLY_NAMES:
        pool = per_measure_pool[name]
        if len(pool) >= 2:
            mu, sd = mean(pool), stdev(pool)
        else:
            mu, sd = (pool[0] if pool else 0.0), 1.0
        stats[name] = (mu, sd if sd > 1e-9 else 1.0)

    by_cand = {c: {name: [] for name in FRIENDLY_NAMES} for c in rec["candidates"]}
    for s in sents:
        for name in FRIENDLY_NAMES:
            v = s["measures"][name]
            if v is not None:
                mu, sd = stats[name]
                by_cand[s["speaker"]][name].append((v - mu) / sd)

    agg = {}
    for c, cols in by_cand.items():
        agg[c] = {name: (round(mean(vals), 4) if vals else None) for name, vals in cols.items()}
    return agg


def chronological_turns(rec):
    """Sentences in original chronological order (turn_id, then index)."""
    def turn_num(tid):
        # turn_id like '1001_t0001' -> 1
        return int(tid.split("_t")[-1])
    return sorted(rec["sentences"], key=lambda s: (turn_num(s["turn_id"]), s["sentence_index_in_turn"]))


def build_cache():
    debates = load_everything()
    manifest = {}
    for d in sorted(debates, key=lambda x: int(x)):
        rec = debates[d]
        cands = list(rec["candidates"].keys())  # canonical order = first appearance
        winner = [c for c, r in rec["candidates"].items() if r == "winner"]
        assert len(winner) == 1, (d, rec["candidates"])
        agg = zscore_and_aggregate(rec)
        turns = chronological_turns(rec)
        out = {
            "debate_id": d,
            "year": rec["year"],
            "candidates": cands,
            "results": rec["candidates"],
            "winner": winner[0],
            "n_sentences": len(turns),
            "descriptors": agg,
            "transcript_turns": [
                {"speaker": s["speaker"], "text": s["text"]} for s in turns
            ],
        }
        (OUT_DIR / f"{d}.json").write_text(json.dumps(out, indent=1))
        manifest[d] = {"year": rec["year"], "n_candidates": len(cands), "n_sentences": len(turns)}
        print(f"[{d}] year={rec['year']} candidates={cands} winner={winner[0]} "
              f"sentences={len(turns)}")
    (OUT_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[done] {len(debates)} debates cached to {OUT_DIR}")


if __name__ == "__main__":
    build_cache()
