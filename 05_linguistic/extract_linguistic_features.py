#!/usr/bin/env python3
"""
Stage 5 — per-sentence linguistic + syntactic features.

For each sentence in outputs/master_sentences.csv, compute:

  SENTENCE TYPE
    is_question, is_imperative, is_exclamatory
    sentence_form          : declarative / interrogative / imperative / exclamatory

  SYNTACTIC (from spaCy parse)
    parse_tree_depth       : depth of dependency tree
    n_clauses              : count of ROOT + clausal-complement nodes
    n_subordinate_clauses  : count of nodes whose dep ∈ {ccomp, xcomp, advcl, acl}
    mean_dependency_length : mean (token.i - token.head.i)
    n_negations            : count of NEG-marked tokens

  LEXICAL / VOCABULARY
    n_chars, n_words, mean_word_length
    type_token_ratio (TTR)
    pct_function_words     : ADP + AUX + CCONJ + DET + PART + PRON + SCONJ
    pct_content_words      : NOUN + VERB + ADJ + ADV
    pct_pronouns_first     : first-person pronouns (I, we, my, our, …)
    pct_pronouns_second    : second-person pronouns (you, your, …)
    pct_pronouns_third     : third-person pronouns (he/she/they, his/her/their, …)

  READABILITY
    flesch_reading_ease, flesch_kincaid_grade
    smog_index, gunning_fog

  POS DISTRIBUTION
    pos_<TAG>_rate         : fraction of tokens with each Universal POS

  SENTIMENT
    vader_compound, vader_pos, vader_neg, vader_neu
    (Optional: RoBERTa label/score, off by default — slow)

  PSYCHOLINGUISTIC (empath) — relevant categories for political discourse
    emp_power, emp_government, emp_negative_emotion, emp_positive_emotion,
    emp_aggression, emp_optimism, emp_pride, emp_certainty, emp_achievement,
    emp_work, emp_money, emp_family, emp_war, emp_violence, emp_religion,
    emp_morality, emp_law, emp_leader, emp_independence, emp_help

  EMBEDDINGS (separate file)
    Saved to outputs/linguistic_embeddings.parquet (sentence_id + 384-dim
    sentence-transformer vector). Optional via --skip-embeddings.

Outputs:
    outputs/linguistic_features.csv
    outputs/linguistic_embeddings.parquet

Run:
    python 05_linguistic/extract_linguistic_features.py
    python 05_linguistic/extract_linguistic_features.py --limit 300
    python 05_linguistic/extract_linguistic_features.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT = Path("/Users/lavanya/debate_analysis")
MASTER_CSV = PROJECT / "outputs" / "master_sentences.csv"
OUT_CSV = PROJECT / "outputs" / "linguistic_features.csv"
OUT_EMB = PROJECT / "outputs" / "linguistic_embeddings.parquet"

EMPATH_CATEGORIES = [
    "power", "government", "politics", "negative_emotion", "positive_emotion",
    "aggression", "anger", "fear", "joy", "optimism", "pride", "certainty",
    "achievement", "work", "money", "business", "economics", "law",
    "leader", "independence", "patriotism", "war", "violence", "weapon",
    "family", "children", "youth", "health", "medical_emergency",
    "religion", "morality", "ethics", "help", "love", "hate",
]

# Spacy POS tags we'll surface as their own columns
PRIMARY_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN",
                "AUX", "DET", "ADP", "CCONJ", "SCONJ", "INTJ", "NUM"]

FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
SUBORDINATE_DEPS = {"ccomp", "xcomp", "advcl", "acl", "relcl"}

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours", "myself", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them",
                "their", "theirs", "himself", "herself", "themselves", "it", "its"}


def tree_depth(token, seen=None) -> int:
    """Max depth of the dependency subtree rooted at `token`."""
    if seen is None:
        seen = set()
    seen.add(token.i)
    children = [c for c in token.children if c.i not in seen]
    if not children:
        return 1
    return 1 + max(tree_depth(c, seen) for c in children)


def compute_syntactic(doc) -> dict:
    """spaCy-based syntactic + sentence-type features."""
    tokens = [t for t in doc if not t.is_space]
    n_tok = len(tokens)
    if n_tok == 0:
        return {}

    # Sentence type — primary cue is final punctuation; refined with parse
    text = doc.text.strip()
    is_question = text.endswith("?")
    is_exclam = text.endswith("!")
    # Imperative heuristic: starts with a VERB whose dep is ROOT and there's no overt subject
    is_imp = False
    if tokens:
        first = tokens[0]
        root = next((t for t in tokens if t.dep_ == "ROOT"), None)
        if root and root.pos_ == "VERB":
            has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in tokens)
            if not has_subject and not is_question:
                is_imp = True
    sentence_form = (
        "interrogative" if is_question else
        "imperative" if is_imp else
        "exclamatory" if is_exclam else
        "declarative"
    )

    # Syntactic complexity
    roots = [t for t in tokens if t.dep_ == "ROOT"]
    depth = max((tree_depth(r) for r in roots), default=1)
    n_clauses = len(roots) + sum(1 for t in tokens if t.dep_ in {"ccomp", "xcomp"})
    n_sub = sum(1 for t in tokens if t.dep_ in SUBORDINATE_DEPS)
    dep_lens = [abs(t.i - t.head.i) for t in tokens if t.head is not t]
    mean_dep_len = float(np.mean(dep_lens)) if dep_lens else 0.0
    n_neg = sum(1 for t in tokens if t.dep_ == "neg" or t.lower_ in {"not", "n't", "no", "never"})

    # POS distribution
    pos_counts = {p: 0 for p in PRIMARY_POS}
    n_function = n_content = 0
    for t in tokens:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
        if t.pos_ in FUNCTION_POS:
            n_function += 1
        elif t.pos_ in CONTENT_POS:
            n_content += 1
    pos_rates = {f"pos_{p.lower()}_rate": pos_counts[p] / n_tok for p in PRIMARY_POS}

    # Lexical
    words_alpha = [t.lower_ for t in tokens if t.is_alpha]
    n_chars = sum(len(w) for w in words_alpha)
    n_words = len(words_alpha)
    mean_word_len = n_chars / n_words if n_words else 0.0
    ttr = len(set(words_alpha)) / n_words if n_words else 0.0

    n_p1 = sum(1 for w in words_alpha if w in FIRST_PERSON)
    n_p2 = sum(1 for w in words_alpha if w in SECOND_PERSON)
    n_p3 = sum(1 for w in words_alpha if w in THIRD_PERSON)

    return {
        "is_question": int(is_question),
        "is_imperative": int(is_imp),
        "is_exclamatory": int(is_exclam),
        "sentence_form": sentence_form,
        "parse_tree_depth": depth,
        "n_clauses": n_clauses,
        "n_subordinate_clauses": n_sub,
        "mean_dependency_length": round(mean_dep_len, 3),
        "n_negations": n_neg,
        "n_chars": n_chars,
        "n_words_spacy": n_words,
        "mean_word_length": round(mean_word_len, 3),
        "type_token_ratio": round(ttr, 4),
        "pct_function_words": round(n_function / n_tok, 4),
        "pct_content_words": round(n_content / n_tok, 4),
        "pct_pronouns_first": round(n_p1 / max(1, n_tok), 4),
        "pct_pronouns_second": round(n_p2 / max(1, n_tok), 4),
        "pct_pronouns_third": round(n_p3 / max(1, n_tok), 4),
        **{k: round(v, 4) for k, v in pos_rates.items()},
    }


def compute_readability(text: str) -> dict:
    """textstat readability scores. Returns NaN on too-short input."""
    import textstat
    try:
        return {
            "flesch_reading_ease":   float(textstat.flesch_reading_ease(text)),
            "flesch_kincaid_grade":  float(textstat.flesch_kincaid_grade(text)),
            "smog_index":            float(textstat.smog_index(text)),
            "gunning_fog":           float(textstat.gunning_fog(text)),
        }
    except Exception:
        return {k: np.nan for k in ["flesch_reading_ease", "flesch_kincaid_grade",
                                     "smog_index", "gunning_fog"]}


def compute_sentiment(vader, text: str) -> dict:
    """VADER lexicon-based sentiment."""
    s = vader.polarity_scores(text)
    return {
        "vader_compound": s["compound"],
        "vader_pos": s["pos"],
        "vader_neg": s["neg"],
        "vader_neu": s["neu"],
    }


def compute_empath(emp_lex, text: str) -> dict:
    """Empath psycholinguistic-category presence scores (0..1 normalized)."""
    try:
        scores = emp_lex.analyze(text, categories=EMPATH_CATEGORIES, normalize=True)
    except Exception:
        scores = {}
    return {f"emp_{c}": float(scores.get(c) or 0.0) for c in EMPATH_CATEGORIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default=str(MASTER_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--out-embeddings", default=str(OUT_EMB))
    ap.add_argument("--debates", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="Skip the sentence-transformer embedding step")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                    help="384-dim, fast. Use 'all-mpnet-base-v2' for 768-dim higher quality.")
    args = ap.parse_args()

    # === Load data and models ==============================================
    import spacy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from empath import Empath

    print(f"[load] {args.master_csv}")
    master = pd.read_csv(args.master_csv, low_memory=False)
    master["debate_audio_id"] = master["debate_audio_id"].astype(str)
    master["sentence_text"] = master["sentence_text"].fillna("").astype(str)
    if args.debates:
        master = master[master["debate_audio_id"].isin(args.debates)]
    if args.limit:
        master = master.head(args.limit)
    print(f"  {len(master):,} sentences across {master['debate_audio_id'].nunique()} debates")

    print("[load] spaCy en_core_web_sm…")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # disable NER for speed

    print("[load] VADER + Empath…")
    vader = SentimentIntensityAnalyzer()
    emp_lex = Empath()

    embedder = None
    if not args.skip_embeddings:
        print(f"[load] sentence-transformer ({args.embedding_model})…")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedding_model)

    # === Per-sentence feature extraction ===================================
    out_rows: list[dict] = []
    print("[run] extracting features…")

    texts = master["sentence_text"].tolist()
    sids = master["sentence_id"].tolist()
    debates_col = master["debate_audio_id"].tolist()

    # Process with spaCy's nlp.pipe for speed
    for i, doc in enumerate(nlp.pipe(texts, batch_size=128)):
        text = texts[i]
        sid = sids[i]
        did = debates_col[i]

        row = {"sentence_id": sid, "debate_audio_id": did}
        row.update(compute_syntactic(doc))
        row.update(compute_readability(text))
        row.update(compute_sentiment(vader, text))
        row.update(compute_empath(emp_lex, text))
        out_rows.append(row)

        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(texts)}")

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}  ({len(out):,} rows × {len(out.columns)} cols)")

    # === Embeddings (separate parquet file to keep CSV small) ==============
    if embedder is not None:
        print(f"\n[run] computing sentence embeddings (batches of 64)…")
        emb = embedder.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        emb_df = pd.DataFrame(emb, columns=[f"e{i:03d}" for i in range(emb.shape[1])])
        emb_df.insert(0, "sentence_id", sids)
        emb_df.to_parquet(args.out_embeddings, index=False)
        print(f"[write] {args.out_embeddings}  ({len(emb_df):,} × {emb.shape[1]+1})")

    # === Summary ===========================================================
    print("\n=== Sentence form distribution ===")
    print(out["sentence_form"].value_counts().to_string())

    print("\n=== Sentiment & key features by speaker role (joined to master) ===")
    m = master[["sentence_id", "speaker_role", "result"]].merge(out, on="sentence_id")
    cand = m[m["speaker_role"] == "candidate"]
    print(cand.groupby("result").agg(
        n=("sentence_id", "count"),
        vader_compound=("vader_compound", "mean"),
        vader_pos=("vader_pos", "mean"),
        vader_neg=("vader_neg", "mean"),
        pct_p1=("pct_pronouns_first", "mean"),
        pct_p2=("pct_pronouns_second", "mean"),
        ttr=("type_token_ratio", "mean"),
        flesch=("flesch_reading_ease", "mean"),
        n_clauses=("n_clauses", "mean"),
        emp_power=("emp_power", "mean"),
        emp_neg=("emp_negative_emotion", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
