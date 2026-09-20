#!/usr/bin/env python3
"""
Stage 5b — rule-based DAMSL discourse + Bias tagger.

Implements binary detectors for the BEADS-TAGS schema (see
BEADS-TAGS.xlsx in the original project). One column per tag; sentences
can carry multiple tags (multi-label).

Tags marked TEXT-UNTRACKED can't be reliably detected from text alone
(e.g. INT interrupts need timing; SE selective-evidence needs world
knowledge). These remain 0 in the output and would require human or
LLM annotation.

Inputs:
    outputs/master_sentences.csv
    outputs/linguistic_features.csv  (uses sentence_form already computed)

Output:
    outputs/damsl_bias_tags.csv  (sentence_id + ~40 binary tag columns + counts)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT = Path("/Users/lavanya/debate_analysis")
MASTER_CSV = PROJECT / "outputs" / "master_sentences.csv"
LING_CSV = PROJECT / "outputs" / "linguistic_features.csv"
OUT_CSV = PROJECT / "outputs" / "damsl_bias_tags.csv"


# ===== Lexicons =============================================================

AUX_VERBS_QYN = {"are", "is", "am", "was", "were", "be", "been", "being",
                 "do", "does", "did", "have", "has", "had", "having",
                 "can", "could", "will", "would", "shall", "should",
                 "may", "might", "must"}
WH_WORDS = {"what", "where", "when", "why", "how", "who", "which", "whom", "whose"}

RHETORICAL_TAILS = (
    r"\b(isn'?t it|aren'?t (they|we|you)|don'?t (you|we|they)|right\?|wouldn'?t (you|we))\b"
)
HEDGE_PATTERNS = re.compile(
    r"\b(I think|I believe|I feel|in my opinion|I would say|I'?d argue|I suppose|maybe|perhaps)\b",
    re.IGNORECASE,
)
STANCE_PATTERNS = re.compile(
    r"\b(I (believe|think|feel|argue|maintain|insist|contend)|in my (view|opinion))\b",
    re.IGNORECASE,
)
SELF_PROMO_PATTERNS = re.compile(
    r"\bI (created|built|rebuilt|made|achieved|delivered|fixed|saved|founded|led|brought|"
    r"established|launched|cut|reduced|increased)\b",
    re.IGNORECASE,
)
GREETING_PATTERNS = re.compile(
    r"\b(good (evening|morning|afternoon|night)|hello|hi everyone|welcome|"
    r"thank you for joining|good night)\b",
    re.IGNORECASE,
)
APOLOGY_PATTERNS = re.compile(
    r"\b(i'?m sorry|i apologi[sz]e|my apologies|i regret|forgive me)\b",
    re.IGNORECASE,
)
THANKS_PATTERNS = re.compile(
    r"\b(thank you|thanks)\b",
    re.IGNORECASE,
)
ACK_PATTERNS = re.compile(
    r"^\s*(i see|right|okay|ok|mm-?hmm|yeah|uh-huh|got it|understood|fair enough)[\.\,!\?]*\s*$",
    re.IGNORECASE,
)
AGR_PATTERNS = re.compile(
    r"\b(i agree|that'?s (right|correct|true)|exactly|absolutely|"
    r"you'?re right|i concur|that'?s a good point)\b",
    re.IGNORECASE,
)
DIS_PATTERNS = re.compile(
    r"\b(i disagree|i don'?t agree|that'?s not (right|correct|true)|i (don'?t|do not) (think|believe) (so|that)|"
    r"i beg to differ)\b",
    re.IGNORECASE,
)
REJ_PATTERNS = re.compile(
    r"^\s*(no\b|absolutely not|that'?s (false|not true|a lie|wrong)|never\b|not at all)",
    re.IGNORECASE,
)
EXPL_PATTERNS = re.compile(
    r"\b(because|the reason (is|why)|what I mean (is|by)|in other words|that is|"
    r"let me explain|to clarify|specifically)\b",
    re.IGNORECASE,
)
CORR_PATTERNS = re.compile(
    r"\b(let me clarify|to be clear|actually,|to correct|i (meant|misspoke))\b",
    re.IGNORECASE,
)
CH_PATTERNS = re.compile(
    r"\b(i (don'?t|do not) believe (that'?s|that is) accurate|that'?s (incorrect|inaccurate|misleading|"
    r"a distortion|simply false))\b",
    re.IGNORECASE,
)
TT_PATTERNS = re.compile(
    r"\b(let me (say|add|finish|continue|respond|jump in)|if i may|may i|excuse me)\b",
    re.IGNORECASE,
)
TG_PATTERNS = re.compile(
    r"\b(what do you (think|say)|your turn|over to you|do you (agree|disagree))\b",
    re.IGNORECASE,
)
MS_PATTERNS = re.compile(
    r"\b(but let me (just )?(ask|say)|i want to address|i'?ll come back to|"
    r"as i (was|just) saying|let me get back to)\b",
    re.IGNORECASE,
)
HS_PATTERNS = re.compile(
    r"\b((19|20)\d{2}|ronald reagan|john kennedy|jfk|martin luther king|"
    r"bill of rights|civil war|founding fathers|world war|cold war|"
    r"during the\s+\w+\s+administration)\b",
    re.IGNORECASE,
)
R_REQ_PATTERNS = re.compile(
    r"\b(can you repeat|i didn'?t hear|could you (say that )?again|"
    r"can you say that again|come again)\b",
    re.IGNORECASE,
)
SEEP_PATTERNS = re.compile(
    r"\b(why (is|do|did|does)|how (is|do|did|does)|please (explain|elaborate|clarify))\b",
    re.IGNORECASE,
)

# Bias-side patterns
FEAR_WORDS = {"disaster", "catastrophe", "destroy", "destroyed", "destroying", "threat",
              "dangerous", "danger", "crisis", "ruin", "ruined", "attack", "invasion",
              "collapse", "doom", "terror", "terrified", "scary"}
EMOTION_APPEAL = re.compile(
    r"\b(if you care|imagine if|think about (your|the)|families|children|the children|"
    r"our kids|grandchildren)\b",
    re.IGNORECASE,
)
PRIDE_APPEAL = re.compile(
    r"\b(great country|the greatest nation|american (people|values|spirit)|patriots?|"
    r"made (in )?america|proud to be|true american)\b",
    re.IGNORECASE,
)
PATRIOTISM = re.compile(
    r"\b(america|american|patriot|patriotism|freedom|liberty|constitution|"
    r"founding fathers|stars and stripes|flag|country first|god bless america)\b",
    re.IGNORECASE,
)
UNITY = re.compile(
    r"\b(together we|we as a (nation|people|country)|all of us|united|in unity|"
    r"common (good|ground|cause)|our shared|regardless of)\b",
    re.IGNORECASE,
)
DEFLECTION = re.compile(
    r"\b(what i regret is|what i'?m focused on|what (we|i) should (be )?talk(ing)? about|"
    r"the real question is|the bigger issue|let'?s not forget)\b",
    re.IGNORECASE,
)
POLITICAL_OUTGROUP = re.compile(
    r"\b(democrats? (are|have)|republicans? (are|have)|the (radical |far )?(left|right)|"
    r"the other side|the establishment|the swamp|liberal (elites?|media)|"
    r"conservative (elites?|media))\b",
    re.IGNORECASE,
)
RACE_LEXICON = re.compile(
    r"\b(black (people|community|americans?)|african americans?|hispanic|latino|latina|"
    r"asian (americans?|people)|whites?|caucasian|race|racial|ethnic)\b",
    re.IGNORECASE,
)

# Personal attack: "you/he/she + negative adjective"
PERSONAL_ATTACK = re.compile(
    r"\b(he|she|you|they|trump|biden|obama|clinton|harris|bush|dukakis|dole|kerry|romney|mccain|gore)\s+"
    r"(is|was|are|were|has been|have been)\s+"
    r"(a )?(liar|corrupt|criminal|incompetent|stupid|dumb|weak|crooked|terrible|"
    r"horrible|awful|fraud|disaster|disgrace|failed|failure|dishonest|untrust\w*)",
    re.IGNORECASE,
)

# A "biased question" is a question containing attack/loaded language
LOADED_TERMS = re.compile(
    r"\b(liar|corrupt|fraud|disaster|incompetent|crooked|criminal|coward|fool)\b",
    re.IGNORECASE,
)

OPPONENT_NAMES = {
    "trump", "biden", "obama", "clinton", "harris", "bush", "dukakis", "dole", "kerry",
    "romney", "mccain", "gore", "perot", "quayle", "bentsen", "lieberman", "cheney",
    "kemp", "stockdale", "edwards", "pence", "kaine", "ryan", "palin", "vance", "walz",
}
BLAME_VERBS = re.compile(
    r"\b(left us|destroyed|ruined|broke|abandoned|failed (to|us|the)|"
    r"was responsible for|caused (the|our|this))\b",
    re.IGNORECASE,
)


# ===== Detectors ============================================================

def detect_question_type(text: str, sentence_form: str) -> dict:
    """Q-W (wh-question), YNQ (yes-no question), Q-RHET (rhetorical),
    OQ (open-ended), SEEP (seek explanation)."""
    flags = {"damsl_Q_W": 0, "damsl_YNQ": 0, "damsl_Q_RHET": 0,
              "damsl_OQ": 0, "damsl_SEEP": 0}
    if sentence_form != "interrogative":
        # Rhetorical questions can be declarative + tail tag ("..., right?")
        if re.search(RHETORICAL_TAILS, text, re.IGNORECASE):
            flags["damsl_Q_RHET"] = 1
        return flags

    # Tokenize first word loosely
    first_token = re.match(r"\s*([A-Za-z']+)", text)
    fw = first_token.group(1).lower() if first_token else ""

    if fw in WH_WORDS:
        flags["damsl_Q_W"] = 1
        if fw == "what" and re.search(r"\bwhat (do|are) you (think|believe|propose|suggest)\b", text, re.IGNORECASE):
            flags["damsl_OQ"] = 1
        if re.search(SEEP_PATTERNS, text):
            flags["damsl_SEEP"] = 1
    elif fw in AUX_VERBS_QYN:
        flags["damsl_YNQ"] = 1

    # Rhetorical signal: question with negative tail or contains hedge
    if re.search(RHETORICAL_TAILS, text, re.IGNORECASE):
        flags["damsl_Q_RHET"] = 1
    return flags


def tag_sentence(text: str, sentence_form: str) -> dict:
    """Return dict of all DAMSL + Bias binary tags."""
    flags: dict[str, int] = {}

    # === DAMSL (discourse) =================================================
    flags.update(detect_question_type(text, sentence_form))

    # Statement: default for declarative w/o other major dialog acts
    flags["damsl_S"] = 1 if sentence_form == "declarative" else 0

    # Imperative → command
    flags["damsl_IAFA"] = 1 if sentence_form == "imperative" else 0

    # Lexically detected
    flags["damsl_ACK"] = 1 if ACK_PATTERNS.match(text) else 0
    flags["damsl_AGR"] = 1 if AGR_PATTERNS.search(text) else 0
    flags["damsl_DIS"] = 1 if DIS_PATTERNS.search(text) else 0
    flags["damsl_REJ"] = 1 if REJ_PATTERNS.search(text) else 0
    flags["damsl_GR"] = 1 if GREETING_PATTERNS.search(text) else 0
    flags["damsl_APO"] = 1 if APOLOGY_PATTERNS.search(text) else 0
    flags["damsl_THK"] = 1 if THANKS_PATTERNS.search(text) else 0
    flags["damsl_EXPL"] = 1 if EXPL_PATTERNS.search(text) else 0
    flags["damsl_CORR"] = 1 if CORR_PATTERNS.search(text) else 0
    flags["damsl_CH"] = 1 if CH_PATTERNS.search(text) else 0
    flags["damsl_TT"] = 1 if TT_PATTERNS.search(text) else 0
    flags["damsl_TG"] = 1 if TG_PATTERNS.search(text) else 0
    flags["damsl_MS"] = 1 if MS_PATTERNS.search(text) else 0
    flags["damsl_HS"] = 1 if HS_PATTERNS.search(text) else 0
    flags["damsl_R_REQ"] = 1 if R_REQ_PATTERNS.search(text) else 0

    # ANS: a Yes/No followed by elaboration (loose) — first token is yes/no + text after
    first = re.match(r"^\s*(yes|no)\b", text, re.IGNORECASE)
    flags["damsl_ANS"] = 1 if first else 0

    # Untracked (need timing / floor-management context)
    flags["damsl_INT"] = 0  # interrupt
    flags["damsl_T_REQ"] = 1 if re.search(r"\b(am i allowed|can i (just )?(respond|finish))\b", text, re.IGNORECASE) else 0
    flags["damsl_TA"] = 0  # turn-accept (context-dependent)

    # IAFA-OF (offering): "Would you like…", "I can…", "Let me…"
    flags["damsl_IAFA_OF"] = 1 if re.search(
        r"\b(would you like|let me (offer|help|do)|i can (offer|help|do))\b", text, re.IGNORECASE
    ) else 0

    # S-Inform = statement carrying new information (numeric/factual content)
    has_number = bool(re.search(r"\b\d", text))
    has_named_entity = bool(re.search(r"\b(percent|%|million|billion|trillion|dollar|year)\b", text, re.IGNORECASE))
    flags["damsl_S_Inform"] = 1 if (flags["damsl_S"] and (has_number or has_named_entity)) else 0

    # === BIAS ==============================================================
    flags["bias_AF"] = 1 if any(w in text.lower() for w in FEAR_WORDS) else 0
    flags["bias_AE"] = 1 if EMOTION_APPEAL.search(text) else 0
    flags["bias_AP"] = 1 if PRIDE_APPEAL.search(text) else 0
    flags["bias_APAT"] = 1 if PATRIOTISM.search(text) else 0
    flags["bias_UF"] = 1 if UNITY.search(text) else 0
    flags["bias_DF"] = 1 if DEFLECTION.search(text) else 0
    flags["bias_PB"] = 1 if POLITICAL_OUTGROUP.search(text) else 0
    flags["bias_PER"] = 1 if PERSONAL_ATTACK.search(text) else 0
    flags["bias_IT"] = 1 if SELF_PROMO_PATTERNS.search(text) else 0
    flags["bias_IP"] = 1 if STANCE_PATTERNS.search(text) else 0

    # BQ: question + loaded term
    flags["bias_BQ"] = 1 if (sentence_form == "interrogative" and LOADED_TERMS.search(text)) else 0

    # ATTR (blame/attribution): names opponent + blame verb
    text_lc = text.lower()
    has_opponent = any(re.search(rf"\b{n}\b", text_lc) for n in OPPONENT_NAMES)
    flags["bias_ATTR"] = 1 if (has_opponent and BLAME_VERBS.search(text)) else 0

    # AEX (adversarial exchange): personal attack + question OR personal attack + emphatic
    flags["bias_AEX"] = 1 if (flags["bias_PER"] and (sentence_form == "interrogative" or
                                                       re.search(r"\b(again|just|simply)\b", text_lc))) else 0

    # REB (rebuttal): challenges + has opponent reference
    flags["bias_REB"] = 1 if ((flags["damsl_CH"] or flags["damsl_DIS"]) and has_opponent) else 0

    flags["bias_RB"] = 1 if RACE_LEXICON.search(text) else 0

    # Text-untracked bias categories (kept as 0 columns)
    flags["bias_GB"] = 0   # gender bias — requires careful semantic detection
    flags["bias_CB"] = 0   # cognitive bias — requires meaning understanding
    flags["bias_GD"] = 0   # gender dismissal — pronoun-based, contextual
    flags["bias_IA"] = 0   # issue avoidance — needs question/answer pairing
    flags["bias_SE"] = 0   # selective evidence — needs world knowledge
    flags["bias_CBias"] = 0  # cultural bias — needs context

    return flags


def process(text: str, sentence_form: str) -> dict:
    text = "" if (text is None or pd.isna(text)) else str(text).strip()
    if not text:
        # Return all zero flags with consistent schema
        empty = tag_sentence("placeholder text", "declarative")
        empty["damsl_S"] = 0  # explicitly zero out
        return {k: 0 for k in empty}
    return tag_sentence(text, sentence_form)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default=str(MASTER_CSV))
    ap.add_argument("--ling-csv", default=str(LING_CSV))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args()

    print(f"[load] {args.master_csv}")
    master = pd.read_csv(args.master_csv, low_memory=False)
    print(f"[load] {args.ling_csv}")
    ling = pd.read_csv(args.ling_csv)
    if "sentence_form" not in ling.columns:
        raise SystemExit("linguistic_features.csv missing 'sentence_form' — re-run stage 5")

    j = master[["sentence_id", "sentence_text"]].merge(
        ling[["sentence_id", "sentence_form"]], on="sentence_id"
    )
    print(f"  {len(j):,} sentences to tag")

    print("[tag] running rule-based DAMSL + Bias…")
    rows = []
    for i, (sid, text, form) in enumerate(
        zip(j["sentence_id"], j["sentence_text"], j["sentence_form"])
    ):
        flags = process(text, form)
        rows.append({"sentence_id": sid, **flags})
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(j):,}")

    out = pd.DataFrame(rows)

    # Counters
    damsl_cols = [c for c in out.columns if c.startswith("damsl_")]
    bias_cols = [c for c in out.columns if c.startswith("bias_")]
    out["n_damsl_tags"] = out[damsl_cols].sum(axis=1)
    out["n_bias_tags"] = out[bias_cols].sum(axis=1)
    out["n_total_tags"] = out["n_damsl_tags"] + out["n_bias_tags"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}  ({len(out):,} rows × {len(out.columns)} cols)")

    print("\n=== DAMSL tag frequency (% of sentences with tag) ===")
    for c in sorted(damsl_cols, key=lambda c: out[c].mean(), reverse=True):
        pct = out[c].mean() * 100
        print(f"  {c:18s} {pct:5.2f}%")

    print("\n=== BIAS tag frequency (% of sentences with tag) ===")
    for c in sorted(bias_cols, key=lambda c: out[c].mean(), reverse=True):
        pct = out[c].mean() * 100
        print(f"  {c:18s} {pct:5.2f}%")

    # Winner-vs-loser quick summary
    cand = master[master["speaker_role"] == "candidate"][["sentence_id", "result"]].merge(out, on="sentence_id")
    print(f"\n=== Mean tag rate per sentence by winner vs loser (candidates only, {len(cand):,} sentences) ===")
    print("--- Top discourse-tag differences ---")
    means = cand.groupby("result")[damsl_cols + bias_cols].mean() * 100
    means_t = means.T
    if "winner" in means_t.columns and "loser" in means_t.columns:
        means_t["delta"] = means_t["winner"] - means_t["loser"]
        # Discourse
        d_diff = means_t.loc[damsl_cols].sort_values("delta", key=lambda s: s.abs(), ascending=False).head(10)
        print(d_diff.round(2).to_string())
        print("\n--- Top bias-tag differences ---")
        b_diff = means_t.loc[bias_cols].sort_values("delta", key=lambda s: s.abs(), ascending=False).head(10)
        print(b_diff.round(2).to_string())


if __name__ == "__main__":
    main()
