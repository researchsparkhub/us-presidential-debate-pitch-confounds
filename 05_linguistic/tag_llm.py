#!/usr/bin/env python3
"""
Stage 5c — LLM DAMSL + Beads tagger (Phase A & B best-guess, Sahana Project v2).

For every sentence, Claude Opus 4.8 predicts BOTH:
  - Phase A: DAMSL discourse-act tags
  - Phase B: BEADS bias tags

The model tags DAMSL first and then uses those discourse acts as context when
judging bias — so the DAMSL prediction feeds the Beads prediction. Both are
multi-label (0+ codes) and constrained to the valid code lists via structured
outputs. Runs through the Batch API (async, 50% cheaper).

Unlike the rule-based tagger, this reaches the DAMSL/Beads codes rules can't
detect (INT, TA; GB, GD, CB, IA, SE, CBias).

Requires: ANTHROPIC_API_KEY in the environment.

Inputs:  outputs/master_sentences.csv
Outputs: outputs/llm_tags.csv                        (sentence_id, damsl_tags, beads_tags, counts)
         Sahana Project/tagging_prompt.md            (the exact prompt used)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05_linguistic/tag_llm.py                  # full run (~35k sentences)
    python 05_linguistic/tag_llm.py --limit 50       # smoke test on 50 sentences
    python 05_linguistic/tag_llm.py --prompt-only    # just (re)write tagging_prompt.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import anthropic

PROJECT = Path(__file__).resolve().parent.parent
MASTER = PROJECT / "outputs" / "master_sentences.csv"
OUT_CSV = PROJECT / "outputs" / "llm_tags.csv"
PROMPT_MD = PROJECT / "Sahana Project" / "tagging_prompt.md"

MODEL = "claude-opus-4-8"
CHUNK = 10          # sentences per request (amortizes the cached codebook)
MAX_TOKENS = 3072

# Phase A — DAMSL discourse acts: (code, name, definition)
DAMSL = [
    ("Q-W", "Wh-question", "Question opening with what/where/when/why/how/who/which."),
    ("YNQ", "Yes-no question", "Opens with an auxiliary verb; expects yes or no."),
    ("Q-RHET", "Rhetorical question", "Not seeking an answer; often declarative + tag (\"..., right?\")."),
    ("OQ", "Open-ended question", "Invites an extended answer (\"what do you think/propose?\")."),
    ("SEEP", "Seek explanation", "Asks for reasoning (\"why is...\", \"please explain/elaborate\")."),
    ("S", "Statement", "Plain declarative assertion; the default act for a declarative sentence."),
    ("S-Inform", "Informative statement", "Statement with new factual content — numbers, percentages, dollars, years."),
    ("IAFA", "Action directive / command", "Imperative telling the listener to do something."),
    ("IAFA-OF", "Offer", "Offers to do or provide something (\"would you like\", \"I can offer\")."),
    ("ACK", "Acknowledgment", "Backchannel signalling receipt (\"I see\", \"right\", \"okay\", \"mm-hmm\", \"got it\")."),
    ("AGR", "Agreement", "Explicit agreement (\"I agree\", \"exactly\", \"you're right\")."),
    ("DIS", "Disagreement", "Explicit disagreement (\"I disagree\", \"that's not right\", \"I beg to differ\")."),
    ("REJ", "Rejection", "Flat rejection (\"no\", \"absolutely not\", \"that's false\", \"never\")."),
    ("ANS", "Answer", "Direct answer, typically opening with yes/no."),
    ("GR", "Greeting", "Greeting or welcome (\"good evening\", \"welcome\", \"thank you for joining\")."),
    ("APO", "Apology", "Apology (\"I'm sorry\", \"my apologies\", \"I regret\")."),
    ("THK", "Thanks", "Expression of thanks (\"thank you\", \"thanks\")."),
    ("EXPL", "Explanation", "Gives reasoning or restatement (\"because\", \"the reason is\", \"in other words\")."),
    ("CORR", "Correction / self-clarification", "Corrects or clarifies own prior statement (\"to be clear\", \"I misspoke\")."),
    ("CH", "Challenge / fact-check", "Challenges the accuracy of a claim (\"that's incorrect/misleading/false\")."),
    ("TT", "Take/hold turn", "Bids to take or hold the floor (\"let me say\", \"if I may\", \"may I\", \"excuse me\")."),
    ("TG", "Turn-give", "Hands the floor to another (\"what do you think?\", \"your turn\", \"over to you\")."),
    ("T-REQ", "Turn request", "Requests permission to speak (\"am I allowed\", \"can I respond/finish\")."),
    ("R-REQ", "Repeat request", "Asks for repetition (\"can you repeat\", \"could you say that again\")."),
    ("MS", "Maintain / steer topic", "Steers back to or holds a topic (\"but let me ask\", \"as I was saying\")."),
    ("HS", "Historical / self reference", "References a year, historical figure, or event."),
    ("INT", "Interrupt", "Cuts off or talks over another speaker."),
    ("TA", "Turn-accept", "Accepts an offered turn."),
]

# Phase B — BEADS bias: (code, name, definition)
BEADS = [
    ("AF", "Appeal to fear", "Fear/threat language (disaster, catastrophe, crisis, danger, invasion, collapse)."),
    ("AE", "Appeal to emotion", "Emotional appeal (\"imagine if\", \"the children\", \"our families\", \"if you care\")."),
    ("AP", "Appeal to pride", "Group pride (\"great country\", \"American values\", \"proud to be\")."),
    ("APAT", "Appeal to patriotism", "Patriotic framing (freedom, liberty, constitution, flag, \"God bless America\")."),
    ("UF", "Unity framing", "Collective unity (\"together we\", \"we as a nation\", \"common good\", \"united\")."),
    ("DF", "Deflection", "Redirects away from the question (\"the real question is\", \"let's not forget\", \"what I'm focused on\")."),
    ("PB", "Political-outgroup bias", "Negative framing of a political outgroup (\"the radical left\", \"the swamp\", \"liberal elites\", \"Democrats are\")."),
    ("PER", "Personal attack", "Attacks a person's character (opponent + liar/corrupt/incompetent/crooked/disgrace)."),
    ("IT", "Self-promotion / taking credit", "Claims personal credit (\"I created\", \"I built\", \"I delivered\", \"I cut/reduced\")."),
    ("IP", "Injecting personal stance", "Frames as personal opinion/stance (\"I believe\", \"in my view\", \"I contend\")."),
    ("BQ", "Biased / loaded question", "A question containing loaded terms (liar, corrupt, fraud, criminal, coward)."),
    ("ATTR", "Blame attribution", "Assigns blame to a named opponent (opponent + destroyed/ruined/failed us/was responsible for)."),
    ("AEX", "Adversarial exchange", "Personal attack combined with a question or an emphatic jab (\"again\", \"just\", \"simply\")."),
    ("REB", "Rebuttal", "Challenge or disagreement that references the opponent."),
    ("RB", "Race-related content", "Mentions race/ethnicity (black/hispanic/asian/white community, racial, ethnic)."),
    ("GB", "Gender bias", "Bias based on gender — stereotyping or unequal framing tied to sex/gender."),
    ("GD", "Gender dismissal", "Dismissing or belittling someone on gendered grounds (talking over, condescension tied to gender)."),
    ("CB", "Cognitive bias", "A reasoning bias: false dilemma, anchoring, hasty generalization, straw man, etc."),
    ("IA", "Issue avoidance", "Avoids answering the substantive issue that was raised; a non-answer or topic dodge."),
    ("SE", "Selective evidence", "Cherry-picks facts or cites one-sided evidence while omitting relevant context."),
    ("CBias", "Cultural bias", "Bias tied to culture, religion, or national/ethnic group beyond race."),
]

DAMSL_CODES = [c for c, _, _ in DAMSL]
BEADS_CODES = [c for c, _, _ in BEADS]


def _codebook(rows):
    return "\n".join(f"- {c} ({name}): {defn}" for c, name, defn in rows)


SYSTEM = (
    "You are an expert political-discourse analyst tagging sentences from U.S. presidential "
    "debates. For each sentence you assign two kinds of tags:\n\n"
    "PHASE A — DAMSL discourse acts (what the sentence DOES conversationally):\n"
    + _codebook(DAMSL)
    + "\n\nPHASE B — BEADS bias (rhetorical bias the sentence exhibits):\n"
    + _codebook(BEADS)
    + "\n\nPROCEDURE for each sentence:\n"
    "1. First determine the DAMSL discourse-act tag(s). A sentence may carry several "
    "(e.g. a statement that also challenges the opponent).\n"
    "2. Then, USING those discourse acts as context, determine the BEADS bias tag(s). "
    "The discourse act often constrains the bias — e.g. a challenge (CH) or rejection (REJ) "
    "aimed at the opponent is evidence for a rebuttal (REB); a loaded question (YNQ/Q-W) is "
    "evidence for a biased question (BQ); a non-answer to a question is issue avoidance (IA).\n"
    "3. Both phases are multi-label: return every code that genuinely applies, or an empty "
    "list if none do. Do not over-tag. Judge each sentence on its plain meaning.\n\n"
    "PRECISION NOTES:\n"
    "- IAFA (command) applies ONLY to grammatically imperative sentences — a bare command "
    "such as \"Look at the record\" or \"Answer the question.\" Do NOT assign IAFA to "
    "declarative sentences that merely express what ought to happen (\"we should\", "
    "\"we've got to\", \"I think we need to\", \"everybody ought to\") — those are statements (S), "
    "and a hortatory appeal to shared action may also be UF.\n"
    "- Neutral factual or procedural sentences — moderator introductions, reading the rules, "
    "giving time limits, or plainly citing a statistic or poll — are NOT bias. Leave Beads "
    "empty unless the sentence itself is slanted, loaded, or one-sided.\n\n"
    "Return results for every sentence given, keyed by its exact sentence_id."
)

USER_TEMPLATE = "Tag these sentences:\n[<sentence_id>] <sentence text>\n... (up to %d per request)" % CHUNK

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_id": {"type": "string"},
                    "damsl": {"type": "array", "items": {"type": "string", "enum": DAMSL_CODES}},
                    "beads": {"type": "array", "items": {"type": "string", "enum": BEADS_CODES}},
                },
                "required": ["sentence_id", "damsl", "beads"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def write_prompt_md():
    PROMPT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = (
        "# Tagging prompt — Sahana Project v2 (LLM DAMSL + Beads)\n\n"
        f"Generated by `05_linguistic/tag_llm.py`. Model: **{MODEL}**, "
        f"{CHUNK} sentences per Batch-API request, structured outputs constrained to the code lists.\n\n"
        "DAMSL (Phase A) is predicted first and used as context for the Beads (Phase B) prediction.\n\n"
        "## System prompt (verbatim)\n\n```\n" + SYSTEM + "\n```\n\n"
        "## User message template (per request)\n\n```\n" + USER_TEMPLATE + "\n```\n\n"
        "## Structured-output JSON schema\n\n```json\n" + json.dumps(SCHEMA, indent=2) + "\n```\n"
    )
    PROMPT_MD.write_text(md)
    print(f"[write] {PROMPT_MD}")


def load_sentences(limit):
    rows = []
    with MASTER.open(newline="") as f:
        for row in csv.DictReader(f):
            txt = (row.get("sentence_text") or "").strip()
            if txt:
                rows.append((row["sentence_id"], txt))
    return rows[:limit] if limit else rows


def build_requests(sentences):
    reqs = []
    for i in range(0, len(sentences), CHUNK):
        chunk = sentences[i:i + CHUNK]
        listing = "\n".join(f"[{sid}] {txt}" for sid, txt in chunk)
        reqs.append({
            "custom_id": f"c{i:06d}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "Tag these sentences:\n" + listing}],
                "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            },
        })
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only tag the first N sentences (smoke test)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between batch status polls")
    ap.add_argument("--prompt-only", action="store_true", help="just (re)write tagging_prompt.md and exit")
    ap.add_argument("--resume", metavar="BATCH_ID", help="reattach to an existing batch instead of creating one")
    args = ap.parse_args()

    write_prompt_md()
    if args.prompt_only:
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    sentences = load_sentences(args.limit)

    if args.resume:
        batch = client.messages.batches.retrieve(args.resume)
        print(f"[batch] resumed {batch.id}  status={batch.processing_status}")
    else:
        reqs = build_requests(sentences)
        print(f"[load] {len(sentences):,} sentences  →  {len(reqs):,} batch requests")
        batch = client.messages.batches.create(requests=reqs)
        print(f"[batch] created {batch.id}  status={batch.processing_status}")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(f"  {b.processing_status}: processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored}")
        time.sleep(args.poll)
    print(f"[batch] ended: succeeded={b.request_counts.succeeded} errored={b.request_counts.errored}")

    damsl_by_id: dict[str, list[str]] = {}
    beads_by_id: dict[str, list[str]] = {}
    errors = 0
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            errors += 1
            continue
        text = next((blk.text for blk in result.result.message.content if blk.type == "text"), "")
        try:
            for item in json.loads(text).get("results", []):
                sid = item.get("sentence_id")
                if not sid:
                    continue
                dset, bset = set(item.get("damsl", [])), set(item.get("beads", []))
                damsl_by_id[sid] = [c for c in DAMSL_CODES if c in dset]  # keep valid, in schema order
                beads_by_id[sid] = [c for c in BEADS_CODES if c in bset]
        except (json.JSONDecodeError, AttributeError):
            errors += 1

    if errors:
        print(f"[warn] {errors} request(s) failed or unparseable — those sentences default to no tags", file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "damsl_tags", "beads_tags", "n_damsl_tags", "n_beads_tags"])
        for sid, _ in sentences:
            d, bd = damsl_by_id.get(sid, []), beads_by_id.get(sid, [])
            w.writerow([sid, "; ".join(d), "; ".join(bd), len(d), len(bd)])

    dtag = sum(1 for sid, _ in sentences if damsl_by_id.get(sid))
    btag = sum(1 for sid, _ in sentences if beads_by_id.get(sid))
    print(f"[write] {OUT_CSV}  ({len(sentences):,} rows; {dtag:,} with DAMSL, {btag:,} with Beads)")


if __name__ == "__main__":
    main()
