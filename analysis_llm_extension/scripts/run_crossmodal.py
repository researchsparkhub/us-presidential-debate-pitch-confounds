#!/usr/bin/env python3
"""
Method 4 (LLM cross-modal reasoning) + explainability/grounding (paper
extension items #7 and #4) + the contamination probe promised in the main
paper's Future Work section.

For every one of the 24 debates in llm_ext_cache/, and for 3 repeats with an
independently randomised Candidate-A/B/[C] letter assignment per repeat, the
model is asked which candidate went on to win that year's presidential
election, under three conditions:
  transcript   - anonymised candidate-turn transcript only
  descriptors  - the same 34 standardised (within-debate z-scored) measures
                 used by Methods 1-3, as a table, no text at all
  both         - transcript + descriptor table together

Every prediction call also asks for a short explanation and a list of cited
evidence (a controlled vocabulary, not free text), so the explanation can be
checked programmatically against the real data (grounding) or scanned for
fabricated numeric claims (transcript-only condition, which is given no
numbers at all).

A separate, single-shot contamination probe (temperature 0, fresh context,
no prediction question) asks the model to identify the debate from (a) the
anonymised transcript and (b) the descriptor table alone.

Model: claude-opus-4-8, via the Anthropic Messages API, structured
(json_schema) outputs. See crossmodal_prompt.md for the exact prompts.

Checkpointed to JSONL; safe to re-run (skips completed rows).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_crossmodal.py                # full run / resume
    python run_crossmodal.py --debates 1001 1003 --workers 4   # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE.parent / "cache" / "llm_ext_cache"
PROMPT_MD = HERE.parent.parent / "Sahana Project" / "crossmodal_prompt.md"
PRED_OUT = HERE.parent / "results" / "crossmodal_predictions.jsonl"
PROBE_OUT = HERE.parent / "results" / "crossmodal_contamination.jsonl"

MODEL = "claude-opus-4-8"
N_REPEATS = 3
TEMPERATURE_MAIN = 1.0
TEMPERATURE_PROBE = 0.0
MAX_TOKENS_PRED = 1024
MAX_TOKENS_PROBE = 512

from llm_ext_data import FRIENDLY_NAMES  # 34 measure names, canonical order

THEMES = [
    "policy substance", "confidence or composure", "clarity or articulation",
    "emotional appeal", "attacks on opponent", "specific facts or numbers",
    "pacing or delivery", "likability or warmth", "other",
]

SYSTEM_COMMON = (
    "You are analysing an anonymised transcript or set of speech-measurement "
    "descriptors from a U.S. presidential general-election debate between two "
    "or three candidates. All candidate names, party labels, and the debate's "
    "date have been removed and replaced with neutral letters (Candidate A, "
    "Candidate B, and Candidate C if present).\n\n"
    "Your task: judge which candidate went on to WIN that year's presidential "
    "election. Base your judgement only on the material given to you in this "
    "message. Do not rely on outside knowledge of real debates, candidates, "
    "or elections, even if the content seems familiar --- reason only from "
    "what is shown here."
)

TASK_TAIL = (
    "\n\nReturn your single best guess as exactly one candidate letter, a short "
    "(1-3 sentence) explanation, and a short list of the specific evidence "
    "that most influenced your judgement, drawn only from the controlled list "
    "given in the schema."
)


def letters_for(n):
    return ["A", "B", "C"][:n]


def pred_schema(letters, mode):
    """mode: 'measures' (descriptor table available) or 'themes' (text only)."""
    if mode == "measures":
        cited_item_enum = FRIENDLY_NAMES + ["other"]
    else:
        cited_item_enum = THEMES
    return {
        "type": "object",
        "properties": {
            "prediction": {"type": "string", "enum": letters},
            "explanation": {"type": "string"},
            "cited_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "enum": cited_item_enum},
                        "favors": {"type": "string", "enum": letters + ["similar"]},
                    },
                    "required": ["item", "favors"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["prediction", "explanation", "cited_evidence"],
        "additionalProperties": False,
    }


PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "recognized": {"type": "boolean"},
        "guessed_year": {"type": ["string", "null"]},
        "guessed_candidates": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["recognized", "guessed_year", "guessed_candidates", "reasoning"],
    "additionalProperties": False,
}

PROBE_SYSTEM = (
    "You are shown an anonymised excerpt from a U.S. presidential debate (names, "
    "party, and date removed). Try to identify which specific real debate this "
    "is, using any cues in the content or phrasing --- word choice, topics, "
    "policy references, quoted lines, writing style, or numeric patterns. Report "
    "your best guess of the year and both/all candidates' real names even if you "
    "are not fully certain; set recognized=true only if you are reasonably "
    "confident, and explain what specific cue led you there."
)


def format_transcript(debate, letter_map):
    lines = []
    for t in debate["transcript_turns"]:
        letter = letter_map[t["speaker"]]
        lines.append(f"Candidate {letter}: {t['text']}")
    return "\n".join(lines)


def format_descriptors(debate, letter_map):
    cands = sorted(letter_map, key=lambda c: letter_map[c])
    lines = ["Standardised measures (z-scores within this debate; 0 = debate "
             "average, positive = above the debate average, negative = below):"]
    for c in cands:
        letter = letter_map[c]
        vals = debate["descriptors"][c]
        parts = [f"{name}={vals[name]:+.2f}" if vals[name] is not None else f"{name}=NA"
                 for name in FRIENDLY_NAMES]
        lines.append(f"\nCandidate {letter}:\n  " + ", ".join(parts))
    return "\n".join(lines)


def build_pred_messages(debate, letter_map, condition):
    letters = sorted(set(letter_map.values()))
    parts = []
    if condition in ("transcript", "both"):
        parts.append("ANONYMISED TRANSCRIPT (candidate turns, in order):\n\n"
                      + format_transcript(debate, letter_map))
    if condition in ("descriptors", "both"):
        parts.append(format_descriptors(debate, letter_map))
    parts.append(TASK_TAIL.strip())
    mode = "themes" if condition == "transcript" else "measures"
    return "\n\n".join(parts), pred_schema(letters, mode), mode


def build_probe_messages(debate, letter_map, condition):
    parts = []
    if condition == "transcript":
        parts.append("ANONYMISED TRANSCRIPT (candidate turns, in order):\n\n"
                      + format_transcript(debate, letter_map))
    else:
        parts.append(format_descriptors(debate, letter_map))
    return "\n".join(parts)


def load_done(path, key_fields):
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(tuple(r[k] for k in key_fields))
    return done


def call_structured(client, system, user, schema, temperature, max_tokens):
    # Note: this model deprecates the `temperature` parameter (server returns
    # a 400 if set) --- sampling variability across repeats comes from the
    # API's own default, not a caller-controlled temperature. Documented as
    # such in the reproducibility appendix.
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if system:
        kwargs["system"] = [{"type": "text", "text": system}]
    resp = client.messages.create(**kwargs)
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


def write_prompt_md():
    PROMPT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Cross-modal LLM prediction prompt --- Method 4 (crossmodal extension)

Generated by `run_crossmodal.py`. Model: **{MODEL}**, Anthropic Messages API,
structured (`json_schema`) outputs, temperature **{TEMPERATURE_MAIN}** for the
prediction task ({N_REPEATS} repeats per debate per condition, each with an
independently randomised Candidate-A/B/[C] letter assignment), temperature
**{TEMPERATURE_PROBE}** for the single-shot contamination probe.

Three conditions per debate: `transcript` (anonymised candidate turns only),
`descriptors` (the 34 within-debate z-scored measures from Methods 1-3, as a
table, no text), `both`.

## System / task prompt (prediction task, verbatim)

```
{SYSTEM_COMMON}
```//task tail//
```
{TASK_TAIL}
```

## Contamination probe prompt (verbatim)

```
{PROBE_SYSTEM}
```

## Controlled evidence vocabulary

Descriptor conditions (`descriptors`, `both`) cite from the 34 measure names
used throughout the paper (Table tab:loadings / names.json). Text-only
condition (`transcript`) cites from a separate qualitative vocabulary:

```
{json.dumps(THEMES, indent=2)}
```

## Prediction JSON schema (descriptor-available conditions; letters vary 2-3
per debate)

```json
{json.dumps(pred_schema(["A", "B"], "measures"), indent=2)}
```

## Prediction JSON schema (transcript-only condition)

```json
{json.dumps(pred_schema(["A", "B"], "themes"), indent=2)}
```

## Contamination-probe JSON schema

```json
{json.dumps(PROBE_SCHEMA, indent=2)}
```
"""
    PROMPT_MD.write_text(md)
    print(f"[write] {PROMPT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-pred", action="store_true")
    args = ap.parse_args()

    write_prompt_md()
    client = anthropic.Anthropic()

    all_ids = sorted(
        (p.stem for p in CACHE_DIR.glob("*.json") if p.stem != "_manifest"),
        key=int,
    )
    debate_ids = args.debates if args.debates else all_ids
    debates = {d: json.loads((CACHE_DIR / f"{d}.json").read_text()) for d in debate_ids}
    print(f"[load] {len(debates)} debates")

    rng = random.Random(20260806)  # fixed seed: reproducible letter assignments

    # ---- build job list: (debate_id, repeat, condition) ----
    letter_maps = {}  # (debate_id, repeat) -> {candidate_name: letter}
    for d, debate in debates.items():
        for rep in range(N_REPEATS):
            cands = list(debate["candidates"])
            rng.shuffle(cands)
            letter_maps[(d, rep)] = {c: L for c, L in zip(cands, letters_for(len(cands)))}

    pred_lock = threading.Lock()
    probe_lock = threading.Lock()

    def run_pred(d, rep, condition):
        debate = debates[d]
        lm = letter_maps[(d, rep)]
        user, schema, mode = build_pred_messages(debate, lm, condition)
        out = call_structured(client, SYSTEM_COMMON, user, schema, TEMPERATURE_MAIN, MAX_TOKENS_PRED)
        inv_lm = {v: k for k, v in lm.items()}
        row = {
            "debate_id": d, "repeat": rep, "condition": condition,
            "letter_map": lm, "winner_candidate": debate["winner"],
            "winner_letter": lm[debate["winner"]],
            "predicted_letter": out.get("prediction"),
            "predicted_candidate": inv_lm.get(out.get("prediction")),
            "explanation": out.get("explanation"),
            "cited_evidence": out.get("cited_evidence"),
            "mode": mode,
        }
        with pred_lock:
            with PRED_OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def run_probe(d, condition):
        debate = debates[d]
        lm = letter_maps[(d, 0)]  # any mapping works; probe doesn't use letters as ground truth
        user = build_probe_messages(debate, lm, condition)
        out = call_structured(client, PROBE_SYSTEM, user, PROBE_SCHEMA, TEMPERATURE_PROBE, MAX_TOKENS_PROBE)
        row = {"debate_id": d, "condition": condition, "year": debate["year"],
               "real_candidates": debate["candidates"], **out}
        with probe_lock:
            with PROBE_OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    jobs = []
    if not args.skip_pred:
        done_pred = load_done(PRED_OUT, ["debate_id", "repeat", "condition"])
        for d in debates:
            for rep in range(N_REPEATS):
                for cond in ("transcript", "descriptors", "both"):
                    if (d, rep, cond) not in done_pred:
                        jobs.append(("pred", d, rep, cond))
    probe_jobs = []
    if not args.skip_probe:
        done_probe = load_done(PROBE_OUT, ["debate_id", "condition"])
        for d in debates:
            for cond in ("transcript", "descriptors"):
                if (d, cond) not in done_probe:
                    probe_jobs.append(("probe", d, cond))

    print(f"[plan] {len(jobs)} prediction calls, {len(probe_jobs)} probe calls to do")

    total = len(jobs) + len(probe_jobs)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for kind, d, a, *rest in jobs:
            futs[ex.submit(run_pred, d, a, rest[0])] = ("pred", d, a, rest[0])
        for kind, d, cond in probe_jobs:
            futs[ex.submit(run_probe, d, cond)] = ("probe", d, cond)
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[warn] {tag} failed: {e}")
                continue
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  {completed}/{total} done")

    print("[done]")


if __name__ == "__main__":
    main()
