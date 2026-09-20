#!/usr/bin/env python3
"""
Build per-debate annotation-verification files — v2 (LLM-tagged).

Same format as v1, but BOTH phases are pre-filled from the LLM tagger
(outputs/llm_tags.csv, produced by 05_linguistic/tag_llm.py):
    Phase A (DAMSL)  <- llm_tags.damsl_tags
    Phase B (Beads)  <- llm_tags.beads_tags

One CSV per debate in debates_v2/, replicated across N annotators, each with a
notes column per phase.

Run 05_linguistic/tag_llm.py first to produce outputs/llm_tags.csv.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MASTER = PROJECT / "outputs" / "master_sentences.csv"
LLM = PROJECT / "outputs" / "llm_tags.csv"
OUTDIR = HERE / "debates_v2"

N_ANNOTATORS = 3

if not LLM.exists():
    sys.exit(f"Missing {LLM}. Run: python 05_linguistic/tag_llm.py")

OUTDIR.mkdir(parents=True, exist_ok=True)

# --- load LLM tags (already semicolon-joined) --------------------------------
llm = {}
with LLM.open(newline="") as f:
    for row in csv.DictReader(f):
        llm[row["sentence_id"]] = (row.get("damsl_tags", ""), row.get("beads_tags", ""))

# --- group master sentences by debate ----------------------------------------
debates = {}
with MASTER.open(newline="") as f:
    for row in csv.DictReader(f):
        debates.setdefault(row["debate_id"], []).append(row)

# --- header ------------------------------------------------------------------
header = ["unit_id", "speaker", "text"]
for a in range(1, N_ANNOTATORS + 1):
    header += [
        f"Annotator {a} - Phase A (DAMSL)",
        f"Annotator {a} - Phase A notes",
        f"Annotator {a} - Phase B (Beads)",
        f"Annotator {a} - Phase B notes",
    ]

total = 0
for debate_id, rows in sorted(debates.items()):
    out = OUTDIR / f"{debate_id}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(rows, start=1):
            damsl, beads = llm.get(row["sentence_id"], ("", ""))
            line = [i, row["speaker_raw"], row["sentence_text"]]
            for _ in range(N_ANNOTATORS):
                line += [damsl, "", beads, ""]
            w.writerow(line)
    total += 1
    print(f"{out.name:12} {len(rows):>5} sentences")

print(f"\nWrote {total} debate files to {OUTDIR}")
