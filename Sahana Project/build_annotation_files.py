#!/usr/bin/env python3
"""
Build per-debate annotation-verification files for the Sahana Project.

For each debate, emit one CSV (sentence-level) with:
    unit_id, speaker, text,
    Annotator N - Phase A (DAMSL), Annotator N - Phase B (Beads), ...
    Annotator N - Phase A notes, Annotator N - Phase B notes, ...

Phase A columns are PRE-FILLED with the model's existing DAMSL tags.
Phase B columns are PRE-FILLED with the model's existing Beads/bias tags.
Each annotator gets their own copy of the pre-filled tags to verify/correct,
plus a notes column to justify a change or propose a new tag.
"""
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MASTER = PROJECT / "outputs" / "master_sentences.csv"
TAGS = PROJECT / "outputs" / "damsl_bias_tags.csv"
OUTDIR = Path(__file__).resolve().parent / "debates"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_ANNOTATORS = 3  # change to 2 if you only have two verifiers


def code(colname: str) -> str:
    """damsl_S_Inform -> S-Inform ; bias_AF -> AF"""
    stem = colname.split("_", 1)[1]
    return stem.replace("_", "-")


# --- load tags, collapse one-hot -> semicolon-joined labels ------------------
tag_rows = {}
with TAGS.open(newline="") as f:
    r = csv.DictReader(f)
    damsl_cols = [c for c in r.fieldnames if c.startswith("damsl_")]
    bias_cols = [c for c in r.fieldnames if c.startswith("bias_")]
    for row in r:
        sid = row["sentence_id"]
        damsl = "; ".join(code(c) for c in damsl_cols if row[c] == "1")
        bias = "; ".join(code(c) for c in bias_cols if row[c] == "1")
        tag_rows[sid] = (damsl, bias)

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
            sid = row["sentence_id"]
            damsl, bias = tag_rows.get(sid, ("", ""))
            line = [i, row["speaker_raw"], row["sentence_text"]]
            for _ in range(N_ANNOTATORS):
                line += [damsl, "", bias, ""]
            w.writerow(line)
    total += 1
    print(f"{out.name:12} {len(rows):>5} sentences")

print(f"\nWrote {total} debate files to {OUTDIR}")
