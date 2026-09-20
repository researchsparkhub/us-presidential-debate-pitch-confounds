# Sahana Project — DAMSL + Beads tag verification

Human verification of the rule-based DAMSL (discourse) and Beads (bias) tags.
Each debate is one file, verified independently by 3 annotators across two phases:

- **Phase A = DAMSL** (discourse act, e.g. `S`, `Q-W`, `CH`)
- **Phase B = Beads** (bias, e.g. `PER`, `ATTR`, `AF`)

The annotator sees the model's existing tag, pre-filled, and **verifies** it: keep,
correct (with a note why), add a missing tag, or **propose a new tag**.

## What's here

| File / folder | Purpose |
|---|---|
| `debates/` | **v1** — 26 CSVs, Phase A from rule-based DAMSL, Phase B from rule-based Beads. |
| `debates_v2/` | **v2** — 26 CSVs, both phases pre-filled from the LLM tagger (recommended). |
| `codebook.csv` | All DAMSL + Beads codes with definitions (also embedded in the importer). |
| `tagging_prompt.md` | The exact prompt + schema used by the LLM tagger (v2). |
| `import_to_sheets.gs` | Google Apps Script that turns the CSVs into formatted Google Sheets. |
| `build_annotation_files.py` | Regenerates `debates/` (v1, rule-based) from `outputs/`. |
| `build_annotation_files_v2.py` | Regenerates `debates_v2/` from `outputs/llm_tags.csv`. |
| `inter_annotator_agreement.py` | Computes agreement (exact-match, Jaccard, Krippendorff α, per-tag Fleiss κ) from verified CSVs in `verified/`. |
| `verified/` | Drop the annotators' completed sheets here (File → Download → CSV) to run agreement. |

## Measuring agreement (after verification)

Once annotators have verified their sheets: in Google Sheets do **File → Download → CSV** for each
debate and drop the files into `Sahana Project/verified/`, then:

```bash
python "Sahana Project/inter_annotator_agreement.py"
```

It reports, per phase (DAMSL / Beads): exact-set-match %, mean Jaccard, **Krippendorff's α**
(chance-corrected reliability using MASI set distance), and **per-tag Fleiss κ** (which codes
annotators disagree on). It writes `agreement/summary.csv`, `agreement/per_tag_kappa_phase_*.csv`,
and `agreement/disagreements_phase_*.csv` (sentences to adjudicate). By default it only scores
rows where ≥2 annotators tagged a phase; add `--include-empty` to also credit agreement-on-empty.

## v2 — LLM-tagged Phase A & B (recommended)

Both phases are pre-filled by **Claude Opus 4.8** (`05_linguistic/tag_llm.py`). The model
tags **DAMSL first, then conditions the Beads prediction on those discourse acts** — one
Batch-API pass, multi-label, constrained to the valid codes. This reaches the codes the
rule-based tagger can never detect (INT, TA; GB, GD, CB, IA, SE, CBias). The exact prompt is
saved to `tagging_prompt.md`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python 05_linguistic/tag_llm.py --limit 50   # smoke test first (~1 min)
python 05_linguistic/tag_llm.py              # full run → outputs/llm_tags.csv (async batch)
python "Sahana Project/build_annotation_files_v2.py"   # → debates_v2/
```

Then import to Sheets with `INPUT_SUBFOLDER = 'debates_v2'` and
`OUTPUT_FOLDER = 'Sahana Debates - Sheets v2'` in `import_to_sheets.gs` (keeps v1 untouched).

### File format (per debate)

```
unit_id | speaker | text | A1 Phase A | A1 Phase A notes | A1 Phase B | A1 Phase B notes | A2 ... | A3 ...
```

Multiple tags in one cell are separated by `; ` (e.g. `S; ATTR`), since a sentence
can carry several tags.

## Recommended workflow: Google Sheets (not emailed CSVs)

Emailing CSVs breaks down fast: no code validation, no version control, and a painful
merge of 26 files × 3 annotators. Google Sheets fixes all three and needs no install.

**One-time setup (you, ~5 min):**

1. In Google Drive, create a folder **`Sahana Debates - CSV`** and drag the whole
   local `debates/` folder into it (Drive keeps it as a `debates` subfolder — that's fine,
   the script looks there by default via `INPUT_SUBFOLDER = 'debates'`). If instead you move
   the 26 CSVs directly into `Sahana Debates - CSV`, set `INPUT_SUBFOLDER = ''`.
2. Open <https://script.google.com> → **New project** → paste `import_to_sheets.gs`.
3. Run `importAll()` and approve the permission prompt.
   - Creates a folder **`Sahana Debates - Sheets`** with one formatted Sheet per debate:
     frozen header + context columns, a locked **Codebook** tab, a **READ ME** tab,
     dropdown validation on the tag columns, and warning-protected source columns.
4. Right-click the `Sahana Debates - Sheets` folder → **Share** → add the annotators as
   **Editor**. Each annotator edits only their own colour-tinted columns.

**Annotators:** open a debate's Sheet, work down the Annotation tab, verify each pre-filled
tag. Codes are on the Codebook tab. Codes like `INT`, `TA`, `GB`, `GD`, `CB`, `IA`, `SE`,
`CBias` are never auto-filled (they need human judgement) — add them where they apply.

## Getting the data back for analysis

**File → Download → CSV** on each verified Sheet, drop them in a folder, and compare the
three annotators' columns for inter-annotator agreement. (A merge/agreement script can be
added once the format is confirmed.)

### Note on agreement metrics

The current side-by-side layout lets annotators see each other's and the model's tags, so
apparent agreement is inflated by *anchoring*. That's fine for efficient verification; if
you later need a defensible inter-annotator agreement number, give each annotator a blind
copy (no other columns visible) and re-run.
