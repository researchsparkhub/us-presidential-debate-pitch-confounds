# Data Availability

## What is included in this repository

| Path | Contents |
|---|---|
| `data/metadata/debates.csv` | One row per candidate per debate: `debate_audio_id`, `debate_id`, `year`, `debate_type` (Presidential / Vice Presidential), `debate_number`, `date`, `speaker`, `party`, `election_winner`, `winning_party`, `result`. |
| `data/transcripts/<debate_audio_id>.txt` | Official debate transcript, plain text. |
| `outputs/*.csv`, `outputs/*.parquet` | Per-sentence derived feature tables (linguistic, semantic, discourse, acoustic) that the paper's analysis runs on directly. |
| `01_ingest/` … `06_analysis/`, `analysis_llm_extension/`, `Sahana Project/` | Every script used to go from raw audio + transcript to the tables above. |

The paper's core analysis (24 debates, 1988–2020) uses only the rows where
`debate_type = Presidential`; `debates.csv` also lists Vice Presidential
debates and 2024, which are part of the broader ingested metadata but not
part of the analytical corpus (see `report/`, Section on Corpus Scope).

## What is not included, and why

**Raw and intermediate audio is not included.** These are broadcast
recordings of televised debates; we do not hold redistribution rights, and
the paper's own data statement (Appendix I) states this explicitly. This
repository's `.gitignore` excludes every `.wav`/`.m4a`/`.mp3`/`.mp4` file
and the intermediate alignment working directories under `02_alignment/`
that would otherwise contain them.

## How to obtain the audio and reproduce the acoustic pipeline

1. **Identify each debate.** `data/metadata/debates.csv` gives the exact
   date and debate type for every `debate_audio_id` (e.g. `1001` =
   Presidential debate 1, 1988-09-25, Dukakis vs. G.H.W. Bush).
2. **Source the recording.** Two public archives cover every debate in
   this corpus:
   - Commission on Presidential Debates transcripts and video index:
     <https://www.debates.org/voter-education/debate-transcripts/>
   - C-SPAN Debate Archive (searchable by date):
     <https://www.c-span.org/debates/>
   Search by the date in `debates.csv`; both sites index every
   general-election presidential debate since 1988.
3. **Place the audio.** Extract or convert the recording to a mono WAV and
   save it as `data/audio/<debate_audio_id>.wav` (matching the ID column in
   `debates.csv`). `data/audio/` is gitignored, so this step happens on
   your own machine after cloning.
4. **Run the pipeline** from `data/metadata/debates.csv` onward, following
   the sequence in the top-level `README.md`. Stage 2
   (`02_alignment/prepare_mfa_corpus.py`, `run_mfa.sh`) is the first stage
   that touches audio and requires a working
   [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
   installation (see Appendix C of the paper for the exact acoustic model
   and dictionary used).

If you only need the derived measurements the paper's statistics run on —
not the audio itself — everything you need is already in `outputs/`; you
do not need to source the audio at all.

## LLM-based stages

`05_linguistic/tag_llm.py`, `analysis_llm_extension/scripts/run_crossmodal.py`,
and `analysis_llm_extension/scripts/run_cot_tagging.py` call the Anthropic
API and expect an `ANTHROPIC_API_KEY` environment variable. No key is
included in this repository. The exact model, prompts and schemas used are
specified in the paper's Appendix G and in `Sahana Project/`.

## Questions

If you are a reviewer, replicator, or would like access to the underlying
audio for research purposes, contact the corresponding author
(see `report/`).
