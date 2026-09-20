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
4. **Generate the Whisper alignment.** Several stage-2 scripts
   (`whisper_canonical.py`, `align_via_whisper.py`, `trim_and_align_2024.py`)
   read word-level timing from `data/whisper_aligned/<debate_audio_id>.json`.
   This repository does not include a script that produces that file, since
   it depends on the audio above; generate it yourself with
   [Whisper](https://github.com/openai/whisper) or
   [WhisperX](https://github.com/m-bain/whisperX) (`openai-whisper` is listed
   in `requirements.txt`) and write its `word_segments` output (each with
   `start`, `end`, `word`) to that path.
5. **Install Montreal Forced Aligner.** `02_alignment/run_mfa.sh` and the
   `retry_*.sh` scripts assume a conda environment named exactly `aligner`
   with [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
   installed, using the `english_us_arpa` acoustic model and pronunciation
   dictionary (see Appendix C of the paper):
   ```bash
   conda create -n aligner -c conda-forge montreal-forced-aligner
   conda activate aligner
   mfa model download acoustic english_us_arpa
   mfa model download dictionary english_us_arpa
   ```
   `retry_v3_final.sh` additionally uses macOS's `caffeinate` to prevent
   sleep during long runs; on Linux, drop that wrapper and run the inner
   command directly.
6. **Run the pipeline** from `data/metadata/debates.csv` onward, following
   the sequence in the top-level `README.md`.

If you only need the derived measurements the paper's statistics run on —
not the audio itself — everything you need is already in `outputs/`; you
do not need to source the audio, run Whisper, or install MFA at all.

## Other external inputs

- **2024 speaker diarization.** `03_speakers/build_2024_speaker_mapping.py`
  and `build_master_sentences.py` read a diarization CSV at
  `data/diarization/segments_raw.csv` (columns: `debate_audio_id`,
  `start_sec`, `end_sec`, `diarized_speaker`), produced with
  [pyannote.audio](https://github.com/pyannote/pyannote-audio) (listed in
  `requirements.txt`) for the two 2024 debates only. This is not needed to
  reproduce the paper's core 1988–2020 analysis.
- **Debate inventory spreadsheet.** `01_ingest/build_debates_metadata.py`
  builds `data/metadata/debates.csv` from a source spreadsheet. Since
  `debates.csv` is already included in this repository, you do not need
  this input unless you are rebuilding the metadata from scratch; if you
  are, point the script at your spreadsheet with the
  `DEBATE_INVENTORY_XLSX` environment variable.

## LLM-based stages

`05_linguistic/tag_llm.py`, `analysis_llm_extension/scripts/run_crossmodal.py`,
and `analysis_llm_extension/scripts/run_cot_tagging.py` call the Anthropic
API and expect an `ANTHROPIC_API_KEY` environment variable. No key is
included in this repository. The exact model, prompts and schemas used are
specified in the paper's Appendix G and in `Sahana Project/`.

## `analysis_llm_extension/` layout

Scripts live in `analysis_llm_extension/scripts/`; the data they read and
write lives in the sibling directories `analysis_llm_extension/results/`
(JSON/CSV results already shipped in this repo) and
`analysis_llm_extension/cache/llm_ext_cache/` (cached per-debate API
responses). Running a script from a different working directory is fine —
each script resolves these paths relative to its own location, not the
current directory.

## Questions

If you are a reviewer, replicator, or would like access to the underlying
audio for research purposes, contact the corresponding author
(see `report/`).
