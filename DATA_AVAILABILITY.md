# Data Availability

## What is included

- `data/metadata/debates.csv` — every debate: date, candidates, party,
  and the outcome (whether that candidate's ticket won the election that
  followed).
- `data/transcripts/` — official debate transcripts, plain text, one file
  per debate.
- `outputs/` — per-sentence derived feature tables (linguistic, semantic,
  discourse, acoustic) and the figures used in the paper. These are the
  measurements the paper's analysis actually runs on.
- Every processing script used to go from raw audio and transcript to
  these tables (`01_ingest/` through `06_analysis/`).

## What is not included

**Raw and intermediate audio.** The debates are broadcast recordings
(network and C-SPAN footage). We do not hold redistribution rights to
this material, and it is not included in this repository, consistent with
the data statement in the paper's Appendix I.

To reproduce the acoustic pipeline from scratch, you will need to source
the audio yourself. Official transcripts and video for U.S. general-election
presidential debates are archived by the
[Commission on Presidential Debates](https://www.debates.org/voter-education/debate-transcripts/)
and by [C-SPAN](https://www.c-span.org/debates/). `data/metadata/debates.csv`
gives the exact date and candidates for each debate in the corpus, which is
enough to locate the corresponding recording.

Once you have the audio, place each file at `data/audio/<debate_id>.wav`
(IDs match `data/metadata/debates.csv`) and the alignment pipeline in
`02_alignment/` will run as documented in the top-level `README.md`.

## Questions

If you are a reviewer, replicator, or would like access to the underlying
audio for research purposes, contact the corresponding author
(see `report/`).
