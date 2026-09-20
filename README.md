# Speaker Identity or Communicative Style? Pitch, Prediction, and Speaker Confounds in U.S. Presidential Debates, 1988–2020

Code and derived data for the paper by Lavanya Prahallad and Radhika Mamidi
(LTRC, IIIT Hyderabad). The paper is in `report/`.

The study asks whether linguistic and vocal properties distinguish
candidates whose tickets went on to win a U.S. presidential election from
those whose tickets did not, across every televised general-election
debate from 1988 to 2020 (24 debates, 24,193 candidate sentences). One
acoustic property — a specific feature of the pitch contour — survives
every validity check we run. It does not generalize to an unseen debate,
and three classical models plus a large language model all appear to
predict the winner for the same reason: they are detecting who is
speaking, or which debate this is, not how winners communicate.

## Repository layout

```
01_ingest/            Build debate/candidate/outcome metadata
02_alignment/          Audio–text alignment (Whisper + Montreal Forced Aligner)
03_speakers/           Speaker-label normalisation, master sentence table
04_acoustic/           Acoustic feature extraction (Praat, openSMILE)
05_linguistic/         Linguistic/semantic/discourse feature extraction,
                       DAMSL + BEADS discourse tagging (rule-based and LLM)
06_analysis/           Statistical analysis and report figures
analysis_llm_extension/  Cross-modal LLM winner-prediction experiment
                          (transcript/descriptor conditions, contamination
                          probe, grounding check) and cached model outputs
Sahana Project/         Discourse-annotation build scripts, prompts, codebook,
                        and inter-annotator-agreement analysis (cited by the
                        paper's Appendix G and E)
data/
  metadata/            Debate/candidate/outcome table
  transcripts/         Official debate transcripts (plain text)
outputs/                Per-sentence feature tables and figures used in the paper
report/                 The paper (journal and IEEE-style versions), figures,
                        bibliography
```

## What's not included, and why

**Audio is not redistributed.** The debates are broadcast recordings; we do
not have redistribution rights, and this repository does not include them.
See `DATA_AVAILABILITY.md` for what that means for reproducing the acoustic
pipeline from scratch, and where to source the recordings yourself.

An earlier, superseded draft of this work (a different corpus scope, before
the analysis settled on the 24-debate window) is not included here to avoid
two versions of the paper circulating with different numbers.

## Reproducing the pipeline

Each numbered directory is one pipeline stage; later stages read the
outputs of earlier ones. Stage 2 (forced alignment) requires the source
audio described in `DATA_AVAILABILITY.md` and a working Montreal Forced
Aligner installation. From `data/metadata/debates.csv` onward:

```bash
python 01_ingest/build_debates_metadata.py
python 02_alignment/prepare_mfa_corpus.py   # requires audio, see DATA_AVAILABILITY.md
python 03_speakers/build_master_sentences.py
python 04_acoustic/extract_acoustic_features.py
python 05_linguistic/extract_linguistic_features.py
python 06_analysis/winner_loser_analysis.py
```

Tool versions and parameter settings for every stage are listed in the
paper's Appendix C; the LLM prompts and schemas used for discourse tagging
and cross-modal prediction are in `Sahana Project/` and
`analysis_llm_extension/`, and specified in full in Appendix G.

## The paper

`report/debate_analysis_report_journal.tex` is the current, actively
maintained version (Springer Nature journal format); build with
`pdflatex` + `bibtex` (see `report/Makefile`). `report/dap_report.tex` is
an earlier report frozen for historical reference.

## License

Code is released under the MIT License (see `LICENSE`). The derived,
per-sentence feature tables in `outputs/` and `data/` are intended for
reuse with attribution; if you need an explicit data license for your use
case, contact the authors.

## Citation

If you use this code or data, please cite the paper (full citation in
`report/`).
