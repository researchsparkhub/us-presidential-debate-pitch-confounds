# Speaker Identity or Communicative Style? Pitch, Prediction, and Speaker Confounds in U.S. Presidential Debates, 1988–2020

Code and derived data for the paper by Lavanya Prahallad and Radhika Mamidi
(LTRC, IIIT Hyderabad). This repository covers the data-processing pipeline
and analysis only; the paper itself is not part of this repository. The
exception is [`paper_supplement/`](paper_supplement/): the paper's
Electronic Supplementary Material (its four "Online Resource" files, cited
as such in the manuscript text), hosted here since they are journal
submission artifacts rather than pipeline code or data.

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
06_analysis/           Statistical analysis and figures
analysis_llm_extension/  Cross-modal LLM winner-prediction experiment
                          (transcript/descriptor conditions, contamination
                          probe, grounding check) and cached model outputs
data/
  metadata/            Debate/candidate/outcome table
  transcripts/         Official debate transcripts (plain text)
outputs/                Per-sentence derived feature tables and figures
```

## Getting the data

- **Already in this repo**: debate/candidate/outcome metadata
  (`data/metadata/debates.csv`), official transcripts (`data/transcripts/`),
  and every derived feature table the analysis actually runs on
  (`outputs/`). If you want to reproduce the *analysis*, you already have
  everything you need — skip straight to the "Reproducing the analysis"
  section below.
- **Not in this repo**: the raw debate audio. These are broadcast
  recordings we don't hold redistribution rights to. **See
  [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for exactly where to
  download each debate's recording (Commission on Presidential Debates and
  C-SPAN archive links, matched to the `debate_audio_id`/date columns in
  `debates.csv`) and where to place it if you want to reproduce the
  *acoustic pipeline from raw audio*.**

## Setup

```bash
git clone <this-repo-url>
cd debate_analysis
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The LLM-based stages (discourse tagging, cross-modal prediction) additionally
require an `ANTHROPIC_API_KEY` environment variable, and prompt files that
are not included in this repository — see `DATA_AVAILABILITY.md`.

## Reproducing the analysis

Everything from `data/metadata/debates.csv` and `outputs/` onward needs no
audio:

```bash
python 06_analysis/winner_loser_analysis.py   # the statistical results
python 06_analysis/generate_report_plots.py   # the figures
```

## Reproducing the full pipeline from raw audio

Each numbered directory is one pipeline stage; later stages read the
outputs of earlier ones. Stage 2 (forced alignment) requires the source
audio described in `DATA_AVAILABILITY.md` and a working Montreal Forced
Aligner installation.

```bash
python 01_ingest/build_debates_metadata.py
python 02_alignment/prepare_mfa_corpus.py   # requires audio, see DATA_AVAILABILITY.md
python 03_speakers/build_master_sentences.py
python 04_acoustic/extract_acoustic_features.py
python 05_linguistic/extract_linguistic_features.py
python 06_analysis/winner_loser_analysis.py
```

Tool versions and parameter settings for every stage are documented inline
in each script's module docstring.

## Not included

This repository is scoped to data and code only: the paper itself, its main
appendix material, and the discourse-annotation prompts/build project are
maintained elsewhere and are not part of this repository — the sole
exception is `paper_supplement/`, the four journal-submitted Online
Resource files (see above). An earlier, superseded draft of the analysis (a
different corpus scope, before it settled on the 24-debate window) is
likewise not included.

## License

Code is released under the MIT License (see `LICENSE`). The derived,
per-sentence feature tables in `outputs/` and `data/` are intended for
reuse with attribution; if you need an explicit data license for your use
case, contact the authors.

## Citation

<!--
DOI badge slot: once a release of this repository is archived on Zenodo
(https://zenodo.org/account/settings/github/ -> toggle this repo -> cut a
GitHub release), Zenodo mints a DOI and gives you a badge snippet to paste
here, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

If you use this code or data, please cite:

```bibtex
@article{prahallad2026speaker,
  author  = {Prahallad, Lavanya and Mamidi, Radhika},
  title   = {Speaker Identity or Communicative Style? Pitch, Prediction, and
             Speaker Confounds in {U.S.} Presidential Debates, 1988--2020},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/researchsparkhub/us-presidential-debate-pitch-confounds}}
}
```

A machine-readable citation is also provided in
[`CITATION.cff`](CITATION.cff) — GitHub uses this to populate the "Cite
this repository" button in the sidebar of this repo.
