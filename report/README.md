# DAP IEEE Report

Self-contained LaTeX source for the conference paper *Speaking to Win: A
Multimodal Acoustic and Linguistic Analysis of U.S. Presidential Debates,
1988–2024*.

## Files

| File | Purpose |
|---|---|
| `dap_report.tex` | main LaTeX source (IEEEtran conference template) |
| `refs.bib` | bibliography |
| `Makefile` | build automation |
| `figures/` | 5 figures, each in `.pdf` (vector) and `.png` |

## To build the PDF

Install a LaTeX distribution if you don't have one:

```bash
brew install --cask mactex-no-gui          # macOS
# or
sudo apt install texlive-full              # Linux
```

Then in this directory:

```bash
make            # produces dap_report.pdf
make view       # macOS: opens the PDF
make clean      # removes .aux/.log/.bbl etc.
```

## To regenerate the figures

```bash
make figures    # runs ../06_analysis/generate_report_plots.py
```

Figures are computed directly from the CSVs in `../outputs/`. If the
underlying analysis is re-run, regenerating the figures and rebuilding
will produce a refreshed paper.

## Figure list

| File | Description |
|---|---|
| `fig1_pipeline.pdf` | End-to-end pipeline overview |
| `fig2_lda.pdf` | LDA projection of openSMILE features (winners vs losers) |
| `fig3_effects.pdf` | Significant winner-vs-loser effect sizes |
| `fig4_pauses.pdf` | Pause distributions (winners vs losers) |
| `fig5_per_debate.pdf` | Per-election-cycle trends 1988–2024 |
