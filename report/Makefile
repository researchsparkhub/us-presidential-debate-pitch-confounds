# Makefile for the DAP IEEE report.
#
# Requires: a working LaTeX distribution (e.g. MacTeX, TeX Live, MiKTeX).
# On macOS:   brew install --cask mactex-no-gui
# On Linux:   sudo apt install texlive-full
#
# Then:       make            # build dap_report.pdf
#             make clean      # remove intermediate files
#             make figures    # regenerate figures from CSVs

PAPER := dap_report
PYTHON := python3

.PHONY: all clean figures view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib figures/fig1_pipeline.pdf
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	@echo "Built $(PAPER).pdf"

figures:
	$(PYTHON) ../06_analysis/generate_report_plots.py

view: $(PAPER).pdf
	open $(PAPER).pdf

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out \
	      $(PAPER).toc $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf
