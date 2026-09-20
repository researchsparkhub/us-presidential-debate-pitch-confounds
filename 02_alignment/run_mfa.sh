#!/usr/bin/env bash
#
# Run Montreal Forced Aligner on the prepared corpus.
# Prerequisite: conda env `aligner` with MFA installed + english_us_arpa
#   acoustic model and dictionary downloaded.
#
# By default, aligns ALL debates in mfa_corpus/. Pass debate IDs as args to
# limit to a pilot subset:
#     ./run_mfa.sh                  # everything
#     ./run_mfa.sh 1010 1012 1002   # just these debates
#
# Output: TextGrid files in mfa_output/<debate_id>/<debate_id>.TextGrid

set -euo pipefail

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate aligner

CORPUS_DIR="mfa_corpus"
OUTPUT_DIR="mfa_output"

# Optional: filter to a pilot subset
if [[ $# -gt 0 ]]; then
  PILOT_DIR="mfa_corpus_pilot"
  rm -rf "$PILOT_DIR" && mkdir -p "$PILOT_DIR"
  for did in "$@"; do
    if [[ -d "$CORPUS_DIR/$did" ]]; then
      ln -s "../$CORPUS_DIR/$did" "$PILOT_DIR/$did"
      echo "[pilot] including $did"
    else
      echo "[skip ] $did not in $CORPUS_DIR"
    fi
  done
  CORPUS_DIR="$PILOT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# --beam / --retry_beam wider than default to accommodate paraphrased
# transcripts AND long debate audio (~90 min). The pilot showed the default
# beam (100/400) timed out on 1003 after 46 min; 400/1000 succeeds.
# --single_speaker forces parallelism across utterances even though each
# debate is one "speaker" folder, so 4 jobs actually run in parallel.
mfa align \
  "$CORPUS_DIR" \
  english_us_arpa \
  english_us_arpa \
  "$OUTPUT_DIR" \
  --clean \
  --num_jobs 4 \
  --single_speaker \
  --beam 400 \
  --retry_beam 1000

echo
n_done=$(find "$OUTPUT_DIR" -name "*.TextGrid" | wc -l | tr -d ' ')
echo "Done. $n_done TextGrid(s) in $OUTPUT_DIR/"
find "$OUTPUT_DIR" -name "*.TextGrid" -exec basename {} \; | sort
