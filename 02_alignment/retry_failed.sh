#!/usr/bin/env bash
#
# Retry MFA alignment on the 6 failed debates in parallel.
#
# Each debate runs in its OWN MFA working directory (so hung jobs can't share
# state) and has a hard watchdog timeout. Up to 3 debates run concurrently
# (configurable via PARALLEL).
#
# Run:
#     ./retry_failed.sh
# Logs:
#     retry_logs/retry_<id>.log
# Outputs (if alignment succeeds):
#     mfa_output/<id>/<id>.TextGrid

set -uo pipefail
cd "$(dirname "$0")"

FAILED_DEBATES=("1001" "1007" "1008" "1019" "1034" "1036")
PARALLEL=3              # how many alignments run concurrently
TIMEOUT_MIN=120         # hard kill any job after this many minutes
BEAM=800
RETRY_BEAM=2000

eval "$(conda shell.bash hook)"
conda activate aligner

mkdir -p retry_logs retry_workdir

# ---- launch a single debate alignment with a watchdog ----------------------
launch_one() {
  local did=$1
  local log="retry_logs/retry_${did}.log"
  local workdir="retry_workdir/${did}"
  # Unique corpus name → unique MFA temp dir under ~/Documents/MFA/
  local corpus_name="corpus_${did}"
  local corpus="${workdir}/${corpus_name}"

  echo "[$(date +%H:%M:%S)] Starting ${did} (beam=${BEAM}/${RETRY_BEAM}, timeout=${TIMEOUT_MIN}m)" > "$log"

  rm -rf "$corpus" && mkdir -p "${corpus}/${did}"
  # Symlink WAV (large), copy the small files
  ln -s "$(realpath "mfa_corpus/${did}/${did}.wav")" "${corpus}/${did}/${did}.wav"
  cp "mfa_corpus/${did}/${did}.lab" "${corpus}/${did}/${did}.lab"
  cp "mfa_corpus/${did}/${did}_turns.json" "${corpus}/${did}/${did}_turns.json"

  # MFA in foreground (caller wraps with timeout)
  mfa align \
    "$corpus" \
    english_us_arpa \
    english_us_arpa \
    mfa_output \
    --clean \
    --num_jobs 1 \
    --single_speaker \
    --beam "$BEAM" \
    --retry_beam "$RETRY_BEAM" \
    >> "$log" 2>&1
  local rc=$?

  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "[$(date +%H:%M:%S)] ${did} SUCCESS" >> "$log"
  else
    echo "[$(date +%H:%M:%S)] ${did} FAILED rc=${rc}" >> "$log"
  fi
}

# ---- run with watchdog timeout --------------------------------------------
launch_with_timeout() {
  local did=$1
  ( launch_one "$did" ) &
  local job_pid=$!

  ( sleep $((TIMEOUT_MIN * 60)); kill -TERM $job_pid 2>/dev/null; sleep 5; kill -KILL $job_pid 2>/dev/null; pkill -KILL -P $job_pid 2>/dev/null ) &
  local watchdog_pid=$!

  wait $job_pid 2>/dev/null
  local rc=$?
  kill $watchdog_pid 2>/dev/null
  wait $watchdog_pid 2>/dev/null
  if [[ $rc -eq 143 ]] || [[ $rc -eq 137 ]]; then
    echo "[$(date +%H:%M:%S)] ${did} TIMED OUT after ${TIMEOUT_MIN}m" >> "retry_logs/retry_${did}.log"
  fi
}

# ---- main: throttle to PARALLEL concurrent jobs ---------------------------
echo "[$(date +%H:%M:%S)] retry_failed.sh launching ${#FAILED_DEBATES[@]} debates (max ${PARALLEL} concurrent)"
echo

active=()
for did in "${FAILED_DEBATES[@]}"; do
  # Wait until we have a free slot
  while (( ${#active[@]} >= PARALLEL )); do
    for i in "${!active[@]}"; do
      pid="${active[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        unset 'active[i]'
      fi
    done
    active=("${active[@]}")  # reindex
    [[ ${#active[@]} -lt PARALLEL ]] && break
    sleep 5
  done

  launch_with_timeout "$did" &
  active+=("$!")
  echo "[$(date +%H:%M:%S)] dispatched ${did} (slot pid=$!)"
done

# Wait for all remaining to finish
echo
echo "[$(date +%H:%M:%S)] All dispatched. Waiting for completion…"
wait

# ---- summary --------------------------------------------------------------
echo
echo "[$(date +%H:%M:%S)] === Done ==="
for did in "${FAILED_DEBATES[@]}"; do
  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "  ✓ ${did}  TextGrid written"
  else
    echo "  ✗ ${did}  no TextGrid — see retry_logs/retry_${did}.log"
  fi
done
