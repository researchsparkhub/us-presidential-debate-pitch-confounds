#!/usr/bin/env bash
#
# Sequential retry for the 4 older debates that failed the parallel retry.
#
# - 1001, 1007, 1019 hit a race condition when 3 jobs simultaneously tried to
#   extract the acoustic model. Running ONE AT A TIME eliminates the race.
# - 1008 ran but MFA reported the beam was too tight. Uses 4000/10000 here.
#
# Each debate has its own corpus folder (so MFA temp dirs don't collide) and
# a 120-minute hard watchdog timeout.
#
# Run:
#     nohup ./retry_v2_sequential.sh > retry_v2_main.log 2>&1 &
#     tail -f retry_v2_main.log retry_logs/retry_*.log
# Output: mfa_output/<id>/<id>.TextGrid for each success

set -uo pipefail
cd "$(dirname "$0")"

# Per-debate (beam, retry_beam, timeout_minutes)
# Indexed lists so each debate gets its own settings.
DEBATES=("1001" "1007" "1019" "1008")
BEAMS=(    "800" "800" "800" "4000")
RETRY_BEAMS=("2000" "2000" "2000" "10000")
TIMEOUTS=( "120" "120" "120" "180")

eval "$(conda shell.bash hook)"
conda activate aligner

mkdir -p retry_logs retry_workdir_v2

run_one_with_timeout() {
  local did=$1
  local beam=$2
  local retry_beam=$3
  local timeout_min=$4
  local log="retry_logs/retry_v2_${did}.log"
  local workdir="retry_workdir_v2/${did}"
  local corpus_name="corpus_v2_${did}"
  local corpus="${workdir}/${corpus_name}"

  echo "[$(date +%H:%M:%S)] Starting ${did} (beam=${beam}/${retry_beam}, timeout=${timeout_min}m)" | tee -a "$log"

  rm -rf "$corpus" && mkdir -p "${corpus}/${did}"
  ln -s "$(realpath "mfa_corpus/${did}/${did}.wav")" "${corpus}/${did}/${did}.wav"
  cp "mfa_corpus/${did}/${did}.lab" "${corpus}/${did}/${did}.lab"
  cp "mfa_corpus/${did}/${did}_turns.json" "${corpus}/${did}/${did}_turns.json"

  # Run with watchdog
  (
    mfa align \
      "$corpus" \
      english_us_arpa \
      english_us_arpa \
      mfa_output \
      --clean \
      --num_jobs 1 \
      --single_speaker \
      --beam "$beam" \
      --retry_beam "$retry_beam" \
      >> "$log" 2>&1
  ) &
  local job_pid=$!

  (
    sleep $((timeout_min * 60))
    kill -TERM "$job_pid" 2>/dev/null
    sleep 5
    kill -KILL "$job_pid" 2>/dev/null
    pkill -KILL -P "$job_pid" 2>/dev/null
  ) &
  local watchdog=$!

  wait "$job_pid" 2>/dev/null
  local rc=$?
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null

  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "[$(date +%H:%M:%S)] ${did} SUCCESS (rc=${rc})" | tee -a "$log"
    return 0
  else
    echo "[$(date +%H:%M:%S)] ${did} FAILED (rc=${rc}) — see ${log}" | tee -a "$log"
    return 1
  fi
}

echo "[$(date +%H:%M:%S)] Sequential retry of ${#DEBATES[@]} debates"
echo

n_success=0
for i in "${!DEBATES[@]}"; do
  did="${DEBATES[$i]}"
  beam="${BEAMS[$i]}"
  retry_beam="${RETRY_BEAMS[$i]}"
  timeout_min="${TIMEOUTS[$i]}"

  echo
  echo "================================================================"
  echo "  Debate ${did}  (${i+1}/${#DEBATES[@]})"
  echo "================================================================"

  if run_one_with_timeout "$did" "$beam" "$retry_beam" "$timeout_min"; then
    n_success=$((n_success + 1))
  fi
done

echo
echo "[$(date +%H:%M:%S)] === Sequential retry done: ${n_success}/${#DEBATES[@]} succeeded ==="
for did in "${DEBATES[@]}"; do
  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "  ✓ ${did}"
  else
    echo "  ✗ ${did}"
  fi
done
