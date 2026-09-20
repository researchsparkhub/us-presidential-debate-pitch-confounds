#!/usr/bin/env bash
#
# Final retry pass on the 6 stragglers. Differences vs prior retries:
#
# 1. Wrapped in `caffeinate -is` so macOS can't sleep / suspend the laptop
#    mid-run. Without this, sleep time counts toward the watchdog and kills
#    MFA before it actually finishes.
# 2. Runs strictly sequential — one debate at a time, no parallel jobs.
# 3. Per-debate beam settings tuned by what we learned:
#       - 1001, 1007, 1019: standard beam 800/2000, 150 min timeout
#       - 1008: wider beam 4000/10000, 240 min timeout
#       - 1034, 1036: re-use the already-trimmed audio (1034_trimmed.wav,
#         1036_trimmed.wav) with beam 800/2000, 150 min timeout
# 4. Each debate runs in its own corpus folder so MFA temp dirs don't collide.
# 5. Time-stamped status lines so you can tell what's been wall-clock vs CPU.
#
# Run:
#     # keep laptop plugged in + lid open; caffeinate handles the rest
#     nohup ./retry_v3_final.sh > retry_v3_main.log 2>&1 &
#     tail -f retry_v3_main.log
#
# Estimated wall time: 6–10 hours total

set -uo pipefail
cd "$(dirname "$0")"

# Per-debate config:  did|beam|retry_beam|timeout_min|wav_basename
# wav_basename: "<did>.wav" for normal, "<did>_trimmed.wav" for the 2024 ones
RUNS=(
  "1001|800|2000|150|1001.wav"
  "1007|800|2000|150|1007.wav"
  "1019|800|2000|150|1019.wav"
  "1008|4000|10000|240|1008.wav"
  "1034|800|2000|150|1034_trimmed.wav"
  "1036|800|2000|150|1036_trimmed.wav"
)

eval "$(conda shell.bash hook)"
conda activate aligner

mkdir -p retry_logs retry_workdir_v3

run_one() {
  local did=$1 beam=$2 retry_beam=$3 timeout_min=$4 wav_name=$5
  local log="retry_logs/retry_v3_${did}.log"
  local workdir="retry_workdir_v3/${did}"
  local corpus_name="corpus_v3_${did}"
  local corpus="${workdir}/${corpus_name}"
  local wav_src

  if [[ "$wav_name" == *"_trimmed.wav" ]]; then
    wav_src="mfa_corpus/${did}/${wav_name}"
    if [[ ! -f "$wav_src" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${did} SKIP — trimmed wav missing at ${wav_src}" | tee -a "$log"
      return 1
    fi
  else
    wav_src="mfa_corpus/${did}/${wav_name}"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${did} (beam=${beam}/${retry_beam}, timeout=${timeout_min}m, wav=${wav_name})" | tee -a "$log"

  rm -rf "$corpus" && mkdir -p "${corpus}/${did}"
  # symlink the wav (real or trimmed), copy the small files
  ln -s "$(realpath "$wav_src")" "${corpus}/${did}/${did}.wav"
  cp "mfa_corpus/${did}/${did}.lab" "${corpus}/${did}/${did}.lab"
  cp "mfa_corpus/${did}/${did}_turns.json" "${corpus}/${did}/${did}_turns.json"

  # MFA writes to the unified mfa_output dir except for 2024 trimmed cases —
  # those need TextGrid shifting back to original timeline. For the 4 older
  # debates, MFA output goes straight into mfa_output/.
  local out_dir="mfa_output"
  if [[ "$wav_name" == *"_trimmed.wav" ]]; then
    out_dir="${workdir}/mfa_output_trim"
    mkdir -p "$out_dir"
  fi

  # Run MFA with watchdog
  (
    mfa align \
      "$corpus" \
      english_us_arpa \
      english_us_arpa \
      "$out_dir" \
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

  # Post-process trimmed outputs: shift TextGrid into original-audio coords
  # (stdlib only — json/re/pathlib — so the system/venv python3 is enough)
  if [[ "$wav_name" == *"_trimmed.wav" && -f "${out_dir}/${did}/${did}.TextGrid" ]]; then
    python3 - <<PY
import json, re
from pathlib import Path
tg_path = Path("${out_dir}/${did}/${did}.TextGrid")
offset_path = Path("mfa_corpus/${did}/${did}_trim_offset.json")
out_path = Path("mfa_output/${did}/${did}.TextGrid")
shift = json.loads(offset_path.read_text())["trim_start"]
text = tg_path.read_text(encoding="utf-8")
text = re.sub(r"(xmin\s*=\s*|xmax\s*=\s*)([\d.]+)",
              lambda m: f"{m.group(1)}{float(m.group(2)) + shift:.6f}", text)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(text, encoding="utf-8")
print(f"  shifted TextGrid by +{shift:.1f}s -> {out_path}")
PY
  fi

  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${did} SUCCESS (rc=${rc})" | tee -a "$log"
    return 0
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${did} FAILED (rc=${rc}) — see ${log}" | tee -a "$log"
    return 1
  fi
}

# ===== main: run sequentially, wrapped in caffeinate to prevent sleep =======
START_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[${START_TS}] retry_v3_final.sh starting — ${#RUNS[@]} debates, sequential, caffeinated"
echo "[${START_TS}] PID: $$"
echo

# Re-execute ourselves under caffeinate -is so the script (and its children)
# can't be sleep-suspended by macOS.
if [[ "${CAFFEINATED:-0}" != "1" ]]; then
  echo "[$(date '+%H:%M:%S')] re-launching under caffeinate -is"
  export CAFFEINATED=1
  exec caffeinate -is "$0" "$@"
fi

n_success=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r did beam retry_beam timeout_min wav_name <<< "$entry"
  echo
  echo "================================================================"
  echo "  Debate ${did}"
  echo "================================================================"
  if run_one "$did" "$beam" "$retry_beam" "$timeout_min" "$wav_name"; then
    n_success=$((n_success + 1))
  fi
done

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === retry_v3_final done: ${n_success}/${#RUNS[@]} succeeded ==="
for entry in "${RUNS[@]}"; do
  IFS='|' read -r did _ _ _ _ <<< "$entry"
  if [[ -f "mfa_output/${did}/${did}.TextGrid" ]]; then
    echo "  ✓ ${did}"
  else
    echo "  ✗ ${did}"
  fi
done
