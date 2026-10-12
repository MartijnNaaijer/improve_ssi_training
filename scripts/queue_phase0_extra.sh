#!/usr/bin/env bash
# Extra Phase 0 runs, started after scripts/queue_phase0.sh has finished.
# Usage (Git Bash, repo root): bash scripts/queue_phase0_extra.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUTF8=1

until grep -q 'queue finished' models/logs/queue.log 2>/dev/null; do sleep 120; done

run() {  # system split [extra args]
  local system=$1 split=$2; shift 2
  local log=models/logs/$system-$split.log
  echo "$(date '+%F %T') start $system $split $*" | tee -a models/logs/queue.log
  if $PY -u scripts/run_upstream.py --system "$system" --split "$split" "$@" > "$log" 2>&1; then
    $PY scripts/evaluate.py --system "$system" --from-predictions --splits "$split" >> "$log" 2>&1
    echo "$(date '+%F %T') done  $system $split" | tee -a models/logs/queue.log
  else
    echo "$(date '+%F %T') FAILED $system $split (see $log)" | tee -a models/logs/queue.log
  fi
}

run chen-bilstm-crf random
echo "$(date '+%F %T') extra queue finished" | tee -a models/logs/queue.log
