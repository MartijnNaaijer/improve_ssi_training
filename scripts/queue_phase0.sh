#!/usr/bin/env bash
# Phase 0 upstream runs, one after another on the single GPU.
# Usage (Git Bash, repo root): bash scripts/queue_phase0.sh
# Each job logs to models/logs/<system>-<split>.log and is scored right away.
# A failing job is logged and skipped; the queue continues.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUTF8=1
mkdir -p models/logs

# Wait for the ssi-seq2seq random run that was started before this queue.
until grep -qE -- '^-> |Traceback' models/logs/ssi-seq2seq-random.log 2>/dev/null; do sleep 60; done
$PY scripts/evaluate.py --system ssi-seq2seq --from-predictions --splits random

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

run chen-transformer random
run ssi-seq2seq heldout-Sir
run chen-transformer heldout-Sir
run ssi-seq2seq heldout-ESNJ
run chen-transformer heldout-ESNJ
# Reduced budget: the headline MDLM configuration but 40 instead of 250 epochs.
run chen-mdlm random --epochs 40
echo "$(date '+%F %T') queue finished" | tee -a models/logs/queue.log
