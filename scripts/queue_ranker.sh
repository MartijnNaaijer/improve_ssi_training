#!/usr/bin/env bash
# Phase 2 ranker runs, after the random run that was started before this queue.
# Usage (Git Bash, repo root): bash scripts/queue_ranker.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUTF8=1 PYTHONPATH=src
until grep -qE -- 'ranker.pred.tsv|Traceback' models/logs/ranker-random.log 2>/dev/null; do sleep 30; done
for part in dev test; do $PY scripts/evaluate.py --system ranker --from-predictions --splits random --part $part; done
for s in heldout-Sir heldout-ESNJ; do
  if $PY -u scripts/run_ranker.py --split $s > models/logs/ranker-$s.log 2>&1; then
    $PY scripts/evaluate.py --system ranker --from-predictions --splits $s >> models/logs/ranker-$s.log 2>&1
  fi
done
echo "ranker queue finished"
