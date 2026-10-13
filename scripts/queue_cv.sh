#!/usr/bin/env bash
# 5-fold cross-validated ranker over the whole corpus (Phase 4 gold-error detection).
# Candidate tables are built on the CPU while the previous fold trains on the GPU.
# Usage (Git Bash, repo root): bash scripts/queue_cv.sh
# Logs: models/logs/{prepare,ranker}-cv5-<k>.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUTF8=1 PYTHONPATH=src
FOLDS="0 1 2 3 4"

(
  for k in $FOLDS; do
    $PY -u scripts/prepare_candidates.py --split cv5-$k > models/logs/prepare-cv5-$k.log 2>&1
  done
) &

for k in $FOLDS; do
  log=models/logs/prepare-cv5-$k.log
  until grep -qE -- 'oracle recall train|Traceback' $log 2>/dev/null; do sleep 30; done
  if grep -q Traceback $log; then echo "prepare cv5-$k failed"; continue; fi
  sleep 5
  # run_ranker can exit non-zero after a successful run, so check its output instead.
  $PY -u scripts/run_ranker.py --split cv5-$k > models/logs/ranker-cv5-$k.log 2>&1
  if [ -f results/cv5-$k/ranker.pred.tsv ]; then
    $PY scripts/evaluate.py --system ranker --from-predictions --splits cv5-$k >> models/logs/ranker-cv5-$k.log 2>&1
    $PY scripts/triage.py --split cv5-$k >> models/logs/ranker-cv5-$k.log 2>&1
  else
    echo "ranker cv5-$k failed"
  fi
done
wait
$PY scripts/collect_cv.py
echo "cv queue finished"
