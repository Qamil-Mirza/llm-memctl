#!/bin/bash
# Experiment 29 (§29) on one Qwen2.5-7B pod (vLLM 0.8.5). Run from the repository (or worktree) root.
#   stage.sh rows          CPU, before the pod: the 500 questions' 32-candidate decisions (reader-free)
#   stage.sh signal URL    pod step 1: 15,509 prompt-logprob calls (+470 tokenize); writes the utility rows
#   stage.sh tune URL      per fold, inner 6/2 split of the TRAINING questions: blend weight from the grid
#   stage.sh train         CPU: final utility / blend / evidence-rows heads per fold, then the controller check
#   stage.sh reader URL    pod step 2: five arms x five test folds, then the one blind judge pass
#   stage.sh report        the pre-registered claims, once
# STUB=1: the free rehearsal against configs/sweeps/exp29/fake_vllm.py; separate stub folders and caches. Its reader
# cells play each fold's training part; its metrics go through memctl.reader_utility stub-metrics, which refuses
# test-fold cells.
set -u -o pipefail
ROOT=/home/qamil-mirza/Code/llm-memctl
PY=$ROOT/.venv/bin/python
GUARD=$ROOT/runs/_pipelines/guard.sh; [ -x "$GUARD" ] || GUARD=""
export PYTHONPATH=$PWD
mkdir -p runs/_logs runs/_sweeps
if [ "${STUB:-0}" = 1 ]; then
  TAG=stub_
  export EXP29_DIR=runs/stub_exp29 EXP29_HEADS=runs/stub_exp29/heads EXP29_CACHE=cache/stub_utility_exp29
else
  TAG=
fi
DIR=${EXP29_DIR:-runs/exp29}
HEADS=${EXP29_HEADS:-configs/sweeps/exp29/heads}
wait_model() {  # gives up after WAIT_MAX seconds (default 900), so a lost network cannot idle a billing pod
  local end=$(( $(date +%s) + ${WAIT_MAX:-900} ))
  until curl -sf -m 10 $1/v1/models | grep -q '"qwen2.5-7b-instruct"'; do
    [ $(date +%s) -ge $end ] && { echo "model not reachable after ${WAIT_MAX:-900} s" >&2; exit 3; }
    sleep 10
  done
}
case "$1" in
rows)
  $GUARD $PY -m memctl.reader_utility rows --workers 6 ;;
signal)
  wait_model $2
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_signal_START
  $GUARD $PY -m memctl.reader_utility signal --base-url $2/v1 --workers 32 2>&1 | tee -a runs/_logs/${TAG}exp29_signal.log
  [ -f $DIR/utility.jsonl.gz ] && date -u +%H:%M:%S > runs/_logs/${TAG}exp29_signal_DONE ;;
tune)
  [ -f $DIR/utility.jsonl.gz ] || { echo "no utility rows: run the signal step first"; exit 1; }
  wait_model $2
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_tune_START
  for k in 0 1 2 3 4; do  # one process per fold: the CPU training is the slow part, the pod waits
    $GUARD $PY -m memctl.reader_utility tune --fold $k --base-url $2/v1 --workers 16 >> runs/_logs/${TAG}exp29_tune.log 2>&1 &
  done
  wait
  for k in 0 1 2 3 4; do [ -f $DIR/tune_f$k.json ] || { echo "fold $k not tuned"; exit 1; }; done
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_tune_DONE ;;
train)
  for k in 0 1 2 3 4; do
    $GUARD $PY -m memctl.reader_utility train --fold $k >> runs/_logs/${TAG}exp29_train.log 2>&1 &
  done
  wait
  for k in 0 1 2 3 4; do $GUARD $PY -m memctl.reader_utility check --fold $k --n 3 || exit 1; done ;;
reader)
  URL=$2
  for k in 0 1 2 3 4; do for h in utility blend evidence; do
    [ -f $HEADS/f$k/$h.pt ] || { echo "missing head $HEADS/f$k/$h.pt: run train first"; exit 1; }
  done; done
  wait_model $URL
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_reader_START
  pids=(); LOGN=$(cat runs/_logs/${TAG}exp29_reader.log 2>/dev/null | wc -l)
  for k in 0 1 2 3 4; do
    OUT=runs/_sweeps/${TAG}exp29_qwen7b_f$k.yaml
    sed -e "s#POD_URL#$URL#g" -e "s#HEADS_DIR#$HEADS#g" configs/sweeps/exp29/exp29_qwen7b_f$k.yaml > $OUT
    if [ -n "$TAG" ]; then
      # The rehearsal plays each fold's TRAINING part, never its test part, so no stub number is a test-fold number.
      sed -i -e "s#cache/generations_exp29_qwen7b#cache/stub_generations_exp29#g" -e "s#^name: exp29_#name: stub_exp29_#" \
          -e "s#detail_episodes: 100000#detail_episodes: ${STUB_DETAIL:-100000}#" -e "s#part: test#part: train#" $OUT
    fi
    $GUARD $PY -m memctl.sweep --config $OUT --workers 3 >> runs/_logs/${TAG}exp29_reader.log 2>&1 &
    pids+=($!)
  done
  status=0  # a marker is written only when its step succeeded (the 2026-10-09 run wrote both after a dead pod)
  for pid in "${pids[@]}"; do wait $pid || status=1; done
  # memctl.sweep exits 0 even when cells fail, so read this run's part of the log too
  tail -n +$((LOGN+1)) runs/_logs/${TAG}exp29_reader.log | grep -q "cells failed" && status=1
  [ $status = 0 ] || { echo "a fold's sweep failed; see runs/_logs/${TAG}exp29_reader.log" >&2; exit 1; }
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_reader_ANSWERS_DONE
  $GUARD $PY configs/sweeps/exp29/judge_report.py judge --base-url $URL/v1 --prefix ${TAG}exp29 \
    --cache cache/${TAG}judge_exp29 > runs/_logs/${TAG}exp29_judge.log 2>&1 || { echo "judge failed" >&2; exit 1; }
  date -u +%H:%M:%S > runs/_logs/${TAG}exp29_reader_DONE ;;
report)
  $PY configs/sweeps/exp29/judge_report.py report --prefix ${TAG}exp29 | tee runs/${TAG}exp29_report.md ;;
*)
  echo "usage: stage.sh rows | signal URL | tune URL | train | reader URL | report"; exit 2 ;;
esac
