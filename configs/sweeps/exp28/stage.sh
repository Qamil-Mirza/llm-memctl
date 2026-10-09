#!/bin/bash
# Experiment 28 (§28) on the shared Qwen2.5-7B pod. Run from the repository (or worktree) root.
#   stage.sh gate N URL   prompt vN: rewrite the 307 LoCoMo dev questions and the 500 LongMemEval questions; print
#                         the LoCoMo dev outputs and dev recall (the only thing a revision may use; LongMemEval text and
#                         rewrites are never shown); score on every open fold's TRAINING part (aggregate output only);
#                         close the gate when every fold passes or after v4. A revision vN+1 is written from the dev
#                         output only, into prompts/vN+1.txt, and committed first.
#   stage.sh reader URL   fixed8 and fixed8_rewrite on the five test folds, then the blind judge. Refuses unless the
#                         gate is final and GO; the rewrites are replayed from the cache only.
# STUB=1: the free rehearsal. Stub backends, separate stub caches and gate folder; no server is called.
set -u
ROOT=/home/qamil-mirza/Code/llm-memctl
PY=$ROOT/.venv/bin/python
GUARD=$ROOT/runs/_pipelines/guard.sh; [ -x "$GUARD" ] || GUARD=""
export PYTHONPATH=$PWD
mkdir -p runs/_logs runs/_sweeps
if [ "${STUB:-0}" = 1 ]; then
  export EXP28_GATE_DIR=runs/stub_exp28_gate EXP28_REWRITE_CACHE=cache/stub_rewrite_exp28
  export EXP28_PROMPT_DIR=${EXP28_PROMPT_DIR:-configs/sweeps/exp28/prompts}
  BACKEND=stub; TAG=stub_
else
  BACKEND=openai; TAG=
fi
GATE_DIR=${EXP28_GATE_DIR:-runs/exp28_gate}
PROMPTS=${EXP28_PROMPT_DIR:-configs/sweeps/exp28/prompts}

case "$1" in
gate)
  N=$2; URL=$3
  if [ "$BACKEND" = openai ]; then
    until curl -sf -m 10 $URL/v1/models | grep -q '"qwen2.5-7b-instruct"'; do sleep 10; done
  fi
  $PY -m memctl.rewrite_gate --backend $BACKEND rewrite --version $N --base-url $URL/v1 || exit 1
  $PY -m memctl.rewrite_gate --backend $BACKEND inspect --version $N | tee runs/_logs/${TAG}exp28_inspect_v$N.log
  $PY -m memctl.rewrite_gate --backend $BACKEND gate --version $N | tee runs/_logs/${TAG}exp28_gate_v$N.log || exit 1
  if $PY -c "import json,sys; s=json.load(open('$GATE_DIR/state.json')); sys.exit(0 if len(s['selected'])==5 or len(s['versions'])>=4 else 1)"; then
    $PY -m memctl.rewrite_gate --backend $BACKEND final | tee runs/_logs/${TAG}exp28_final.log
  else
    echo "gate open: folds without a passing version remain; write v$((N + 1)) from the LoCoMo dev output only, or stop"
  fi
  ;;
reader)
  URL=$2
  $PY -c "import json,sys; s=json.load(open('$GATE_DIR/selected.json')); sys.exit(0 if s['go'] else 1)" \
    || { echo "arm (a) gate not final or not GO: the reader stage does not run"; exit 1; }
  if [ "$BACKEND" = openai ]; then
    until curl -sf -m 10 $URL/v1/models | grep -q '"qwen2.5-7b-instruct"'; do sleep 10; done
  fi
  date -u +%H:%M:%S > runs/_logs/${TAG}exp28_reader_START
  for k in 0 1 2 3 4; do
    V=$($PY -c "import json; print(json.load(open('$GATE_DIR/selected.json'))['selected']['$k'])")
    OUT=runs/_sweeps/${TAG}exp28_qwen7b_f$k.yaml
    sed -e "s#POD_URL#$URL#g" -e "s#REWRITE_PROMPT#$PROMPTS/v$V.txt#g" \
        configs/sweeps/exp28/exp28_qwen7b_f$k.yaml > $OUT
    if [ "$BACKEND" = stub ]; then
      sed -i -e "s#backend: openai#backend: stub#g" -e "s#cache/generations_exp27_qwen7b#cache/stub_generations_exp28#g" \
          -e "s#cache/rewrite_exp28#cache/stub_rewrite_exp28#g" -e "s#^name: exp28_#name: stub_exp28_#" $OUT
    fi
    $GUARD $PY -m memctl.sweep --config $OUT --workers 3 >> runs/_logs/${TAG}exp28_reader.log 2>&1 &
  done
  wait
  date -u +%H:%M:%S > runs/_logs/${TAG}exp28_reader_ANSWERS_DONE
  $GUARD $PY configs/sweeps/exp28/judge_report.py judge --backend $BACKEND --base-url $URL/v1 --prefix ${TAG}exp28 --cache cache/${TAG}judge_exp28 \
    --out runs/${TAG}exp28_verdicts.json > runs/_logs/${TAG}exp28_judge.log 2>&1
  date -u +%H:%M:%S > runs/_logs/${TAG}exp28_reader_DONE
  ;;
*)
  echo "usage: stage.sh gate N URL | stage.sh reader URL"; exit 2 ;;
esac
