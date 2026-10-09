#!/bin/bash
# Experiment 31, reader stage (§31; NEEDS SPEND, not approved). Usage: exp31_stage.sh POD_URL [WORKTREE]
# The pod runs memctl/ttt_server.py (pinned commit) with Qwen2.5-3B-Instruct. This answers all 5 folds x 4 cells
# (stub judge) and enforces the pre-registered time gate: once 100 qTTT answers are served, if their mean server
# time (prefill + adaptation + decoding) divided by the 4 replicas sharing the GPU (the effective GPU-seconds per
# answer) is above 10 s, every sweep is stopped and GATE_FAILED is written.
URL=$1; W=${2:-$(pwd)}
cd $W
PY=/home/qamil-mirza/Code/llm-memctl/.venv/bin/python
G=/home/qamil-mirza/Code/llm-memctl/runs/_pipelines/guard.sh
mkdir -p runs/_logs runs/_sweeps
until curl -sf -m 10 $URL/v1/models | grep -q qttt; do sleep 10; done
date -u +%H:%M:%S > runs/_logs/exp31_READY
PIDS=""
for k in 0 1 2 3 4; do
  sed "s#POD_URL#$URL#g" configs/sweeps/exp31/exp31_qwen3b_f$k.yaml > runs/_sweeps/exp31_qwen3b_f$k.yaml
  $G $PY -m memctl.sweep --config runs/_sweeps/exp31_qwen3b_f$k.yaml --workers 4 >> runs/_logs/exp31_f$k.log 2>&1 &
  PIDS="$PIDS $!"
done
echo $PIDS > runs/_logs/exp31_sweeps.pids
gate() {  # prints "n mean_seconds" for the qTTT name
  curl -sf -m 10 $URL/stats | $PY -c "import json,sys; s=json.load(sys.stdin)['qwen2.5-3b-instruct-hf-qttt']; print(s['n'], (s['prefill_s']+s['adapt_s']+s['generate_s'])/4)"
}
while kill -0 $PIDS 2>/dev/null; do
  read N MEAN <<< "$(gate)"
  if [ -n "$N" ] && [ "$N" -ge 100 ]; then
    echo "$N $MEAN" > runs/_logs/exp31_GATE
    if $PY -c "import sys; sys.exit(0 if float('$MEAN') > 10 else 1)"; then
      kill $PIDS; echo "mean $MEAN s over $N qTTT answers" > runs/_logs/exp31_GATE_FAILED; exit 1
    fi
    break
  fi
  sleep 30
done
wait $PIDS
date -u +%H:%M:%S > runs/_logs/exp31_DONE
