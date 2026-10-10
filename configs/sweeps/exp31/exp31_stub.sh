#!/bin/bash
# Experiment 31 full-size stub run (§31; FREE): a random small model (STUB_MODEL; default 42M parameters) of the Qwen2.5 architecture with the
# real Qwen2.5 tokenizer, served by memctl/ttt_server.py on CPU, all 5 folds x 4 cells, at the paid run's worker counts.
# Usage: exp31_stub.sh PORT [WORKTREE]   (run again with a different port for the resume rehearsal)
W=${2:-$(pwd)}
cd $W
PY=/home/qamil-mirza/Code/llm-memctl/.venv/bin/python
G=/home/qamil-mirza/Code/llm-memctl/runs/_pipelines/guard.sh
PORT=$1; URL=http://127.0.0.1:$PORT
mkdir -p runs/_logs runs/_sweeps
[ -n "$STUB_MODEL" ] || STUB_MODEL='{"hidden_size":256,"intermediate_size":704,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2}'
export HF_HOME=$W/external/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3
GUARD_VM_GB=60 /usr/bin/time -v $G $PY -m memctl.ttt_server serve --model Qwen/Qwen2.5-3B-Instruct \
  --served-name qwen2.5-3b-instruct-hf --dtype float32 --device cpu --port $PORT --replicas 4 --steps ${STUB_STEPS:-32} \
  --random-init "$STUB_MODEL" \
  --log runs/_logs/stub_server_$PORT.jsonl > runs/_logs/stub_server_$PORT.log 2>&1 &
echo $! > runs/_logs/stub_server_$PORT.pid
until curl -sf -m 5 $URL/v1/models | grep -q qttt; do
  kill -0 $(cat runs/_logs/stub_server_$PORT.pid) 2>/dev/null || { echo "server died" > runs/_logs/STUB_FAILED_$PORT; exit 1; }
  sleep 5; done
date -u +%H:%M:%S > runs/_logs/STUB_READY_$PORT
PIDS=""
for k in 0 1 2 3 4; do
  # stub only: step logs off (disk); logging is not part of a cell's resume identity
  sed -e "s#POD_URL#$URL#g" -e "s#detail_episodes: 100000#detail_episodes: 0#" configs/sweeps/exp31/exp31_qwen3b_f$k.yaml > runs/_sweeps/exp31_qwen3b_f$k.yaml
  /usr/bin/time -v $G $PY -m memctl.sweep --config runs/_sweeps/exp31_qwen3b_f$k.yaml --workers 2 >> runs/_logs/stub_sweep_f$k.log 2>&1 &
  PIDS="$PIDS $!"
done
echo $PIDS > runs/_logs/stub_sweeps_$PORT.pids
wait $PIDS
date -u +%H:%M:%S > runs/_logs/STUB_DONE_$PORT
kill $(pgrep -P $(cat runs/_logs/stub_server_$PORT.pid))
