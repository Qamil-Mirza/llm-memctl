#!/bin/bash
# Experiment 32 (§32) reader stage. NOT APPROVED: it needs spend and the user's approval. Run from the repository (or
# worktree) root, after the free pass (it replays the cross-encoder scores the free pass cached; a miss is an error).
#   stage.sh URL          e.g. https://<pod>-8000.proxy.runpod.net : Qwen 7B answers all §32 arms on the five test
#                         folds (three workers per fold), then (unless NOJUDGE=1) the one blind Qwen 7B judge pass.
# Resume: run the same command again (a new pod URL is fine: base_url is not part of a cell's identity, ENDPOINT_KEYS).
# Hard stop: DEADLINE=<unix seconds> kills every step's whole process group (sweep parents AND their pool workers) at
#   that time; set it a few minutes before the pod's own hard stop. Stopping by hand: kill this script (INT/TERM); its
#   trap kills the same groups. (A plain `timeout` was not used: it signals only the sweep parent, and the stub
#   rehearsal showed a group kill of the script leaves the sweeps running in timeout's own groups.) Marks: runs/_logs/${TAG}exp32_{READY,ANSWERS_DONE,DONE} (no pgrep -f waits; `wait` on our
#   own children only).
# STUB=1: the free rehearsal. Stub reader and judge, separate stub caches and run names; no server is called.
set -u
ROOT=/home/qamil-mirza/Code/llm-memctl
PY=$ROOT/.venv/bin/python
GUARD=$ROOT/runs/_pipelines/guard.sh; [ -x "$GUARD" ] || GUARD=""
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=${HF_HOME:-$ROOT/cache/hf_exp32} HF_HUB_OFFLINE=1
URL=${1:?usage: stage.sh URL}
if [ "${STUB:-0}" = 1 ]; then BACKEND=stub; TAG=stub_; else BACKEND=openai; TAG=; fi
mkdir -p runs/_logs runs/_sweeps
left() { [ -z "${DEADLINE:-}" ] && { echo 86400; return; }; echo $(( DEADLINE - $(date +%s) )); }
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
[ "$FREE" -lt 15 ] && { echo "only ${FREE} GB free; need 15" >&2; exit 4; }
for k in 0 1 2 3 4; do for name in zero dense; do
  [ -s cache/exp32_ce/${name}_f$k.jsonl ] || { echo "missing cache/exp32_ce/${name}_f$k.jsonl: run the free pass first" >&2; exit 5; }
done; done
if [ "$BACKEND" = openai ]; then
  until curl -sf -m 10 $URL/v1/models | grep -q '"qwen2.5-7b-instruct"'; do
    [ "$(left)" -le 0 ] && { echo "deadline reached before the server was ready" >&2; exit 3; }; sleep 10; done
fi
date -u +%H:%M:%S > runs/_logs/${TAG}exp32_READY
pids=()
stop() { for p in "${pids[@]}"; do kill -KILL -- -$p 2>/dev/null; done; }
trap 'stop; exit 130' INT TERM
if [ -n "${DEADLINE:-}" ]; then
  ( sleep "$(left)"; echo "DEADLINE reached $(date -u +%H:%M:%S)" >> runs/_logs/${TAG}exp32_reader.log; kill -TERM $$ ) &
  WATCHDOG=$!
fi
for k in 0 1 2 3 4; do
  OUT=runs/_sweeps/${TAG}exp32_qwen7b_f$k.yaml
  sed "s#POD_URL#$URL#g" configs/sweeps/exp32/exp32_qwen7b_f$k.yaml > $OUT
  if [ "$BACKEND" = stub ]; then
    sed -i -e "s#backend: openai#backend: stub#g" -e "s#cache/generations_exp32_qwen7b#cache/stub_generations_exp32#g" \
        -e "s#^name: exp32_#name: stub_exp32_#" $OUT
  fi
  setsid $GUARD $PY -m memctl.sweep --config $OUT --workers 3 >> runs/_logs/${TAG}exp32_reader.log 2>&1 &
  pids+=($!)  # setsid execs in place: this pid leads its own process group
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
[ $status = 0 ] || { echo "a fold's sweep failed or hit the deadline; see runs/_logs/${TAG}exp32_reader.log" >&2; exit 1; }
date -u +%H:%M:%S > runs/_logs/${TAG}exp32_ANSWERS_DONE
[ "${NOJUDGE:-0}" = 1 ] && { [ -n "${WATCHDOG:-}" ] && kill $WATCHDOG 2>/dev/null; exit 0; }
JUDGE_CACHE=cache/judge_exp32; [ "$BACKEND" = stub ] && JUDGE_CACHE=cache/stub_judge_exp32
setsid $GUARD $PY configs/sweeps/exp32/judge.py --backend $BACKEND --base-url $URL/v1 \
  --prefix ${TAG}exp32_qwen7b --cache-dir $JUDGE_CACHE --out runs/${TAG}exp32_verdicts.json \
  > runs/_logs/${TAG}exp32_judge.log 2>&1 &
pids=($!)
wait ${pids[0]} || { echo "judge failed or hit the deadline" >&2; exit 1; }
[ -n "${WATCHDOG:-}" ] && kill $WATCHDOG 2>/dev/null
date -u +%H:%M:%S > runs/_logs/${TAG}exp32_DONE
