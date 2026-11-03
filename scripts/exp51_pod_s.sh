#!/usr/bin/env bash
# §51 pod S (two-pod plan, D-73): strict fp32 A40, batched layout 64. Needs its own quote and the user's line in the
# Executor pane. Order: preflight -> watchdog -> GPU + install -> download (pinned revision) -> D-22 smoke (the dummy
# tiny Qwen2: replay + s-arms + s-select on 1 base per task, smoke root) -> §10 replay check (exit 8 stops before S)
# -> s-arms (deadline, resumable) -> s-select -> f2_candidate.json -> pod.done. Copy back runs/exp51/gpu.
#   RUNPOD_POD_ID=<id> HARD_STOP_S=.. SELF_STOP_S=.. bash scripts/exp51_pod_s.sh
#   ENGINE=stub [INSTALL=1 PYTHON=<venv>/bin/python] bash scripts/exp51_pod_s.sh       # CPU, dummy only (D-30)
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"; INSTALL="${INSTALL:-}"
HARD_STOP_S="${HARD_STOP_S:-34000}"; SELF_STOP_S="${SELF_STOP_S:-35500}"
OUT=runs/exp51/gpu; PY="${PYTHON:-python}"; [ "$ENGINE" = stub ] && PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "exp51 pod S; engine $ENGINE; hard stop +$HARD_STOP_S s; self-stop +$SELF_STOP_S s"
if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo self-stop >> '$OUT/pod.log'; if command -v runpodctl >/dev/null && \
    [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
fi
python3 -c "import json; r=[json.loads(l) for l in open('runs/exp51/cases51.jsonl')]; \
assert r and all(x['split'] in ('E','S') for x in r), 'sealed rows in the S bundle'" || { log "preflight: sealed rows present"; exit 3; }
for f in runs/exp51/gpu/b5_fit.json runs/exp51/gpu/subs/1.json scripts/exp51_stage4.py; do test -e "$f" || { log "missing $f"; exit 3; }; done
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==5.17.0" accelerate huggingface_hub \
    safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import transformers, sklearn, scripts.exp51_stage4; assert transformers.__version__ == '5.17.0'" || { log "install check failed"; exit 4; }
}
if [ "$ENGINE" != stub ]; then
  until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
  install_line; DEV=cuda
  SNAP=$($PY -c "import huggingface_hub; print(huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct', \
revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))")
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""; [ -n "$INSTALL" ] && install_line; DEV=cpu
  SNAP=$($PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; print(E.SNAPSHOT)")
fi
log "ready at +$(( $(date +%s) - T0 )) s; snapshot $SNAP"
$PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; from pathlib import Path; E.make_dummy(Path('$OUT/dummy'), Path('$SNAP'))" 2>&1 | { grep -v "Writing model shards" || true; }
SM="$OUT/smoke"; mkdir -p "$SM"; cp -r "$OUT/edits" "$OUT/b5_fit.json" "$SM/" 2>/dev/null || true
step() {   # step <name> <args...>
  local name="$1"; shift; local rc=0
  set +e; timeout "$(left)" $PY scripts/exp51_stage4.py "$@" 2>&1 | grep -v "Loading weights" | tee -a "$OUT/$name.log"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
# D-22 smoke on the dummy (its own root; the replay check is skipped: the dummy cannot match the laptop)
step smoke_arms --stage s-arms --root "$SM" --subs runs/exp51/gpu/subs --backend hf --snapshot "$OUT/dummy" \
  --device "$DEV" --limit-bases 1 || { log "SMOKE FAILED"; exit 5; }
step smoke_select --stage s-select --root "$SM" --subs runs/exp51/gpu/subs --backend hf --snapshot "$OUT/dummy" \
  --device "$DEV" --limit-bases 1 --boot 200 || { log "SMOKE FAILED"; exit 5; }
test -e "$SM/f2_candidate.json" || { log "SMOKE FAILED: no f2_candidate.json"; exit 5; }
[ "$ENGINE" = stub ] && { log "stub: smoke only"; exit 0; }
step replay --stage replay --root "$OUT" --subs runs/exp51/gpu/subs --snapshot "$SNAP" --device cuda || { log "REPLAY CHECK FAILED: S not run"; exit 8; }
step s_arms --stage s-arms --root "$OUT" --subs runs/exp51/gpu/subs --snapshot "$SNAP" --device cuda --deadline-s "$(( $(left) - 60 ))" || exit $?
step s_select --stage s-select --root "$OUT" --subs runs/exp51/gpu/subs --snapshot "$SNAP" --device cuda --deadline-s "$(( $(left) - 60 ))" || exit $?
log "pod S done at +$(( $(date +%s) - T0 )) s: copy back $OUT (f2_candidate.json) now"
