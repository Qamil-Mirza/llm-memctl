#!/usr/bin/env bash
# §51 pod H + T-city (two-pod plan, D-73): runs only after the F2 commit; its own quote and the user's line in the
# Executor pane. Order: preflight -> watchdog -> GPU + install -> the git-verified F2 guard (exit 4: nothing sealed is
# read) -> download -> D-22 smoke (the dummy, S rows only, smoke root) -> h-run (H once, then T-city; deadline,
# resumable) -> h-report (prints nothing about outcomes) -> pod.done. Copy back runs/exp51/gpu.
#   RUNPOD_POD_ID=<id> HARD_STOP_S=.. SELF_STOP_S=.. bash scripts/exp51_pod_ht.sh
#   ENGINE=stub [INSTALL=1 PYTHON=<venv>/bin/python] bash scripts/exp51_pod_ht.sh      # CPU, dummy only (D-30)
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"; INSTALL="${INSTALL:-}"
HARD_STOP_S="${HARD_STOP_S:-22000}"; SELF_STOP_S="${SELF_STOP_S:-23500}"
OUT=runs/exp51/gpu; PY="${PYTHON:-python}"; [ "$ENGINE" = stub ] && PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "exp51 pod H+T; engine $ENGINE; hard stop +$HARD_STOP_S s; self-stop +$SELF_STOP_S s"
if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo self-stop >> '$OUT/pod.log'; if command -v runpodctl >/dev/null && \
    [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  git config --global --add safe.directory '*'
fi
for f in "$OUT/design.json" "$OUT/design_committed.json" .git scripts/exp51_stage4.py scripts/exp51_smoke_s.py; do test -e "$f" || { log "missing $f"; exit 3; }; done
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==5.17.0" accelerate huggingface_hub \
    safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import transformers, sklearn, scripts.exp51_stage4; assert transformers.__version__ == '5.17.0'" || { log "install check failed"; exit 4; }
}
if [ "$ENGINE" != stub ]; then
  until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
  install_line; DEV=cuda
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""; [ -n "$INSTALL" ] && install_line; DEV=cpu
fi
set +e   # errexit + pipefail would end the chain at this pipe with exit 1, before the logged exit 4
$PY -c "import sys; sys.path.insert(0,'.'); from memctl.analysis import abst49 as A; from pathlib import Path; \
A.require_design(Path('$OUT')); print('F2 guard ok')" 2>&1 | tee -a "$OUT/pod.log"; GRC=${PIPESTATUS[0]}; set -e
[ "$GRC" -eq 0 ] || { log "F2 GUARD FAILED: no sealed row read"; exit 4; }
if [ "$ENGINE" != stub ]; then
  SNAP=$($PY -c "import huggingface_hub; print(huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct', \
revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))")
else
  SNAP=$($PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; print(E.SNAPSHOT)")
fi
$PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; from pathlib import Path; E.make_dummy(Path('$OUT/dummy'), Path('$SNAP'))" 2>&1 | { grep -v "Writing model shards" || true; }
SM="$OUT/smoke_ht"; mkdir -p "$SM"; cp -r "$OUT/edits" "$OUT/b5_fit.json" "$SM/"
step() {
  local name="$1"; shift; local rc=0
  set +e; timeout "$(left)" $PY "${STEP_SCRIPT:-scripts/exp51_stage4.py}" "$@" 2>&1 | grep -v "Loading weights" | tee -a "$OUT/$name.log"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
# D-22 smoke on dummy-shaped stand-ins of the submissions, B5 and B4 (scripts/exp51_smoke_s.py, as pod S, D4): the real
# 2048-wide edits cannot apply to the 64-wide dummy. S rows only, its own root; nothing sealed is read.
STEP_SCRIPT=scripts/exp51_smoke_s.py step smoke --stage s-arms --root "$SM" --subs "$OUT/subs" --backend hf --snapshot "$OUT/dummy" --device "$DEV" \
  --limit-bases 1 || { log "SMOKE FAILED"; exit 5; }
[ "$ENGINE" = stub ] && { log "stub: guard + smoke only"; exit 0; }
step h_run --stage h-run --root "$OUT" --subs "$OUT/subs" --snapshot "$SNAP" --device cuda --deadline-s "$(( $(left) - 300 ))" || exit $?
step h_report --stage h-report --root "$OUT" --subs "$OUT/subs" --snapshot "$SNAP" --device cuda || exit $?
log "pod H+T done at +$(( $(date +%s) - T0 )) s: copy back $OUT now"
