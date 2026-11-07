#!/usr/bin/env bash
# §53 pod (one pod; its own quote and the user's line in the Executor pane). Order: preflight -> watchdog -> GPU +
# install -> the git-verified manifest guard (exit 4: no N-held row is read) -> download -> smoke on N-dev rows only
# with the real model (exit 5) -> pod stage (B0 on N-held, the sample, every arm; deadline, resumable) -> report ->
# pod.done. Copy back runs/exp53/pod and runs/exp53/smoke.
#   RUNPOD_POD_ID=<id> HARD_STOP_S=.. SELF_STOP_S=.. bash scripts/exp53_pod.sh
#   ENGINE=stub [INSTALL=1 PYTHON=<venv>/bin/python] bash scripts/exp53_pod.sh      # CPU: guard + install only (D-30)
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"; INSTALL="${INSTALL:-}"
HARD_STOP_S="${HARD_STOP_S:-17000}"; SELF_STOP_S="${SELF_STOP_S:-17600}"
OUT=runs/exp53/pod; SM=runs/exp53/smoke; PY="${PYTHON:-python}"; [ "$ENGINE" = stub ] && PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT" "$SM"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "exp53 pod; engine $ENGINE; hard stop +$HARD_STOP_S s; self-stop +$SELF_STOP_S s"
if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo self-stop >> '$OUT/pod.log'; if command -v runpodctl >/dev/null && \
    [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  git config --global --add safe.directory '*'
fi
for f in runs/exp53/manifest.json runs/exp53/manifest_committed.json .git scripts/exp53_run.py runs/exp53/lme_rows.jsonl; do
  test -e "$f" || { log "missing $f"; exit 3; }; done
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==5.17.0" accelerate huggingface_hub \
    safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import transformers, sklearn, scripts.exp53_run; assert transformers.__version__ == '5.17.0'" || { log "install check failed"; exit 4; }
}
if [ "$ENGINE" != stub ]; then
  until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
  install_line
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""; [ -n "$INSTALL" ] && install_line
fi
set +e   # errexit + pipefail would end the chain at this pipe with exit 1 (the §51 D6b lesson)
$PY -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts'); import exp53_run as R; from pathlib import Path; \
R.require_manifest(Path('$OUT')); print('manifest guard ok')" 2>&1 | tee -a "$OUT/pod.log"; GRC=${PIPESTATUS[0]}; set -e
[ "$GRC" -eq 0 ] || { log "MANIFEST GUARD FAILED: no N-held row read"; exit 4; }
[ "$ENGINE" = stub ] && { log "stub: guard + install only"; exit 0; }
SNAP=$($PY -c "import huggingface_hub; print(huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct', \
revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))")
log "snapshot $SNAP"
step() {
  local name="$1"; shift; local rc=0
  set +e; timeout "$(left)" $PY scripts/exp53_run.py "$@" 2>&1 | grep -v "Loading weights" | tee -a "$OUT/$name.log"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
step smoke --stage pod --smoke --root "$SM" --limit-rows 39 --snapshot "$SNAP" --device cuda || { log "SMOKE FAILED"; exit 5; }
step smoke_report --stage report --smoke --root "$SM" || { log "SMOKE REPORT FAILED"; exit 5; }
set +e; step run --stage pod --root "$OUT" --snapshot "$SNAP" --device cuda --deadline-s "$(( $(left) - 300 ))"; RC=$?; set -e
step report --stage report --root "$OUT" || true
log "pod done at +$(( $(date +%s) - T0 )) s (run rc $RC): copy back runs/exp53/pod and runs/exp53/smoke now"
exit $RC
