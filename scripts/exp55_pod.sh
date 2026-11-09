#!/usr/bin/env bash
# §55 pod (rev 0b, bc8e145; user approval in the Executor pane: one A40, ~$1.00, hard stop $1.50 / 150 min).
# Order: preflight -> GPU + install (D-30 line) -> snapshot -> smoke (D-22: 8 W rows + 2 LME rows on a scratch root,
# exit 5) -> check20 -> W free (P0, P1, P2) -> W prefix (P1, P2) -> LME P0, P1, P2 (P2 last: the frozen drop order;
# a stage stops between rows at the deadline, exit 9, resumable) -> pod.done. Copy back runs/exp55/pod.
#   RUNPOD_POD_ID=<id> RUN_S=<seconds of run budget> bash scripts/exp55_pod.sh
#   ENGINE=stub PYTHON=<venv>/bin/python INSTALL=1 bash scripts/exp55_pod.sh      # CPU: preflight + install only (D-30)
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"; INSTALL="${INSTALL:-}"; RUN_S="${RUN_S:-7200}"
OUT=runs/exp55/pod; PY="${PYTHON:-python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + RUN_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "exp55 pod; engine $ENGINE; run budget $RUN_S s"
for f in runs/exp55/freeze.json runs/exp55/rows_W.jsonl runs/exp55/rows_LME.jsonl runs/exp55/lme_prompts.jsonl \
         runs/exp54/cases54.jsonl runs/exp51/cases51_counts.json scripts/exp55_run.py scripts/exp51_engine.py; do
  test -e "$f" || { log "missing $f"; exit 3; }; done
sha() { sha256sum "$1" | cut -d' ' -f1; }
[ "$(sha runs/exp55/freeze.json)" = 24ec9eb6d9e8b4f3c8b6620657325cb6c91b4581d6c639b090f039512347399f ] || { log "freeze sha mismatch"; exit 3; }
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==5.17.0" accelerate huggingface_hub \
    safetensors 2>&1 | tail -2
  $PY -c "import sys; sys.path.insert(0,'scripts'); import transformers, exp55_run; assert transformers.__version__ == '5.17.0'" \
    || { log "install check failed"; exit 4; }
}
if [ "$ENGINE" = stub ]; then
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""; [ -n "$INSTALL" ] && install_line; log "stub: preflight + install only"; exit 0
fi
until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$OUT/pod.log"
install_line
SNAP=$($PY -c "import huggingface_hub; print(huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct', \
revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))")
log "snapshot $SNAP"
step() {
  local name="$1"; shift; local rc=0
  set +e; timeout "$(left)" $PY -u scripts/exp55_run.py "$@" --pod --device cuda --snapshot "$SNAP" --deadline-s "$(( $(left) - 60 ))" \
    2>&1 | grep -v "Loading weights" | tee -a "$OUT/$name.log"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
step smoke --stage smoke --root "$OUT/smoke" || { log "SMOKE FAILED"; exit 5; }
RC=0
for s in "check20 --stage check20" "wfree --stage wfree" "wprefix --stage wprefix" \
         "lmeP0 --stage lme --arms P0" "lmeP1 --stage lme --arms P1" "lmeP2 --stage lme --arms P2"; do
  set -- $s; name=$1; shift
  set +e; step "$name" "$@" --root "$OUT/run"; rc=$?; set -e
  if [ $rc -ne 0 ]; then RC=$rc; log "stopping after $name (rc $rc)"; break; fi
done
log "pod done at +$(( $(date +%s) - T0 )) s (rc $RC): copy back runs/exp55/pod"
exit $RC
