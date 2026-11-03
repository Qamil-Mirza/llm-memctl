#!/usr/bin/env bash
# §51 timed GPU measurement pod (proposed; runs only after its own quote and the user's approval line in the
# Executor pane). Qwen2.5-3B in strict float32 (no TF32) on CUDA, transformers 5.17.0. Never two pods.
#
#   RUNPOD_POD_ID=<id> bash scripts/exp51_pod_measure.sh                 # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp51_pod_measure.sh                        # CPU: the dummy model, no download
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp51_pod_measure.sh   # D-30 (same install line)
#
# Order: preflight -> watchdog (self-stop at SELF_STOP_S) -> GPU check + install -> inputs sha (exit 3) -> download the
# 3B -> stub check = D-22 smoke (the dummy tiny Qwen2 built from the downloaded config and tokenizer, the whole
# measurement on 8 rows; exit 5) -> the measurement (64 E rows, budget MEASURE_S) -> pod.done. The laptop DELETE
# watchdog is armed separately at launch. Copy back: $OUT.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"; INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
SELF_STOP_S="${SELF_STOP_S:-1800}"
MEASURE_S="${MEASURE_S:-900}"
OUT="${OUT:-runs/exp51/gpu_measure/pod}"
PY="${PYTHON:-python}"
[ "$ENGINE" = stub ] && PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s)
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$*" | tee -a "$OUT/pod.log"; }
log "exp51 measurement chain; engine $ENGINE; self-stop +$SELF_STOP_S s; measure budget $MEASURE_S s"
if [ "$ENGINE" != stub ]; then
  [ -n "${RUNPOD_POD_ID:-}" ] || log "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire"
  nohup setsid bash -c "sleep $SELF_STOP_S; echo self-stop >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" > "$OUT/watchdog.log" 2>&1 < /dev/null &
  log "watchdog pid $!"
fi
for f in pyproject.toml scripts/exp51_engine.py scripts/exp51_common.py scripts/exp51_gpu_measure.py \
         memctl/analysis/edit51.py memctl/analysis/grade51.py runs/exp51/gpu_measure/inputs.jsonl \
         runs/exp51/cases51_counts.json; do
  test -e "$f" || { log "preflight: missing $f"; exit 3; }
done
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==${TRANSFORMERS_VERSION}" accelerate \
    huggingface_hub safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import torch, transformers, numpy, safetensors, huggingface_hub; \
import memctl.analysis.edit51, memctl.analysis.grade51, scripts.exp51_engine, scripts.exp51_gpu_measure; \
print('torch', torch.__version__, 'transformers', transformers.__version__); \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 | tee -a "$OUT/pod.log" || { log "install check failed"; exit 4; }
}
if [ "$ENGINE" != stub ]; then
  until nvidia-smi > /dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "GPU not visible"; exit 4; }; sleep 5; done
  install_line
  DEV=cuda
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  [ -n "$INSTALL" ] && install_line
  DEV=cpu
fi
if [ -e bundle_info.json ]; then
  WANT=$($PY -c "import json; print(json.load(open('bundle_info.json'))['inputs_sha256'])")
  [ "$(sha256sum runs/exp51/gpu_measure/inputs.jsonl | cut -d' ' -f1)" = "$WANT" ] || { log "inputs sha mismatch"; exit 3; }
fi
log "preflight ok at +$(( $(date +%s) - T0 )) s"
if [ "$ENGINE" != stub ]; then
  SNAP=$($PY - <<'PY'
import huggingface_hub
print(huggingface_hub.snapshot_download("Qwen/Qwen2.5-3B-Instruct",
      revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"]))
PY
)
  log "download done at +$(( $(date +%s) - T0 )) s -> $SNAP"
else
  SNAP=$($PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; print(E.SNAPSHOT)")
fi
# D-22 smoke = stub check: the dummy tiny Qwen2 (real config shape scaled down, real tokenizer), the whole measurement
$PY -c "import sys; sys.path.insert(0,'scripts'); import exp51_engine as E; from pathlib import Path; \
E.make_dummy(Path('$OUT/dummy'), Path('$SNAP'))" 2>&1 | { grep -v -E "Writing model shards" || true; } | tee -a "$OUT/pod.log"
set +e
timeout 600 $PY scripts/exp51_gpu_measure.py --backend hf --snapshot "$OUT/dummy" --device "$DEV" --n 8 --batches 4,8 \
  --budget-s 300 --out "$OUT/smoke" 2>&1 | grep -v "Loading weights" | tee -a "$OUT/pod.log"
SM=${PIPESTATUS[0]}
set -e
{ [ "$SM" -eq 0 ] && grep -q '"done": true' "$OUT/smoke/measure.json"; } || { log "SMOKE FAILED (rc $SM)"; exit 5; }
log "smoke ok at +$(( $(date +%s) - T0 )) s"
[ "$ENGINE" = stub ] && { log "stub engine: measurement on the dummy only"; exit 0; }
set +e
timeout $(( MEASURE_S + 300 )) $PY scripts/exp51_gpu_measure.py --backend hf --snapshot "$SNAP" --device cuda --n 64 \
  --batches 32,64 --budget-s "$MEASURE_S" --out "$OUT/measure" 2>&1 | grep -v "Loading weights" | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
log "measurement rc $RC at +$(( $(date +%s) - T0 )) s; copy back $OUT now"
exit "$RC"
