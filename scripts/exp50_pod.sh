#!/usr/bin/env bash
# §50 pod stage (research/proposals/2026-11-02-exp50-why-false-abstention.md rev 1 + rev 1a; D-70): one pod session,
# Qwen2.5-3B in float32 on CUDA (dev AND held on one device and precision), transformers 5.17.0, single rows. Never
# two pods. Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch kept). Built from scripts/exp49_pod.sh.
#
#   RUNPOD_POD_ID=<id> bash scripts/exp50_pod.sh                     # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp50_pod.sh                            # CPU: tiny random Qwen2 stub check + the fake chain
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp50_pod.sh   # D-30: the SAME install line
#   ENGINE=stub PROJECT_LIMIT=0 bash scripts/exp50_pod.sh            # the projection must stop the chain (exit 7)
#
# Order: preflight -> watchdog -> GPU check + install -> population (cases sha from bundle_info; exit 3) -> stub check
# (tiny random Qwen2: 3 dev rows measured; a rerun redoes none; exit 5) -> download the 3B -> D-22 smoke (the whole
# chain on dev bases only, runs/exp50_smoke/run, stage 2 forced; no-op replay must hold; exit 5) -> projection (exit 7
# above PROJECT_LIMIT = $3.00 at RATE) -> dev measure -> stage2-dev -> dev-summary (writes dev.ready; no held file
# exists yet) -> wait-mark (the laptop pulls the dev files for the Advisor's gate read and audit, writes and commits
# design.json after "design OK", ships it with a shallow clone holding that commit as .git and the marker; the
# git-verified guard must pass; exit 10 timeout, 11 refused) -> held measure -> stage1-report -> stage2-held (stops at
# STAGE2_STOP_EPOCH, rev 1a item 3) -> report -> status.
# Deadlines: HARD_STOP_S after start all model work ends (stages stop between rows, exit 9, resumable). The watchdog
# stops the pod at SELF_STOP_S with `runpodctl stop pod $RUNPOD_POD_ID`; the laptop DELETE is armed separately.
# No outcome is printed (the smoke prints dev-base counts only). Copy back: $OUT, runs/exp50/run, runs/exp50_smoke.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-9000}"
SELF_STOP_S="${SELF_STOP_S:-9900}"
MARK_WAIT_S="${MARK_WAIT_S:-3600}"
STAGE2_STOP_EPOCH="${STAGE2_STOP_EPOCH:-0}"
PROJECT_LIMIT="${PROJECT_LIMIT:-$([ "${ENGINE:-hf}" = stub ] && echo 1000 || echo 3.00)}"
RATE="${RATE:-0.59}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp50_pod_stub}"; PY="${PYTHON:-.venv/bin/python}"; DEV=cpu
  RES="${RES:-$OUT/run}"; SMOKE="$OUT/smoke/run"; BK="--backend fake"; MAIN_W=""
else
  OUT="${OUT:-runs/exp50_pod}"; PY="${PYTHON:-python}"; DEV=cuda
  RES=runs/exp50/run; SMOKE=runs/exp50_smoke/run; BK="--backend hf --dtype float32"; MAIN_W=""
fi
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
STUBS="$OUT/stub_models"
DONE_FILE="$OUT/pod.done"
TIMES="$OUT/stage_times.json"
mkdir -p "$OUT"
rm -f "$DONE_FILE" "$TIMES"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$DONE_FILE"' EXIT
echo "exp50 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; run -> $RES" | tee -a "$OUT/pod.log"
echo "deadlines: HARD_STOP_S=$HARD_STOP_S s, SELF_STOP_S=$SELF_STOP_S s, MARK_WAIT_S=$MARK_WAIT_S s; projection limit \$$PROJECT_LIMIT at \$$RATE/h" | tee -a "$OUT/pod.log"

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(pyproject.toml runs/exp50/cases50.jsonl runs/exp50/cases50_counts.json
        scripts/exp45_cases.py scripts/exp49_cases.py scripts/exp49_run.py scripts/exp49_pod_run.py scripts/exp50_cases.py
        scripts/exp50_run.py scripts/exp50_report.py scripts/exp50_pod_run.py
        memctl/analysis/abst49.py memctl/analysis/why50.py memctl/analysis/patch46.py memctl/analysis/probe46.py
        memctl/analysis/maps.py memctl/analysis/steering.py memctl/analysis/answer_selection.py)
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
command -v git >/dev/null || missing "git (the held guard verifies the design commit with git)"
EXPECT=""
if [ -e bundle_info.json ]; then
  EXPECT=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('cases50_sha256',''))" 2>/dev/null || true)
fi
echo "preflight ok: ${#INPUTS[@]} files present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
stage_done() {   # stage_done <name> <status> <rc> [note]
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp50_pod_run.py stage-time --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
    --elapsed $(( now - STAGE_START )) --t-offset $(( STAGE_START - T0 )) --note "${4:-}" \
    --hard-stop-s "$HARD_STOP_S" --self-stop-s "$SELF_STOP_S" --engine "$ENGINE" \
    || echo "stage-time: could not write $TIMES" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
left() { echo $(( DEADLINE - $(date +%s) )); }
status_of() { [ "$1" -eq 0 ] && echo ok || { { [ "$1" -eq 124 ] || [ "$1" -eq 9 ]; } && echo partial || echo failed; }; }

install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==${TRANSFORMERS_VERSION}" accelerate \
    huggingface_hub safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import torch, transformers, numpy, sklearn, huggingface_hub, safetensors, accelerate; \
import memctl.analysis.abst49, memctl.analysis.why50, memctl.analysis.steering, memctl.analysis.probe46, \
scripts.exp49_run, scripts.exp50_run, scripts.exp50_report, scripts.exp50_pod_run; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'sklearn', sklearn.__version__); \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
}

if [ "$ENGINE" != stub ]; then
  GPU_DEADLINE=$(( $(date +%s) + 120 ))
  until nvidia-smi > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$GPU_DEADLINE" ]; then echo "GPU not visible after 120 s" | tee -a "$OUT/pod.log"; exit 4; fi
    sleep 5
  done
  install_line
  $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; p = torch.cuda.get_device_properties(0); \
print('gpu', p.name, round(p.total_memory / 2**30, 1), 'GiB')" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "CUDA check failed" | tee -a "$OUT/pod.log"; exit 4; }
  export TOKENIZERS_PARALLELISM=false
  stage_done install ok 0
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  if [ -n "$INSTALL" ]; then install_line; stage_done install ok 0; fi
fi

set +e
$PY scripts/exp50_pod_run.py population ${EXPECT:+--expect-cases-sha "$EXPECT"} 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "population check failed (exit $RC)" | tee -a "$OUT/pod.log"; stage_done population failed "$RC"; exit 3; }
stage_done population ok 0

run_step() {   # run_step <log name> <timeout s> <command...>
  local name="$1" t="$2"; shift 2
  if [ "$t" -le 30 ]; then echo "SKIP $name: only $t s left" | tee -a "$OUT/pod.log"; return 124; fi
  set +e
  timeout "$t" "$@" 2>&1 | grep -v -E "unauthenticated|Writing model shards|Loading weights" | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}
DL() { echo $(( $(left) - 30 )); }

# Stub check on this device: tiny random Qwen2 (36 layers) + the real tokenizer, 3 dev rows measured; a rerun must
# redo none.
$PY scripts/exp50_pod_run.py make-stub --out "$STUBS" 2>&1 | { grep -v -E "unauthenticated|Writing model shards" || true; } | tee -a "$OUT/pod.log"
SC=0
run_step stubcheck "$(left)" $PY scripts/exp50_run.py --stage measure --split dev --limit 3 --backend hf --model "$STUBS/qwen3b" --dtype float32 --out "$OUT/stubcheck" || SC=$?
if [ "$SC" -eq 0 ]; then
  N1=$(ls "$OUT/stubcheck/rows" | wc -l)
  M1=$(stat -c %Y "$OUT"/stubcheck/rows/*.json | sort | tail -1)
  run_step stubcheck "$(left)" $PY scripts/exp50_run.py --stage measure --split dev --limit 3 --backend hf --model "$STUBS/qwen3b" --dtype float32 --out "$OUT/stubcheck" || SC=$?
  M2=$(stat -c %Y "$OUT"/stubcheck/rows/*.json | sort | tail -1)
  echo "stub check: $N1 rows, rerun rewrote $([ "$M1" = "$M2" ] && echo none || echo some)" | tee -a "$OUT/pod.log"
  { [ "$N1" = 3 ] && [ "$M1" = "$M2" ]; } || SC=6
fi
if [ "$SC" -ne 0 ]; then echo "STUB CHECK FAILED (rc $SC): chain aborted before any download" | tee -a "$OUT/pod.log"; stage_done stub_check failed "$SC"; exit 5; fi
stage_done stub_check ok 0

if [ "$ENGINE" != stub ]; then
  set +e
  timeout $(( $(left) > 1800 ? 1800 : $(left) )) $PY - <<'PYEOF' 2>&1 | tee -a "$OUT/pod.log"
import time, huggingface_hub
t = time.time()
path = huggingface_hub.snapshot_download("Qwen/Qwen2.5-3B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
print(f"download Qwen/Qwen2.5-3B-Instruct: {time.time() - t:.0f} s -> {path}", flush=True)
PYEOF
  RC=${PIPESTATUS[0]}
  set -e
  [ "$RC" -eq 0 ] || { stage_done download failed "$RC"; echo "download failed (exit $RC)" | tee -a "$OUT/pod.log"; exit 4; }
  stage_done download ok 0
fi

# D-22 smoke: the whole chain on dev bases only (k >= 3 relabelled as stand-in held), outside runs/exp50/.
SM=0
run_step smoke "$(left)" $PY scripts/exp50_run.py --smoke 6 $BK --out "$SMOKE" || SM=$?
if [ "$SM" -ne 0 ]; then echo "SMOKE FAILED (rc $SM): chain aborted before the full run" | tee -a "$OUT/pod.log"; stage_done smoke failed "$SM"; exit 5; fi
stage_done smoke ok 0

set +e
$PY scripts/exp50_pod_run.py project --smoke "$SMOKE/smoke.json" --elapsed $(( $(date +%s) - T0 )) --rate "$RATE" \
  --limit "$PROJECT_LIMIT" 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then echo "PROJECTION above \$$PROJECT_LIMIT (rc $RC): chain stopped; ask before continuing" | tee -a "$OUT/pod.log"; stage_done projection stop "$RC"; exit 7; fi
stage_done projection ok 0

step() {   # step <name> <args...>: one exp50_run stage under the deadline; stops the chain unless it completed
  local name="$1"; shift
  local rc=0
  run_step "$name" "$(left)" $PY scripts/exp50_run.py "$@" --out "$RES" || rc=$?
  stage_done "$name" "$(status_of "$rc")" "$rc"
  if [ "$rc" -ne 0 ]; then echo "stage $name did not complete (rc $rc): chain stops here (resumable)" | tee -a "$OUT/pod.log"; exit "$rc"; fi
}
step measure_dev --stage measure --split dev $BK --deadline-s "$(DL)"
step stage2_dev --stage stage2-dev $BK --deadline-s "$(DL)"
step dev_summary --stage dev-summary
touch "$RES/dev.ready"
echo "dev.ready: pull the dev files for the gate read and audit; after design OK the laptop writes, commits and ships design.json + .git + design_committed.json (waiting up to $MARK_WAIT_S s)" | tee -a "$OUT/pod.log"
W=$(( MARK_WAIT_S < $(left) ? MARK_WAIT_S : $(left) ))
set +e
$PY scripts/exp50_pod_run.py wait-mark --root "$RES" --timeout "$W" 2>&1 | tee -a "$OUT/pod.log"
WM=${PIPESTATUS[0]}
set -e
stage_done wait_mark "$([ "$WM" -eq 0 ] && echo ok || echo failed)" "$WM"
[ "$WM" -eq 0 ] || { echo "no verified design mark (rc $WM): held stages not run" | tee -a "$OUT/pod.log"; exit "$WM"; }
step measure_held --stage measure --split held $BK --deadline-s "$(DL)"
step stage1_report --stage stage1-report
S2=0; run_step stage2_held "$(left)" $PY scripts/exp50_run.py --stage stage2-held $BK --out "$RES" --deadline-s "$(DL)" \
  $([ "$STAGE2_STOP_EPOCH" != 0 ] && echo "--stop-at-epoch $STAGE2_STOP_EPOCH") || S2=$?
stage_done stage2_held "$(status_of "$S2")" "$S2" "$([ "$S2" -eq 9 ] && echo 'stage 2 not completed (time budget)')"
RP=0; run_step report "$(left)" $PY scripts/exp50_run.py --stage report --out "$RES" > /dev/null || RP=$?
stage_done report "$(status_of "$RP")" "$RP"
$PY scripts/exp50_pod_run.py status --out "$OUT" --root "$RES" 2>&1 | tee -a "$OUT/pod.log"
echo "all stages done in $(( $(date +%s) - T0 )) s (report $RP); copy back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
[ "$RP" -eq 0 ]
