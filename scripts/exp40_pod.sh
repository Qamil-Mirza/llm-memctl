#!/usr/bin/env bash
# §40 pod stage (research/proposals/2026-10-23-exp40-fixed-setting-validation.md rev 1 at 877f86d; the Advisor's
# R/C selection ruling of 2026-10-23; D-34): one A40, transformers 5.17.0, bf16, CUDA; the 3B with forward hooks and
# Qwen2.5-7B-Instruct (the §35 judge) resident in one process. No vLLM. Image:
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch is kept: pip is constrained to the installed version).
#
#   RUNPOD_POD_ID=<id> bash scripts/exp40_pod.sh            # on the pod, from the unpacked tree (scripts/exp40_bundle.sh)
#   ENGINE=stub bash scripts/exp40_pod.sh                   # CPU, stub engine + stub judge, no install, runs/exp40_stub
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp40_pod.sh   # D-30: the SAME install line, then stub
#   ENGINE=stub STUB_JUDGE=silent bash scripts/exp40_pod.sh       # the smoke must fail (exit 5)
#   ENGINE=stub STUB_JUDGE=disagree bash scripts/exp40_pod.sh     # the agreement check must fail (exit 7)
#   EXP=exp42 ...                                                # §42 (proposal 2026-10-25 rev 1): seed 42, §40's
#                                                                # 640 ids excluded, rule42.json; out runs/exp42_pod
#
# Chain: preflight inputs -> install -> GPU check -> self-stop watchdog -> run (one process: the set and the direction
# asserts, both models loaded, D-22 smoke [exit 5], judge agreement on 60 cached §35 rows [exit 7, then the pod stops
# itself at once], N pass on 640 rows batched, A pass, the judge on the A answers, R/C selection, arms in primary-first
# order, the judge on every R/C output) -> report --confirm -> pod.done.
# Deadlines: all model work ends by HARD_STOP_S (default 2700 s = 45 min) after this script starts (the run stage runs
# under `timeout` with what is left; inside it the arms stop JUDGE_RESERVE_S, default 300 s, before the deadline so the
# final judge pass fits). A watchdog started here stops the pod itself at SELF_STOP_S (default 3300 s = 55 min) with
# `runpodctl stop pod $RUNPOD_POD_ID` whatever state the chain is in; copy $OUT back before then. RUNPOD_POD_ID must be
# passed in by the launch command. Every wait loop has a deadline. No `pgrep -f`. $OUT/pod.done is written on any exit.
# Exit 2: setting refused. 3: input missing. 4: GPU / install. 5: smoke failed. 7: judge agreement below 0.90 (the
# pod is stopped by this script right away, as the watchdog would). 124: the run stage was cut by the deadline.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-2700}"
SELF_STOP_S="${SELF_STOP_S:-3300}"
JUDGE_RESERVE_S="${JUDGE_RESERVE_S:-300}"
STUB_JUDGE="${STUB_JUDGE:-d1}"
RULE="${RULE:-runs/exp41_dev/qwen3b/rule.json}"
EXP="${EXP:-exp40}"
RULE42="${RULE42:-runs/exp42_inputs/rule42.json}"
case "$EXP" in exp40|exp42) ;; *) echo "EXP must be exp40 or exp42, not $EXP" >&2; exit 2;; esac
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/${EXP}_stub}"; PY="${PYTHON:-.venv/bin/python}"; BACKEND=stub; REPORT="${REPORT:-$OUT/${EXP}_report_stub.md}"
else
  OUT="${OUT:-runs/${EXP}_pod}"; PY="${PYTHON:-python}"; BACKEND=hf; REPORT="${REPORT:-$OUT/${EXP}_report_confirm.md}"
fi
DONE_FILE="$OUT/pod.done"
mkdir -p "$OUT"
rm -f "$DONE_FILE"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT
echo "$EXP pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; hard stop +${HARD_STOP_S} s, self-stop +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"

stop_pod_now() {  # what the watchdog does, now
  echo "self-stop ($1) at $(date -u +%FT%TZ)" | tee -a "$OUT/pod.log"
  if [ "$ENGINE" = stub ]; then echo "stub: no pod to stop" | tee -a "$OUT/pod.log"; return 0; fi
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "self-stop: runpodctl failed; STOP THE POD BY HAND" | tee -a "$OUT/pod.log"
  else
    echo "self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND" | tee -a "$OUT/pod.log"
  fi
}

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

# Preflight: every input the run and the report read (scripts/exp40_bundle.sh packs exactly these and the code).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/locomo/locomo10.json runs/exp38_dev/qwen3b/directions.npz runs/exp40_inputs/agreement60.jsonl
        scripts/exp40_pod_run.py scripts/exp40_report.py scripts/exp38_pod_run.py scripts/exp35_generate.py
        memctl/locomo_view.py memctl/analysis/exp40.py memctl/analysis/steering.py memctl/analysis/conditional.py
        memctl/analysis/answer_selection.py pyproject.toml)
if [ "$EXP" = exp42 ]; then INPUTS+=("$RULE42" runs/exp40_pod/qwen3b/view.json); fi
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
if [ -e "$RULE" ]; then echo "preflight: §41 rule present ($RULE)" | tee -a "$OUT/pod.log"
else echo "preflight: no §41 rule at $RULE: signals are logged, the rule is not scored" | tee -a "$OUT/pod.log"; fi
if [ -z "${HF_TOKEN:-}" ] && [ -s /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(tr -d '[:space:]' < /workspace/.hf_token); fi
if [ "$ENGINE" != stub ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN not set and /workspace/.hf_token absent; Hugging Face may rate-limit the model download" >&2
fi
echo "preflight ok: ${#INPUTS[@]} files present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$T0
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}

# The install line (the pod's, and the D-30 fresh-venv test's: one line, used by both).
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"   # keep the installed torch; never let pip replace it
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==${TRANSFORMERS_VERSION}" accelerate scipy 2>&1 | tail -2
  $PY -c "import torch, transformers, scipy, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, \
'scipy', scipy.__version__, 'numpy', numpy.__version__); assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 \
    | tee -a "$OUT/pod.log" || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
}

if [ "$ENGINE" != stub ]; then
  GPU_DEADLINE=$(( $(date +%s) + 120 ))
  until nvidia-smi > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$GPU_DEADLINE" ]; then echo "GPU not visible after 120 s" | tee -a "$OUT/pod.log"; exit 4; fi
    sleep 5
  done
  install_line
  $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('gpu', torch.cuda.get_device_name(0))" \
    2>&1 | tee -a "$OUT/pod.log" || { echo "CUDA check failed" | tee -a "$OUT/pod.log"; exit 4; }
  export TOKENIZERS_PARALLELISM=false
  stage_done install
else
  export HF_HUB_OFFLINE=1
  if [ -n "$INSTALL" ]; then install_line; stage_done install; fi
fi

BUDGET=$(( DEADLINE - $(date +%s) ))
if [ "$BUDGET" -le 0 ]; then echo "deadline reached before the run" | tee -a "$OUT/pod.log"; exit 124; fi
set +e
timeout "$BUDGET" $PY scripts/exp40_pod_run.py run --out "$OUT" --backend "$BACKEND" --stub-judge "$STUB_JUDGE" \
  --rule "$RULE" --exp "$EXP" --rule42 "$RULE42" --deadline "$DEADLINE" --judge-reserve "$JUDGE_RESERVE_S" 2>&1 | tee -a "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e
stage_done run
if [ "$RC" -eq 7 ]; then
  echo "JUDGE AGREEMENT BELOW 0.90: chain ends before any arm (exit 7)" | tee -a "$OUT/pod.log"
  stop_pod_now "agreement below the bar"
  exit 7
fi
if [ "$RC" -eq 5 ]; then echo "SMOKE FAILED: chain aborted before the note pass (exit 5)" | tee -a "$OUT/pod.log"; exit 5; fi
if [ "$RC" -eq 124 ]; then echo "run stage cut by the deadline; reporting the rows written so far" | tee -a "$OUT/pod.log"; fi
if [ "$RC" -ne 0 ] && [ "$RC" -ne 124 ]; then echo "run stage failed: exit $RC" | tee -a "$OUT/pod.log"; exit "$RC"; fi
# The report is CPU only; it runs after the deadline if need be (before the self-stop), also on partial rows.
set +e
timeout 600 $PY scripts/exp40_report.py --run-dir "$OUT/qwen3b" --confirm --rule "$RULE" --exp "$EXP" --rule42 "$RULE42" --out "$REPORT" > "$OUT/report.log" 2>&1
REP=$?
set -e
echo "report exit $REP: $REPORT" | tee -a "$OUT/pod.log"
stage_done report
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
if [ "$RC" -ne 0 ]; then exit "$RC"; fi
exit "$REP"
