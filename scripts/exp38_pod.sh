#!/usr/bin/env bash
# §38 pod stage (research/proposals/2026-10-22-exp38-refusal-direction.md §6, amendments 1-2; dev review addendum
# 7e2c90d): one A40, one engine for every role: transformers 5.17.0, bf16, CUDA, the 3B with forward hooks for the
# steering and Qwen2.5-7B-Instruct for the §35 judge (greedy, 4 tokens). No vLLM. Image:
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch is kept: pip is constrained to the installed version).
#
#   RUNPOD_POD_ID=<id> bash scripts/exp38_pod.sh        # on the pod, from the unpacked tree (see scripts/exp38_bundle.sh)
#   ENGINE=stub bash scripts/exp38_pod.sh               # CPU, stub engine + stub judge, no install, writes runs/exp38_stub
#   ENGINE=stub STUB_JUDGE=silent bash scripts/exp38_pod.sh   # the smoke must fail (exit 5)
#
# Chain: preflight inputs -> install -> GPU check -> self-stop watchdog -> D-22 smoke (5 dev rows: reproduction, one
# Steer, the judge; a reply that is not a yes/no verdict aborts the chain, exit 5) -> run (reproduction folds 0-4,
# directions, sanity S, primary arms, then secondary arms while the deadline allows) -> judge (7B, after the 3B process
# has exited) -> report with --confirm --design runs/exp38_dev/qwen3b/gate.json -> pod.done.
# Deadlines: all model work ends by HARD_STOP_S (default 2700 s = 45 min) after this script starts: each stage runs
# under `timeout` with what is left, and the run keeps JUDGE_RESERVE_S (default 420 s) free for the judge, skipping
# secondary arms (logged in plan.json) that its estimate says would not fit. A watchdog started here stops the pod
# itself at SELF_STOP_S (default 3300 s = 55 min) with `runpodctl stop pod $RUNPOD_POD_ID` whatever state the chain is
# in; copy $OUT back before then. RUNPOD_POD_ID must be passed in by the launch command; it is exported here.
# Every wait loop has a deadline. No `pgrep -f`. The sentinel $OUT/pod.done is written on any exit.
# Exit 3: input missing (preflight) or reproduction below 0.8. 4: GPU did not come up, or sanity S failed. 5: smoke
# failed. 6: population counts differ from the §35 cache. 124: a stage was cut by the deadline.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-2700}"
SELF_STOP_S="${SELF_STOP_S:-3300}"
JUDGE_RESERVE_S="${JUDGE_RESERVE_S:-420}"
STUB_JUDGE="${STUB_JUDGE:-d1}"
DESIGN="${DESIGN:-runs/exp38_dev/qwen3b/gate.json}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp38_stub}"; PY="${PYTHON:-.venv/bin/python}"; BACKEND=stub; REPORT="${REPORT:-runs/exp38_report_stub.md}"
else
  OUT="${OUT:-runs/exp38_pod}"; PY="${PYTHON:-python}"; BACKEND=hf; REPORT="${REPORT:-runs/exp38_report_confirm.md}"
fi
DONE_FILE="$OUT/pod.done"
mkdir -p "$OUT"
rm -f "$DONE_FILE"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT
echo "§38 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; hard stop +${HARD_STOP_S} s, self-stop +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"

# The self-stop watchdog: detached, so it outlives this script; its pid is kept in a file (never searched by name).
if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

# Preflight: every input the population loader, the report and the judge read (scripts/exp38_bundle.sh packs them).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json "$DESIGN" runs/exp38_dev/qwen3b/directions.npz
        scripts/exp38_pod_run.py scripts/exp38_report.py scripts/exp38_dev.py scripts/exp35_generate.py scripts/exp35_report.py
        memctl/analysis/steering.py memctl/analysis/answer_selection.py)
for READER in qwen3b qwen7b; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp23_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
  test -d "cache/generations_exp23_${READER}" || missing "cache/generations_exp23_${READER}/"
done
for FOLD in 0 1 2 3 4; do
  for ARM in A B Ncut judge; do INPUTS+=("runs/exp35/qwen3b/${ARM}/fold${FOLD}.jsonl"); done
done
test -d runs/exp34_jlens_own/qwen3b || missing runs/exp34_jlens_own/qwen3b/
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
if [ -z "${HF_TOKEN:-}" ] && [ -s /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(tr -d '[:space:]' < /workspace/.hf_token); fi
if [ "$ENGINE" != stub ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN not set and /workspace/.hf_token absent; Hugging Face may rate-limit the model download" >&2
fi
echo "preflight ok: ${#INPUTS[@]} files and 3 folders present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$T0
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
run_stage() {  # run_stage NAME CMD...: killed at the deadline; any nonzero exit ends the chain with that code.
  local name=$1; shift
  local budget=$(( DEADLINE - $(date +%s) ))
  if [ "$budget" -le 0 ]; then echo "deadline reached before stage $name" | tee -a "$OUT/pod.log"; exit 124; fi
  set +e
  timeout "$budget" "$@" 2>&1 | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 124 ]; then echo "stage $name cut by the deadline" | tee -a "$OUT/pod.log"; exit 124; fi
  if [ "$rc" -eq 5 ]; then echo "stage $name: SMOKE FAILED; chain aborted before the full run (exit 5)" | tee -a "$OUT/pod.log"; exit 5; fi
  if [ "$rc" -ne 0 ]; then echo "stage $name failed: exit $rc" | tee -a "$OUT/pod.log"; exit "$rc"; fi
  stage_done "$name"
}

if [ "$ENGINE" != stub ]; then
  GPU_DEADLINE=$(( $(date +%s) + 120 ))
  until nvidia-smi > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$GPU_DEADLINE" ]; then echo "GPU not visible after 120 s" | tee -a "$OUT/pod.log"; exit 4; fi
    sleep 5
  done
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"   # keep the image's CUDA torch; never let pip replace it
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==${TRANSFORMERS_VERSION}" accelerate 2>&1 | tail -2
  $PY -c "import torch, transformers; assert torch.cuda.is_available(), 'CUDA not available'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'transformers', transformers.__version__); \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}', transformers.__version__" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
  export TOKENIZERS_PARALLELISM=false
  stage_done install
else
  export HF_HUB_OFFLINE=1
fi

COMMON=(--out "$OUT" --design "$DESIGN" --backend "$BACKEND" --stub-judge "$STUB_JUDGE")
run_stage smoke $PY scripts/exp38_pod_run.py smoke "${COMMON[@]}"
run_stage run $PY scripts/exp38_pod_run.py run "${COMMON[@]}" --deadline "$DEADLINE" --judge-reserve "$JUDGE_RESERVE_S"
run_stage judge $PY scripts/exp38_pod_run.py judge "${COMMON[@]}" --deadline "$DEADLINE"
REPORT_DESIGN="$DESIGN"
if [ "$ENGINE" = stub ]; then REPORT_DESIGN="$OUT/qwen3b/design_stub.json"; fi
# The report is CPU only and quick; it runs after the deadline if need be (before the self-stop).
set +e
timeout 600 $PY scripts/exp38_report.py --run-dir "$OUT/qwen3b" --confirm --design "$REPORT_DESIGN" --out "$REPORT" \
  > "$OUT/report.log" 2>&1
RC=$?
set -e
cp -f "${REPORT%.md}.md" "${REPORT%.md}.json" "$OUT/" 2>/dev/null || true
echo "report exit $RC: $REPORT (copied into $OUT/)" | tee -a "$OUT/pod.log"
stage_done report
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
exit "$RC"
