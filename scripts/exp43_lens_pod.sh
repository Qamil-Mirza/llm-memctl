#!/usr/bin/env bash
# §43 Part B pod stage, one family per pod (proposal 2026-10-26-exp43-cross-family.md amendment 1 Part B; rev 1
# rulings B1-B3; the Advisor's deadline ruling of 2026-10-26: HARD_STOP_S 9900, SELF_STOP_S 10500, under a 3 h
# laptop DELETE). One A40, bf16, CUDA, transformers 5.17.0, the vendored jacobian-lens (581d398). Image:
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch is kept: pip is constrained to the installed version).
#
#   FAMILY=llama RUNPOD_POD_ID=<id> bash scripts/exp43_lens_pod.sh     # on the pod, from the unpacked bundle
#   bash scripts/exp43_lens_pod.sh --family gemma                      # the same; --family overrides FAMILY
#   FAMILY=llama ENGINE=stub bash scripts/exp43_lens_pod.sh            # CPU, tiny random models, runs/exp43_lens_stub_<f>
#   FAMILY=llama ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp43_lens_pod.sh   # D-30: the SAME install line
#   FAMILY=llama ENGINE=stub STUB_BREAK=bos bash scripts/exp43_lens_pod.sh   # the smoke must fail (exit 5)
#
# Order (the deciding logit rows land first): preflight -> install -> GPU check -> watchdog -> population (counts
# 61/65/66/68/70 and the bundle's fingerprint, exit 3) -> downloads (both sizes, bf16 safetensors) -> D-22 smoke (5 dev
# rows through a small and a large logit read: depth, width, BOS count, refusal ids, finite logits, and for the small
# model the final-layer top-1 at P1 against the laptop fp32 dev read; exit 5 before any full read) -> logit reads,
# small (330 rows) -> logit reads, large (330 rows; Phi-4 14B alone on the GPU, reads only) -> J-lens fit, small (56
# WikiText paragraphs, min 32, chunk 4, dim batch 32, as §34; Phi-4-mini with BOS off per families.py) -> J-lens reads,
# small (330 rows, folds 0-1 first, both variants) -> lens eval, small (the library sets, for the §34 gate) -> pod.done.
# No report here: scripts/exp43_lens_report.py runs on the laptop from the copied rows.
#
# Deadlines: HARD_STOP_S (default 9900 s) after this script starts, all model work ends. A stage is skipped (and logged
# as skipped in stage_times.json) when its minimum expected time no longer fits; a read stage that starts stops between
# rows before the deadline (exit 9 in the log, rows so far kept). The fit's time box is what is left after the J-lens
# reads and eval are reserved. A watchdog started here stops the pod itself at SELF_STOP_S (default 10500 s) with
# `runpodctl stop pod $RUNPOD_POD_ID`; copy $OUT back before then. RUNPOD_POD_ID must be passed in the launch command.
# $OUT/pod.done on any exit. $OUT/stage_times.json: seconds per stage, copied back (later quotes are set from it).
# Exit 2: bad family. 3: input / population. 4: GPU / install / download. 5: smoke failed. 124: deadline before a
# deciding stage could start.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
while [ $# -gt 0 ]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2;;
    --family=*) FAMILY="${1#--family=}"; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
FAMILY="${FAMILY:-}"
case "$FAMILY" in
  llama) SMALL=llama3b; LARGE=llama8b; SMALL_ID=meta-llama/Llama-3.2-3B-Instruct; LARGE_ID=meta-llama/Llama-3.1-8B-Instruct;;
  gemma) SMALL=gemma4b; LARGE=gemma12b; SMALL_ID=google/gemma-3-4b-it; LARGE_ID=google/gemma-3-12b-it;;
  phi)   SMALL=phi4mini; LARGE=phi4; SMALL_ID=microsoft/Phi-4-mini-instruct; LARGE_ID=microsoft/phi-4;;
  *) echo "FAMILY (or --family) must be llama, gemma or phi, not '${FAMILY}'" >&2; exit 2;;
esac
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-9900}"
SELF_STOP_S="${SELF_STOP_S:-10500}"
N_PROMPTS="${N_PROMPTS:-56}"
MIN_PROMPTS="${MIN_PROMPTS:-32}"
FIT_MAX_MIN="${FIT_MAX_MIN:-150}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp43_lens_stub_${FAMILY}}"; PY="${PYTHON:-.venv/bin/python}"; DEVICE=cpu; DTYPE=float32
  N_PROMPTS="${STUB_N_PROMPTS:-4}"; MIN_PROMPTS="${STUB_MIN_PROMPTS:-2}"; EVAL_LIMIT="--limit 3"; THREADS="--threads 4"
  ENG_ARGS="--engine stub"
else
  OUT="${OUT:-runs/exp43_lens_pod_${FAMILY}}"; PY="${PYTHON:-python}"; DEVICE=cuda; DTYPE=bfloat16; EVAL_LIMIT=""; THREADS=""
  ENG_ARGS="--engine hf"
fi
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
JL="$OUT/jl"                                   # J-lens root in exp34_report's layout: jl/<small>/, jl/lenses/, jl/eval/
DONE_FILE="$OUT/pod.done"
TIMES="$OUT/stage_times.json"
mkdir -p "$OUT" "$JL/lenses"
rm -f "$DONE_FILE" "$TIMES"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc family=$FAMILY engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$DONE_FILE"' EXIT
echo "exp43 Part B pod chain; family $FAMILY ($SMALL + $LARGE); engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}" | tee -a "$OUT/pod.log"
echo "deadlines: HARD_STOP_S=$HARD_STOP_S s (all model work ends), SELF_STOP_S=$SELF_STOP_S s (the pod stops itself)" | tee -a "$OUT/pod.log"

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

# Preflight: every input the chain reads (scripts/exp43_lens_bundle.sh packs exactly these and the code).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(pyproject.toml data/longmemeval/longmemeval_s_cleaned.json runs/exp24_verdicts.json
        data/jlens_corpus/wikitext-103-raw-v1-test.parquet external/jacobian-lens/pyproject.toml
        scripts/exp43_lens_pod_run.py scripts/jlens_read.py scripts/jlens_fit.py scripts/jlens_eval.py
        memctl/analysis/families.py memctl/analysis/jlens_readout.py memctl/jacobian_data.py)
for SET in association multihop multilingual order-ops poetry typo; do
  INPUTS+=("external/jacobian-lens/data/evaluations/lens-eval-${SET}.json")
done
for READER in "$SMALL" "$LARGE"; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp24_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
done
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
for CACHE in "cache/generations_exp24_$SMALL" "cache/generations_exp24_$LARGE"; do
  test -d "$CACHE" && test -n "$(ls -A "$CACHE")" || missing "$CACHE (directory, non-empty)"
done
REFERENCE="runs/exp43_lens_dev/$SMALL"
if [ "$ENGINE" != stub ] && [ ! -d "$REFERENCE" ]; then missing "$REFERENCE (laptop fp32 dev reads for the smoke)"; fi
EXPECT_FP=""
if [ -e bundle_info.json ]; then
  EXPECT_FP=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('population_fingerprint',''))" 2>/dev/null || true)
fi
if [ -z "${HF_TOKEN:-}" ] && [ -s /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(tr -d '[:space:]' < /workspace/.hf_token); fi
if [ "$ENGINE" != stub ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN not set and /workspace/.hf_token absent; the gated Llama and Gemma downloads will fail" >&2
fi
echo "preflight ok: ${#INPUTS[@]} files and 2 cache folders present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
# stage_done <name> <status> <rc> [est key] [note]: one record in stage_times.json and a pod.log line.
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp43_lens_pod_run.py stage-time --family "$FAMILY" --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
    --elapsed $(( now - STAGE_START )) --t-offset $(( STAGE_START - T0 )) ${4:+--est-key "$4"} --note "${5:-}" \
    --hard-stop-s "$HARD_STOP_S" --self-stop-s "$SELF_STOP_S" --engine "$ENGINE" \
    || echo "stage-time: could not write $TIMES" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
est() { $PY scripts/exp43_lens_pod_run.py estimate --family "$FAMILY" --stage "$1" --engine "$ENGINE"; }
left() { echo $(( DEADLINE - $(date +%s) )); }

# The install line (the pod's, and the D-30 fresh-venv test's: one line, used by both). Every import the chain makes
# is checked after it.
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"   # keep the installed torch; never let pip replace it
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . -e external/jacobian-lens \
    "transformers==${TRANSFORMERS_VERSION}" accelerate huggingface_hub safetensors pyarrow scipy 2>&1 | tail -2
  $PY -c "import torch, transformers, jlens, jlens.hooks, numpy, scipy, pyarrow, pyarrow.parquet, huggingface_hub, \
safetensors, accelerate, yaml; import memctl.analysis.families, memctl.analysis.jlens_readout, memctl.analysis.steering, \
memctl.analysis.jacobian_stats, memctl.jacobian_data, scripts.jlens_read, scripts.jlens_fit, scripts.jlens_eval, \
scripts.exp43_lens_pod_run; print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', \
numpy.__version__, 'scipy', scipy.__version__, 'pyarrow', pyarrow.__version__); \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
  echo "jlens commit $(git -C external/jacobian-lens rev-parse --short HEAD 2>/dev/null || cat external/jacobian-lens/COMMIT 2>/dev/null || echo unknown) (expected 581d398)" | tee -a "$OUT/pod.log"
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
  df -h . 2>/dev/null | tail -1 | tee -a "$OUT/pod.log" || true
  export TOKENIZERS_PARALLELISM=false
  stage_done install ok 0 install
else
  export HF_HUB_OFFLINE=1
  if [ -n "$INSTALL" ]; then install_line; stage_done install ok 0 install; fi
fi

# Population: counts per fold and the bundle's fingerprint (ids, prompts, outputs, labels; no outcome is printed).
set +e
$PY scripts/exp43_lens_pod_run.py population --family "$FAMILY" --out "$OUT" ${EXPECT_FP:+--expect-fingerprint "$EXPECT_FP"} \
  2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "population check failed (exit $RC)" | tee -a "$OUT/pod.log"; stage_done population failed "$RC"; exit 3; }
stage_done population ok 0

# Downloads (both sizes) or, for the stub, tiny random models with the real tokenizers.
if [ "$ENGINE" = stub ]; then
  $PY scripts/exp43_lens_pod_run.py make-stub --family "$FAMILY" --out "$OUT/stub_models" 2>&1 | { grep -v "Writing model shards" || true; } | tee -a "$OUT/pod.log"
  W_SMALL="--weights $OUT/stub_models/$SMALL"; W_LARGE="--weights $OUT/stub_models/$LARGE"; W_SMALL_PATH="$OUT/stub_models/$SMALL"
  stage_done stub_models ok 0
else
  set +e
  timeout $(( $(left) > 1800 ? 1800 : $(left) )) $PY - "$SMALL_ID" "$LARGE_ID" <<'EOF' 2>&1 | tee -a "$OUT/pod.log"
import sys, time, huggingface_hub
for repo in sys.argv[1:]:
    t = time.time()
    path = huggingface_hub.snapshot_download(repo, ignore_patterns=["original/*", "*.pth", "*.bin", "*.gguf", "*.onnx", "*.msgpack", "*.h5"])
    print(f"download {repo}: {time.time() - t:.0f} s -> {path}", flush=True)
EOF
  RC=${PIPESTATUS[0]}
  set -e
  [ "$RC" -eq 0 ] || { stage_done download failed "$RC" download; echo "download failed (exit $RC)" | tee -a "$OUT/pod.log"; exit 4; }
  W_SMALL=""; W_LARGE=""; W_SMALL_PATH=""
  stage_done download ok 0 download
fi

# run_read <stage> <est key> <args...>: skip when even the minimum does not fit; otherwise run with the deadline.
run_read() {
  local stage="$1" key="$2"; shift 2
  local need; need=$(est "$key")
  local min_need=$(( need / 4 ))
  if [ "$(left)" -le "$min_need" ]; then
    echo "SKIP $stage: $(left) s left, needs about $need s (minimum $min_need s)" | tee -a "$OUT/pod.log"
    stage_done "$stage" skipped -1 "$key" "left $(left) s, estimate $need s"; return 0
  fi
  set +e
  timeout "$(left)" $PY scripts/exp43_lens_pod_run.py read --family "$FAMILY" $ENG_ARGS --device "$DEVICE" --dtype "$DTYPE" \
    $THREADS --deadline "$DEADLINE" "$@" 2>&1 | tee -a "$OUT/${stage}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  case "$rc" in
    0) stage_done "$stage" ok 0 "$key";;
    9|124) echo "$stage cut by the deadline (exit $rc); rows so far kept" | tee -a "$OUT/pod.log"; stage_done "$stage" partial "$rc" "$key";;
    *) echo "$stage failed (exit $rc)" | tee -a "$OUT/pod.log"; stage_done "$stage" failed "$rc" "$key";;
  esac
  return "$rc"
}

# D-22 smoke: 5 dev rows through both sizes; any failure ends the chain before the full reads.
SMOKE_RC=0
run_read smoke_small smoke --size small --smoke 5 --out "$OUT/smoke/$SMALL" $W_SMALL ${REFERENCE:+--reference "$REFERENCE"} || SMOKE_RC=$?
if [ "$SMOKE_RC" -eq 0 ]; then run_read smoke_large smoke --size large --smoke 5 --out "$OUT/smoke/$LARGE" $W_LARGE || SMOKE_RC=$?; fi
if [ "$SMOKE_RC" -ne 0 ] || ! grep -q '"pass": true' "$OUT/smoke/$SMALL/_smoke.json" 2>/dev/null \
   || ! grep -q '"pass": true' "$OUT/smoke/$LARGE/_smoke.json" 2>/dev/null; then
  echo "SMOKE FAILED (exit $SMOKE_RC): chain aborted before the full reads" | tee -a "$OUT/pod.log"; exit 5
fi

# The deciding rows: logit-lens reads of both sizes on the 330 rows.
run_read logit_small logit_small --size small --variant logit --out "$OUT/logit/$SMALL" $W_SMALL || true
run_read logit_large logit_large --size large --variant logit --out "$OUT/logit/$LARGE" $W_LARGE || true

# J-lens fit, small model only, time box = what is left after the J-lens reads and eval (and one chunk) are reserved.
RESERVE=$(( $(est jread) + $(est eval) ))
FIT_EST=$(est fit); FIT_MIN=$(est fit_min)
CHUNK_S=$(( FIT_EST * 4 / 56 ))
FIT_BUDGET=$(( $(left) - RESERVE - CHUNK_S - 120 ))
FIT_OK=0
if [ "$FIT_BUDGET" -lt "$FIT_MIN" ]; then
  echo "SKIP fit: budget $FIT_BUDGET s after reserving $RESERVE s, the 32-prompt minimum needs about $FIT_MIN s" | tee -a "$OUT/pod.log"
  stage_done fit skipped -1 fit "budget $FIT_BUDGET s < minimum $FIT_MIN s"
else
  BOX_MIN=$(( FIT_BUDGET / 60 )); [ "$BOX_MIN" -le "$FIT_MAX_MIN" ] || BOX_MIN=$FIT_MAX_MIN
  echo "fit $SMALL: $N_PROMPTS prompts (min $MIN_PROMPTS), time box $BOX_MIN min" | tee -a "$OUT/pod.log"
  set +e
  timeout "$(left)" $PY scripts/jlens_fit.py --reader "$SMALL" --n-prompts "$N_PROMPTS" --min-prompts "$MIN_PROMPTS" \
    --max-minutes "$BOX_MIN" --dim-batch 32 --device "$DEVICE" --dtype "$DTYPE" --out "$JL/lenses" \
    ${W_SMALL_PATH:+--weights "$W_SMALL_PATH"} 2>&1 | tee -a "$OUT/fit.log"
  RC=${PIPESTATUS[0]}
  set -e
  if [ "$RC" -eq 0 ] && grep -Eq '"below_minimum":[[:space:]]*false' "$JL/lenses/$SMALL.json" 2>/dev/null; then
    FIT_OK=1; stage_done fit ok 0 fit "$(grep -Eo '"n_prompts":[[:space:]]*[0-9]+' "$JL/lenses/$SMALL.json")"
  else
    echo "fit: no usable lens (exit $RC; sidecar $(grep -Eo '"(n_prompts|below_minimum)":[^,]*' "$JL/lenses/$SMALL.json" 2>/dev/null | tr '\n' ' '))" | tee -a "$OUT/pod.log"
    stage_done fit failed "$RC" fit
  fi
fi

if [ "$FIT_OK" -eq 1 ]; then
  # J-lens reads: folds 0-1 first (the gate), stopping $(est eval) s before the deadline so the eval still fits.
  SAVE_DEADLINE=$DEADLINE; DEADLINE=$(( SAVE_DEADLINE - $(est eval) ))
  run_read jread jread --size small --variant jlens --lens "$JL/lenses/$SMALL.pt" --dev-first --out "$JL/$SMALL" $W_SMALL || true
  DEADLINE=$SAVE_DEADLINE
  if [ "$(left)" -le $(( $(est eval) / 2 )) ]; then
    echo "SKIP eval: $(left) s left" | tee -a "$OUT/pod.log"; stage_done eval skipped -1 eval "left $(left) s"
  else
    set +e
    timeout "$(left)" $PY scripts/jlens_eval.py --reader "$SMALL" --lens "$JL/lenses/$SMALL.pt" --device "$DEVICE" --dtype "$DTYPE" \
      --out "$JL/eval" $EVAL_LIMIT ${W_SMALL_PATH:+--weights "$W_SMALL_PATH"} 2>&1 | tee -a "$OUT/eval.log"
    RC=${PIPESTATUS[0]}
    set -e
    stage_done eval "$([ "$RC" -eq 0 ] && echo ok || echo failed)" "$RC" eval
  fi
else
  stage_done jread skipped -1 jread "no usable lens"
  stage_done eval skipped -1 eval "no usable lens"
fi

# Minimal status (row counts only; no outcome): the laptop report reads the rows.
$PY - "$OUT" "$SMALL" "$LARGE" <<'EOF' 2>&1 | tee -a "$OUT/pod.log"
import json, sys
from pathlib import Path
out, small, large = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
count = lambda d: sum(1 for p in Path(d).glob("*.json") if not p.name.startswith("_")) if Path(d).exists() else 0
status = {"logit_small": count(out / "logit" / small), "logit_large": count(out / "logit" / large),
          "jlens_small": count(out / "jl" / small), "eval": (out / "jl" / "eval" / f"{small}.json").exists(),
          "lens": json.loads((out / "jl" / "lenses" / f"{small}.json").read_text()).get("n_prompts")
                  if (out / "jl" / "lenses" / f"{small}.json").exists() else None}
(out / "status.json").write_text(json.dumps(status, indent=1))
print("status", json.dumps(status))
EOF
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ back (the lens .pt/.ckpt.pt are not needed by the report) before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
