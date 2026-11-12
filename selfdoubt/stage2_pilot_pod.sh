#!/usr/bin/env bash
# selfdoubt Stage 2 pilot pod (PROPOSAL decision 8): Qwen3-1.7B, GSM8K DEV rows only (selfdoubt/stage2_pilot.py select,
# run on the laptop). Teacher-forced forwards with HF transformers; no vLLM. Measures only; readings are frozen by the
# Advisor before any held-out run. The Executor needs the user's own approval line before launch.
#   RUNPOD_POD_ID=<id> RUN_S=<s> SELF_STOP_S=<s> OUT=<dir> ROWS=<rows.jsonl> ROWS_SHA=<sha256> bash selfdoubt/stage2_pilot_pod.sh
#   ENGINE=plan PYTHON=<fake> ...    # $0 test: no watchdog/GPU/install/download; the runner gets the run/report args
#   SPLIT_NAME=heldout HELDOUT_CONFIRMED=1 ...   # held-out replication (PROPOSAL c69f28e); default dev
# Order: preflight (files, rows sha, dev-only check) -> self-stop watchdog -> GPU -> hub + background download ->
# transformers install -> run (deadline, resumable) -> report. Output: $OUT/res.jsonl, $OUT/res.report.json.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-pilot}"; RUN_S="${RUN_S:-1500}"; SELF_STOP_S="${SELF_STOP_S:-}"
OUT="${OUT:-runs/selfdoubt/s2_pilot_pod}"; ROWS="${ROWS:-}"; ROWS_SHA="${ROWS_SHA:-}"; PY="${PYTHON:-python}"
OFFSETS="${OFFSETS:-0,32,128,end}"; SPLIT_NAME="${SPLIT_NAME:-dev}"; HELDOUT_CONFIRMED="${HELDOUT_CONFIRMED:-}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + RUN_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "selfdoubt stage2 pilot pod; out $OUT; rows $ROWS; split $SPLIT_NAME; offsets $OFFSETS; engine $ENGINE; run budget $RUN_S s; self-stop ${SELF_STOP_S:-none} s"
case "$SPLIT_NAME" in
  dev) ;;
  heldout) [ "$HELDOUT_CONFIRMED" = 1 ] || { log "held-out rows need HELDOUT_CONFIRMED=1"; exit 3; } ;;
  *) log "SPLIT_NAME must be dev or heldout"; exit 3 ;;
esac

TF_PIN="transformers==4.57.1"; HUB_PIN="huggingface_hub[hf_xet]<1.0"
M17="${M17:-Qwen/Qwen3-1.7B}"; R17="${R17:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
SPLIT_SHA=c9e99ab227e45d10298d91d4f0a17e150cfefa3a24d872555a2330c7d9392703

for f in selfdoubt/__init__.py selfdoubt/common.py selfdoubt/engines.py selfdoubt/stage2_pilot.py \
         selfdoubt/vendor/interrupt_lrm/__init__.py selfdoubt/vendor/interrupt_lrm/prompt_utils_qwen.py \
         runs/selfdoubt/split.json "$ROWS"; do
  test -n "$f" && test -e "$f" || { log "missing ${f:-ROWS}"; exit 3; }; done
sha() { sha256sum "$1" | cut -d' ' -f1; }
[ "$(sha runs/selfdoubt/split.json)" = "$SPLIT_SHA" ] || { log "split.json sha mismatch"; exit 3; }
[ -n "$ROWS_SHA" ] && [ "$(sha "$ROWS")" = "$ROWS_SHA" ] || { log "rows sha mismatch (or ROWS_SHA unset)"; exit 3; }
# every row's item must be in split[SPLIT_NAME]["gsm8k"] (stdlib only, before any install)
python3 - "$ROWS" "$SPLIT_NAME" <<'PY' | tee -a "$OUT/pod.log" || { log "rows are not all $SPLIT_NAME GSM8K"; exit 3; }
import json, sys
name = sys.argv[2]
ok = set(json.load(open("runs/selfdoubt/split.json"))[name]["gsm8k"])
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
bad = [r["item_id"] for r in rows if r["item_id"] not in ok]
assert rows and not bad, f"{len(bad)} rows outside {name}"
print(f"rows ok: {len(rows)} {name} GSM8K rows, {len({r['item_id'] for r in rows})} items")
PY

if [ "$ENGINE" = plan ]; then
  log "plan: no watchdog, GPU, install or download; runner $PY"
  SNAP=PLAN17
else
  if [ -n "$SELF_STOP_S" ]; then
    nohup setsid bash -c "sleep $SELF_STOP_S; echo \"\$(date -u +%T) self-stop\" >> '$OUT/pod.log'; \
      for kv in \$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); do export \"\$kv\"; done; \
      if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl remove pod \"\$RUNPOD_POD_ID\" \
      || runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
      > "$OUT/watchdog.log" 2>&1 < /dev/null &
    log "self-stop watchdog armed at +$SELF_STOP_S s (runpodctl $(command -v runpodctl || echo MISSING))"
  fi
  until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$OUT/pod.log"
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  [ "$TORCH_VERSION" = 2.8.0 ] || { log "image torch $TORCH_VERSION != 2.8.0"; exit 4; }
  timeout 300 $PY -m pip install -q -c "$OUT/constraints.txt" "$HUB_PIN" 2>&1 | tail -2
  ( timeout 900 $PY -c "import huggingface_hub as h; print(h.snapshot_download('$M17', revision='$R17', \
allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges.txt','vocab.json']))" > "$OUT/snap17.path" \
    2> "$OUT/snap17.err"; echo $? > "$OUT/snap17.rc" ) &
  log "download started; installing $TF_PIN"
  timeout 600 $PY -m pip install -q -c "$OUT/constraints.txt" "$TF_PIN" "$HUB_PIN" pyarrow numpy \
    latex2sympy2==1.9.1 word2number==1.1 2>&1 | tail -3     # the report's R2 readings import the Stage 0 scorer
  $PY -c "import transformers, torch; assert transformers.__version__ == '4.57.1'; assert torch.__version__.startswith('2.8.0'); \
print(transformers.__version__, torch.__version__)" | tee -a "$OUT/pod.log" || { log "install check failed"; exit 4; }
  $PY -m pip freeze 2>/dev/null | grep -iE "^(torch|transformers|tokenizers|huggingface|hf.xet|safetensors|latex2sympy2|antlr4|sympy)" > "$OUT/env.txt" || true
  until [ -s "$OUT/snap17.rc" ]; do [ "$(left)" -gt 60 ] || { log "download not done in time"; exit 6; }; sleep 3; done
  [ "$(cat "$OUT/snap17.rc")" = 0 ] || { log "snapshot failed: $(tail -2 "$OUT/snap17.err")"; exit 6; }
  SNAP=$(tail -1 "$OUT/snap17.path")
  log "setup done at +$(( $(date +%s) - T0 )) s; snapshot $SNAP"
fi

# report-path smoke check BEFORE the paid generation (v2 pod 2026-11-09: the report failed after the run on a
# missing latex2sympy2); a failure here stops the pod with nothing generated
set +e
$PY -u selfdoubt/stage2_pilot.py check 2>&1 | tee -a "$OUT/pod.log"; ck=${PIPESTATUS[0]}
set -e
[ "$ck" = 0 ] || { log "report path check failed (rc $ck)"; exit 4; }
guard=$(( $(left) - 60 ))
[ "$guard" -gt 120 ] || { log "no time left to run ($guard s)"; exit 8; }
log "run: guard $guard s"
set +e
timeout "$guard" $PY -u selfdoubt/stage2_pilot.py run --rows "$ROWS" --out "$OUT/res.jsonl" --model "$M17" \
  --snapshot "$SNAP" --offsets "$OFFSETS" --deadline-s "$(( guard - 120 ))" 2>&1 | tee -a "$OUT/run.log" \
  | grep -vE "^(Loading|\[transformers\])"; rc=${PIPESTATUS[0]}
$PY -u selfdoubt/stage2_pilot.py report --res "$OUT/res.jsonl" --label "$SPLIT_NAME" 2>&1 | tee -a "$OUT/pod.log"; rc2=${PIPESTATUS[0]}
set -e
log "pod done at +$(( $(date +%s) - T0 )) s (run rc $rc, report rc $rc2): copy back $OUT"
[ "$rc" = 0 ] && exit "$rc2"; exit "$rc"
