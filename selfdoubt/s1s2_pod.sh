#!/usr/bin/env bash
# selfdoubt combined dev pod (Advisor item 9, 2026-11-09): Stage 2 pilot, then the Stage 1 dev run, on ONE pod (one
# setup). Dev only. The Executor needs the user's own approval line before launch.
#   1. selfdoubt/stage2_pilot_pod.sh   OUT=$OUT/s2, RUN_S=$PILOT_S (graceful deadline between rows), HF transformers
#   2. selfdoubt/stage0_full_pod.sh    OUT=$OUT/s1, RUNNER=selfdoubt/stage1_run.py, STEPS=$S1_STEPS, EXTRA=$S1_EXTRA,
#      RUN_S = what is left; its step starts only if its ceiling still fits (rc 8 otherwise)
# One self-stop watchdog here (the sub-scripts get none). $OUT/pod.done is written last, so the waiter copies both.
#   RUNPOD_POD_ID=<id> RUN_S=<s> SELF_STOP_S=<s> OUT=<dir> PILOT_S=<s> ROWS=... ROWS_SHA=... S1_STEPS=... S1_EXTRA=...
#   ENGINE=plan PYTHON=<fake> ...   # $0 test of the wiring
#   S2_SPLIT=heldout S2_HELDOUT_CONFIRMED=1 ...   # Stage 2 held-out replication (PROPOSAL c69f28e); Stage 1 stays dev
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-}"; RUN_S="${RUN_S:-3600}"; SELF_STOP_S="${SELF_STOP_S:-}"; OUT="${OUT:-runs/selfdoubt/s1s2_pod}"
PILOT_S="${PILOT_S:-3000}"; S1_STEPS="${S1_STEPS:-}"; S1_EXTRA="${S1_EXTRA:-}"
S2_SPLIT="${S2_SPLIT:-dev}"; S2_HELDOUT_CONFIRMED="${S2_HELDOUT_CONFIRMED:-}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + RUN_S ))
trap 'rc=$?; echo "exit=$rc pilot_rc=${RCP:-} s1_rc=${RCS:-} at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "selfdoubt s1+s2 pod; out $OUT; run budget $RUN_S s; pilot $PILOT_S s (split $S2_SPLIT); s1 steps '$S1_STEPS' extra '$S1_EXTRA'"
[[ "$S1_EXTRA" == *heldout* ]] && { log "Stage 1 on this pod is dev only (S1_EXTRA mentions heldout)"; exit 3; }
[[ "$S1_EXTRA" =~ --cut\ (0\.3|0\.5|0\.9) ]] || { log "S1_EXTRA must fix --cut (0.3/0.5/0.9)"; exit 3; }
[[ "$S1_STEPS" =~ ^[a-z0-9_]+\|17\|gsm8k\|dev\| ]] || { log "S1_STEPS must be one 1.7B GSM8K dev step"; exit 3; }
if [ "$ENGINE" != plan ] && [ -n "$SELF_STOP_S" ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"\$(date -u +%T) self-stop\" >> '$OUT/pod.log'; \
    for kv in \$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); do export \"\$kv\"; done; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl remove pod \"\$RUNPOD_POD_ID\" \
    || runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  log "self-stop watchdog armed at +$SELF_STOP_S s (runpodctl $(command -v runpodctl || echo MISSING))"
fi
P_ENGINE=pilot; S_ENGINE=vllm; [ "$ENGINE" = plan ] && { P_ENGINE=plan; S_ENGINE=plan; }
set +e
ENGINE=$P_ENGINE OUT="$OUT/s2" RUN_S="$PILOT_S" SELF_STOP_S= SPLIT_NAME="$S2_SPLIT" \
  HELDOUT_CONFIRMED="$S2_HELDOUT_CONFIRMED" bash selfdoubt/stage2_pilot_pod.sh; RCP=$?
log "pilot rc $RCP at +$(( $(date +%s) - T0 )) s; $(left) s left for Stage 1"
ENGINE=$S_ENGINE OUT="$OUT/s1" RUN_S="$(left)" SELF_STOP_S= RUNNER=selfdoubt/stage1_run.py STEPS="$S1_STEPS" \
  EXTRA="$S1_EXTRA" bash selfdoubt/stage0_full_pod.sh; RCS=$?
set -e
log "pod done at +$(( $(date +%s) - T0 )) s (pilot rc $RCP, stage1 rc $RCS): copy back $OUT"
[ "$RCP" = 0 ] && exit "$RCS"; exit "$RCP"
