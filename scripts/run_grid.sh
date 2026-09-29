#!/usr/bin/env bash
# Drive the full LoCoMo grid, in batches, on a machine with a GPU of about 12 GB or more.
#
#   scripts/run_grid.sh first          keep_newest and oracle at B = 10% and 25%  (4 runs)
#   scripts/run_grid.sh rest           the other rule controllers, and B = 50%    (11 runs)
#   scripts/run_grid.sh full_context   full_context on its own, watching GPU memory
#   scripts/run_grid.sh all            first, then rest, then full_context
#
# Runs "first" before "rest" so the keep_newest-against-oracle question can be looked at
# before committing hours to the remaining controllers. Every batch writes
# runs/locomo_full/comparison.md at the end, and every run is logged with its wall clock.
#
# Already-finished runs are skipped, so the script can be re-run after an interruption.
# A run counts as finished when its folder holds a metrics.json for the same controller,
# budget and git commit. Nothing is ever overwritten: memctl.run stamps every folder with
# the commit it was made from (docs/decisions.md entry 14).
#
# The jev controller is not here: it needs an API key. Add it when there is one.

set -euo pipefail
cd "$(dirname "$0")/.."

# Overridable so the driver itself can be smoke-tested against a stub config:
#   CONFIG=configs/smoke_keep_newest.yaml scripts/run_grid.sh first
CONFIG=${CONFIG:-configs/locomo_full.yaml}
# Read where the runs go from the config, so the two cannot drift apart.
OUT=$(sed -n 's/^output_dir:[[:space:]]*//p' "$CONFIG" | head -1)
OUT=${OUT:-runs}
LOG=$OUT/grid.log
SERVICE=${SERVICE:-memctl}          # SERVICE=cpu to run without a GPU
COMPOSE=${COMPOSE:-docker compose}

# On a rented pod that is already a container, Docker-in-Docker is usually unavailable.
# Set NO_DOCKER=1 to call a local interpreter instead:
#   NO_DOCKER=1 scripts/run_grid.sh first
# RUNNER then defaults to the venv scripts/provision_gpu_box.sh creates.
NO_DOCKER=${NO_DOCKER:-}
if [ -n "$NO_DOCKER" ]; then
    RUNNER=${RUNNER:-.venv/bin/python}
else
    RUNNER=${RUNNER:-python}
fi
run_memctl() {
    if [ -n "$NO_DOCKER" ]; then
        "$RUNNER" "$@"
    else
        $COMPOSE run --rm "$SERVICE" "$RUNNER" "$@"
    fi
}

commit_tag() {   # the suffix memctl.run puts on a folder name for the current tree
    local short dirty
    short=$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)
    dirty=$(git status --porcelain 2>/dev/null)
    [ -n "$dirty" ] && printf '%s-dirty' "$short" || printf '%s' "$short"
}

already_done() {  # $1 controller, $2 budget label used in folder names (e.g. B10pct)
    compgen -G "$OUT/*_$1_*_$2_seed*_$(commit_tag)" > /dev/null 2>&1
}

say() { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

one_run() {  # $1 controller, $2 budget fraction ("" for full_context)
    local controller=$1 fraction=${2:-} label started elapsed
    if [ -n "$fraction" ]; then
        label="B$(python3 -c "print(round(float('$fraction')*100))")pct"
    else
        label="B25pct"   # full_context ignores the budget but the folder still carries the label
    fi
    if already_done "$controller" "$label"; then
        say "SKIP  $controller $label (already run at this commit)"
        return 0
    fi
    say "START $controller $label"
    started=$(date +%s)
    local args=(-m memctl.run --config "$CONFIG" --controller "$controller")
    [ -n "$fraction" ] && args+=(--budget-fraction "$fraction")
    if run_memctl "${args[@]}" >> "$LOG" 2>&1; then
        elapsed=$(( $(date +%s) - started ))
        say "DONE  $controller $label in $((elapsed / 60))m$((elapsed % 60))s"
    else
        elapsed=$(( $(date +%s) - started ))
        say "FAIL  $controller $label after $((elapsed / 60))m$((elapsed % 60))s -- see $LOG"
        return 1
    fi
}

compare() {
    say "writing $OUT/comparison.md"
    run_memctl -m memctl.compare "$OUT" >> "$LOG" 2>&1 || say "compare failed, see $LOG"
}

batch_first() {
    say "=== batch: keep_newest and oracle at B = 10% and 25% ==="
    for fraction in 0.10 0.25; do
        for controller in keep_newest oracle; do one_run "$controller" "$fraction"; done
    done
    compare
    say "=== batch done. Read $OUT/comparison.md before running 'rest'. ==="
}

batch_rest() {
    say "=== batch: the remaining controllers and budgets ==="
    for fraction in 0.10 0.25 0.50; do
        for controller in lru random file_everything; do one_run "$controller" "$fraction"; done
    done
    for controller in keep_newest oracle; do one_run "$controller" 0.50; done
    compare
    say "=== batch done ==="
}

# full_context has no budget and its prompts are the whole history, about 22,000 model
# tokens. It did not fit on the development Mac (docs/decisions.md entry 35). GPU memory
# is sampled every 20 seconds while it runs so that a stall shows up in the log.
batch_full_context() {
    say "=== batch: full_context (prompts of about 22,000 model tokens) ==="
    if command -v nvidia-smi > /dev/null 2>&1; then
        ( while true; do
              printf 'gpu  %s  %s MiB used\n' "$(date -u +%H:%M:%S)" \
                  "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$OUT/full_context_gpu.log"
              sleep 20
          done ) &
        local watcher=$!
        trap 'kill $watcher 2>/dev/null || true' RETURN
        say "watching GPU memory in $OUT/full_context_gpu.log"
    fi
    if one_run full_context; then
        say "full_context finished"
    else
        say "full_context did NOT complete -- report it as not run, with the error from $LOG"
    fi
    compare
}

case "${1:-}" in
    first|rest|full_context|all) mkdir -p "$OUT" ;;
esac

case "${1:-}" in
    first)        batch_first ;;
    rest)         batch_rest ;;
    full_context) batch_full_context ;;
    all)          batch_first; batch_rest; batch_full_context ;;
    *)            sed -n '2,15p' "$0"; exit 2 ;;
esac
