#!/usr/bin/env bash
# One-off cleanup for the 2026-10-10 reset (approved by the user; see research/RESET_HANDOFF.md).
# Every step checks before it deletes, and the script stops at the first failure.
# Run from the repo root:  bash scripts/reset_cleanup_2026-10-10.sh [step ...]
# Steps: verify worktrees quarantine dirty compress tidy branches   (no argument = all, in that order)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$(git branch --show-current)" = reset-2026-10-10 ] || { echo "check out reset-2026-10-10 first"; exit 1; }
LOG=runs/_deleted_2026-10-10.txt

step_verify() {
    # The worktree-only runs were copied into runs/ and listed with checksums; prove the copies are intact.
    (cd runs && sha256sum --quiet -c _from_worktrees/MANIFEST.sha256) && echo "verify: all rescued files match"
}

step_worktrees() {
    step_verify
    git worktree unlock .claude/worktrees/agent-a98a784ae7effa943 2>/dev/null || true
    for w in .claude/worktrees/agent-*; do
        [ -d "$w" ] || continue
        if [ -n "$(git -C "$w" status --porcelain | grep -v '^!!' || true)" ]; then echo "$w has uncommitted work, skipped"; continue; fi
        git worktree remove --force "$w" && echo "removed $w (branch kept on origin)"
    done
    git worktree prune
}

step_quarantine() {
    # Failed launches and unfinished cells that were never reported (EXPERIMENTS.md §15, §20, §22c).
    for d in runs/_aborted runs/_aborted_exp20 runs/_aborted_exp20_b runs/exp22c_peek_quarantine; do
        [ -d "$d" ] || continue
        echo "$(date -u +%FT%TZ) deleted $d ($(du -sh "$d" | cut -f1)): aborted or quarantined, never reported" >> "$LOG"
        rm -rf "$d"; echo "deleted $d"
    done
}

step_dirty() {
    # Pre-2026-10-05 dirty runs, replaced by the clean rerun. Keep the small files the research note cites
    # (reports, summaries, metadata, configs, logs); delete episodes and per-step traces.
    d=runs/_archive_dirty_2026-09-30
    [ -d "$d" ] || return 0
    before=$(du -sh "$d" | cut -f1)
    find "$d" -type f ! \( -name '*.md' -o -name 'summary.json' -o -name 'report.json' -o -name 'metadata.json' \
        -o -name '*.yaml' -o -name '*.log' -o -name 'train_log.jsonl' \) -delete
    find "$d" -type d -empty -delete
    echo "$(date -u +%FT%TZ) trimmed $d from $before to $(du -sh "$d" | cut -f1): dirty runs, kept reports/summaries/configs only" >> "$LOG"
    echo "trimmed $d: $before -> $(du -sh "$d" | cut -f1)"
}

compress_one() {
    f=$1
    gzip -k -6 "$f"
    gzip -t "$f.gz"
    [ "$(wc -l < "$f")" = "$(zcat "$f.gz" | wc -l)" ] || { echo "LINE MISMATCH $f"; rm -f "$f.gz"; return 1; }
    rm "$f"
}
export -f compress_one

step_compress() {
    # Per-step traces only. memctl.runlog.open_trace reads name.gz when name is gone (PR #21).
    # episodes/failures/regret stay plain: some readers open them directly.
    before=$(du -sh runs | cut -f1)
    find runs -type f \( -name items.jsonl -o -name steps.jsonl -o -name rewards.jsonl -o -name memory_actions.jsonl \) \
        ! -path 'runs/_from_worktrees/*' -print0 | xargs -0 -r -n 1 -P "$(nproc)" bash -c 'compress_one "$0"'
    echo "compress: runs $before -> $(du -sh runs | cut -f1)"
}

step_tidy() {
    # Some __pycache__ folders are root-owned (left by the Docker image); those are skipped.
    find . -path ./.venv -prune -o -type d -name __pycache__ -user "$(id -u)" -print0 | xargs -0 -r rm -rf 2>/dev/null || true
    rm -rf .pytest_cache
    rmdir .worktrees 2>/dev/null || true
    echo "tidy: caches removed (memctl.egg-info kept, the editable install may use it)"
}

step_branches() {
    # Local copies of branches already merged into origin/main; all remain on origin.
    for b in $(git for-each-ref --format='%(refname:short)' refs/heads); do
        case $b in main|reset-2026-10-10|phase2-locomo-baseline) continue;; esac
        [ "$(git rev-list --count origin/main.."$b")" = 0 ] || { echo "kept $b (not in origin/main)"; continue; }
        git branch -D "$b" >/dev/null && echo "deleted local branch $b"
    done
}

steps=${*:-verify worktrees quarantine dirty compress tidy branches}
for s in $steps; do echo "== $s"; "step_$s"; done
df -h . | tail -1
