#!/usr/bin/env bash
# §47: the transfer bundle for the pod (scripts/exp47_pod.sh), runs/_transfer/exp47_tree.tar.gz. Exactly what the
# chain preflights and imports:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, exp47_pod.sh,
#     exp47_pod_run.py, exp47_patch.py, exp47_report.py;
#   - inputs: runs/exp47_bridge/rows47.jsonl + rows47_counts.json (the 36 aligned rows with their token ids; no §44
#     prompt files are needed on the pod);
#   - bundle_info.json: HEAD, dirty flag, the rows47 sha the pod re-checks.
# No tokenizer, no weights (downloaded on the pod), no .env, no tests, no outputs. Never anything under writing/.
#   bash scripts/exp47_bundle.sh [out]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp47_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp47_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp47_pod.sh exp47_pod_run.py exp47_patch.py exp47_report.py; do echo "scripts/$f"; done
  echo runs/exp47_bridge/rows47.jsonl
  echo runs/exp47_bridge/rows47_counts.json
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '\.(npz|npy|pt)$' "$LIST"; then echo "bundle: refusing to pack arrays or lenses" >&2; exit 3; fi
if grep -q '^writing/' "$LIST"; then echo "bundle: refusing to pack writing/" >&2; exit 3; fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "rows47_sha256": "$(sha runs/exp47_bridge/rows47.jsonl)", "n_files": $(wc -l < "$LIST"), "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); rows47 $(sha runs/exp47_bridge/rows47.jsonl | cut -c1-16)"
