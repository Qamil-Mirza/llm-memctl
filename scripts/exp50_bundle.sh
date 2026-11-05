#!/usr/bin/env bash
# §50: the transfer bundle for the pod (scripts/exp50_pod.sh), runs/_transfer/exp50_tree.tar.gz. Exactly what the chain
# preflights and imports:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, exp45_cases.py, exp49_cases.py,
#     exp49_run.py, exp49_pod_run.py (reused engine and helpers), exp50_cases.py, exp50_run.py, exp50_report.py,
#     exp50_pod_run.py, exp50_pod.sh;
#   - inputs: runs/exp50/cases50.jsonl + cases50_counts.json;
#   - bundle_info.json: HEAD, dirty flag, the cases50 sha the pod re-checks.
# No tokenizer, no weights, no .env, no tests, no outputs, no arrays, no .git (the design commit is shipped later as a
# shallow clone). Never anything under writing/.
#   bash scripts/exp50_bundle.sh [out]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp50_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp45_cases.py exp49_cases.py exp49_run.py exp49_pod_run.py exp50_cases.py exp50_run.py \
           exp50_report.py exp50_pod_run.py exp50_pod.sh; do echo "scripts/$f"; done
  echo runs/exp50/cases50.jsonl
  echo runs/exp50/cases50_counts.json
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '\.(npz|npy|pt)$' "$LIST"; then echo "bundle: refusing to pack arrays" >&2; exit 3; fi
if grep -q '^writing/' "$LIST"; then echo "bundle: refusing to pack writing/" >&2; exit 3; fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "cases50_sha256": "$(sha runs/exp50/cases50.jsonl)", "n_files": $(wc -l < "$LIST"),
 "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); cases50 $(sha runs/exp50/cases50.jsonl | cut -c1-16)"
