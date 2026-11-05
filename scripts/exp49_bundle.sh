#!/usr/bin/env bash
# §49: the transfer bundle for the pod (scripts/exp49_pod.sh), runs/_transfer/exp49_tree.tar.gz. Exactly what the chain
# preflights and imports:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, exp45_cases.py (templates),
#     exp49_pod.sh, exp49_pod_run.py, exp49_run.py, exp49_report.py;
#   - inputs: runs/exp49/cases49.jsonl + cases49_counts.json; the R2 vector file runs/exp38_dev/qwen3b/directions.npz
#     (+ .json), checked byte-exact by exp40.check_direction on the pod (the only array packed);
#   - bundle_info.json: HEAD, dirty flag, the cases49 sha the pod re-checks.
# No tokenizer, no weights (downloaded on the pod), no .env, no tests, no outputs, no .git (the design commit is
# shipped later as a shallow clone). Never anything under writing/.
#   bash scripts/exp49_bundle.sh [out]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp49_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
R2=runs/exp38_dev/qwen3b/directions.npz
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp45_cases.py exp49_pod.sh exp49_pod_run.py exp49_run.py exp49_report.py; do echo "scripts/$f"; done
  echo runs/exp49/cases49.jsonl
  echo runs/exp49/cases49_counts.json
  echo "$R2"
  echo runs/exp38_dev/qwen3b/directions.json
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -v -x -F "$R2" "$LIST" | grep -q -E '\.(npz|npy|pt)$'; then echo "bundle: refusing to pack arrays other than $R2" >&2; exit 3; fi
if grep -q '^writing/' "$LIST"; then echo "bundle: refusing to pack writing/" >&2; exit 3; fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "cases49_sha256": "$(sha runs/exp49/cases49.jsonl)", "r2_npz_sha256": "$(sha $R2)", "n_files": $(wc -l < "$LIST"),
 "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); cases49 $(sha runs/exp49/cases49.jsonl | cut -c1-16)"
