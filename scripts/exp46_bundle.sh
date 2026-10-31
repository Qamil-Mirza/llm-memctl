#!/usr/bin/env bash
# §46: the transfer bundle for the pod (scripts/exp46_pod.sh), runs/_transfer/exp46_tree.tar.gz. Exactly what the
# chain preflights and imports:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, exp46_pod.sh,
#     exp46_pod_run.py, exp46_capture.py, exp46_patch.py, exp46_probe.py, exp46_report.py, exp45_cases.py (pools,
#     render);
#   - inputs: runs/exp45_maps/cases.jsonl + cases_counts.json, the 576 RC / TU §45 stage-1 records
#     (runs/exp45_maps/j56-jrefit-logit/stage1/{RC,TU}*.json: chat sha, last prompt index, §45 outcome; no npz), the
#     §46 addition runs/exp46_probe/cases46.jsonl + cases46_counts.json;
#   - bundle_info.json: HEAD, dirty flag, the cases / cases46 shas the pod re-checks.
# No tokenizer, no weights (downloaded on the pod), no .env, no tests, no maps, no outputs.
#   bash scripts/exp46_bundle.sh [out]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp46_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp46_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp46_pod.sh exp46_pod_run.py exp46_capture.py exp46_patch.py exp46_probe.py exp46_report.py \
           exp45_cases.py; do echo "scripts/$f"; done
  echo runs/exp45_maps/cases.jsonl
  echo runs/exp45_maps/cases_counts.json
  ls runs/exp45_maps/j56-jrefit-logit/stage1/RC*.json runs/exp45_maps/j56-jrefit-logit/stage1/TU*.json
  echo runs/exp46_probe/cases46.jsonl
  echo runs/exp46_probe/cases46_counts.json
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '\.(npz|npy|pt)$' "$LIST"; then echo "bundle: refusing to pack arrays or lenses" >&2; exit 3; fi
N_S1=$(grep -c 'stage1/' "$LIST")
[ "$N_S1" -eq 576 ] || { echo "bundle: $N_S1 stage-1 records, want 576" >&2; exit 3; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "cases_sha256": "$(sha runs/exp45_maps/cases.jsonl)", "cases46_sha256": "$(sha runs/exp46_probe/cases46.jsonl)",
 "n_stage1_records": $N_S1, "n_files": $(wc -l < "$LIST"), "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); cases46 $(sha runs/exp46_probe/cases46.jsonl | cut -c1-16)"
