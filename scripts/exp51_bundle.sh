#!/usr/bin/env bash
# §51: the transfer bundle for the timed GPU measurement pod (scripts/exp51_pod_measure.sh),
# runs/_transfer/exp51_measure.tar.gz. Code the chain imports, the case-list counts, the 64 E rows with their laptop
# B0 records (runs/exp51/gpu_measure/inputs.jsonl, E only), bundle_info.json. No weights, no tokenizer, no .env, no
# tests, no outputs, no arrays, no .git, nothing sealed (H / T-city rows are not in the inputs), never writing/.
#   bash scripts/exp51_bundle.sh [out]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp51_measure.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp); STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp45_cases.py exp49_cases.py exp50_cases.py exp51_cases.py exp51_common.py exp51_engine.py \
           exp51_gpu_measure.py exp51_pod_measure.sh; do echo "scripts/$f"; done
  echo runs/exp51/cases51_counts.json
  echo runs/exp51/gpu_measure/inputs.jsonl
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '\.(npz|npy|pt|safetensors)$' "$LIST"; then echo "bundle: refusing to pack arrays" >&2; exit 3; fi
if grep -q '^writing/' "$LIST"; then echo "bundle: refusing to pack writing/" >&2; exit 3; fi
python3 - <<'PY' || { echo "bundle: inputs hold a non-E row" >&2; exit 3; }
import json, sys
rows = [json.loads(l) for l in open("runs/exp51/gpu_measure/inputs.jsonl")]
sys.exit(0 if rows and all(r["row"]["split"] == "E" for r in rows) else 1)
PY
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "inputs_sha256": "$(sha runs/exp51/gpu_measure/inputs.jsonl)", "n_files": $(wc -l < "$LIST"), "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16)"
