#!/usr/bin/env bash
# §44 rev 8: the transfer bundle for the pod (scripts/exp44_pod.sh), runs/_transfer/exp44_rev8_tree.tar.gz. Exactly
# what the chain preflights and imports, nothing else:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, the pod chain and its imports
#     (exp44_pod.sh, exp44_pod_run.py, exp44_patch.py, exp44_swap.py, exp44_swap_gate.py) and, for the D-30 test from
#     the tree, exp44_swap_report.py and exp44_device_check.py;
#   - the §34 population sources: data/longmemeval/longmemeval_s_cleaned.json, runs/exp23_verdicts.json, the §23
#     fixed8 traces of qwen3b and qwen7b (items, steps, episodes; folds 0-4) and both generation caches
#     (cache/generations_exp23_qwen{3b,7b}); runs/exp34_jlens/qwen7b/*.json (exp44_patch's §34 prompt-sha check);
#   - the gate outputs: runs/exp44_swapgate/gate.json (rev 3, the rev 8 run asserts it) and runs/exp44_swapgate/rev8/
#     (gate.json, eligible_qwen3b.jsonl);
#   - bundle_info.json: HEAD, dirty flag, the population fingerprint the pod re-checks, input shas.
# No tokenizer, no weights (downloaded on the pod), no .env, no tests, no result directory (no exp44_patch/exp44_swap
# rows), no steering directions (U does not rerun).
#   bash scripts/exp44_bundle.sh [out]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp44_rev8_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp44_rev8_tree.tar.gz}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp44_pod.sh exp44_pod_run.py exp44_patch.py exp44_swap.py exp44_swap_gate.py \
           exp44_swap_report.py exp44_device_check.py; do echo "scripts/$f"; done
  echo data/longmemeval/longmemeval_s_cleaned.json
  echo runs/exp23_verdicts.json
  for R in qwen3b qwen7b; do for F in 0 1 2 3 4; do for X in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
    echo "runs/exp23_${R}_f${F}/fixed8__fraction0.05/$X"; done; done
    find "cache/generations_exp23_$R" -type f -name '*.json'; done
  find runs/exp34_jlens/qwen7b -maxdepth 1 -type f -name '*.json'
  echo runs/exp44_swapgate/gate.json
  for f in gate.json gate.md eligible_qwen3b.jsonl; do echo "runs/exp44_swapgate/rev8/$f"; done
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '^runs/exp44_(patch|swap)/' "$LIST"; then echo "bundle: refusing to pack result rows" >&2; exit 3; fi
FP=$($PY -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from scripts.exp44_pod_run import fingerprint_parts
print(fingerprint_parts()['fingerprint'])" 2>/dev/null)
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "population_fingerprint": "$FP",
 "longmemeval_sha256": "$(sha data/longmemeval/longmemeval_s_cleaned.json)",
 "verdicts_sha256": "$(sha runs/exp23_verdicts.json)",
 "rev8_eligible_sha256": "$(sha runs/exp44_swapgate/rev8/eligible_qwen3b.jsonl)",
 "n_files": $(wc -l < "$LIST"), "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); population fingerprint ${FP:0:16}"
