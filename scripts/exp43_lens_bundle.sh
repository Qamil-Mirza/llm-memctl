#!/usr/bin/env bash
# §43 Part B: the transfer bundle for one family's lens pod, runs/_transfer/exp43_lens_<family>_tree.tar.gz. Exactly
# what scripts/exp43_lens_pod.sh preflights and imports, nothing else:
#   - code: pyproject.toml, the memctl package (git-visible files), the pod chain and its imports
#     (scripts/exp43_lens_pod.sh, exp43_lens_pod_run.py, jlens_read.py, jlens_fit.py, jlens_eval.py) and the laptop
#     report with its imports (exp43_lens_report.py, exp34_report.py, exp43_lens_profile.py, exp43_lens_dev.py), so the
#     D-30 test can run the report from the tree; memctl/analysis/families.py is in the package;
#   - the vendored jacobian-lens (its git-tracked files, no .git) with its commit in external/jacobian-lens/COMMIT;
#   - data/longmemeval/longmemeval_s_cleaned.json (question metadata and evidence ids), the WikiText parquet (the fit);
#   - runs/exp24_verdicts.json and the family's two §24 readers' fixed8 traces (items, steps, episodes, folds 0-4);
#   - the §24 cache files of the 330 population rows for both readers (scripts/exp43_lens_pod_run.py bundle-list);
#   - the laptop fp32 dev reads of the 5 smoke rows (runs/exp43_lens_dev/<small>/) for the smoke's top-1 check;
#   - bundle_info.json: HEAD, dirty flag, family, the population fingerprint the pod re-checks, input shas.
# No tokenizer, no weights, no .env, no tests, no other run directory.
#   FAMILY=llama bash scripts/exp43_lens_bundle.sh [out]      (or: bash scripts/exp43_lens_bundle.sh --family llama [out])
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp43_lens_<family>_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = --family ]; then FAMILY="$2"; shift 2; fi
FAMILY="${FAMILY:-}"
case "$FAMILY" in
  llama) SMALL=llama3b; LARGE=llama8b;; gemma) SMALL=gemma4b; LARGE=gemma12b;; phi) SMALL=phi4mini; LARGE=phi4;;
  *) echo "bundle: FAMILY must be llama, gemma or phi, not '$FAMILY'" >&2; exit 2;;
esac
OUT="${1:-runs/_transfer/exp43_lens_${FAMILY}_tree.tar.gz}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
CACHE_LIST=$(mktemp)
trap 'rm -f "$LIST" "$CACHE_LIST"; rm -rf "$STAGE"' EXIT
$PY scripts/exp43_lens_pod_run.py bundle-list --family "$FAMILY" --with-reference > "$CACHE_LIST" \
  || { echo "bundle: bundle-list failed" >&2; exit 3; }
JL_COMMIT=$(git -C external/jacobian-lens rev-parse --short HEAD)
JL_DIRTY=$(git -C external/jacobian-lens status --porcelain --untracked-files=no | wc -l)
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp43_lens_pod.sh exp43_lens_pod_run.py jlens_read.py jlens_fit.py jlens_eval.py \
           exp43_lens_report.py exp34_report.py exp43_lens_profile.py exp43_lens_dev.py; do echo "scripts/$f"; done
  git -C external/jacobian-lens ls-files | sed 's|^|external/jacobian-lens/|'
  echo data/longmemeval/longmemeval_s_cleaned.json
  echo data/jlens_corpus/wikitext-103-raw-v1-test.parquet
  echo runs/exp24_verdicts.json
  for R in "$SMALL" "$LARGE"; do for F in 0 1 2 3 4; do for X in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
    echo "runs/exp24_${R}_f${F}/fixed8__fraction0.05/$X"; done; done; done
  cat "$CACHE_LIST"
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
FP=$($PY -c "
import sys; sys.path.insert(0, '.')
from memctl.analysis import families
from scripts.exp43_lens_pod_run import fingerprint, pop34_pairs
print(fingerprint(pop34_pairs(*families.pair('$FAMILY'))))")
sha() { sha256sum "$1" | cut -d' ' -f1; }
mkdir -p "$STAGE/external/jacobian-lens"
echo "$JL_COMMIT" > "$STAGE/external/jacobian-lens/COMMIT"
cat > "$STAGE/bundle_info.json" <<EOF
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "family": "$FAMILY", "small": "$SMALL", "large": "$LARGE", "population_fingerprint": "$FP",
 "jlens_commit": "$JL_COMMIT", "jlens_dirty_files": $JL_DIRTY,
 "longmemeval_sha256": "$(sha data/longmemeval/longmemeval_s_cleaned.json)",
 "verdicts_sha256": "$(sha runs/exp24_verdicts.json)",
 "n_cache_files": $(grep -c '^cache/' "$CACHE_LIST"), "built_utc": "$(date -u +%FT%TZ)"}
EOF
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json external/jacobian-lens/COMMIT
echo "bundle ($FAMILY): $(( $(wc -l < "$LIST") + 2 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); population fingerprint ${FP:0:16}; jlens $JL_COMMIT"
