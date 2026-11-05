#!/usr/bin/env bash
# §45: the transfer bundle for the pod (scripts/exp45_pod.sh), runs/_transfer/exp45_tree.tar.gz. Exactly what the
# chain preflights and imports, nothing else:
#   - code: pyproject.toml, the memctl package (git-visible files), scripts/__init__.py, the pod chain and its imports
#     (exp45_pod.sh, exp45_pod_run.py, exp45_maps.py, jlens_fit.py, jlens_eval.py) and, for the D-30 test from the
#     tree, exp45_cases.py, exp45_render.py, exp45_report.py;
#   - the vendored jacobian-lens (its git-tracked files, no .git) with its commit in external/jacobian-lens/COMMIT;
#   - the WikiText parquet (the refit), the §34 56-prompt lens runs/exp34_jlens/lenses/qwen3b.{pt,json} (sha ddbdb32d...);
#   - the committed case list runs/exp45_maps/cases.jsonl + cases_counts.json, the stage 3 prompts rows44.jsonl +
#     rows44_counts.json, and the laptop fp32 smoke reference runs/exp45_maps/smoke_ref/reference.json (required;
#     ALLOW_NO_REF=1 packs without it, for the D-30 stub test only);
#   - bundle_info.json: HEAD, dirty flag, the cases / rows44 / lens shas the pod re-checks.
# No tokenizer, no weights (downloaded on the pod), no .env, no tests, no map outputs, no §44 rows.
#   bash scripts/exp45_bundle.sh [out]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp45_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp45_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
REF=runs/exp45_maps/smoke_ref/reference.json
if [ ! -e "$REF" ] && [ -z "${ALLOW_NO_REF:-}" ]; then
  echo "bundle: $REF missing (run: .venv/bin/python scripts/exp45_maps.py --write-reference --device cpu --dtype float32 --out runs/exp45_maps/smoke_ref)" >&2
  exit 3
fi
JL_COMMIT=$(git -C external/jacobian-lens rev-parse --short HEAD)
JL_DIRTY=$(git -C external/jacobian-lens status --porcelain --untracked-files=no | wc -l)
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in __init__.py exp45_pod.sh exp45_pod_run.py exp45_maps.py jlens_fit.py jlens_eval.py exp45_cases.py \
           exp45_render.py exp45_report.py; do echo "scripts/$f"; done
  git -C external/jacobian-lens ls-files | sed 's|^|external/jacobian-lens/|'
  echo data/jlens_corpus/wikitext-103-raw-v1-test.parquet
  echo runs/exp34_jlens/lenses/qwen3b.pt
  echo runs/exp34_jlens/lenses/qwen3b.json
  for f in cases.jsonl cases_counts.json rows44.jsonl rows44_counts.json; do echo "runs/exp45_maps/$f"; done
  if [ -e "$REF" ]; then echo "$REF"; fi
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
if grep -q -E '^runs/exp45_maps/(j56|lenses|eval)' "$LIST"; then echo "bundle: refusing to pack map outputs" >&2; exit 3; fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
LENS_SHA=$(sha runs/exp34_jlens/lenses/qwen3b.pt)
[ "$LENS_SHA" = ddbdb32de0eefd4ed0c2ca42d64620c322e931a24813df5fa25da95c482de21b ] || { echo "bundle: the §34 lens sha differs" >&2; exit 3; }
mkdir -p "$STAGE/external/jacobian-lens"
echo "$JL_COMMIT" > "$STAGE/external/jacobian-lens/COMMIT"
cat > "$STAGE/bundle_info.json" <<JSON
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "cases_sha256": "$(sha runs/exp45_maps/cases.jsonl)", "rows44_sha256": "$(sha runs/exp45_maps/rows44.jsonl)",
 "lens56_sha256": "$LENS_SHA", "smoke_reference": $([ -e "$REF" ] && echo true || echo false),
 "jlens_commit": "$JL_COMMIT", "jlens_dirty_files": $JL_DIRTY,
 "n_files": $(wc -l < "$LIST"), "built_utc": "$(date -u +%FT%TZ)"}
JSON
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json external/jacobian-lens/COMMIT
echo "bundle: $(( $(wc -l < "$LIST") + 2 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); cases $(sha runs/exp45_maps/cases.jsonl | cut -c1-16); smoke reference $([ -e "$REF" ] && echo yes || echo NO)"
