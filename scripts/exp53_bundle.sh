#!/usr/bin/env bash
# §53 pod bundle (runs/_transfer/exp53_pod.tar.gz), built only after the manifest commit: code; the N cases; the
# manifest and its marker; the edits, vectors and calibration; the LongMemEval arm C rows; submission 2's frozen f
# files; a shallow clone's .git holding the manifest commit (for the git-verified guard). Refuses when the guard fails
# here. No .env, no tests, no weights, never writing/.
#   bash scripts/exp53_bundle.sh [out]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp53_pod.tar.gz}"; mkdir -p "$(dirname "$OUT")"
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts'); import exp53_run as R; from pathlib import Path; \
R.require_manifest(Path('runs/exp53/pod')); print('guard ok on the laptop')" || { echo "bundle: manifest guard fails" >&2; exit 4; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
git clone -q --depth 50 "file://$(pwd)" "$STAGE/clone"
COMMIT=$(python3 -c "import json; print(json.load(open('runs/exp53/manifest_committed.json'))['commit'])")
git -C "$STAGE/clone" cat-file -e "${COMMIT}^{commit}" || { echo "bundle: manifest commit not in the shallow clone" >&2; exit 4; }
{ git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in pyproject.toml scripts/__init__.py scripts/exp45_cases.py scripts/exp49_cases.py scripts/exp50_cases.py \
           scripts/exp51_cases.py scripts/exp51_common.py scripts/exp51_engine.py scripts/exp51_run.py scripts/exp51_mech.py \
           scripts/exp51_stage4.py scripts/exp53_cases.py scripts/exp53_run.py scripts/exp53_pod.sh \
           runs/exp51/cases51_counts.json runs/exp51/gpu/subs/2.json runs/exp51/gpu/subs/2_f_direction.npy \
           runs/exp53/cases53.jsonl runs/exp53/cases53_counts.json runs/exp53/manifest.json runs/exp53/manifest_committed.json \
           runs/exp53/vectors.npz runs/exp53/calibration.json runs/exp53/edits_built.json runs/exp53/lme_rows.jsonl \
           runs/exp53/lme_rows_meta.json; do echo "$f"; done
  find runs/exp53/edits -type f
} | sort -u > "$STAGE/list"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$STAGE/list"
grep -q -E '(^|/)\.env$|^writing/' "$STAGE/list" && { echo "bundle: forbidden file" >&2; exit 3; }
mv "$STAGE/clone/.git" "$STAGE/.git"
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$STAGE/list" -C "$STAGE" .git
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); manifest commit $COMMIT"
