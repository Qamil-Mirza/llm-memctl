#!/usr/bin/env bash
# §51 pod H + T-city bundle (runs/_transfer/exp51_podHT.tar.gz), built only after the Advisor has committed F2:
# code; the FULL case list; the submissions, every edit (submissions, B5, the B6 seeds made on pod S); pod S's S
# records the H report needs (b0/S, f/S.json, f/S_pred.json); runs/exp51/gpu/design.json (= the committed F2) and
# design_committed.json; a shallow clone's .git holding that commit (for the git-verified guard). Refuses when the
# guard fails here. No .env, no tests, no weights, never writing/.
#   bash scripts/exp51_ht_bundle.sh [out]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp51_podHT.tar.gz}"
G=runs/exp51/gpu
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from memctl.analysis import abst49 as A; from pathlib import Path; \
A.require_design(Path('$G')); print('guard ok on the laptop')" || { echo "bundle: F2 guard fails; no H bundle" >&2; exit 4; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
git clone -q --depth 50 "file://$(pwd)" "$STAGE/clone"
COMMIT=$(python3 -c "import json; print(json.load(open('$G/design_committed.json'))['commit'])")
git -C "$STAGE/clone" cat-file -e "${COMMIT}^{commit}" || { echo "bundle: F2 commit not in the shallow clone" >&2; exit 4; }
{ git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)'
  for f in pyproject.toml scripts/__init__.py scripts/exp45_cases.py scripts/exp49_cases.py scripts/exp50_cases.py \
           scripts/exp51_cases.py scripts/exp51_common.py scripts/exp51_engine.py scripts/exp51_run.py \
           scripts/exp51_mech.py scripts/exp51_stage4.py scripts/exp51_pod_ht.sh runs/exp51/cases51.jsonl \
           runs/exp51/cases51_counts.json $G/design.json $G/design_committed.json $G/b5_fit.json; do echo "$f"; done
  find $G/subs $G/edits $G/b0/S $G/f -type f | grep -E '/(S|S_pred)\.json$|/subs/|/edits/|/b0/S/'
} | sort -u > "$STAGE/list"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$STAGE/list"
grep -q -E '(^|/)\.env$|^writing/' "$STAGE/list" && { echo "bundle: forbidden file" >&2; exit 3; }
mv "$STAGE/clone/.git" "$STAGE/.git"
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$STAGE/list" -C "$STAGE" .git
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); F2 commit $COMMIT"
