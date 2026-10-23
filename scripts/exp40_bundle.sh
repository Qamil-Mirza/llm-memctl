#!/usr/bin/env bash
# §40: build the transfer bundle for the pod, runs/_transfer/exp40_tree.tar.gz: the code (git-visible files under
# memctl/ and scripts/, pyproject.toml) and exactly the inputs scripts/exp40_pod.sh preflights:
#   - data/locomo/locomo10.json (the new set; the views are rebuilt from it on the pod, shas recorded);
#   - runs/exp38_dev/qwen3b/directions.npz (v_29, byte-exact; sha asserted by the run);
#   - runs/exp40_inputs/agreement60.jsonl (the 60 cached §35 judge rows, built by scripts/exp40_prepare.py);
#   - runs/exp41_dev/qwen3b/rule.json if it exists (the frozen §41 rule; its commit is recorded in bundle_info.json);
#   - bundle_info.json at the root: HEAD commit, the rule's last commit, the input shas.
# No tokenizer, no model, no .env, no tests, no other run directory. Usage:
#   bash scripts/exp40_bundle.sh [out=runs/_transfer/exp40_tree.tar.gz]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp40_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp40_tree.tar.gz}"
RULE=runs/exp41_dev/qwen3b/rule.json
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
STAGE=$(mktemp -d)
trap 'rm -f "$LIST"; rm -rf "$STAGE"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl scripts | grep -v -E '(__pycache__|\.pyc$)'
  echo data/locomo/locomo10.json
  echo runs/exp38_dev/qwen3b/directions.npz
  echo runs/exp40_inputs/agreement60.jsonl
  if [ -e "$RULE" ]; then echo "$RULE"; fi
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
RULE_COMMIT=none
if [ -e "$RULE" ]; then
  RULE_COMMIT=$(git log -1 --format=%H -- "$RULE" 2>/dev/null || true)
  if [ -z "$RULE_COMMIT" ]; then RULE_COMMIT=uncommitted; echo "bundle: WARNING: $RULE is not committed" >&2; fi
  if ! git diff --quiet HEAD -- "$RULE" 2>/dev/null; then RULE_COMMIT="$RULE_COMMIT+modified"; echo "bundle: WARNING: $RULE differs from its commit" >&2; fi
fi
sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "$STAGE/bundle_info.json" <<EOF
{"head_commit": "$(git rev-parse HEAD)", "dirty": $(git diff --quiet HEAD -- memctl scripts && echo false || echo true),
 "rule_commit": "$RULE_COMMIT", "rule_sha256": "$( [ -e "$RULE" ] && sha "$RULE" || echo none)",
 "locomo_sha256": "$(sha data/locomo/locomo10.json)", "directions_sha256": "$(sha runs/exp38_dev/qwen3b/directions.npz)",
 "agreement60_sha256": "$(sha runs/exp40_inputs/agreement60.jsonl)", "built_utc": "$(date -u +%FT%TZ)"}
EOF
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C "$STAGE" bundle_info.json
echo "bundle: $(( $(wc -l < "$LIST") + 1 )) entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16); rule commit $RULE_COMMIT"
