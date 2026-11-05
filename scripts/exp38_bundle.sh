#!/usr/bin/env bash
# §38: build the transfer bundle for the pod, runs/_transfer/exp38_tree.tar.gz: the code (git-visible files under
# memctl/ and scripts/, pyproject.toml) and exactly the inputs scripts/exp38_pod.sh preflights:
#   - the population loader (memctl.jacobian_data.load_records, both readers, since the §35 population is the
#     intersection): runs/exp23_{qwen3b,qwen7b}_f{0..4}/fixed8__fraction0.05/{items.jsonl.gz,steps.jsonl.gz,episodes.jsonl},
#     runs/exp23_verdicts.json, cache/generations_exp23_{qwen3b,qwen7b}/, data/longmemeval/longmemeval_s_cleaned.json
#     (its .shards/ folder is rebuilt from it on first use, so it is not shipped);
#   - the §35 cache of the 3B (rows, judge verdicts, manifest): runs/exp35/qwen3b/;
#   - the §35 token check: runs/exp34_jlens_own/qwen3b/*.json (the .npz lens reads are not read);
#   - the laptop design: runs/exp38_dev/qwen3b/gate.json (the primary setting) and directions.npz (cosine, descriptive).
# No .env, no tests, no other run directory. Usage:  bash scripts/exp38_bundle.sh [out=runs/_transfer/exp38_tree.tar.gz]
# On the pod:  mkdir -p /workspace/llm-memctl && tar xzf exp38_tree.tar.gz -C /workspace/llm-memctl
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/_transfer/exp38_tree.tar.gz}"
mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
{
  echo pyproject.toml
  git ls-files --cached --others --exclude-standard -- memctl scripts | grep -v -E '(__pycache__|\.pyc$)'
  for R in qwen3b qwen7b; do
    for F in 0 1 2 3 4; do
      for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do echo "runs/exp23_${R}_f${F}/fixed8__fraction0.05/${FILE}"; done
    done
    echo "cache/generations_exp23_${R}"
  done
  echo runs/exp23_verdicts.json
  echo data/longmemeval/longmemeval_s_cleaned.json
  echo runs/exp35/qwen3b
  ls runs/exp34_jlens_own/qwen3b/*.json
  echo runs/exp38_dev/qwen3b/gate.json
  echo runs/exp38_dev/qwen3b/directions.npz
} | sort -u > "$LIST"
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$LIST"
if grep -q -E '(^|/)\.env$' "$LIST"; then echo "bundle: refusing to pack a .env file" >&2; exit 3; fi
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST"
echo "bundle: $(wc -l < "$LIST") entries -> $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16)"
