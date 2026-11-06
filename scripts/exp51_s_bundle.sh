#!/usr/bin/env bash
# §51 pod S bundle (runs/_transfer/exp51_podS.tar.gz): code; runs/exp51/cases51.jsonl FILTERED TO E AND S (no H or
# T-city row is in the bundle; the full list's sha and the filtered sha are in bundle_info.json); the counts file;
# the submissions (json + f_direction.npy); the edits the pod applies (submissions, B5) and b5_fit.json; the laptop
# references for the §10 replay check (B0 records and submission-1 outputs of the first 200 E rows by case id). No
# .env, no tests, no weights, no .git, never writing/.
#   bash scripts/exp51_s_bundle.sh <submissions dir> [out]
set -euo pipefail
cd "$(dirname "$0")/.."
SUBS="$1"; OUT="${2:-runs/_transfer/exp51_podS.tar.gz}"
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$(dirname "$OUT")" "$STAGE/runs/exp51/run/b0/E" "$STAGE/runs/exp51/gpu/subs" "$STAGE/runs/exp51/gpu/edits"
git ls-files --cached --others --exclude-standard -- memctl | grep -v -E '(__pycache__|\.pyc$)' > "$STAGE/list"
for f in pyproject.toml scripts/__init__.py scripts/exp45_cases.py scripts/exp49_cases.py scripts/exp50_cases.py \
         scripts/exp51_cases.py scripts/exp51_common.py scripts/exp51_engine.py scripts/exp51_run.py \
         scripts/exp51_mech.py scripts/exp51_stage4.py scripts/exp51_pod_s.sh scripts/exp51_smoke_s.py \
         runs/exp51/cases51_counts.json \
         runs/exp38_dev/qwen3b/directions.npz; do
  echo "$f" >> "$STAGE/list"
done
while read -r f; do test -e "$f" || { echo "bundle: missing $f" >&2; exit 3; }; done < "$STAGE/list"
python3 - "$STAGE" "$SUBS" <<'PY'
import json, shutil, sys, hashlib
from pathlib import Path
stage, subs = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in open("runs/exp51/cases51.jsonl")]
keep = [r for r in rows if r["split"] in ("E", "S")]
assert not [r for r in keep if r["split"] in ("H", "Tcity")]
(stage / "runs/exp51/cases51.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep))
for p in sorted(subs.glob("[0-9]*")):
    shutil.copy(p, stage / "runs/exp51/gpu/subs" / p.name)
ids = {json.loads(p.read_text())["edit_id"] for p in subs.glob("[0-9].json")}
fit = json.loads(Path("runs/exp51/run/b5_fit.json").read_text())
ids |= set(fit["edit_ids"].values())
for eid in ids:
    for ext in ("json", "npz"):
        shutil.copy(f"runs/exp51/run/edits/{eid}.{ext}", stage / "runs/exp51/gpu/edits")
shutil.copy("runs/exp51/run/b5_fit.json", stage / "runs/exp51/gpu/b5_fit.json")
e_rows = sorted((r for r in rows if r["split"] == "E"), key=lambda r: r["case_id"])[:200]
s1 = json.loads((subs / "1.json").read_text())["edit_id"]
dst = stage / f"runs/exp51/run/arms/E/EDIT_{s1}_d1/total"
dst.mkdir(parents=True)
for r in e_rows:
    shutil.copy(f"runs/exp51/run/b0/E/{r['case_id']}.json", stage / "runs/exp51/run/b0/E")
    shutil.copy(f"runs/exp51/run/arms/E/EDIT_{s1}_d1/total/{r['case_id']}.json", dst)
full = hashlib.sha256(open("runs/exp51/cases51.jsonl", "rb").read()).hexdigest()
filt = hashlib.sha256((stage / "runs/exp51/cases51.jsonl").read_bytes()).hexdigest()
(stage / "bundle_info.json").write_text(json.dumps({"cases_full_sha256": full, "cases_E_S_sha256": filt,
                                                    "n_rows": len(keep), "splits": ["E", "S"]}))
PY
echo "$(git rev-parse HEAD)" > "$STAGE/head_commit"
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' -T "$STAGE/list" -C "$STAGE" runs bundle_info.json head_commit
tar tzf "$OUT" | grep -q -E '(^|/)\.env$|^writing/' && { echo "bundle: forbidden file" >&2; exit 3; }
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)); sha256 $(sha256sum "$OUT" | cut -c1-16)"
