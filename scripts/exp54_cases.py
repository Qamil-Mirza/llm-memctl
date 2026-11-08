#!/usr/bin/env python
"""§54: the W split (seed 55) with the §51 F2 generator, through scripts/exp53_cases.build. Data only (tokenizer
read offline for the §51 checks).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp54_cases.py          # -> runs/exp54/cases54.jsonl + counts

Spec: research/proposals/2026-11-07-exp54-native-switch.md rev 0 and the stage-0 freeze runs/exp54/freeze.json
(66fcafe). Same build as N (§53): 50 bases per task, 13 rows per base, E and S keys excluded; in addition no W prompt
equals any N prompt. Seed 56 (sealed, stage 5) is never generated here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_cases as X  # noqa: E402
import exp53_cases as N  # noqa: E402

OUT = ROOT / "runs/exp54"
CASES53 = ROOT / "runs/exp53/cases53.jsonl"
FREEZE = json.loads((OUT / "freeze.json").read_text())
SEED = FREEZE["seeds"]["dev"]
SPLIT = FREEZE["data"]["split_name"]
assert SEED == 55 and SPLIT == "W"


def build():
    n_sha = {json.loads(l)["prompt_sha256"] for l in CASES53.read_text().splitlines() if l.strip()}
    old = X.old_hashes

    def old_plus_n(*a, **k):
        sha, mem, info = old(*a, **k)
        return set(sha) | n_sha, mem, {**info, "n_rows_added": len(n_sha)}
    N.SEED, N.SPLIT = SEED, SPLIT
    X.old_hashes = old_plus_n
    try:
        rows, info = N.build(FREEZE["data"]["bases_per_task"])
    finally:
        X.old_hashes = old
    for r in rows:
        r.pop("nsplit", None)
    return rows, info, n_sha


def main() -> int:
    rows, info, n_sha = build()
    pr = N.checks(rows)
    pr["n_prompt"] = [r["case_id"] for r in rows if r["prompt_sha256"] in n_sha]
    assert all(r["seed"] == SEED and r["split"] == SPLIT for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cases54.jsonl"
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    r51 = [json.loads(l) for l in N.CASES51.read_text().splitlines() if l.strip()]
    r53 = [json.loads(l) for l in CASES53.read_text().splitlines() if l.strip()]
    keyed = [r for r in rows if r["effective_key"]]
    overlap = {}
    for name, src in (("E", r51), ("S", r51), ("H", r51), ("Tcity", r51), ("N", r53)):
        ks = {tuple(r["effective_key"]) for r in src if (r["split"] == name or name == "N") and r["effective_key"]}
        overlap[name] = {"rows_with_key_in_split": sum(tuple(r["effective_key"]) in ks for r in keyed),
                         "rows_with_key": len(keyed)}
    counts = Counter(f"{r['task']}|{r['set']}" for r in rows)
    s = {"pass": not any(pr.values()), "cases54_sha256": sha, "n_rows": len(rows),
         "n_bases": len({r["base_key"] for r in rows}), "seed": SEED, "split": SPLIT,
         "excluded_key_splits": list(N.EXCLUDED), "key_overlap": overlap, "problems": pr,
         "counts": dict(sorted(counts.items())), **info}
    (OUT / "cases54_counts.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({"pass": s["pass"], "sha256": sha, "n": len(rows), "bases": s["n_bases"],
                      "problems": {k: len(v) for k, v in pr.items()}, "overlap": overlap}))
    return 0 if s["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
