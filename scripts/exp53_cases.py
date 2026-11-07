#!/usr/bin/env python
"""§53: the N split (seed 54) with the §51 generator, rows and checks. Data only (tokenizer read offline).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp53_cases.py          # -> runs/exp53/cases53.jsonl + counts

Spec: research/proposals/2026-11-05-exp53-head-controls.md rev 1 (Data). Fixed choices of this build:
- N_PER_TASK bases per task (DE, RC, TU), 13 rows per base exactly as §51 (ANS a/b x H0-H2, UNA-rel x H0-H2,
  UNA-swap H1, CF x H0-H2), F2 format, the §51 instruction block, prompt_b2 / prompt_b3 as §51.
- Disjointness (rev 1a ruling 2): no N effective key is an E key (the editor saw E) or an S key (where selection
  happened); N may share keys with H or T-city. Every N prompt differs (sha256) from every §49, §50 and §51 row and from every other
  N row, and no N memory text equals an E memory text.
- DE / TU: the 144 (name, city) cells minus E's and S's cells, in the §51 slot order; N takes the first contiguous run with
  N_PER_TASK eligible cells (names with >= 2 cells in the run, for the CF), as §51's allocate. RC: A, B redrawn
  until neither key is an E or S key and the gold key is new within N. The overlap with H and T-city is reported.
- N-dev = bases with index % 5 == 0 (20%); N-held = the rest (80%), read once.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp45_cases as C  # noqa: E402
import exp49_cases as G  # noqa: E402
import exp51_cases as X  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

OUT = ROOT / "runs/exp53"
CASES51 = ROOT / "runs/exp51/cases51.jsonl"
SEED = 54
SPLIT = "N"
N_PER_TASK = 50
DEV_MOD = 5
EXCLUDED = ("E", "S")


def nsplit(index: int) -> str:
    return "dev" if index % DEV_MOD == 0 else "held"


def build(n: int = N_PER_TASK) -> tuple[list[dict], dict]:
    old_sha, _, old_info = X.old_hashes()
    r51 = [json.loads(l) for l in CASES51.read_text().splitlines() if l.strip()]
    e_keys = {tuple(r["effective_key"]) for r in r51 if r["split"] in EXCLUDED and r["effective_key"]}
    e_mem = {M.sha256_text(X.memory_text(r["prompt"])) for r in r51 if r["split"] == "E"}
    used_sha = set(old_sha) | {r["prompt_sha256"] for r in r51}
    names, cities = list(C.NAMES), list(C.CITIES)
    e_cells = {(k[1], k[3]) for k in e_keys if k[0] == "DE"} | {(k[1], k[3]) for k in e_keys if k[0] == "TU"}
    order = [c for c in X.grid_order(names, cities) if c not in e_cells]
    length = n
    while len(X.eligible(order[:length])) < n:
        length += 1
        if length > len(order):
            raise ValueError("N does not fit the DE / TU cells outside E")
    cells = order[:length]
    elig = X.eligible(cells)
    rows, info = [], {"attempts": Counter(), "redraw_reasons": Counter()}
    gold_keys = set()
    for task in X.TASKS:
        if task == "RC":
            golds, cfs = X.rc_labels(SEED, SPLIT, n, cities)
        for k in range(n):
            for attempt in range(X.MAX_ATTEMPTS):
                rng = random.Random(f"{SEED}-{SPLIT}-{task}-{k}-{attempt}")
                if task == "RC":
                    gold, cf = golds[k], cfs[k]
                    A, B = rng.sample(names, 2)
                    ks = [X.effective_key(task, A, B, gold), X.effective_key(task, A, B, cf)]
                    if any(x in e_keys for x in ks) or ks[0] in gold_keys:
                        info["redraw_reasons"]["rc_key"] += 1
                        continue
                else:
                    A, gold = elig[k]
                    others = [c for a, c in cells if a == A and c != gold]
                    cf = random.Random(f"{SEED}-{SPLIT}-{task}-{k}-cf").choice(others)
                    B = rng.choice([x for x in names if x != A])
                b = X.draw_rest(rng, task, k, A, B, gold, cf, names, cities)
                b.update({"split": SPLIT, "seed": SEED})
                rs = X.base_rows(b, cities)
                if rs is None:
                    info["redraw_reasons"]["no_swap_entity"] += 1
                    continue
                if any(r["prompt_sha256"] in used_sha for r in rs):
                    info["redraw_reasons"]["prompt_repeats"] += 1
                    continue
                if any(M.sha256_text(X.memory_text(r["prompt"])) in e_mem for r in rs):
                    info["redraw_reasons"]["memory_text_in_E"] += 1
                    continue
                if any(tuple(r["effective_key"]) in e_keys for r in rs if r["effective_key"]):
                    info["redraw_reasons"]["e_key"] += 1
                    continue
                break
            else:
                raise RuntimeError(f"N {task} base {k}: no valid draw")
            info["attempts"][attempt] += 1
            used_sha.update(r["prompt_sha256"] for r in rs)
            gold_keys.add(X.effective_key(task, A, B, gold))
            for r in rs:
                r["nsplit"] = nsplit(k)
            rows += rs
    info["attempts"] = {str(k): v for k, v in sorted(info["attempts"].items())}
    info["redraw_reasons"] = dict(info["redraw_reasons"])
    info["old_cases"] = old_info
    info["cases51_sha256"] = hashlib.sha256(CASES51.read_bytes()).hexdigest()
    return rows, info


def checks(rows: list[dict]) -> dict:
    pr = X.check_cases(rows, [], None)
    r51 = [json.loads(l) for l in CASES51.read_text().splitlines() if l.strip()]
    e_keys = {tuple(r["effective_key"]) for r in r51 if r["split"] == "E" and r["effective_key"]}
    s_keys = {tuple(r["effective_key"]) for r in r51 if r["split"] == "S" and r["effective_key"]}
    pr["e_key_overlap"] = [r["case_id"] for r in rows if r["effective_key"] and tuple(r["effective_key"]) in e_keys]
    pr["s_key_overlap"] = [r["case_id"] for r in rows if r["effective_key"] and tuple(r["effective_key"]) in s_keys]
    old = X.old_hashes()[0] | {r["prompt_sha256"] for r in r51}
    pr["old_prompt"] = [r["case_id"] for r in rows if r["prompt_sha256"] in old]
    return pr


def main() -> int:
    rows, info = build()
    pr = checks(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cases53.jsonl"
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = Counter(f"{r['nsplit']}|{r['task']}|{r['set']}" for r in rows)
    r51 = [json.loads(l) for l in CASES51.read_text().splitlines() if l.strip()]
    overlap = {}
    for sp in ("E", "S", "H", "Tcity"):
        ks = {tuple(r["effective_key"]) for r in r51 if r["split"] == sp and r["effective_key"]}
        keyed = [r for r in rows if r["effective_key"]]
        overlap[sp] = {"rows_with_key_in_split": sum(tuple(r["effective_key"]) in ks for r in keyed),
                       "rows_with_key": len(keyed)}
    s = {"pass": not any(pr.values()), "cases53_sha256": sha, "n_rows": len(rows),
         "n_bases": len({r["base_key"] for r in rows}), "n_per_task": N_PER_TASK, "seed": SEED,
         "dev_rule": f"base index % {DEV_MOD} == 0", "excluded_key_splits": list(EXCLUDED),
         "key_overlap": overlap, "problems": pr, "counts": dict(sorted(counts.items())), **info}
    (OUT / "cases53_counts.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({"pass": s["pass"], "sha256": sha, "n": len(rows), "problems": {k: len(v) for k, v in pr.items()},
                      "attempts": info["attempts"], "redraws": info["redraw_reasons"]}))
    return 0 if s["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
