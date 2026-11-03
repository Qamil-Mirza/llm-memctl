#!/usr/bin/env python
"""§51: the case list for agent-proposed weight edits (E, S at seed 51; H at 52; T-city and T-format at 53). Data
only; the Qwen2.5 tokenizer is read (offline) for the T-city single-token and pool-token checks; no model is loaded.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_cases.py --feasibility       # the finite-pool check only
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_cases.py [--scale r]          # -> runs/exp51/cases51.jsonl

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a (§3, §13.2; D-72). The §49 / §50 base
draw, nuisance fact, distractor scheme, UNA-rel replacement lines, UNA-swap entity and the F2 renderer are reused
(scripts/exp49_cases.py, scripts/exp50_cases.py); this file adds the split plan, CF twins at every hardness, the
T-city pool, the T-format wording and the closure checks.

Rows per base (13): ANS rep a and rep b (+7 days) x H0-H2; UNA-rel x H0-H2; UNA-swap at H1; CF (the evidence city
changed to the base's CF city, rep a dates) x H0-H2.

Fixed choices of this build (each stamped into cases51_counts.json "design"):
- Effective key (§13.2) = (task, A, B for RC else None, the city stated in the variant's evidence): the gold for ANS,
  the CF city for CF. TU's evidence city is the moved-to city (the old city is not in the key, as the proposal
  lists it). UNA variants state no evidence city and carry no effective key; they inherit the base's split.
- No effective key is shared by two splits. DE and TU have only 12 names x 12 cities = 144 keys per pool, so their
  keys are allocated to splits before any draw: the 144 (name, city) cells are ordered by the cyclic slot
  (city index - name index) mod 12, then name, and each split takes a contiguous run, lengthened until it holds as
  many cells of names with >= 2 cells as it has bases (a base needs its gold cell and another cell of the same name
  for its CF). Bases take the run's eligible cells in order (gold = the cell, CF = a seeded choice among the name's
  other cells in the run). Gold cities are balanced within a split up to a partial slot. RC (12 x 11 x 12 keys)
  draws A, B from the base stream and resamples until neither of its keys belongs to another split and its gold key
  is new in its split (<= 1,000 attempts, then abort). The T-city split has its own city pool, so its own grid.
- Finite-pool feasibility (§13.2): the requested sizes (E 48, S 32, H 48, T-city 24, T-format 24 bases per task) are
  checked against these capacities first; if they do not fit, every size is multiplied by the largest r in steps
  of 0.001 that fits (floor), the same r for every split and task, and r is reported.
- Each base is redrawn (attempt counter in its stream, <= 1,000) until it has a UNA-swap entity (§50 rule), none of
  its prompts has the exact sha256 of a §49 or §50 row or of an earlier §51 base's row, and none of its memory texts
  is an earlier base's memory text in another split; so every base has its 13 distinct rows (H0 prompts are short:
  "A lives in X." + the nuisance line, so a CF H0 row can equal another base's ANS H0 row).
- T-city: 12 new cities from a fixed candidate list (UK first), each one token in the " X" form and with first
  content tokens (both forms) distinct from every pool item; every city slot of a T-city base (gold, CF, old,
  distractors, UNA-rel replacement) uses this pool.
- T-format: the F2 instruction's first sentence block reworded (T_FORMAT_INSTRUCTION); everything else unchanged.
- prompt_b2 (§35): " Do not reply unknown." appended to the end of the instruction block (after the reasoning
  instructions). prompt_b3 (§49 R4): both "unknown" sentences removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp45_cases as C  # noqa: E402
import exp49_cases as G  # noqa: E402
import exp50_cases as F  # noqa: E402
from memctl.agents.llm import REASONING_INSTRUCTIONS  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

OUT = Path("runs/exp51")
TASKS = ("DE", "RC", "TU")
SPLITS = ("E", "S", "H", "Tcity", "Tformat")                    # generation order (fixed)
SEEDS = {"E": 51, "S": 51, "H": 52, "Tcity": 53, "Tformat": 53}
SIZES = {"E": 48, "S": 32, "H": 48, "Tcity": 24, "Tformat": 24}   # bases per task (§3), before any reduction
MAX_ATTEMPTS = 1000
SHIFT_DAYS = F.SHIFT_DAYS
OLD_CASES = (Path("runs/exp49/cases49.jsonl"), Path("runs/exp50/cases50.jsonl"))
T_CITY_CANDIDATES = ("London", "Oxford", "Bristol", "Bath", "Glasgow", "Belfast", "Sheffield", "Leicester", "Exeter",
                     "Plymouth", "Coventry", "Swansea", "Aberdeen", "Dundee", "Bradford", "Nottingham", "Southampton",
                     "Portsmouth", "Dublin", "Paris", "Berlin", "Rome", "Madrid", "Boston", "Dallas", "Denver",
                     "Seattle", "Chicago", "Houston", "Tokyo", "Sydney", "Toronto", "Vienna", "Prague", "Lisbon")
T_FORMAT_INSTRUCTION = ("The text below is your memory of an earlier long conversation, and after it comes a question "
                        "about that conversation. Reply with a brief phrase giving the requested fact. If the answer "
                        "is not in the memory, reply: unknown")
B2_SUFFIX = " Do not reply unknown."
INSTRUCTION_BLOCK = f"{C.F23_INSTRUCTION}\n{REASONING_INSTRUCTIONS}"


# ---------------------------------------------------------------------------------------------- pools

def t_city_pool(tokenizer) -> list[str]:
    """The first 12 candidates that are one token in the ' X' form and whose first content tokens (both forms) are
    distinct from every pool item's and from each other's."""
    owner = {}
    for pool in (C.NAMES, C.CITIES, C.JOBS, G.PETS, ("unknown",)):
        for item in pool:
            for form in M.FORMS:
                owner[M.first_content_token(tokenizer, item, form)[0]] = item
    out = []
    for c in T_CITY_CANDIDATES:
        if len(tokenizer(" " + c, add_special_tokens=False).input_ids) != 1:
            continue
        toks = {M.first_content_token(tokenizer, c, form)[0] for form in M.FORMS}
        if any(t in owner for t in toks):
            continue
        for t in toks:
            owner[t] = c
        out.append(c)
        if len(out) == 12:
            return out
    raise ValueError(f"only {len(out)} T-city candidates pass the token checks")


T_CITIES_FALLBACK = None      # filled by main() from the tokenizer; tests pass a pool explicitly


# ---------------------------------------------------------------------------------------------- keys

def key(r: dict) -> tuple:
    """§50 canonical core tuple (task, A, B for RC, base gold, old city): `drop_overlap`'s lambda, extracted (§13.2)."""
    return (r["task"], r["A"], r["B"] if r["task"] == "RC" else None, r["base_gold"], r["old_city"])


def effective_key(task: str, A: str, B: str | None, city: str) -> tuple:
    return (task, A, B if task == "RC" else None, city)


def base_effective_keys(b: dict) -> list[tuple]:
    return [effective_key(b["task"], b["A"], b["B"], b["gold"]), effective_key(b["task"], b["A"], b["B"], b["cf"])]


# ---------------------------------------------------------------------------------------------- DE / TU allocation

def grid_order(names, cities) -> list[tuple[str, str]]:
    n = len(names)
    cells = [((j - i) % n, i, names[i], cities[j]) for i in range(n) for j in range(len(cities))]
    return [(a, c) for _, _, a, c in sorted(cells)]


def eligible(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cnt = Counter(a for a, _ in cells)
    return [cell for cell in cells if cnt[cell[0]] >= 2]


def allocate(sizes: dict[str, int], names, cities) -> dict[str, list[tuple[str, str]]] | None:
    """Contiguous runs over the slot order, one per split in `sizes` order; None when they do not fit."""
    order = grid_order(names, cities)
    out, pos = {}, 0
    for split, n in sizes.items():
        if n == 0:
            out[split] = []
            continue
        length = n
        while True:
            if pos + length > len(order):
                return None
            run = order[pos:pos + length]
            if len(eligible(run)) >= n:
                break
            length += 1
        out[split] = run
        pos += length
    return out


def scaled(sizes: dict[str, int], r: float) -> dict[str, int]:
    return {k: int(math.floor(v * r + 1e-9)) for k, v in sizes.items()}


def feasibility(sizes: dict[str, int] = SIZES, t_pool_size: int = 12) -> dict:
    """§13.2: does the finite pool hold the requested sizes? DE / TU share one grid for E, S, H, T-format and use a
    second (T-city) grid; RC's grid is 12 x 11 x 12 per split pool. Returns the largest uniform r (step 0.001)."""
    names, cities = list(C.NAMES), list(C.CITIES)
    t_cities = [f"T{i}" for i in range(t_pool_size)]

    def fits(r: float) -> bool:
        s = scaled(sizes, r)
        main = {k: v for k, v in s.items() if k != "Tcity"}
        rc_cap = len(names) * (len(names) - 1) * len(cities)
        return (allocate(main, names, cities) is not None and allocate({"Tcity": s["Tcity"]}, names, t_cities)
                is not None and sum(main.values()) * 2 <= rc_cap)
    r = 1.0
    while r > 0 and not fits(r):
        r = round(r - 0.001, 3)
    need_de = sum(v for k, v in sizes.items() if k != "Tcity")
    return {"requested": dict(sizes), "de_tu_cells": len(names) * len(cities), "de_tu_bases_requested": need_de,
            "fits_at_1": fits(1.0), "r": r, "sizes": scaled(sizes, r)}


# ---------------------------------------------------------------------------------------------- base draw

def draw_rest(rng: random.Random, task: str, index: int, A: str, B: str, gold: str, cf: str, names, cities) -> dict:
    """§49 draw_base after the core (old city, J, pet, 10 distractors, gaps), over the given pools."""
    old = rng.choice([c for c in cities if c not in (gold, cf)]) if task == "TU" else None
    core_cities = {gold, cf} | ({old} if old else set())
    J, pet = rng.choice(C.JOBS), rng.choice(G.PETS)
    other_names = [n for n in names if n not in (A, B)]
    other_cities = [c for c in cities if c not in core_cities]
    other_jobs = [j for j in C.JOBS if j != J]
    rng.shuffle(other_names); rng.shuffle(other_cities); rng.shuffle(other_jobs)
    relations = list(C.DISTRACTOR_RELATIONS)
    rng.shuffle(relations)
    if relations[9] == relations[0]:
        j = next(j for j in range(1, 9) if relations[j] != relations[0])
        relations[9], relations[j] = relations[j], relations[9]
    facts, city_iter, job_iter = [], iter(other_cities), iter(other_jobs)
    for f in range(10):
        name = other_names[f] if f < 9 else other_names[0]
        rel = relations[f]
        facts.append({"name": name, "relation": rel, "value": next(job_iter) if rel == "job" else next(city_iter)})
    gap_u = [rng.random() for _ in range(10)]
    return {"task": task, "index": index, "A": A, "B": B, "gold": gold, "cf": cf, "old": old, "J": J, "pet": pet,
            "distractors": facts, "gap_u": gap_u}


def replacement_facts(b: dict, seed: int, cities) -> list[str]:
    """§50 UNA-rel replacement lines over the base's city pool (N = a pool name no distractor uses)."""
    used_names = {b["A"], b["B"]} | {d["name"] for d in b["distractors"]}
    free_names = [n for n in C.NAMES if n not in used_names]
    used_cities = {b["gold"], b["cf"]} | ({b["old"]} if b["old"] else set()) | \
        {d["value"] for d in b["distractors"] if d["relation"] != "job"}
    free_cities = [c for c in cities if c not in used_cities]
    used_jobs = {b["J"]} | {d["value"] for d in b["distractors"] if d["relation"] == "job"}
    free_jobs = [j for j in C.JOBS if j not in used_jobs]
    rng = random.Random(f"{seed}-{b['split']}-{b['task']}-{b['index']}-replacement")
    n, city, job = rng.choice(free_names), rng.choice(free_cities), rng.choice(free_jobs)
    k = 2 if b["task"] == "TU" else 1
    return [f"{n} lives in {city}.", f"{n} works as {C.article(job)} {job}."][:k]


# ---------------------------------------------------------------------------------------------- rows

def render(facts, question, shift, instruction_block):
    lines = [f"user, {F.f2_date(i, shift)} 10:00: {fact}" for i, fact in enumerate(facts)]
    asked = f"Question: (asked on {F.f2_date(len(facts), shift)} 12:00) {question}"
    return "\n".join([instruction_block, "", "## Memory", *lines, "", "## Current input", asked, "", "Note:"])


def instruction_for(split: str) -> str:
    return f"{T_FORMAT_INSTRUCTION}\n{REASONING_INSTRUCTIONS}" if split == "Tformat" else INSTRUCTION_BLOCK


def b2_prompt(prompt: str, block: str) -> str:
    if prompt.count(block) != 1:
        raise ValueError("instruction block not found once")
    return prompt.replace(block, block + B2_SUFFIX)


def b3_prompt(prompt: str, split: str) -> str:
    if split == "Tformat":
        sents = (" If the answer is not in the memory, reply: unknown", G.R4_SENTENCE_NOTE)
    else:
        sents = (G.R4_SENTENCE, G.R4_SENTENCE_NOTE)
    for s in sents:
        if prompt.count(s) != 1:
            raise ValueError(f"{s!r} is not in the prompt exactly once")
        prompt = prompt.replace(s, "")
    return prompt


def core51(b: dict, kind: str, cities) -> dict:
    task = b["task"]
    if kind in ("ANS", "UNA-swap"):
        return G.core(task, b, "ANS")
    if kind == "CF":
        return G.core(task, b, "CF")
    c = G.core(task, b, "UNA")
    rep = replacement_facts(b, b["seed"], cities)
    old = c["replacement_lines"]
    return {**c, "facts": [rep[old.index(f)] if f in old else f for f in c["facts"]], "replacement_lines": rep}


def make_row(b: dict, kind: str, rep: str, hard: str, cities) -> dict | None:
    task, split = b["task"], b["split"]
    c = core51(b, kind, cities)
    facts, core_idx = C.place_distractors(c["facts"], b["distractors"], b["gap_u"], C.HARDNESS[hard])
    question, gold = c["question"], c["gold"]
    query = b["B"] if task == "RC" else b["A"]
    swap_d = None
    if kind == "UNA-swap":
        swap_d = F.swap_entity(facts, b["distractors"][:C.HARDNESS[hard]])
        if swap_d is None:
            return None
        question, gold, query = F.SWAP_Q[task].format(D=swap_d), None, swap_d
    if kind == "UNA-rel":
        gold = None
    shift = SHIFT_DAYS if rep == "b" else 0
    block = instruction_for(split)
    prompt = render(facts, question, shift, block)
    base_key = f"{split}-{task}{b['index']:02d}"
    tag = {"ANS": f"ANS{rep}", "UNA-rel": "UNArel", "UNA-swap": "UNAswap", "CF": "CF"}[kind]
    ev_city = b["cf"] if kind == "CF" else (b["gold"] if kind == "ANS" else None)
    return {"case_id": f"{base_key}_{tag}_{hard}", "base_key": base_key, "split": split, "seed": b["seed"],
            "task": task, "base_index": b["index"], "set": kind, "rep": rep, "format": "F2", "hardness": hard,
            "facts": facts, "core_fact_indices": core_idx, "question": question, "gold": gold,
            "base_gold": b["gold"], "cf_city": b["cf"], "old_city": b["old"],
            "competitors": c["competitors"] if kind in ("ANS", "CF") else [], "A": b["A"], "B": b["B"],
            "query_entity": query, "swap_entity": swap_d, "nuisance": G.nuisance(task, b),
            "distractors": b["distractors"][:C.HARDNESS[hard]],
            "una_replacements": c["replacement_lines"] if kind == "UNA-rel" else [],
            "effective_key": list(effective_key(task, b["A"], b["B"], ev_city)) if ev_city else None,
            "city_pool": "T" if split == "Tcity" else "main",
            "prompt": prompt, "prompt_sha256": M.sha256_text(prompt),
            "prompt_b2": b2_prompt(prompt, block), "prompt_b3": b3_prompt(prompt, split),
            "cities_in_prompt": [x for x in cities if re.search(r"\b" + re.escape(x) + r"\b", prompt)]}


def base_rows(b: dict, cities) -> list[dict] | None:
    rows = []
    for hard in ("H0", "H1", "H2"):
        for rep in ("a", "b"):
            rows.append(make_row(b, "ANS", rep, hard, cities))
        rows.append(make_row(b, "UNA-rel", "a", hard, cities))
        rows.append(make_row(b, "CF", "a", hard, cities))
    sw = make_row(b, "UNA-swap", "a", "H1", cities)
    if sw is None:
        return None
    rows.append(sw)
    return rows


def old_hashes(paths=OLD_CASES) -> tuple[set, set, dict]:
    shas, keys, info = set(), set(), {}
    for p in paths:
        if not Path(p).exists():
            info[str(p)] = "missing"
            continue
        rs = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
        shas |= {r["prompt_sha256"] for r in rs}
        keys |= {key({**r, "old_city": r.get("old_city")}) for r in rs}
        info[str(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return shas, keys, info


def rc_labels(seed: int, split: str, n: int, cities) -> tuple[list[str], list[str]]:
    """RC gold / CF lists: seeded permutations of the pool (cf deranged per block), truncated to n (§49 scheme)."""
    rng = random.Random(f"{seed}-RC-{split}-labels")
    golds, cfs = [], []
    while len(golds) < n:
        g = rng.sample(list(cities), len(cities))
        while True:
            c = rng.sample(list(cities), len(cities))
            if all(a != b for a, b in zip(g, c)):
                break
        golds += g
        cfs += c
    return golds[:n], cfs[:n]


def build_cases(sizes: dict[str, int], t_cities: list[str], old_paths=OLD_CASES) -> tuple[list[dict], dict]:
    old_sha, old_keys, old_info = old_hashes(old_paths)
    names = list(C.NAMES)
    alloc = {"main": allocate({k: v for k, v in sizes.items() if k != "Tcity"}, names, list(C.CITIES)),
             "T": allocate({"Tcity": sizes["Tcity"]}, names, t_cities)}
    if alloc["main"] is None or alloc["T"] is None:
        raise ValueError(f"sizes {sizes} do not fit the pools: run --feasibility")
    owner: dict[tuple, str] = {}
    for pool, a in alloc.items():
        for split, cells in a.items():
            for task in ("DE", "TU"):
                for A, city in cells:
                    owner[effective_key(task, A, None, city)] = split
    rows, info = [], {"attempts": Counter(), "redraw_reasons": Counter(), "old_core_overlap_bases": 0}
    gold_keys: dict[str, set] = defaultdict(set)
    used_sha: set = set()
    used_mem: dict[str, str] = {}
    for split in SPLITS:
        seed = SEEDS[split]
        cities = t_cities if split == "Tcity" else list(C.CITIES)
        cells = alloc["T" if split == "Tcity" else "main"][split]
        elig = eligible(cells)
        for task in TASKS:
            n = sizes[split]
            if task == "RC":
                golds, cfs = rc_labels(seed, split, n, cities)
            for k in range(n):
                for attempt in range(MAX_ATTEMPTS):
                    rng = random.Random(f"{seed}-{split}-{task}-{k}-{attempt}")
                    if task == "RC":
                        gold, cf = golds[k], cfs[k]
                        A, B = rng.sample(names, 2)
                        ks = [effective_key(task, A, B, gold), effective_key(task, A, B, cf)]
                        if any(owner.get(x, split) != split for x in ks) or ks[0] in gold_keys[split]:
                            info["redraw_reasons"]["rc_key"] += 1
                            continue
                    else:
                        A, gold = elig[k]
                        others = [c for a, c in cells if a == A and c != gold]
                        cf = random.Random(f"{seed}-{split}-{task}-{k}-cf").choice(others)
                        B = rng.choice([x for x in names if x != A])
                    b = draw_rest(rng, task, k, A, B, gold, cf, names, cities)
                    b.update({"split": split, "seed": seed})
                    rs = base_rows(b, cities)
                    if rs is None:
                        info["redraw_reasons"]["no_swap_entity"] += 1
                        continue
                    if any(r["prompt_sha256"] in old_sha for r in rs):
                        info["redraw_reasons"]["old_prompt_hash"] += 1
                        continue
                    if any(r["prompt_sha256"] in used_sha for r in rs):
                        info["redraw_reasons"]["prompt_repeats_another_base"] += 1
                        continue
                    if any(used_mem.get(M.sha256_text(memory_text(r["prompt"])), split) != split for r in rs):
                        info["redraw_reasons"]["memory_text_in_another_split"] += 1
                        continue
                    break
                else:
                    raise RuntimeError(f"{split} {task} base {k}: no valid draw in {MAX_ATTEMPTS} attempts")
                info["attempts"][attempt] += 1
                used_sha.update(r["prompt_sha256"] for r in rs)
                for r in rs:
                    used_mem[M.sha256_text(memory_text(r["prompt"]))] = split
                for x in base_effective_keys(b):
                    owner[x] = split
                gold_keys[split].add(effective_key(task, A, B, gold))
                info["old_core_overlap_bases"] += key(rs[0]) in old_keys
                rows += rs
    info["attempts"] = {str(k): v for k, v in sorted(info["attempts"].items())}
    info["redraw_reasons"] = dict(info["redraw_reasons"])
    info["old_cases"] = old_info
    info["allocation"] = {pool: {s: len(c) for s, c in a.items()} for pool, a in alloc.items()}
    return rows, info


# ---------------------------------------------------------------------------------------------- checks

def content_groups(rows: list[dict]) -> dict[str, int]:
    """§13.2 closure: bases linked when any of their variants share an effective key; returns base -> group id."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        parent.setdefault(r["base_key"], r["base_key"])
    first = {}
    for r in rows:
        if r["effective_key"] is None:
            continue
        k = tuple(r["effective_key"])
        if k in first:
            a, b = find(first[k]), find(r["base_key"])
            if a != b:
                parent[a] = b
        else:
            first[k] = r["base_key"]
    roots = sorted({find(x) for x in parent})
    gid = {r: i for i, r in enumerate(roots)}
    return {x: gid[find(x)] for x in parent}


def memory_text(prompt: str) -> str:
    return prompt.split("## Memory")[1].split("## Current input")[0]


def check_cases(rows: list[dict], t_cities: list[str], tokenizer=None) -> dict:
    pr = {k: [] for k in ("rows_per_base", "closure", "canonical_key", "memory_text", "duplicate", "cf", "swap",
                          "una_city", "t_city", "b2_b3", "pool_tokens")}
    per_base = Counter(r["base_key"] for r in rows)
    pr["rows_per_base"] = [f"{k}: {n}" for k, n in per_base.items() if n != 13]
    split_of = {r["base_key"]: r["split"] for r in rows}
    groups = content_groups(rows)
    gsplits = defaultdict(set)
    for b, g in groups.items():
        gsplits[g].add(split_of[b])
    pr["closure"] = [f"group {g}: {sorted(s)}" for g, s in gsplits.items() if len(s) > 1]
    ck = defaultdict(set)
    for r in rows:
        ck[key(r)].add(r["split"])
    pr["canonical_key"] = [str(k) for k, s in ck.items() if len(s) > 1]
    mt = defaultdict(set)
    for r in rows:
        mt[M.sha256_text(memory_text(r["prompt"]))].add(r["split"])
    pr["memory_text"] = [k for k, s in mt.items() if len(s) > 1]
    shas = Counter(r["prompt_sha256"] for r in rows)
    pr["duplicate"] = [s for s, n in shas.items() if n > 1]
    by = {r["case_id"]: r for r in rows}
    for r in rows:
        pool = t_cities if r["split"] == "Tcity" else list(C.CITIES)
        if r["set"] == "CF":
            a = by[r["case_id"].replace("_CF_", "_ANSa_")]
            if r["gold"] != r["cf_city"] or r["cf_city"] == r["base_gold"] or \
                    re.search(r"\b" + re.escape(r["base_gold"]) + r"\b", r["prompt"]) or \
                    len(r["facts"]) != len(a["facts"]):
                pr["cf"].append(r["case_id"])
        if r["set"] == "UNA-swap":
            a = by[r["case_id"].replace("_UNAswap_", "_ANSa_")]
            if memory_text(a["prompt"]) != memory_text(r["prompt"]):
                pr["swap"].append(r["case_id"])
        if r["set"] == "UNA-rel":
            q = r["query_entity"]
            if [f for f in r["facts"] if re.search(r"\b" + re.escape(q) + r"\b", f) and
                    any(re.search(r"\b" + re.escape(c) + r"\b", f) for c in pool)]:
                pr["una_city"].append(r["case_id"])
        if r["split"] == "Tcity" and [c for c in C.CITIES if re.search(r"\b" + re.escape(c) + r"\b", r["prompt"])]:
            pr["t_city"].append(r["case_id"])
        if r["prompt_b2"].count(B2_SUFFIX) != 1 or "unknown" in r["prompt_b3"].lower():
            pr["b2_b3"].append(r["case_id"])
    if tokenizer is not None:
        for c in t_cities:
            if len(tokenizer(" " + c, add_special_tokens=False).input_ids) != 1:
                pr["pool_tokens"].append(c)
    return pr


def summary(rows: list[dict], info: dict, pr: dict, sizes: dict, feas: dict, t_cities: list[str], sha: str) -> dict:
    counts = Counter(f"{r['split']}|{r['task']}|{r['set']}" for r in rows)
    golds = {f"{s}|{t}": dict(sorted(Counter(r["gold"] for r in rows if r["split"] == s and r["task"] == t and
                                               r["set"] == "ANS" and r["rep"] == "a" and r["hardness"] == "H1").items()))
             for s in SPLITS for t in TASKS}
    partial = Counter()
    pairs = defaultdict(set)
    for r in rows:
        if r["set"] == "ANS":
            pairs[(r["task"], r["A"], r["base_gold"])].add(r["split"])
    partial["person_city_pairs_in_2plus_splits"] = sum(len(s) > 1 for s in pairs.values())
    return {"pass": not any(pr.values()), "cases51_sha256": sha, "n_rows": len(rows),
            "n_bases": len({r["base_key"] for r in rows}), "sizes": sizes, "feasibility": feas,
            "problems": pr, "counts": dict(sorted(counts.items())), "gold_balance_ANSa_H1": golds,
            "partial_overlap": dict(partial), **info,
            "design": {"seeds": SEEDS, "t_cities": t_cities, "t_format_instruction": T_FORMAT_INSTRUCTION,
                       "b2_suffix": B2_SUFFIX, "shift_days": SHIFT_DAYS, "max_attempts": MAX_ATTEMPTS,
                       "effective_key": "(task, A, B if RC, evidence city); TU evidence city = moved-to city",
                       "rows_per_base": 13}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feasibility", action="store_true")
    p.add_argument("--scale", type=float, default=None, help="uniform size factor (default: the feasibility r)")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    feas = feasibility()
    if a.feasibility:
        print(json.dumps(feas))
        return 0
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(G.TOKENIZER_ID)
    t_cities = t_city_pool(tok)
    r = feas["r"] if a.scale is None else min(a.scale, feas["r"])
    sizes = scaled(SIZES, r)
    rows, info = build_cases(sizes, t_cities)
    pr = check_cases(rows, t_cities, tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases51.jsonl"
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    s = summary(rows, info, pr, sizes, {**feas, "r_used": r}, t_cities, sha)
    (out / "cases51_counts.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({"pass": s["pass"], "sha256": sha, "n": len(rows), "sizes": sizes, "r": r,
                      "problems": {k: len(v) for k, v in pr.items()}, "attempts": info["attempts"],
                      "redraws": info["redraw_reasons"], "old_core_overlap_bases": info["old_core_overlap_bases"]}))
    return 0 if s["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
