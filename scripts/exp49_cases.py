#!/usr/bin/env python
"""§49: the legible-abstentions case list (independent bases, labels balanced, matched twins in one split). Data only:
the Qwen2.5 tokenizer is read (offline) for the pool-token check; no model is loaded.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp49_cases.py          # -> runs/exp49/cases49.jsonl + cases49_counts.json

Spec: research/proposals/2026-11-01-exp49-legible-abstentions.md §2 (rev 0a); D-63. Templates, pools, distractor
scheme and the F1 / F2 renderers are §45's (scripts/exp45_cases.py), imported unchanged. City-answer questions only:
DE v0 "Where does {A} live?", RC v0 "In which city does {A}'s supervisor work?", TU v0 "Where does {A} live now?"
(competitor: the old city).

Fixed choices (each recorded in cases49_counts.json "design"):
- Per task, 96 bases: even index dev, odd held (48 each). Within each task x split the gold city list is four seeded
  permutations of the 12 cities concatenated (4 bases per gold city; the first 24 bases hold every city twice), and
  the counterfactual (CF) city list is four permutations each deranged against its gold block (cf != gold; TU also
  cf != old). Everything else of a base (A, B, TU's old city, the UNA job and pet, the 10 distractor facts and their
  gap draws) is drawn by random.Random("49-{task}-{index}"), independent of the label.
- No two bases of a task share their core (DE: A and the gold; RC: A, B, gold; TU: A, old, gold): A, B and the old
  city are redrawn from the base's own stream until the core is new, so no H0 prompt repeats within or across splits.
- Distractors follow §45: 9 pool names other than A, B (the 10th fact reuses name 0 with another relation), cities
  other than every core city of the base and its twins, jobs other than the UNA job; H1 = the first 3, H2 = all 10.
- Sets per split: ANS-F2 (H0 / H1 / H2, 432 rows), CF-F2 (H1, gold -> the CF city, 144), UNA-F2 (H1, 144), ANS-F1
  (H1, the first 24 bases per task, 72). Twins share the base's distractors and gap draws.
- Rev 0b B1 (overrides rev 0a's UNA wording): every prompt carries one nuisance fact about the queried entity after
  the city fact(s): DE "{A} works as {a J}.", RC "{B} works as {a J}.", TU "{A} has {a pet}." (one more line than
  §45's templates, stated). UNA = the ANS prompt with the query's city fact(s) replaced, in place, by the same number
  of ordinary distractor lines about other people (the base's distractors 4 and 5, which H1 never uses). Same line
  count, same hardness slots; the only difference is whether the queried relation exists.
- R4 (proposal §3, build review fix 1): `prompt_r4` = the F2 prompt with both "unknown" sentences removed: " If the
  memory does not contain the answer, reply: unknown" (§23 instruction) and " Reply unknown only if the memory has
  nothing at all on the topic." (Chain-of-Note text); checked: no "unknown" (any case) remains in any prompt_r4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp45_cases as C  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

TOKENIZER_ID = C.TOKENIZER_ID
OUT = Path("runs/exp49")
SEED = 49
TASKS = ("DE", "RC", "TU")
N_BASE = 96                      # per task: 48 dev (even) + 48 held (odd)
N_F1 = 24                        # per task x split: the first 24 bases (every gold city twice)
PETS = ("cat", "dog", "rabbit", "parrot")
SETS = ("ANS", "CF", "UNA")
SIZES = {("ANS", "F2"): 432, ("CF", "F2"): 144, ("UNA", "F2"): 144, ("ANS", "F1"): 72}   # rows per split
R4_SENTENCE = " If the memory does not contain the answer, reply: unknown"
R4_SENTENCE_NOTE = " Reply unknown only if the memory has nothing at all on the topic."   # build review fix 1
QUESTIONS = {"DE": "Where does {A} live?", "RC": "In which city does {A}'s supervisor work?",
             "TU": "Where does {A} live now?"}


def split_of(index: int) -> str:
    return "dev" if index % 2 == 0 else "held"


def label_lists(task: str, split: str, seed: int = SEED) -> tuple[list[str], list[str]]:
    """The 48 gold cities and 48 CF cities of one task x split, in base order (k = index // 2)."""
    rng = random.Random(f"{seed}-{task}-{split}-labels")
    golds, cfs = [], []
    for _ in range(4):
        g = rng.sample(C.CITIES, 12)
        while True:
            c = rng.sample(C.CITIES, 12)
            if all(a != b for a, b in zip(g, c)):
                break
        golds += g
        cfs += c
    return golds, cfs


def core_key(task: str, A: str, B: str, gold: str, old: str | None) -> tuple:
    """What makes two bases' H0 ANS prompts identical."""
    return {"DE": (A, gold), "RC": (A, B, gold), "TU": (A, old, gold)}[task]


def draw_base(task: str, index: int, gold: str, cf: str, seed: int = SEED, used: set | None = None) -> dict:
    """`used`: core keys of earlier bases of this task (both splits); A, B (and TU's old city) are redrawn from the
    same stream until the key is new, so no two bases share an H0 prompt (and no prompt is in both splits)."""
    rng = random.Random(f"{seed}-{task}-{index}")
    while True:
        A, B = rng.sample(C.NAMES, 2)
        old = rng.choice([c for c in C.CITIES if c not in (gold, cf)]) if task == "TU" else None
        key = core_key(task, A, B, gold, old)
        if used is None or key not in used:
            break
    if used is not None:
        used.add(key)
    core_cities = {gold, cf} | ({old} if old else set())
    J, pet = rng.choice(C.JOBS), rng.choice(PETS)
    other_names = [n for n in C.NAMES if n not in (A, B)]
    other_cities = [c for c in C.CITIES if c not in core_cities]
    other_jobs = [j for j in C.JOBS if j != J]
    rng.shuffle(other_names); rng.shuffle(other_cities); rng.shuffle(other_jobs)
    relations = list(C.DISTRACTOR_RELATIONS)
    rng.shuffle(relations)
    if relations[9] == relations[0]:                    # §45: the 10th fact reuses name 0, so another relation
        j = next(j for j in range(1, 9) if relations[j] != relations[0])
        relations[9], relations[j] = relations[j], relations[9]
    facts, city_iter, job_iter = [], iter(other_cities), iter(other_jobs)
    for f in range(10):
        name = other_names[f] if f < 9 else other_names[0]
        rel = relations[f]
        facts.append({"name": name, "relation": rel, "value": next(job_iter) if rel == "job" else next(city_iter)})
    gap_u = [rng.random() for _ in range(10)]
    return {"task": task, "index": index, "split": split_of(index), "A": A, "B": B, "gold": gold, "cf": cf,
            "old": old, "J": J, "pet": pet, "distractors": facts, "gap_u": gap_u}


def nuisance(task: str, b: dict) -> str:
    """Rev 0b B1: one nuisance-relation fact about the queried entity, in ANS, CF and UNA alike."""
    if task == "DE":
        return f"{b['A']} works as {C.article(b['J'])} {b['J']}."
    if task == "RC":
        return f"{b['B']} works as {C.article(b['J'])} {b['J']}."
    return f"{b['A']} has {C.article(b['pet'])} {b['pet']}."


def core(task: str, b: dict, kind: str) -> dict:
    """Core facts, gold, competitors and entities of one set (ANS / CF / UNA). Rev 0b B1: the query's city fact(s)
    are followed by the nuisance fact; UNA replaces the city fact(s) with the same number of ordinary distractor
    lines about other people, the base's distractors 4 and 5 (never in its H1 list; H1 = the first 3)."""
    A, B = b["A"], b["B"]
    city = b["cf"] if kind == "CF" else b["gold"]
    if task == "DE":
        city_facts = [f"{A} lives in {city}."]
        ents = [A, city]
    elif task == "RC":
        city_facts = [f"{B} works in {city}."]
        ents = [A, B, city]
    else:
        city_facts = [f"{A} lived in {b['old']}.", f"Later, {A} moved to {city}."]
        ents = [A, b["old"], city]
    if kind == "UNA":
        city_facts = [C.distractor_sentence(d) for d in b["distractors"][3:3 + len(city_facts)]]
        ents = [e for e in ents if e not in (city, b["old"])]
    pre = [f"{A}'s supervisor is {B}."] if task == "RC" else []
    facts = pre + city_facts + [nuisance(task, b)]
    comps = [b["old"]] if (task == "TU" and kind != "UNA") else []
    return {"facts": facts, "gold": None if kind == "UNA" else city, "competitors": comps, "entities": ents,
            "question": QUESTIONS[task].format(A=A), "replacement_lines": city_facts if kind == "UNA" else []}


def r4_prompt(prompt: str) -> str:
    """Both "unknown" sentences removed (the §23 instruction's and the Chain-of-Note line, build review fix 1)."""
    for sent in (R4_SENTENCE, R4_SENTENCE_NOTE):
        if prompt.count(sent) != 1:
            raise ValueError(f"{sent!r} is not in the F2 prompt exactly once")
        prompt = prompt.replace(sent, "")
    return prompt


def cities_in(text: str) -> list[str]:
    return [c for c in C.CITIES if re.search(r"\b" + re.escape(c) + r"\b", text)]


def make_row(b: dict, kind: str, fmt: str, hard: str) -> dict:
    c = core(b["task"], b, kind)
    facts, core_idx = C.place_distractors(c["facts"], b["distractors"], b["gap_u"], C.HARDNESS[hard])
    prompt = C.render_prompt(fmt, facts, c["question"])
    base_key = f"{b['task']}{b['index']:02d}"
    return {"case_id": f"{base_key}_{kind}_{fmt}_{hard}", "base_key": base_key, "task": b["task"],
            "base_index": b["index"], "split": b["split"], "set": kind, "format": fmt, "hardness": hard,
            "facts": facts, "core_fact_indices": core_idx, "question": c["question"], "gold": c["gold"],
            "base_gold": b["gold"], "cf_city": b["cf"], "old_city": b["old"], "competitors": c["competitors"],
            "entities": c["entities"], "A": b["A"], "B": b["B"], "nuisance_job": b["J"], "nuisance_pet": b["pet"],
            "distractors": b["distractors"][:C.HARDNESS[hard]], "una_replacements": c["replacement_lines"],
            "prompt": prompt,
            "prompt_sha256": M.sha256_text(prompt), "prompt_r4": r4_prompt(prompt) if fmt == "F2" else None,
            "cities_in_prompt": cities_in(prompt)}


def build_cases(seed: int = SEED) -> list[dict]:
    rows = []
    for task in TASKS:
        lists = {s: label_lists(task, s, seed) for s in ("dev", "held")}
        used: set = set()
        for index in range(N_BASE):
            split, k = split_of(index), index // 2
            gold, cf = lists[split][0][k], lists[split][1][k]
            b = draw_base(task, index, gold, cf, seed, used)
            for hard in ("H0", "H1", "H2"):
                rows.append(make_row(b, "ANS", "F2", hard))
            rows.append(make_row(b, "CF", "F2", "H1"))
            rows.append(make_row(b, "UNA", "F2", "H1"))
            if k < N_F1:
                rows.append(make_row(b, "ANS", "F1", "H1"))
    return rows


# ---------------------------------------------------------------------------------------------- checks

def queried_entity(r: dict) -> str:
    return r["B"] if r["task"] == "RC" else r["A"]


def check_cases(rows: list[dict], tokenizer=None) -> dict:
    """Counts per split, label balance, split integrity, UNA without a queried-relation city, gold present once in
    its own fact only, no distractor touching a core entity, distinct prompts, pool tokens (with the tokenizer)."""
    pr = {k: [] for k in ("counts", "balance", "split_integrity", "una_city", "gold_placement", "distractor_overlap",
                          "prompt_duplicate", "pool_tokens")}
    for split in ("dev", "held"):
        c = Counter((r["set"], r["format"]) for r in rows if r["split"] == split)
        for key, n in SIZES.items():
            if c[key] != n:
                pr["counts"].append(f"{split} {key}: {c[key]}, want {n}")
    for (split, task, kind, fmt, hard), golds in _group(rows, ("split", "task", "set", "format", "hardness"), "gold"):
        if kind == "UNA":
            golds = [g for g in golds if g is not None]
            if golds:
                pr["balance"].append(f"{split} {task} UNA has golds")
            continue
        cnt = Counter(golds)
        if set(cnt) != set(C.CITIES) or len(set(cnt.values())) != 1:
            pr["balance"].append(f"{split} {task} {kind} {fmt} {hard}: {dict(cnt)}")
    by_base = {}
    for r in rows:
        by_base.setdefault(r["base_key"], set()).add(r["split"])
    pr["split_integrity"] = [k for k, s in by_base.items() if len(s) != 1 or s != {split_of(int(k[2:]))}]
    for r in rows:
        if r["set"] == "UNA":
            q = queried_entity(r)
            for f in r["facts"][:]:
                if re.search(r"\b" + re.escape(q) + r"\b", f) and cities_in(f):
                    pr["una_city"].append(f"{r['case_id']}: {f}")
            for city in {r["base_gold"], r["cf_city"]} | ({r["old_city"]} if r["old_city"] else set()):
                if re.search(r"\b" + re.escape(city) + r"\b", r["prompt"]):
                    pr["una_city"].append(f"{r['case_id']}: core city {city} in the prompt")
        else:
            hits = [f for f in r["facts"] if re.search(r"\b" + re.escape(r["gold"]) + r"\b", f)]
            core_facts = [r["facts"][i] for i in r["core_fact_indices"]]
            if len(hits) != 1 or hits[0] not in core_facts:
                pr["gold_placement"].append(r["case_id"])
            other = r["base_gold"] if r["set"] == "CF" else r["cf_city"]
            if re.search(r"\b" + re.escape(other) + r"\b", r["prompt"]):
                pr["gold_placement"].append(f"{r['case_id']}: twin city {other} present")
        core_names = {r["A"], r["B"]}
        core_cities = {r["base_gold"], r["cf_city"]} | ({r["old_city"]} if r["old_city"] else set())
        for d in r["distractors"]:
            if d["name"] in core_names or d["value"] in core_cities or d["value"] == r["nuisance_job"]:
                pr["distractor_overlap"].append(f"{r['case_id']}: {d}")
        if r["prompt_r4"] is not None and "unknown" in r["prompt_r4"].lower():
            pr["counts"].append(f"{r['case_id']}: 'unknown' in prompt_r4")
    nfacts = {}
    for r in rows:
        if r["format"] == "F2" and r["hardness"] == "H1":
            nfacts.setdefault(r["base_key"], {})[r["set"]] = len(r["facts"])
    pr["counts"] += [f"{k}: line counts {v}" for k, v in nfacts.items() if len(set(v.values())) != 1]
    shas = Counter(r["prompt_sha256"] for r in rows)
    pr["prompt_duplicate"] = [s for s, n in shas.items() if n > 1]
    if tokenizer is not None:
        owner = {}
        for pool in (C.NAMES, C.CITIES, C.JOBS, PETS, ("unknown",)):
            for item in pool:
                for form in M.FORMS:
                    tid, _ = M.first_content_token(tokenizer, item, form)
                    if tid in owner and owner[tid] != item:
                        pr["pool_tokens"].append(f"{item!r} and {owner[tid]!r} share {tokenizer.decode([tid])!r}")
                    owner[tid] = item
    return pr


def _group(rows, keys, field):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r[field])
    return sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0]))


def counts(rows: list[dict]) -> dict:
    def by(*keys):
        c = Counter("|".join(str(r[k]) for k in keys) for r in rows)
        return dict(sorted(c.items()))
    return {"n_rows": len(rows), "by_split_set_format": by("split", "set", "format"),
            "by_split_task_set_format_hardness": by("split", "task", "set", "format", "hardness"),
            "gold_by_split_task_ans_f2_h1": {f"{s}|{t}": dict(sorted(Counter(
                r["gold"] for r in rows if r["split"] == s and r["task"] == t and r["set"] == "ANS"
                and r["format"] == "F2" and r["hardness"] == "H1").items())) for s in ("dev", "held") for t in TASKS},
            "n_bases": len({r["base_key"] for r in rows})}


def write_cases(rows: list[dict], out: Path, problems: dict, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases49.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"seed": seed, "cases49_sha256": sha, "pass": not any(problems.values()), "problems": problems,
               "counts": counts(rows),
               "design": {"names": C.NAMES, "cities": C.CITIES, "jobs": C.JOBS, "pets": PETS,
                          "questions": QUESTIONS, "sizes_per_split": {f"{k[0]}-{k[1]}": v for k, v in SIZES.items()},
                          "r4_removed": [R4_SENTENCE, R4_SENTENCE_NOTE], "f2_instruction": C.F23_INSTRUCTION,
                          "dev": "even base index", "held": "odd base index", "tokenizer": TOKENIZER_ID,
                          "labels": "4 seeded permutations per task x split; CF deranged per block"}}
    (out / "cases49_counts.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--no-tokenizer", action="store_true")
    args = p.parse_args(argv)
    tok = None
    if not args.no_tokenizer:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    rows = build_cases(args.seed)
    problems = check_cases(rows, tok)
    s = write_cases(rows, Path(args.out), problems, args.seed)
    print(json.dumps({"pass": s["pass"], "sha256": s["cases49_sha256"], "n": len(rows),
                      "problems": {k: len(v) for k, v in problems.items()}}))
    return 0 if s["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
