#!/usr/bin/env python
"""§45: instantiate the fixed case list (templates x pools x seed) and the §44 rev 8 stage 3 rows. Data only: the
Qwen2.5 tokenizer is read (offline) for the pinned ids; no model is loaded.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp45_cases.py --seed 45            # -> runs/exp45_maps/cases.jsonl + cases_counts.json
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp45_cases.py --rows44             # -> runs/exp45_maps/rows44.jsonl (stage 3)

Spec: research/proposals/2026-10-30-exp45-jlens-maps.md §3 (rev 0, 78d5ad1). Three task types x 12 base cases x four
variants (v0 base, v1 substitution, v2 question change, v3 order change) x two formats (F1 plain chat, F2 the §23
memory-note wrapper) x three hardness levels (H0 bare, H1 + 3 distractor facts, H2 + 10) = 864 prompts. Base cases
with an even index are the dev half, odd the held half.

Fixed choices (rev 0 leaves them open; each is recorded in cases_counts.json "design"):
- Per task type, A, C1, J are permutations of the pools over the 12 base cases; B = the next name in that
  permutation, B' the one after, C2 the next city: so every (task, variant) has 12 distinct golds across base cases.
- Distractor facts (H1, H2) are about pool people other than A, B, B' (9 names: the 10th fact reuses the first
  name with another relation), with cities other than C1, C2 (drawn without replacement) and jobs other than J. Three
  relations, 4 / 3 / 3 of the 10: "{X} lives in {Y}.", "{X} works as a {J}.", "{X} works in {Y}.". H1 is the first 3
  of the H2 list (H1 is a subset of H2). Each distractor gets a gap (0..n_core) by a uniform draw scaled to the
  variant's number of core facts; core facts keep their order; the same draws serve every variant and format of a
  base case, so variants differ only in their manipulation.
- DE v3 places the second fact before A's fact for base cases 0, 1, 4, 5, 8, 9 and after it for 2, 3, 6, 7, 10, 11
  (dev and held each get both placements).
- Competitors: the core-fact entities of the gold's kind other than the gold (TU: the other city; RC v2: A; DE:
  none). RC's intermediate is B (B' in v1) except in v2, where B is the gold.
- F1: the facts one per line, a blank line, the question. F2: the §23 instruction and Chain-of-Note text verbatim,
  "## Memory", one "user, <date> (<day>) 10:00: <fact>" line per fact (one day apart from 2023/03/01), "## Current
  input", "Question: (asked on <next day> (<day>) 12:00) <question>", "Note:".
- The months pool is defined (the proposal lists it) but no template uses it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.agents.llm import REASONING_INSTRUCTIONS  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT = Path("runs/exp45_maps")
SEED = 45
TASKS = ("DE", "RC", "TU")
VARIANTS = ("v0", "v1", "v2", "v3")
FORMATS = ("F1", "F2")
HARDNESS = {"H0": 0, "H1": 3, "H2": 10}
N_BASE = 12

# Pools: single-token in the " X" form, and every first content token (both forms) distinct across all three pools.
NAMES = ("Alice", "Ben", "David", "Emma", "Henry", "James", "Kate", "Laura", "Paul", "Rachel", "Sarah", "Peter")
CITIES = ("York", "Manchester", "Liverpool", "Leeds", "Cardiff", "Cambridge", "Edinburgh", "Brighton", "Norwich",
          "Durham", "Derby", "Chester")
JOBS = ("teacher", "doctor", "chef", "artist", "writer", "cook", "driver", "manager", "actor", "editor", "scientist",
        "designer")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

F23_INSTRUCTION = ("Below is your memory of a long conversation, followed by a question about it. Answer with a short "
                   "phrase that states the fact asked for. If the memory does not contain the answer, reply: unknown")
F2_START = dt.date(2023, 3, 1)
DISTRACTOR_RELATIONS = ("lives", "lives", "lives", "lives", "job", "job", "job", "works_in", "works_in", "works_in")


def article(job: str) -> str:
    return "an" if job[0].lower() in "aeiou" else "a"


def distractor_sentence(d: dict) -> str:
    if d["relation"] == "lives":
        return f"{d['name']} lives in {d['value']}."
    if d["relation"] == "job":
        return f"{d['name']} works as {article(d['value'])} {d['value']}."
    return f"{d['name']} works in {d['value']}."


def draw_bases(seed: int = SEED) -> dict[str, list[dict]]:
    """Per task type, the 12 base cases' entities, distractor lists (10) and gap draws."""
    out = {}
    for task in TASKS:
        rng = random.Random(f"{seed}-{task}")
        names, cities, jobs = rng.sample(NAMES, 12), rng.sample(CITIES, 12), rng.sample(JOBS, 12)
        bases = []
        for i in range(N_BASE):
            A, B, B2 = names[i], names[(i + 1) % 12], names[(i + 2) % 12]
            C1, C2, J = cities[i], cities[(i + 1) % 12], jobs[i]
            other_names = [n for n in NAMES if n not in (A, B, B2)]
            other_cities = [c for c in CITIES if c not in (C1, C2)]
            other_jobs = [j for j in JOBS if j != J]
            rng.shuffle(other_names); rng.shuffle(other_cities); rng.shuffle(other_jobs)
            relations = list(DISTRACTOR_RELATIONS)
            rng.shuffle(relations)
            if relations[9] == relations[0]:                    # the 10th fact reuses name 0: another relation
                j = next(j for j in range(1, 9) if relations[j] != relations[0])
                relations[9], relations[j] = relations[j], relations[9]
            facts, city_iter, job_iter = [], iter(other_cities), iter(other_jobs)
            for f in range(10):
                name = other_names[f] if f < 9 else other_names[0]
                rel = relations[f]
                value = next(job_iter) if rel == "job" else next(city_iter)
                facts.append({"name": name, "relation": rel, "value": value})
            gaps_u = [rng.random() for _ in range(10)]
            bases.append({"index": i, "A": A, "B": B, "B2": B2, "C1": C1, "C2": C2, "J": J, "distractors": facts,
                          "gap_u": gaps_u, "v3_before": (i // 2) % 2 == 0})
        out[task] = bases
    return out


def core_variant(task: str, b: dict, variant: str) -> dict:
    """Core facts, question, gold, competitors, intermediate and entities (role, surface) of one variant."""
    A, B, B2, C1, C2, J = b["A"], b["B"], b["B2"], b["C1"], b["C2"], b["J"]
    inter = None
    if task == "DE":
        city = C2 if variant == "v1" else C1
        facts = [f"{A} lives in {city}."]
        if variant == "v3":
            extra = f"{B} works as {article(J)} {J}."
            facts = [extra] + facts if b["v3_before"] else facts + [extra]
        q, gold = (f"Who lives in {city}?", A) if variant == "v2" else (f"Where does {A} live?", city)
        ents = [A, city] + ([B, J] if variant == "v3" else [])
        competitors = []
    elif task == "RC":
        sup = B2 if variant == "v1" else B
        facts = [f"{A}'s supervisor is {sup}.", f"{sup} works in {C1}."]
        if variant == "v3":
            facts = facts[::-1]
        if variant == "v2":
            q, gold, competitors = f"Who is {A}'s supervisor?", sup, [A]
        else:
            q, gold, competitors, inter = f"In which city does {A}'s supervisor work?", C1, [], sup
        ents = [A, sup, C1]
    else:
        first, second = (C2, C1) if variant == "v1" else (C1, C2)
        if variant == "v3":
            facts = [f"{A} now lives in {second}.", f"Before that, {A} lived in {first}."]
        else:
            facts = [f"{A} lived in {first}.", f"Later, {A} moved to {second}."]
        if variant == "v2":
            q, gold, competitors = f"Where did {A} live before moving?", first, [second]
        else:
            q, gold, competitors = f"Where does {A} live now?", second, [first]
        ents = [A, first, second]
    return {"facts": facts, "question": q, "gold": gold, "competitors": competitors, "intermediate": inter,
            "entities": ents}


def place_distractors(core: list[str], distractors: list[dict], gap_u: list[float], n: int) -> tuple[list[str], list[int]]:
    """The fact list of one hardness: core facts in order, the first n distractors in their gaps (draw order kept).
    Returns (facts, indices of the core facts in it)."""
    gaps = [[] for _ in range(len(core) + 1)]
    for d, u in zip(distractors[:n], gap_u[:n]):
        gaps[min(int(u * (len(core) + 1)), len(core))].append(distractor_sentence(d))
    facts, core_idx = list(gaps[0]), []
    for c, tail in zip(core, gaps[1:]):
        core_idx.append(len(facts))
        facts.append(c)
        facts.extend(tail)
    return facts, core_idx


def f2_date(i: int) -> str:
    d = F2_START + dt.timedelta(days=i)
    return f"{d:%Y/%m/%d} ({d:%a})"


def render_prompt(fmt: str, facts: list[str], question: str) -> str:
    if fmt == "F1":
        return "\n".join(facts) + "\n\n" + question
    lines = [f"user, {f2_date(i)} 10:00: {fact}" for i, fact in enumerate(facts)]
    asked = f"Question: (asked on {f2_date(len(facts))} 12:00) {question}"
    return "\n".join([f"{F23_INSTRUCTION}\n{REASONING_INSTRUCTIONS}", "", "## Memory", *lines, "", "## Current input",
                      asked, "", "Note:"])


def build_cases(tokenizer, seed: int = SEED) -> list[dict]:
    bases = draw_bases(seed)
    rows = []
    for task in TASKS:
        for b in bases[task]:
            split = "dev" if b["index"] % 2 == 0 else "held"
            for variant in VARIANTS:
                core = core_variant(task, b, variant)
                roles = [("gold", core["gold"])] + [("competitor", c) for c in core["competitors"]]
                if core["intermediate"]:
                    roles.append(("intermediate", core["intermediate"]))
                roles += [("entity", e) for e in core["entities"]]
                pinned = M.pin_tokens(tokenizer, roles)
                for hard, n in HARDNESS.items():
                    facts, core_idx = place_distractors(core["facts"], b["distractors"], b["gap_u"], n)
                    for fmt in FORMATS:
                        prompt = render_prompt(fmt, facts, core["question"])
                        case_id = f"{task}{b['index']:02d}_{variant}_{fmt}_{hard}"
                        rows.append({"case_id": case_id, "task": task, "base_index": b["index"], "split": split,
                                     "variant": variant, "format": fmt, "hardness": hard, "facts": facts,
                                     "core_fact_indices": core_idx, "question": core["question"], "gold": core["gold"],
                                     "competitors": core["competitors"], "intermediate": core["intermediate"],
                                     "entities": core["entities"], "distractors": b["distractors"][:n],
                                     "v3_before": b["v3_before"] if (task == "DE" and variant == "v3") else None,
                                     "prompt": prompt, "prompt_sha256": M.sha256_text(prompt), "pinned": pinned})
    return rows


def check_cases(rows: list[dict], tokenizer) -> dict:
    """The review checks of §3: counts, distinct golds per (task, variant) across base cases, every competitor and
    the intermediate distinct from the gold, distinct pinned first content tokens (within a case and across all pool
    items), no distractor touching a core entity, dev / held parity. Returns {check: problems list}."""
    problems = {k: [] for k in ("counts", "duplicate_gold", "competitor_is_gold", "pinned", "pool_tokens",
                                "distractor_overlap", "parity", "prompt_duplicate")}
    if len(rows) != 864:
        problems["counts"].append(f"{len(rows)} prompts, want 864")
    golds = {}
    for r in rows:
        golds.setdefault((r["task"], r["variant"], r["format"], r["hardness"]), []).append(r["gold"])
        if r["gold"] in r["competitors"] or r["intermediate"] == r["gold"]:
            problems["competitor_is_gold"].append(r["case_id"])
        ids = [p["id"] for p in r["pinned"]]
        if len(ids) != len(set(ids)):
            problems["pinned"].append(r["case_id"])
        core_names = set(r["entities"])
        for d in r["distractors"]:
            if d["name"] in core_names or d["value"] in core_names:
                problems["distractor_overlap"].append(f"{r['case_id']}: {d}")
    for key, gs in golds.items():
        if len(gs) != len(set(gs)):
            problems["duplicate_gold"].append(f"{key}: {Counter(gs).most_common(1)}")
    owner = {}
    for pool in (NAMES, CITIES, JOBS, ("unknown",)):
        for item in pool:
            for form in M.FORMS:
                tid, _ = M.first_content_token(tokenizer, item, form)
                if tid in owner and owner[tid] != item:
                    problems["pool_tokens"].append(f"{item!r} and {owner[tid]!r} share {tokenizer.decode([tid])!r}")
                owner[tid] = item
    shas = Counter(r["prompt_sha256"] for r in rows)
    problems["prompt_duplicate"] = [s for s, n in shas.items() if n > 1]
    for task in TASKS:
        for cell in ("variant", "format", "hardness"):
            c = Counter((r[cell], r["split"]) for r in rows if r["task"] == task)
            for value in {v for v, _ in c}:
                if c[(value, "dev")] != c[(value, "held")]:
                    problems["parity"].append(f"{task} {cell} {value}: dev {c[(value, 'dev')]} held {c[(value, 'held')]}")
    return problems


def counts(rows: list[dict]) -> dict:
    def by(*keys, subset=None):
        c = Counter(tuple(str(r[x]) for x in keys) for r in (subset if subset is not None else rows))
        return {"|".join(k): n for k, n in sorted(c.items())}
    gold_use = {}
    for split in ("dev", "held"):
        gold_use[split] = dict(sorted(Counter(r["gold"] for r in rows if r["split"] == split).items()))
    n_tokens = sorted(len(r["prompt"]) for r in rows)
    return {"n_prompts": len(rows), "by_task": by("task"), "by_split": by("split"), "by_task_split": by("task", "split"),
            "by_task_variant_split": by("task", "variant", "split"), "by_format_split": by("format", "split"),
            "by_hardness_split": by("hardness", "split"), "gold_use_by_split": gold_use,
            "pinned_per_prompt": dict(sorted(Counter(len(r["pinned"]) for r in rows).items())),
            "prompt_chars_min_median_max": [n_tokens[0], n_tokens[len(n_tokens) // 2], n_tokens[-1]],
            "de_v3_placement": by("split", "v3_before", subset=[r for r in rows if r["v3_before"] is not None])}


def write_cases(rows: list[dict], out: Path, problems: dict, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"seed": seed, "cases_sha256": sha, "pass": not any(problems.values()), "problems": problems,
               "counts": counts(rows),
               "design": {"names": NAMES, "cities": CITIES, "jobs": JOBS, "months_unused": MONTHS,
                          "distractor_relations": DISTRACTOR_RELATIONS, "f2_instruction": F23_INSTRUCTION,
                          "tokenizer": TOKENIZER_ID, "dev": "even base index", "held": "odd base index"}}
    (out / "cases_counts.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------------------------- stage 3 rows

def build_rows44(eligible_path: Path = Path("runs/exp44_swapgate/rev8/eligible_qwen3b.jsonl")) -> list[dict]:
    """The 70 rev 8 rows x (clean, swap): prompt text re-derived with the gate's own functions and checked against the
    eligible file's shas, the evidence line's char span, the question span, gold / substitute and their read ids, the
    §23 label and the cached §23 output (clean only). Reads the §23 traces (no model, no tokenizer)."""
    import exp44_swap as S
    import exp44_swap_gate as G

    records = {r.question_id: r for r in S.load_population("qwen3b")}
    order, eligible_sha = S.load_eligible("qwen3b", gate_rev=8)
    out = []
    for e in order:
        rec = records[e["question_id"]]
        why, place = G.place_gold(rec, G.Checker())
        swap, old, new = S.rebuild_swap(rec, e)
        for kind, prompt, line in (("clean", rec.prompt, old), ("swap", swap, new)):
            start = place["start"]
            if prompt[start:start + len(line)] != line:
                raise SystemExit(f"rows44: {rec.question_id} {kind}: evidence line not at the gate's offset")
            qs, qe = G.question_span(prompt)
            out.append({"key": f"{rec.question_id}_{kind}", "question_id": rec.question_id, "fold": rec.fold,
                        "kind": kind, "prompt": prompt, "prompt_sha256": M.sha256_text(prompt),
                        "evidence_span": [start, start + len(line)], "question_span": [qs, qe],
                        "gold": rec.gold, "substitute": e["substitute"],
                        "gold_read_id": e["gold_read_id"], "substitute_read_id": e["substitute_read_id"],
                        "gold_read_text": e["gold_read_text"], "substitute_read_text": e["substitute_read_text"],
                        "label23": S.label23(rec), "question_type": rec.question_type,
                        "output23": rec.output if kind == "clean" else None, "eligible_sha256": eligible_sha,
                        "expected_sha256": e["clean_prompt_sha256"] if kind == "clean" else e["swap_prompt_sha256"]})
    bad = [r["key"] for r in out if r["prompt_sha256"] != r["expected_sha256"]]
    if bad:
        raise SystemExit(f"rows44: prompt sha differs from the gate's for {bad[:3]}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--rows44", action="store_true", help="write rows44.jsonl (stage 3) instead of the case list")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.rows44:
        rows = build_rows44()
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows44.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        info = {"n_rows": len(rows), "n_questions": len({r["question_id"] for r in rows}),
                "by_kind": dict(Counter(r["kind"] for r in rows)), "by_label23": dict(Counter(r["label23"] for r in rows if r["kind"] == "clean")),
                "sha256": hashlib.sha256((out / "rows44.jsonl").read_bytes()).hexdigest()}
        (out / "rows44_counts.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))
        return 0 if info["n_rows"] == 140 else 3
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    rows = build_cases(tok, args.seed)
    problems = check_cases(rows, tok)
    summary = write_cases(rows, out, problems, args.seed)
    print(json.dumps({"pass": summary["pass"], "sha256": summary["cases_sha256"], "n": len(rows),
                      "problems": {k: len(v) for k, v in problems.items()}}))
    return 0 if summary["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
