#!/usr/bin/env python
"""§50: the case list for "why false abstention" (newly generated from the §49 generator at seed 50). Data only; the
Qwen2.5 tokenizer is read (offline) for the pool-token and single-token checks; no model is loaded.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp50_cases.py [--bases-per-split 48|24]   # -> runs/exp50/cases50.jsonl

Spec: research/proposals/2026-11-02-exp50-why-false-abstention.md rev 1 §3 (+ rev 1a); D-70. The §49 base draw, gold
balance, nuisance fact, distractor scheme and F1 / F2 renderers are reused unchanged (scripts/exp49_cases.py,
scripts/exp45_cases.py), with seed 50.

Sets per split (48 bases per task; 24 under the rev 1a scale-down, the first two gold blocks, still balanced):
- ANS rep a (H0 / H1 / H2): the §49 ANS-F2 prompt.
- ANS rep b (H0 / H1 / H2): rep a with every timestamp, the question's included, shifted by +7 days (weekday labels
  unchanged; order and relative dates kept). Nothing else differs.
- UNA-rel (H0 / H1 / H2): the §49 relation-replaced twin at every hardness. Fixed choice of this build: the
  replacement lines are about the base's one pool name that no distractor uses (N), with a city and (TU) a job no
  other line of that base uses, so they never repeat a distractor line at H2 (§49's distractors 4-5 would).
- UNA-swap (H1): rep a's H1 prompt, byte-identical memory, with the question about D, the first H1 distractor person
  (prompt order) whose only line in the prompt is a "works as" line: DE "Where does {D} live?", RC "In which city does
  {D}'s supervisor work?", TU "Where does {D} live now?". No such D -> no row (counted).
- ANS-F1 (H1): the first 24 bases per task x split (the first two gold blocks); 12 under the scale-down.
Splits: even base index dev (dev-fit: index mod 4 == 0, dev-check: index mod 4 == 2), odd held. Exact prompt duplicates
of any §49 row (runs/exp49/cases49.jsonl) are dropped before the split; core-tuple overlap is reported, not removed.
Every row also carries its S prompt (memory block + the frozen yes/no question), its polarity prompt and its K_v
template (§4.1-4.2).
"""

from __future__ import annotations

import argparse
import datetime as dt
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
import exp49_cases as G  # noqa: E402
from memctl.agents.llm import REASONING_INSTRUCTIONS  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

OUT = Path("runs/exp50")
SEED = 50
SHIFT_DAYS = 7
CASES49 = Path("runs/exp49/cases49.jsonl")
TASKS = G.TASKS
SWAP_Q = {"DE": "Where does {D} live?", "RC": "In which city does {D}'s supervisor work?",
          "TU": "Where does {D} live now?"}
S_Q = {"DE": "Does the memory give enough information to tell where {X} lives? Reply yes or no.",
       "RC": "Does the memory give enough information to tell in which city {X}'s supervisor works? Reply yes or no.",
       "TU": "Does the memory give enough information to tell where {X} lives now? Reply yes or no."}
POL_Q = {"DE": "Is information needed to tell where {X} lives missing from the memory? Reply yes or no.",
         "RC": "Is information needed to tell in which city {X}'s supervisor works missing from the memory? Reply yes or no.",
         "TU": "Is information needed to tell where {X} lives now missing from the memory? Reply yes or no."}
KV_Q = {"DE": "How sure are you that {X} lives in {city}? Reply with one digit from 0 (not at all sure) to 9 (certain).",
        "RC": "How sure are you that {X}'s supervisor works in {city}? Reply with one digit from 0 (not at all sure) to 9 (certain).",
        "TU": "How sure are you that {X} lives now in {city}? Reply with one digit from 0 (not at all sure) to 9 (certain)."}
DATE_RE = re.compile(r"\d{4}/\d{2}/\d{2} \([A-Z][a-z]{2}\)")


def f2_date(i: int, shift: int = 0) -> str:
    d = C.F2_START + dt.timedelta(days=i + shift)
    return f"{d:%Y/%m/%d} ({d:%a})"


def render_f2(facts: list[str], question: str, shift: int = 0) -> str:
    """exp45_cases.render_prompt's F2 branch with every date shifted by `shift` days (shift 0 reproduces it)."""
    lines = [f"user, {f2_date(i, shift)} 10:00: {fact}" for i, fact in enumerate(facts)]
    asked = f"Question: (asked on {f2_date(len(facts), shift)} 12:00) {question}"
    return "\n".join([f"{C.F23_INSTRUCTION}\n{REASONING_INSTRUCTIONS}", "", "## Memory", *lines, "", "## Current input",
                      asked, "", "Note:"])


def memory_block(facts: list[str], shift: int = 0) -> str:
    return "\n".join(["## Memory", *[f"user, {f2_date(i, shift)} 10:00: {f}" for i, f in enumerate(facts)]])


def side_prompt(facts: list[str], question: str, shift: int = 0) -> str:
    """S / polarity / K_v prompts: the row's own memory block (its dates), a blank line, then the question."""
    return memory_block(facts, shift) + "\n\n" + question


def replacement_facts(b: dict, seed: int = SEED) -> list[str]:
    """UNA-rel replacement lines (see module doc): about N, the pool name no distractor uses."""
    used_names = {b["A"], b["B"]} | {d["name"] for d in b["distractors"]}
    free_names = [n for n in C.NAMES if n not in used_names]
    used_cities = {b["gold"], b["cf"]} | ({b["old"]} if b["old"] else set()) | \
        {d["value"] for d in b["distractors"] if d["relation"] != "job"}
    free_cities = [c for c in C.CITIES if c not in used_cities]
    used_jobs = {b["J"]} | {d["value"] for d in b["distractors"] if d["relation"] == "job"}
    free_jobs = [j for j in C.JOBS if j not in used_jobs]
    rng = random.Random(f"{seed}-{b['task']}-{b['index']}-replacement")
    n, city, job = rng.choice(free_names), rng.choice(free_cities), rng.choice(free_jobs)
    k = 2 if b["task"] == "TU" else 1
    return [f"{n} lives in {city}.", f"{n} works as {C.article(job)} {job}."][:k]


def core50(task: str, b: dict, kind: str) -> dict:
    """§49's core (ANS) or the UNA-rel core with this build's replacement lines."""
    if kind == "ANS":
        return G.core(task, b, "ANS")
    c = G.core(task, b, "UNA")
    rep = replacement_facts(b)
    old = c["replacement_lines"]
    facts = [rep[old.index(f)] if f in old else f for f in c["facts"]]
    return {**c, "facts": facts, "replacement_lines": rep}


def swap_entity(facts: list[str], distractors: list[dict]) -> str | None:
    """The first H1 distractor name (prompt order) whose only prompt line is a 'works as' line."""
    for f in facts:
        for d in distractors:
            n = d["name"]
            if not re.match(re.escape(n) + r"\b", f):
                continue
            lines = [x for x in facts if re.search(r"\b" + re.escape(n) + r"\b", x)]
            if len(lines) == 1 and " works as " in lines[0]:
                return n
    return None


def part_of(index: int) -> str:
    return "held" if index % 2 else ("dev-fit" if index % 4 == 0 else "dev-check")


def make_row(b: dict, kind: str, rep: str, fmt: str, hard: str) -> dict | None:
    task = b["task"]
    c = core50(task, b, "ANS" if kind in ("ANS", "UNA-swap") else "UNA")
    facts, core_idx = C.place_distractors(c["facts"], b["distractors"], b["gap_u"], C.HARDNESS[hard])
    question, query, gold = c["question"], (b["B"] if task == "RC" else b["A"]), c["gold"]
    swap_d = None
    if kind == "UNA-swap":
        swap_d = swap_entity(facts, b["distractors"][:C.HARDNESS[hard]])
        if swap_d is None:
            return None
        question = SWAP_Q[task].format(D=swap_d)
        gold, query = None, swap_d
    if kind == "UNA-rel":
        gold = None
    shift = SHIFT_DAYS if rep == "b" else 0
    prompt = render_f2(facts, question, shift) if fmt == "F2" else C.render_prompt("F1", facts, question)
    ask = swap_d if kind == "UNA-swap" else b["A"]           # the S question names A (RC: A's supervisor) or D
    base_key = f"{task}{b['index']:02d}"
    tag = {"ANS": f"ANS{rep}", "UNA-rel": "UNArel", "UNA-swap": "UNAswap"}[kind]
    return {"case_id": f"{base_key}_{tag}_{fmt}_{hard}", "base_key": base_key, "task": task, "base_index": b["index"],
            "split": "held" if b["index"] % 2 else "dev", "part": part_of(b["index"]), "set": kind, "rep": rep,
            "format": fmt, "hardness": hard, "facts": facts, "core_fact_indices": core_idx, "question": question,
            "gold": gold, "base_gold": b["gold"], "competitors": c["competitors"] if kind == "ANS" else [],
            "old_city": b["old"], "A": b["A"], "B": b["B"], "query_entity": query, "swap_entity": swap_d,
            "nuisance": G.nuisance(task, b), "distractors": b["distractors"][:C.HARDNESS[hard]],
            "una_replacements": c["replacement_lines"] if kind == "UNA-rel" else [],
            "prompt": prompt, "prompt_sha256": M.sha256_text(prompt),
            "s_prompt": side_prompt(facts, S_Q[task].format(X=ask), shift),
            "pol_prompt": side_prompt(facts, POL_Q[task].format(X=ask), shift),
            "kv_template": side_prompt(facts, KV_Q[task].format(X=ask, city="{city}"), shift),
            "cities_in_prompt": G.cities_in(prompt)}


def build_cases(per_split: int = 48, seed: int = SEED) -> tuple[list[dict], dict]:
    rows, no_swap = [], []
    for task in TASKS:
        lists = {s: G.label_lists(task, s, seed) for s in ("dev", "held")}
        used: set = set()
        for index in range(2 * 48):
            split, k = G.split_of(index), index // 2
            gold, cf = lists[split][0][k], lists[split][1][k]
            b = G.draw_base(task, index, gold, cf, seed, used)      # all 96 drawn (same streams at 48 and 24)
            if k >= per_split:
                continue
            for hard in ("H0", "H1", "H2"):
                for rep in ("a", "b"):
                    rows.append(make_row(b, "ANS", rep, "F2", hard))
                rows.append(make_row(b, "UNA-rel", "a", "F2", hard))
            sw = make_row(b, "UNA-swap", "a", "F2", "H1")
            if sw is None:
                no_swap.append(f"{task}{index:02d}")
            else:
                rows.append(sw)
            if k < per_split // 2:
                rows.append(make_row(b, "ANS", "a", "F1", "H1"))
    info = {"no_swap_entity": no_swap}
    return rows, info


def drop_overlap(rows: list[dict], cases49: Path = CASES49) -> tuple[list[dict], dict]:
    """Exact prompt duplicates of §49 rows dropped; core-tuple overlap (task, A, B, gold, old) reported."""
    if not cases49.exists():
        return rows, {"cases49": "missing"}
    old = [json.loads(l) for l in cases49.read_text().splitlines() if l.strip()]
    old_sha = {r["prompt_sha256"] for r in old}
    key = lambda r: (r["task"], r["A"], r["B"] if r["task"] == "RC" else None, r["base_gold"], r["old_city"])
    old_keys = {key(r) for r in old}
    kept = [r for r in rows if r["prompt_sha256"] not in old_sha]
    dropped = [r["case_id"] for r in rows if r["prompt_sha256"] in old_sha]
    bases = {r["base_key"]: key(r) for r in rows}
    return kept, {"cases49_sha256": hashlib.sha256(cases49.read_bytes()).hexdigest(), "dropped_exact": dropped,
                  "core_tuple_overlap_bases": sum(k in old_keys for k in bases.values()), "n_bases": len(bases)}


def check_cases(rows: list[dict], tokenizer=None) -> dict:
    pr = {k: [] for k in ("rep_b_diff", "swap_memory", "nuisance", "una_city", "balance", "split", "single_token",
                          "duplicate")}
    by = {r["case_id"]: r for r in rows}
    for r in rows:
        if r["set"] == "ANS" and r["rep"] == "b":
            a = by.get(r["case_id"].replace("_ANSb_", "_ANSa_"))
            if a is not None and DATE_RE.sub("D", a["prompt"]) != DATE_RE.sub("D", r["prompt"]):
                pr["rep_b_diff"].append(r["case_id"])
            if a is not None and [m.group()[-4:] for m in DATE_RE.finditer(a["prompt"])] != \
                    [m.group()[-4:] for m in DATE_RE.finditer(r["prompt"])]:
                pr["rep_b_diff"].append(f"{r['case_id']}: weekday changed")
        if r["set"] == "UNA-swap":
            a = by.get(r["case_id"].replace("_UNAswap_", "_ANSa_"))
            if a is not None and a["prompt"].split("## Current input")[0] != r["prompt"].split("## Current input")[0]:
                pr["swap_memory"].append(r["case_id"])
            d = r["swap_entity"]
            if [f for f in r["facts"] if d in f and " works as " not in f]:
                pr["swap_memory"].append(f"{r['case_id']}: D has a non-job line")
        if r["set"] in ("UNA-rel", "UNA-swap"):
            q = r["query_entity"]
            for f in r["facts"]:
                if r["set"] == "UNA-rel" and re.search(r"\b" + re.escape(q) + r"\b", f) and G.cities_in(f):
                    pr["una_city"].append(f"{r['case_id']}: {f}")
        if r["nuisance"] not in r["facts"]:
            pr["nuisance"].append(r["case_id"])
        if r["split"] != ("held" if r["base_index"] % 2 else "dev"):
            pr["split"].append(r["case_id"])
    for split in ("dev", "held"):
        for task in TASKS:
            g = Counter(r["gold"] for r in rows if (r["split"], r["task"], r["set"], r["rep"], r["format"],
                                                    r["hardness"]) == (split, task, "ANS", "a", "F2", "H1"))
            if g and len(set(g.values())) != 1:
                pr["balance"].append(f"{split} {task}: {dict(g)}")
    shas = Counter(r["prompt_sha256"] for r in rows)
    pr["duplicate"] = [s for s, n in shas.items() if n > 1]
    if tokenizer is not None:
        for item in list(C.CITIES) + ["unknown"]:
            ids = tokenizer(" " + item, add_special_tokens=False).input_ids
            if len(ids) != 1:
                pr["single_token"].append(item)
    return pr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases-per-split", type=int, choices=(48, 24), default=48)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--no-tokenizer", action="store_true")
    a = p.parse_args(argv)
    tok = None
    if not a.no_tokenizer:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(G.TOKENIZER_ID)
    rows, info = build_cases(a.bases_per_split)
    rows, overlap = drop_overlap(rows)
    pr = check_cases(rows, tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases50.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = Counter(f"{r['split']}|{r['set']}{r['rep'] if r['set'] == 'ANS' else ''}|{r['format']}" for r in rows)
    summary = {"seed": SEED, "bases_per_split": a.bases_per_split, "cases50_sha256": sha,
               "pass": not any(pr.values()), "problems": pr, "overlap_with_49": overlap, **info,
               "counts": dict(sorted(counts.items())),
               "by_part": dict(sorted(Counter(r["part"] for r in rows).items()))}
    (out / "cases50_counts.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"pass": summary["pass"], "sha256": sha, "n": len(rows), "counts": summary["counts"],
                      "dropped_exact": len(overlap.get("dropped_exact", [])),
                      "core_overlap_bases": overlap.get("core_tuple_overlap_bases"), "no_swap": len(info["no_swap_entity"]),
                      "problems": {k: len(v) for k, v in pr.items()}}))
    return 0 if summary["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
