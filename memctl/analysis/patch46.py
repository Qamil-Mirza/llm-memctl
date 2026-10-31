"""§46 measurement B: aligned P / P' / P'' triples, the first-city readout, transfer and the §4 criterion.

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §4 with rev 1 rulings 1-2 (6ff90c8). Pure functions
(no model); the passes are in scripts/exp46_patch.py.

Triples (rev 1 ruling 1): for every RC v0 / v3 case of runs/exp45_maps/cases.jsonl, two facts "{B'} works in {C1'}"
and "{B''} works in {C1}" are inserted together at one seeded slot of the fact list (seed 46, the same slot and the same
B', B'', C1' in the three prompts; all three unused anywhere in the base prompt). Sentence 1 names B (P, gold C1),
B' (P', gold C1') or B'' (P'', gold C1, the k2 same-answer donor). Alignment per triple: equal token length, the
supervisor's token at the same index, exactly that one token differing.

Readout (rev 4, 34d1b4d): the conclusion. F2: the first pool city or "unknown" on the "Answer:" line (§45 stop rule,
max_new 160; no Answer line -> other). F1: the last pool city or "unknown" in the generation, run to end of turn or
max_new 96. Outcome by that item: "C1" / "C1p" / "other" (another city, or none) / "abstain" ("unknown"). The same
rule decides the solved filter, k3, Transfer and S2. Beside it: which of the triple's supervisor names occur in the
text before the conclusion item (the name-copy route). The rev 1 first-city rule and the §45 rule are kept for the
validate report only.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from memctl.analysis.maps import sha256_text, wilson

METHOD_VERSION = "exp46_patch_v0"
SEED = 46
ROLES = ("P", "Pp", "Ppp")
MAX_NEW_FIRST_CITY = {"F1": 48, "F2": 64}  # rev 1 first-city rule: reported by validate only
MAX_NEW_CONCLUSION = {"F1": 96, "F2": 160}  # rev 4: F1 to end of turn (max 96), F2 to the end of the Answer line
MAX_NEW_45 = MAX_NEW_CONCLUSION
TRANSFER_MIN, TRANSFER_LOWER_MIN, CONTROL_MAX = 0.70, 0.55, 0.10
MID_STACK_L = 10
HELD_PAIRS_MIN = 30
K3_MIN = 0.95
S2_LAYERS = tuple(range(0, 36, 3))
ADDITION = "§46 addition"


class TripleAlignmentError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------- triples

def _present(word: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


H2_POLICIES = ("drop", "reuse")


def triple_choice(case: dict, names: Sequence[str], cities: Sequence[str], seed: int = SEED,
                  h2_policy: str = "reuse") -> dict:
    """B', B'', C1' unused in the base prompt and the insertion slot (0..n facts), seeded per base case id.
    With one free name (every H2 row): policy "drop" raises; "reuse" keeps B' = the free name and takes B'' (the k2
    donor's supervisor only) from the distractor names (rev 2 ruling 1: no "works in" fact; "works as" only first,
    else "lives in"), flagged b2_reuses_distractor and, with a "lives in" fact, b2_has_rival_city."""
    rng = random.Random(f"{seed}-{case['case_id']}")
    free_names = [n for n in names if not _present(n, case["prompt"])]
    free_cities = [c for c in cities if not _present(c, case["prompt"])]
    if not free_cities or not free_names:
        raise TripleAlignmentError(f"{case['case_id']}: {len(free_names)} free names, {len(free_cities)} free cities")
    reuse = False
    rival = False
    if len(free_names) >= 2:
        bp, bpp = rng.sample(free_names, 2)
    elif h2_policy == "reuse":
        bp, bpp = free_names[0], b2_from_distractors(case, rng)
        reuse = True
    else:
        raise TripleAlignmentError(f"{case['case_id']}: {len(free_names)} free name, policy {h2_policy!r}")
    c1p = rng.choice(free_cities)
    slot = rng.randint(0, len(case["facts"]))
    if reuse:
        rival = any(d["name"] == bpp and d["relation"] == "lives" for d in case["distractors"])
    return {"Bp": bp, "Bpp": bpp, "C1p": c1p, "slot": slot, "b2_reuses_distractor": reuse, "b2_has_rival_city": rival}


def b2_from_distractors(case: dict, rng: random.Random) -> str:
    """Rev 2 ruling 1: B'' from the distractor names (never a core entity, never one with a "works in" fact),
    preferring names whose facts carry no city ("works as"), else one with a "lives in" fact."""
    core = set(case["entities"])
    rel: dict[str, set[str]] = {}
    for d in case["distractors"]:
        rel.setdefault(d["name"], set()).add(d["relation"])
    ok = {n: r for n, r in rel.items() if n not in core and "works_in" not in r}
    first = sorted(n for n, r in ok.items() if r == {"job"})
    second = sorted(n for n, r in ok.items() if "lives" in r)
    for pool in (first, second):
        if pool:
            return rng.choice(pool)
    raise TripleAlignmentError(f"{case['case_id']}: no distractor name without a workplace for B''")


def triple_prompts(case: dict, choice: dict, render) -> dict[str, dict]:
    """The three prompts of one triple. `render(fmt, facts, question)` = exp45_cases.render_prompt."""
    A, B, C1 = case["entities"]
    sup_sentence = f"{A}'s supervisor is {B}."
    if sup_sentence not in case["facts"]:
        raise TripleAlignmentError(f"{case['case_id']}: supervisor sentence not in the fact list")
    extra = [f"{choice['Bp']} works in {choice['C1p']}.", f"{choice['Bpp']} works in {C1}."]
    base_facts = list(case["facts"])
    facts = base_facts[:choice["slot"]] + extra + base_facts[choice["slot"]:]
    sup_at = facts.index(sup_sentence)
    out = {}
    for role, sup, gold, comp in (("P", B, C1, choice["C1p"]), ("Pp", choice["Bp"], choice["C1p"], C1),
                                  ("Ppp", choice["Bpp"], C1, choice["C1p"])):
        f = list(facts)
        f[sup_at] = f"{A}'s supervisor is {sup}."
        prompt = render(case["format"], f, case["question"])
        out[role] = {"facts": f, "prompt": prompt, "prompt_sha256": sha256_text(prompt), "supervisor": sup,
                     "gold": gold, "competitors": [comp], "entities": [A, sup, gold]}
    return out


def build_triple_rows(case: dict, names, cities, render, seed: int = SEED, h2_policy: str = "reuse") -> list[dict]:
    choice = triple_choice(case, names, cities, seed, h2_policy)
    prompts = triple_prompts(case, choice, render)
    tid = f"{case['case_id']}_T"
    rows = []
    for role in ROLES:
        p = prompts[role]
        rows.append({"case_id": f"{tid}{role}", "triple_id": tid, "role": role, "source_case": case["case_id"],
                     "addition": ADDITION, "task": "RC", "base_index": case["base_index"], "split": case["split"],
                     "variant": case["variant"], "format": case["format"], "hardness": case["hardness"],
                     "question": case["question"], "intermediate": p["supervisor"], "C1": case["entities"][2],
                     "C1p": choice["C1p"], "Bp": choice["Bp"], "Bpp": choice["Bpp"], "B": case["entities"][1],
                     "slot": choice["slot"],
                     "b2_reuses_distractor": choice["b2_reuses_distractor"],
                     "b2_has_rival_city": choice["b2_has_rival_city"], "distractors": case["distractors"], **p})
    return rows


def check_triple_alignment(ids: dict[str, Sequence[int]], site: dict[str, int], where: str = "") -> int:
    """Equal length, P_B at the same index, exactly the one token at P_B differing (P vs P', P vs P''). Returns P_B."""
    n = {len(v) for v in ids.values()}
    if len(n) != 1:
        raise TripleAlignmentError(f"{where}: lengths differ {sorted(n)}")
    sites = set(site.values())
    if len(sites) != 1:
        raise TripleAlignmentError(f"{where}: P_B differs {site}")
    pb = sites.pop()
    for role in ("Pp", "Ppp"):
        diff = [i for i, (a, b) in enumerate(zip(ids["P"], ids[role])) if a != b]
        if diff != [pb]:
            raise TripleAlignmentError(f"{where}: P vs {role} differ at {diff[:5]}, want only [{pb}]")
    return pb


# ---------------------------------------------------------------------------------------------- readout

def _first(pattern_items: Sequence[str], text: str, flags=0) -> tuple[int, str] | None:
    best = None
    for item in pattern_items:
        m = re.search(r"\b" + re.escape(item) + r"\b", text, flags)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), item)
    return best


def first_city(text: str, cities: Sequence[str]) -> tuple[int, str] | None:
    """(char offset, item) of the first pool city or "unknown" in a generated text, else None."""
    c = _first(cities, text)
    u = _first(("unknown",), text, re.IGNORECASE)
    if u is not None:
        u = (u[0], "unknown")
    hits = [h for h in (c, u) if h is not None]
    return min(hits) if hits else None


def stop_now(text: str, cities: Sequence[str]) -> bool:
    return first_city(text, cities) is not None


def classify_first(text: str, c1: str, c1p: str, cities: Sequence[str]) -> str:
    hit = first_city(text, cities)
    if hit is None:
        return "other"
    item = hit[1]
    if item == "unknown":
        return "abstain"
    return "C1" if item == c1 else ("C1p" if item == c1p else "other")


def _items(text: str, cities: Sequence[str]) -> list[tuple[int, str]]:
    hits = []
    for item in list(cities) + ["unknown"]:
        flags = re.IGNORECASE if item == "unknown" else 0
        hits += [(m.start(), item) for m in re.finditer(r"\b" + re.escape(item) + r"\b", text, flags)]
    return sorted(hits)


def conclusion(text: str, fmt: str, cities: Sequence[str]) -> tuple[int, str] | None:
    """(char offset in `text`, item) of the conclusion: F2 the first item on the last "Answer:" line; F1 the last item
    in the generation. None if there is none (F2: no Answer line, or no item on it)."""
    if fmt == "F2":
        lower = text.lower()
        marker = lower.rfind("answer:")
        if marker < 0:
            return None
        start = marker + len("answer:")
        line = text[start:].split("\n")[0]
        hits = _items(line, cities)
        return (start + hits[0][0], hits[0][1]) if hits else None
    hits = _items(text, cities)
    return hits[-1] if hits else None


def classify_conclusion(text: str, fmt: str, c1: str, c1p: str, cities: Sequence[str]) -> str:
    hit = conclusion(text, fmt, cities)
    if hit is None:
        return "other"
    if hit[1] == "unknown":
        return "abstain"
    return "C1" if hit[1] == c1 else ("C1p" if hit[1] == c1p else "other")


def names_before_city(text: str, names: Sequence[str], cities: Sequence[str], fmt: str = "F1") -> list[str]:
    """Supervisor names in the text before the conclusion item (all of it when there is none)."""
    hit = conclusion(text, fmt, cities)
    head = text if hit is None else text[:hit[0]]
    return [n for n in names if _present(n, head)]


# ---------------------------------------------------------------------------------------------- statistics

def k1_vector(h_p: np.ndarray, h_pp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """§44 rev 6 difference-matched control: h_P + r, r a random direction of norm ||h_P' - h_P||."""
    r = rng.standard_normal(h_p.shape).astype(np.float32)
    r *= np.linalg.norm(h_pp - h_p) / max(np.linalg.norm(r), 1e-12)
    return (h_p + r).astype(np.float32)


def k1_seed(triple_id: str, site: str, layer: int, seed: int = SEED) -> int:
    return int(hashlib.sha256(f"{seed}-{triple_id}-{site}-{layer}".encode()).hexdigest()[:8], 16)


def transfer_profile(outcomes: dict[int, list[str]], target: str = "C1p") -> dict[int, dict]:
    """Per layer: share of pairs whose outcome is `target`, Wilson interval."""
    out = {}
    for layer, outs in sorted(outcomes.items()):
        k = sum(o == target for o in outs)
        p, lo, hi = wilson(k, len(outs))
        out[layer] = {"n": len(outs), "k": k, "share": p, "lower": lo, "upper": hi}
    return out


def carried_through(patch: dict[int, dict], k1: dict[int, dict], k2: dict[int, dict]) -> int | None:
    """§4 criterion: the largest L* such that every layer l <= L* has Transfer >= 0.70, lower >= 0.55, k1 <= 0.10 and
    k2 <= 0.10 (k2 share of C1' with the same-answer donor). None if layer 0 already fails."""
    best = None
    for layer in sorted(patch):
        t, a, b = patch[layer], k1.get(layer), k2.get(layer)
        ok = (t["share"] is not None and t["share"] >= TRANSFER_MIN and t["lower"] >= TRANSFER_LOWER_MIN
              and a is not None and a["share"] is not None and a["share"] <= CONTROL_MAX
              and b is not None and b["share"] is not None and b["share"] <= CONTROL_MAX)
        if not ok:
            break
        best = layer
    return best


def verdict(first_decodable_p_last: int | None, l_star: int | None, held_pairs: int,
            layer_max: int = 29) -> dict:
    """§5 decision rule, computed. Under HELD_PAIRS_MIN held pairs the row is under-powered (rev 1 ruling 1)."""
    a = first_decodable_p_last is not None and first_decodable_p_last <= layer_max
    b = l_star is not None and l_star >= MID_STACK_L
    label = {(True, True): "instrument blind spot", (False, False): "route late, lenses adequate",
             (True, False): "represented, not used", (False, True): "route late at P_last, instrument blind at P_B"}[(a, b)]
    return {"probe_decodable_le_29": a, "l_star_ge_10": b, "verdict": label,
            "under_powered": held_pairs < HELD_PAIRS_MIN, "held_pairs": held_pairs,
            "first_decodable_p_last": first_decodable_p_last, "l_star": l_star}
