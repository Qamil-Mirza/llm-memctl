#!/usr/bin/env python
"""§44 rev 1 §A: the swap-subset feasibility gate, from the data alone (tokenizers only, no model forward pass).

    .venv/bin/python scripts/exp44_swap_gate.py            # rev 2 rules, writes runs/exp44_swapgate/
    .venv/bin/python scripts/exp44_swap_gate.py --rev 1 --out runs/exp44_swapgate/rev1   # the 63882c1 gate

Rev 2 (the Advisor's rulings 2 and 3 of the proposal's "Rev 2" record, 5146f5b) changes two things and nothing else:
- substitute assignment: row i (its index in the family's (fold, question_id) order, 0..329) shuffles its own
  candidate list (the other 329 rows in that order) with random.Random(44 + i) and walks it; a substitute string
  (D1-normalised) is used at most CAP = 3 times across the family's eligible rows, rows processed in (fold, id) order;
  a candidate at its cap is skipped. A use is counted only once the row is eligible (alignment passed).
- surface types: "number" splits into number:currency (leading $/£/€ or a currency word), number:percent (% or
  "percent"), number:duration (a time unit word), number:bare (everything else the number regex takes, including a
  number with a non-time, non-currency unit such as "10 pounds" or "38 subjects"); the substitute must share the
  subtype. "other" substitutes must come from a row of the same LongMemEval question type, with the same
  first-character capitalisation (surface form, str.isupper of the first character) and |token length diff| <= 1.
  Every other rule (including the +-2 length tolerance for the other types) is the rev 1 rule below.

Rev 3 (7acce03, the default; --rev 2 reproduces 88b2a19): "different first token" becomes "different first token
after the shared prefix" (the longest common content-token prefix of G and S, after the leading whitespace tokens,
memctl.analysis.swap.shared_prefix); a candidate that is a token-prefix of the gold or vice versa (no divergent token)
counts as first_token_equal. The alignment check is memctl.analysis.swap.build_swap_inputs (clean / swap / cut with the
prefix forced: it ends every sequence, identical). Eligible rows record the prefix and the read (divergent) ids.

Rev 8 (83fe60a, D-47; --rev 8 --out runs/exp44_swapgate/rev8, Qwen 3B only; everything else rev 3): (1) gold
surface type "other" excluded as type_other_rev8 (applied after the placement exclusions, before the substitute walk);
(2) number:bare candidates form-matched: same form (digit / number word), digit-form = same digit count of the leading
number (a range "5-2" / "2-3 eggs" / "3:1" counts every digit of the string), same unit presence (a trailing word after
the number, "points", "-inch"; the unit need not match); (3) every kept type: leading determiner (the/a/an/my/our/his/
her/their/its, membership case-insensitive, followed by at least one more word) the same word as written in G and S
("The" != "the": a case-only difference would put the read on the determiner), or absent in both;
(4) the walk unchanged, so a row whose rev 3 substitute passes (2)-(3) keeps it: asserted against the recorded rev 3
eligible set (runs/exp44_swapgate/eligible_qwen3b.jsonl), every unchanged row's eligible record identical; the changed
rows are listed in gate.json `rev8_changes`.

Spec: research/proposals/2026-10-29-exp44-patching.md, "Rev 1 amendment" §A (D-45). Population: the §34 330 rows per
family (126 dev = folds 0-1, 204 confirmation = folds 2-4). Qwen 2.5 3B via scripts/exp44_patch.population (paired
with the 7B: same rule, same prompts, so the 7B shares these rows); Llama 3.2 3B via load_family_records paired with
Llama 3.1 8B under the §34 rule of runs/exp43_stage0/pop34.py (in view for both, same prompt sha, judge label on both).

Rules implemented here (every choice the spec leaves open is named in RULES and written to gate.json):

- verbatim: the gold's D1 token sequence (answer_selection._tokens: lowercase, every char outside [a-z0-9] a
  separator) occurs as a contiguous run of whole tokens in the text's D1 token sequence. Implemented as an ASCII,
  case-insensitive regex t1[^a-z0-9]+t2...tn bounded by (?<![a-z0-9]) / (?![a-z0-9]); cross-checked against the
  normalised form on every segment tested (a disagreement is counted and reported). Stricter than D1's substring
  branch (" 5" in "25" is not verbatim) and without D1's all-tokens branch.
- placement: the prompt is cut by jacobian.segment_prompt; a memory line's span is the line itself (its separator
  excluded). Evidence lines = record.evidence_lines. "In the question" = the question line (from "Question:" to
  "\\n\\nNote:"). "Elsewhere" = any verbatim match in the whole prompt not inside the one evidence line (instruction,
  non-evidence memory lines, the rest of the input block, or a match spanning a line boundary).
- swap: every verbatim match of the gold inside that one evidence line, widened to the gold's own non-alphanumeric
  prefix / suffix when the line has it adjacent ("$" of "$800", "%" of "20%"), is replaced by the substitute's surface
  form (its raw gold string, one trailing period dropped), so the evidence carries S exactly as P1 scores it
  (ids(" " + S)); every other byte of the prompt is unchanged (asserted).
- substitute pool: all 330 population rows of the same family, ordered by (fold, question_id), shuffled once with
  random.Random(44); for each row the pool is iterated from the start and the first candidate passing every rule is
  taken (the same order for every row). Candidate rules, in order: not the row itself and a different D1-normalised
  gold; same surface type; |len(ids(" " + S)) - len(ids(" " + G))| <= 2; S not verbatim anywhere in the clean prompt;
  G not verbatim anywhere in the swap prompt; the same teacher-forced leading whitespace tokens as G (so the swap run
  is forced exactly like the clean one and P1 predicts S's first content token); a different first content token.
- first token after "Answer:": the §34 tracked token, jlens_readout.tracked_token_ids (first non-whitespace token of
  ids(" " + text)). Also recorded the steering.continuation_first_id form, ids("Answer:" + " " + text) minus
  ids("Answer:"), and the prefix-stability of that split; disagreements are counted.
- alignment: patching.check_alignment on RowInputs(clean, swap) built by readout_for with the gold forced as §34
  (P1 last, same P1 token, identical question-to-P1 tail); the length check of the cut does not apply
  (n_evidence_lines passed as 0), and the swap must leave the tracked ids unchanged.
- exclusions, in this order: gold in zero evidence lines (or no alphanumeric token); gold in several evidence lines;
  gold in the question line; gold elsewhere in the prompt; no substitute (no candidate passes the rules before the
  first-token rule); first token equal (candidates pass every rule but the first-token one, none passes it);
  alignment failure (the chosen substitute's swap fails the check: the row is excluded, no retry).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.answer_selection import _tokens  # noqa: E402

SEED = 44
LEN_TOL = 2
CAP = 3                 # rev 2: uses per substitute string across a family's eligible rows
OTHER_LEN_TOL = 1       # rev 2: "other" substitutes within +-1 token
OUT = Path("runs/exp44_swapgate")
FOLD_SETS = {"dev": (0, 1), "confirmation": (2, 3, 4), "all": (0, 1, 2, 3, 4)}
LABELS = ("success", "refusal", "wrong")
EXCLUSIONS = ("gold_zero_evidence", "gold_several_evidence", "gold_in_question", "gold_elsewhere",
              "no_substitute", "first_token_equal", "alignment_failure")
EXCLUSIONS_REV8 = EXCLUSIONS[:4] + ("type_other_rev8",) + EXCLUSIONS[4:]
REV3_ELIGIBLE = Path("runs/exp44_swapgate/eligible_qwen3b.jsonl")     # the rev 3 / rev 4 recorded set (S rows)
DETERMINERS = ("the", "a", "an", "my", "our", "his", "her", "their", "its")


def exclusions(rev: int) -> tuple[str, ...]:
    return EXCLUSIONS_REV8 if rev >= 8 else EXCLUSIONS

# --- surface types (fixed regex, tested in this order: date, number, name, else other) ---------------------------
MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
WEEKDAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_RE = re.compile(
    rf"^(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS})\b.*|(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b.*"
    rf"|(?:{MONTHS})\s+\d{{4}}|(?:{WEEKDAYS})\b.*|\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?"
    rf"|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm|a\.m\.|p\.m\.))\.?$", re.I)
NUMBER_WORDS = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
                r"a hundred|a thousand|thousand|half|once|twice|a couple")
NUMBER_RE = re.compile(
    rf"^(?:about\s+|around\s+|approximately\s+)?(?:[$£€+-]\s?)?(?:\d[\d,]*(?:\.\d+)?|\.\d+|{NUMBER_WORDS})"
    r"(?:\s?%|\s?[-:/x]\s?\d[\d,]*(?:\.\d+)?)?(?:[\s-][A-Za-z]+(?:\s[A-Za-z]+)?)?\.?$", re.I)
NAME_WORD = r"(?:[A-ZÀ-Þ][\w'’.&-]*|of|the|and|at|on|in|de|la|del|du|von|van|for|&)"
NAME_RE = re.compile(rf"^(?:The\s+|the\s+)?[A-ZÀ-Þ][\w'’.&-]*(?:\s+{NAME_WORD}){{0,5}}\.?$")
NOT_NAMES = {"yes", "no", "none", "unknown"}
SURFACE_RULES = {"order": ["date", "number", "name", "other"], "date": DATE_RE.pattern, "number": NUMBER_RE.pattern,
                 "name": NAME_RE.pattern + "  (no digit; not yes/no/none/unknown)", "other": "everything else"}


CURRENCY_WORDS = r"dollars?|bucks|euros?|cents?|usd|eur|gbp|yen|rupees?"
TIME_UNITS = (r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|nights?|weeks?|months?|years?|yrs?|decades?|"
              r"centur(?:y|ies)")
NUMBER_SUBTYPE_RULES = {"order": ["currency", "percent", "duration", "bare"],
                        "currency": rf"leading [$£€] or a word in ({CURRENCY_WORDS})",
                        "percent": "'%' or the word percent", "duration": rf"a word in ({TIME_UNITS})",
                        "bare": "every other number-type gold (incl. a number with a non-time, non-currency unit)"}


def number_subtype(gold: str) -> str:
    g = gold.strip().lower()
    if re.match(r"^(?:about\s+|around\s+|approximately\s+)?[$£€]", g) or re.search(rf"\b(?:{CURRENCY_WORDS})\b", g):
        return "currency"
    if "%" in g or re.search(r"\bper\s?cent\b", g):
        return "percent"
    if re.search(rf"(?:\b|\d)(?:{TIME_UNITS})\b", g):
        return "duration"
    return "bare"


BARE_RE = re.compile(
    rf"^(?:about\s+|around\s+|approximately\s+)?(?:[$£€+-]\s?)?(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+|{NUMBER_WORDS})"
    r"(?P<range>\s?[-:/x]\s?\d[\d,]*(?:\.\d+)?)?(?P<unit>[\s-][A-Za-z]+(?:\s[A-Za-z]+)?)?\.?$", re.I)


def bare_form(gold: str) -> dict | None:
    """Rev 8: (form, digit count, unit presence) of a number:bare gold; None if it does not parse."""
    m = BARE_RE.match(gold.strip())
    if m is None:
        return None
    digit = bool(re.match(r"[\d.]", m.group("num")))
    is_range = m.group("range") is not None
    digits = (len(re.findall(r"\d", gold)) if is_range else len(re.findall(r"\d", m.group("num")))) if digit else None
    return {"form": "digit" if digit else "word", "digits": digits, "unit": m.group("unit") is not None,
            "range": is_range}


def leading_determiner(text: str) -> str | None:
    """Rev 8: the surface form's first word (as written) if it is a determiner (membership case-insensitive) followed
    by at least one more word. "The same word" is compared as written: "The" vs "the" would differ in their first
    token, so the read would land on the determiner's case, the very fault rule (3) removes (resolved by the Executor
    build; the case-insensitive comparison gives 2 such rows on Qwen 3B)."""
    words = surface(text).split()
    return words[0] if len(words) >= 2 and words[0].lower() in DETERMINERS else None


def rev8_reject(gold: str, cand: str, type_: str) -> str | None:
    """Rev 8 candidate rules (2)-(3): None if `cand` passes, else the rule it fails."""
    if type_ == "number:bare":
        g, c = bare_form(gold), bare_form(cand)
        if g is None or c is None:
            return "bare_unparsed"
        if g["form"] != c["form"]:
            return "bare_form"
        if g["form"] == "digit" and g["digits"] != c["digits"]:
            return "bare_digits"
        if g["unit"] != c["unit"]:
            return "bare_unit"
    if leading_determiner(gold) != leading_determiner(cand):
        return "determiner"
    return None


def type_key(gold: str, rev: int = 2) -> str:
    """The surface type a substitute must share: rev 1 = surface_type; rev 2 splits number into four subtypes."""
    t = surface_type(gold)
    return f"number:{number_subtype(gold)}" if rev >= 2 and t == "number" else t


def surface_type(gold: str) -> str:
    g = gold.strip()
    if DATE_RE.match(g):
        return "date"
    if NUMBER_RE.match(g):
        return "number"
    if NAME_RE.match(g) and not re.search(r"\d", g) and g.rstrip(".").lower() not in NOT_NAMES:
        return "name"
    return "other"


# --- verbatim matching ----------------------------------------------------------------------------------------------

def gold_pattern(gold: str) -> re.Pattern | None:
    toks = _tokens(gold)
    if not toks:
        return None
    body = r"[^a-z0-9]+".join(re.escape(t) for t in toks)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.I | re.A)


def normalised_contains(text: str, gold: str) -> bool:
    """The reference: the gold's D1 tokens as a contiguous whole-token run of the text's D1 tokens."""
    toks = _tokens(gold)
    return bool(toks) and f" {' '.join(toks)} " in f" {' '.join(_tokens(text))} "


def core(text: str) -> str:
    m = re.search(r"[A-Za-z0-9].*[A-Za-z0-9]|[A-Za-z0-9]", text, re.S)
    return m.group(0) if m else ""


# --- placement ------------------------------------------------------------------------------------------------------

def line_spans(record) -> list[tuple[str, int, int, str]]:
    """(region, start, end, line) for every memory line, the line's own span (no separator)."""
    from memctl.analysis.jacobian import segment_prompt

    out = []
    for (region, start, _), line in zip(segment_prompt(record.prompt, record.memory_lines, record.evidence_lines)[1:-1],
                                         record.memory_lines):
        assert record.prompt.startswith(line, start)
        out.append((region, start, start + len(line), line))
    return out


def question_span(prompt: str) -> tuple[int, int]:
    from memctl.analysis.jlens_readout import NOTE_SUFFIX, QUESTION_MARK

    q = prompt.rindex(QUESTION_MARK)
    end = prompt.index(NOTE_SUFFIX, q) if NOTE_SUFFIX in prompt[q:] else len(prompt)
    return q, end


class Checker:
    """Counts disagreements between the regex and the normalised reference."""

    def __init__(self):
        self.disagreements = 0
        self.checks = 0

    def occurs(self, text: str, pat: re.Pattern, gold: str) -> bool:
        hit = bool(pat.search(text))
        self.checks += 1
        if hit != normalised_contains(text, gold):
            self.disagreements += 1
        return hit


def place_gold(record, chk: Checker) -> tuple[str | None, dict]:
    """(exclusion or None, placement) for the gold of one row."""
    pat = gold_pattern(record.gold)
    if pat is None:
        return "gold_zero_evidence", {}
    spans = line_spans(record)
    evidence = set(record.evidence_lines)
    ev_hits = [(i, s, e) for i, (region, s, e, line) in enumerate(spans)
               if line in evidence and chk.occurs(line, pat, record.gold)]
    if not ev_hits:
        return "gold_zero_evidence", {}
    if len(ev_hits) > 1:
        return "gold_several_evidence", {"n_evidence_lines_with_gold": len(ev_hits)}
    qs, qe = question_span(record.prompt)
    if chk.occurs(record.prompt[qs:qe], pat, record.gold):
        return "gold_in_question", {}
    i, s, e = ev_hits[0]
    matches = [m.span() for m in pat.finditer(record.prompt)]
    inside = [(a, b) for a, b in matches if s <= a and b <= e]
    if len(matches) != len(inside):
        return "gold_elsewhere", {}
    return None, {"line_index": i, "start": s, "end": e, "n_occurrences": len(inside)}


def affixes(text: str) -> tuple[str, str, str]:
    """(prefix, core, suffix) of a gold string, one trailing period dropped first ("$800" -> "$", "800", "")."""
    raw = text.strip()
    raw = raw[:-1] if raw.endswith(".") else raw
    c = core(raw)
    if not c:
        return "", "", ""
    i = raw.index(c)
    return raw[:i], c, raw[i + len(c):]


def surface(text: str) -> str:
    """The text inserted into the evidence line: the substitute's raw gold, one trailing period dropped."""
    return "".join(affixes(text))


def swap_prompt(record, place: dict, substitute: str) -> tuple[str, str, str]:
    """(swap prompt, clean evidence line, swap evidence line): each verbatim match of the gold in that one line,
    widened to the gold's own prefix / suffix when the line has them next to it ("$" before "800"), is replaced by
    the substitute's surface form (its raw gold, one trailing period dropped), so the line carries S as P1 scores it."""
    pat = gold_pattern(record.gold)
    pre, _, suf = affixes(record.gold)
    s, e = place["start"], place["end"]
    line = record.prompt[s:e]
    out, cursor = [], 0
    for m in pat.finditer(line):
        a, b = m.span()
        if pre and line[max(0, a - len(pre)):a].lower() == pre.lower():
            a -= len(pre)
        if suf and line[b:b + len(suf)].lower() == suf.lower():
            b += len(suf)
        out += [line[cursor:a], surface(substitute)]
        cursor = b
    new_line = "".join(out) + line[cursor:]
    prompt = record.prompt[:s] + new_line + record.prompt[e:]
    assert prompt[:s] == record.prompt[:s] and prompt[s + len(new_line):] == record.prompt[e:]
    return prompt, line, new_line


# --- tokens ---------------------------------------------------------------------------------------------------------

def token_info(tokenizer, text: str) -> dict:
    from memctl.analysis.jacobian import _ids, _margin_length
    from memctl.analysis.jlens_readout import ANSWER_PREFIX

    answer = _ids(tokenizer, " " + text)
    k = _margin_length(tokenizer, answer)
    first = answer[k - 1]
    a = _ids(tokenizer, ANSWER_PREFIX)
    b = _ids(tokenizer, ANSWER_PREFIX + " " + text)
    stable = b[:len(a)] == a and len(b) > len(a)
    return {"n_tokens": len(answer), "leading": tuple(answer[:k - 1]), "first_id": int(first),
            "content": tuple(answer[k - 1:]),
            "first_text": tokenizer.decode([first]), "cont_first_id": int(b[len(a)]) if stable else None,
            "cont_tail_equal": stable and b[len(a):] == answer}


# --- substitute -----------------------------------------------------------------------------------------------------

def pool_order(records) -> list:
    pool = sorted(records, key=lambda r: (r.fold, r.question_id))
    random.Random(SEED).shuffle(pool)
    return pool


def row_candidates(records_sorted, i: int) -> list:
    """Rev 2: row i's own candidate list (every other row, (fold, id) order) shuffled with random.Random(44 + i)."""
    cands = [r for j, r in enumerate(records_sorted) if j != i]
    random.Random(SEED + i).shuffle(cands)
    return cands


def norm(text: str) -> str:
    return " ".join(_tokens(text))


def first_cap(text: str) -> bool:
    s = surface(text)
    return bool(s) and s[0].isupper()


def choose_substitute(record, place: dict, pool, tinfo: dict, types: dict, chk: Checker,
                      uses: Counter | None = None, rev: int = 1, stats: Counter | None = None):
    """(substitute record | None, exclusion | None, swap prompt, lines). `tinfo`/`types` keyed by question id.
    Rev 2 (`rev=2`, `uses` = substitute-string use counts so far): "other" substitutes need the same LongMemEval
    question type, the same first-character capitalisation and +-1 token length; a candidate whose normalised string
    already has CAP uses is skipped (counted in `stats["cap_skips"]`; a row whose only passing candidates were at
    their cap is counted in `stats["no_substitute_cap_only"]` and excluded as no_substitute)."""
    gnorm = norm(record.gold)
    g = tinfo[record.question_id]
    gpat = gold_pattern(record.gold)
    other = types[record.question_id] == "other"
    tol = OTHER_LEN_TOL if rev >= 2 and other else LEN_TOL
    passed_but_first = passed_but_cap = False
    for cand in pool:
        if cand.question_id == record.question_id or norm(cand.gold) == gnorm:
            continue
        if types[cand.question_id] != types[record.question_id]:
            continue
        if rev >= 2 and other and (cand.question_type != record.question_type
                                   or first_cap(cand.gold) != first_cap(record.gold)):
            continue
        if rev >= 8:
            why8 = rev8_reject(record.gold, cand.gold, types[record.question_id])
            if why8 is not None:
                if stats is not None:
                    stats[f"rev8_reject_{why8}"] += 1
                continue
        c = tinfo[cand.question_id]
        if abs(c["n_tokens"] - g["n_tokens"]) > tol:
            continue
        cpat = gold_pattern(cand.gold)
        if cpat is None or not surface(cand.gold) or chk.occurs(record.prompt, cpat, cand.gold):
            continue
        prompt, line, new_line = swap_prompt(record, place, cand.gold)
        if chk.occurs(prompt, gpat, record.gold):
            continue
        if c["leading"] != g["leading"]:
            continue
        if rev >= 3:
            from memctl.analysis.swap import shared_prefix
            _, gd, sd = shared_prefix(g["content"], c["content"])
            if gd is None or sd is None:
                passed_but_first = True
                continue
        elif c["first_id"] == g["first_id"]:
            passed_but_first = True
            continue
        if uses is not None and uses[norm(cand.gold)] >= CAP:
            passed_but_cap = True
            if stats is not None:
                stats["cap_skips"] += 1
            continue
        return cand, None, prompt, (line, new_line)
    if passed_but_cap and stats is not None:
        stats["no_substitute_cap_only"] += 1
    return None, ("first_token_equal" if passed_but_first and not passed_but_cap else "no_substitute"), None, None


def alignment(tokenizer, record, prompt: str, line: str, new_line: str, chat_text_fn) -> str | None:
    from memctl.analysis import patching as pt
    from memctl.analysis.jlens_readout import readout_for

    memory = tuple(new_line if l == line else l for l in record.memory_lines)
    evidence = tuple(new_line if l == line else l for l in record.evidence_lines)
    try:
        clean = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                            chat_text_fn=chat_text_fn)
        swap = readout_for(tokenizer, prompt, memory, evidence, record.gold, chat_text_fn=chat_text_fn)
        x = pt.RowInputs(record.question_id, record.fold, record.prompt, prompt, tuple(clean.input_ids),
                         tuple(swap.input_ids), clean.positions["P1"][0], swap.positions["P1"][0],
                         clean.positions["Q"][0], swap.positions["Q"][0], clean.tracked["gold"],
                         clean.tracked["unknown_sp"], clean.gold_token_text, clean.n_leading_ws, 0)
        if swap.tracked != clean.tracked or swap.n_leading_ws != clean.n_leading_ws:
            return "tracked ids or forced whitespace differ"
        pt.check_alignment(x)
    except Exception as exc:  # noqa: BLE001 - every failure of the check is an exclusion, recorded with its reason
        return f"{type(exc).__name__}: {exc}"
    return None


# --- population -----------------------------------------------------------------------------------------------------

def label23(record) -> str:
    if record.correct:
        return "success"
    return "refusal" if record.unknown else "wrong"


def qwen_population(runs_dir: str = "runs"):
    from scripts.exp44_patch import population

    rows = population("qwen3b", range(5), runs_dir)
    assert all(r.correct is not None for r in rows), "a Qwen population row lacks its §23 label"
    verdicts = {v["question_id"]: v for v in json.loads(Path(runs_dir, "exp23_verdicts.json").read_text())
                if v["reader"] == "qwen3b" and v["arm"] == "fixed8"}
    for r in rows:
        v = verdicts[r.question_id]
        assert (v["correct"], v["unknown"]) == (r.correct, r.unknown) and v["fold"] == r.fold, r.question_id
    return rows


def llama_population(runs_dir: str = "runs"):
    from memctl.analysis import families as F
    from memctl.jacobian_data import family_population, load_family_records

    small, large = F.get("llama3b"), F.get("llama8b")
    rows = []
    for fold in range(5):
        a = {r.question_id: r for r in family_population(load_family_records(small, fold, runs_dir)[0])}
        b = {r.question_id: r for r in family_population(load_family_records(large, fold, runs_dir)[0])}
        for q in sorted(set(a) & set(b)):
            if hashlib.sha256(a[q].prompt.encode()).hexdigest() == hashlib.sha256(b[q].prompt.encode()).hexdigest():
                rows.append(a[q])
    verdicts = {(v["question_id"], v["fold"]): v for v in json.loads(Path(runs_dir, "exp24_verdicts.json").read_text())
                if v["reader"] == "llama3b" and v["arm"] == "fixed8"}
    for r in rows:
        v = verdicts[(r.question_id, r.fold)]
        assert (v["correct"], v["unknown"]) == (r.correct, r.unknown), r.question_id
    return rows


# --- the gate -------------------------------------------------------------------------------------------------------

def evaluate(records, tokenizer, chat_text_fn=None, log=print, rev: int = 2) -> dict:
    records = sorted(records, key=lambda r: (r.fold, r.question_id))
    chk = Checker()
    tinfo = {r.question_id: token_info(tokenizer, r.gold) for r in records}
    types = {r.question_id: type_key(r.gold, rev) for r in records}
    pool = pool_order(records) if rev < 2 else None
    uses, stats = (Counter(), Counter()) if rev >= 2 else (None, None)
    removed = Counter()
    eligible, align_reasons, removed_rows = [], [], []
    for i, r in enumerate(records):
        why, place = place_gold(r, chk)
        if why is None and rev >= 8 and types[r.question_id] == "other":
            why = "type_other_rev8"
        if why is None:
            cands = pool if rev < 2 else row_candidates(records, i)
            cand, why, prompt, lines = choose_substitute(r, place, cands, tinfo, types, chk, uses=uses, rev=rev,
                                                         stats=stats)
        if why is None and rev >= 3:
            from memctl.analysis.swap import build_swap_inputs
            try:
                x = build_swap_inputs(tokenizer, r, prompt, lines[0], lines[1], cand.gold, (), chat_text_fn)
                bad = None
            except Exception as exc:  # noqa: BLE001 - every failure of the check is an exclusion, with its reason
                bad = f"{type(exc).__name__}: {exc}"
            if bad is not None:
                why = "alignment_failure"
                align_reasons.append({"question_id": r.question_id, "reason": bad})
        elif why is None:
            bad = alignment(tokenizer, r, prompt, lines[0], lines[1], chat_text_fn)
            if bad is not None:
                why = "alignment_failure"
                align_reasons.append({"question_id": r.question_id, "reason": bad})
        if why is not None:
            removed[why] += 1
            removed_rows.append({"question_id": r.question_id, "fold": r.fold, "label": label23(r),
                                 "question_type": r.question_type, "gold": r.gold, "type": types[r.question_id],
                                 "exclusion": why})
            continue
        if uses is not None:
            uses[norm(cand.gold)] += 1
        g, s = tinfo[r.question_id], tinfo[cand.question_id]
        eligible.append({
            "question_id": r.question_id, "fold": r.fold, "question_type": r.question_type,
            "gold": r.gold, "substitute": cand.gold, "substitute_from": cand.question_id,
            "gold_surface": surface(r.gold), "substitute_surface": surface(cand.gold),
            "surface_type": types[r.question_id], "substitute_type": types[cand.question_id],
            "first_cap": first_cap(r.gold), "substitute_first_cap": first_cap(cand.gold),
            "substitute_question_type": cand.question_type,
            "gold_first_id": g["first_id"], "gold_first_text": g["first_text"],
            "substitute_first_id": s["first_id"], "substitute_first_text": s["first_text"],
            "gold_n_tokens": g["n_tokens"], "substitute_n_tokens": s["n_tokens"], "leading_ws_ids": list(g["leading"]),
            "evidence_line_index": place["line_index"], "n_occurrences_in_line": place["n_occurrences"],
            "clean_prompt_sha256": hashlib.sha256(r.prompt.encode()).hexdigest(),
            "swap_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "row_index": i, "_label": label23(r)})
        if rev >= 3:
            eligible[-1].update({"shared_prefix_ids": list(x.prefix_ids),
                                 "shared_prefix_text": tokenizer.decode(list(x.prefix_ids)),
                                 "gold_read_id": x.g_id, "gold_read_text": x.gold_token_text,
                                 "substitute_read_id": x.s_id, "substitute_read_text": x.sub_token_text})
    cont_disagree = sum(1 for e in eligible for k in ("gold", "substitute")
                        if not tinfo[e["question_id"] if k == "gold" else e["substitute_from"]]["cont_tail_equal"])
    return {"records": records, "eligible": eligible, "removed": removed, "align_reasons": align_reasons,
            "types": types, "rev": rev, "removed_rows": removed_rows, "cap_stats": dict(stats or {}), "regex_checks": chk.checks, "regex_disagreements": chk.disagreements,
            "continuation_split_unstable": cont_disagree}


def tables(res) -> dict:
    recs, el = res["records"], res["eligible"]
    out = {"population": {k: sum(r.fold in f for r in recs) for k, f in FOLD_SETS.items()},
           "eligible": {k: sum(e["fold"] in f for e in el) for k, f in FOLD_SETS.items()}}
    qtypes = sorted({r.question_type for r in recs})
    out["by_question_type"] = {k: {t: sum(e["fold"] in f and e["question_type"] == t for e in el) for t in qtypes}
                               for k, f in FOLD_SETS.items()}
    out["by_label"] = {k: {lab: sum(e["fold"] in f and e["_label"] == lab for e in el) for lab in LABELS}
                       for k, f in (("confirmation", FOLD_SETS["confirmation"]), ("all", FOLD_SETS["all"]))}
    out["by_surface_type"] = {k: dict(Counter(e["surface_type"] for e in el if e["fold"] in f))
                              for k, f in FOLD_SETS.items()}
    out["population_surface_types"] = dict(Counter(res["types"].values()))
    out["by_surface_type_and_label_all"] = {ty: {lab: sum(e["surface_type"] == ty and e["_label"] == lab for e in el)
                                                 for lab in LABELS} for ty in sorted({e["surface_type"] for e in el})}
    out["exclusions_in_order"] = {k: res["removed"].get(k, 0) for k in exclusions(res["rev"])}
    left, steps = len(recs), []
    for k in exclusions(res["rev"]):
        left -= res["removed"].get(k, 0)
        steps.append({"exclusion": k, "removed": res["removed"].get(k, 0), "remaining": left})
    out["exclusion_steps"] = steps
    out["distinct_substitutes"] = len({e["substitute_from"] for e in el})
    mult = Counter(norm(e["substitute"]) for e in el)
    out["distinct_substitute_strings"] = len(mult)
    out["max_substitute_multiplicity"] = max(mult.values()) if mult else 0
    out["top_substitutes"] = mult.most_common(8)
    out["cap_stats"] = res.get("cap_stats", {})
    pre = [e for e in el if e.get("shared_prefix_ids")]
    out["shared_prefix"] = {"n_nonempty": len(pre), "prefixes": dict(Counter(e["shared_prefix_text"] for e in pre)),
                            "rows": [{"question_id": e["question_id"], "fold": e["fold"], "label": e["_label"],
                                      "gold": e["gold"], "substitute": e["substitute"],
                                      "prefix": e["shared_prefix_text"], "gold_read": e["gold_read_text"],
                                      "substitute_read": e["substitute_read_text"]} for e in pre]}
    out["alignment_failures"] = res["align_reasons"]
    out["substitute_exclusions"] = [x for x in res.get("removed_rows", [])
                                    if x["exclusion"] in ("no_substitute", "first_token_equal", "type_other_rev8")]
    out["regex_vs_normalised"] = {"checks": res["regex_checks"], "disagreements": res["regex_disagreements"]}
    out["continuation_split_unstable"] = res["continuation_split_unstable"]
    return out


def public(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def rev8_changes(res: dict, rev3_rows: list[dict]) -> dict:
    """Rev 8 rule (4), the honesty clause's basis: compare with the recorded rev 3 / rev 4 eligible set. A rev 3 row
    whose substitute passes the rev 8 rules must be eligible under rev 8 with the identical record (asserted: same
    substitute string, same swap prompt sha, same read ids); every other rev 3 row is either excluded as "other",
    re-drawn (a changed substitute) or lost (no passing candidate / alignment). No row may be new."""
    new = {e["question_id"]: public(e) for e in res["eligible"]}
    removed = {x["question_id"]: x["exclusion"] for x in res["removed_rows"]}
    unchanged, changed, lost, other = [], [], [], []
    for old in rev3_rows:
        q = old["question_id"]
        if old["surface_type"] == "other":
            assert removed.get(q) == "type_other_rev8", (q, removed.get(q))
            other.append(q)
            continue
        why = rev8_reject(old["gold"], old["substitute"], old["surface_type"])
        if why is None:
            assert q in new, f"rev 8: {q} keeps a passing rev 3 substitute {old['substitute']!r} but is {removed.get(q)}"
            assert new[q] == old, (q, {k: (old.get(k), new[q].get(k)) for k in set(old) | set(new[q])
                                       if old.get(k) != new[q].get(k)})
            unchanged.append(q)
            continue
        base = {"question_id": q, "fold": old["fold"], "surface_type": old["surface_type"], "gold": old["gold"],
                "rev3_substitute": old["substitute"], "rev3_fails": why}
        if q in new:
            e = new[q]
            assert e["substitute"] != old["substitute"]
            changed.append({**base, "rev8_substitute": e["substitute"], "rev8_read": [e["gold_read_text"],
                            e["substitute_read_text"]], "shared_prefix": e["shared_prefix_text"],
                            "read_differs_by_case_only": e["gold_read_text"].strip().lower()
                            == e["substitute_read_text"].strip().lower()})
        else:
            lost.append({**base, "exclusion": removed.get(q)})
    extra = sorted(set(new) - {o["question_id"] for o in rev3_rows})
    assert not extra, f"rev 8: rows eligible now but not under rev 3: {extra}"
    labels = {e["question_id"]: e["_label"] for e in res["eligible"]}
    for c in changed:
        c["label"] = labels[c["question_id"]]
    return {"rev3_eligible": len(rev3_rows), "rev3_eligible_sha256": None, "excluded_other": len(other),
            "unchanged": len(unchanged), "changed": len(changed), "lost": len(lost),
            "unchanged_identity_asserted": True, "changed_rows": changed, "lost_rows": lost,
            "unchanged_question_ids": unchanged}


def gate(t: dict, rev: int = 1) -> dict:
    n = t["eligible"]["confirmation"]
    qt = {k: v for k, v in t["by_question_type"]["confirmation"].items() if v >= 15}
    fail = t["by_label"]["confirmation"]["refusal"] + t["by_label"]["confirmation"]["wrong"]
    fail_all = t["by_label"]["all"]["refusal"] + t["by_label"]["all"]["wrong"]
    if n < 50 or fail < 15:
        verdict = "stop"
    elif len(qt) < 2 and rev < 2:
        verdict = "stop (fewer than two question types of >= 15; the spec names no pool route for this)"
    elif fail >= 30:
        verdict = "proceed"
    else:
        verdict = "pool (claims read on all 330)"
    if rev >= 2 and len(qt) < 2 and verdict != "stop":       # rev 2 ruling 7: proceed with a type-diversity caveat
        verdict += " + type-diversity caveat (fewer than two question types of >= 15 on folds 2-4; rev 2 ruling 7)"
    return {"eligible_folds_2_4": n, "question_types_ge_15_folds_2_4": qt, "n_question_types_ge_15": len(qt),
            "inview_failures_folds_2_4": fail, "inview_failures_all_330": fail_all, "verdict": verdict}


RULES_REV2 = {
    "pool": "rev 2: row i (index in the family's (fold, question_id) order) shuffles the other 329 rows with"
            " random.Random(44 + i) and walks that list; first candidate passing every rule and not at its cap",
    "cap": f"each D1-normalised substitute string used at most {CAP} times across the family's eligible rows; rows in"
           " (fold, id) order; a candidate at its cap is skipped; a use counts once the row is eligible; a row whose"
           " only passing candidates were capped is excluded as no_substitute (count in cap_stats)",
    "surface_types": "rev 2: number split into currency / percent / duration / bare (number_subtype; the substitute"
                     " must share the subtype); date, name as rev 1",
    "other": f"rev 2: 'other' substitutes from rows of the same LongMemEval question type, same first-character"
             f" capitalisation of the surface form, |n_tokens diff| <= {OTHER_LEN_TOL}",
    "number_subtypes": NUMBER_SUBTYPE_RULES,
}

RULES = {
    "verbatim": "the gold's D1 tokens (answer_selection._tokens) as a contiguous whole-token run of the text's D1 tokens"
                " (ASCII case-insensitive regex, cross-checked against the normalised form); stricter than D1's"
                " substring branch, no all-tokens branch",
    "placement": "segment_prompt line spans (separator excluded); question = 'Question:' to '\\n\\nNote:'; elsewhere ="
                 " any whole-prompt match outside the one evidence line",
    "swap": "every match of the gold inside that one evidence line (widened to the gold's own adjacent non-alphanumeric"
            " prefix/suffix, e.g. '$') replaced by the substitute's surface form (its raw gold, one trailing period"
            " dropped); all other bytes identical (asserted)",
    "pool": "the family's 330 population rows sorted by (fold, question_id), shuffled once with random.Random(44);"
            " each row iterates the same order from the start, first candidate passing every rule",
    "candidate_rules_in_order": ["not self, different normalised gold", "same surface type",
                                 f"|n_tokens(' '+S) - n_tokens(' '+G)| <= {LEN_TOL}",
                                 "S not verbatim anywhere in the clean prompt",
                                 "G not verbatim anywhere in the swap prompt",
                                 "same teacher-forced leading-whitespace tokens as G", "different first content token"],
    "first_token": "jlens_readout.tracked_token_ids: first non-whitespace token of ids(' ' + text) (the token P1"
                   " predicts in §34); continuation_first_id form ids('Answer:' + ' ' + text) - ids('Answer:') checked",
    "alignment": "patching.check_alignment on readout_for(clean) vs readout_for(swap), gold forced as §34; the cut-is-"
                 "shorter clause off (n_evidence_lines=0); tracked ids and forced whitespace must match",
    "exclusion_order": list(EXCLUSIONS),
    "labels": "§23: success = correct; refusal = not correct and unknown; wrong = otherwise (Qwen exp23 fixed8,"
              " Llama exp24 fixed8)",
    "surface_types": SURFACE_RULES,
}


def markdown(per: dict, rev: int = 2) -> str:
    title = ("# §44 rev 8: repaired swap gate (no 'other'; bare numbers form-matched; determiner matched), Qwen 3B"
             if rev >= 8 else
             "# §44 rev 3: swap-subset gate recomputed with the shared-prefix rule" if rev >= 3
             else "# §44 rev 2: swap-subset gate recomputed under rulings 2-3" if rev >= 2
             else "# §44 rev 1 §A: swap-subset feasibility gate")
    L = [title, "",
         "Data only (tokenizers, no forward pass). Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 §A"
         + (" with the rev 2 rulings 2 (per-row draw, cap 3) and 3 (number subtypes, same-type 'other')" if rev >= 2
            else "") + (" and rev 8 (83fe60a, D-47: no 'other', bare form-matched, determiner matched)" if rev >= 8
                         else "") + "; rules in gate.json `rules` and in scripts/exp44_swap_gate.py. Counts only.", ""]
    for fam, d in per.items():
        t, g = d["tables"], d["gate"]
        L += [f"## {fam}", "", "| | dev (0-1) | confirmation (2-4) | all |", "|---|---|---|---|",
              f"| population | {t['population']['dev']} | {t['population']['confirmation']} | {t['population']['all']} |",
              f"| eligible | {t['eligible']['dev']} | {t['eligible']['confirmation']} | {t['eligible']['all']} |", "",
              "Eligible by LongMemEval question type:", "", "| type | dev | confirmation | all |", "|---|---|---|---|"]
        for qt in t["by_question_type"]["all"]:
            L.append(f"| {qt} | " + " | ".join(str(t["by_question_type"][k][qt]) for k in FOLD_SETS) + " |")
        L += ["", "Eligible by §23 label:", "", "| label | confirmation | all |", "|---|---|---|"]
        for lab in LABELS:
            L.append(f"| {lab} | {t['by_label']['confirmation'][lab]} | {t['by_label']['all'][lab]} |")
        L += ["", "Exclusions in order of application (of 330):", "", "| exclusion | removed | remaining |", "|---|---|---|"]
        for s in t["exclusion_steps"]:
            L.append(f"| {s['exclusion']} | {s['removed']} | {s['remaining']} |")
        L += ["", f"Eligible by surface type (all): {t['by_surface_type']['all']}; population surface types: "
                  f"{t['population_surface_types']}; distinct substitute rows used: {t['distinct_substitutes']}; "
                  f"distinct substitute strings: {t['distinct_substitute_strings']}, maximum multiplicity "
                  f"{t['max_substitute_multiplicity']} (most used: {t['top_substitutes']}); cap stats: "
                  f"{t['cap_stats']}.",
              f"Eligible by surface type x §23 label (all): {t['by_surface_type_and_label_all']}.",
              f"Regex vs normalised cross-check: {t['regex_vs_normalised']}; continuation split unstable: "
              f"{t['continuation_split_unstable']}.", "",
              f"Gate arithmetic: eligible folds 2-4 = {g['eligible_folds_2_4']} (need >= 50); question types with "
              f">= 15 = {g['question_types_ge_15_folds_2_4']} (need >= 2); in-view failures folds 2-4 = "
              f"{g['inview_failures_folds_2_4']} (>= 30 proceed, 15-29 pool, < 15 stop); all 330: "
              f"{g['inview_failures_all_330']}. **Verdict: {g['verdict']}**"
              + ("" if fam == "qwen3b" else " (descriptive; the gate binds on Qwen 3B only)"), ""]
        if rev >= 8 and "rev8_changes" in d:
            ch = d["rev8_changes"]
            L += [f"Rev 8 against the recorded rev 3 / rev 4 set ({ch['rev3_eligible']} rows): 'other' excluded "
                  f"{ch['excluded_other']}; unchanged substitute {ch['unchanged']} (record identity asserted); "
                  f"changed substitute {ch['changed']}; lost (no passing candidate) {ch['lost']}.", "",
                  "Changed-substitute rows (the only fresh evidence, honesty clause):", ""]
            L += [f"- {c['question_id']} f{c['fold']} {c['label']} [{c['surface_type']}] {c['gold']!r}: "
                  f"{c['rev3_substitute']!r} (fails {c['rev3_fails']}) -> {c['rev8_substitute']!r}, read "
                  f"{c['rev8_read'][0]!r} vs {c['rev8_read'][1]!r}" + (" [read differs by case only]"
                                                                      if c["read_differs_by_case_only"] else "")
                  for c in ch["changed_rows"]]
            L += ["", "Lost rows (rev 3 substitute fails rev 8, no passing candidate):", ""]
            L += [f"- {c['question_id']} f{c['fold']} [{c['surface_type']}] {c['gold']!r}: {c['rev3_substitute']!r} "
                  f"(fails {c['rev3_fails']}) -> {c['exclusion']}" for c in ch["lost_rows"]]
            L.append("")
        if rev >= 2 and fam == "qwen3b":
            c = g["rev2_check"]
            L += [f"Rev 2 check (5146f5b: the POOL verdict stands if >= 85 eligible and >= 40 failures on all 330): "
                  f"eligible {c['eligible_all']}, failures {c['failures_all']} -> **{c['result']}**", ""]
        if rev >= 2:
            if rev >= 3:
                sp = t["shared_prefix"]
                L += [f"Rows with a non-empty shared prefix: {sp['n_nonempty']} (prefixes {sp['prefixes']}):", ""]
                L += [f"- {x['question_id']} f{x['fold']} {x['label']} {x['gold']!r} -> {x['substitute']!r}: prefix "
                      f"{x['prefix']!r}, read {x['gold_read']!r} vs {x['substitute_read']!r}" for x in sp["rows"]]
                L.append("")
            L += ["Rows excluded at the substitute step (no_substitute / first_token_equal"
                  + (" / type_other_rev8" if rev >= 8 else "") + "):", ""]
            L += [f"- {x['question_id']} f{x['fold']} {x['label']} [{x['type']}] {x['gold']!r}: {x['exclusion']}"
                  for x in t["substitute_exclusions"]]
            L += ["", "Eligible rows (gold -> substitute, type):", ""]
            L += [f"- {e['question_id']} f{e['fold']} [{e['surface_type']}] {e['gold']!r} -> {e['substitute']!r}"
                  for e in d["eligible"]]
            L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="default runs/exp44_swapgate (rev 8: runs/exp44_swapgate/rev8)")
    ap.add_argument("--rev", type=int, default=3, choices=(1, 2, 3, 8))
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from memctl.analysis import families as F

    out = Path(args.out or (OUT / "rev8" if args.rev >= 8 else OUT))
    if args.rev >= 8 and out.resolve() == OUT.resolve():
        raise SystemExit("exp44_swap_gate: rev 8 must not overwrite the rev 3 outputs in runs/exp44_swapgate/")
    out.mkdir(parents=True, exist_ok=True)
    per = {}
    for fam in (("qwen3b",) if args.rev >= 8 else ("qwen3b", "llama3b")):
        if fam == "qwen3b":
            records, tok, fn = qwen_population(), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct"), None
        else:
            cfg = F.get("llama3b")
            tok = AutoTokenizer.from_pretrained(cfg.model_id)
            records, fn = llama_population(), (lambda p, tok=tok, cfg=cfg: F.chat_text(tok, cfg, p))
        assert len(records) == 330 and sum(r.fold < 2 for r in records) == 126, (fam, len(records))
        res = evaluate(records, tok, fn, rev=args.rev)
        t = tables(res)
        per[fam] = {"tables": t, "gate": gate(t, args.rev)}
        if args.rev >= 8:      # the rev 1 gate arithmetic does not bind: the pooled 330-row design is fixed by rev 4
            per[fam]["gate"]["verdict_rev1_arithmetic"] = per[fam]["gate"]["verdict"]
            per[fam]["gate"]["verdict"] = "n/a under rev 8 (POOL fixed by rev 4; rev 1 arithmetic recorded beside)"
        if args.rev >= 2 and fam == "qwen3b":
            ea = t["eligible"]["all"]
            fa = t["by_label"]["all"]["refusal"] + t["by_label"]["all"]["wrong"]
            per[fam]["gate"]["rev2_check"] = {
                "eligible_all": ea, "failures_all": fa, "need": {"eligible_all": 85, "failures_all": 40},
                "result": ("recorded for rev 4 (rev 3: POOL regardless; the 85 / 40 bar was a re-read trigger)"
                           if args.rev >= 3 else
                           "verdict stands (POOL)" if ea >= 85 and fa >= 40 else "FLAG: Advisor re-reads the gate (rev 3)")}
        if args.rev >= 8:
            prev = json.loads((OUT / "gate.json").read_text())
            assert prev["rev"] == 3, "runs/exp44_swapgate/gate.json is not the rev 3 gate"
            rev3_rows = [json.loads(l) for l in REV3_ELIGIBLE.read_text().splitlines() if l.strip()]
            per[fam]["rev8_changes"] = rev8_changes(res, rev3_rows)
            per[fam]["rev8_changes"]["rev3_eligible_sha256"] = hashlib.sha256(REV3_ELIGIBLE.read_bytes()).hexdigest()
            print(fam, "rev 8 vs rev 3:", {k: v for k, v in per[fam]["rev8_changes"].items()
                                           if isinstance(v, int)}, flush=True)
        if args.rev >= 2:
            per[fam]["eligible"] = [{k: e[k] for k in ("question_id", "fold", "surface_type", "gold", "substitute")}
                                    for e in res["eligible"]]
        with open(out / f"eligible_{fam}.jsonl", "w") as fh:
            for e in res["eligible"]:
                fh.write(json.dumps({k: v for k, v in e.items() if not k.startswith("_")}) + "\n")
        print(fam, json.dumps(per[fam]["gate"]), flush=True)
    rules = dict(RULES, **({"rev2": RULES_REV2} if args.rev >= 2 else {}))
    if args.rev >= 8:
        rules["rev8"] = {"other": "surface_type 'other' excluded as type_other_rev8 (after gold_elsewhere, before the"
                                  " substitute walk)",
                         "bare": "number:bare: same form (digit / word); digit-form same digit count of the leading"
                                 " number (range: every digit of the string); same unit presence (trailing word)",
                         "bare_regex": BARE_RE.pattern,
                         "determiner": f"first word in {list(DETERMINERS)} (membership case-insensitive, followed by"
                                       " another word): the same word as written in S ('The' != 'the', else the"
                                       " read is the determiner's case), or none in both; every kept type",
                         "walk": "unchanged (random.Random(44 + i), first passing candidate, cap 3); a rev 3 row"
                                 " whose substitute passes keeps it (record identity asserted)"}
    if args.rev >= 3:
        rules["rev3"] = {"first_token": "different first token after the shared prefix (longest common content-token"
                                        " prefix of G and S after the leading whitespace); no divergent token ->"
                                        " first_token_equal",
                         "alignment": "memctl.analysis.swap.build_swap_inputs: clean / swap / cut with the prefix"
                                      " forced; it ends every sequence, identical"}
    payload = {"spec": "research/proposals/2026-10-29-exp44-patching.md rev 1 §A (D-45)"
                       + (" + rev 2 rulings 2-3 (5146f5b)" if args.rev >= 2 else "")
                       + (" + rev 3 shared-prefix rule (7acce03)" if args.rev >= 3 else "")
                       + (" + rev 8 repaired swap (83fe60a, D-47)" if args.rev >= 8 else ""), "rev": args.rev, "seed": SEED,
               "rules": rules, "families": per,
               "note": "Qwen 2.5 7B shares the Qwen 3B rows and prompts (exp44_patch.population asserts it)."}
    (out / "gate.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (out / "gate.md").write_text(markdown(per, args.rev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
