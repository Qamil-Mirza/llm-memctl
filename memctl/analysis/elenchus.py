"""§37: a three-turn elenchus (cite lines, derive blind, confront) as a self-check for the 3B's forced answers.

Pre-registration: research/proposals/2026-10-16-exp37-elenchus.md at 714da06 (revision 2). Pure logic, no GPU.

- Numbered block: the cached §23 prompt's "## Memory" block (reasoning instructions and goal line not shown, see
  reading 1) with "[k] " before the first character of each memory ITEM, k = 1..n in prompt order; one number per
  item, never per text line; the header line is not numbered. Asserted against PromptRecord.memory_lines.
- Cut-B / Cut-F: `ablate_evidence` on the unnumbered cached prompt, then the remaining items numbered 1..n'.
- E1 grounds: numbered block, the question, "You answered: <candidate>", the E1 text; up to 24 tokens; G = valid item
  numbers in the reply; g = 1 iff G is non-empty.
- E2 blind: only the cited items (with their numbers), the question, the E2 text; candidate never shown; 32 tokens;
  d = the reply's first line. E2 in dialogue: E1's user turn, E1's reply, then the dialogue text naming G.
- E3: no memory; Question, Answer 1, Answer 2 (candidate and d, order by the parity of the first hex digit of
  sha256(question_id)), the E3 text; one token, top-20 logprobs, the §36 v2 scorer; a = (pA + 0.5 pB)/(pA+pB+pC).
- Scores: e = 0 if g = 0 or d is "not determined" or a refusal, else a; m (normalised agreement); e_mech = g * m.

Readings taken where the proposal is open (reported to the Advisor):
1. "The memory block" is the "## Memory" header line plus the items (verification.memory_block), without the §23
   goal line: the goal line asks for an answer ("Answer with a short phrase ... reply: unknown"), the cue §36b removed,
   and §2's table lists only the block, the question and the turn's text. V-num, the §36b frame (a) one-shot, keeps
   frame (a)'s goal line, with the items numbered.
2. Message layout: block, blank line, the cached "Question: ..." line byte-exact, then "You answered: <c>" and the
   E1 text on their own lines (E1); cited items one per line, blank line, question, E2 text (E2); "Question: ...",
   "Answer 1: x", "Answer 2: y", E3 text, one per line (E3). The question line includes "(asked on <date>)" as cached.
3. E1 parse: every integer in the reply is a candidate number; numbers outside 1..n are dropped. The reply "parses"
   (compliance gate) iff it holds at least one integer in 1..n or the word NONE.
4. "<G>" in the dialogue text is rendered as the cited numbers in brackets, ascending, space-separated ("[2] [5]").
5. The E3 fallback "Are Answer 1 and Answer 2 the same answer?" replaces the first sentence only; the letter
   instructions stay.
6. Not-determined: the normalised reply equals or contains (as a whole-word phrase) one of the six normalised phrases.
7. Normaliser number rule: a "number" is a run of digits with an optional decimal part, after the number-word mapping;
   the numeric rule applies only when each side has exactly one and neither side looks like a time or a date (h:mm,
   d/m, a month name), so dates and times always go to the phrase rule.
8. Calibration rows: the first 20 development candidate rows by question id; "another question's candidate" is the
   next row's candidate in that order (cyclic), skipping any whose normalised form equals this one's.
9. Smoke E3: E3 runs on (candidate, d) when d is a real answer, else on (candidate, candidate), so every smoke row
   tests the E3 format.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from memctl.analysis import verification as ver
from memctl.analysis.answer_selection import is_refusal_answer
from memctl.analysis.jacobian import MEMORY_HEADER, ablate_evidence, segment_prompt

# --- pre-registered texts (§2), byte-exact ---------------------------------------------------------------------
E1_TEXT = ("Which memory lines is this answer based on? Reply with the line numbers only, for example [2] [5]. "
           "If no line bears on the question, reply NONE.")
E1_FALLBACK = "List the numbers of the memory lines that support your answer, separated by spaces, or write NONE."
YOU_ANSWERED = "You answered: "
E2_TEXT = ("Using only the lines above, what is the answer? Answer with a short phrase. If the lines above do not "
           "determine the answer, reply: not determined.")
E2_DIALOGUE = "Based only on lines {G}, what do those lines say the answer is? Short phrase, or: not determined."
E3_TEXT = ("Do Answer 1 and Answer 2 give the same answer to the question? Reply with one letter. A = same, B = partly "
           "the same or unclear, C = different")
E3_FALLBACK_SENTENCE = "Are Answer 1 and Answer 2 the same answer?"
E3_FIRST_SENTENCE = "Do Answer 1 and Answer 2 give the same answer to the question?"
NOT_DETERMINED = ("not determined", "cannot be determined", "undetermined", "unknown", "not specified", "not mentioned")
MAX_TOKENS = {"E1": 24, "E2": 32, "E2d": 32, "E3": 1, "E3d": 1, "Vnum": 1}
CALIBRATION_N, CALIBRATION_PASS = 20, 18
GATE_AUROC, GATE_MARGIN = 0.65, 0.05
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "greedy": True}
NORMALISER_VERSION = "v1"


def e3_text(fallback: bool = False) -> str:
    return E3_TEXT.replace(E3_FIRST_SENTENCE, E3_FALLBACK_SENTENCE, 1) if fallback else E3_TEXT


def wording(e1_fallback: bool = False, e3_fallback: bool = False) -> dict:
    return {"e1": E1_FALLBACK if e1_fallback else E1_TEXT, "you_answered": YOU_ANSWERED, "e2": E2_TEXT,
            "e2_dialogue": E2_DIALOGUE, "e3": e3_text(e3_fallback), "not_determined": NOT_DETERMINED,
            "max_tokens": MAX_TOKENS, "normaliser": NORMALISER_VERSION, "order_rule": "parity of first hex digit of "
            "sha256(question_id): even -> (candidate, d), odd -> (d, candidate)", "vnum": "§36b frame a, items numbered",
            "scorer": ver.SCORER_VERSION}


def wording_sha256(e1_fallback: bool = False, e3_fallback: bool = False) -> str:
    return ver.sha256(json.dumps(wording(e1_fallback, e3_fallback), sort_keys=True))


# --- the numbered block ---------------------------------------------------------------------------------------

def _item_starts(prompt: str, memory_lines) -> list[int]:
    spans = segment_prompt(prompt, tuple(memory_lines), ())
    starts = [s for (_, s, _) in spans[1:-1]]
    if len(starts) != len(memory_lines):
        raise AssertionError("numbering: item count mismatch")
    for s, line in zip(starts, memory_lines):
        if prompt[s:s + len(line)] != line:
            raise AssertionError("numbering: item not at its span")
    return starts


def number_prompt(prompt: str, memory_lines) -> str:
    """The prompt with "[k] " inserted before the first character of each memory item (k = 1..n). Checked by position:
    every item k starts right after its own "[k] ", and deleting exactly those n prefixes gives the prompt back."""
    starts = _item_starts(prompt, memory_lines)
    out, cursor, positions = [], 0, []
    for k, s in enumerate(starts, start=1):
        out.append(prompt[cursor:s])
        positions.append(sum(len(x) for x in out))
        out.append(f"[{k}] ")
        cursor = s
    out.append(prompt[cursor:])
    numbered = "".join(out)
    for k, (pos, line) in enumerate(zip(positions, memory_lines), start=1):
        tag = f"[{k}] "
        if numbered[pos:pos + len(tag) + len(line)] != tag + line:
            raise AssertionError(f"numbering: item {k} not right after its number")
    rebuilt, cursor = [], 0
    for k, pos in enumerate(positions, start=1):
        rebuilt.append(numbered[cursor:pos])
        cursor = pos + len(f"[{k}] ")
    rebuilt.append(numbered[cursor:])
    if "".join(rebuilt) != prompt:
        raise AssertionError("numbering: not reversible")
    return numbered


def number_positions(prompt: str, memory_lines) -> list[int]:
    """Character offsets of the inserted "[k] " tags in number_prompt's output."""
    starts = _item_starts(prompt, memory_lines)
    return [s + sum(len(f"[{j}] ") for j in range(1, k)) for k, s in enumerate(starts, start=1)]


def numbered_block(prompt: str, memory_lines) -> str:
    """"## Memory" + the numbered items (the block shown in E1)."""
    numbered = number_prompt(prompt, memory_lines)
    block = ver.memory_block(numbered)
    if not block.startswith(MEMORY_HEADER) or block.count("[1] ") < 1:
        raise AssertionError("numbering: block malformed")
    return block


def cut_view(prompt: str, memory_lines, evidence_lines) -> tuple[str, tuple[str, ...]]:
    """(cut prompt, remaining items): ablate_evidence on the unnumbered prompt; numbering is applied afterwards."""
    cut = ver.cut_prompt(prompt, memory_lines, evidence_lines)
    evidence = set(evidence_lines)
    return cut, tuple(x for x in memory_lines if x not in evidence)


def question_line(prompt: str) -> str:
    return ver.split_prompt(ver.strip_reasoning(prompt))[1]


# --- turn builders (each returns chat messages) ----------------------------------------------------------------

def e1_messages(prompt: str, memory_lines, candidate: str, fallback: bool = False) -> list[dict]:
    text = (f"{numbered_block(prompt, memory_lines)}\n\n{question_line(prompt)}\n{YOU_ANSWERED}{candidate}\n"
            f"{E1_FALLBACK if fallback else E1_TEXT}")
    return [{"role": "user", "content": text}]


def cited_lines(memory_lines, cited: list[int]) -> str:
    return "\n".join(f"[{k}] {memory_lines[k - 1]}" for k in sorted(cited))


def e2_blind_messages(prompt: str, memory_lines, cited: list[int]) -> list[dict]:
    if not cited:
        raise ValueError("E2 needs a non-empty G")
    return [{"role": "user", "content": f"{cited_lines(memory_lines, cited)}\n\n{question_line(prompt)}\n{E2_TEXT}"}]


def g_text(cited: list[int]) -> str:
    return " ".join(f"[{k}]" for k in sorted(cited))


def e2_dialogue_messages(e1: list[dict], e1_reply: str, cited: list[int]) -> list[dict]:
    return e1 + [{"role": "assistant", "content": e1_reply},
                 {"role": "user", "content": E2_DIALOGUE.format(G=g_text(cited))}]


def order_flip(question_id: str) -> bool:
    """True when the order is (d, candidate): the first hex digit of sha256(question_id) is odd."""
    return int(hashlib.sha256(question_id.encode()).hexdigest()[0], 16) % 2 == 1


def e3_messages(prompt: str, question_id: str, candidate: str, d: str, fallback: bool = False) -> list[dict]:
    x, y = (d, candidate) if order_flip(question_id) else (candidate, d)
    return e3_pair_messages(prompt, x, y, fallback)


def e3_pair_messages(prompt: str, x: str, y: str, fallback: bool = False) -> list[dict]:
    return [{"role": "user", "content": f"{question_line(prompt)}\nAnswer 1: {x}\nAnswer 2: {y}\n{e3_text(fallback)}"}]


def vnum_messages(prompt: str, memory_lines, candidate: str) -> list[dict]:
    """V-num: the §36b frame (a) one-shot (P2 frame, nothing appended) on the numbered prompt."""
    system, user, prefix = ver.frame_messages(number_prompt(prompt, memory_lines), candidate, "a")
    assert system is None and prefix == ""
    return [{"role": "user", "content": user}]


# --- parsing --------------------------------------------------------------------------------------------------

def parse_e1(reply: str, n: int) -> dict:
    """{'G': sorted valid numbers, 'g': 0/1, 'parses': bool, 'none': bool, 'dropped': out-of-range numbers}."""
    nums = [int(x) for x in re.findall(r"\d+", reply or "")]
    valid = sorted({k for k in nums if 1 <= k <= n})
    dropped = sorted({k for k in nums if not 1 <= k <= n})
    none = re.search(r"\bnone\b", reply or "", re.IGNORECASE) is not None
    return {"G": valid, "g": 1 if valid else 0, "parses": bool(valid) or none, "none": none, "dropped": dropped}


def first_line(reply: str) -> str:
    for line in (reply or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


# --- normaliser (§2) ------------------------------------------------------------------------------------------
_QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
           "«": '"', "»": '"', "′": "'", "″": '"'}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-"}
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
     "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]) if i >= 1}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _ascii(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(_QUOTES.get(c, _DASHES.get(c, c)) for c in text)


def looks_like_time_or_date(text: str) -> bool:
    t = _ascii(text or "").lower()
    if re.search(r"\d{1,2}:\d{2}", t) or re.search(r"\d{1,4}[/-]\d{1,2}", t):
        return True
    return any(re.search(rf"\b{m}\b", t) for m in _MONTHS)


def normalise(text: str) -> str:
    """§2's normalisation: label, quotes, trailing period, ASCII, lower case, currency and thousands commas, number
    words (one..twenty, 'a dozen'), punctuation, articles."""
    t = _ascii(text or "").strip()
    t = re.sub(r"^\s*(answer|a)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().strip("\"'").strip()
    if t.endswith("."):
        t = t[:-1]
    t = t.strip("\"'").strip().lower()
    t = re.sub(r"[$€£¥₹]", "", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\ba dozen\b", "12", t)
    t = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: _NUMBER_WORDS[m.group(1)], t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)      # periods, except a decimal point
    t = re.sub(r"[^\w\s.]", " ", t)
    t = t.replace("_", " ")
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def is_not_determined(text: str) -> bool:
    t = f" {normalise(text)} "
    return any(f" {normalise(p)} " in t for p in NOT_DETERMINED)


def same_answer(d: str, candidate: str) -> int:
    """m: 1 if d and the candidate agree after normalisation (numeric rule when each side has exactly one number and
    neither is a time or date; else equality or whole-word containment), else 0; empty after normalisation -> 0."""
    a, b = normalise(d), normalise(candidate)
    if not a or not b:
        return 0
    na, nb = _NUMBER.findall(a), _NUMBER.findall(b)
    if len(na) == 1 and len(nb) == 1 and not looks_like_time_or_date(d) and not looks_like_time_or_date(candidate):
        return int(float(na[0]) == float(nb[0]))
    if a == b:
        return 1
    return int(f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def real_answer(d: str | None) -> bool:
    """d is a usable derivation: non-empty, not 'not determined', not a refusal by the §23 rule."""
    return bool(d and normalise(d)) and not is_not_determined(d) and not is_refusal_answer(d)


# --- scores ---------------------------------------------------------------------------------------------------

def e3_score(top: dict[str, float]) -> dict:
    """a and the label fields from E3's top-20 (the §36 v2 scorer)."""
    lp = ver.label_probs(top)
    return {"a": lp["s"], "label_mass": lp["label_mass"], "label_mass_tab": lp["label_mass_tab"], "argmax": lp["argmax"],
            "p_A": lp["p_A"], "p_B": lp["p_B"], "p_C": lp["p_C"]}


def elenchus_scores(g: int, d: str | None, a: float | None, candidate: str) -> dict:
    """e, m, e_mech for one row. e = 0 when g = 0 or d is not a real answer; else a (None if E3 is missing)."""
    m = same_answer(d, candidate) if g and d else 0
    e = 0.0 if (not g or not real_answer(d)) else a
    return {"g": g, "e": e, "m": m, "e_mech": g * m}


def evidence_hit(cited: list[int], memory_lines, evidence_lines) -> int:
    """1 if the cited set holds at least one annotated evidence item."""
    ev = set(evidence_lines)
    return int(any(memory_lines[k - 1] in ev for k in cited))


# --- planted thesis -------------------------------------------------------------------------------------------

def planted(gold: str, name_pool: list[str], key: str, prompt: str) -> str | None:
    """A wrong value of the gold's kind: number -> n + 2 (§36 substitute_value); name -> another question's
    name-kind gold not in the prompt. None when the gold is neither. Never equal to the gold after normalisation."""
    kind = ver.gold_kind(gold)
    if kind is None:
        return None
    value = ver.substitute_value(gold, kind, name_pool, key, prompt)
    if normalise(value) == normalise(gold) or same_answer(value, gold):
        raise AssertionError("planted value equals the gold")
    return value


# --- calibration and gates -------------------------------------------------------------------------------------

def calibration_pairs(rows: list[tuple[str, str, str]]) -> list[dict]:
    """rows: (question_id, prompt, candidate) sorted by question id; the first CALIBRATION_N give the identical pair
    (expect A) and the pair with the next different candidate (expect C)."""
    rows = sorted(rows)[:CALIBRATION_N]
    out = []
    for i, (q, prompt, cand) in enumerate(rows):
        other = next(c for j in range(1, len(rows)) for (_, _, c) in [rows[(i + j) % len(rows)]]
                     if normalise(c) != normalise(cand))
        out.append({"question_id": q, "kind": "identical", "x": cand, "y": cand, "expect": "A", "prompt": prompt})
        out.append({"question_id": q, "kind": "different", "x": cand, "y": other, "expect": "C", "prompt": prompt})
    return out


def calibration_pass(results: list[dict]) -> dict:
    """results: dicts with 'kind', 'argmax', 'expect'. Pass when each kind has >= 18 of 20 right."""
    out = {}
    for kind in ("identical", "different"):
        got = [r for r in results if r["kind"] == kind]
        out[kind] = {"n": len(got), "right": sum(r["argmax"] == r["expect"] for r in got)}
    out["pass"] = all(v["n"] == CALIBRATION_N and v["right"] >= CALIBRATION_PASS for v in (out["identical"], out["different"]))
    return out


def dev_gate(auroc_e: float | None, auroc_s: float | None) -> dict:
    """§5: pass iff the development AUROC(e) point >= 0.65 and >= AUROC(s, dev) + 0.05."""
    bar = None if auroc_s is None else auroc_s + GATE_MARGIN
    ok = auroc_e is not None and bar is not None and auroc_e >= GATE_AUROC and auroc_e >= bar
    return {"auroc_e": auroc_e, "auroc_s": auroc_s, "bar_abs": GATE_AUROC, "bar_vs_s": bar, "pass": bool(ok)}
