"""§36: can the 3B judge evidence support for its own forced answers? Pure logic, no GPU.

Pre-registration: research/proposals/2026-10-15-exp36-verification.md (frozen at d2a67f2 + a3a3f9c; reviewer step
waived by the user). The candidates are §35's: arm B's `final_answer` (frozen note, "unknown" prohibited) and arm
Fcut's on the evidence-cut prompt; the labels are §35's judge verdicts. Nothing here generates an answer.

The verifier input (proposal §2) is one user message, chat-templated with the generation prompt, followed by the
literal `Answer: ` (completion style, as §35's frozen-note arms):

    <goal line>                                  the cached §23 prompt, byte-exact, with "\n" + the reasoning
                                                 instructions removed (they must occur exactly once)
    <blank line>
    ## Memory
    <memory lines>                               byte-exact
    <blank line>
    [A previous reader wrote this note: <note>]  V+n only
    Question: (asked on ...) <question>          the cached "## Current input" text, byte-exact
    Proposed answer: <candidate>
    <instruction: V, P2 or P3>

The support score is s = (p_A + 0.5 p_B) / (p_A + p_B + p_C) from the first generated token's top-20 logprobs, each
label summing its two spellings (`A` and ` A`); a spelling outside the top 20 counts 0. The label mass is stored.

Readings of the proposal that it leaves open (the simplest one is taken each time; reported to the Advisor):

1. The "## Current input" header and the trailing "Note:" line of the §23 prompt are dropped; the current-input text
   already begins with "Question: ", so it is kept byte-exact as the proposal's "Question: <question>" line. The goal
   line is kept byte-exact, so it still says "If the memory does not contain the answer, reply: unknown" (the proposal
   asks for the goal line byte-exact).
2. V+n shows the frozen note stripped of leading and trailing whitespace (it is shown inside a sentence, not
   continued; inner newlines kept) after "A previous reader wrote this note: ", then a newline. The note is the 3B's for every 3B candidate, also
   when the 7B verifies (the note belongs to the candidate). Paraphrases P2 and P3 are the V setting (no note) with
   the instruction and label texts reworded; the letters A/B/C keep their meaning so the score is the same formula.
3. A §35 candidate (arm B or Fcut) needs an "Answer:" marker and must not be a refusal (the §23 rule: starts with
   "unknown"). Rows whose candidate fails this get no verifier prompt; the report keeps them in the denominator as
   forced abstentions. Abs takes the answer §23 graded (the whole output when it has no marker, as §23 did) unless it
   is a refusal; the marker flag is stored in the row's `extra`.
4. Length control: each evidence line is replaced, at its exact span, by the memory line of another population
   question (non-evidence, not in this prompt, not containing this question's gold) closest in character length; ties
   broken by sha256(question id, line index, line). Cutting the replacements from the length-control prompt gives
   exactly the Cut-B prompt (tested).
5. Counterfactual: the gold as a whole must be a number (an integer written in digits) or a name (1-4 words, every
   word capitalised, the first a strict token by scripts/exp34_gold_kinds.py's rule, not a number word or yes/no).
   It must occur exactly once, verbatim and case-sensitive, with word boundaries (a number also not inside a longer
   number), in the content part of the evidence lines (after the "speaker, date: " label). Integers become n + 2;
   names become another population question's name-kind gold, picked by sha256 of the question id among those that
   differ from the gold and do not occur in the prompt. Occurrences outside the evidence are counted and stored.
6. C4 scores, after ids(chat text) + ids(note + "Answer:"), the suffix ids(" " + candidate) and ids(" unknown") by
   prompt logprobs (two passes); m = sum over candidate tokens - sum over unknown tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from memctl.analysis.answer_selection import ANSWER_MARKER, agent_answer, is_refusal_answer, split_note
from memctl.analysis.jacobian import INPUT_HEADER, MEMORY_HEADER, ablate_evidence, segment_prompt

# --- pre-registered wording (§36 section 2) -------------------------------------------------------------------
# memctl/agents/llm.py REASONING_INSTRUCTIONS, copied (analysis may not import agents; a test keeps the two equal).
REASONING_INSTRUCTIONS = (
    "Each memory line shows who said it and when. First write a brief note of at most 80 words: name the "
    "facts that bear on the question in a few words each (do not copy whole lines), then do any counting, "
    "adding or date arithmetic. Give dates as absolute dates (for example 7 May 2023), never relative ones "
    "such as 'yesterday' or 'last year'. If the question asks for advice or a suggestion, answer with a "
    "brief suggestion that uses what the memory says about the user. Reply unknown only if the memory has "
    "nothing at all on the topic. End with one line of the form 'Answer: <short phrase>'."
)
PROPOSED = "Proposed answer: "
QUESTION = "Question: "
NOTE_PREFIX = "A previous reader wrote this note: "
ANSWER_LITERAL = "Answer: "
INSTRUCTIONS = {
    "V": ("Does the memory above support the proposed answer? Reply with one letter.\n"
          "A = the memory states or directly implies this answer\n"
          "B = the memory is partly relevant but does not settle it\n"
          "C = the memory does not support it, or contradicts it"),
    # Paraphrases, fixed before launch (robustness only, never selected).
    "P2": ("Is the proposed answer backed by the memory above? Answer with a single letter.\n"
           "A = yes: the memory says this or clearly implies it\n"
           "B = in part: the memory touches on it but does not decide it\n"
           "C = no: the memory does not back it, or says something different"),
    "P3": ("Check the proposed answer against the memory above and reply with one letter only.\n"
           "A = supported: the memory contains this answer or directly entails it\n"
           "B = unclear: related information is there, but not enough to confirm it\n"
           "C = unsupported: nothing in the memory supports it, or the memory contradicts it"),
}
LABELS = ("A", "B", "C")
SPELLINGS = {label: (label, " " + label) for label in LABELS}
B_WEIGHT = 0.5
# v1: matched the stored token strings as they were (vLLM stored byte-level BPE strings, so ' A' never matched).
# v2: tokens decoded through the byte-level alphabet first; primary spellings unchanged ('A', ' A'); label_mass_tab
#     beside it also counts tab-prefixed labels ('\tA'), never used by a claim.
SCORER_VERSION = "v2-bytedecode"
MASS_FLOOR = 0.5         # a row is low-mass below this
MASS_SHARE = 0.1         # the readout is "not reliable" when more than this share of rows is low-mass
TOP_LOGPROBS = 20
CF_INT_OFFSET = 2
C4_UNKNOWN = " unknown"
R_STAR = 0.30            # Claim 4 risk cap
CLAIM1_MEANINGFUL = 0.65
CLAIM3_MEANINGFUL = 0.05
CF_DESCRIPTIVE_BELOW = 20

# Items. Label items give one first-token readout per (question, variant); C4 gives two prompt-logprob passes.
ITEMS = ("V", "Vn", "P2", "P3", "CutB", "CutF", "Abs", "CF", "LC", "V_own", "C4")
WORDING_OF_ITEM = {"V": "V", "Vn": "V", "P2": "P2", "P3": "P3", "CutB": "V", "CutF": "V", "Abs": "V", "CF": "V",
                   "LC": "V", "V_own": "V"}
CF_VARIANTS = ("orig@intact", "sub@intact", "orig@cf", "sub@cf")
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "greedy": True, "max_tokens": 1, "logprobs": TOP_LOGPROBS}
C4_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "greedy": True, "max_tokens": 1, "prompt_logprobs": 1}

# scripts/exp34_gold_kinds.py's rule, copied (a library module does not import scripts; a test keeps them equal).
OPENERS = {"the", "a", "an", "i", "you", "he", "she", "it", "they", "we", "my", "your", "his", "her", "their", "our",
           "in", "on", "at", "to", "for", "with", "of", "by", "from", "about", "around", "over", "there", "this",
           "that", "when", "after", "before", "during", "it's", "its", "as", "if", "one", "some", "both", "all"}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred",
                "first", "second", "third", "half", "twice"}


def token_kind(token: str) -> str:
    s = token.strip()
    low = s.lower()
    if not s:
        return "function"
    if re.fullmatch(r"[\d$€£%.,:/-]+", s):
        return "strict"
    if low in NUMBER_WORDS or low in {"yes", "no"}:
        return "strict"
    if s[0].isupper() and low not in OPENERS and len(s) > 1:
        return "strict"
    return "function"


def wording() -> dict:
    return {"instructions": INSTRUCTIONS, "proposed": PROPOSED, "note_prefix": NOTE_PREFIX,
            "answer_literal": ANSWER_LITERAL, "spellings": SPELLINGS, "b_weight": B_WEIGHT, "top_logprobs": TOP_LOGPROBS,
            "cf_int_offset": CF_INT_OFFSET, "c4_unknown": C4_UNKNOWN, "c4_prefix_marker": ANSWER_MARKER,
            "reasoning_removed": REASONING_INSTRUCTIONS, "lc_rule": "closest char length, sha256 tie-break, v1",
            "cf_rule": "digits n+2 / title-case name 1-4 words, exactly once in evidence content, v1"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def wording_sha256() -> str:
    return sha256(json.dumps(wording(), sort_keys=True))


def _h(*parts) -> int:
    return int(sha256("|".join(str(p) for p in parts))[:16], 16)


# --- the verifier prompt --------------------------------------------------------------------------------------

def strip_reasoning(prompt: str) -> str:
    """The §23 prompt without "\\n" + the reasoning instructions. Raises ValueError unless they occur exactly once."""
    target = "\n" + REASONING_INSTRUCTIONS
    count = prompt.count(target)
    if count != 1:
        raise ValueError(f"reasoning instructions occur {count} times in the prompt (need exactly 1)")
    return prompt.replace(target, "", 1)


def split_prompt(prompt: str) -> tuple[str, str]:
    """(goal line + blank line + memory block, current-input text) of a §23 prompt with the reasoning removed. The
    first part ends at the last memory line; the second is the text after "## Current input\\n" without the
    trailing "\\n\\nNote:" and must start with "Question: "."""
    if MEMORY_HEADER not in prompt or INPUT_HEADER not in prompt:
        raise ValueError("prompt has no memory or input header")
    head, rest = prompt.rsplit(INPUT_HEADER, 1)
    if not rest.startswith("\n"):
        raise ValueError("input header not followed by a newline")
    rest = rest[1:]
    for tail in ("\n\nNote:", "\n\nAnswer:"):
        if rest.endswith(tail):
            rest = rest[: -len(tail)]
            break
    else:
        raise ValueError("prompt does not end with the Note:/Answer: line")
    if not rest.startswith(QUESTION):
        raise ValueError(f"current input does not start with {QUESTION!r}")
    return head, rest


def verifier_prompt(prompt: str, candidate: str, wording_key: str = "V", note: str | None = None) -> str:
    """The verifier's user message for one §23 prompt (intact, cut, counterfactual or length-controlled)."""
    if wording_key not in INSTRUCTIONS:
        raise ValueError(f"unknown wording {wording_key!r}")
    head, question = split_prompt(strip_reasoning(prompt))
    note_line = "" if note is None else NOTE_PREFIX + note.strip() + "\n"
    return f"{head}\n\n{note_line}{question}\n{PROPOSED}{candidate}\n{INSTRUCTIONS[wording_key]}"


# --- scores ---------------------------------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """The byte-level BPE alphabet (GPT-2's bytes_to_unicode, used by the Qwen2.5 tokenizer): byte -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_DECODER = {c: b for b, c in _bytes_to_unicode().items()}


def decode_token(token: str) -> str:
    """A token string as text. vLLM 0.8.5 stored the top-20 as raw byte-level BPE strings ('ĠA' for ' A', 'ĉ' for a
    tab); those are mapped back to bytes and decoded as UTF-8 (a partial character becomes U+FFFD). A string with any
    character outside the byte-level alphabet (a space, say) is already text and is returned unchanged."""
    if not token or any(ch not in BYTE_DECODER for ch in token):
        return token
    return bytes(BYTE_DECODER[ch] for ch in token).decode("utf-8", errors="replace")


def normalize_top(top: dict[str, float]) -> dict[str, float]:
    """{decoded text: logprob}, tokens that decode to the same text merged by log-sum-exp."""
    out: dict[str, float] = {}
    for token, lp in top.items():
        text = decode_token(token)
        out[text] = lp if text not in out else max(out[text], lp) + math.log1p(math.exp(-abs(out[text] - lp)))
    return out


def label_probs(top: dict[str, float]) -> dict:
    """From {token: logprob} (the first generated position's top-k; raw byte-level BPE strings are decoded first):
    per-label probability (both spellings summed), per-spelling probability, mass, s, argmax label. s and argmax are
    None when the mass is 0."""
    top = normalize_top(top)
    spell = {sp: math.exp(top[sp]) if sp in top else 0.0 for label in LABELS for sp in SPELLINGS[label]}
    p = {label: sum(spell[sp] for sp in SPELLINGS[label]) for label in LABELS}
    mass = sum(p.values())
    tab = {label: math.exp(top["\t" + label]) if "\t" + label in top else 0.0 for label in LABELS}
    s = (p["A"] + B_WEIGHT * p["B"]) / mass if mass > 0 else None
    argmax = max(LABELS, key=lambda lab: (p[lab], -LABELS.index(lab))) if mass > 0 else None
    return {"p_A": p["A"], "p_B": p["B"], "p_C": p["C"], "p_spellings": spell, "label_mass": mass, "s": s,
            "argmax": argmax, "label_mass_tab": mass + sum(tab.values()), "scorer_version": SCORER_VERSION}


def reliability(masses: list[float]) -> dict:
    """The proposal's fallback rule: "not reliable" when the mass is below 0.5 on more than 0.1 of rows."""
    n = len(masses)
    low = sum(m < MASS_FLOOR for m in masses)
    share = low / n if n else None
    return {"n": n, "low_mass": low, "low_share": share, "reliable": n > 0 and share <= MASS_SHARE}


def confidence_margin(candidate_lps: list[float], unknown_lps: list[float]) -> float:
    return float(sum(candidate_lps) - sum(unknown_lps))


# --- candidates -----------------------------------------------------------------------------------------------

def candidate_of_row(row: dict | None) -> str | None:
    """A §35 generation row's candidate: its final answer when it has a marker and is not a refusal."""
    if row is None or not row.get("marker") or row.get("refusal"):
        return None
    answer = row.get("final_answer")
    return answer if answer and answer.strip() else None


def candidate_of_output(output: str) -> str | None:
    """A cached §23 output's candidate (Abs): the answer §23 graded (the phrase after the last marker, or the whole
    output when there is none) unless it is a refusal."""
    answer, _ = agent_answer(output)
    if is_refusal_answer(answer) or not answer.strip():
        return None
    return answer


# --- cut, length control, counterfactual ----------------------------------------------------------------------

def cut_prompt(prompt: str, memory_lines, evidence_lines) -> str:
    cut = ablate_evidence(prompt, tuple(memory_lines), tuple(evidence_lines))
    if evidence_lines and cut == prompt:
        raise ValueError("evidence cut changed nothing")
    return cut


def _line_spans(prompt: str, memory_lines, evidence_lines) -> list[tuple[int, int, str, bool]]:
    """(start, end, line, is_evidence) per memory line: the line's own characters (no separator)."""
    spans = segment_prompt(prompt, tuple(memory_lines), tuple(evidence_lines))
    evidence = set(evidence_lines)
    return [(start, start + len(line), line, line in evidence) for (_, start, _), line in zip(spans[1:-1], memory_lines)]


def length_control(prompt: str, memory_lines, evidence_lines, pool: list[str], key: str,
                   gold: str = "") -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """(prompt, new memory lines, replacement lines): every evidence line replaced at its exact span by the pool
    line closest in character length (not in this prompt, not containing the gold, not used twice)."""
    present = set(memory_lines)
    g = gold.strip().lower()
    usable = sorted({x for x in pool if x not in present and not (len(g) >= 3 and g in x.lower())})
    used: set[str] = set()
    pieces, cursor, new_lines, replacements = [], 0, [], []
    for index, (start, end, line, is_ev) in enumerate(_line_spans(prompt, memory_lines, evidence_lines)):
        if not is_ev:
            new_lines.append(line)
            continue
        choices = [x for x in usable if x not in used and "\n" not in x]
        if not choices:
            raise ValueError("length control: pool exhausted")
        best = min(abs(len(x) - len(line)) for x in choices)
        pick = min((x for x in choices if abs(len(x) - len(line)) == best), key=lambda x: (_h(key, index, x), x))
        used.add(pick)
        pieces += [prompt[cursor:start], pick]
        cursor = end
        new_lines.append(pick)
        replacements.append(pick)
    pieces.append(prompt[cursor:])
    out = "".join(pieces)
    if evidence_lines and out == prompt:
        raise ValueError("length control changed nothing")
    return out, tuple(new_lines), tuple(replacements)


def gold_kind(gold: str) -> str | None:
    """'number' (an integer in digits), 'name' (1-4 capitalised words, first a strict non-number token) or None."""
    g = gold.strip()
    if re.fullmatch(r"\d+", g):
        return "number"
    words = g.split()
    if not 1 <= len(words) <= 4:
        return None
    first = words[0]
    low = first.lower().strip(".,")
    if token_kind(first) != "strict" or low in NUMBER_WORDS or low in {"yes", "no"} or re.match(r"[\d$€£]", first):
        return None
    if not all(w[0].isupper() and any(c.isalpha() for c in w) for w in words):
        return None
    return "name"


def _pattern(gold: str, kind: str) -> re.Pattern:
    g = re.escape(gold.strip())
    if kind == "number":
        return re.compile(r"(?<![\w.,/:-])" + g + r"(?![\w]|[.,/:-]\d)")
    return re.compile(r"(?<!\w)" + g + r"(?!\w)")


def _content_start(line: str) -> int:
    i = line.find(": ")
    return i + 2 if i >= 0 else 0


def find_once_in_evidence(prompt: str, memory_lines, evidence_lines, gold: str) -> dict | None:
    """The gold's single verbatim occurrence inside the evidence content: {'start', 'end', 'kind', 'outside'}
    (absolute offsets in the prompt; 'outside' = occurrences in other memory lines), or None when the gold is not a
    number or a name or does not occur exactly once in the evidence."""
    kind = gold_kind(gold)
    if kind is None or not evidence_lines:
        return None
    pattern = _pattern(gold, kind)
    hits, outside = [], 0
    for start, end, line, is_ev in _line_spans(prompt, memory_lines, evidence_lines):
        c0 = start + _content_start(line)
        found = [m for m in pattern.finditer(prompt, c0, end)]
        if is_ev:
            hits += [(m.start(), m.end()) for m in found]
        else:
            outside += len(found)
    if len(hits) != 1:
        return None
    return {"start": hits[0][0], "end": hits[0][1], "kind": kind, "outside": outside}


def substitute_value(gold: str, kind: str, name_pool: list[str], key: str, prompt: str) -> str:
    g = gold.strip()
    if kind == "number":
        return str(int(g) + CF_INT_OFFSET)
    low = prompt.lower()
    options = sorted({x.strip() for x in name_pool
                      if gold_kind(x) == "name" and x.strip().lower() != g.lower() and x.strip().lower() not in low})
    if not options:
        raise ValueError("counterfactual: no substitute name")
    return options[_h("cf", key) % len(options)]


def counterfactual(prompt: str, memory_lines, evidence_lines, gold: str, name_pool: list[str], key: str) -> dict | None:
    """The counterfactual prompt, or None when the row does not qualify. Asserts the substitution by exact span."""
    hit = find_once_in_evidence(prompt, memory_lines, evidence_lines, gold)
    if hit is None:
        return None
    start, end = hit["start"], hit["end"]
    original = prompt[start:end]
    if original != gold.strip():
        raise AssertionError("counterfactual: span is not the gold")
    sub = substitute_value(gold, hit["kind"], name_pool, key, prompt)
    new = prompt[:start] + sub + prompt[end:]
    assert new[:start] == prompt[:start] and new[start + len(sub):] == prompt[end:] and new[start:start + len(sub)] == sub
    lines, ev = [], []
    for s, e, line, is_ev in _line_spans(prompt, memory_lines, evidence_lines):
        if s <= start and end <= e:
            line = line[: start - s] + sub + line[end - s:]
        lines.append(line)
        if is_ev:
            ev.append(line)
    return {"prompt": new, "memory_lines": tuple(lines), "evidence_lines": tuple(ev), "original": original,
            "substitute": sub, "kind": hit["kind"], "start": start, "end": end, "outside": hit["outside"]}


# --- inputs and rows ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifyInput:
    item: str
    question_id: str
    variant: str            # "" except CF (CF_VARIANTS)
    fold: int
    prompt: str             # the verifier's user message (C4: the §23 prompt)
    continuation: str       # ANSWER_LITERAL (C4: note + "Answer:")
    candidate: str
    candidate_source: str   # e.g. "qwen3b/B", "qwen3b/Fcut", "qwen3b/cached", "gold", "substitute"
    intact_prompt_sha256: str
    suffixes: tuple[str, ...] = ()   # C4: (" " + candidate, " unknown")
    extra: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def kind(self) -> str:
        return "c4" if self.item == "C4" else "label"

    @property
    def prompt_sha256(self) -> str:
        return sha256(self.prompt)

    @property
    def input_sha256(self) -> str:
        return sha256(json.dumps([self.prompt, self.continuation, list(self.suffixes)]))


def label_input(item: str, record, prompt: str, candidate: str, source: str, variant: str = "",
                note: str | None = None, extra: dict | None = None) -> VerifyInput:
    text = verifier_prompt(prompt, candidate, WORDING_OF_ITEM[item], note)
    return VerifyInput(item, record.question_id, variant, record.fold, text, ANSWER_LITERAL, candidate, source,
                       sha256(record.prompt), (), extra or {})


def c4_input(record, candidate: str, source: str) -> VerifyInput | None:
    """C4 on the reader's own §35 arm-A prefix (chat text + its cached note + "Answer:"). None without a note."""
    note, _ = split_note(record.output)
    if note is None:
        return None
    return VerifyInput("C4", record.question_id, "", record.fold, record.prompt, note + ANSWER_MARKER, candidate, source,
                       sha256(record.prompt), (" " + candidate, C4_UNKNOWN))


def make_row(item: VerifyInput, reader: str, engine: str, model: str, raw_text: str, n_input_tokens: int | None,
             seconds: float, result: dict) -> dict:
    """One output row. `result`: for label items {'top': {token: logprob}}; for C4 {'suffix_logprobs': [[...], [...]],
    'n_prefix_tokens': int}."""
    row = {"question_id": item.question_id, "fold": item.fold, "reader": reader, "item": item.item,
           "variant": item.variant, "engine": engine, "model": model,
           "sampling": C4_SAMPLING if item.kind == "c4" else SAMPLING, "wording_sha256": wording_sha256(),
           "input_sha256": item.input_sha256, "prompt_sha256": item.prompt_sha256,
           "intact_prompt_sha256": item.intact_prompt_sha256, "raw_input_sha256": sha256(raw_text),
           "n_input_tokens": n_input_tokens, "seconds": seconds, "candidate": item.candidate,
           "candidate_source": item.candidate_source, "extra": item.extra}
    if item.kind == "c4":
        cand, unk = result["suffix_logprobs"]
        row.update({"lp_candidate": cand, "lp_unknown": unk, "n_prefix_tokens": result.get("n_prefix_tokens"),
                    "m": confidence_margin(cand, unk)})
    else:
        top = result["top"]
        row.update(label_probs(top))
        row["top1"] = decode_token(max(top, key=top.get)) if top else None
        row["top_logprobs"] = sorted(([t, round(v, 4)] for t, v in top.items()), key=lambda x: -x[1])
    return row


def row_key(row) -> tuple[str, str]:
    if isinstance(row, VerifyInput):
        return row.question_id, row.variant
    return row["question_id"], row.get("variant", "")


class StaleRowError(RuntimeError):
    """A row on disk was produced from a different input, wording or engine than the current run would use."""


def pending_inputs(inputs: list[VerifyInput], existing: list[dict], engine: str) -> list[VerifyInput]:
    """Inputs with no row yet. A row for the same (question, variant) with another input, wording or engine is an
    error (never mix two wordings or engines in one item file)."""
    done = {row_key(r): r for r in existing}
    todo = []
    for item in inputs:
        row = done.get(row_key(item))
        if row is None:
            todo.append(item)
            continue
        if row["input_sha256"] != item.input_sha256 or row.get("wording_sha256") != wording_sha256():
            raise StaleRowError(f"{item.item}/{item.question_id}/{item.variant}: row on disk has a different input or "
                                "wording; move the old file aside")
        if row["engine"] != engine:
            raise StaleRowError(f"{item.item}/{item.question_id}: row on disk came from engine {row['engine']!r}, "
                                f"this run uses {engine!r}")
    return todo


def to_dict(item: VerifyInput) -> dict:
    return asdict(item)


# --- statistics (paired bootstrap over questions; every variant of a question in one unit) --------------------

class Bootstrap:
    """Multinomial resampling of question ids: `counts[b, j]` is how often unit j is drawn in resample b. Every
    statistic is a weighted one, a row weighted by its question's count, so all rows of a question move together."""

    def __init__(self, units, samples: int = 10000, seed: int = 0) -> None:
        self.units = sorted(set(units))
        self.col = {u: j for j, u in enumerate(self.units)}
        n = len(self.units)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(n, [1.0 / n] * n, size=samples).astype(float) if n else np.zeros((samples, 0))

    def weights(self, qids) -> np.ndarray:
        return self.counts[:, [self.col[q] for q in qids]] if len(qids) else np.zeros((self.counts.shape[0], 0))


def _ci(draws: np.ndarray) -> tuple[float | None, float | None]:
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None, None
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _stat(point, draws) -> dict:
    low, high = _ci(draws)
    return {"estimate": None if point is None or not np.isfinite(point) else float(point), "ci_low": low, "ci_high": high}


def auroc_stat(pos: list[tuple[str, float]], neg: list[tuple[str, float]], boot: Bootstrap) -> dict:
    """AUROC (ties 0.5) of positives over negatives, (qid, score) rows, with a cluster-bootstrap interval."""
    out = {"n_pos": len(pos), "n_neg": len(neg)}
    if not pos or not neg:
        return {**out, "estimate": None, "ci_low": None, "ci_high": None}
    ps = np.array([s for _, s in pos], float)
    ns = np.array([s for _, s in neg], float)
    cmp = (ps[:, None] > ns[None, :]) + 0.5 * (ps[:, None] == ns[None, :])
    point = cmp.mean()
    wp, wn = boot.weights([q for q, _ in pos]), boot.weights([q for q, _ in neg])
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.einsum("bi,ij,bj->b", wp, cmp, wn) / (wp.sum(1) * wn.sum(1))
    return {**out, **_stat(point, draws)}


def average_precision(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """Average precision (PR-AUC) with tied scores taken as one threshold."""
    if not pos_scores:
        return None
    rows = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores], key=lambda r: -r[0])
    ap, tp, seen, i = 0.0, 0, 0, 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        group_tp = sum(r[1] for r in rows[i:j])
        tp += group_tp
        seen += j - i
        ap += group_tp / len(pos_scores) * (tp / seen)
        i = j
    return ap


def ratio_stat(rows: list[tuple[str, float, float]], boot: Bootstrap) -> dict:
    """sum(num) / sum(den) over (qid, num, den) rows."""
    if not rows or sum(r[2] for r in rows) == 0:
        return {"n": len(rows), "estimate": None, "ci_low": None, "ci_high": None}
    num = np.array([r[1] for r in rows], float)
    den = np.array([r[2] for r in rows], float)
    w = boot.weights([r[0] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ num) / (w @ den)
    return {"n": len(rows), "num": float(num.sum()), "den": float(den.sum()), **_stat(num.sum() / den.sum(), draws)}


def ratio_difference(rows: list[tuple[str, float, float, float, float]], boot: Bootstrap) -> dict:
    """sum(n1)/sum(d1) - sum(n2)/sum(d2) over (qid, n1, d1, n2, d2) rows, paired by question."""
    a = np.array([r[1:] for r in rows], float).reshape(-1, 4)
    if not rows or a[:, 1].sum() == 0 or a[:, 3].sum() == 0:
        return {"n": len(rows), "estimate": None, "ci_low": None, "ci_high": None}
    w = boot.weights([r[0] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ a[:, 0]) / (w @ a[:, 1]) - (w @ a[:, 2]) / (w @ a[:, 3])
    point = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    return {"n": len(rows), **_stat(point, draws)}


def mean_stat(rows: list[tuple[str, float]], boot: Bootstrap) -> dict:
    return ratio_stat([(q, v, 1.0) for q, v in rows], boot) | {"n": len(rows)}


# --- decisions ------------------------------------------------------------------------------------------------

def top_k(scores: dict[str, float | None], k: int) -> set[str]:
    """The k rows with the highest score (None lowest), ties broken by question id."""
    order = sorted(scores, key=lambda q: (-(scores[q] if scores[q] is not None else -math.inf), q))
    return set(order[:k])


def emit_at(scores: dict[str, float | None], tau: float | None) -> set[str]:
    if tau is None:
        return set()
    return {q for q, s in scores.items() if s is not None and s >= tau}


def fit_tau(scores: dict[str, float | None], wrong: dict[str, bool], r_star: float = R_STAR) -> float | None:
    """The lowest threshold whose risk (wrong share among emitted rows, s >= tau) is at most r_star; None if none."""
    for tau in sorted({s for s in scores.values() if s is not None}):
        emitted = [q for q, s in scores.items() if s is not None and s >= tau]
        if emitted and sum(wrong[q] for q in emitted) / len(emitted) <= r_star:
            return tau
    return None


def fixed_sequence(claim1_verdict: str, claim2: dict, claim3: dict) -> tuple[str, str]:
    """Verdicts of Claims 2 and 3 under fixed-sequence testing: each is tested only if the previous one is SHOWN; a
    risk difference is SHOWN when its upper bound is below 0."""
    def shown(stat):
        return stat.get("ci_high") is not None and stat["ci_high"] < 0
    if claim1_verdict != "SHOWN":
        return "not tested (Claim 1 not SHOWN)", "not tested (Claim 1 not SHOWN)"
    v2 = "SHOWN" if shown(claim2) else "NOT SHOWN"
    if v2 != "SHOWN":
        return v2, "not tested (Claim 2 not SHOWN)"
    return v2, "SHOWN" if shown(claim3) else "NOT SHOWN"


def rescore_row(row: dict) -> dict:
    """A label row with p_A/p_B/p_C, spellings, mass, s, argmax and top1 recomputed from its stored top_logprobs
    (decoded); C4 rows and rows without top_logprobs are returned unchanged."""
    if row.get("item") == "C4" or not row.get("top_logprobs"):
        return row
    top = {t: lp for t, lp in row["top_logprobs"]}
    new = {**row, **label_probs(top)}
    new["top1"] = decode_token(max(top, key=top.get))
    new["rescored"] = True
    new["scorer_version"] = SCORER_VERSION
    return new


# --- wording probe (coordinator request 2026-10-15; NOT the frozen wording) ------------------------------------
# On dev V the 3B put label mass >= 0.5 on only 0.11 of rows: it answered the question (the prompt keeps the §23 goal
# line and ends with "Answer: "). These variants are tested on format compliance only (no correctness-linked number).
# They are chat messages; the engine applies the chat template with the generation prompt and appends nothing.
PROBE_LABEL_LINES = ("A = the memory states or directly implies this answer\n"
                     "B = the memory is partly relevant but does not settle it\n"
                     "C = the memory does not support it, or contradicts it")
PROBE_REPLY = "Reply with exactly one letter: A, B or C."
PROBE_TASK = ("You will check whether a proposed answer is supported by a memory log. Below are the memory, a question "
              "and a proposed answer. Does the memory support the proposed answer?\n" + PROBE_LABEL_LINES)
PROBE_SYSTEM = "You are a strict fact checker. You reply with a single letter."
PROBE_WORDINGS = ("W1", "W2", "W3")


def memory_block(prompt: str) -> str:
    """'## Memory' and the memory lines of a §23 prompt, byte-exact (no goal line, no reasoning instructions)."""
    head, _ = split_prompt(strip_reasoning(prompt))
    return head[head.index(MEMORY_HEADER):]


def probe_messages(prompt: str, candidate: str, key: str) -> list[dict]:
    """W1: memory block + question + proposed answer + the A/B/C lines + the reply line. W2: W1 with the task
    sentence and the A/B/C lines once before the memory. W3: W2 with a system message."""
    _, question = split_prompt(strip_reasoning(prompt))
    body = f"{memory_block(prompt)}\n\n{question}\n{PROPOSED}{candidate}\n{PROBE_LABEL_LINES}\n{PROBE_REPLY}"
    if key == "W1":
        return [{"role": "user", "content": body}]
    if key == "W2":
        return [{"role": "user", "content": f"{PROBE_TASK}\n\n{body}"}]
    if key == "W3":
        return [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": f"{PROBE_TASK}\n\n{body}"}]
    raise ValueError(f"unknown probe wording {key!r}")


def probe_wording_sha256(key: str) -> str:
    material = {"key": key, "label_lines": PROBE_LABEL_LINES, "reply": PROBE_REPLY, "proposed": PROPOSED,
                "task": PROBE_TASK if key in ("W2", "W3") else None, "system": PROBE_SYSTEM if key == "W3" else None,
                "assistant_prefix": ""}
    return sha256(json.dumps(material, sort_keys=True))
