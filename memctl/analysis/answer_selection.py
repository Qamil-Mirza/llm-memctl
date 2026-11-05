"""§35: separating derivation from answer selection. Pure logic, no GPU.

Pre-registration: research/proposals/2026-10-13-exp35-answer-selection.md at commit 12c9168 (reviewer round 2, conditional ACCEPT). Six arms per in-view question:

- N, normal: the intact §23 prompt regenerated on this run's engine (the within-engine baseline for F).
- A, reproduce: chat text + the cached note + "Answer:", generate the phrase (the within-engine baseline for B).
- B, frozen note, abstention prohibited: chat text + note + newline + `B_INSTRUCTION` + newline + "Answer:".
- F, forced choice: the §23 prompt with both "unknown" clauses replaced (`F_REPLACEMENTS`), regenerated in full.
- Fcut: arm F's prompt on the evidence-cut prompt (`ablate_evidence`, as §34c). Ncut: the intact instruction on it.

The note is the cached output's text before its last "Answer:" marker, byte-exact (nothing is normalised). A and B
are completion-style continuations: the engine appends the continuation's ids after the chat text's ids, the way
§34b teacher-forced the own-mode readout. Outputs without the marker are excluded from A and B and counted.

Every pre-registered string is a named constant below so a reviewer's wording change is a one-line edit; the row
schema records a hash of the wording (`wording_sha256`) so rows written under an older wording are refused on resume.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from memctl.analysis.jacobian import ablate_evidence
from memctl.judge import LONGMEMEVAL_PROMPTS
from memctl.metrics import is_refusal

# --- pre-registered wording (§35 section 2) -------------------------------------------------------------------
ANSWER_MARKER = "Answer:"
B_INSTRUCTION = "Based on the note above, give the answer as a short phrase. Do not reply unknown."
# Arm F's textual transformation of the §23 prompt, applied in order; each original must occur exactly once.
# 1. The goal line's clause (the LongMemEval task goal; same words as agents/llm.py DEFAULT_INSTRUCTIONS).
# 2. The reasoning line's sentence (agents/llm.py REASONING_INSTRUCTIONS).
F_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("If the memory does not contain the answer, reply: unknown",
     "Always answer with your best guess, even if the memory does not contain the answer; never reply unknown"),
    ("Reply unknown only if the memory has nothing at all on the topic.",
     "Never reply unknown; if the memory has little on the topic, give your best guess anyway."),
)
ARMS = ("N", "A", "B", "F", "Fcut", "Ncut")
FROZEN_NOTE_ARMS = ("A", "B")
CUT_ARMS = ("Fcut", "Ncut")
# §23 gave the whole note + answer 256 new tokens; A and B only write the phrase after "Answer:".
MAX_NEW_TOKENS = {"N": 256, "A": 64, "B": 64, "F": 256, "Fcut": 256, "Ncut": 256}
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "greedy": True}
JUDGE_MAX_NEW_TOKENS = 4  # as §23 (memctl/judge.py)
UNDERPOWERED_BELOW = 15   # |S| on the confirmation folds
GATE_G = 0.8              # arm A reproduces the cached verdict and refusal flag on at least this share of marked rows
FLAGGED_TYPES = ("single-session-preference", "knowledge-update")

# Qwen2.5's chat template with one user message and the generation prompt (what `wrap_chat` renders for both
# readers). The stub engine uses it so a CPU run builds the same raw text the pod builds; a test checks it against
# the real tokenizer when one is cached locally.
QWEN25_CHAT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")


def wording() -> dict:
    return {"answer_marker": ANSWER_MARKER, "b_instruction": B_INSTRUCTION, "f_replacements": [list(p) for p in F_REPLACEMENTS],
            "max_new_tokens": MAX_NEW_TOKENS}


def wording_sha256() -> str:
    return sha256(json.dumps(wording(), sort_keys=True))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- the note and D1 ------------------------------------------------------------------------------------------

def split_note(output: str) -> tuple[str | None, str]:
    """(note, answer) around the last 'Answer:'; note is None when the output has no marker. Byte-exact."""
    i = output.rfind(ANSWER_MARKER)
    return (output[:i], output[i + len(ANSWER_MARKER):]) if i >= 0 else (None, output)


def _tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if t]


def d1(note: str | None, gold: str) -> bool | None:
    """Criterion D1: the normalised gold string, or every gold token, occurs in the note. None when there is no note
    or the gold has no alphanumeric token."""
    gold_tokens = _tokens(gold)
    if note is None or not gold_tokens:
        return None
    words = _tokens(note)
    return (" ".join(gold_tokens) in " ".join(words)) or all(t in words for t in gold_tokens)


def gold_in_note(record) -> bool | None:
    """D1 for a PromptRecord's cached output."""
    return d1(split_note(record.output)[0], record.gold)


# --- answers and refusals -------------------------------------------------------------------------------------

def final_answer(output: str) -> str | None:
    """memctl/agents/llm.py final_answer, copied (analysis may not import agents; a test keeps the two equal)."""
    marker = output.lower().rfind("answer:")
    if marker < 0:
        return None
    rest = output[marker + len("answer:"):].strip()
    return rest.splitlines()[0].strip() if rest else ""


def agent_answer(output: str) -> tuple[str, bool]:
    """(graded answer, marker present) exactly as LLMAgent.act with reasoning: the phrase after the last
    'answer:' (case-insensitive), 'unknown' when that phrase is empty, the whole output when there is no marker."""
    output = output.strip()
    parsed = final_answer(output)
    if parsed is None:
        return output, False
    return (parsed or "unknown"), True


def is_refusal_answer(answer: str) -> bool:
    """The §23 rule (runs/_pipelines/exp23_judge.py): the graded answer (the text after the last 'Answer:', or the
    whole output without a marker) starts with 'unknown'."""
    return (answer or "").strip().lower().startswith("unknown")


def is_refusal_list(answer: str) -> bool:
    """memctl/metrics.py REFUSALS substring rule; descriptive only, not the §35 rule."""
    return is_refusal(answer or "")


def outcome(correct: bool, answer: str) -> str:
    """Three-way label: judged correct, else a refusal (§23 rule), else wrong."""
    if correct:
        return "correct"
    return "refusal" if is_refusal_answer(answer) else "wrong"


# --- arm inputs -----------------------------------------------------------------------------------------------

def forced_prompt(prompt: str) -> str:
    """Arm F's prompt. Raises ValueError unless every replaced string occurs exactly once."""
    for old, new in F_REPLACEMENTS:
        count = prompt.count(old)
        if count != 1:
            raise ValueError(f"arm F: {old!r} occurs {count} times in the prompt (need exactly 1)")
        prompt = prompt.replace(old, new)
    return prompt


def frozen_continuation(arm: str, note: str) -> str:
    """Text appended after the chat text for A and B: the note byte-exact, then the arm's suffix (B: newline, the
    instruction, newline, 'Answer:', exactly as pre-registered; a note ending in a newline gives a blank line)."""
    if arm == "A":
        return note + ANSWER_MARKER
    if arm == "B":
        return note + "\n" + B_INSTRUCTION + "\n" + ANSWER_MARKER
    raise ValueError(f"not a frozen-note arm: {arm!r}")


@dataclass(frozen=True)
class ArmInput:
    arm: str
    question_id: str
    fold: int
    reader: str
    prompt: str            # the user message (chat-formatted by the engine)
    continuation: str      # appended after the chat text; "" for the regenerated arms
    max_new_tokens: int
    intact_prompt_sha256: str
    prompt_sha256: str     # sha256 of `prompt` (for F: the arm-F prompt hash)
    input_sha256: str      # sha256 of json [prompt, continuation]: what was generated from, engine-independent
    note: str | None = None  # A and B: the cached note (byte-exact)


def build_arm_input(record, arm: str) -> ArmInput | None:
    """The arm's input for one PromptRecord. None for A/B when the cached output has no marker. Raises ValueError
    when arm F's replacement or the evidence cut fails."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (known: {ARMS})")
    continuation = ""
    if arm in FROZEN_NOTE_ARMS:
        note, _ = split_note(record.output)
        if note is None:
            return None
        prompt, continuation = record.prompt, frozen_continuation(arm, note)
    elif arm == "N":
        prompt = record.prompt
    elif arm == "F":
        prompt = forced_prompt(record.prompt)
    else:
        cut = ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines)
        if record.evidence_lines and cut == record.prompt:
            raise ValueError("evidence cut changed nothing")
        prompt = forced_prompt(cut) if arm == "Fcut" else cut
    return ArmInput(arm, record.question_id, record.fold, record.reader, prompt, continuation, MAX_NEW_TOKENS[arm],
                    sha256(record.prompt), sha256(prompt), sha256(json.dumps([prompt, continuation])),
                    split_note(record.output)[0] if arm in FROZEN_NOTE_ARMS else None)


def full_output(arm_input: ArmInput, generated: str) -> str:
    """The text the answer is parsed from: the continuation (note + suffix) plus the generated text for A and B, the
    generated text alone (stripped, as the §23 server returned it) for the regenerated arms."""
    if arm_input.continuation:
        return arm_input.continuation + generated
    return generated.strip()


@dataclass
class GenerationRow:
    question_id: str
    fold: int
    reader: str
    arm: str
    engine: str            # e.g. "vllm-0.8.5", "hf-transformers-5.17.0", "stub"
    model: str
    sampling: dict         # SAMPLING plus max_new_tokens
    wording_sha256: str
    input_sha256: str
    prompt_sha256: str
    intact_prompt_sha256: str
    raw_input_sha256: str  # sha256 of the raw text: chat text + continuation
    n_input_tokens: int | None      # ids(chat text) + ids(continuation); None when the engine has no tokenizer
    n_teacher_forced: int | None    # ids(continuation): the teacher-forced tokens of A and B (0 otherwise)
    n_prefix_tokens: int | None     # A, B: ids(chat text) + ids(note), the note's own tokenisation
    max_new_tokens: int
    generated: str
    final_answer: str
    marker: bool
    refusal: bool          # the §23 rule
    refusal_list: bool     # the REFUSALS substring rule, descriptive

    def to_dict(self) -> dict:
        return asdict(self)


def make_row(arm_input: ArmInput, generated: str, engine: str, model: str, raw_text: str,
             counts: tuple[int | None, int | None, int | None] = (None, None, None)) -> GenerationRow:
    """`raw_text`: chat text + continuation as the engine built it; `counts`: (input, teacher-forced, prefix) tokens."""
    answer, marker = agent_answer(full_output(arm_input, generated))
    return GenerationRow(arm_input.question_id, arm_input.fold, arm_input.reader, arm_input.arm, engine, model,
                         {**SAMPLING, "max_new_tokens": arm_input.max_new_tokens}, wording_sha256(),
                         arm_input.input_sha256, arm_input.prompt_sha256, arm_input.intact_prompt_sha256,
                         sha256(raw_text), *counts, arm_input.max_new_tokens, generated, answer, marker,
                         is_refusal_answer(answer), is_refusal_list(answer))


class StaleRowError(RuntimeError):
    """A row on disk was generated from a different input, wording or engine than the current run would use."""


def pending_inputs(inputs: list[ArmInput], existing: list[dict], engine: str) -> list[ArmInput]:
    """Inputs with no row on disk yet. A row for the same question with a different input, wording or engine is
    an error (never mix two wordings or engines in one arm file)."""
    done = {}
    for row in existing:
        done[row["question_id"]] = row
    todo = []
    for item in inputs:
        row = done.get(item.question_id)
        if row is None:
            todo.append(item)
            continue
        if row["input_sha256"] != item.input_sha256 or row.get("wording_sha256") != wording_sha256():
            raise StaleRowError(f"{item.arm}/{item.question_id}: row on disk has a different input or wording; "
                                "move the old file aside")
        if row["engine"] != engine:
            raise StaleRowError(f"{item.arm}/{item.question_id}: row on disk came from engine {row['engine']!r}, "
                                f"this run uses {engine!r}")
    return todo


# --- population and judging -----------------------------------------------------------------------------------

def population(records_by_reader: dict[str, list]) -> tuple[list[str], list[str]]:
    """(question ids, prompt mismatches): in-view answerable questions with at least one evidence line and a §23
    judge label for every reader, whose prompt is identical across readers (the §34 pairing)."""
    keep = []
    by_reader = {r: {x.question_id: x for x in rows} for r, rows in records_by_reader.items()}
    sets = [{q for q, x in rows.items() if x.answerable and x.all_in_view and x.evidence_lines and x.correct is not None}
            for rows in by_reader.values()]
    mismatched = []
    for q in sorted(set.intersection(*sets)) if sets else []:
        prompts = {by_reader[r][q].prompt for r in by_reader}
        (keep if len(prompts) == 1 else mismatched).append(q)
    return keep, mismatched


def judge_question(instance: dict) -> str:
    """The question text the §23 judge saw (runs/_pipelines/exp23_judge.py)."""
    return f"(asked on {instance.get('question_date', '')}) {instance['question']}"


def judge_prompt(question: str, gold: str, answer: str, question_type: str) -> str:
    """memctl/judge.py Judge(style='longmemeval').prompt, without building a model."""
    template = LONGMEMEVAL_PROMPTS.get(question_type or "", LONGMEMEVAL_PROMPTS["default"])
    return template.format(question=question, gold=gold, answer=answer)


def judge_cache_key(prompt: str, settings: dict) -> str:
    """The TrackedLLM cache key (memctl/llm.py): sha256 of {prompt, max_new_tokens, **settings}, sorted json."""
    material = {"prompt": prompt, "max_new_tokens": JUDGE_MAX_NEW_TOKENS, **settings}
    return sha256(json.dumps(material, sort_keys=True))


def judge_settings(backend: str) -> dict:
    """The settings part of a judge cache key. §23's judge was backend 'openai' (vLLM 0.8.5 server)."""
    return {"backend": backend, "name": "qwen2.5-7b-instruct", "revision": None, "load_in_4bit": False, "temperature": 0.0}


def verdict_of(reply: str) -> bool:
    return reply.strip().lower().startswith("yes")
