"""§34: reading a Jacobian lens at chosen positions of a cached reader prompt.

Positions (indices into the chat text plus ``Answer:``): P1 the last token (the first answer token is
predicted here), P0 the last token of the generation prompt, E the last token of each gold evidence
line, Q the last token of the question line. Tracked tokens: the gold answer's first content token
(leading whitespace-only tokens skipped, as the §33 margin does), " unknown" and "unknown".
The rank/outcome arithmetic over lens logits lives further down (Task 2). ``jlens`` and ``torch`` are
imported only inside the functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memctl.analysis.jacobian import _ids, _margin_length, segment_prompt, unknown_id, wrap_chat

ANSWER_PREFIX = "Answer:"
POSITION_NAMES = ("P1", "P0", "E", "Q")
QUESTION_MARK = "Question:"
NOTE_SUFFIX = "\n\nNote:"


@dataclass(frozen=True)
class Readout:
    text: str
    input_ids: tuple[int, ...]
    positions: dict[str, tuple[int, ...]]
    tracked: dict[str, int]
    gold_token_text: str


def tracked_token_ids(tokenizer, gold: str) -> tuple[dict[str, int], str]:
    answer = _ids(tokenizer, " " + gold)
    first = answer[_margin_length(tokenizer, answer) - 1]
    ids = {"gold": first, "unknown_sp": unknown_id(tokenizer), "unknown": _ids(tokenizer, "unknown")[0]}
    return ids, tokenizer.decode([first])


def _token_at_char(offsets: list[tuple[int, int]], char: int) -> int:
    """Index of the token whose span contains ``char`` (the last token starting at or before it)."""
    best = 0
    for index, (start, end) in enumerate(offsets):
        if start <= char:
            best = index
        if start > char:
            break
    return best


def readout_for(tokenizer, prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...], gold: str) -> Readout:
    missing = [l for l in evidence_lines if l not in memory_lines]
    if missing:
        raise ValueError(f"evidence line not among the memory lines: {missing[0][:60]!r}")
    chat = wrap_chat(tokenizer, prompt)
    text = chat + ANSWER_PREFIX
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    prompt_start = text.index(prompt)
    p0 = _token_at_char(offsets, len(chat) - 1)          # the token holding the chat text's last char, robust to merges
    p1 = len(ids) - 1
    evidence_positions = []
    for region, start, end in segment_prompt(prompt, memory_lines, evidence_lines):
        if region != "evidence":
            continue
        line = next(l for l in evidence_lines if prompt.startswith(l, start))
        evidence_positions.append(_token_at_char(offsets, prompt_start + start + len(line) - 1))
    q_start = prompt.rindex(QUESTION_MARK)
    q_end = prompt.index(NOTE_SUFFIX, q_start) if NOTE_SUFFIX in prompt[q_start:] else len(prompt)
    q_position = _token_at_char(offsets, prompt_start + q_end - 1)
    tracked, gold_text = tracked_token_ids(tokenizer, gold)
    return Readout(text, tuple(ids), {"P1": (p1,), "P0": (p0,), "E": tuple(evidence_positions), "Q": (q_position,)},
                   tracked, gold_text)
