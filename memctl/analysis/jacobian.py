"""The Jacobian lens (§33): how much a reader's output depends on each region of its prompt.

Pure functions over strings, token offsets and numpy arrays, plus one gradient pass that needs torch
and transformers (imported inside the functions that use them). Regions: the chat template, the
instruction, the evidence memory lines, the distractor memory lines and the question.
"""

from __future__ import annotations

import numpy as np

REGIONS = ("template", "instruction", "evidence", "distractor", "question")
MEMORY_HEADER = "## Memory\n"
INPUT_HEADER = "\n\n## Current input"


def segment_prompt(prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Char spans (region, start, end) tiling the prompt. Memory lines are found by exact search in order,
    so a line whose content contains a newline stays one line. Gaps (separators) join the span before them."""
    header = prompt.index(MEMORY_HEADER) + len(MEMORY_HEADER)
    spans: list[tuple[str, int, int]] = [("instruction", 0, header)]
    cursor = header
    evidence = set(evidence_lines)
    for line in memory_lines:
        start = prompt.find(line, cursor)
        if start < 0:
            raise ValueError(f"memory line not found in prompt: {line[:60]!r}")
        if start > cursor:  # separator before this line belongs to the previous span
            region, s, _ = spans[-1]
            spans[-1] = (region, s, start)
        end = start + len(line)
        spans.append(("evidence" if line in evidence else "distractor", start, end))
        cursor = end
    question = prompt.index(INPUT_HEADER, cursor)
    if question > cursor:
        region, s, _ = spans[-1]
        spans[-1] = (region, s, question)
    spans.append(("question", question, len(prompt)))
    return spans


def token_regions(offsets: list[tuple[int, int]], prompt_start: int, prompt_len: int,
                  segments: list[tuple[str, int, int]]) -> list[str]:
    """A region per token. A token is placed by its start char; tokens starting outside the prompt are template."""
    regions = []
    for start, end in offsets:
        if end <= prompt_start or start >= prompt_start + prompt_len:
            regions.append("template")
            continue
        char = max(start, prompt_start) - prompt_start
        region = next((name for name, s, e in segments if s <= char < e), None)
        regions.append(region or "template")
    return regions
