"""§55 prompt arms (pure): P0 = the frozen reader prompt (7b5fc30), byte-identical; P1 = one consistent abstention rule;
P2 = both "unknown" sentences removed (the §51 prompt_b3 text). Built by exact string replacement; a check asserts the
only difference from P0 is the two sentences. Spec: research/proposals/2026-11-08-exp55-clean-prompt.md rev 0."""
from __future__ import annotations

import difflib

TASK0 = " If the memory does not contain the answer, reply: unknown"
NOTE0 = " Reply unknown only if the memory has nothing at all on the topic."
TASK1 = " If the memory does not state the answer, reply: unknown"
NOTE1 = " Reply unknown only if the memory does not state the answer."
ARMS = ("P0", "P1", "P2")
REPL = {"P0": (TASK0, NOTE0), "P1": (TASK1, NOTE1), "P2": ("", "")}


def build(prompt: str, arm: str) -> str:
    for s in (TASK0, NOTE0):
        if prompt.count(s) != 1:
            raise ValueError(f"{s!r} occurs {prompt.count(s)} times, want 1")
    t, n = REPL[arm]
    out = prompt.replace(TASK0, t).replace(NOTE0, n)
    check(prompt, out, arm)
    return out


def check(p0: str, px: str, arm: str) -> None:
    """The only edits are the two sentence replacements, in place."""
    t, n = REPL[arm]
    i, j = p0.index(TASK0), p0.index(NOTE0)
    assert i < j
    want = p0[:i] + t + p0[i + len(TASK0):j] + n + p0[j + len(NOTE0):]
    assert px == want, arm
    sm = difflib.SequenceMatcher(None, p0, px, autojunk=False)
    changed = [(tag, p0[a:b], px[c:d]) for tag, a, b, c, d in sm.get_opcodes() if tag != "equal"]
    removed = "".join(x[1] for x in changed)
    assert len(removed) <= len(TASK0) + len(NOTE0), (arm, changed)
