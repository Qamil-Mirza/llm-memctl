"""§34: reading a Jacobian lens at chosen positions of a cached reader prompt.

The read sequence is built by id concatenation, never by re-tokenising a joined string:
ids(chat text) + ids("Answer:") + the gold's leading whitespace-only tokens (teacher-forced; Qwen splits
" 4 hours" into a bare-space token and "4", so the digit is predicted one position after "Answer:").
Positions: P1 the last token (it predicts the gold's first content token), P0 the last token of the
generation prompt, E the last token of each gold evidence line, Q the last token of the question line, and
P1_raw the token right after "Answer:" (equal to P1 when the gold starts with a content token). Tracked
tokens: the gold answer's first content token (leading whitespace-only tokens skipped, as the §33 margin
does), " unknown" and "unknown". ``jlens`` and ``torch`` are imported only inside the functions that need them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from memctl.analysis.jacobian import _ids, _margin_length, segment_prompt, unknown_id, wrap_chat

ANSWER_PREFIX = "Answer:"
POSITION_NAMES = ("P1", "P0", "E", "Q", "P1_raw")   # array layout order; P1_raw last
QUESTION_MARK = "Question:"
NOTE_SUFFIX = "\n\nNote:"


def file_sha256(path) -> str:
    """sha256 of a file's bytes, streamed (a 7B lens is ~700 MB). Reads and evals key on it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Readout:
    text: str                      # for display only; the authoritative sequence is input_ids
    input_ids: tuple[int, ...]
    positions: dict[str, tuple[int, ...]]
    tracked: dict[str, int]
    gold_token_text: str
    chat_text: str = ""
    n_chat: int = 0                # input_ids[:n_chat] are the chat text's ids
    n_leading_ws: int = 0          # teacher-forced whitespace-only gold tokens at the end of input_ids


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
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    chat_ids = list(encoded["input_ids"])
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    answer = _ids(tokenizer, " " + gold)
    leading = answer[:_margin_length(tokenizer, answer) - 1]            # whitespace-only gold tokens, usually [Ġ] or []
    ids = chat_ids + _ids(tokenizer, ANSWER_PREFIX) + leading
    text = chat + ANSWER_PREFIX + tokenizer.decode(leading)
    prompt_start = chat.index(prompt)
    p0 = _token_at_char(offsets, len(chat) - 1)          # the token holding the chat text's last char, robust to merges
    p1 = len(ids) - 1                                    # predicts the gold's first content token
    p1_raw = p1 - len(leading)                           # the token right after "Answer:"
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
    positions = {"P1": (p1,), "P0": (p0,), "E": tuple(evidence_positions), "Q": (q_position,), "P1_raw": (p1_raw,)}
    return Readout(text, tuple(ids), positions, tracked, gold_text, chat_text=chat, n_chat=len(chat_ids),
                   n_leading_ws=len(leading))


OUTCOMES = ("A", "B", "C")


def ranks_of(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """1 + the number of logits strictly greater than each tracked token's, per position. A NaN would compare
    false everywhere and give a silently wrong rank, so non-finite logits are refused."""
    chosen = logits[:, list(token_ids)]                       # [n_pos, n_tracked]
    if not np.isfinite(logits).all():
        raise ValueError(f"non-finite logits ({int((~np.isfinite(logits)).sum())} entries); ranks would be meaningless")
    return (logits[:, None, :] > chosen[:, :, None]).sum(-1) + 1


def _layer_logits(lens_logits: dict[int, np.ndarray], model_logits: np.ndarray, n_layers: int):
    final = n_layers - 1
    layers = sorted(l for l in lens_logits if l != final) + [final]
    return layers, [np.asarray(lens_logits[l]) if l != final else np.asarray(model_logits) for l in layers]


def rank_table(lens_logits, model_logits, n_layers: int, token_ids: Sequence[int]) -> tuple[np.ndarray, list[int]]:
    layers, per_layer = _layer_logits(lens_logits, model_logits, n_layers)
    return np.stack([ranks_of(lg, token_ids) for lg in per_layer]).astype(np.int32), layers


def topk_table(lens_logits, model_logits, n_layers: int, k: int = 10):
    layers, per_layer = _layer_logits(lens_logits, model_logits, n_layers)
    ids, values = [], []
    for lg in per_layer:
        k = min(k, lg.shape[-1])
        part = np.argpartition(-lg, k - 1, axis=-1)[:, :k]          # the k largest, unordered (O(vocab))
        order = np.argsort(-np.take_along_axis(lg, part, axis=-1), axis=-1, kind="stable")
        top = np.take_along_axis(part, order, axis=-1)
        ids.append(top)
        values.append(np.take_along_axis(lg, top, axis=-1))
    return np.stack(ids).astype(np.int32), np.stack(values).astype(np.float32), layers


def best_rank(ranks_1d: np.ndarray) -> int:
    return int(np.min(ranks_1d))


def emergence_depth(ranks_1d: np.ndarray, layers: Sequence[int], n_layers: int, k: int) -> float | None:
    hits = [layer for layer, rank in zip(layers, ranks_1d) if rank <= k]
    return hits[0] / n_layers if hits else None


def outcome(ranks_1d: np.ndarray, k: int) -> str:
    if ranks_1d[-1] <= k:
        return "C"
    return "B" if np.min(ranks_1d) <= k else "A"


def _first_top1_depth(ranks_1d: np.ndarray, layers: Sequence[int], n_layers: int) -> float | None:
    return emergence_depth(ranks_1d, layers, n_layers, k=1)


def _one(ranks: np.ndarray, layers, n_layers, tracked_names, pos_idx: int, k: int) -> dict:
    out = {}
    for t_idx, name in enumerate(tracked_names):
        r = ranks[:, pos_idx, t_idx]
        entry = {"best_rank": best_rank(r), "final_rank": int(r[-1])}
        if name == "gold":
            entry.update({"emergence_depth": emergence_depth(r, layers, n_layers, k), "outcome": outcome(r, k)})
        else:
            entry["first_top1_depth"] = _first_top1_depth(r, layers, n_layers)
        out[name] = entry
    return out


def question_measures(ranks, layers, n_layers, tracked_names, position_names, k: int = 10) -> dict:
    out = {"k": k, "layers": list(layers)}
    for name, indices in position_names.items():
        entries = [_one(ranks, layers, n_layers, tracked_names, i, k) for i in indices]
        out[name] = entries if name == "E" else (entries[0] if entries else None)
    return out
