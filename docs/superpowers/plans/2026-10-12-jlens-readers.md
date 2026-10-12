# J-lens on small readers (§34) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit one Jacobian lens per Qwen 2.5 reader (3B, 7B), read the lens at the answer position of every cached §23 prompt at every layer, track the gold answer's first content token and `unknown`, classify each question as never-present / present-then-lost / present-to-end, and report the pre-registered claims with a development/confirmation fold guard.

**Architecture:** A pure analysis module turns a prompt record into readout positions and tracked token ids, and turns per-layer lens logits into ranks, emergence depth and outcomes. Three scripts drive the vendored `jlens` package: fit a lens on a WikiText corpus under a time box, read every record resumably (J-lens and logit-lens baseline from the same activations), and score the package's own lens-quality evaluation sets. A report script computes the lens gate, the claims, the decomposition and the descriptives. A pod script runs fit, read, eval, and then the §33 secondary.

**Tech Stack:** Python 3.12 in `.venv` (torch 2.14 CPU, transformers 5.17, numpy, scipy, pyarrow), `jlens` 0.1.0 installed editable from `external/jacobian-lens` (pinned `581d398`), pytest. Reuses `memctl/jacobian_data.py`, `memctl/analysis/jacobian.py` (`wrap_chat`, `segment_prompt`, `_ids`, `_margin_length`, `unknown_id`) and `memctl/analysis/jacobian_stats.py` from §33.

**Spec:** `docs/superpowers/specs/2026-10-12-jlens-readers-design.md`

## Global Constraints

- Branch `jacobian-lens` (shared with §33), never `main`. Commit after every task; no session-link lines in commit messages (user rule).
- `.venv/bin/python` / `.venv/bin/pytest`. Tests go in a new file `tests/test_jlens.py`. The §33 `tiny` fixture pattern (Qwen2.5-0.5B tokenizer + tiny random `Qwen2ForCausalLM`) is reused; copy it into the new file rather than importing across test modules.
- `memctl/analysis/` must not import `memctl.harness`, `memctl.controllers`, `memctl.envs`, `memctl.agents`, `memctl.memory`; `jlens`, `torch`, `transformers` only lazily inside functions. `tests/test_import_boundaries.py` has two pre-existing failures that are not ours.
- Inputs only: §23 cache and traces through `load_records`; `data/jlens_corpus/wikitext-103-raw-v1-test.parquet` (present; column `text`, 4,358 rows, 591 with ≥150 words); `external/jacobian-lens/data/evaluations/*.json` (`items[*]` with `prompt`, `intermediates`, optional `target`). No new reader generations.
- Outputs under `runs/exp34_jlens/` only: `lenses/<reader>.pt` + `.json`, `<reader>/<qid>.{npz,json}`, `<reader>/_timing.json`, `_failed.json`, `_dropped.json`, `eval/<reader>.json`. Nothing under `runs/` (other than new files there), `cache/`, `data/`, `external/` is ever modified or deleted. Scripts never delete files.
- Fold rule: development folds {0, 1}; the report refuses other folds without `--confirm` (exit 2). The pod computes all five folds; no number from folds 2–4 is produced before the pre-registration commit.
- Spending: no agent creates any RunPod resource. The main session quotes and waits for the user.
- The `jlens` API, exactly: `jlens.from_hf(hf_model, tokenizer) -> LensModel` (freezes params, `.eval()`; `model.n_layers`, `model.d_model`, `model.layers`, `model.encode(text, max_length)`, `model.unembed(residual)`); `jlens.fit(model, prompts, *, source_layers=None, target_layer=None, dim_batch=8, max_seq_len=128, skip_first=16, checkpoint_path=None, checkpoint_every=1, resume=True) -> JacobianLens` (resume tracks `next_idx` over the same prompt list, so calling `fit` again with a longer prefix of the list processes only the new prompts); `JacobianLens.save(path)`, `.load(path)` (`weights_only=True`), `.from_pretrained(repo, filename=...)`, `.source_layers`, `.n_prompts`, `.apply(model, prompt, *, layers=None, positions=None, max_seq_len=512, use_jacobian=True) -> (lens_logits: dict[layer, Tensor[n_pos, vocab]], model_logits: Tensor[n_pos, vocab], input_ids)`. `apply` encodes `prompt` with the tokenizer itself (no special tokens added for Qwen), so pass `max_seq_len=4096` and compute positions on the identical ids.

## Review Focus

1. A question whose gold first content token is also the top token at `P0` for trivial reasons (for example a gold of "The" or "I"): the outcome logic must not count it as C by accident. Test in Task 2 with a tracked id that is top-1 at every layer and a distinct `unknown` id.
2. A prompt longer than `apply`'s default 512 tokens: positions computed on truncated ids would be silently wrong. The read script passes `max_seq_len=4096` and asserts `len(input_ids) == len(our_ids)`. Test in Task 1 (positions on a real 1,000-token prompt) and Task 4 (the assert).
3. A reader with fewer prompts than the minimum when the time box expires: the fit script must still save a lens and a sidecar saying `n_prompts < minimum`, and the pod script must stop before reading with that lens (exit 5). Test in Task 3 and Task 7.
4. A lens whose `source_layers` miss the final layer (fit never includes it): the readout must take the last layer's ranks from `model_logits`, not from the lens. Test in Task 2 and Task 4.
5. The two lens variants (J-lens, logit lens) must be read from the same forward activations and stored side by side so the gate compares like with like. Test in Task 4 (both arrays present, same shape, final-layer ranks identical between variants).

---

### Task 1: Readout positions and tracked tokens

**Files:**
- Create: `memctl/analysis/jlens_readout.py`
- Test: `tests/test_jlens.py`

**Interfaces:**
- Consumes: `memctl.analysis.jacobian.wrap_chat`, `segment_prompt`, `_ids`, `_margin_length`, `unknown_id`.
- Produces:
  ```python
  ANSWER_PREFIX = "Answer:"
  POSITION_NAMES = ("P1", "P0", "E", "Q")   # E and Q may hold several / zero positions

  @dataclass(frozen=True)
  class Readout:
      text: str                       # chat text + ANSWER_PREFIX
      input_ids: tuple[int, ...]      # tokenizer(text, add_special_tokens=False)
      positions: dict[str, tuple[int, ...]]   # "P1": (last,), "P0": (last token of generation prompt,), "E": last token of each evidence line, "Q": (last token of the question line,)
      tracked: dict[str, int]         # "gold": first content token id, "unknown_sp": id(" unknown"), "unknown": id("unknown")
      gold_token_text: str

  def readout_for(tokenizer, prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...], gold: str) -> Readout
  def tracked_token_ids(tokenizer, gold: str) -> tuple[dict[str, int], str]
  ```
  Positions are indices into `input_ids`. `P0` is the index of the last token of the generation prompt, i.e. `len(ids(chat_text)) - 1`. `E` positions: for each evidence line, the index of the token whose char span contains the line's last char (found through `segment_prompt` char spans and the tokenizer's offset mapping on `text`). `Q`: the token containing the last char of the question line (the line starting `Question:` inside the `## Current input` block; its end is the char before `\n\nNote:`; if `Note:` is absent, the last char of the prompt).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jlens.py
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.analysis.jlens_readout import ANSWER_PREFIX, Readout, readout_for, tracked_token_ids

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nassistant, d1: turn b\nhas newline\n"
          "user, d2: I was born in Leeds.\n\n## Current input\nQuestion: (asked on 2023/05/30) Where was I born?\n\nNote:")
LINES = ("user, d1: turn a", "assistant, d1: turn b\nhas newline", "user, d2: I was born in Leeds.")


@pytest.fixture(scope="module")
def tiny():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    return model, tok


def test_tracked_token_ids_skip_a_bare_space(tiny):
    _, tok = tiny
    ids, text = tracked_token_ids(tok, "4 hours")
    assert tok.decode([ids["gold"]]) == "4" and text == "4"
    ids2, text2 = tracked_token_ids(tok, "Leeds")
    assert tok.decode([ids2["gold"]]) == " Leeds" and text2 == " Leeds"
    assert tok.decode([ids["unknown_sp"]]) == " unknown" and tok.decode([ids["unknown"]]) == "unknown"


def test_readout_positions_land_on_the_right_tokens(tiny):
    _, tok = tiny
    r = readout_for(tok, PROMPT, LINES, (LINES[2],), gold="Leeds")
    assert isinstance(r, Readout) and r.text.endswith(ANSWER_PREFIX)
    ids = list(r.input_ids)
    assert ids == tok(r.text, add_special_tokens=False).input_ids
    (p1,) = r.positions["P1"]; (p0,) = r.positions["P0"]; (e,) = r.positions["E"]; (q,) = r.positions["Q"]
    assert p1 == len(ids) - 1 and p0 < p1
    assert tok.decode(ids[p0 + 1:p1 + 1]) == ANSWER_PREFIX          # P0 is the token right before "Answer:"
    assert tok.decode(ids[e]).strip().endswith(".") and "Leeds" in tok.decode(ids[e - 3:e + 1])
    assert tok.decode(ids[q]).strip().endswith("?")
    assert r.tracked["gold"] == tracked_token_ids(tok, "Leeds")[0]["gold"] and r.gold_token_text == " Leeds"


def test_readout_with_two_evidence_lines_and_no_note(tiny):
    _, tok = tiny
    prompt = PROMPT.replace("\n\nNote:", "")
    r = readout_for(tok, prompt, LINES, (LINES[0], LINES[2]), gold="Leeds")
    assert len(r.positions["E"]) == 2 and r.positions["E"][0] < r.positions["E"][1]
    (q,) = r.positions["Q"]
    assert tok.decode(list(r.input_ids)[q]).strip().endswith("?")


def test_readout_rejects_an_evidence_line_not_in_memory(tiny):
    _, tok = tiny
    with pytest.raises(ValueError):
        readout_for(tok, PROMPT, LINES, ("user, d9: not there",), gold="Leeds")


RUNS = Path("runs")
needs_exp23 = pytest.mark.skipif(not (RUNS / "exp23_qwen3b_f0").exists(), reason="§23 traces not present")


@needs_exp23
def test_readout_on_a_real_cached_prompt(tiny):
    from memctl.jacobian_data import load_records
    _, tok = tiny
    records, _ = load_records("qwen3b", 0)
    record = next(r for r in records if r.answerable and r.all_in_view and r.evidence_lines)
    r = readout_for(tok, record.prompt, record.memory_lines, record.evidence_lines, record.gold)
    assert len(r.input_ids) > 512                                   # longer than apply's default truncation
    assert len(r.positions["E"]) == len(record.evidence_lines)
    assert all(0 <= p < len(r.input_ids) for ps in r.positions.values() for p in ps)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'memctl.analysis.jlens_readout'`

- [ ] **Step 3: Write the implementation**

```python
# memctl/analysis/jlens_readout.py
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
```

Note for the implementer: `segment_prompt` extends an evidence span over the separator that follows it, so the line's last char is `start + len(line) - 1`, never `end - 1`. If `prompt.startswith(l, start)` matches more than one evidence line (identical strings), the first is fine, since both have the same length.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: 5 passed (the real-data test passes on this machine)

- [ ] **Step 5: Run the boundary tests**

Run: `.venv/bin/pytest tests/test_import_boundaries.py -q`
Expected: the same two pre-existing failures, no new one.

- [ ] **Step 6: Commit**

```bash
git add memctl/analysis/jlens_readout.py tests/test_jlens.py
git commit -m "§34: readout positions and tracked tokens for the J-lens"
```

---

### Task 2: Ranks, emergence depth and outcomes

**Files:**
- Modify: `memctl/analysis/jlens_readout.py` (append)
- Test: `tests/test_jlens.py` (append)

**Interfaces:**
- Consumes: nothing from `jlens`; pure numpy over logits arrays.
- Produces:
  ```python
  OUTCOMES = ("A", "B", "C")

  def ranks_of(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray
  # logits [n_pos, vocab] -> ranks [n_pos, n_tracked], 1 = top (ties: number of strictly greater logits + 1)
  def rank_table(lens_logits: dict[int, np.ndarray], model_logits: np.ndarray, n_layers: int, token_ids: Sequence[int]) -> tuple[np.ndarray, list[int]]
  # -> (ranks [len(layers), n_pos, n_tracked], layers): layers = sorted(lens_logits) plus the final layer n_layers-1 taken from model_logits (never from the lens)
  def topk_table(lens_logits: dict[int, np.ndarray], model_logits: np.ndarray, n_layers: int, k: int = 10) -> tuple[np.ndarray, np.ndarray, list[int]]
  # -> (ids [len(layers), n_pos, k] int32, logits [len(layers), n_pos, k] float32, layers)
  def best_rank(ranks_1d: np.ndarray) -> int
  def emergence_depth(ranks_1d: np.ndarray, layers: Sequence[int], n_layers: int, k: int) -> float | None
  # first layer with rank <= k, as layer / n_layers; None if never
  def outcome(ranks_1d: np.ndarray, k: int) -> str
  # "C" if ranks_1d[-1] <= k; else "B" if min(ranks_1d) <= k; else "A"
  def question_measures(ranks: np.ndarray, layers: Sequence[int], n_layers: int, tracked_names: Sequence[str], position_names: dict[str, Sequence[int]], k: int = 10) -> dict
  # {"k": k, "layers": layers, "P1": {"gold": {"best_rank", "final_rank", "emergence_depth", "outcome"}, "unknown_sp": {"best_rank", "final_rank", "first_top1_depth"}, ...}, "P0": {...}, "E": [ {...} per position ], "Q": {...}}
  ```
  `ranks` is indexed `[layer_idx, pos_idx, tracked_idx]` where `pos_idx` runs over the concatenation of all positions in the order P1, P0, E..., Q (the read script builds that order and passes `position_names` as the mapping name → list of pos_idx).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jlens.py
from memctl.analysis.jlens_readout import (OUTCOMES, best_rank, emergence_depth, outcome, question_measures,
                                           rank_table, ranks_of, topk_table)


def test_ranks_of_counts_strictly_greater_plus_one():
    logits = np.array([[0.1, 0.9, 0.9, 0.5], [3.0, 2.0, 1.0, 0.0]])
    assert ranks_of(logits, [1, 3, 0]).tolist() == [[1, 3, 4], [2, 4, 1]]


def test_rank_table_takes_the_final_layer_from_model_logits():
    vocab = 6
    lens = {0: np.zeros((1, vocab)), 2: np.zeros((1, vocab))}
    lens[0][0, 4] = 5.0           # gold top at layer 0
    model = np.zeros((1, vocab)); model[0, 1] = 9.0; model[0, 4] = 1.0   # at the final layer, token 1 is top, gold second
    ranks, layers = rank_table(lens, model, n_layers=4, token_ids=[4, 1])
    assert layers == [0, 2, 3] and ranks.shape == (3, 1, 2)
    assert ranks[0, 0].tolist() == [1, 2] and ranks[2, 0].tolist() == [2, 1]
    ids, lg, layers2 = topk_table(lens, model, n_layers=4, k=2)
    assert layers2 == layers and ids.shape == (3, 1, 2) and ids[2, 0, 0] == 1 and lg.dtype == np.float32


def test_outcome_and_depth_rules():
    assert outcome(np.array([50, 5, 3, 40]), k=10) == "B"
    assert outcome(np.array([50, 5, 3, 7]), k=10) == "C"
    assert outcome(np.array([50, 30, 20, 40]), k=10) == "A"
    assert outcome(np.array([1, 1, 1, 1]), k=10) == "C"          # always-top token is C, never B (review focus 1)
    assert best_rank(np.array([50, 5, 3, 40])) == 3
    assert emergence_depth(np.array([50, 5, 3, 40]), layers=[0, 8, 16, 23], n_layers=24, k=10) == pytest.approx(8 / 24)
    assert emergence_depth(np.array([50, 30, 20, 40]), layers=[0, 8, 16, 23], n_layers=24, k=10) is None
    assert set(OUTCOMES) == {"A", "B", "C"}


def test_question_measures_shapes_every_position():
    layers = [0, 1, 2, 3]
    ranks = np.full((4, 4, 3), 99)             # 4 layers, 4 positions (P1, P0, E, Q), 3 tracked
    ranks[1:, 0, 0] = [4, 2, 50]               # gold at P1: emerges at layer 1, lost at the end -> B
    ranks[:, 0, 1] = [99, 99, 3, 1]            # " unknown" at P1 reaches top-1 at the last layer
    m = question_measures(ranks, layers, n_layers=4, tracked_names=["gold", "unknown_sp", "unknown"],
                          position_names={"P1": [0], "P0": [1], "E": [2], "Q": [3]}, k=10)
    assert m["k"] == 10 and m["P1"]["gold"] == {"best_rank": 2, "final_rank": 50, "emergence_depth": pytest.approx(0.25), "outcome": "B"}
    assert m["P1"]["unknown_sp"]["first_top1_depth"] == pytest.approx(0.75)
    assert m["P0"]["gold"]["outcome"] == "A" and len(m["E"]) == 1 and m["Q"]["gold"]["best_rank"] == 99
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q -k "ranks or rank_table or outcome or measures"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

```python
# append to memctl/analysis/jlens_readout.py
from typing import Sequence

OUTCOMES = ("A", "B", "C")


def ranks_of(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    chosen = logits[:, list(token_ids)]                       # [n_pos, n_tracked]
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
        top = np.argsort(-lg, axis=-1)[:, :k]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add memctl/analysis/jlens_readout.py tests/test_jlens.py
git commit -m "§34: ranks, emergence depth and A/B/C outcomes from lens logits"
```

---

### Task 3: The lens-fitting script with a time box

**Files:**
- Create: `scripts/jlens_fit.py`
- Test: `tests/test_jlens.py` (append)

**Interfaces:**
- Consumes: `jlens.from_hf`, `jlens.fit`, `JacobianLens.save`; the parquet corpus.
- Produces: `runs/exp34_jlens/lenses/<reader>.pt` and `<reader>.json` sidecar `{"reader","model_id","n_prompts","n_target","n_minimum","below_minimum","settings": {...},"wall_s","jlens_commit","corpus_sha256","prompt_sha256s"}`; checkpoint at `<reader>.ckpt.pt`.
- Pure helpers (tested): `load_corpus(parquet_path, min_words=150) -> list[str]`; `select_prompts(paragraphs, n, seed=0) -> list[str]` (seeded shuffle, deterministic, first `n`); `fit_under_time_box(model, prompts, *, max_minutes, chunk, checkpoint_path, **fit_kwargs) -> tuple[JacobianLens, int]` which calls `jlens.fit(model, prompts[:m], checkpoint_path=..., resume=True, **fit_kwargs)` for `m = chunk, 2*chunk, ...` and stops when the elapsed time exceeds `max_minutes` or all prompts are done; returns the last lens and the number of prompts processed.
- CLI: `scripts/jlens_fit.py --reader {qwen05b,qwen3b,qwen7b} [--model-id ID] --n-prompts 128 --min-prompts 32 --max-minutes 90 --chunk 4 --dim-batch 8 --max-seq-len 128 --skip-first 16 [--source-layers all|0:k] --out runs/exp34_jlens/lenses --device auto --dtype bfloat16 --corpus data/jlens_corpus/wikitext-103-raw-v1-test.parquet --seed 0`. Exit 0 always when a lens was saved; the sidecar carries `below_minimum`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jlens.py
import importlib.util, time


def _script(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_prompts_is_deterministic_and_filters_short_paragraphs(tmp_path):
    s = _script("jlens_fit")
    import pyarrow as pa, pyarrow.parquet as pq
    rows = ["short one", " ".join(["word"] * 160), " ".join(["other"] * 200), " ".join(["third"] * 151)]
    pq.write_table(pa.table({"text": rows}), tmp_path / "c.parquet")
    paragraphs = s.load_corpus(tmp_path / "c.parquet", min_words=150)
    assert len(paragraphs) == 3
    a = s.select_prompts(paragraphs, 2, seed=0); b = s.select_prompts(paragraphs, 2, seed=0)
    assert a == b and len(a) == 2 and all(len(x.split()) >= 150 for x in a)


def test_fit_under_time_box_stops_early_and_resumes(tiny, tmp_path):
    jlens = pytest.importorskip("jlens")
    s = _script("jlens_fit")
    model, tok = tiny
    lens_model = jlens.from_hf(model, tok)
    prompts = [" ".join(f"w{i}" for i in range(200))] * 6
    ckpt = tmp_path / "ckpt.pt"
    lens, n = s.fit_under_time_box(lens_model, prompts, max_minutes=0.0, chunk=2, checkpoint_path=str(ckpt),
                                   source_layers=[1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    assert n == 2 and lens.n_prompts == 2                        # one chunk, then the box is over
    lens2, n2 = s.fit_under_time_box(lens_model, prompts, max_minutes=10.0, chunk=2, checkpoint_path=str(ckpt),
                                     source_layers=[1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    assert n2 == 6 and lens2.n_prompts == 6                      # resumed from the checkpoint, finished the list


def test_sidecar_marks_below_minimum(tmp_path):
    s = _script("jlens_fit")
    side = s.sidecar("qwen05b", "m", n_prompts=3, n_target=8, n_minimum=4, settings={"a": 1}, wall_s=1.0,
                     corpus_sha256="x", prompt_sha256s=["a", "b", "c"])
    assert side["below_minimum"] is True and side["n_prompts"] == 3 and side["jlens_commit"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q -k "select_prompts or time_box or sidecar"`
Expected: FAIL (script not found)

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""§34: fit a Jacobian lens for one reader on WikiText paragraphs, under a time box.

    .venv/bin/python scripts/jlens_fit.py --reader qwen05b --n-prompts 4 --min-prompts 2 --max-minutes 30 --device cpu --dtype float32
    python scripts/jlens_fit.py --reader qwen7b --n-prompts 128 --min-prompts 32 --max-minutes 90 --dim-batch 32 --device cuda

Writes <out>/<reader>.pt (fp16 Jacobians, one per source layer), <out>/<reader>.json (n_prompts, settings,
wall time, corpus and prompt hashes, jlens commit) and keeps <out>/<reader>.ckpt.pt for resume. Never
deletes anything. Exit 0 whenever a lens was saved; the sidecar's below_minimum says whether it is usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
JLENS_DIR = Path(__file__).resolve().parents[1] / "external" / "jacobian-lens"


def load_corpus(parquet_path, min_words: int = 150) -> list[str]:
    import pyarrow.parquet as pq

    texts = pq.read_table(str(parquet_path)).column("text").to_pylist()
    return [t.strip() for t in texts if len(t.split()) >= min_words]


def select_prompts(paragraphs: list[str], n: int, seed: int = 0) -> list[str]:
    order = list(range(len(paragraphs)))
    random.Random(seed).shuffle(order)
    return [paragraphs[i] for i in order[:n]]


def jlens_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(JLENS_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit_under_time_box(model, prompts, *, max_minutes: float, chunk: int, checkpoint_path: str, **fit_kwargs):
    """Call jlens.fit on growing prefixes of ``prompts`` (resume processes only the new ones) until the
    time box is spent or the list is done. Returns (lens, n_prompts_processed)."""
    import jlens

    started = time.perf_counter()
    lens, done = None, 0
    for upto in range(chunk, len(prompts) + chunk, chunk):
        upto = min(upto, len(prompts))
        lens = jlens.fit(model, prompts[:upto], checkpoint_path=checkpoint_path, checkpoint_every=1, resume=True, **fit_kwargs)
        done = upto
        if upto >= len(prompts) or (time.perf_counter() - started) / 60.0 >= max_minutes:
            break
    return lens, done


def sidecar(reader, model_id, *, n_prompts, n_target, n_minimum, settings, wall_s, corpus_sha256, prompt_sha256s) -> dict:
    return {"reader": reader, "model_id": model_id, "n_prompts": n_prompts, "n_target": n_target, "n_minimum": n_minimum,
            "below_minimum": n_prompts < n_minimum, "settings": settings, "wall_s": wall_s, "jlens_commit": jlens_commit(),
            "corpus_sha256": corpus_sha256, "prompt_sha256s": prompt_sha256s}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--model-id", default=None)
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--min-prompts", type=int, default=32)
    p.add_argument("--max-minutes", type=float, default=90.0)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--skip-first", type=int, default=16)
    p.add_argument("--source-layers", default="all", help="'all' (every layer below the last) or 'a:b' (half-open range)")
    p.add_argument("--out", default="runs/exp34_jlens/lenses")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--corpus", default="data/jlens_corpus/wikitext-103-raw-v1-test.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch, transformers, jlens

    model_id = args.model_id or MODEL_IDS[args.reader]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paragraphs = load_corpus(args.corpus)
    prompts = select_prompts(paragraphs, args.n_prompts, seed=args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, args.dtype)).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    source_layers = None if args.source_layers == "all" else list(range(*map(int, args.source_layers.split(":"))))
    settings = {"dim_batch": args.dim_batch, "max_seq_len": args.max_seq_len, "skip_first": args.skip_first,
                "source_layers": args.source_layers, "seed": args.seed, "chunk": args.chunk, "max_minutes": args.max_minutes,
                "device": device, "dtype": args.dtype}
    print(f"{model} fitting on {len(prompts)} prompts (min {args.min_prompts}, box {args.max_minutes} min)", flush=True)
    started = time.perf_counter()
    lens, n_done = fit_under_time_box(model, prompts, max_minutes=args.max_minutes, chunk=args.chunk,
                                      checkpoint_path=str(out / f"{args.reader}.ckpt.pt"), source_layers=source_layers,
                                      dim_batch=args.dim_batch, max_seq_len=args.max_seq_len, skip_first=args.skip_first)
    wall = time.perf_counter() - started
    lens.save(str(out / f"{args.reader}.pt"))
    side = sidecar(args.reader, model_id, n_prompts=lens.n_prompts, n_target=args.n_prompts, n_minimum=args.min_prompts,
                   settings=settings, wall_s=wall, corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                   prompt_sha256s=[hashlib.sha256(x.encode()).hexdigest() for x in prompts[:n_done]])
    (out / f"{args.reader}.json").write_text(json.dumps(side, indent=1))
    print(json.dumps({k: v for k, v in side.items() if k != "prompt_sha256s"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: all pass (the tiny-model fit runs in seconds: `d_model` 32 → 4 backward passes per prompt)

- [ ] **Step 5: Fit a real local development lens (0.5B, CPU)** so later tasks can read with it: `.venv/bin/python scripts/jlens_fit.py --reader qwen05b --n-prompts 4 --min-prompts 2 --max-minutes 30 --chunk 2 --device cpu --dtype float32 --out runs/exp34_jlens/lenses` (about 11 minutes: 161 s per prompt measured). Record the sidecar line in the report.

- [ ] **Step 6: Commit**

```bash
git add scripts/jlens_fit.py tests/test_jlens.py
git commit -m "§34: lens fitting under a time box with checkpoint resume"
```

---

### Task 4: The read script (resumable; J-lens and logit lens from one forward)

**Files:**
- Create: `scripts/jlens_read.py`
- Test: `tests/test_jlens.py` (append)

**Interfaces:**
- Consumes: `load_records` (§33), `readout_for`, `rank_table`, `topk_table`, `question_measures` (Tasks 1–2), `jlens.JacobianLens.load`, `jlens.from_hf`, `jlens.hooks.ActivationRecorder`, `lens.transport`, `model.unembed`.
- Produces per record: `runs/exp34_jlens/<reader>/<qid>.npz` with `ranks_jlens [L, P, T] int32`, `ranks_logit [L, P, T] int32`, `top_ids_jlens [L, P, 10] int32`, `top_logits_jlens [L, P, 10] float32`, `top_ids_logit`, `top_logits_logit`, `layers [L] int32`, `position_index [P] int32`; and `<qid>.json` = `{"question_id","fold","reader","model_id","lens_path","lens_n_prompts","prompt_sha256","n_tokens","positions": {...},"tracked": {...},"gold_token_text","measures": {"jlens": question_measures(...), "logit": question_measures(...)},"elapsed_s"}`. Also `_timing.json` (append semantics as §33), `_failed.json` (per-record isolation as §33), `_dropped.json`.
- Core function (tested with the tiny model): `read_record(lens, lens_model, tokenizer, record, k=10) -> tuple[dict, dict[str, np.ndarray]]` that (1) builds the readout, (2) encodes `readout.text` with `lens_model.encode(text, max_length=4096)` and asserts the ids equal `readout.input_ids` (raise `ValueError` otherwise), (3) runs one forward under `ActivationRecorder(lens_model.layers, at=lens.source_layers + [final])`, (4) for every recorded layer selects the residuals at the concatenated positions, computes J-lens logits (`lens.transport` then `unembed`) and logit-lens logits (`unembed` directly), the final layer from the final residual only, (5) builds both rank tables and top-k tables and both measure dicts. Model logits at the final layer are identical for both variants (review focus 5).
- CLI: `scripts/jlens_read.py --reader {qwen05b,qwen3b,qwen7b} [--records-from R] --lens PATH --folds 0,1,2,3,4 [--limit N] [--k 10] [--out DIR] [--device auto] [--dtype bfloat16] [--model-id ID]`. Resume keyed on a readable json; npz written before json; everything computed before any write; "nothing to do" without loading the model.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jlens.py
from memctl.jacobian_data import PromptRecord


@pytest.fixture(scope="module")
def tiny_lens(tiny, tmp_path_factory):
    jlens = pytest.importorskip("jlens")
    model, tok = tiny
    lens_model = jlens.from_hf(model, tok)
    prompts = [" ".join(f"w{i}" for i in range(200))] * 2
    lens = jlens.fit(lens_model, prompts, source_layers=[0, 1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    path = tmp_path_factory.mktemp("lens") / "tiny.pt"
    lens.save(str(path))
    return lens, lens_model, tok, path


def test_read_record_produces_both_variants_from_one_forward(tiny_lens):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_read")
    record = PromptRecord("qX", 0, "fake", PROMPT, "Answer: Leeds", "Leeds", "single-session-user", True, LINES, (LINES[2],), True, True, False)
    payload, arrays = s.read_record(lens, lens_model, tok, record, k=5)
    L, P, T = arrays["ranks_jlens"].shape
    assert arrays["layers"].tolist() == [0, 1, 2, 3] and P == 4 and T == 3
    assert arrays["ranks_logit"].shape == (L, P, T)
    assert (arrays["ranks_jlens"][-1] == arrays["ranks_logit"][-1]).all()          # final layer identical (model logits)
    assert arrays["top_ids_jlens"].shape == (L, P, 5)
    assert payload["measures"]["jlens"]["P1"]["gold"]["outcome"] in ("A", "B", "C")
    assert payload["positions"]["P1"] == [payload["n_tokens"] - 1] and payload["gold_token_text"] == " Leeds"
    assert len(payload["prompt_sha256"]) == 64


def test_read_record_rejects_truncated_ids(tiny_lens, monkeypatch):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_read")
    record = PromptRecord("qX", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    monkeypatch.setattr(s, "MAX_SEQ_LEN", 8)                      # simulate apply-style truncation
    with pytest.raises(ValueError):
        s.read_record(lens, lens_model, tok, record, k=5)


def test_run_records_writes_resumes_and_isolates_failures(tiny_lens, tmp_path):
    lens, lens_model, tok, path = tiny_lens
    s = _script("jlens_read")
    good = PromptRecord("q1", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    bad = PromptRecord("q2", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, ("not a memory line",), True, True, False)
    timing = s.run_records(lens, lens_model, tok, [good, bad], tmp_path, k=5, model_id="tiny", lens_path=str(path))
    assert set(p.name for p in tmp_path.glob("q1.*")) == {"q1.npz", "q1.json"} and not list(tmp_path.glob("q2.*"))
    failed = json.loads((tmp_path / "_failed.json").read_text())
    assert "q2" in failed and "q1" in timing
    assert s.run_records(lens, lens_model, tok, [good], tmp_path, k=5, model_id="tiny", lens_path=str(path)) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q -k "read_record or run_records"`
Expected: FAIL (script not found)

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""§34: read a fitted Jacobian lens (and the logit-lens baseline) at the answer, prompt-end, evidence and
question positions of every cached §23 prompt. One forward per question; resumable per question.

    .venv/bin/python scripts/jlens_read.py --reader qwen05b --records-from qwen3b --lens runs/exp34_jlens/lenses/qwen05b.pt --folds 0 --limit 10 --device cpu --dtype float32
    python scripts/jlens_read.py --reader qwen3b --lens runs/exp34_jlens/lenses/qwen3b.pt --folds 0,1,2,3,4 --device cuda

Outputs under runs/exp34_jlens/<reader>/: <qid>.npz (ranks and top-k for both variants), <qid>.json
(positions, tracked ids, measures), _timing.json, _failed.json (retried next run), _dropped.json.
Never deletes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jlens_readout import POSITION_NAMES, question_measures, rank_table, readout_for, topk_table  # noqa: E402
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
MAX_SEQ_LEN = 4096
TRACKED = ("gold", "unknown_sp", "unknown")


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _readable(path: Path) -> bool:
    try:
        json.loads(path.read_text()); return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def is_done(out: Path, question_id: str) -> bool:
    return _readable(out / f"{question_id}.json")


def _flat_positions(readout) -> tuple[list[int], dict[str, list[int]]]:
    flat, index = [], {}
    for name in POSITION_NAMES:
        index[name] = list(range(len(flat), len(flat) + len(readout.positions[name])))
        flat.extend(readout.positions[name])
    return flat, index


def read_record(lens, lens_model, tokenizer, record: PromptRecord, k: int = 10) -> tuple[dict, dict[str, np.ndarray]]:
    import torch
    from jlens.hooks import ActivationRecorder

    started = time.perf_counter()
    readout = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold)
    input_ids = lens_model.encode(readout.text, max_length=MAX_SEQ_LEN)
    if input_ids.shape[1] != len(readout.input_ids) or input_ids[0].tolist() != list(readout.input_ids):
        raise ValueError(f"{record.question_id}: lens encode ({input_ids.shape[1]} ids) differs from the readout ids ({len(readout.input_ids)})")
    final = lens_model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final})
    flat, index = _flat_positions(readout)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        residuals = {l: recorder.activations[l][0][flat].float() for l in layers}
    tracked_ids = [readout.tracked[name] for name in TRACKED]
    jl, ll = {}, {}
    with torch.no_grad():
        for l in layers:
            if l == final:
                continue
            jl[l] = lens_model.unembed(lens.transport(residuals[l], l)).float().cpu().numpy()
            ll[l] = lens_model.unembed(residuals[l]).float().cpu().numpy()
        model_logits = lens_model.unembed(residuals[final]).float().cpu().numpy()
    arrays = {}
    for variant, logits in (("jlens", jl), ("logit", ll)):
        ranks, layer_list = rank_table(logits, model_logits, lens_model.n_layers, tracked_ids)
        top_ids, top_logits, _ = topk_table(logits, model_logits, lens_model.n_layers, k)
        arrays[f"ranks_{variant}"], arrays[f"top_ids_{variant}"], arrays[f"top_logits_{variant}"] = ranks, top_ids, top_logits
    arrays["layers"] = np.array(layer_list, dtype=np.int32)
    arrays["position_index"] = np.array(flat, dtype=np.int32)
    measures = {variant: question_measures(arrays[f"ranks_{variant}"], layer_list, lens_model.n_layers, TRACKED, index, k)
                for variant in ("jlens", "logit")}
    payload = {"question_id": record.question_id, "fold": record.fold, "reader": record.reader, "n_tokens": len(readout.input_ids),
               "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "positions": {name: list(readout.positions[name]) for name in POSITION_NAMES}, "tracked": readout.tracked,
               "gold_token_text": readout.gold_token_text, "measures": measures, "elapsed_s": time.perf_counter() - started}
    return payload, arrays


def run_records(lens, lens_model, tokenizer, records, out: Path, *, k: int, model_id: str, lens_path: str) -> dict[str, float]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    failed_path = out / "_failed.json"
    failed = json.loads(failed_path.read_text()) if _readable(failed_path) else {}
    timing = {}
    for number, record in enumerate(records):
        if is_done(out, record.question_id):
            continue
        try:
            payload, arrays = read_record(lens, lens_model, tokenizer, record, k=k)
        except (ValueError, RuntimeError) as error:
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            print(f"[{number + 1}/{len(records)}] {record.question_id} FAILED {error!r}", flush=True)
            continue
        payload.update({"model_id": model_id, "lens_path": lens_path, "lens_n_prompts": getattr(lens, "n_prompts", None)})
        np.savez_compressed(out / f"{record.question_id}.npz", **arrays)
        write_json(out / f"{record.question_id}.json", payload)
        if record.question_id in failed:
            failed.pop(record.question_id); write_json(failed_path, failed)
        timing[record.question_id] = payload["elapsed_s"]
        print(f"[{number + 1}/{len(records)}] {record.question_id} {payload['elapsed_s']:.1f}s "
              f"{payload['measures']['jlens']['P1']['gold']['outcome']}", flush=True)
    return timing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--records-from", default=None)
    p.add_argument("--lens", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model-id", default=None)
    args = p.parse_args(argv)
    out = Path(args.out or f"runs/exp34_jlens/{args.reader}"); out.mkdir(parents=True, exist_ok=True)
    records, dropped = [], {}
    for fold in (int(f) for f in args.folds.split(",")):
        rs, d = load_records(args.records_from or args.reader, fold)
        records.extend(rs[: args.limit] if args.limit else rs); dropped[str(fold)] = d
    write_json(out / "_dropped.json", dropped)
    pending = [r for r in records if not is_done(out, r.question_id)]
    if not pending:
        print("nothing to do", flush=True); return 0
    import torch, transformers, jlens

    model_id = args.model_id or MODEL_IDS[args.reader]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, args.dtype)).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    lens_model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    started = time.perf_counter()
    timing = run_records(lens, lens_model, tok, records, out, k=args.k, model_id=model_id, lens_path=args.lens)
    summary = {"model_id": model_id, "lens": args.lens, "device": device, "dtype": args.dtype, "questions": len(timing),
               "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    path = out / "_timing.json"
    existing = json.loads(path.read_text()) if _readable(path) else {}
    existing.setdefault("runs", []).append(summary); write_json(path, existing)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `lens.transport(residual, layer)` expects `[n, d_model]` float on the lens's device; check `jlens/lens.py` `transport` for the device/dtype it wants and move accordingly (the Jacobians are fp32 on CPU after `load`; moving the lens's `jacobians[layer]` to the model device once at start is the efficient path — add a small `_lens_to(lens, device)` helper if `transport` is slow on CPU for the 7B).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: all pass

- [ ] **Step 5: Local end-to-end read with the development lens from Task 3** on ten real prompts: `.venv/bin/python scripts/jlens_read.py --reader qwen05b --records-from qwen3b --lens runs/exp34_jlens/lenses/qwen05b.pt --folds 0 --limit 10 --device cpu --dtype float32` then rerun to see "nothing to do". Record the per-question outcomes printed and the timing line.

- [ ] **Step 6: Commit**

```bash
git add scripts/jlens_read.py tests/test_jlens.py
git commit -m "§34: resumable lens readout, J-lens and logit lens from one forward"
```

---

### Task 5: Lens-quality evaluation on the package's prompt sets

**Files:**
- Create: `scripts/jlens_eval.py`
- Test: `tests/test_jlens.py` (append)

**Interfaces:**
- Consumes: `external/jacobian-lens/data/evaluations/lens-eval-{association,multihop,multilingual,order-ops,poetry,typo}.json` (`items[*]`: `prompt` (string, or a list of chat turns), `intermediates` (list of target strings), optional `target`); `jlens` apply-style readout at one position across all layers; `ranks_of`.
- Produces: `runs/exp34_jlens/eval/<reader>.json` = per set: `{"n_items","pass_at_k": {"jlens": x, "logit": y}, "k"}` and overall means. Readout position per the data README: for `multihop`/`multilingual`/`association`/`typo`/`order-ops`, the token immediately preceding `target` when `target` is present (encode `prompt` and `prompt + target` and take `len(ids(prompt)) - 1`), else the last token; for `poetry`, the last newline token. A chat-list prompt is rendered with the tokenizer's chat template (no generation prompt). An intermediate's token id is the first content token of `" " + intermediate` (same rule as the gold); pass@k per item = fraction of intermediates whose min-over-layers rank ≤ k, over the middle band of layers (`[n_layers//3, 2*n_layers//3)` by default, `--band a:b` to override).
- Pure helpers tested: `readout_position(tokenizer, item, set_name) -> tuple[str, int]` (text, position); `pass_at_k(ranks_by_layer: np.ndarray [L, n_intermediates], band_mask: np.ndarray [L] bool, k) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jlens.py
def test_eval_readout_position_before_target_and_poetry_newline(tiny):
    _, tok = tiny
    s = _script("jlens_eval")
    text, pos = s.readout_position(tok, {"prompt": "The capital of France is", "target": " Paris", "intermediates": ["France"]}, "multihop")
    assert text == "The capital of France is" and pos == len(tok(text, add_special_tokens=False).input_ids) - 1
    text2, pos2 = s.readout_position(tok, {"prompt": "Roses are red, violets are blue\nSugar is sweet", "intermediates": ["blue"]}, "poetry")
    ids = tok(text2, add_special_tokens=False).input_ids
    assert "\n" in tok.decode([ids[pos2]])
    text3, _ = s.readout_position(tok, {"prompt": [{"role": "user", "content": "hi"}], "intermediates": ["x"]}, "association")
    assert "hi" in text3 and "<|im_start|>" in text3


def test_pass_at_k_uses_only_the_band():
    s = _script("jlens_eval")
    ranks = np.array([[1, 50], [20, 50], [30, 3], [1, 1]])      # 4 layers, 2 intermediates
    band = np.array([False, True, True, False])
    assert s.pass_at_k(ranks, band, k=10) == pytest.approx(0.5)  # only the second intermediate hits inside the band
    assert s.pass_at_k(ranks, np.ones(4, bool), k=10) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q -k "eval_readout or pass_at_k"`
Expected: FAIL (script not found)

- [ ] **Step 3: Write the implementation** — `scripts/jlens_eval.py` with `readout_position`, `pass_at_k`, a `score_set(lens, lens_model, tokenizer, items, set_name, band, k) -> dict` that reuses the single-forward pattern of `jlens_read.read_record` (import `read_record`'s helpers? No: implement a small `logits_at(lens, lens_model, text, position) -> (jl: dict[layer, np.ndarray[vocab]], ll, model_logits)` here; keep it under 40 lines), and `main` with `--reader --lens --sets all|names --band auto|a:b --k 10 --out runs/exp34_jlens/eval --device --dtype --model-id --limit`. The docstring states the README conventions it follows. Skip items whose readout text exceeds 4096 tokens and count them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: all pass

- [ ] **Step 5: Run the eval locally with the development lens**: `.venv/bin/python scripts/jlens_eval.py --reader qwen05b --lens runs/exp34_jlens/lenses/qwen05b.pt --sets multihop,association --limit 20 --device cpu --dtype float32`. Record the pass@10 for both variants (a 4-prompt lens may not beat the logit lens; that is fine locally, the number is the plumbing proof).

- [ ] **Step 6: Commit**

```bash
git add scripts/jlens_eval.py tests/test_jlens.py
git commit -m "§34: lens-quality evaluation on the package's prompt sets"
```

---

### Task 6: The report

**Files:**
- Create: `scripts/exp34_report.py`
- Modify: `memctl/analysis/jacobian_stats.py` (append `bootstrap_proportion`)
- Test: `tests/test_jlens.py` (append)

**Interfaces:**
- Consumes: `load_records` (`--records-from` as in §33), the `<qid>.json` outputs of Task 4 for two readers, `runs/exp34_jlens/eval/<reader>.json`, `runs/exp34_jlens/lenses/<reader>.json` sidecars, `paired_claim`, `bootstrap_auc`, `bootstrap_mean_difference` (§33 stats).
- Produces: `build_report(readers=(small, large), folds, k_label, root, samples, records_from=None) -> (markdown, numbers)`; CLI `--small qwen3b --large qwen7b --folds 0,1 [--confirm] [--root runs/exp34_jlens] [--out] [--samples 10000] [--records-from] [--community-root DIR]`; exit 2 on confirmation folds without `--confirm`, exit 4 on prompt-hash mismatch (as §33).
- Sections and numbers (`numbers[...]` keys in brackets):
  1. **Lens gate** [`gate`]: per reader (a) paired mean difference of `log10(best rank)` of the gold at `P1`, logit lens minus J-lens, over the middle band only (recompute best rank from the npz over band layers; band = `[n_layers//3, 2*n_layers//3)`), `paired_claim`-style with verdict `PASS` if `ci_low > 0`; (b) lens-eval pass@10 J-lens vs logit per set, mean over sets, `PASS` if J-lens ≥ logit on the mean; gate verdict = (a) and (b); sidecar `n_prompts` and `below_minimum` shown.
  2. **Claim 1** [`claim1`]: paired `log10(best_rank)` at `P1`, small minus large (positive = large better), `paired_claim` verdict; secondary: share of outcome A, small minus large, via `paired_claim` on 0/1 indicators.
  3. **Claim 2** [`claim2`]: within each reader, `bootstrap_auc(wrong_best_logranks, right_best_logranks)` so AUC > 0.5 means failures have worse rank; verdict SHOWN if `ci_low > 0.5`; `bootstrap_mean_difference` of the means.
  4. **Decomposition** [`decomposition`]: per reader, among in-view failures split by `unknown` (refusal) vs answered-wrong, the share of A, B, C with `bootstrap_proportion` intervals; the same among successes.
  5. **Descriptives** [`descriptive`]: emergence-depth mean and quartiles per reader among questions with an emergence; `E` and `Q` best ranks; the `unknown_sp` first-top1 depth on refusals; per question type outcome shares; if `--community-root` is given, the top-10 overlap at `P1` between our large-reader lens and the community lens per layer band (mean Jaccard) and the Claim 1 numbers recomputed with the community lens.
  6. **Dropped and failed** counts, prompt identity check (as §33).
- `bootstrap_proportion(flags: list[bool], samples=10000, seed=0) -> {"n","p","ci_low","ci_high"}` appended to `jacobian_stats.py` with a test.

- [ ] **Step 1: Write the failing tests** — a synthetic-outputs test in the style of §33's `test_report_builds_claims_from_outputs`: write fake `<qid>.json` files for two readers (40 questions; small reader gold best rank worse and more A outcomes; `measures.jlens` and `measures.logit` both present, logit worse), fake `eval/<reader>.json`, fake sidecars; monkeypatch `load_records`; assert `numbers["gate"][reader]["verdict"] == "PASS"`, `numbers["claim1"]["best_logrank"]["verdict"] == "SHOWN"`, `numbers["claim2"]["qwen3b"]["auc"] > 0.5`, decomposition shares per reader sum to 1 within each failure class, and the markdown contains "Lens gate", "Claim 1", "Decomposition". Plus `test_bootstrap_proportion_interval_contains_p`, `test_report_refuses_confirmation_folds_without_confirm` and `test_report_exit_4_on_prompt_mismatch` mirroring §33's tests.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jlens.py -q -k "report or proportion"`
Expected: FAIL

- [ ] **Step 3: Write the implementation** following `scripts/exp33_report.py`'s structure (`load_outputs` by directory and fold, `population`, `_fmt`, `PromptIdentityError`, the fold guard, markdown tables). Keep functions under 60 lines; compute band best ranks from the npz with a helper `band_best_logrank(npz_path, variant, position_idx, tracked_idx, band) -> float`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jlens.py -q`
Expected: all pass

- [ ] **Step 5: Run the report on the local ten-question outputs** (both readers pointing at the `qwen05b` folder through a temp root with symlinks `qwen3b` and `qwen7b` → the 0.5B outputs, `--records-from qwen3b`), expecting zero differences and a report that runs end to end.

- [ ] **Step 6: Commit**

```bash
git add scripts/exp34_report.py memctl/analysis/jacobian_stats.py tests/test_jlens.py
git commit -m "§34: report with lens gate, claims, decomposition and fold guard"
```

---

### Task 7: Pod runbook, README how-to, EXPERIMENT_INDEX pin

**Files:**
- Create: `scripts/exp34_pod.sh`
- Modify: `README_RESEARCH.md` (append a §34 how-to), `research/EXPERIMENT_INDEX.md` is Advisor-owned: do NOT edit it; instead add the pin line to the README how-to and note in the report that the index needs the row `external/jacobian-lens@581d398`.
- Test: `tests/test_jlens.py` (append, pod-script tests in the style of §33's)

**Interfaces:**
- `scripts/exp34_pod.sh [max_minutes_7b=90] [max_minutes_3b=40] [n_prompts=128]`:
  1. Preflight (before pip or any download): every §33 input path (reuse the list from `scripts/exp33_pod.sh`) plus `data/jlens_corpus/wikitext-103-raw-v1-test.parquet`, `external/jacobian-lens/pyproject.toml`; exit 3 naming the missing path.
  2. `pip install -q -e . -e external/jacobian-lens pyarrow scipy "transformers==<pinned>"`; print torch and transformers versions and the jlens commit.
  3. Fit: `python scripts/jlens_fit.py --reader qwen3b --n-prompts $N --min-prompts 32 --max-minutes $M3 --dim-batch 32 --device cuda` then the same for `qwen7b` with `$M7`; after each, read the sidecar; if `below_minimum` is true, print it and exit 5 (never read with a lens below the minimum).
  4. Read: `python scripts/jlens_read.py --reader qwen3b --lens runs/exp34_jlens/lenses/qwen3b.pt --folds 0,1,2,3,4 --device cuda`, then `qwen7b`.
  5. Eval: `python scripts/jlens_eval.py --reader qwen3b --lens ... --device cuda`, then `qwen7b`.
  6. Community cross-check: `huggingface_hub.hf_hub_download("andyx10/jacobian-lens-qwen2.5-7b-instruct", "qwen7b_lens.pt")` into `runs/exp34_jlens/lenses/community_qwen7b.pt`, then `jlens_read.py --reader qwen7b --lens <that> --out runs/exp34_jlens/community_qwen7b --folds 0,1` (development folds only).
  7. Secondary: `bash scripts/exp33_pod.sh`.
  8. Print elapsed time per stage; `set -euo pipefail`; everything resumable.
- README how-to: local commands (fit 0.5B dev lens, read 10, eval, report with `--records-from qwen3b`), the paid flow, the paths to copy (the §33 list plus the corpus parquet and `external/jacobian-lens/`), the fold rule, and the pin line `external/jacobian-lens@581d398 (jlens 0.1.0, Apache-2.0)`.

- [ ] **Step 1: Write the failing tests** (pod script: preflight exits 3 naming a missing path before any python call; the sidecar check exits 5 when `below_minimum` is true, tested by stubbing `python` with a shell function in a temp PATH that writes a sidecar with `below_minimum: true`; the transformers pin equals the local version).

- [ ] **Step 2: Run the tests to verify they fail**

- [ ] **Step 3: Write the script and the README section**

- [ ] **Step 4: Run `.venv/bin/pytest tests/test_jlens.py tests/test_jacobian.py tests/test_import_boundaries.py -q`** — expect all §33/§34 tests green and only the two pre-existing boundary failures.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp34_pod.sh README_RESEARCH.md tests/test_jlens.py
git commit -m "§34: pod runbook (fit, read, eval, cross-check, then §33) and README how-to"
```

---

## After the plan (main session, not an implementation task)

1. Pre-launch checklist: the local 0.5B fit/read/eval/report chain ran (Tasks 3–6 steps 5); time per prompt and per question measured; resume tested in a fresh process; peak memory recorded. Quote the pod (A40 secure $0.59/h or H100 fallback), the time boxes and the hard stop; wait for the user.
2. Paid run; copy `runs/exp34_jlens/` and `runs/exp33_jacobian/` back; check sidecars (`n_prompts`), `_failed.json`, `_regen_check.json`.
3. Development-fold report (folds 0,1): lens gate; fix k, the band and thresholds; pre-registration in `research/proposals/2026-10-XX-exp34-jlens.md` with the §34 text for EXPERIMENTS.md and the recorded prediction; reviewer sign-off; commit hash.
4. Confirmation report (`--confirm`); result sections §34a and §33a in EXPERIMENTS.md; report copies under `docs/research/`; gate decision on stage 2; ask the Advisor to add the `external/jacobian-lens@581d398` row to `research/EXPERIMENT_INDEX.md`.
