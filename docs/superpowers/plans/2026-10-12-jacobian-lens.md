# Jacobian Lens (§33, stage 1 gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, per question and per reader, how much a Qwen 2.5 3B and 7B reader's answer depends on the evidence tokens of the identical cached prompt, using the gradient of teacher-forced targets with respect to the input embeddings, and report the pre-registered claims.

**Architecture:** A data module rebuilds each §23 `fixed8` prompt record from the generation cache and the run traces (prompt, output, gold, evidence lines, in-view flag, verdict). A pure analysis module segments a prompt into regions, wraps it in the chat template, builds the three teacher-forced targets, runs one forward and backward pass with gradients on the input embeddings, and turns per-token gradient norms into region shares. A CLI drives the models over the records with per-question resumable outputs. A report pipeline computes the claims.

**Tech Stack:** Python 3.12 in `.venv` (torch 2.14 CPU, transformers 5.17, numpy, scipy), pytest. Hugging Face models `Qwen/Qwen2.5-0.5B-Instruct` (development), `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (paid run on a RunPod A40).

**Spec:** `docs/superpowers/specs/2026-10-12-jacobian-lens-design.md`

## Global Constraints

- Branch `jacobian-lens`, created from `origin/main`. Never work on `main`. Commit after every task; no `Claude-Session` lines in commit messages (user rule).
- Use `.venv/bin/python` and `.venv/bin/pytest` for everything. Run tests with `.venv/bin/pytest tests/test_jacobian.py -q` and the boundary test `.venv/bin/pytest tests/test_import_boundaries.py -q` (two pre-existing failures there are not yours: `memory/engine.py` importing `memctl.retrieval`, and one analysis-module boundary; do not touch them).
- `memctl/analysis/` must not import `memctl.harness`, `memctl.controllers`, `memctl.envs`, `memctl.agents` or `memctl.memory` (tests/test_import_boundaries.py). The data loader therefore lives at `memctl/jacobian_data.py`, next to `memctl/rerank_data.py`.
- Imports of `torch` and `transformers` are lazy (inside functions) in `memctl/analysis/jacobian.py` so the module imports without them; tests use `pytest.importorskip`.
- No new reader generations. Prompts come only from `cache/generations_exp23_<reader>/`; traces from `runs/exp23_<reader>_f<k>/fixed8__fraction0.05/`; verdicts from `runs/exp23_verdicts.json`; questions from `data/longmemeval/longmemeval_s_cleaned.json`.
- Outputs go to `runs/exp33_jacobian/<reader>/` in the main tree (symlink `runs/` if in a worktree). Nothing under `runs/`, `cache/`, `data/` or `external/` is ever deleted.
- Fold rule: folds 0 and 1 are development. The paid run computes all folds in one pass and stores raw per-token gradients for both mass variants, but **no report, table or number is produced from folds 2 to 4** until the pre-registration is committed. The report CLI takes `--folds` and defaults to `0,1`.
- Spending: no RunPod resource is created by any agent. The main session quotes the price and waits for the user. Development runs use the 0.5B on CPU.
- Chat template: exactly as `memctl/llm.py` `HFLLM.generate` does it (user message = prompt, `add_generation_prompt=True`, `enable_thinking=False` when accepted).

## Review Focus

1. A memory line whose content contains a newline: the line count of the memory section is not the item count. Segmentation must find lines by exact substring in order, never by splitting on `\n`. Test in Task 2.
2. A gold answer whose first token tokenises differently after `Answer:` than alone (for example a gold starting with a digit or quote). Continuation ids must be built as `tok("Answer:") + tok(" " + gold)` and scored on exactly those ids. Test in Task 3.
3. A question with two or more evidence turns, or with fallback-session evidence (no `has_answer` mark): all such in-view lines are `evidence`. Test in Task 1.
4. A prompt where an evidence line cannot be found (text truncated or altered): the record is dropped with a reason, counted, and the run continues. Test in Task 1 and Task 6.
5. Resume: a crashed run restarts and skips every (question, target) that already has an output file, and a partial file (crash mid-write) is rewritten. Outputs are written to a temp file and renamed. Test in Task 6.

---

### Task 1: Prompt records from the cache and traces

**Files:**
- Create: `memctl/jacobian_data.py`
- Test: `tests/test_jacobian.py`

**Interfaces:**
- Consumes: `memctl.envs.longmemeval.parse_instance`, `_instance`, `_index` (existing); `memctl.memory.items.label_prefix` (existing).
- Produces:
  ```python
  @dataclass(frozen=True)
  class PromptRecord:
      question_id: str
      fold: int
      reader: str
      prompt: str                      # exactly what the reader saw (user message)
      output: str                      # the reader's cached output
      gold: str
      question_type: str
      answerable: bool
      memory_lines: tuple[str, ...]    # every memory line, prompt order, "speaker, date: content"
      evidence_lines: tuple[str, ...]  # the subset carrying gold evidence
      all_in_view: bool                # evidence_complete_rate == 1.0 in the trace
      correct: bool | None             # §23 judge verdict (None if missing)
      unknown: bool | None

  def memory_line(item: dict) -> str
  def load_records(reader: str, fold: int, runs_dir: str = "runs", cache_dir: str | None = None,
                   data_path: str = "data/longmemeval/longmemeval_s_cleaned.json",
                   arm: str = "fixed8__fraction0.05") -> tuple[list[PromptRecord], dict[str, int]]
  # returns (records, dropped) where dropped counts reasons: "no_cached_prompt", "evidence_not_found"
  ```

- [ ] **Step 1: Write the failing tests** (synthetic run folder built in `tmp_path`; a real-data test skipped when `runs/exp23_qwen3b_f0` is absent)

```python
# tests/test_jacobian.py
import gzip, json
from pathlib import Path

import pytest

from memctl.jacobian_data import PromptRecord, load_records, memory_line

RUNS = Path("runs")
CACHE = Path("cache/generations_exp23_qwen3b")
needs_exp23 = pytest.mark.skipif(not (RUNS / "exp23_qwen3b_f0").exists() or not CACHE.exists(),
                                 reason="§23 traces or cache not present")


def _item(ep, item_id, content, date="2023/05/20 (Sat) 02:21", speaker="user", created_at=1):
    return {"experiment_id": "fixed8__fraction0.05", "episode_id": ep, "id": item_id, "content": content,
            "created_at": created_at, "source_type": speaker, "metadata": {"speaker": speaker, "date": date}}


def test_memory_line_matches_compact_reader_format():
    item = _item("ep00000", "s1:0", "I was born in Leeds.")
    assert memory_line(item) == "user, 2023/05/20 (Sat) 02:21: I was born in Leeds."
    item["metadata"] = {"speaker": "assistant"}
    assert memory_line(item) == "assistant: I was born in Leeds."


def _write_fake_run(tmp_path, prompt_lines, evidence_ids, question="(asked on 2023/05/30 (Tue) 23:40) Where was I born?",
                    output="Note\nAnswer: Leeds", qid="q_abc", complete=1.0):
    """A one-episode fixed8 run folder plus a cache folder whose prompt is rebuilt from the lines."""
    run = tmp_path / "runs" / "exp23_fake_f0" / "fixed8__fraction0.05"
    run.mkdir(parents=True)
    items = [_item("ep00000", f"s1:{n}", content, created_at=n + 1) for n, content in enumerate(prompt_lines)]
    obs = _item("ep00000", "q:0", f"Question: {question}", created_at=99)
    with gzip.open(run / "items.jsonl.gz", "wt") as fh:
        for it in items + [obs]:
            fh.write(json.dumps(it) + "\n")
    step = {"episode_id": "ep00000", "step": 5, "observation_id": "q:0", "scored": True,
            "memory": {"active_ids": [it["id"] for it in items] + ["q:0"]}, "agent_output": output}
    with gzip.open(run / "steps.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"episode_id": "ep00000", "step": 1, "scored": False, "memory": {"active_ids": []}}) + "\n")
        fh.write(json.dumps(step) + "\n")
    (run / "episodes.jsonl").write_text(json.dumps(
        {"episode_id": "ep00000", "seed": 0, "evidence_complete_rate": complete, "env_stats": {"episode": qid}}) + "\n")
    memory = "\n".join(memory_line(it) for it in items)
    prompt = f"Instructions here.\n\n## Memory\n{memory}\n\n## Current input\nQuestion: {question}\n\nNote:"
    cache = tmp_path / "cache"
    (cache / "00").mkdir(parents=True)
    (cache / "00" / "deadbeef.json").write_text(json.dumps({"prompt": prompt, "output": output}))
    verdicts = [{"reader": "fake", "arm": "fixed8", "question_id": qid, "type": "single-session-user",
                 "question": question, "gold": "Leeds", "answer": "Leeds", "unknown": False, "correct": True}]
    (tmp_path / "runs" / "exp23_verdicts.json").write_text(json.dumps(verdicts))
    return run, cache, prompt


def test_load_records_rebuilds_prompt_and_marks_evidence(tmp_path, monkeypatch):
    lines = ["Hello, how are you?", "I was born in Leeds, by the way.\nSecond line of the same turn.", "Nice weather."]
    run, cache, prompt = _write_fake_run(tmp_path, lines, evidence_ids=("s1:1",))
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids",
                        lambda data_path, question_id: (frozenset({"s1:1"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta",
                        lambda data_path, question_id: ("Leeds", "single-session-user", True))
    records, dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert dropped == {"no_cached_prompt": 0, "evidence_not_found": 0}
    [r] = records
    assert isinstance(r, PromptRecord)
    assert r.prompt == prompt and r.output == "Note\nAnswer: Leeds"
    assert r.memory_lines == tuple(memory_line(_item("ep00000", f"s1:{n}", c)) for n, c in enumerate(lines))
    assert r.evidence_lines == (r.memory_lines[1],)
    assert r.all_in_view is True and r.correct is True and r.unknown is False and r.gold == "Leeds"


def test_load_records_fallback_session_marks_every_in_view_turn(tmp_path, monkeypatch):
    lines = ["turn a", "turn b", "turn c"]
    run, cache, _ = _write_fake_run(tmp_path, lines, evidence_ids=())
    # a fallback session: any one of s1:0 / s1:2 counts, so both in-view lines are evidence
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids",
                        lambda data_path, question_id: (frozenset({"s1:0", "s1:2"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta",
                        lambda data_path, question_id: ("x", "multi-session", True))
    [r], _ = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert r.evidence_lines == (r.memory_lines[0], r.memory_lines[2])


def test_load_records_drops_prompt_without_cache_entry(tmp_path, monkeypatch):
    run, cache, _ = _write_fake_run(tmp_path, ["turn a"], evidence_ids=())
    (cache / "00" / "deadbeef.json").unlink()
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids", lambda d, q: (frozenset({"s1:0"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta", lambda d, q: ("x", "t", True))
    records, dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert records == [] and dropped["no_cached_prompt"] == 1


def test_load_records_drops_prompt_whose_evidence_is_not_in_view(tmp_path, monkeypatch):
    run, cache, _ = _write_fake_run(tmp_path, ["turn a"], evidence_ids=(), complete=0.0)
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids", lambda d, q: (frozenset({"zzz:9"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta", lambda d, q: ("x", "t", True))
    [r], dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert r.evidence_lines == () and r.all_in_view is False and dropped["evidence_not_found"] == 0


@needs_exp23
def test_real_fold0_qwen3b_records_all_match():
    records, dropped = load_records("qwen3b", 0)
    assert dropped["no_cached_prompt"] == 0
    assert len(records) == 100
    in_view = [r for r in records if r.all_in_view and r.answerable]
    assert all(r.evidence_lines for r in in_view)
    assert all(line in r.prompt for r in records for line in r.memory_lines)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'memctl.jacobian_data'`

- [ ] **Step 3: Write the implementation**

```python
# memctl/jacobian_data.py
"""Prompt records for the Jacobian lens (§33): the exact §23 `fixed8` prompt each reader saw, rebuilt
from the generation cache and the run traces, with the gold evidence lines marked.

The reader's prompt is in the generation cache (`cache/generations_exp23_<reader>/**.json`, key
`prompt`). It is matched to a question by rebuilding the memory section from the trace: the active
items at the scored step, in arrival order, in the compact "speaker, date: content" form of
agents/llm.py. Evidence lines are the in-view items that carry the gold evidence
(envs/longmemeval.py: `has_answer` turns, or any turn of an answer session with no marked turn).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance, parse_instance
from memctl.memory.items import label_prefix

MEMORY_HEADER = "## Memory\n"
INPUT_HEADER = "\n\n## Current input"


@dataclass(frozen=True)
class PromptRecord:
    question_id: str
    fold: int
    reader: str
    prompt: str
    output: str
    gold: str
    question_type: str
    answerable: bool
    memory_lines: tuple[str, ...]
    evidence_lines: tuple[str, ...]
    all_in_view: bool
    correct: bool | None
    unknown: bool | None


def memory_line(item: dict) -> str:
    """The compact line a `labels: compact` reader saw for a trace item (agents/llm.py compact_line)."""
    return label_prefix(item.get("metadata", {}), item.get("source_type", "observation")) + item["content"]


def _memory_section(prompt: str) -> str | None:
    if MEMORY_HEADER not in prompt or INPUT_HEADER not in prompt:
        return None
    return prompt.split(MEMORY_HEADER, 1)[1].split(INPUT_HEADER, 1)[0]


def cached_prompts(cache_dir: str) -> dict[str, tuple[str, str]]:
    """memory section -> (prompt, output) for every cached generation."""
    index: dict[str, tuple[str, str]] = {}
    for path in sorted(Path(cache_dir).glob("*/*.json")):
        try:
            entry = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        section = _memory_section(entry.get("prompt", ""))
        if section is not None:
            index[section] = (entry["prompt"], entry.get("output", ""))
    return index


def _read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=4)
def _instance_number(data_path: str) -> dict[str, int]:
    return {str(entry["question_id"]): number for number, entry in enumerate(_index(data_path))}


def evidence_item_ids(data_path: str, question_id: str) -> tuple[frozenset[str], ...]:
    """One set per evidence requirement: the marked turn, or every turn of an unmarked answer session."""
    episode, fallback = parse_instance(_instance(data_path, _instance_number(data_path)[question_id]))
    marked = tuple(frozenset({item_id}) for item_id in episode.questions[0].evidence_ids)
    return marked + tuple(frozenset(ids) for ids in fallback)


def question_meta(data_path: str, question_id: str) -> tuple[str, str, bool]:
    """(gold, question_type, answerable)."""
    instance = _instance(data_path, _instance_number(data_path)[question_id])
    return str(instance["answer"]), str(instance["question_type"]), not str(question_id).endswith("_abs")


def _verdicts(runs_dir: str, reader: str) -> dict[str, dict]:
    path = Path(runs_dir) / "exp23_verdicts.json"
    if not path.exists():
        return {}
    return {v["question_id"]: v for v in json.loads(path.read_text()) if v["reader"] == reader and v["arm"] == "fixed8"}


def load_records(reader: str, fold: int, runs_dir: str = "runs", cache_dir: str | None = None,
                 data_path: str = "data/longmemeval/longmemeval_s_cleaned.json",
                 arm: str = "fixed8__fraction0.05") -> tuple[list[PromptRecord], dict[str, int]]:
    run = Path(runs_dir) / f"exp23_{reader}_f{fold}" / arm
    cache = cached_prompts(cache_dir or f"cache/generations_exp23_{reader}")
    items: dict[tuple[str, str], dict] = {(it["episode_id"], it["id"]): it for it in _read_jsonl(run / "items.jsonl.gz")}
    scored = {s["episode_id"]: s for s in _read_jsonl(run / "steps.jsonl.gz") if s.get("scored")}
    episodes = {e["episode_id"]: e for e in _read_jsonl(run / "episodes.jsonl")}
    verdicts = _verdicts(runs_dir, reader)
    dropped = {"no_cached_prompt": 0, "evidence_not_found": 0}
    records = []
    for episode_id, step in scored.items():
        question_id = str(episodes[episode_id]["env_stats"]["episode"])
        active = [items[(episode_id, i)] for i in step["memory"]["active_ids"] if i != step["observation_id"]]
        active.sort(key=lambda it: it.get("created_at", 0))
        lines = tuple(memory_line(it) for it in active)
        hit = cache.get("\n".join(lines))
        if hit is None:
            dropped["no_cached_prompt"] += 1
            continue
        prompt, output = hit
        gold, question_type, answerable = question_meta(data_path, question_id)
        complete = (episodes[episode_id].get("evidence_complete_rate") or 0.0) >= 1.0
        requirement_ids = evidence_item_ids(data_path, question_id) if answerable else ()
        needed = set().union(*requirement_ids) if requirement_ids else set()
        evidence = tuple(line for it, line in zip(active, lines) if it["id"] in needed)
        if complete and answerable and not evidence:
            dropped["evidence_not_found"] += 1
            continue
        verdict = verdicts.get(question_id, {})
        records.append(PromptRecord(
            question_id, fold, reader, prompt, output, gold, question_type, answerable, lines, evidence, complete,
            verdict.get("correct"), verdict.get("unknown"),
        ))
    records.sort(key=lambda r: r.question_id)
    return records, dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: 5 passed (the real-data test passes on this machine; it is skipped elsewhere)

- [ ] **Step 5: Run the boundary tests**

Run: `.venv/bin/pytest tests/test_import_boundaries.py -q`
Expected: the same two pre-existing failures as before, no new one.

- [ ] **Step 6: Commit**

```bash
git add memctl/jacobian_data.py tests/test_jacobian.py
git commit -m "§33: prompt records from the §23 cache and traces (jacobian_data)"
```

---

### Task 2: Prompt segmentation and token regions

**Files:**
- Create: `memctl/analysis/jacobian.py`
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 (pure strings).
- Produces:
  ```python
  REGIONS = ("template", "instruction", "evidence", "distractor", "question")

  def segment_prompt(prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...]) -> list[tuple[str, int, int]]
  # char spans (region, start, end) that tile [0, len(prompt)) exactly, in order

  def token_regions(offsets: list[tuple[int, int]], prompt_start: int, prompt_len: int,
                    segments: list[tuple[str, int, int]]) -> list[str]
  # one region name per token of the full chat text; tokens outside the prompt are "template"
  ```

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jacobian.py
from memctl.analysis.jacobian import REGIONS, segment_prompt, token_regions

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nassistant, d1: turn b\nhas newline\n"
          "user, d2: turn c\n\n## Current input\nQuestion: what?\n\nNote:")
LINES = ("user, d1: turn a", "assistant, d1: turn b\nhas newline", "user, d2: turn c")


def test_segment_prompt_tiles_the_prompt_in_order():
    segments = segment_prompt(PROMPT, LINES, evidence_lines=(LINES[1],))
    assert segments[0][1] == 0 and segments[-1][2] == len(PROMPT)
    assert all(a[2] == b[1] for a, b in zip(segments, segments[1:]))
    assert [s[0] for s in segments] == ["instruction", "distractor", "evidence", "distractor", "question"]
    region_text = {s[0]: PROMPT[s[1]:s[2]] for s in segments}
    assert region_text["evidence"].startswith("assistant, d1: turn b\nhas newline")
    assert region_text["question"].startswith("\n\n## Current input")
    assert region_text["instruction"] == "Use only the memory.\n\n## Memory\n"


def test_segment_prompt_handles_a_line_with_a_newline_by_exact_search():
    # a naive split on "\n" would see four memory lines; exact search sees three
    segments = segment_prompt(PROMPT, LINES, evidence_lines=())
    assert sum(1 for s in segments if s[0] == "distractor") == 3


def test_segment_prompt_rejects_missing_line():
    with pytest.raises(ValueError):
        segment_prompt(PROMPT, LINES + ("user, d9: not there",), evidence_lines=())


def test_token_regions_assigns_by_token_start_and_marks_template():
    segments = [("instruction", 0, 5), ("evidence", 5, 9), ("question", 9, 12)]
    # chat text = "<s>" + prompt(12 chars) + "<e>"; offsets in chat-text coordinates
    offsets = [(0, 3), (3, 5), (5, 8), (8, 10), (10, 14), (14, 15), (15, 18)]
    regions = token_regions(offsets, prompt_start=3, prompt_len=12, segments=segments)
    assert regions == ["template", "instruction", "instruction", "evidence", "evidence", "question", "template"]
    assert set(regions) <= set(REGIONS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k "segment or token_regions"`
Expected: FAIL with `ImportError` on `memctl.analysis.jacobian`

- [ ] **Step 3: Write the implementation**

```python
# memctl/analysis/jacobian.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add memctl/analysis/jacobian.py tests/test_jacobian.py
git commit -m "§33: prompt segmentation and token regions"
```

---

### Task 3: Chat wrapping, teacher-forced targets and the gradient pass

**Files:**
- Modify: `memctl/analysis/jacobian.py` (append)
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: `segment_prompt`, `token_regions` (Task 2).
- Produces:
  ```python
  TARGETS = ("gold", "margin", "own")

  def wrap_chat(tokenizer, prompt: str) -> str          # chat text; prompt appears verbatim inside it
  def continuation_ids(tokenizer, target: str, gold: str, output: str) -> tuple[list[int], int]
  # (ids appended after the chat text, index into those ids where scoring starts)
  # gold:   tok("Answer:") + tok(" " + gold), scoring starts at len(tok("Answer:"))
  # margin: tok("Answer:") + [first id of tok(" " + gold)], scoring index = len(tok("Answer:"))
  # own:    tok(output), scoring starts at 0
  def unknown_id(tokenizer) -> int                       # first id of tok(" unknown")

  @dataclass
  class JacobianResult:
      regions: list[str]            # per prompt-side token (chat text only, continuation excluded)
      g_l2: np.ndarray              # per-token L2 norm of d target / d embedding
      g_gxi: np.ndarray             # per-token |gradient · embedding|
      target_value: float
      n_tokens: int

  def jacobian_pass(model, tokenizer, prompt: str, memory_lines, evidence_lines, gold: str, output: str,
                    targets: tuple[str, ...] = ("gold", "margin")) -> dict[str, JacobianResult]
  # one forward for gold+margin (shared continuation prefix), one for own
  def target_value(model, tokenizer, prompt: str, gold: str, output: str, target: str) -> float
  # no-grad value of a target (used by the ablation check)
  ```

- [ ] **Step 1: Write the failing tests** (a tiny random Qwen2 model plus the real 0.5B tokenizer; skipped offline)

```python
# append to tests/test_jacobian.py
import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.analysis.jacobian import (JacobianResult, TARGETS, continuation_ids, jacobian_pass, target_value,
                                      unknown_id, wrap_chat)


@pytest.fixture(scope="module")
def tiny():
    """The Qwen2.5 tokenizer (chat template) with a 2-layer random Qwen2 model of the same vocabulary."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:  # offline
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    return model, tok


def test_wrap_chat_contains_prompt_verbatim_and_generation_prompt(tiny):
    model, tok = tiny
    text = wrap_chat(tok, PROMPT)
    assert PROMPT in text
    assert text.rstrip().endswith("<|im_start|>assistant")


def test_continuation_ids_gold_scores_exactly_the_gold_tokens(tiny):
    _, tok = tiny
    ids, start = continuation_ids(tok, "gold", gold="4 hours", output="")
    assert ids[:start] == tok("Answer:", add_special_tokens=False).input_ids
    assert ids[start:] == tok(" 4 hours", add_special_tokens=False).input_ids
    m_ids, m_start = continuation_ids(tok, "margin", gold="4 hours", output="")
    assert m_ids == ids[: start + 1] and m_start == start
    o_ids, o_start = continuation_ids(tok, "own", gold="", output="Note\nAnswer: unknown")
    assert o_ids == tok("Note\nAnswer: unknown", add_special_tokens=False).input_ids and o_start == 0
    assert unknown_id(tok) == tok(" unknown", add_special_tokens=False).input_ids[0]


def test_jacobian_pass_shapes_and_target_values(tiny):
    model, tok = tiny
    out = jacobian_pass(model, tok, PROMPT, LINES, (LINES[1],), gold="4 hours", output="Answer: nope", targets=TARGETS)
    assert set(out) == set(TARGETS)
    for result in out.values():
        assert isinstance(result, JacobianResult)
        assert len(result.regions) == result.n_tokens == len(result.g_l2) == len(result.g_gxi)
        assert np.isfinite(result.g_l2).all() and (result.g_l2 >= 0).all()
        assert "evidence" in result.regions and "template" in result.regions
    # the gold target equals a direct teacher-forced log-prob sum
    text = wrap_chat(tok, PROMPT)
    ids, start = continuation_ids(tok, "gold", "4 hours", "")
    full = tok(text, add_special_tokens=False).input_ids + ids
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([full])).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    n = len(full) - len(ids)
    expected = sum(logp[n + j - 1, ids[j]].item() for j in range(start, len(ids)))
    assert out["gold"].target_value == pytest.approx(expected, rel=1e-4, abs=1e-4)
    assert target_value(model, tok, PROMPT, "4 hours", "", "gold") == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_gradient_is_nonzero_only_where_it_should_be(tiny):
    model, tok = tiny
    out = jacobian_pass(model, tok, PROMPT, LINES, (LINES[1],), gold="4 hours", output="", targets=("gold",))
    assert out["gold"].g_l2.sum() > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k "wrap_chat or continuation or jacobian_pass or gradient"`
Expected: FAIL with `ImportError` (names not defined)

- [ ] **Step 3: Write the implementation**

```python
# append to memctl/analysis/jacobian.py
from dataclasses import dataclass

TARGETS = ("gold", "margin", "own")
ANSWER_PREFIX = "Answer:"


def wrap_chat(tokenizer, prompt: str) -> str:
    """The chat text the vLLM server built: one user message, generation prompt, no thinking (as llm.py)."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def unknown_id(tokenizer) -> int:
    return _ids(tokenizer, " unknown")[0]


def continuation_ids(tokenizer, target: str, gold: str, output: str) -> tuple[list[int], int]:
    if target == "own":
        return _ids(tokenizer, output), 0
    prefix = _ids(tokenizer, ANSWER_PREFIX)
    answer = _ids(tokenizer, " " + gold)
    if target == "gold":
        return prefix + answer, len(prefix)
    if target == "margin":
        return prefix + answer[:1], len(prefix)
    raise ValueError(f"unknown target {target!r} (known: {TARGETS})")


@dataclass
class JacobianResult:
    regions: list[str]
    g_l2: np.ndarray
    g_gxi: np.ndarray
    target_value: float
    n_tokens: int


def _prompt_tokens(tokenizer, prompt: str, memory_lines, evidence_lines):
    text = wrap_chat(tokenizer, prompt)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    segments = segment_prompt(prompt, memory_lines, evidence_lines)
    regions = token_regions(encoded["offset_mapping"], text.index(prompt), len(prompt), segments)
    return text, list(encoded["input_ids"]), regions


def _scalar(logits, full_ids: list[int], n_prompt: int, cont: list[int], start: int, target: str, unk: int):
    import torch

    logp = torch.log_softmax(logits.float(), -1)
    if target == "margin":
        position = n_prompt + start - 1
        return logits[position, cont[start]].float() - logits[position, unk].float()
    return sum(logp[n_prompt + j - 1, cont[j]] for j in range(start, len(cont)))


def jacobian_pass(model, tokenizer, prompt: str, memory_lines, evidence_lines, gold: str, output: str,
                  targets: tuple[str, ...] = ("gold", "margin")) -> dict[str, JacobianResult]:
    import torch

    text, prompt_ids, regions = _prompt_tokens(tokenizer, prompt, memory_lines, evidence_lines)
    n_prompt = len(prompt_ids)
    unk = unknown_id(tokenizer)
    device = model.get_input_embeddings().weight.device
    results: dict[str, JacobianResult] = {}
    # gold and margin share the continuation prefix "Answer: <first gold token>", so one forward serves both
    groups = []
    if "gold" in targets or "margin" in targets:
        groups.append(tuple(t for t in ("gold", "margin") if t in targets))
    if "own" in targets:
        groups.append(("own",))
    for group in groups:
        cont, start = continuation_ids(tokenizer, "gold" if group[0] != "own" else "own", gold, output)
        if group == ("margin",):
            cont, start = continuation_ids(tokenizer, "margin", gold, output)
        full = torch.tensor([prompt_ids + cont], device=device)
        embeds = model.get_input_embeddings()(full).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeds).logits[0]
        for number, target in enumerate(group):
            t_cont = cont if target != "margin" else cont[: start + 1]
            value = _scalar(logits, prompt_ids + cont, n_prompt, t_cont, start, target, unk)
            (grad,) = torch.autograd.grad(value, embeds, retain_graph=number < len(group) - 1)
            g = grad[0, :n_prompt].float()
            x = embeds[0, :n_prompt].detach().float()
            results[target] = JacobianResult(
                regions, g.norm(dim=-1).cpu().numpy(), (g * x).sum(-1).abs().cpu().numpy(), float(value), n_prompt,
            )
        del logits, embeds
    return results


def target_value(model, tokenizer, prompt: str, gold: str, output: str, target: str) -> float:
    """The scalar target without gradients (the ablation check)."""
    import torch

    text = wrap_chat(tokenizer, prompt)
    prompt_ids = _ids(tokenizer, text)
    cont, start = continuation_ids(tokenizer, target, gold, output)
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids + cont], device=device)).logits[0]
        return float(_scalar(logits, prompt_ids + cont, len(prompt_ids), cont, start, target, unknown_id(tokenizer)))
```

Note for the implementer: `_scalar` for `margin` scores at the position that predicts `cont[start]`, which is the first gold token; the margin continuation is the prefix plus that one token, so the scored logits come from the prompt and `Answer:` only. In `jacobian_pass` the gold/margin group uses the gold continuation for the forward pass and slices for the margin target, so the two share one forward. `target_value` for `own` passes `output` through `continuation_ids`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass (the tokenizer download is about 10 MB on first run)

- [ ] **Step 5: Commit**

```bash
git add memctl/analysis/jacobian.py tests/test_jacobian.py
git commit -m "§33: chat wrapping, teacher-forced targets and the input-embedding gradient pass"
```

---

### Task 4: Region measures and the ablation prompt

**Files:**
- Modify: `memctl/analysis/jacobian.py` (append)
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: `JacobianResult` (Task 3), `REGIONS` (Task 2).
- Produces:
  ```python
  def region_shares(g: np.ndarray, regions: list[str]) -> dict[str, float]
  # share of gradient mass per non-template region; sums to 1; a region with no tokens has share 0.0
  def region_counts(regions: list[str]) -> dict[str, int]
  def evidence_lift(shares: dict[str, float], counts: dict[str, int]) -> float | None
  # shares["evidence"] / (counts["evidence"] / non-template tokens); None if no evidence tokens
  def topk_fraction(g: np.ndarray, regions: list[str], k: int = 50) -> float
  # fraction of the k largest-g non-template tokens that are evidence
  def measures(result: JacobianResult, k: int = 50) -> dict
  # {"l2": {"shares": ..., "lift": ..., "topk": ...}, "gxi": {...}, "counts": ..., "target_value": ..., "n_tokens": ...}
  def ablate_evidence(prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...]) -> str
  # the prompt with each evidence line (and the newline separating it) removed, cut on segment_prompt's exact spans; ValueError on a missing line
  ```

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jacobian.py
from memctl.analysis.jacobian import ablate_evidence, evidence_lift, measures, region_counts, region_shares, topk_fraction


def test_region_shares_sum_to_one_and_ignore_template():
    regions = ["template", "instruction", "evidence", "evidence", "distractor", "question", "template"]
    g = np.array([100.0, 1.0, 2.0, 2.0, 4.0, 1.0, 100.0])
    shares = region_shares(g, regions)
    assert shares == pytest.approx({"instruction": 0.1, "evidence": 0.4, "distractor": 0.4, "question": 0.1})
    assert region_counts(regions) == {"template": 2, "instruction": 1, "evidence": 2, "distractor": 1, "question": 1}
    assert evidence_lift(shares, region_counts(regions)) == pytest.approx(0.4 / (2 / 5))
    assert evidence_lift({"evidence": 0.0}, {"evidence": 0, "instruction": 3}) is None


def test_topk_fraction_counts_evidence_among_largest():
    regions = ["template", "instruction", "evidence", "evidence", "distractor", "question"]
    g = np.array([9.0, 1.0, 5.0, 4.0, 3.0, 0.5])
    assert topk_fraction(g, regions, k=2) == 1.0
    assert topk_fraction(g, regions, k=4) == 0.5
    assert topk_fraction(g, regions, k=50) == pytest.approx(2 / 5)


def test_measures_bundles_both_variants():
    result = JacobianResult(["template", "evidence", "distractor"], np.array([1.0, 3.0, 1.0]), np.array([1.0, 1.0, 3.0]), -2.5, 3)
    m = measures(result, k=1)
    assert m["l2"]["shares"]["evidence"] == 0.75 and m["gxi"]["shares"]["evidence"] == 0.25
    assert m["l2"]["topk"] == 1.0 and m["gxi"]["topk"] == 0.0
    assert m["target_value"] == -2.5 and m["n_tokens"] == 3 and m["counts"]["evidence"] == 1


def test_ablate_evidence_removes_the_lines_and_keeps_the_rest_intact():
    ablated = ablate_evidence(PROMPT, (LINES[1],))
    assert LINES[1] not in ablated
    assert ablated == ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nuser, d2: turn c\n\n## Current input\n"
                       "Question: what?\n\nNote:")
    assert ablate_evidence(PROMPT, ()) == PROMPT
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k "shares or topk or measures or ablate"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

```python
# append to memctl/analysis/jacobian.py
CONTENT_REGIONS = tuple(r for r in REGIONS if r != "template")


def region_counts(regions: list[str]) -> dict[str, int]:
    return {name: sum(1 for r in regions if r == name) for name in REGIONS if name in regions}


def region_shares(g: np.ndarray, regions: list[str]) -> dict[str, float]:
    mask = np.array([r != "template" for r in regions])
    total = float(g[mask].sum())
    shares = {}
    for name in CONTENT_REGIONS:
        chosen = np.array([r == name for r in regions])
        shares[name] = float(g[chosen].sum()) / total if total > 0 and chosen.any() else 0.0
    return shares


def evidence_lift(shares: dict[str, float], counts: dict[str, int]) -> float | None:
    evidence = counts.get("evidence", 0)
    content = sum(n for name, n in counts.items() if name != "template")
    if evidence == 0 or content == 0:
        return None
    return shares.get("evidence", 0.0) / (evidence / content)


def topk_fraction(g: np.ndarray, regions: list[str], k: int = 50) -> float:
    content = [i for i, r in enumerate(regions) if r != "template"]
    top = sorted(content, key=lambda i: -float(g[i]))[: min(k, len(content))]
    return sum(1 for i in top if regions[i] == "evidence") / len(top) if top else 0.0


def measures(result: JacobianResult, k: int = 50) -> dict:
    counts = region_counts(result.regions)
    out = {"counts": counts, "target_value": result.target_value, "n_tokens": result.n_tokens, "k": k}
    for name, g in (("l2", result.g_l2), ("gxi", result.g_gxi)):
        shares = region_shares(g, result.regions)
        out[name] = {"shares": shares, "lift": evidence_lift(shares, counts), "topk": topk_fraction(g, result.regions, k)}
    return out


def ablate_evidence(prompt: str, evidence_lines: tuple[str, ...]) -> str:
    """The prompt with every evidence line removed, along with the newline that separated it."""
    for line in evidence_lines:
        if "\n" + line in prompt:
            prompt = prompt.replace("\n" + line, "", 1)
        else:  # the first memory line has the header before it, not a newline
            prompt = prompt.replace(line + "\n", "", 1)
    return prompt
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add memctl/analysis/jacobian.py tests/test_jacobian.py
git commit -m "§33: region shares, evidence lift, top-k fraction and the ablation prompt"
```

---

### Task 5: Claim statistics

**Files:**
- Create: `memctl/analysis/jacobian_stats.py`
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: `memctl.analysis.stats.paired_bootstrap` (existing: `paired_bootstrap(differences, samples, seed) -> {"n","mean","ci_low","ci_high","p"}`).
- Produces:
  ```python
  def auc(positives: list[float], negatives: list[float]) -> float          # Mann-Whitney AUC, ties count 0.5
  def bootstrap_auc(positives, negatives, samples: int = 10000, seed: int = 0) -> dict   # {"n_pos","n_neg","auc","ci_low","ci_high"}
  def bootstrap_spearman(x: list[float], y: list[float], samples: int = 10000, seed: int = 0) -> dict  # {"n","rho","ci_low","ci_high"}
  def paired_claim(a: dict[str, float], b: dict[str, float], samples: int = 10000, seed: int = 0) -> dict
  # a - b over shared keys; the paired_bootstrap dict plus "verdict": "SHOWN" if ci_low > 0 else "NOT SHOWN"
  ```

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_jacobian.py
from memctl.analysis.jacobian_stats import auc, bootstrap_auc, bootstrap_spearman, paired_claim


def test_auc_is_mann_whitney_with_half_credit_for_ties():
    assert auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_bootstrap_auc_interval_covers_point_and_is_in_unit_range():
    rng = np.random.default_rng(0)
    pos = list(rng.normal(1.0, 1.0, 60)); neg = list(rng.normal(0.0, 1.0, 60))
    r = bootstrap_auc(pos, neg, samples=500)
    assert r["n_pos"] == 60 and r["n_neg"] == 60
    assert 0 <= r["ci_low"] <= r["auc"] <= r["ci_high"] <= 1 and r["ci_low"] > 0.5


def test_bootstrap_spearman_detects_a_monotone_relation():
    x = list(range(50)); y = [v ** 2 + (1 if v % 2 else -1) for v in x]
    r = bootstrap_spearman(x, y, samples=500)
    assert r["n"] == 50 and r["rho"] > 0.9 and r["ci_low"] > 0.5


def test_paired_claim_uses_shared_keys_and_gives_a_verdict():
    a = {"q1": 0.5, "q2": 0.6, "q3": 0.7, "q4": 0.8, "zz": 9.0}
    b = {"q1": 0.1, "q2": 0.2, "q3": 0.3, "q4": 0.4}
    r = paired_claim(a, b, samples=500)
    assert r["n"] == 4 and r["mean"] == pytest.approx(0.4) and r["verdict"] == "SHOWN"
    assert paired_claim(b, a, samples=500)["verdict"] == "NOT SHOWN"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k "auc or spearman or paired_claim"`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# memctl/analysis/jacobian_stats.py
"""Statistics for the §33 claims: paired region-share differences, an AUC for within-reader
separation, and a rank correlation for the ablation check. Bootstraps resample questions."""

from __future__ import annotations

import random

from scipy.stats import spearmanr

from memctl.analysis.stats import paired_bootstrap


def auc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def _interval(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def bootstrap_auc(positives: list[float], negatives: list[float], samples: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    draws = [auc(rng.choices(positives, k=len(positives)), rng.choices(negatives, k=len(negatives))) for _ in range(samples)]
    low, high = _interval(draws)
    return {"n_pos": len(positives), "n_neg": len(negatives), "auc": auc(positives, negatives), "ci_low": low, "ci_high": high}


def bootstrap_spearman(x: list[float], y: list[float], samples: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    pairs = list(zip(x, y))
    draws = []
    for _ in range(samples):
        sample = rng.choices(pairs, k=len(pairs))
        draws.append(float(spearmanr([a for a, _ in sample], [b for _, b in sample]).statistic))
    low, high = _interval([d for d in draws if d == d])
    return {"n": len(pairs), "rho": float(spearmanr(x, y).statistic), "ci_low": low, "ci_high": high}


def paired_claim(a: dict[str, float], b: dict[str, float], samples: int = 10000, seed: int = 0) -> dict:
    keys = sorted(set(a) & set(b))
    result = paired_bootstrap([a[k] - b[k] for k in keys], samples=samples, seed=seed)
    result["verdict"] = "SHOWN" if result["ci_low"] is not None and result["ci_low"] > 0 else "NOT SHOWN"
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add memctl/analysis/jacobian_stats.py tests/test_jacobian.py
git commit -m "§33: AUC, Spearman and paired-claim bootstraps"
```

---

### Task 6: The run script (resumable, per-question outputs, regeneration check)

**Files:**
- Create: `scripts/jacobian_lens.py`
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: `load_records` (Task 1), `jacobian_pass`, `measures`, `ablate_evidence`, `target_value`, `wrap_chat` (Tasks 3–4).
- Ruling (controller): when `record.output.strip()` is empty, skip the `own` target for that record and write `"own_skipped": true` into the gold json instead; an all-zero gradient would give 0/0 shares. `ablate_evidence` takes `(prompt, memory_lines, evidence_lines)` after the Task 4 review.
- Produces files under `--out` (default `runs/exp33_jacobian/<reader>`):
  - `<question_id>.<target>.json`: `{"question_id","fold","reader","target","model_id","measures": measures(...), "elapsed_s"}`
  - `<question_id>.<target>.npz`: `g_l2`, `g_gxi` (float32), `regions` (int8 codes into `REGIONS`)
  - `<question_id>.ablation.json` (with `--ablate`): `{"question_id","gold_full","gold_ablated","delta"}`
  - `_timing.json`: per-question seconds, peak RSS, model id, device, dtype
  - `_regen_check.json` (with `--regen-check N`): matches, N, per-prompt pairs
  - `_dropped.json`: the `dropped` counts per fold
- CLI: `scripts/jacobian_lens.py --reader qwen3b --folds 0,1 [--targets gold,margin,own] [--ablate] [--out DIR] [--limit N] [--device auto|cpu|cuda] [--dtype bfloat16|float32] [--model-id ID] [--regen-check N] [--regen-min M] [--k 50]`
- Reader to model id: `{"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}`; `qwen05b` reads the `qwen3b` prompts and verdicts (`--records-from qwen3b`), since no 0.5B ran in §23.

- [ ] **Step 1: Write the failing tests** (the script's pure helpers are importable; the loop is tested with a fake model)

```python
# append to tests/test_jacobian.py
import importlib.util, sys


def _script():
    spec = importlib.util.spec_from_file_location("jacobian_lens", "scripts/jacobian_lens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_paths_and_resume_skip(tmp_path):
    s = _script()
    out = tmp_path / "o"
    assert s.output_path(out, "q1", "gold").name == "q1.gold.json"
    assert not s.is_done(out, "q1", ("gold", "margin"))
    out.mkdir()
    s.write_json(s.output_path(out, "q1", "gold"), {"ok": 1})
    s.write_json(s.output_path(out, "q1", "margin"), {"ok": 1})
    assert s.is_done(out, "q1", ("gold", "margin"))
    (out / "q1.margin.json").write_text("{not json")  # a partial file from a crash
    assert not s.is_done(out, "q1", ("gold", "margin"))


def test_write_json_is_atomic(tmp_path):
    s = _script()
    path = tmp_path / "x.json"
    s.write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_run_records_writes_measures_npz_and_ablation(tiny, tmp_path):
    s = _script()
    model, tok = tiny
    record = PromptRecord("qX", 0, "fake", PROMPT, "Answer: nope", "4 hours", "single-session-user", True,
                          LINES, (LINES[1],), True, False, False)
    timing = s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=True, k=5, model_id="tiny")
    assert set(p.name for p in tmp_path.glob("qX.*")) == {"qX.gold.json", "qX.gold.npz", "qX.margin.json", "qX.margin.npz",
                                                          "qX.own.json", "qX.own.npz", "qX.ablation.json"}
    gold = json.loads((tmp_path / "qX.gold.json").read_text())
    assert gold["measures"]["l2"]["shares"]["evidence"] >= 0 and gold["target"] == "gold" and gold["model_id"] == "tiny"
    arrays = np.load(tmp_path / "qX.gold.npz")
    assert arrays["g_l2"].dtype == np.float32 and arrays["regions"].dtype == np.int8
    ablation = json.loads((tmp_path / "qX.ablation.json").read_text())
    assert ablation["delta"] == pytest.approx(ablation["gold_full"] - ablation["gold_ablated"])
    assert timing["qX"] > 0
    # a second call skips everything
    assert s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=True, k=5, model_id="tiny") == {}


def test_regen_check_counts_matches(tiny, tmp_path):
    s = _script()
    model, tok = tiny
    record = PromptRecord("qX", 0, "fake", PROMPT, "whatever the cache said", "4 hours", "t", True, LINES, (), True, None, None)
    result = s.regen_check(model, tok, [record], n=1, max_new_tokens=4)
    assert result["n"] == 1 and result["matches"] in (0, 1) and len(result["pairs"]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k "output_paths or atomic or run_records or regen"`
Expected: FAIL (file not found)

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""§33 Jacobian lens: per-question input-embedding gradients of a reader on its cached §23 prompts.

    .venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 5 --device cpu
    python scripts/jacobian_lens.py --reader qwen3b --folds 0,1,2,3,4 --ablate --regen-check 20 --device cuda

Outputs one json (measures) and one npz (per-token gradient mass) per (question, target) under
runs/exp33_jacobian/<reader>/, plus _timing.json, _dropped.json and _regen_check.json. Resumable: existing,
readable outputs are skipped. Never deletes anything.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian import REGIONS, ablate_evidence, jacobian_pass, measures, target_value, wrap_chat  # noqa: E402
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

MODEL_IDS = {
    "qwen05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
REGION_CODE = {name: code for code, name in enumerate(REGIONS)}


def output_path(out: Path, question_id: str, target: str) -> Path:
    return Path(out) / f"{question_id}.{target}.json"


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _readable(path: Path) -> bool:
    try:
        json.loads(Path(path).read_text())
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def is_done(out: Path, question_id: str, targets: tuple[str, ...], ablate: bool = False) -> bool:
    """Done when every target's json is readable. The npz is written before the json, so a readable json
    means its npz exists; a crash between them leaves no json and the question is redone."""
    paths = [output_path(out, question_id, t) for t in targets]
    if ablate:
        paths.append(output_path(out, question_id, "ablation"))
    return all(_readable(p) for p in paths)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_records(model, tokenizer, records: list[PromptRecord], out: Path, targets: tuple[str, ...], ablate: bool,
                k: int, model_id: str) -> dict[str, float]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    for number, record in enumerate(records):
        if is_done(out, record.question_id, targets, ablate):
            continue
        started = time.perf_counter()
        results = jacobian_pass(model, tokenizer, record.prompt, record.memory_lines, record.evidence_lines,
                                record.gold, record.output, targets=targets)
        for target, result in results.items():
            np.savez_compressed(output_path(out, record.question_id, target).with_suffix(".npz"),
                                g_l2=result.g_l2.astype(np.float32), g_gxi=result.g_gxi.astype(np.float32),
                                regions=np.array([REGION_CODE[r] for r in result.regions], dtype=np.int8))
            write_json(output_path(out, record.question_id, target), {
                "question_id": record.question_id, "fold": record.fold, "reader": record.reader, "target": target,
                "model_id": model_id, "measures": measures(result, k=k), "elapsed_s": time.perf_counter() - started,
            })
        if ablate:
            full = target_value(model, tokenizer, record.prompt, record.gold, record.output, "gold")
            ablated = target_value(model, tokenizer, ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines),
                                   record.gold, record.output, "gold")
            write_json(output_path(out, record.question_id, "ablation"), {
                "question_id": record.question_id, "gold_full": full, "gold_ablated": ablated, "delta": full - ablated,
                "n_evidence_lines": len(record.evidence_lines),
            })
        timing[record.question_id] = time.perf_counter() - started
        print(f"[{number + 1}/{len(records)}] {record.question_id} {timing[record.question_id]:.1f}s "
              f"rss {peak_rss_mb():.0f} MB", flush=True)
    return timing


def regen_check(model, tokenizer, records: list[PromptRecord], n: int, max_new_tokens: int = 256) -> dict:
    """Greedy regeneration of the first n prompts against the cached vLLM outputs."""
    import torch

    pairs = []
    device = model.get_input_embeddings().weight.device
    for record in records[:n]:
        ids = tokenizer(wrap_chat(tokenizer, record.prompt), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                       do_sample=False)
        text = tokenizer.decode(generated[0][ids.shape[1]:], skip_special_tokens=True).strip()
        pairs.append({"question_id": record.question_id, "cached": record.output, "regenerated": text,
                      "match": text == record.output.strip()})
    return {"n": len(pairs), "matches": sum(p["match"] for p in pairs), "pairs": pairs}


def load_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--records-from", default=None, help="reader whose §23 prompts to use (default: --reader)")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--targets", default="gold,margin,own")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N records per fold (development)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--regen-check", type=int, default=0)
    parser.add_argument("--regen-min", type=int, default=15, help="abort if fewer than this many of the regen prompts match")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args(argv)

    model_id = args.model_id or MODEL_IDS[args.reader]
    out = Path(args.out or f"runs/exp33_jacobian/{args.reader}")
    out.mkdir(parents=True, exist_ok=True)
    targets = tuple(args.targets.split(","))
    records, dropped = [], {}
    for fold in (int(f) for f in args.folds.split(",")):
        fold_records, fold_dropped = load_records(args.records_from or args.reader, fold)
        records.extend(fold_records[: args.limit] if args.limit else fold_records)
        dropped[str(fold)] = fold_dropped
    write_json(out / "_dropped.json", dropped)
    print(f"{len(records)} records, dropped {dropped}; model {model_id}", flush=True)

    model, tokenizer, device = load_model(model_id, args.device, args.dtype)
    if args.regen_check:
        check = regen_check(model, tokenizer, records, args.regen_check)
        write_json(out / "_regen_check.json", check)
        print(f"regen check: {check['matches']}/{check['n']} match", flush=True)
        if check["matches"] < min(args.regen_min, check["n"]):
            print("regen check FAILED; stopping", flush=True)
            return 2
    started = time.perf_counter()
    timing = run_records(model, tokenizer, records, out, targets, args.ablate, args.k, model_id)
    summary = {"model_id": model_id, "device": device, "dtype": args.dtype, "questions": len(timing),
               "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(), "per_question": timing}
    existing = json.loads((out / "_timing.json").read_text()) if (out / "_timing.json").exists() else {}
    existing.setdefault("runs", []).append(summary)
    write_json(out / "_timing.json", existing)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass

- [ ] **Step 5: Smoke the CLI on CPU with the 0.5B on three real prompts** (downloads about 1 GB once)

Run: `.venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 3 --device cpu --dtype float32 --ablate --regen-check 1 --regen-min 0 --out runs/exp33_jacobian/_smoke_qwen05b`
Expected: three questions processed, files written, a seconds-per-question figure printed, exit 0. Then run it again and confirm `questions: 0` (resume).

- [ ] **Step 6: Commit**

```bash
git add scripts/jacobian_lens.py tests/test_jacobian.py
git commit -m "§33: resumable Jacobian run script with regeneration check"
```

---

### Task 7: The report pipeline

**Files:**
- Create: `runs/_pipelines/exp33_report.py` (note: `runs/` is gitignored; also copy to `scripts/exp33_report.py` so it is versioned, and have the `runs/_pipelines` file import from it: `from scripts.exp33_report import main`). Simpler: create only `scripts/exp33_report.py` and document the command. Do that.
- Create: `scripts/exp33_report.py`
- Test: `tests/test_jacobian.py` (append)

**Interfaces:**
- Consumes: `load_records` (Task 1), the output files of Task 6, `paired_claim`, `bootstrap_auc`, `bootstrap_spearman` (Task 5).
- Produces:
  ```python
  def load_outputs(out_dir: Path, folds: set[int], target: str) -> dict[str, dict]   # question_id -> measures json
  def load_ablation(out_dir: Path, folds: set[int]) -> dict[str, float]              # question_id -> delta
  def population(records_by_reader: dict[str, list[PromptRecord]]) -> list[str]
  # answerable, all_in_view under fixed8, present for every reader given
  def build_report(readers: tuple[str, str], folds: set[int], variant: str, k_label: str, out_root: Path,
                   samples: int = 10000) -> tuple[str, dict]    # (markdown, numbers)
  ```
- CLI: `scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2 --out runs/exp33_report_dev.md`

- [ ] **Step 1: Write the failing tests** (synthetic outputs in `tmp_path`)

```python
# append to tests/test_jacobian.py
def _fake_outputs(root, reader, rows, folds=(0,), target="gold"):
    """rows: list of (question_id, fold, evidence_share, delta)."""
    d = root / reader
    d.mkdir(parents=True, exist_ok=True)
    for qid, fold, share, delta in rows:
        m = {"l2": {"shares": {"instruction": 0.2, "evidence": share, "distractor": 0.7 - share, "question": 0.1},
                    "lift": share / 0.2, "topk": share},
             "gxi": {"shares": {"instruction": 0.2, "evidence": share / 2, "distractor": 0.7 - share / 2, "question": 0.1},
                     "lift": share / 0.4, "topk": share / 2},
             "counts": {"evidence": 20, "distractor": 60, "instruction": 15, "question": 5}, "target_value": -3.0, "n_tokens": 100}
        (d / f"{qid}.{target}.json").write_text(json.dumps({"question_id": qid, "fold": fold, "reader": reader, "target": target,
                                                            "model_id": "x", "measures": m}))
        (d / f"{qid}.ablation.json").write_text(json.dumps({"question_id": qid, "gold_full": 0.0, "gold_ablated": -delta, "delta": delta}))


def test_report_builds_claims_from_outputs(tmp_path, monkeypatch):
    from scripts import exp33_report as rep
    qids = [f"q{i}" for i in range(40)]
    small = [(q, 0, 0.1 + 0.002 * i, 0.5 + 0.01 * i) for i, q in enumerate(qids)]
    large = [(q, 0, 0.3 + 0.002 * i, 1.0 + 0.01 * i) for i, q in enumerate(qids)]
    _fake_outputs(tmp_path, "qwen3b", small); _fake_outputs(tmp_path, "qwen7b", large)

    def records(reader, fold, **kw):
        rs = [PromptRecord(q, 0, reader, "p", "o", "g", "single-session-user", True, ("l",), ("l",), True,
                           correct=(i % 2 == 0) if reader == "qwen3b" else True, unknown=False) for i, q in enumerate(qids)]
        return rs, {}
    monkeypatch.setattr(rep, "load_records", records)
    markdown, numbers = rep.build_report(("qwen3b", "qwen7b"), {0}, "l2", "k50", tmp_path, samples=300)
    assert numbers["claim1"]["gold"]["verdict"] == "SHOWN" and numbers["claim1"]["gold"]["n"] == 40
    assert 0 <= numbers["claim2"]["qwen3b"]["auc"] <= 1
    assert numbers["ablation"]["qwen3b"]["rho"] > 0.9
    assert "Claim 1" in markdown and "qwen7b" in markdown and "folds: 0" in markdown
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_jacobian.py -q -k report`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'` or `scripts.exp33_report`

- [ ] **Step 3: Write the implementation** (and an empty `scripts/__init__.py` if `scripts` is not a package; check with `ls scripts/__init__.py` first and create it only if missing)

```python
# scripts/exp33_report.py
"""§33 report: the pre-registered claims from runs/exp33_jacobian/<reader>/ outputs.

    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2 --out runs/exp33_report_dev.md

Folds default to the development folds 0,1. Confirmation folds (2,3,4) are read only after the
pre-registration is committed; the script refuses them unless --confirm is passed."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian_stats import bootstrap_auc, bootstrap_spearman, paired_claim  # noqa: E402
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

DEV_FOLDS = {0, 1}
TARGETS = ("gold", "margin", "own")


def load_outputs(out_dir: Path, folds: set[int], target: str) -> dict[str, dict]:
    found = {}
    for path in sorted(Path(out_dir).glob(f"*.{target}.json")):
        entry = json.loads(path.read_text())
        if entry["fold"] in folds:
            found[entry["question_id"]] = entry["measures"]
    return found


def load_ablation(out_dir: Path, folds: set[int]) -> dict[str, float]:
    """question_id -> delta for questions whose gold output is in the chosen folds."""
    folds_of = {}
    for path in Path(out_dir).glob("*.gold.json"):
        entry = json.loads(path.read_text())
        folds_of[entry["question_id"]] = entry["fold"]
    deltas = {}
    for path in Path(out_dir).glob("*.ablation.json"):
        entry = json.loads(path.read_text())
        if folds_of.get(entry["question_id"]) in folds:
            deltas[entry["question_id"]] = entry["delta"]
    return deltas


def population(records_by_reader: dict[str, list[PromptRecord]]) -> list[str]:
    sets = []
    for records in records_by_reader.values():
        sets.append({r.question_id for r in records if r.answerable and r.all_in_view and r.evidence_lines})
    return sorted(set.intersection(*sets)) if sets else []


def _fmt(stat: dict, key: str = "mean") -> str:
    if stat.get("ci_low") is None:
        return f"{stat.get(key)}"
    return f"{stat[key]:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def build_report(readers: tuple[str, str], folds: set[int], variant: str, k_label: str, out_root: Path,
                 samples: int = 10000) -> tuple[str, dict]:
    small, large = readers
    records_by_reader = {}
    for reader in readers:
        rows = []
        for fold in sorted(folds):
            fold_rows, _ = load_records(reader, fold)
            rows.extend(fold_rows)
        records_by_reader[reader] = rows
    qids = population(records_by_reader)
    by_q = {reader: {r.question_id: r for r in rows} for reader, rows in records_by_reader.items()}
    numbers: dict = {"folds": sorted(folds), "variant": variant, "population": len(qids), "claim1": {}, "claim2": {},
                     "ablation": {}, "shares": {}, "cells": {}}
    lines = [f"# §33 Jacobian lens report ({small} vs {large}; folds: {', '.join(map(str, sorted(folds)))}; "
             f"variant {variant}; population {len(qids)} in-view answerable questions)", ""]
    outputs = {reader: {t: load_outputs(out_root / reader, folds, t) for t in TARGETS} for reader in readers}

    lines += ["## Claim 1: evidence share, large minus small (paired by question)", "",
              "| target | n | mean difference (95% CI) | verdict |", "|---|---|---|---|"]
    for target in ("gold", "margin"):
        a = {q: outputs[large][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[large][target]}
        b = {q: outputs[small][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[small][target]}
        stat = paired_claim(a, b, samples=samples)
        numbers["claim1"][target] = stat
        lines.append(f"| {target} | {stat['n']} | {_fmt(stat)} | {stat['verdict']} |")

    lines += ["", "## Claim 2: within each reader, evidence share on correct vs wrong (gold target, AUC)", "",
              "| reader | n right | n wrong | AUC (95% CI) | mean right | mean wrong |", "|---|---|---|---|---|---|"]
    for reader in readers:
        shares = outputs[reader]["gold"]
        right = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is True]
        wrong = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is False]
        stat = bootstrap_auc(right, wrong, samples=samples) if right and wrong else {"n_pos": len(right), "n_neg": len(wrong),
                                                                                    "auc": None, "ci_low": None, "ci_high": None}
        numbers["claim2"][reader] = stat
        lines.append(f"| {reader} | {len(right)} | {len(wrong)} | {_fmt(stat, 'auc')} | "
                     f"{mean(right) if right else float('nan'):.3f} | {mean(wrong) if wrong else float('nan'):.3f} |")

    lines += ["", "## Ablation check: Spearman(evidence share, Δ gold log-prob when evidence is removed)", "",
              "| reader | n | rho (95% CI) | mean Δ |", "|---|---|---|---|"]
    for reader in readers:
        deltas = load_ablation(out_root / reader, folds)
        shares = outputs[reader]["gold"]
        keys = [q for q in qids if q in deltas and q in shares]
        if len(keys) >= 3:
            stat = bootstrap_spearman([shares[q][variant]["shares"]["evidence"] for q in keys], [deltas[q] for q in keys], samples=samples)
            stat["mean_delta"] = mean(deltas[q] for q in keys)
        else:
            stat = {"n": len(keys), "rho": None, "ci_low": None, "ci_high": None, "mean_delta": None}
        numbers["ablation"][reader] = stat
        lines.append(f"| {reader} | {stat['n']} | {_fmt(stat, 'rho')} | {stat['mean_delta']} |")

    lines += ["", "## Descriptive: mean region shares and evidence lift per reader and target", "",
              "| reader | target | instruction | evidence | distractor | question | lift | top-k evidence |",
              "|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        for target in TARGETS:
            rows = [outputs[reader][target][q][variant] for q in qids if q in outputs[reader][target]]
            if not rows:
                continue
            avg = {name: mean(r["shares"][name] for r in rows) for name in ("instruction", "evidence", "distractor", "question")}
            lift = mean(r["lift"] for r in rows if r["lift"] is not None)
            topk = mean(r["topk"] for r in rows)
            numbers["shares"][f"{reader}/{target}"] = {**avg, "lift": lift, "topk": topk, "n": len(rows)}
            lines.append(f"| {reader} | {target} | {avg['instruction']:.3f} | {avg['evidence']:.3f} | {avg['distractor']:.3f} | "
                         f"{avg['question']:.3f} | {lift:.2f} | {topk:.3f} |")

    lines += ["", f"## Descriptive: the four outcome cells ({small} right?, {large} right?) and mean evidence share (gold)", "",
              f"| {small} | {large} | n | {small} share | {large} share |", "|---|---|---|---|---|"]
    cells = defaultdict(list)
    for q in qids:
        if q in outputs[small]["gold"] and q in outputs[large]["gold"]:
            cells[(by_q[small][q].correct, by_q[large][q].correct)].append(q)
    for key in sorted(cells, key=str):
        qs = cells[key]
        s = mean(outputs[small]["gold"][q][variant]["shares"]["evidence"] for q in qs)
        l = mean(outputs[large]["gold"][q][variant]["shares"]["evidence"] for q in qs)
        numbers["cells"][str(key)] = {"n": len(qs), small: s, large: l}
        lines.append(f"| {key[0]} | {key[1]} | {len(qs)} | {s:.3f} | {l:.3f} |")

    lines += ["", f"Top-k label: {k_label}. Verdict rule: SHOWN if the lower bound is above 0. "
              "Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3)."]
    return "\n".join(lines) + "\n", numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", default="qwen3b")
    parser.add_argument("--large", default="qwen7b")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--variant", default="l2", choices=("l2", "gxi"))
    parser.add_argument("--root", default="runs/exp33_jacobian")
    parser.add_argument("--out", default=None)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--confirm", action="store_true", help="allow folds outside the development set")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    markdown, numbers = build_report((args.small, args.large), folds, args.variant, "k50", Path(args.root), args.samples)
    out = Path(args.out or f"runs/exp33_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.write_text(markdown)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_jacobian.py -q`
Expected: all pass

- [ ] **Step 5: Run the report on the smoke outputs** (both readers pointing at the smoke folder is fine for a plumbing check)

Run: `.venv/bin/python scripts/exp33_report.py --small qwen05b --large qwen05b --folds 0 --root runs/exp33_jacobian --out runs/exp33_jacobian/_smoke_report.md` after copying or symlinking `runs/exp33_jacobian/_smoke_qwen05b` to `runs/exp33_jacobian/qwen05b`
Expected: a markdown report with three-question tables, no exception.

- [ ] **Step 6: Commit**

```bash
git add scripts/exp33_report.py scripts/__init__.py tests/test_jacobian.py
git commit -m "§33: report pipeline for the pre-registered claims"
```

---

### Task 8: Pod runbook and the pre-launch stub (no spend; main session launches)

**Files:**
- Create: `scripts/exp33_pod.sh`
- Modify: `README_RESEARCH.md` (append a short "§33 Jacobian lens" how-to)

**Interfaces:**
- Consumes: `scripts/jacobian_lens.py` (Task 6).
- Produces: a shell script that, on a RunPod PyTorch pod with the repo and `runs/` present, installs the package, runs the regeneration check, then both readers over all folds with `--ablate`, writing to `runs/exp33_jacobian/<reader>/`. It must be idempotent (resume) and print the elapsed time per reader.

- [ ] **Step 1: Write the pod script**

```bash
#!/usr/bin/env bash
# §33 on one A40: both Qwen readers, all folds, with the ablation check. Resumable; rerun to continue.
#   bash scripts/exp33_pod.sh [folds]      default folds 0,1,2,3,4
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
python -m pip install -q -e . transformers accelerate scipy 2>&1 | tail -1
export TOKENIZERS_PARALLELISM=false
for READER in qwen3b qwen7b; do
  START=$(date +%s)
  python scripts/jacobian_lens.py --reader "$READER" --folds "$FOLDS" --targets gold,margin,own --ablate \
    --regen-check 20 --regen-min 15 --device cuda --dtype bfloat16 --out "runs/exp33_jacobian/$READER" \
    2>&1 | tee -a "runs/exp33_jacobian/${READER}_pod.log"
  echo "$READER done in $(( $(date +%s) - START )) s" | tee -a "runs/exp33_jacobian/${READER}_pod.log"
done
```

- [ ] **Step 2: Local timing stub for the cost quote** (CPU, 3B in bf16 is slow but gives the token count and memory; the 0.5B gives the per-call shape)

Run: `.venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 10 --device cpu --dtype float32 --ablate --out runs/exp33_jacobian/_stub_qwen05b`
Record from `_timing.json`: seconds per question, peak RSS. Write the two numbers into the README how-to as "local 0.5B CPU reference". The A40 estimate is made by the main session from these and the token counts (about 1,100 prompt tokens).

- [ ] **Step 3: README how-to**

Append to `README_RESEARCH.md`:

```markdown
## §33 Jacobian lens (how to run)

Prompts come from the §23 cache; no reader is called. Local development on the 0.5B:

    .venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 10 --device cpu --dtype float32 --ablate
    .venv/bin/python scripts/exp33_report.py --small qwen05b --large qwen05b --folds 0

Paid run (one A40, user-approved): `bash scripts/exp33_pod.sh`, then copy `runs/exp33_jacobian/` back.
Report on development folds only until the pre-registration is committed:

    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2
    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 2,3,4 --variant l2 --confirm

Local 0.5B CPU reference: <seconds per question> s/question, peak RSS <MB> MB (fill from `_timing.json`).
```

- [ ] **Step 4: Run the whole test file and the boundary test once more**

Run: `.venv/bin/pytest tests/test_jacobian.py tests/test_import_boundaries.py -q`
Expected: all §33 tests pass; the two pre-existing boundary failures only.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp33_pod.sh README_RESEARCH.md
git commit -m "§33: pod runbook and local timing reference"
```

---

## After the plan (main session, not an implementation task)

1. Pre-launch checklist from the stub numbers; quote the A40 price and hard stop; wait for the user.
2. Paid run of both readers over all folds; copy outputs back; verify `_regen_check.json` and `_dropped.json`.
3. Development-fold report (folds 0,1); fix the variant (`l2` or `gxi`), `k`, and the ablation bar; draft the pre-registration in `research/proposals/2026-10-XX-exp33-jacobian-lens.md` with the EXPERIMENTS.md §33 text; reviewer sign-off; commit with hash.
4. Confirmation report (folds 2,3,4, `--confirm`); results section §33a in EXPERIMENTS.md; report copy to `docs/research/exp33_jacobian_report.md`; gate decision on stage 2.
