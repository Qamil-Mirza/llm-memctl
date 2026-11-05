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


# --- §43: the §24 records of another family (rev 1 ruling B3) --------------------------------------------------
# `load_records` above stays the §33-§35 Qwen loader (exp23 paths) unchanged. This is the §24 version (stage 0's
# runs/exp43_stage0/data.py `load_records24`, moved here): runs/exp24_<key>_f<fold>, cache/generations_exp24_<key>,
# and runs/exp24_verdicts.json, whose rows carry a `fold` key and several arms (fixed8 / fixed16 / fifo_top5_t3000).

def _family_verdicts(path: str, reader: str, arm: str, fold: int) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return {v["question_id"]: v for v in json.loads(p.read_text())
            if v["reader"] == reader and v["arm"] == arm and v.get("fold", fold) == fold}


def load_family_records(cfg, fold: int, runs_dir: str = "runs",
                        data_path: str = "data/longmemeval/longmemeval_s_cleaned.json",
                        arm: str = "fixed8__fraction0.05") -> tuple[list[PromptRecord], dict[str, int]]:
    """PromptRecords of one fold for a memctl.analysis.families.FamilyConfig `cfg` (the same construction as
    `load_records`, pointed at the family's run, cache and verdict file)."""
    run = Path(cfg.run_dir(fold, runs_dir)) / arm
    cache = cached_prompts(cfg.cache_dir)
    items: dict[tuple[str, str], dict] = {(it["episode_id"], it["id"]): it for it in _read_jsonl(run / "items.jsonl.gz")}
    scored = {s["episode_id"]: s for s in _read_jsonl(run / "steps.jsonl.gz") if s.get("scored")}
    episodes = {e["episode_id"]: e for e in _read_jsonl(run / "episodes.jsonl")}
    verdicts = _family_verdicts(str(Path(runs_dir) / Path(cfg.verdicts).name), cfg.key, cfg.verdict_arm, fold)
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
            question_id, fold, cfg.key, prompt, output, gold, question_type, answerable, lines, evidence, complete,
            verdict.get("correct"), verdict.get("unknown"),
        ))
    records.sort(key=lambda r: r.question_id)
    return records, dropped


def family_population(records: list[PromptRecord]) -> list[PromptRecord]:
    """The §35 population rule for one reader: answerable, all evidence in view, at least one evidence line, a judge
    label present (stage 0's 126 dev rows per family). Marked rows (an `Answer:` in the cached output) are a further
    filter the caller applies."""
    return [r for r in records if r.answerable and r.all_in_view and r.evidence_lines and r.correct is not None]
