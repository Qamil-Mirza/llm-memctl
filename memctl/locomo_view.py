"""§40: the LoCoMo in-view set, built in the §23 fixed8 prompt family (pre-registration
research/proposals/2026-10-23-exp40-fixed-setting-validation.md, rev 1 at 877f86d).

Pure logic, no model. From data/locomo/locomo10.json:

- Turns: `memctl.envs.locomo.parse_conversation` (sessions in number order, turns in file order = arrival order). A turn
  with a photo has " [shares a photo: <blip_caption>]" appended to its text, after one space, at the END of the turn's
  content (rev 1 iii: inside the content, so the line reads "speaker, date: <text> [shares a photo: <caption>]").
- Line: the §23 compact line `label_prefix(metadata) + content` = "<speaker>, <session date>: <content>", the session
  date being the conversation's `session_<n>_date_time` string verbatim (e.g. "1:56 pm on 8 May, 2023"; rev 1 ii).
- Evidence: every id the `D<n>:<m>` pattern finds in the question's evidence entries (the file has typos such as
  "D8:6; D9:17"), de-duplicated in order. A question is EXCLUDED before the shuffle when an entry yields no id or an id
  is not a turn of its conversation ("unresolved"), when it has no evidence, or when it has more than 8 evidence turns.
- View: all evidence turns plus the BM25 top turns of the same conversation (`LexicalRetriever`, content words, the
  question text as the query, document frequencies over all the conversation's turns, ties to the earlier turn) that
  are not evidence, until 8 items; if BM25 has fewer positive-score turns than needed, the earliest remaining turns
  fill the view (`n_zero_fill`, counted; expected 0). The 8 items are shown in arrival order.
- Prompt: `LLMAgent.build_prompt` itself (reasoning on, compact labels, arrival order) with the QA environment's goal
  and the observation "Question: <question>" (what `QAEnvironment` shows for LoCoMo; LongMemEval adds "(asked on ...)").
  So the instruction, headers and "Note:" are the §23 fixed8 template byte-for-byte; only memory and question differ.
- Rows: answerable = categories 1-4. A seeded shuffle (random.Random(40)) of the eligible answerable questions in file
  order takes 600 (`N_ROWS`); a separate random.Random(40) shuffle of the eligible category-5 questions takes 40
  (`N_ADVERSARIAL`), the descriptive adversarial block (answer rate only, no judge).
- §42 (proposal 2026-10-25 rev 1 F2): the same builder with seed 42 and §40's 600 + 40 question ids removed from
  both eligible pools before the shuffles (`build_set(seed=, exclude_ids=)`; the defaults are §40's set, unchanged).
- Judge type map (rev 1 iv): 4 -> single-session-user, 1 -> multi-session, 2 -> temporal-reasoning, 3 -> default.
  (LONGMEMEVAL_PROMPTS has no single-session-user or multi-session entry, so those fall to the default prompt; only
  category 2 gets a different wording, the temporal off-by-one clause.)
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from memctl.agents.llm import LLMAgent
from memctl.envs.locomo import CATEGORIES, DIALOGUE_ID, parse_conversation
from memctl.envs.qa import QAEnvironment
from memctl.memory.items import MemoryItem, SourceType, count_tokens, label_prefix
from memctl.memory.state import MemoryView
from memctl.retrieval import LexicalRetriever
from memctl.task import Observation, TaskState

LOCOMO_PATH = "data/locomo/locomo10.json"
SEED = 40
N_ROWS = 600
N_ADVERSARIAL = 40
VIEW_K = 8
MAX_EVIDENCE = 8
ANSWERABLE = (1, 2, 3, 4)
ADVERSARIAL = 5
JUDGE_TYPE = {4: "single-session-user", 1: "multi-session", 2: "temporal-reasoning", 3: "default"}
CAPTION_FORMAT = " [shares a photo: {caption}]"   # appended to the turn text (envs/locomo.parse_conversation)
AGENT_CONFIG = {"reasoning": True, "labels": "compact", "order": "arrival", "model": {"backend": "stub"}}


@dataclass(frozen=True)
class LocomoRecord:
    question_id: str             # "<sample_id>_q<index in the file's qa list>"
    conversation: str
    qa_index: int
    category: int
    category_name: str
    judge_type: str
    question: str
    gold: str                    # str(answer); category 5: str(adversarial_answer) or "" (never judged)
    evidence_ids: tuple[str, ...]
    view_ids: tuple[str, ...]    # the 8 shown turns, arrival order
    memory_lines: tuple[str, ...]
    evidence_lines: tuple[str, ...]
    prompt: str
    prompt_sha256: str
    n_bm25_fill: int
    n_zero_fill: int
    fold: int = 0                # the steering rows' schema needs a fold; LoCoMo has none

    def to_dict(self) -> dict:
        return asdict(self)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str = LOCOMO_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


def resolve_evidence(entries, text_of: dict) -> tuple[tuple[str, ...], str | None]:
    """(ids, problem): ids de-duplicated in order; problem 'unresolved' when an entry yields no id or an id is not a
    turn of the conversation, 'none' when there is no evidence, 'too_many' above MAX_EVIDENCE, else None."""
    ids: list[str] = []
    for entry in entries or []:
        found = DIALOGUE_ID.findall(str(entry))
        if not found:
            return tuple(ids), "unresolved"
        for i in found:
            if i not in text_of:
                return tuple(ids), "unresolved"
            if i not in ids:
                ids.append(i)
    if not ids:
        return (), "none"
    if len(ids) > MAX_EVIDENCE:
        return tuple(ids), "too_many"
    return tuple(ids), None


def _item(turn: Observation, index: int) -> MemoryItem:
    return MemoryItem(turn.id, turn.content, count_tokens(turn.content), index, SourceType.USER,
                      metadata=dict(turn.metadata))


def line_of(turn: Observation) -> str:
    return label_prefix(turn.metadata, SourceType.USER) + turn.content


def fill_view(question: str, items: list[MemoryItem], evidence_ids: tuple[str, ...], k: int = VIEW_K
              ) -> tuple[list[MemoryItem], int, int]:
    """(items in arrival order, n BM25 fill, n zero-score fill)."""
    ev = set(evidence_ids)
    chosen = [it for it in items if it.id in ev]
    need = k - len(chosen)
    hits = [it for it, _ in LexicalRetriever().search(question, items, len(items)) if it.id not in ev][:max(need, 0)]
    n_bm25 = len(hits)
    taken = ev | {it.id for it in hits}
    zero = [it for it in items if it.id not in taken][:max(need - n_bm25, 0)]
    view = sorted(chosen + hits + zero, key=lambda it: it.created_at)
    return view, n_bm25, len(zero)


def build_prompt(agent: LLMAgent, view: list[MemoryItem], question: str) -> str:
    obs = Observation("question", f"Question: {question}", SourceType.USER, requires_response=True)
    memory = MemoryView(step=0, budget=10 ** 9, archive_budget=None, active=tuple(view), archived=(),
                        active_tokens=sum(it.token_count for it in view), archive_tokens=0)
    return agent.build_prompt(memory, obs, TaskState(0, QAEnvironment.goal, obs))


@dataclass(frozen=True)
class Candidate:
    sample: int                  # index of the conversation in the file
    qa_index: int
    category: int
    evidence_ids: tuple[str, ...]


def candidates(samples: list[dict]) -> tuple[list[Candidate], dict, dict]:
    """Every question that passes the evidence filters, in file order; exclusion counts; question counts by category."""
    out, excluded = [], {}
    total = {c: 0 for c in CATEGORIES}
    for si, sample in enumerate(samples):
        episode = parse_conversation(sample, include_adversarial=True)
        for number, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            total[cat] += 1
            ids, problem = resolve_evidence(qa.get("evidence", []), episode.turn_text)
            if problem:
                key = f"{problem}:cat{cat}"
                excluded[key] = excluded.get(key, 0) + 1
                continue
            out.append(Candidate(si, number, cat, ids))
    return out, dict(sorted(excluded.items())), total


def record_of(sample: dict, cand: Candidate, agent: LLMAgent, cache: dict) -> LocomoRecord:
    """The view, lines and prompt of one question (`cache` keeps the parsed turns per conversation)."""
    sid = str(sample["sample_id"])
    if sid not in cache:
        episode = parse_conversation(sample, include_adversarial=True)
        cache[sid] = ([_item(t, n) for n, t in enumerate(episode.turns)], {t.id: line_of(t) for t in episode.turns})
    items, line = cache[sid]
    qa = sample["qa"][cand.qa_index]
    view, n_bm25, n_zero = fill_view(qa["question"], items, cand.evidence_ids)
    prompt = build_prompt(agent, view, qa["question"])
    cat = cand.category
    gold = qa.get("adversarial_answer", "") if cat == ADVERSARIAL else qa.get("answer", "")
    return LocomoRecord(f"{sid}_q{cand.qa_index}", sid, cand.qa_index, cat, CATEGORIES[cat],
                        JUDGE_TYPE.get(cat, "default"), qa["question"], str(gold), cand.evidence_ids,
                        tuple(it.id for it in view), tuple(line[it.id] for it in view),
                        tuple(line[i] for i in cand.evidence_ids), prompt, sha256(prompt), n_bm25, n_zero)


@dataclass
class LocomoSet:
    rows: list[LocomoRecord]          # the 600, shuffle order
    adversarial: list[LocomoRecord]   # the 40, shuffle order
    counts: dict
    shas: dict

    def by_q(self) -> dict[str, LocomoRecord]:
        return {r.question_id: r for r in self.rows + self.adversarial}


def question_id(samples: list[dict], cand: Candidate) -> str:
    return f"{samples[cand.sample]['sample_id']}_q{cand.qa_index}"


def build_set(path: str = LOCOMO_PATH, n_rows: int = N_ROWS, n_adv: int = N_ADVERSARIAL, seed: int = SEED,
              exclude_ids=None) -> LocomoSet:
    """Filters, then the two seeded shuffles (file order in), then the views of the selected questions only.

    §42 (rev 1 F2): `exclude_ids` (question ids, e.g. §40's 600 + 40) are removed from both eligible pools BEFORE the
    shuffles; `seed` seeds both shuffles. The defaults (seed 40, no exclusion) give §40's set byte-identically; only
    with `exclude_ids` given do the counts gain the pool keys and the shas the exclusion sha."""
    agent = LLMAgent(dict(AGENT_CONFIG))
    samples = load(path)
    cands, excluded, total = candidates(samples)
    answerable = [c for c in cands if c.category in ANSWERABLE]
    adversarial = [c for c in cands if c.category == ADVERSARIAL]
    eligible_a, eligible_x = len(answerable), len(adversarial)
    if exclude_ids is not None:
        drop = set(exclude_ids)
        answerable = [c for c in answerable if question_id(samples, c) not in drop]
        adversarial = [c for c in adversarial if question_id(samples, c) not in drop]
    pool_a, pool_x = len(answerable), len(adversarial)
    random.Random(seed).shuffle(answerable)
    random.Random(seed).shuffle(adversarial)
    if len(answerable) < n_rows or len(adversarial) < n_adv:
        raise ValueError(f"not enough eligible questions: {len(answerable)}/{n_rows} answerable, "
                         f"{len(adversarial)}/{n_adv} adversarial")
    cache: dict = {}
    rows = [record_of(samples[c.sample], c, agent, cache) for c in answerable[:n_rows]]
    adv = [record_of(samples[c.sample], c, agent, cache) for c in adversarial[:n_adv]]
    counts = {"questions_by_category": {str(c): n for c, n in total.items()}, "excluded": excluded,
              "eligible_answerable": eligible_a, "eligible_adversarial": eligible_x, "rows": len(rows),
              "adversarial": len(adv),
              "rows_by_category": {str(c): sum(r.category == c for r in rows) for c in ANSWERABLE},
              "zero_fill_rows": sum(r.n_zero_fill > 0 for r in rows + adv),
              "evidence_turns_hist": {str(n): sum(len(r.evidence_ids) == n for r in rows) for n in range(1, 9)},
              "prompt_chars_max": max(len(r.prompt) for r in rows + adv)}
    if exclude_ids is not None:
        counts.update({"seed": seed, "exclude_ids": len(set(exclude_ids)),
                       "excluded_from_answerable": eligible_a - pool_a, "excluded_from_adversarial": eligible_x - pool_x,
                       "pool_answerable": pool_a, "pool_adversarial": pool_x})
    shas = {"locomo10.json": file_sha256(path),
            "rows": sha256(json.dumps([r.to_dict() for r in rows], sort_keys=True)),
            "adversarial": sha256(json.dumps([r.to_dict() for r in adv], sort_keys=True)),
            "row_ids": sha256(json.dumps([r.question_id for r in rows])),
            "adversarial_ids": sha256(json.dumps([r.question_id for r in adv]))}
    if exclude_ids is not None:
        shas["exclude_ids"] = sha256(json.dumps(sorted(set(exclude_ids))))
    return LocomoSet(rows, adv, counts, shas)
