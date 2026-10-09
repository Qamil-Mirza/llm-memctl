"""UtilMem (Qing, Shi and Vosoughi, 2026; arXiv 2608.30508): long-form questions that need evidence spread over
several sessions, in a haystack of look-alike distractor sessions.

Source: github.com/peijunallin/UtilMem at commit b7ebd4a1ba31379a67540f2b1c95ec48351f6d79 (a README only; the
evaluation code is "coming soon"). Data: the Hugging Face dataset KrisQ/utilmem at revision
15a774f0d630498e17ad4ba3eedaf06d08c5796e, one file, multi_domain_eval_strong.json (54 MB, sha256 2816fb72...).
Neither the repository nor the dataset states a licence (the paper is CC BY 4.0; its ethics statement says the
benchmark is "intended for research evaluation"). It is read here for research only and not redistributed.

The release: 100 bundles. A bundle is one user's history of 65-85 sessions (about 120k tokens) from five domains
(studychat, finance, mental_health, fitness, edgar_10k), in date order. Each domain has 3-4 questions and 3-7
evidence sessions ("gt"); every other session is a distractor ("noise": strong ones written to share words with
the evidence, and the other domains' sessions). 1,717 questions in all. There is no gold answer: the paper scores
an answer 1-10 with an LLM judge against a reference answer that the same reader wrote from the evidence sessions
alone (`oracle: true` builds that episode here). The judge prompt is the paper's Appendix F (JUDGE_PROMPT).

One episode is one question, as in LongMemEval: the bundle's sessions turn by turn, then the question, asked
after the whole history (the paper gives each question the full history). Seed s plays the s-th question of the
chosen pool. Labels (evaluation only, never shown to the agent or controller): one requirement per evidence
session of the question's domain, met when any turn of that session is in view. So "all in view" means every
evidence session is touched, the session-level recall of the paper's §5.2, not that every evidence turn is in view.

Split (§30, fixed before any look at evidence positions or outcomes): 20 of the 100 bundles, drawn with seed 30,
are the development side (`part: dev`); the other 80 are the evaluation side (`part: eval`). Any check or design
choice uses the development side only.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path

from memctl.envs.qa import QAEnvironment, QAEpisode, QAItem
from memctl.memory.items import SourceType
from memctl.task import Dependency, EvidenceRequirement, Observation, StepResult

SOURCE = "https://huggingface.co/datasets/KrisQ/utilmem"
SOURCE_REVISION = "15a774f0d630498e17ad4ba3eedaf06d08c5796e"
SOURCE_SHA256 = "2816fb722c970f13b3fb404845aeddda3cb88a0f8bbd8e0417ee1c9748d113db"
CODE = "https://github.com/peijunallin/UtilMem"
CODE_COMMIT = "b7ebd4a1ba31379a67540f2b1c95ec48351f6d79"
DOMAINS = ("studychat", "finance", "mental_health", "fitness", "edgar_10k")
SPLIT_SEED = 30
DEV_BUNDLES = 20

GOAL = (
    "You are the user's long-term AI assistant. Your memory below holds parts of your past conversations with the "
    "user, each line labelled with who spoke and when. Answer the user's current request using what the memory "
    "says. Ground every detail in the memory and do not invent facts; some remembered conversations only look "
    "related, so use only the ones that are about this request. Write a complete, well-organised answer. If the "
    "memory holds nothing relevant to the request, reply: unknown"
)

# ---- from the paper (arXiv 2608.30508, CC BY 4.0), Appendix F, Figure 8: the noise-robustness judge prompt ------
JUDGE_PROMPT = """You are an expert judge evaluating how well a memory-augmented QA system tolerates retrieval noise.

# Setup
The same QA model (temperature=0, deterministic) was asked the same question twice: once with ONLY oracle context (the relevant past sessions), and once with oracle context PLUS distractor sessions (retrieval noise). Because the QA model is deterministic and the only difference is the injected noise, the Reference Answer below is treated as a strong baseline (score = 10). Your job is to score the Noisy Answer on how faithfully it preserves the quality of the Reference Answer.

# Context (Ground-Truth Oracle Sessions)
{context}

# Question
{question}

# Reference Answer (Oracle context, score = 10 by definition)
{answer_oracle}

# Noisy Answer (Oracle + distractor context, TO BE EVALUATED)
{answer_noisy}

# Step 1: Assess each of the five sub-dimensions
For each dimension below, classify the Noisy Answer into exactly one severity level: intact (no meaningful difference), mild (a small, bounded issue a reader would still trust), clear (a meaningful issue that affects usefulness), or severe (a failure that would mislead the reader).
1. Factual Fidelity. Are the claims consistent with either the Context or the Reference Answer? A claim absent from the Reference but supported by the Context is acceptable. Classify as severe when the Noisy Answer contradicts the Context or the Reference on an important claim.
2. Completeness Preservation. Does the Noisy Answer cover a comparable level of question-relevant information from the Context as the Reference does? Judge relative to what the question actually asks. Classify as clear or worse when the Noisy Answer omits or covers less question-relevant information.
3. Hallucination Absence. Does the Noisy Answer add information that is NOT in the Reference AND NOT supported by the Context? Content grounded in the Context but absent from the Reference is NOT a hallucination. Penalise fabricated facts, invented specifics, or distractor-sourced content.
4. Semantic Equivalence. Would a reader reach a similar conclusion or take a similar action from both answers? Classify as clear or worse when the noise changed the overall meaning, recommendation, or practical takeaway.
5. Ungrounded Inference. Does the Noisy Answer infer, assume, or speculate beyond what the Context supports? Distinguish from hallucination: hallucination = fabricated facts, ungrounded inference = fabricated reasoning or assumptions presented as fact.

# Step 2: Assign the final integer score (1--10)
Use the severity classifications from Step 1. When an answer sits between two anchors, choose the LOWER score, since a severe failure on a single dimension caps the score regardless of strength elsewhere.
- 10. All five dimensions intact. Factually identical in substance, complete, hallucination-free, same meaning. Differences are purely surface-level.
- 9. Up to two dimensions are mild, the rest intact. No clear or severe issues, no contradiction, no hallucination, no change in conclusion.
- 8. Three or more dimensions are mild, but NO dimension is clear or severe. Fully trustworthy, with only minor bounded drift.
- 7. One dimension is clear, any number of other dimensions may be mild. A single noticeable issue, but core facts and overall conclusion intact.
- 6. One dimension is clear with bounded impact (a secondary fact missing or a single unsupported inference). Answer remains mostly useful.
- 5. Two dimensions are clear. Partially reliable.
- 4. Two or more dimensions are clear AND at least one is severe: a key fact missing, one significant hallucination, or the conclusion has shifted.
- 3. Multiple severe failures: large omissions, notable hallucinations, or distractor content visibly bleeding into the answer.
- 2. Mostly wrong. Contradicts the Reference on important claims, distractor content dominates, or the conclusion is fundamentally different.
- 1. Completely wrong, refuses to answer, or contradicts the core facts entirely.
The final score MUST be an integer in [1, 10] and MUST be consistent with the Step 1 severities.

# Failure-Mode Tag
Pick the SINGLE best tag for the PRIMARY source of degradation (use "none" only when the score is 10): none, style_only, omission, hallucination, contradiction, ungrounded_inference, meaning_shift, refusal, mixed.

Respond with ONLY a JSON object containing analysis (2--4 sentence comparison), the five sub_dimensions severity labels, the integer score, and the failure_mode tag."""
# ---- end of the paper's prompt -----------------------------------------------------------------------------------

DEGRADED_AT = 6  # the paper's DR: the share of answers scored 6 or lower
# The §24 refusal phrases; an answer that starts with "unknown" or carries one of them in its first 200 characters
# counts as a refusal (the "unknown" share).
REFUSAL_PHRASES = ("don't know", "do not know", "not mention", "no information", "cannot determine", "unable to")


def is_refusal(answer: str) -> bool:
    text = (answer or "").strip().lower()
    return text.startswith("unknown") or any(p in text[:200] for p in REFUSAL_PHRASES)


def parse_score(output: str) -> int | None:
    """The judge's integer score (1-10), or None when the output has none."""
    text = output or ""
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start: end + 1])
            score = payload.get("score") if isinstance(payload, dict) else None
            if isinstance(score, (int, float)) or (isinstance(score, str) and score.strip().isdigit()):
                score = int(float(score))
                return score if 1 <= score <= 10 else None
        except (json.JSONDecodeError, ValueError):
            pass
    found = re.search(r'"score"\s*:\s*"?(\d+)', text)
    if found and 1 <= int(found.group(1)) <= 10:
        return int(found.group(1))
    return None


def normalized_robustness(scores: list[float]) -> float:
    """NR on a 0-1 scale: (RS - 1) / 9, RS the mean score (the paper rescales RS to 0-100; 1 maps to 0)."""
    return (sum(scores) / len(scores) - 1) / 9 if scores else float("nan")


@lru_cache(maxsize=4)
def _sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _shard_dir(path: str) -> Path:
    """`<file>.shards/`: index.json (every question: bundle, domain, number) and one NNNN.json per bundle, so a
    process holds one bundle, not the whole file. Rebuilt when the source's sha256 differs from the one recorded."""
    source = Path(path)
    folder = source.with_name(source.name + ".shards")
    stamp = {"sha256": _sha256(source)}
    index = folder / "index.json"
    if index.exists() and json.loads(index.read_text()).get("source") == stamp:
        return folder
    folder.mkdir(exist_ok=True)
    with open(folder / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if index.exists() and json.loads(index.read_text()).get("source") == stamp:
            return folder
        bundles = json.loads(source.read_text())
        questions = []
        for number, bundle in enumerate(bundles):
            (folder / f"{number:04d}.json").write_text(json.dumps(bundle))
            pool = bundle["question_pool"]
            for domain in [d for d in DOMAINS if d in pool] + sorted(set(pool) - set(DOMAINS)):
                for q in range(len(pool[domain])):
                    questions.append({"bundle": number, "sample_id": bundle["sample_id"], "domain": domain, "number": q})
        tmp = folder / "index.json.tmp"
        tmp.write_text(json.dumps({"source": stamp, "bundles": len(bundles), "questions": questions}))
        os.replace(tmp, index)
    return folder


@lru_cache(maxsize=4)
def _index(path: str) -> dict:
    return json.loads((_shard_dir(path) / "index.json").read_text())


@lru_cache(maxsize=4)
def _bundle(path: str, number: int) -> dict:
    return json.loads((_shard_dir(path) / f"{number:04d}.json").read_text())


def split_bundles(count: int, seed: int = SPLIT_SEED, dev: int = DEV_BUNDLES) -> tuple[list[int], list[int]]:
    """(development bundles, evaluation bundles), each sorted."""
    chosen = sorted(random.Random(seed).sample(range(count), dev))
    return chosen, [n for n in range(count) if n not in set(chosen)]


def question_pool(path: str, part: str = "eval", sample: dict | None = None) -> list[dict]:
    """The questions of one side (dev, eval or all), in file order; with `sample` {seed, per_domain}, a fixed-seed
    sample of per_domain questions from each domain, sorted back into file order."""
    index = _index(path)
    dev, evaluation = split_bundles(index["bundles"])
    bundles = {"dev": set(dev), "eval": set(evaluation), "all": set(range(index["bundles"]))}[part]
    pool = [q for q in index["questions"] if q["bundle"] in bundles]
    if sample and sample.get("per_domain") is not None:
        rng = random.Random(int(sample.get("seed", 0)))
        chosen = []
        for domain in DOMAINS:
            of_domain = [q for q in pool if q["domain"] == domain]
            chosen.extend(rng.sample(of_domain, min(int(sample["per_domain"]), len(of_domain))))
        order = {id(q): n for n, q in enumerate(pool)}
        pool = sorted(chosen, key=lambda q: order[id(q)])
    return pool


def parse_question(bundle: dict, domain: str, number: int, oracle: bool = False,
                   max_turn_tokens: int | None = None) -> tuple[QAEpisode, tuple[tuple[str, ...], ...]]:
    """(episode, evidence sessions as groups of turn ids) for one question. `oracle`: only the question domain's
    evidence sessions, the context the paper's reference answer is written from."""
    evidence = set(bundle["answer_session_ids"][domain])
    order = sorted(range(len(bundle["haystack_sessions"])), key=lambda n: bundle["haystack_dates"][n])
    turns, text_of, groups = [], {}, {}
    for position in order:
        session_id = bundle["haystack_session_ids"][position]
        if oracle and session_id not in evidence:
            continue
        date = bundle["haystack_dates"][position]
        for n, turn in enumerate(bundle["haystack_sessions"][position]):
            item_id = f"{session_id}:{n}"
            if item_id in text_of:
                item_id = f"{session_id}#{position}:{n}"
            text = turn["content"]
            if max_turn_tokens:
                from memctl.memory.items import truncate_tokens

                text = truncate_tokens(text, max_turn_tokens)
            source = SourceType.USER if turn["role"] == "user" else SourceType.OBSERVATION
            turns.append(Observation(item_id, text, source, metadata={"speaker": turn["role"], "session": session_id,
                                                                      "date": date}))
            text_of[item_id] = text
            if session_id in evidence:
                groups.setdefault(session_id, []).append(item_id)
    question_id = f"{bundle['sample_id']}/{domain}/q{number}"
    flat = tuple(i for ids in groups.values() for i in ids)
    question = QAItem(question_id, bundle["question_pool"][domain][number], "", domain, flat)
    return QAEpisode(question_id, turns, [question], text_of), tuple(tuple(ids) for ids in groups.values())


class UtilMemEnv(QAEnvironment):
    name = "utilmem"
    goal = GOAL

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.path = config.get("path", "data/utilmem/multi_domain_eval_strong.json")
        if not Path(self.path).exists():
            raise FileNotFoundError(f"UtilMem not found at {self.path}. Download {SOURCE} (revision {SOURCE_REVISION})")
        self.part = config.get("part", "eval")
        if self.part not in ("dev", "eval", "all"):
            raise ValueError(f"unknown part {self.part!r}")
        self.sample = dict(config.get("sample") or {})
        self.oracle = bool(config.get("oracle", False))
        self.max_turn_tokens = config.get("max_turn_tokens")
        self.pool = question_pool(self.path, self.part, self.sample)
        shard = dict(config.get("shard") or {})
        if shard:
            self.pool = self.pool[int(shard["index"])::int(shard["k"])]
        self._groups: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._answers: list[dict] = []

    def load_episode(self, seed: int) -> QAEpisode:
        if not self.pool:
            raise ValueError("no question in this part, sample and shard")
        entry = self.pool[seed % len(self.pool)]
        episode, groups = parse_question(_bundle(self.path, entry["bundle"]), entry["domain"], entry["number"],
                                         self.oracle, self.max_turn_tokens)
        self._groups = {episode.questions[0].id: groups}
        return episode

    def reset(self, seed: int) -> Observation:
        self._answers = []
        return super().reset(seed)

    def step(self, agent_action: str | None) -> StepResult:
        """No gold answer: the answer is recorded (scored, not correct) and graded afterwards by the rubric judge
        against the oracle reference, so every question's evidence status is logged."""
        current = self._observations[self._index]
        info: dict = {"scored": False, "correct": None}
        if current.requires_response:
            question = self._questions[current.id]
            answer = agent_action or ""
            self._answers.append({"question_id": question.id, "domain": question.category, "refusal": is_refusal(answer)})
            self._scores.append({"correct": False, "f1": 0.0, "category": question.category})
            info = {"scored": True, "correct": False, "gold": "", "query_id": current.id, "category": question.category,
                    "question_id": question.id}
        self._index += 1
        self._last_reward = 0.0
        return StepResult(self.get_observation(), 0.0, self.is_done(), info)

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        """One requirement per evidence session of the question's domain; any of its turns meets it."""
        seen = {observation.id for observation in self._observations[: self._index + 1]}
        found = []
        for query_id, question in self._questions.items():
            if query_id not in seen:
                continue
            requirements = tuple(EvidenceRequirement(ids, "") for ids in self._groups.get(question.id, ()))
            found.append(Dependency(query_id, self._step_of[query_id], requirements, None, question.category))
        return found

    def episode_stats(self) -> dict:
        return {"episode": self._episode.id if self._episode else None, "answers": self._answers,
                "scored_by": "rubric judge, afterwards (no gold answer)"}
