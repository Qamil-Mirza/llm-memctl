"""StreamMemBench (Liu et al., 2026; arXiv 2606.14571): a day of a person's life, then two-step tasks with
user feedback in between.

Source: github.com/landian60/StreamMemBench at commit b32965525da5f982599af9794954c616ddeb34ff (code MIT;
data derived from EgoLife, under EgoLife's licence and Hugging Face terms). The English release
(`data/streammembench_v1_en`) is copied to data/streammembench/streammembench_v1_en.

The stream is text only: every observation is a first-person narration line ("I put my phone away.") or a
dialogue line ("Jake: Okay."), grouped into five-minute segments. Lines are about 8 tokens long, so they are
grouped into chunks of about `chunk_tokens` (default 150, between LongMemEval's median turn of 91 and mean
of 210 tokens); a chunk never crosses a segment. A chunk becomes one memory item, like a LongMemEval turn.

One episode is one participant-day (about 126k tokens, close to LongMemEval_s's 103k). The paper keeps one
memory per participant over 7 days; a day is used here so a stream fits the scale the controllers were
built for. Segments are shown in order. Right after an evaluated segment, each of its evidence anchors runs
the benchmark's trajectory (their runner.py, run_evidence_anchor):

    1. initial request (the agent answers)                 -> initial_evidence_use = the user affirms
    2. if the user simulator asks for a revision: the feedback, with the request and the first answer
       (the agent answers again)                           -> feedback_incorporation = the user affirms
    3. the interaction is stored (an observation; their store_interaction text)
    4. follow-up request (the agent answers)               -> followup_reuse = the user affirms

The user simulator is their prompt (prompts/evaluation/user_feedback_simulator.md, English), sent as one
user message (system part, a blank line, then the user part), one vote at temperature 0; the paper uses
three votes at 0.5 with DeepSeek-V4-Pro. `simulator.style: lexical` is their deterministic smoke-test rule
(token overlap), for tests only. Fidelity (their LLM audit of saved records) is not computed: memctl keeps
every chunk, so it would be 1 for every arm; the label-based "evidence in view" replaces it as a diagnostic.

Labels (evaluation only, never shown to the agent or controller): an anchor's evidence chunks are those
holding its supporting observations whose day-level stream_index points at the same text in the same
segment. In the release, 20,644 of 25,671 supporting observations match that way; 1,498 of 8,107 anchors
have none, and then every chunk of the evidence segment counts.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

from memctl.envs.base import TaskEnvironment
from memctl.memory.items import SourceType, count_tokens, label_prefix
from memctl.task import Dependency, EvidenceRequirement, Observation, StepResult

SOURCE = "https://github.com/landian60/StreamMemBench"
SOURCE_COMMIT = "b32965525da5f982599af9794954c616ddeb34ff"
PARTICIPANTS = ("A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE")
# §25: A1 and A2 are the training side (reserved for any tuning); evaluation samples only A3-A6.
EVALUATION_PARTICIPANTS = PARTICIPANTS[2:]

# ---- vendored from StreamMemBench (MIT) -----------------------------------------------------------------
# prompts/evaluation/user_feedback_simulator.md, split as their utils/llm.py prompt_sections does.
SIMULATOR_SYSTEM = """You are simulating the real user. Compare the memory system answer with the expected behavior and the ground-truth evidence anchor.

The memory system did not see the evidence statement or expected behavior. Your feedback may reveal the missing or corrected fact in a natural user voice, because the feedback step is part of the benchmark trajectory.

Use `expected_behavior` only as a judging reference. The `feedback` text must not mention benchmark field names such as `expected_behavior` or `evidence_statement`, and must not sound like a grading rubric.

Judge whether the answer:

- uses the relevant evidence;
- satisfies the expected behavior;
- avoids unsupported facts;
- preserves the evidential strength of the anchor.

Use `affirm` when the answer is already good. Use `revise` when the answer misses key evidence, contradicts the anchor, fabricates details, or fails the task.

The feedback should be 1 to 3 sentences and sound like the user directly correcting or confirming the assistant.

## Output

Return only JSON:

```json
{
  "type": "affirm or revise",
  "reason": "Brief reason for the feedback decision.",
  "feedback": "Natural user feedback to the assistant.",
  "evidence": [
    {
      "evidence_statement": "Relevant ground-truth statement.",
      "quote_hints": ["Original quote or observation hint"]
    }
  ]
}
```"""

SIMULATOR_USER = """You are {user}. Evaluate the assistant response and provide feedback.

user_request:
{user_task}

assistant_answer:
{agent_response}

expected_behavior:
{expected_behavior}

evidence_statement:
{evidence_statement}

supporting_observations:
{supporting_observations}

raw_evidence:
{raw_evidence}"""

# evaluation/user_feedback_simulator.py, DeterministicUserFeedbackSimulator (English strings).
LEXICAL_AFFIRM = "This answer is fine."
LEXICAL_REVISE = "A key piece of information needs to be added: {evidence} Please re-answer my question based on this."
_WORD = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def token_overlap_score(query: str, text: str) -> float:
    """utils/text.py: share of the query's word types found in the text."""
    wanted = {m.group(0).lower() for m in _WORD.finditer(query)}
    if not wanted:
        return 0.0
    return len(wanted & {m.group(0).lower() for m in _WORD.finditer(text)}) / len(wanted)


def extract_json(value: str):
    """utils/llm.py extract_json; returns None instead of raising."""
    candidates = [value.strip()]
    candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*(.*?)```", value, re.DOTALL))
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = value.find(opener), value.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(value[start: end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def interaction_text(name: str, request: str, answer: str, feedback: str, revised: str) -> str:
    """memory/baselines/rag_extracted.py _interaction_text, English labels: what store_interaction saves."""
    parts = [f"{name} says to AI:\n{request}", f"AI Answer:\n{answer}"]
    if feedback:
        parts.append(f"{name} Feedback:\n{feedback}")
    if revised:
        parts.append(f"AI Revised Answer:\n{revised}")
    return "\n\n".join(parts)
# ---- end of vendored code -------------------------------------------------------------------------------


class UserSimulator:
    """Their user-feedback simulator: affirm or revise, with a reason and the feedback text."""

    def __init__(self, config: dict, llm=None) -> None:
        self.style = config.get("style", "llm")
        if self.style not in ("llm", "lexical"):
            raise ValueError(f"unknown simulator style {self.style!r}")
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.llm = None
        if self.style == "llm":
            from memctl.llm import build_llm

            self.llm = llm or build_llm({k: v for k, v in config.items() if k not in ("style", "max_new_tokens")})
        self.calls = 0
        self.parse_failures = 0

    def prompt(self, anchor: dict, request: str, answer: str, expected: str) -> str:
        supporting = json.dumps(anchor.get("supporting_observations", []), ensure_ascii=False)
        user = SIMULATOR_USER.format(
            user=anchor.get("subject", ""), user_task=request, agent_response=answer, expected_behavior=expected,
            evidence_statement=anchor.get("evidence_statement", ""), supporting_observations=supporting,
            raw_evidence=supporting,
        )
        return f"{SIMULATOR_SYSTEM}\n\n{user}"

    def judge(self, anchor: dict, request: str, answer: str, expected: str) -> dict:
        self.calls += 1
        statement = anchor.get("evidence_statement", "")
        if self.style == "lexical":
            good = token_overlap_score(statement, answer) >= 0.5 and token_overlap_score(expected, answer) >= 0.3
            return {"type": "affirm" if good else "revise", "reason": "lexical rule",
                    "feedback": LEXICAL_AFFIRM if good else LEXICAL_REVISE.format(evidence=statement), "parsed": True}
        raw = self.llm.generate(self.prompt(anchor, request, answer, expected), self.max_new_tokens)
        payload = extract_json(raw)
        if not isinstance(payload, dict):
            # Their runner raises here; one failure must not stop a day, so it counts as a revision with
            # their deterministic feedback text, and is counted (simulator_parse_failures).
            self.parse_failures += 1
            return {"type": "revise", "reason": "unparsable simulator output",
                    "feedback": LEXICAL_REVISE.format(evidence=statement), "parsed": False}
        kind = str(payload.get("type", "revise")).strip().lower()
        return {"type": "affirm" if kind == "affirm" else "revise", "reason": str(payload.get("reason", "")),
                "feedback": str(payload.get("feedback", "")), "parsed": True}


def person_of(participant: str) -> str:
    return participant.split("_", 1)[1].capitalize() if "_" in participant else participant


@lru_cache(maxsize=64)
def _json(path: str):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=4)
def evidence_items(root: str, participants: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """(participant, day, segment_id) of every evidence-task item of these participants, sorted."""
    found = []
    for participant in participants:
        for path in sorted((Path(root) / "evidence_tasks" / participant).glob("*.clean.json")):
            for item in _json(str(path))["items"]:
                found.append((item["participant"], item["day"], int(item["segment_id"])))
    return sorted(found)


def sample_items(root: str, participants: tuple[str, ...], seed: int, count: int | None) -> list[tuple[str, str, int]]:
    """A fixed-seed sample of evidence segments (all of them when count is None), sorted."""
    pool = evidence_items(root, participants)
    if count is None or count >= len(pool):
        return list(pool)
    return sorted(random.Random(seed).sample(pool, count))


def shard_days(root: str, items: list[tuple[str, str, int]], k: int) -> list[list[tuple[str, str]]]:
    """The sampled participant-days dealt to k shards with about equal numbers of anchors (largest day first,
    each to the lightest shard), so parallel cells finish together. Each shard is sorted."""
    anchors: dict[tuple[str, str], int] = {}
    for participant, day, number in items:
        found = _json(str(Path(root, "evidence_tasks", participant, f"{day}.clean.json")))["items"]
        count = sum(len(i["evidence_anchors"]) for i in found if int(i["segment_id"]) == number)
        anchors[(participant, day)] = anchors.get((participant, day), 0) + count
    shards: list[list[tuple[str, str]]] = [[] for _ in range(k)]
    load = [0] * k
    for day in sorted(anchors, key=lambda d: (-anchors[d], d)):
        lightest = min(range(k), key=lambda n: (load[n], n))
        shards[lightest].append(day)
        load[lightest] += anchors[day]
    return [sorted(s) for s in shards]


def chunk_segment(observations: list[dict], chunk_tokens: int) -> list[list[dict]]:
    """Consecutive lines up to about chunk_tokens each; a short tail joins the chunk before it."""
    chunks, current, size = [], [], 0
    for observation in observations:
        current.append(observation)
        size += count_tokens(observation["text"])
        if size >= chunk_tokens:
            chunks.append(current)
            current, size = [], 0
    if current:
        if chunks and size < chunk_tokens / 3:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


class StreamMemBenchEnv(TaskEnvironment):
    name = "streammembench"
    stream_depends_on_agent = True  # the feedback turn and the stored interaction quote the agent's answers

    GOAL = (
        "You are {person}'s personal AI assistant. Your memory below holds {person}'s first-person record of "
        "the day (narration and dialogue, each line labelled with the day and time) and your past "
        "conversations with {person}. Answer the current request from {person} using the memory. Only use "
        "the provided information; do not fabricate. If the information is insufficient, say you don't know. "
        "Answer in natural conversational language, in one short paragraph."
    )

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.root = config.get("path", "data/streammembench/streammembench_v1_en")
        if not Path(self.root, "segments").exists():
            raise FileNotFoundError(f"StreamMemBench not found at {self.root}. Clone {SOURCE} and copy "
                                    "data/streammembench_v1_en there")
        self.participants = tuple(config.get("participants", EVALUATION_PARTICIPANTS))
        sample = dict(config.get("sample") or {})
        self.items = sample_items(self.root, self.participants, int(sample.get("seed", 0)), sample.get("items"))
        days = sorted({(p, d) for p, d, _ in self.items})
        shard = dict(config.get("shard") or {})
        if shard:
            days = shard_days(self.root, self.items, int(shard["k"]))[int(shard["index"])]
        # Dropped after sampling and sharding, so the sample and the other shards stay as they are (§25 excludes
        # A3_TASHA DAY1: a stub probe printed its label-based evidence-in-view before the split was set up).
        excluded = {tuple(day) for day in config.get("exclude_days") or []}
        self.days = [day for day in days if day not in excluded]
        self.chunk_tokens = int(config.get("chunk_tokens", 150))
        self.unit = config.get("unit", "chunk")
        if self.unit not in ("chunk", "segment"):
            raise ValueError(f"unknown unit {self.unit!r}")
        self.simulator = UserSimulator(config.get("simulator") or {"style": "lexical"})
        self.goal = ""
        self._queue: list[Observation] = []
        self._index = 0
        self._last_reward = 0.0
        self._results: list[dict] = []

    # ---- the static stream ---------------------------------------------------------------------------
    def day_of(self, seed: int) -> tuple[str, str]:
        if not self.days:
            raise ValueError("no participant-day in this sample and shard")
        return self.days[seed % len(self.days)]

    def _build(self, participant: str, day: str):
        """(observations, anchors by segment, evidence chunk ids by (segment, evidence_id)) for one day."""
        person = person_of(participant)
        segments = _json(str(Path(self.root, "segments", participant, f"{day}.json")))["segments"]
        evaluated = {s for p, d, s in self.items if (p, d) == (participant, day)}
        tasks = _json(str(Path(self.root, "evidence_tasks", participant, f"{day}.clean.json")))["items"]
        anchors = {int(item["segment_id"]): item["evidence_anchors"] for item in tasks if int(item["segment_id"]) in evaluated}
        last = max(evaluated)
        stream, chunk_of, chunks_in = [], {}, {}
        for segment in segments:
            number = int(segment["segment_id"])
            if number > last:
                break
            observations = segment["stream_segment"]["observations"]
            groups = [observations] if self.unit == "segment" else chunk_segment(observations, self.chunk_tokens)
            metadata = {"speaker": person, "date": f"{day} {segment.get('time_range', '')}".strip(), "segment": number}
            ids = []
            for position, group in enumerate(groups):
                item_id = f"{participant}/{day}/s{number:03d}/c{position:02d}"
                stream.append(Observation(item_id, "\n".join(o["text"] for o in group), SourceType.USER, metadata=dict(metadata)))
                ids.append(item_id)
                for o in group:
                    chunk_of[int(o["metadata"]["stream_index"])] = (item_id, o["text"], number)
            chunks_in[number] = ids
        evidence = {}
        for number, found in anchors.items():
            for anchor in found:
                ids = []
                for o in anchor.get("supporting_observations", []):
                    hit = chunk_of.get(int(o.get("metadata", {}).get("stream_index", -1)))
                    if hit and hit[1] == o.get("text") and hit[2] == number and hit[0] not in ids:
                        ids.append(hit[0])
                evidence[(number, anchor["evidence_id"])] = tuple(ids) or tuple(chunks_in[number])
        return stream, anchors, evidence

    def history_tokens(self, seed: int, count_labels: bool = False) -> int:
        """Tokens of the day's stream up to the last evaluated segment: the 100% budget. The requests, the
        feedback and the stored interactions depend on the agent and are not counted (no reference pass)."""
        stream, _, _ = self._build(*self.day_of(seed))
        return sum(count_tokens(label_prefix(o.metadata, o.source_type) + o.content if count_labels else o.content)
                   for o in stream)

    # ---- the episode ---------------------------------------------------------------------------------
    def reset(self, seed: int) -> Observation:
        participant, day = self.day_of(seed)
        self.person = person_of(participant)
        self.goal = self.GOAL.format(person=self.person)
        stream, anchors, self._evidence = self._build(participant, day)
        self._episode_id = f"{participant}/{day}"
        self._queue = []
        self._anchor_of: dict[str, dict] = {}
        last_of = {o.metadata["segment"]: o.id for o in stream}  # the last chunk of each segment
        for observation in stream:
            self._queue.append(observation)
            number = observation.metadata["segment"]
            if number in anchors and observation.id == last_of[number]:
                for anchor in anchors[number]:
                    self._queue.append(self._request(participant, day, number, anchor, "init",
                                                     anchor["tasks"]["initial_task"]["user_request"]))
        self.horizon = len(self._queue) + 2 * sum(len(a) for a in anchors.values())  # plus revisions and records
        self._index, self._last_reward, self._results = 0, 0.0, []
        self._open: dict[str, dict] = {}
        self._step_of: dict[str, int] = {}
        self.simulator.calls, self.simulator.parse_failures = 0, 0
        return self._queue[0]

    def _request(self, participant, day, number, anchor, kind, text) -> Observation:
        key = f"{participant}/{day}/s{number:03d}/{anchor['evidence_id']}"
        observation = Observation(f"{key}#{kind}", f"{self.person} asks you: {text}", SourceType.USER,
                                  requires_response=True, metadata={"speaker": self.person, "kind": kind})
        self._anchor_of[observation.id] = {"key": key, "segment": number, "anchor": anchor}
        return observation

    def step(self, agent_action: str | None) -> StepResult:
        current = self._queue[self._index]
        self._step_of.setdefault(current.id, self._index + 1)
        info: dict = {"scored": False, "correct": None}
        self._last_reward = 0.0
        follow: list[Observation] = []
        if current.requires_response:
            entry = self._anchor_of[current.id]
            anchor, key = entry["anchor"], entry["key"]
            kind = current.id.rsplit("#", 1)[1]
            answer = agent_action or ""
            initial, followup = anchor["tasks"]["initial_task"], anchor["tasks"]["followup_task"]
            record = self._open.setdefault(key, {"query_id": key, "segment": entry["segment"],
                                                 "evidence_id": anchor["evidence_id"]})
            expected = followup["expected_behavior"] if kind == "follow" else initial["expected_behavior"]
            request = followup["user_request"] if kind == "follow" else initial["user_request"]
            # No answer (a reference pass with a null agent): no simulator call, counted as a revision.
            verdict = (self.simulator.judge(anchor, request, answer, expected) if agent_action is not None
                       else {"type": "revise", "reason": "no answer", "feedback": "", "parsed": True})
            affirmed = verdict["type"] == "affirm"
            if kind == "init":
                record.update(initial_answer=answer, initial_evidence_use=int(affirmed), feedback_type=verdict["type"],
                              feedback=verdict["feedback"], feedback_reason=verdict["reason"],
                              feedback_incorporation=None, feedback_incorporation_applicable=not affirmed,
                              revised_answer="")
                if affirmed:
                    follow = self._record_and_followup(entry, record)
                else:
                    text = (f"{self.person} asked you: {request}\nYou answered: {answer}\n{self.person} replies: "
                            f"{verdict['feedback']}\nAnswer {self.person}'s request again, using this feedback.")
                    revise = Observation(f"{key}#revise", text, SourceType.USER, requires_response=True,
                                         metadata={"speaker": self.person, "kind": "revise"})
                    self._anchor_of[revise.id] = entry
                    follow = [revise]
            elif kind == "revise":
                record.update(revised_answer=answer, feedback_incorporation=int(affirmed),
                              incorporation_reason=verdict["reason"])
                follow = self._record_and_followup(entry, record)
            else:
                record.update(followup_answer=answer, followup_reuse=int(affirmed), followup_reason=verdict["reason"])
                self._results.append(self._open.pop(key))
            self._last_reward = float(affirmed)
            info = {"scored": True, "correct": affirmed, "gold": expected, "query_id": current.id, "category": kind}
        self._index += 1
        self._queue[self._index:self._index] = follow
        return StepResult(self.get_observation(), self._last_reward, self.is_done(), info)

    def _record_and_followup(self, entry: dict, record: dict) -> list[Observation]:
        """Their store_interaction, then the follow-up request."""
        anchor, key = entry["anchor"], entry["key"]
        text = interaction_text(self.person, anchor["tasks"]["initial_task"]["user_request"], record["initial_answer"],
                                record["feedback"] or record["feedback_type"], record["revised_answer"])
        stored = Observation(f"{key}#record", text, SourceType.USER,
                             metadata={"speaker": f"{self.person} and assistant", "kind": "record"})
        participant, day = key.split("/")[:2]
        followup = self._request(participant, day, entry["segment"], anchor, "follow",
                                 anchor["tasks"]["followup_task"]["user_request"])
        return [stored, followup]

    def get_observation(self) -> Observation | None:
        return None if self.is_done() else self._queue[self._index]

    def is_done(self) -> bool:
        return self._index >= len(self._queue)

    def get_reward(self) -> float:
        return self._last_reward

    def task_success(self) -> float:
        """Follow-up reuse: the share of finished anchors whose follow-up answer the user affirmed."""
        return sum(r["followup_reuse"] for r in self._results) / len(self._results) if self._results else 0.0

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        """The evidence chunks for the initial and revised answers; for the follow-up, the evidence chunks or
        the stored interaction (either carries what the follow-up needs)."""
        found = []
        for query_id, step in self._step_of.items():
            if query_id not in self._anchor_of:
                continue
            entry = self._anchor_of[query_id]
            ids = self._evidence[(entry["segment"], entry["anchor"]["evidence_id"])]
            kind = query_id.rsplit("#", 1)[1]
            if kind == "follow":
                ids = ids + (f"{entry['key']}#record",)
            found.append(Dependency(query_id, step, (EvidenceRequirement(tuple(ids), ""),), None, kind))
        return found

    def episode_stats(self) -> dict:
        done = self._results

        def mean(key, rows):
            values = [r[key] for r in rows if r.get(key) is not None]
            return sum(values) / len(values) if values else None

        applicable = [r for r in done if r["feedback_incorporation_applicable"]]
        return {
            "episode": getattr(self, "_episode_id", None),
            "anchors": len(done),
            "initial_evidence_use": mean("initial_evidence_use", done),
            "feedback_incorporation": mean("feedback_incorporation", applicable),
            "feedback_incorporation_applicable": len(applicable),
            "followup_reuse": mean("followup_reuse", done),
            "simulator_calls": self.simulator.calls,
            "simulator_parse_failures": self.simulator.parse_failures,
            "per_anchor": [{k: v for k, v in r.items() if not k.endswith("_answer")} for r in done],
        }
