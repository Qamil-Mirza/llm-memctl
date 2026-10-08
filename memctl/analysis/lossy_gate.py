"""The free oracle gate of Experiment 21 (EXPERIMENTS.md §21, pre-registered): LoCoMo under lossy memory.

A scripted simulation, no model. The conversation streams session by session. Each question is asked right after
the session that follows its last evidence session (at the end if that is the last). Raw turns older than w
sessions are gone; what survives is one extractive note per earlier session (a set of its turns within a token
budget, stored as one searchable item). At a question the reader sees the top k by fusion search over the
survivors; a question is correct iff every evidence turn is in view.

Note policies: `oracle` (hindsight: the session's evidence turns of later questions, most questions served first),
`learned` (cross-fitted: a logistic model, trained on the other conversations, scores each turn's chance of being
needed by a question asked w or more sessions later, from question-blind features: log length, salience, digit and
capitalised-word counts, position in the session, whether the turn asks a question, speaker, and its bge-small
embedding; the session's highest-scoring turns fill the note), and the question-blind rules `salience`, `first`, `shortest`, `random`. The `unlimited` control keeps every raw
turn searchable, with the same notes beside them. B questions ("Earlier you were asked ... what did you answer?")
test retention of the stored answer items.

    python -m memctl.analysis.lossy_gate --out runs/exp21_gate.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from memctl.envs import build_env
from memctl.envs.locomo import _load as load_locomo, parse_conversation
from memctl.memory.items import MemoryItem, count_tokens
from memctl.retrieval import FusionRetriever, labelled_text
from memctl.sysinfo import collect_metadata

POLICIES = ("oracle", "learned", "salience", "first", "shortest", "random")
RULES = ("salience", "first", "shortest", "random")


def session_of(turn_id: str) -> int:
    return int(turn_id.split(":")[0][1:])  # "D3:4" -> 3


def salience(text: str) -> float:
    words = text.split()
    marked = sum(bool(re.search(r"\d", w)) or (i > 0 and w[:1].isupper()) for i, w in enumerate(words))
    return marked / max(1, len(words))


def choose(policy: str, turns: list, budget: int, served: dict[str, int], rng: random.Random,
           scores: dict[str, float] | None = None) -> list:
    """The turns of one session that go into its note, within `budget` tokens."""
    if policy == "oracle":
        order = sorted((t for t in turns if served.get(t.id)), key=lambda t: (-served[t.id], count_tokens(labelled_text(t))))
    elif policy == "learned":
        order = sorted(turns, key=lambda t: -scores[t.id])
    elif policy == "salience":
        order = sorted(turns, key=lambda t: -salience(t.content))
    elif policy == "first":
        order = list(turns)
    elif policy == "shortest":
        order = sorted(turns, key=lambda t: count_tokens(labelled_text(t)))
    elif policy == "random":
        order = rng.sample(turns, len(turns))
    else:
        raise ValueError(policy)
    picked, used = [], 0
    for turn in order:
        cost = count_tokens(labelled_text(turn))
        if used + cost <= budget:
            picked.append(turn)
            used += cost
    return picked


def item(turn, step: int) -> MemoryItem:
    return MemoryItem(turn.id, turn.content, count_tokens(turn.content), step, metadata=dict(turn.metadata))


def load(conversation: int):
    sample = load_locomo(build_env({"name": "locomo", "include_adversarial": False}).path)[conversation]
    return parse_conversation(sample, include_adversarial=False)


def placed(episode, w: int):
    """(asked-after session, oldest evidence session, question) for each question with evidence, and the hindsight
    count of how many questions asked w or more sessions later each turn serves."""
    last = max(int(t.metadata["session"]) for t in episode.turns)
    questions = []
    for q in episode.questions:
        if not q.evidence_ids:
            continue
        sessions = [session_of(e) for e in q.evidence_ids]
        # The delay after the last evidence session, 1 to 6 sessions, fixed per question (a hash of its id, not
        # Python's salted hash), so ages spread over the bins 1-2, 3-5 and 6+ (amended after the smoke test, §21).
        delay = 1 + int.from_bytes(q.id.encode()[-4:], "big") * 2654435761 % 6
        questions.append((min(max(sessions) + delay, last), min(sessions), q))
    served = defaultdict(int)
    for asked, _, q in questions:
        for e in q.evidence_ids:
            if asked - session_of(e) >= w:
                served[e] += 1
    return questions, served


def features(episode, encoder) -> tuple[list[str], list[list[float]]]:
    import math

    turns = episode.turns
    speakers = sorted({t.metadata.get("speaker", "") for t in turns})
    by_session = defaultdict(list)
    for t in turns:
        by_session[int(t.metadata["session"])].append(t)
    position = {t.id: i / max(1, len(ts) - 1) for ts in by_session.values() for i, t in enumerate(ts)}
    vectors = encoder.vectors([labelled_text(t) for t in turns])
    rows = []
    for t, v in zip(turns, vectors):
        words = t.content.split()
        rows.append([math.log1p(count_tokens(t.content)), salience(t.content), sum(bool(re.search(r"\d", x)) for x in words),
                     sum(x[:1].isupper() for x in words[1:]), position[t.id], float("?" in t.content),
                     float(speakers.index(t.metadata.get("speaker", ""))), *v.tolist()])
    return [t.id for t in turns], rows


def learned_scores(conversations: int, w: int, encoder) -> tuple[dict[int, dict[str, float]], dict]:
    """Cross-fitted needed-turn scores: each conversation is scored by a model fit on the other conversations."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    data = {}
    for c in range(conversations):
        episode = load(c)
        _, served = placed(episode, w)
        ids, x = features(episode, encoder)
        data[c] = (ids, np.array(x), np.array([float(served.get(i, 0) > 0) for i in ids]))
    scores, aucs, rates = {}, {}, {}
    for c in data:
        x = np.concatenate([data[o][1] for o in data if o != c])
        y = np.concatenate([data[o][2] for o in data if o != c])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(x), y)
        ids, xc, yc = data[c]
        p = model.predict_proba(scaler.transform(xc))[:, 1]
        scores[c] = dict(zip(ids, p.tolist()))
        aucs[c] = float(roc_auc_score(yc, p)) if 0 < yc.sum() < len(yc) else None
        rates[c] = float(yc.mean())
    return scores, {"auc": aucs, "base_rate": rates}


def simulate(conversation: int, w: int, budget: int, k: int, retriever: FusionRetriever,
             scores: dict[str, float] | None = None) -> list[dict]:
    episode = load(conversation)
    turns = episode.turns
    by_session = defaultdict(list)
    for t in turns:
        by_session[int(t.metadata["session"])].append(t)
    last = max(by_session)
    questions, served = placed(episode, w)
    rows = []
    for policy in POLICIES:
        rng = random.Random(conversation)
        notes = {s: choose(policy, by_session[s], budget, served, rng, scores) for s in sorted(by_session)}
        note_items = {s: MemoryItem(f"note:{s}", "\n".join(labelled_text(t) for t in picked) or "(empty)",
                                    sum(count_tokens(labelled_text(t)) for t in picked) or 1, s,
                                    metadata={"note_of": [t.id for t in picked]})
                      for s, picked in notes.items()}
        answers = {}
        for asked, oldest, q in sorted(questions, key=lambda x: x[0]):
            for setting in ("lossy", "unlimited"):
                window = [item(t, s) for s in range(max(1, asked - w + 1), asked + 1) for t in by_session.get(s, [])]
                older = [item(t, s) for s in range(1, asked - w + 1) for t in by_session.get(s, [])] if setting == "unlimited" else []
                kept = [note_items[s] for s in range(1, asked - w + 1)]
                survivors = window + older + kept
                view = set()
                for found, _ in retriever.search(q.question, survivors, k):
                    view |= set(found.metadata.get("note_of", [found.id]))
                alive = {i.id for i in window + older} | {t for n in kept for t in n.metadata["note_of"]}
                rows.append({"conversation": conversation, "policy": policy, "setting": setting, "question": q.id,
                             "kind": "A", "category": q.category, "age": asked - oldest, "old": asked - oldest >= w,
                             "correct": all(e in view for e in q.evidence_ids),
                             "survives": all(e in alive for e in q.evidence_ids)})
                if setting == "lossy":
                    answers[q.id] = (asked, q)
        # B: retention of the stored answer item, d = 3 sessions after A
        for qid, (asked, q) in answers.items():
            b_at = asked + 3
            if b_at > last:
                continue
            window = [item(t, s) for s in range(max(1, b_at - w + 1), b_at + 1) for t in by_session.get(s, [])]
            kept = [note_items[s] for s in range(1, b_at - w + 1)]
            answer = MemoryItem(f"answer:{qid}", f"Answer to '{q.question}': (the reader's answer)", 12, asked)
            others = [MemoryItem(f"answer:{o}", f"Answer to '{oq.question}': (the reader's answer)", 12, oa)
                      for o, (oa, oq) in answers.items() if oa <= b_at and o != qid]
            hits = {f.id for f, _ in retriever.search(f"Earlier you were asked: '{q.question}'. What did you answer?",
                                                      window + kept + [answer] + others, k)}
            rows.append({"conversation": conversation, "policy": policy, "setting": "lossy", "question": qid, "kind": "B",
                         "category": q.category, "age": 3, "old": True, "correct": answer.id in hits, "survives": True})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--conversations", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    retriever = FusionRetriever()
    scores, diagnostics = learned_scores(args.conversations, args.w, retriever.dense)
    rows = [r for c in range(args.conversations) for r in simulate(c, args.w, args.budget, args.k, retriever, scores[c])]
    args.out.write_text(json.dumps({"settings": vars(args) | {"out": str(args.out)}, "learned": diagnostics, "rows": rows,
                                    "metadata": collect_metadata()}))
    print(json.dumps(diagnostics))
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
