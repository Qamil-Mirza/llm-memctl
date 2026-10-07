"""Retrieval headroom: how much of a question's evidence a search over the whole history finds in its top k.

No controller and no reader: every turn of the conversation is a candidate, the question is the query,
and a requirement counts as found when any of its items is in the top k. Two rules per question: the share of its requirements found (`recall@k`), and
whether all of them are (`complete@k`, the one that bounds a controller on multi-evidence questions).
This bounds what a retrieval
floor of k (or a pick-k-of-N head over a shortlist of N) can reach, per question type, before any LLM
evaluation (peer review T2).

Searches:
- `bm25`: the lexical retriever controllers use, over the turn text.
- `bm25_label`: the same, with "speaker (date):" prefixed to each turn and the question date kept.
- `bm25_bridge`: `bm25` top k plus the follow-the-clue search's 4 items (so k + 4 items).
- `dense` (with --dense): cosine similarity of sentence embeddings (default bge-small) of labelled text.
- `fusion` (with --dense): reciprocal rank fusion of `bm25_label` and `dense`.

    python -m memctl.rl.headroom --dataset locomo --dense --out runs/t2_headroom/locomo.json
    python -m memctl.rl.headroom --dataset longmemeval --out runs/t2_headroom/longmemeval.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from memctl.envs.locomo import _load as load_locomo
from memctl.envs.locomo import parse_conversation
from memctl.envs.longmemeval import _load as load_longmemeval
from memctl.envs.longmemeval import parse_instance
from memctl.memory.items import MemoryItem, count_tokens
from memctl.retrieval import LexicalRetriever, bridge_search
from memctl.runlog import collect_metadata

KS = (5, 8, 12, 16, 20)


def label(turn) -> str:
    return f"{turn.metadata.get('speaker', '')} ({turn.metadata.get('date', '')}): {turn.content}"


def questions(dataset: str, path: str):
    """(conversation number, category, question text, turns, requirements) for every question with
    evidence, refusal-scored ones excepted."""
    if dataset == "locomo":
        for number, sample in enumerate(load_locomo(path)):
            episode = parse_conversation(sample, include_adversarial=False)
            for question in episode.questions:
                if question.evidence_ids:
                    yield number, question.category, question.question, episode.turns, [(i,) for i in question.evidence_ids]
    else:
        for number, instance in enumerate(load_longmemeval(path)):
            episode, fallback = parse_instance(instance)
            [question] = episode.questions
            if question.unanswerable:
                continue
            requirements = [(i,) for i in question.evidence_ids] + [tuple(ids) for ids in fallback]
            if requirements:
                yield number, question.category, question.question, episode.turns, requirements


def ranks(order: list[str], requirements: list[tuple[str, ...]]) -> list[int]:
    """For each requirement, the 1-based rank of its best item in `order` (a large number if absent)."""
    position = {item_id: rank for rank, item_id in enumerate(order, 1)}
    return [min((position.get(i, 10**9) for i in ids), default=10**9) for ids in requirements]


def measure(dataset: str, path: str, dense_model: str | None = None) -> dict:
    retriever = LexicalRetriever()
    encoder = None
    if dense_model:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(dense_model, device="cpu")
    depth = max(KS)
    found: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    cache: dict[int, tuple] = {}
    for key, category, text, turns, requirements in questions(dataset, path):
        if key not in cache:
            plain = [MemoryItem(t.id, t.content, count_tokens(t.content), n) for n, t in enumerate(turns)]
            labelled = [MemoryItem(t.id, label(t), count_tokens(t.content), n) for n, t in enumerate(turns)]
            vectors = encoder.encode([label(t) for t in turns], normalize_embeddings=True, batch_size=64) if encoder else None
            cache = {key: (plain, labelled, vectors)}  # one conversation at a time
        plain, labelled, vectors = cache[key]
        orders = {
            "bm25": [item.id for item, _ in retriever.search(text, plain, depth)],
            "bm25_label": [item.id for item, _ in retriever.search(text, labelled, depth)],
        }
        bridged = [item.id for item, _ in bridge_search(retriever, text, [], plain, 4)]
        orders["bm25_bridge"] = {k: orders["bm25"][:k] + [i for i in bridged if i not in orders["bm25"][:k]] for k in KS}
        if encoder is not None:
            query = encoder.encode([text], normalize_embeddings=True)[0]
            scores = np.asarray(vectors) @ query
            dense = [plain[n].id for n in np.argsort(-scores)[:depth * 3]]
            orders["dense"] = dense[:depth]
            fused = defaultdict(float)
            for order in (orders["bm25_label"], dense):
                for rank, item_id in enumerate(order, 1):
                    fused[item_id] += 1.0 / (60 + rank)
            orders["fusion"] = sorted(fused, key=lambda i: -fused[i])[:depth]
        for name, order in orders.items():
            if isinstance(order, dict):  # one list per k (bm25 top k plus the bridge items)
                hits = [[r < 10**9 for r in ranks(order[k], requirements)] for k in KS]
            else:
                rs = ranks(order, requirements)
                hits = [[r <= k for r in rs] for k in KS]
            # any: share of the question's requirements found; all: 1 if every requirement is found
            found[name][category].append([sum(h) / len(h) for h in hits] + [float(all(h)) for h in hits])
    table = {}
    for name, by_category in found.items():
        table[name] = {}
        pooled = [row for rows in by_category.values() for row in rows]
        for category, rows in sorted(by_category.items()) + [("all", pooled)]:
            table[name][category] = {
                "n": len(rows),
                **{f"recall@{k}": round(float(np.mean([r[j] for r in rows])), 4) for j, k in enumerate(KS)},
                **{f"complete@{k}": round(float(np.mean([r[len(KS) + j] for r in rows])), 4) for j, k in enumerate(KS)},
            }
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--path")
    parser.add_argument("--dense", nargs="?", const="BAAI/bge-small-en-v1.5", default=None,
                        help="also measure a dense retriever (sentence-transformers model) and fusion")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.path or ("data/locomo/locomo10.json" if args.dataset == "locomo"
                         else "data/longmemeval/longmemeval_s_cleaned.json")
    table = measure(args.dataset, path, args.dense)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"dataset": args.dataset, "path": path, "dense": args.dense, "table": table,
                                    "metadata": collect_metadata()}, indent=2))
    for name, by_category in table.items():
        print(f"\n{name}")
        for category, row in by_category.items():
            print(f"  {category:28s} n={row['n']:4d} share " + " ".join(f"@{k}={row[f'recall@{k}']:.3f}" for k in KS)
                  + "  | all " + " ".join(f"@{k}={row[f'complete@{k}']:.3f}" for k in KS))


if __name__ == "__main__":
    main()
