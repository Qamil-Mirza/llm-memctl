"""Train the LRE baseline per fold on the training folds only, and compare its labels with ours (EXPERIMENTS.md §27).

    python -m memctl.lre_train --fold 0 --out runs/lre_exp27_f0     # one fold; writes lre.json and train.json

Rows: every turn of every training question's history (the `part: train` questions of the fold, abstention
questions included, as LRE's loader includes them), in the order the controller sees them (sessions by date). A
turn's label is LRE's rule against that question's own gold answer. The held-out fold is never read: only the
training indices are loaded.

Also written (descriptive, training side only): LRE's labels against our evidence labels. Our evidence turn is a
turn marked `has_answer`, or, for an answer session with no marked turn, any of its turns (the environment's
fallback: any one of them counts as evidence).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance, parse_instance
from memctl.lre import LREModel, answer_overlap_labels, traj_features
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"


def training_rows(path: str, k: int, fold: int, part: str = "train"):
    """(texts, trajs, lre labels, evidence labels, row kinds) over the training questions of one fold."""
    index = _index(path)
    texts, trajs, labels, evidence, kinds = [], [], [], [], []
    for number in fold_indices(index, k, fold, part):
        episode, fallback = parse_instance(_instance(path, number))
        question = episode.questions[0]
        turns = episode.turns
        marked = set(question.evidence_ids) | {i for ids in fallback for i in ids}
        n = len(turns)
        texts += [t.content for t in turns]
        trajs += [traj_features(t.content, i, n) for i, t in enumerate(turns)]
        labels += answer_overlap_labels([t.content for t in turns], [question.gold])
        evidence += [int(t.id in marked) for t in turns]
        kinds += ["abstention" if question.unanswerable else "answerable"] * n
    return texts, trajs, labels, evidence, kinds


def agreement(labels: list[int], evidence: list[int]) -> dict:
    both = sum(1 for a, b in zip(labels, evidence) if a and b)
    lre_pos, ev_pos, n = sum(labels), sum(evidence), len(labels)
    return {
        "rows": n,
        "lre_positive_rate": round(lre_pos / n, 5) if n else None,
        "evidence_positive_rate": round(ev_pos / n, 5) if n else None,
        "agreement": round(sum(1 for a, b in zip(labels, evidence) if a == b) / n, 5) if n else None,
        "precision_lre_vs_evidence": round(both / lre_pos, 4) if lre_pos else None,
        "recall_lre_vs_evidence": round(both / ev_pos, 4) if ev_pos else None,
        "lre_positives": lre_pos, "evidence_positives": ev_pos, "both": both,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--path", default=PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    texts, trajs, labels, evidence, kinds = training_rows(args.path, args.k, args.fold)
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    model = LREModel.fit(texts, trajs, labels, {"fold": args.fold, "k": args.k, "part": "train", "data": args.path})
    seconds = time.perf_counter() - started
    size = model.save(args.out / "lre.json")
    by_kind = {kind: agreement([l for l, k in zip(labels, kinds) if k == kind], [e for e, k in zip(evidence, kinds) if k == kind])
               for kind in ("answerable", "abstention")}
    report = {
        "fold": args.fold, "rows": len(texts), "load_seconds": round(loaded, 1), "train_seconds": round(seconds, 1),
        "model_bytes": size, "parameters": model.n_parameters, "vocabulary": len(model.vocabulary),
        "n_iter": model.data["train"]["n_iter"],
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
        "labels_vs_evidence": {"all": agreement(labels, evidence), **by_kind},
    }
    (args.out / "train.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
