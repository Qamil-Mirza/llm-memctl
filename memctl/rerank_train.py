"""Fit and select the §32 pointwise rerankers on the TRAINING part only (EXPERIMENTS.md §32).

    python -m memctl.rerank_train --data runs/exp32_rerankers/data --models configs/sweeps/exp32/models \
        --out runs/exp32_rerankers/inner_selection.json

Per fold k, from memctl/rerank_data.py's rows:
- fit on half A of the fold's training part (the §19 head A's own rows, expert-driven, no DAgger);
- score every setting on half B (the inner validation split; head A never saw it), as the test folds are played;
- the setting with the highest inner all-found@8 is kept (ties: higher requirement recall, then the simpler setting).
The model fitted on half A with that setting is the fold's model; it is not refitted. Nothing reads a test part.

Grids (fixed before any fit; written in EXPERIMENTS.md §32):
- lr_pointwise: standardised L2 logistic regression, C in {0.001, 0.01, 0.1, 1, 10, 100} x class_weight in
  {none, balanced}; lbfgs, max_iter 5000.
- gbdt_pointwise: LightGBM is not installed, so sklearn HistGradientBoostingClassifier: max_depth in {3, 6, none} x
  max_leaf_nodes in {7, 15, 31}, learning_rate 0.05, rounds in {25, 50, 100, 200, 400} (read off one 400-round fit
  with staged_decision_function), no internal early stopping, random_state 0.
The best classical reranker (lr or gbdt) is the one with the higher inner all-found@8 pooled over the five folds' half
B (ties: recall, then lr). Descriptive inner-split rows for bm25, rrf, the head (fixed8) and the zero-shot cross-encoder
are computed the same way.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from memctl.rerank import LogisticModel, gbdt_size, rows, top_k

K = 8
LR_GRID = [(c, w) for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0) for w in (None, "balanced")]
GBDT_GRID = [(d, leaves) for d in (3, 6, None) for leaves in (7, 15, 31)]
GBDT_ROUNDS = (25, 50, 100, 200, 400)


def load(data: Path, fold: int, part: str) -> list[dict]:
    with open(data / f"f{fold}_{part}.pkl", "rb") as handle:
        return pickle.load(handle)["records"]


def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([rows(r["items"], r["globals"]) for r in records])
    y = np.concatenate([r["label"] for r in records]).astype(np.int64)
    group = np.concatenate([np.full(len(r["label"]), n) for n, r in enumerate(records)])
    return x, y, group


def metrics(records: list[dict], scores: list[np.ndarray], k: int = K) -> dict:
    """All-found@k, requirement recall@k and precision@k (share of the shown that is evidence), over the questions
    with at least one requirement, from the candidates' own requirement sets (no budget, no forced removal)."""
    found_all, recall, precision, n = 0, 0.0, 0.0, 0
    for record, score in zip(records, scores):
        if not record["n_requirements"]:
            continue
        shown = set(top_k(score, k))
        hits = [bool(shown & set(found)) for found in record["requirements"]]
        found_all += all(hits)
        recall += sum(hits) / len(hits)
        precision += len(shown & set(record["evidence"])) / max(1, len(shown))
        n += 1
    return {"all_found": found_all / n, "recall": recall / n, "precision": precision / n, "questions": n}


def split_scores(flat: np.ndarray, records: list[dict]) -> list[np.ndarray]:
    out, start = [], 0
    for record in records:
        out.append(flat[start:start + len(record["label"])])
        start += len(record["label"])
    return out


def better(a: dict, b: dict | None) -> bool:
    return b is None or (a["all_found"], a["recall"]) > (b["all_found"], b["recall"])


def fit_lr(x, y, c, weight) -> LogisticModel:
    from sklearn.linear_model import LogisticRegression

    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    model = LogisticRegression(C=c, class_weight=weight, max_iter=5000)
    model.fit((x - mean) / scale, y)
    return LogisticModel(mean, scale, model.coef_[0], model.intercept_[0],
                         {"C": c, "class_weight": weight, "n_iter": int(model.n_iter_[0])})


def fit_gbdt(x, y, depth, leaves):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_depth=depth, max_leaf_nodes=leaves, learning_rate=0.05,
                                           max_iter=max(GBDT_ROUNDS), early_stopping=False, random_state=0)
    return model.fit(x, y)


def truncate_gbdt(model, rounds: int):
    """The same fitted model cut to its first `rounds` trees (equal to a fit with max_iter = rounds)."""
    import copy

    cut = copy.deepcopy(model)
    cut._predictors = cut._predictors[:rounds]
    cut.max_iter = rounds
    return cut


def select_fold(data: Path, fold: int, models: Path) -> dict:
    train, valid = load(data, fold, "train_a"), load(data, fold, "train_b")
    x, y, _ = matrix(train)
    xv, _, _ = matrix(valid)
    result = {"fold": fold, "train_rows": int(len(y)), "train_positive": int(y.sum()), "train_questions": len(train),
              "valid_questions": len(valid), "lr": [], "gbdt": []}
    best_lr = None
    for c, weight in LR_GRID:
        started = time.time()
        model = fit_lr(x, y, c, weight)
        score = metrics(valid, split_scores(model.decision(xv), valid))
        row = {"C": c, "class_weight": weight, **score, "seconds": round(time.time() - started, 2)}
        result["lr"].append(row)
        if better(score, best_lr and best_lr[0]):
            best_lr = (score, model, row)
    best_gbdt = None
    for depth, leaves in GBDT_GRID:
        started = time.time()
        model = fit_gbdt(x, y, depth, leaves)
        staged = list(model.staged_decision_function(xv))
        for rounds in GBDT_ROUNDS:
            score = metrics(valid, split_scores(np.ravel(staged[rounds - 1]), valid))
            row = {"max_depth": depth, "max_leaf_nodes": leaves, "rounds": rounds, **score}
            result["gbdt"].append(row)
            if better(score, best_gbdt and best_gbdt[0]):
                best_gbdt = (score, (model, rounds), row)
        result["gbdt"][-1]["seconds"] = round(time.time() - started, 2)
    models.mkdir(parents=True, exist_ok=True)
    lr_model = best_lr[1]
    (models / f"lr_f{fold}.json").write_text(json.dumps(lr_model.to_json()))
    gbdt_model = truncate_gbdt(*best_gbdt[1])
    check = metrics(valid, split_scores(gbdt_model.decision_function(xv), valid))
    if abs(check["all_found"] - best_gbdt[0]["all_found"]) > 1e-12:
        raise RuntimeError("truncated GBDT does not reproduce its staged score")
    with open(models / f"gbdt_f{fold}.pkl", "wb") as handle:
        pickle.dump(gbdt_model, handle)
    result["chosen"] = {"lr": {**best_lr[2], "parameters": lr_model.parameters},
                        "gbdt": {**best_gbdt[2], "parameters": gbdt_size(gbdt_model)}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and select the §32 pointwise rerankers (training part only).")
    parser.add_argument("--data", type=Path, default=Path("runs/exp32_rerankers/data"))
    parser.add_argument("--models", type=Path, default=Path("configs/sweeps/exp32/models"))
    parser.add_argument("--out", type=Path, default=Path("runs/exp32_rerankers/inner_selection.json"))
    args = parser.parse_args()
    folds = []
    for fold in range(5):
        result = select_fold(args.data, fold, args.models)
        folds.append(result)
        print(f"fold {fold}: lr {result['chosen']['lr']}\n        gbdt {result['chosen']['gbdt']}", flush=True)
    pooled = {}
    for family in ("lr", "gbdt"):
        n = sum(f["chosen"][family]["questions"] for f in folds)
        pooled[family] = {key: sum(f["chosen"][family][key] * f["chosen"][family]["questions"] for f in folds) / n
                          for key in ("all_found", "recall", "precision")}
    lr, gbdt = pooled["lr"], pooled["gbdt"]
    best = "gbdt" if (gbdt["all_found"], gbdt["recall"]) > (lr["all_found"], lr["recall"]) else "lr"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"folds": folds, "pooled": pooled, "best_classical": best}, indent=1))
    print("pooled inner validation (half B):", json.dumps(pooled), "-> best classical:", best)


if __name__ == "__main__":
    main()
