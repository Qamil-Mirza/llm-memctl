"""§32 cross-encoder: inner-split rows for the unlearned arms, and CPU fine-tuning (training part only).

    python -m memctl.rerank_ce inner --fold 0      # half-B metrics of bm25, rrf, the head (fixed8), ce zero-shot
    python -m memctl.rerank_ce tune --fold 0       # fine-tune on half A, pick the epoch on half B

Fine-tuning (fixed before any run, EXPERIMENTS.md §32): cross-encoder/ms-marco-MiniLM-L-6-v2 on (question, turn
text) pairs from half A's rows (memctl/rerank_data.py): every expert-labelled positive (the designated carrier, the
head's label) and 7 negatives per question drawn uniformly from its other candidates (seed 0); binary cross-entropy
on the logit; AdamW, lr 2e-5, batch 16, linear warm-up over the first 10% of steps then linear decay, max_length 512,
2 epochs. After each epoch, half B (the inner validation split) is scored as the test folds are played; the epoch with
the higher all-found@8 is kept (ties: recall, then the earlier epoch). Nothing reads a test part.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from memctl.rerank import RRF_CONSTANT, rrf_scores
from memctl.rerank_train import load, metrics
from memctl.runlog import peak_rss_mb

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HEAD = "/home/qamil-mirza/Code/llm-memctl/runs/lme_n4_head_f{fold}_a/checkpoints/policy_best.pt"


def ce_scores(model, records: list[dict], batch: int = 64) -> list[np.ndarray]:
    out = []
    for record in records:
        values = model.predict([(record["query"], text) for text in record["texts"]], batch_size=batch,
                               show_progress_bar=False)
        out.append(np.asarray(values, dtype=np.float64))
    return out


def head_scores(fold: int, records: list[dict]) -> list[np.ndarray]:
    from memctl.rl.policy import RETRIEVE_COLUMN, ItemPolicy

    checkpoint = torch.load(HEAD.format(fold=fold), weights_only=True)
    s = checkpoint["policy"]
    policy = ItemPolicy(s["item_dim"], s["global_dim"], s["hidden"], s["architecture"], s.get("residual"),
                        s.get("residual_index"))
    policy.load_state_dict(checkpoint["state_dict"])
    out = []
    with torch.no_grad():
        for record in records:
            logits, _ = policy(torch.from_numpy(record["items"]), torch.from_numpy(record["globals"]))
            out.append(logits[:, RETRIEVE_COLUMN].numpy().astype(np.float64))
    return out


def inner(fold: int, data: Path) -> dict:
    from sentence_transformers import CrossEncoder

    from memctl.retrieval import DenseRetriever

    valid = load(data, fold, "train_b")
    dense = DenseRetriever()
    rrf = [rrf_scores(dense.vectors(r["labelled"]) @ dense.vectors([r["query"]])[0]) for r in valid]
    rows = {
        "bm25_top8": metrics(valid, [-np.arange(len(r["label"]), dtype=np.float64) for r in valid]),
        "rrf_top8": metrics(valid, rrf),
        "fixed8": metrics(valid, head_scores(fold, valid)),
        "cross_encoder_zero": metrics(valid, ce_scores(CrossEncoder(BASE, device="cpu", max_length=512), valid)),
    }
    return {"fold": fold, "rrf_constant": RRF_CONSTANT, **rows}


def pairs_of(records: list[dict], negatives: int, rng: random.Random) -> list[tuple[str, str, float]]:
    pairs = []
    for record in records:
        labels = record["label"]
        positive = [j for j in range(len(labels)) if labels[j]]
        negative = [j for j in range(len(labels)) if not labels[j]]
        chosen = positive + rng.sample(negative, min(negatives, len(negative)))
        pairs += [(record["query"], record["texts"][j], float(labels[j])) for j in chosen]
    return pairs


def tune(fold: int, data: Path, out: Path, epochs: int = 2, negatives: int = 7, lr: float = 2e-5, batch: int = 16,
         max_length: int = 512, seed: int = 0) -> dict:
    from sentence_transformers import CrossEncoder

    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, valid = load(data, fold, "train_a"), load(data, fold, "train_b")
    pairs = pairs_of(train, negatives, rng)
    ce = CrossEncoder(BASE, device="cpu", max_length=max_length)
    model, tokenizer = ce.model, ce.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(pairs) + batch - 1) // batch)
    warm = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    log, best, started = [], None, time.time()
    folder = out / f"ce_tuned_f{fold}"
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch):
            chunk = [pairs[i] for i in order[start:start + batch]]
            encoded = tokenizer([p[0] for p in chunk], [p[1] for p in chunk], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits.view(-1)
            loss = loss_fn(logits, torch.tensor([p[2] for p in chunk]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss) * len(chunk)
        model.eval()
        score = metrics(valid, ce_scores(ce, valid))
        row = {"epoch": epoch, "train_loss": total / len(pairs), **score, "seconds": round(time.time() - started, 1)}
        log.append(row)
        print(f"fold {fold} epoch {epoch}: {row}", flush=True)
        if best is None or (score["all_found"], score["recall"]) > (best["all_found"], best["recall"]):
            best = row
            ce.save(str(folder))
    result = {"fold": fold, "pairs": len(pairs), "positives": int(sum(p[2] for p in pairs)), "epochs": log,
              "chosen_epoch": best["epoch"], "chosen": best, "model_dir": str(folder),
              "parameters": sum(p.numel() for p in model.parameters()), "peak_rss_mb": peak_rss_mb(),
              "threads": torch.get_num_threads()}
    (out / f"ce_tuned_f{fold}.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="§32 cross-encoder: inner rows and CPU fine-tuning.")
    parser.add_argument("command", choices=("inner", "tune"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/exp32_rerankers/data"))
    parser.add_argument("--out", type=Path, default=Path("runs/exp32_rerankers"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "inner":
        result = inner(args.fold, args.data)
        (args.out / f"inner_unlearned_f{args.fold}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result))
    else:
        tune(args.fold, args.data, args.out)


if __name__ == "__main__":
    main()
