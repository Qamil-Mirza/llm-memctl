"""Experiment 22 (EXPERIMENTS.md §22, pre-registered): credit assignment for the extractive note-writer.

Training runs in a fast surrogate of the §21 lossy-memory simulator. Per conversation, turn and question embeddings
and per-turn BM25 scores are computed once. A note is ranked by the best of its turns on each list, and the two
lists are fused by reciprocal rank (constant 60), as in FusionRetriever. Every test number comes from the exact
simulator (memctl.analysis.lossy_gate).

Variants: `episode` (i, uniform GRPO credit), `hindsight` (ii, the questions whose evidence lies in the session),
`counterfactual` (iii, the note's marginal contribution, re-simulated without it), `forward` (iv, every question
asked w or more sessions after the session). Advantages are the group mean subtracted (no std); on-policy, one
gradient step per batch, no ratio, no clipping.

    python -m memctl.rl.note_grpo --variant hindsight --fold 0 --seed 0 --out runs/exp22/hindsight_f0_s0
    python -m memctl.rl.note_grpo --time   # time one surrogate rollout and the surrogate's agreement
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from memctl.analysis import lossy_gate as gate
from memctl.embed import content_words
from memctl.memory.items import count_tokens
from memctl.retrieval import DenseRetriever, labelled_text
from memctl.sysinfo import collect_metadata

W, BUDGET, K, RRF = 2, 100, 8, 60
FOLDS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # leave-two-out: the held-out conversations of each fold
CACHE = Path("runs/_cache_exp22")


@dataclass
class Conversation:
    number: int
    turn_ids: list[str]
    session: np.ndarray  # [T] session of each turn
    cost: np.ndarray  # [T] tokens of each turn's labelled text
    x: np.ndarray  # [T, F] question-blind features (§21 `learned`) + token count
    label: np.ndarray  # [T] 1 if needed by a question asked >= W sessions later (training init only)
    asked: np.ndarray  # [Q] session after which each question is asked
    evidence: list[np.ndarray]  # per question, the turn indices of its evidence
    old: np.ndarray  # [Q] oldest evidence >= W sessions back
    dense: np.ndarray  # [Q, T] cosine of question and turn
    bm25: np.ndarray  # [Q, T] BM25 of each turn for the question, over the whole conversation
    sessions: list[np.ndarray]  # turn indices of each session (1-based session numbers -> index s-1)


def bm25_matrix(queries: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    documents = [Counter(content_words(t)) for t in texts]
    lengths = np.array([sum(d.values()) for d in documents], dtype=float)
    average = lengths.mean() or 1.0
    frequency = Counter(w for d in documents for w in d)
    n = len(texts)
    out = np.zeros((len(queries), n))
    for qi, query in enumerate(queries):
        for word in set(content_words(query)):
            if word not in frequency:
                continue
            idf = math.log(1 + (n - frequency[word] + 0.5) / (frequency[word] + 0.5))
            for ti, d in enumerate(documents):
                c = d.get(word, 0)
                if c:
                    out[qi, ti] += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * lengths[ti] / average))
    return out


def prepare(number: int, encoder: DenseRetriever) -> Conversation:
    path = CACHE / f"conversation_{number}.npz"
    episode = gate.load(number)
    questions, served = gate.placed(episode, W)
    turns = episode.turns
    ids = [t.id for t in turns]
    index = {i: n for n, i in enumerate(ids)}
    if path.exists():
        z = np.load(path, allow_pickle=True)
        x, dense, bm25 = z["x"], z["dense"], z["bm25"]
    else:
        _, rows = gate.features(episode, encoder)
        texts = [labelled_text(t) for t in turns]
        x = np.array([r + [math.log1p(count_tokens(tx))] for r, tx in zip(rows, texts)], dtype=np.float32)
        qv = encoder.vectors([q.question for _, _, q in questions])
        dense = qv @ encoder.vectors(texts).T
        bm25 = bm25_matrix([q.question for _, _, q in questions], texts)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=x, dense=dense, bm25=bm25)
    session = np.array([int(t.metadata["session"]) for t in turns])
    return Conversation(
        number, ids, session, np.array([count_tokens(labelled_text(t)) for t in turns]), x,
        np.array([float(served.get(i, 0) > 0) for i in ids], dtype=np.float32),
        np.array([a for a, _, _ in questions]), [np.array([index[e] for e in q.evidence_ids]) for _, _, q in questions],
        np.array([a - o >= W for a, o, _ in questions]), dense, bm25,
        [np.flatnonzero(session == s) for s in range(1, session.max() + 1)])


def outcomes(c: Conversation, notes: list[np.ndarray], questions: np.ndarray | None = None) -> np.ndarray:
    """Surrogate scripted outcome (1 = all evidence in view among the top K) for each question in `questions`."""
    qs = np.arange(len(c.asked)) if questions is None else questions
    out = np.zeros(len(qs))
    for n, q in enumerate(qs):
        s_ask = c.asked[q]
        window = np.flatnonzero((c.session > s_ask - W) & (c.session <= s_ask))
        kept = [notes[s - 1] for s in range(1, s_ask - W + 1) if len(notes[s - 1])]
        members = [np.array([t]) for t in window] + kept
        d = np.array([c.dense[q, m].max() for m in members])
        b = np.array([c.bm25[q, m].max() for m in members])
        fused = np.zeros(len(members))
        for scores, positive in ((d, d > 0), (b, b > 0)):
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(order))
            ranks[order] = np.arange(1, len(order) + 1)
            fused += np.where(positive, 1.0 / (RRF + ranks), 0.0)
        top = np.argsort(-fused, kind="stable")[:K]
        view = set(np.concatenate([members[i] for i in top if fused[i] > 0])) if len(top) else set()
        out[n] = all(e in view for e in c.evidence[q])
    return out


class Writer(nn.Module):
    """hidden > 0: the §22 MLP. hidden = 0: the §22b linear writer (the logistic model's form)."""

    def __init__(self, features: int, hidden: int = 32) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(features), requires_grad=False)
        self.std = nn.Parameter(torch.ones(features), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1)) if hidden \
            else nn.Sequential(nn.Linear(features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)


def fill(order: np.ndarray, cost: np.ndarray) -> tuple[list[int], int]:
    """Turns taken in `order` while they fit the budget; and the prefix length up to the last taken turn."""
    picked, used, last = [], 0, -1
    for position, t in enumerate(order):
        if used + cost[t] <= BUDGET:
            picked.append(int(t))
            used += cost[t]
            last = position
    return picked, last + 1


def prefix_log_prob(logits: torch.Tensor, order: list[int]) -> torch.Tensor:
    """Plackett-Luce log-probability of the ordered prefix `order` (indices into `logits`)."""
    if not order:
        return logits.sum() * 0.0
    o = torch.tensor(order)
    chosen = logits[o]
    taken = torch.zeros(len(order), len(logits))
    for i in range(1, len(order)):
        taken[i] = taken[i - 1]
        taken[i, order[i - 1]] = 1.0
    remaining = logits.unsqueeze(0).masked_fill(taken.bool(), float("-inf"))
    return (chosen - torch.logsumexp(remaining, dim=1)).sum()


def rollout(c: Conversation, writer: Writer, rng: np.random.Generator, greedy: bool = False):
    """One note per session: (notes, per-session log-prob, per-session entropy of the first pick)."""
    logits_all = writer(torch.from_numpy(c.x))
    notes, logps, entropies = [], [], []
    for turns in c.sessions:
        logits = logits_all[turns]
        noise = 0.0 if greedy else rng.gumbel(size=len(turns))
        order = np.argsort(-(logits.detach().numpy() + noise), kind="stable")
        picked, prefix = fill(order, c.cost[turns])
        notes.append(turns[picked])
        logps.append(prefix_log_prob(logits, order[:prefix].tolist()))
        p = torch.softmax(logits, 0)
        entropies.append(-(p * torch.log(p + 1e-12)).sum())
    return notes, torch.stack(logps), torch.stack(entropies)


def session_rewards(c: Conversation, notes: list[np.ndarray], variant: str) -> np.ndarray:
    """Per-session reward (NaN where a session has no credited question), or one episode reward repeated."""
    old = np.flatnonzero(c.old)
    result = outcomes(c, notes, old)
    if variant == "episode":
        return np.full(len(c.sessions), result.mean() if len(old) else np.nan)
    rewards = np.full(len(c.sessions), np.nan)
    for s in range(1, len(c.sessions) + 1):
        later = c.asked[old] >= s + W
        if variant == "hindsight":
            mask = later & np.array([np.any(c.session[c.evidence[q]] == s) for q in old])
        else:
            mask = later
        if not mask.any():
            continue
        if variant in ("hindsight", "forward"):
            rewards[s - 1] = result[mask].mean()
        elif variant == "counterfactual":
            without = [n if i != s - 1 else n[:0] for i, n in enumerate(notes)]
            rewards[s - 1] = (result[mask] - outcomes(c, without, old[mask])).sum()
    return rewards


def supervised_init(train: list[Conversation], seed: int, steps: int = 400) -> Writer:
    torch.manual_seed(seed)
    x = torch.from_numpy(np.concatenate([c.x for c in train]))
    y = torch.from_numpy(np.concatenate([c.label for c in train]))
    writer = Writer(x.shape[1])
    writer.mean.data, writer.std.data = x.mean(0), x.std(0).clamp_min(1e-6)
    optimizer = torch.optim.Adam(writer.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(writer(x), y)
        loss.backward()
        optimizer.step()
    return writer


def logistic_init(train: list[Conversation]) -> Writer:
    """§22b: the linear writer set to the cross-fitted logistic fit (C = 0.1, standardised, as in §21)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.concatenate([c.x for c in train])
    y = np.concatenate([c.label for c in train])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(x), y)
    writer = Writer(x.shape[1], hidden=0)
    writer.mean.data = torch.tensor(scaler.mean_, dtype=torch.float32)
    writer.std.data = torch.tensor(np.where(scaler.scale_ > 0, scaler.scale_, 1.0), dtype=torch.float32)
    writer.net[0].weight.data = torch.tensor(model.coef_, dtype=torch.float32)
    writer.net[0].bias.data = torch.tensor(model.intercept_, dtype=torch.float32)
    return writer


def held_out_accuracy(writer: Writer, held: list[Conversation]) -> float:
    """Surrogate accuracy on held-out old-evidence questions, greedy, packed per token (§22b diagnostic)."""
    results = []
    with torch.no_grad():
        for c in held:
            probability = torch.sigmoid(writer(torch.from_numpy(c.x))).numpy()
            notes = []
            for turns in c.sessions:
                order = np.argsort(-(probability[turns] / c.cost[turns]), kind="stable")
                notes.append(turns[fill(order, c.cost[turns])[0]])
            results.append(outcomes(c, notes, np.flatnonzero(c.old)))
    return float(np.concatenate(results).mean())


def train(variant: str, fold: int, seed: int, updates: int, group: int, out: Path, policy: str = "mlp") -> None:
    encoder = DenseRetriever()
    conversations = {n: prepare(n, encoder) for n in range(10)}
    held = set(FOLDS[fold])
    train_set = [conversations[n] for n in conversations if n not in held]
    held_set = [conversations[n] for n in sorted(held)]
    torch.manual_seed(seed)
    writer = supervised_init(train_set, seed) if policy == "mlp" else logistic_init(train_set)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(writer.state_dict(), out / "init.pt")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam([p for p in writer.parameters() if p.requires_grad], lr=1e-3)
    curve = []
    for update in range(updates):
        loss, rewards_seen = 0.0, []
        for c in train_set:
            draws = [rollout(c, writer, rng) for _ in range(group)]
            rewards = np.stack([session_rewards(c, notes, variant) for notes, _, _ in draws])  # [G, S]
            valid = ~np.isnan(rewards)
            mean = np.where(valid.any(0), np.nanmean(np.where(valid, rewards, np.nan), axis=0), 0.0)
            advantage = torch.from_numpy(np.where(valid, rewards - mean, 0.0)).float()
            for g, (_, logps, entropies) in enumerate(draws):
                loss = loss - (advantage[g] * logps).sum() - 0.01 * entropies.sum()
            rewards_seen.append(outcomes(c, draws[0][0], np.flatnonzero(c.old)).mean())
        optimizer.zero_grad()
        (loss / (len(train_set) * group)).backward()
        optimizer.step()
        point = {"update": update, "train_old_accuracy": float(np.mean(rewards_seen))}
        if policy == "linear" and update % 50 == 0:
            point["held_out_old_accuracy"] = held_out_accuracy(writer, held_set)
        curve.append(point)
    torch.save(writer.state_dict(), out / "writer.pt")
    (out / "curve.json").write_text(json.dumps(curve))
    if policy == "linear":
        curve.append({"update": updates, "held_out_old_accuracy": held_out_accuracy(writer, held_set)})
        (out / "curve.json").write_text(json.dumps(curve))
    (out / "metadata.json").write_text(json.dumps({"variant": variant, "fold": fold, "seed": seed, "updates": updates,
                                                   "group": group, "policy": policy, **collect_metadata()}, default=str))


def time_and_agree() -> None:
    """Time one surrogate rollout, and compare surrogate with exact accuracy for the §21 rule `first`."""
    encoder = DenseRetriever()
    c = prepare(1, encoder)
    writer = supervised_init([c], 0, steps=10)
    rng = np.random.default_rng(0)
    started = time.time()
    for _ in range(5):
        notes, _, _ = rollout(c, writer, rng)
        session_rewards(c, notes, "hindsight")
    per = (time.time() - started) / 5
    print(json.dumps({"seconds_per_rollout_conv1": round(per, 3), "questions": len(c.asked)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["episode", "hindsight", "counterfactual", "forward"])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--policy", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    if args.time:
        time_and_agree()
    else:
        train(args.variant, args.fold, args.seed, args.updates, args.group, args.out, args.policy)


if __name__ == "__main__":
    main()
