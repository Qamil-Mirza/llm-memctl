"""Experiment 29 (§29): a reader-utility training signal for the §19 head.

The §19 head scores 32 BM25 candidates and shows its top 8 (`fixed8`). It was trained on evidence labels. Here the
same architecture is trained on what the reader gets from each candidate:

    utility(q, c) = log P(gold | question + turn c) - log P(gold | question alone)

from Qwen2.5-7B prompt logprobs, in the frozen 7b5fc30 reader prompt shape (see `reader_prompt`, `chat_text`).

Stages (python -m memctl.reader_utility <stage> ...):
  rows     CPU, free.     Every LongMemEval question's decision as the head sees it at test (the 32 candidates, their
                          features, evidence labels and reader lines). Reader-free; no label is used to build it.
  signal   pod step 1.    One prompt-logprob call per (answerable question, candidate) plus one baseline per question.
                          Cached per prompt (replayable; resumes against any base_url). Writes the utility rows.
  tune     CPU + pod.     Per fold, on an inner 6/2 split of that fold's TRAINING questions only: train a blend head
                          for each weight of the pre-declared grid on the 6/8, score the reader's log P(gold | question
                          + the head's top 8) on the 2/8, keep the best weight.
  train    CPU, free.     The final utility, blend and evidence-rows heads per fold, on the fold's training part.
  check    CPU, free.     A trained head inside the real controller picks the same 8 as on the rows (training
                          questions only).

The fold guard. The utility rows are fold-invariant (one reader; each row is made from its own question's gold
answer), so one signal pass serves all five folds. But a head for fold k must never train on rows of fold k's test
questions: `training_rows` takes the fold and part, reads only those questions, and raises if any test question of
the fold reaches it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
OUT = Path(os.environ.get("EXP29_DIR", "runs/exp29"))
HEADS = Path(os.environ.get("EXP29_HEADS", "configs/sweeps/exp29/heads"))
MODEL = "qwen2.5-7b-instruct"
K = 5
SHOW = 8
CANDIDATES = 32
BUDGET_FRACTION = 0.05  # the §23 evaluation setting, so training rows and test decisions share their features
GRID = (0.1, 0.25, 0.5, 0.75, 0.9)  # blend weight on the utility term; declared before any row exists
TAU = 1.0  # temperature (nats) of the utility target softmax; fixed, not tuned
EPOCHS, LR, BATCH, SEED = 40, 3e-3, 64, 0
# Qwen2.5-Instruct's chat template with no system message (it inserts this default), as vLLM's chat endpoint
# applies it to the reader's single user message.
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# ---- prompts ----------------------------------------------------------------------------------------------------

def reader_prompt(goal: str, lines: list[str], question: str) -> str:
    """The frozen 7b5fc30 reader prompt (labels: compact, memory in arrival order) with `reasoning: false`: the
    note instruction is left out and the cue is "Answer:", so the gold answer can follow directly. Equal to
    LLMAgent.build_prompt for that setting (checked on every row at capture)."""
    return "\n".join([goal, "", "## Memory", *(lines or ["(empty)"]), "", "## Current input", question, "",
                      "Answer:"])


def chat_text(user: str, gold: str) -> str:
    """The text whose tail is scored: the chat-formatted user message, then the gold answer as the assistant reply
    (no end token). The gold's tokens are the last n prompt tokens (n from the server's tokenizer)."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{gold}")


# ---- rows ---------------------------------------------------------------------------------------------------------

def capture_config(index: int) -> dict:
    """One question, as the evaluation plays it: the §23 memory settings, the §19 head's controller settings with an
    untrained policy (the decision's inputs do not depend on the weights: keep_none, one question per episode)."""
    return {
        "schema_version": 1, "seed": 0, "episodes": 1,
        "env": {"name": "longmemeval", "path": PATH, "subset": [index, index + 1]},
        "agent": {"name": "null"},
        "controller": {"name": "rl", "retrieval_method": "lexical", "retrieve_floor": SHOW,
                       "retrieve_candidates": CANDIDATES, "floor_head": True, "keep_none": True},
        "memory": {"allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"],
                   "embedder": "hashing", "count_labels": True, "budget": {"fraction": BUDGET_FRACTION}},
        "logging": {"detail_episodes": 0},
    }


def capture_one(index: int) -> dict:
    from memctl.agents.llm import LLMAgent, compact_line
    from memctl.controllers.rl import RLController
    from memctl.envs.longmemeval import _index, _instance
    from memctl.harness.runner import Experiment
    from memctl.rl.expert import make_expert

    agent = LLMAgent({"reasoning": False, "labels": "compact", "model": {"backend": "stub"}})
    seen: dict = {}

    class Capture(RLController):
        def decide(self, memory, task):
            actions = super().decide(memory, task)
            if task.observation.requires_response and self.recorded:
                decision = self.recorded[-1]
                items = [memory.get(i) for i in decision.item_ids]
                seen.update(decision=decision, items=items, task=task)
                for item in items:  # the reader prompt we build equals the reader's own, item by item
                    view = SimpleNamespace(active=[item])
                    assert agent.build_prompt(view, task.observation, task) == reader_prompt(
                        task.goal, [compact_line(item)], task.observation.content)
            return actions

    config = capture_config(index)
    experiment = Experiment(config)
    controller = Capture(experiment.config["controller"], 0)
    experiment.controller = controller
    controller.record = True
    controller.expert = make_expert(experiment.hindsight(0), kind="regret", retrieval_risk=0.36)
    experiment.run_episode(seed=0, detail=False)
    meta = _index(PATH)[index]
    instance = _instance(PATH, index)
    row = {"index": index, "question_id": meta["question_id"], "type": meta["question_type"],
           "unanswerable": meta["question_id"].endswith("_abs"), "gold": str(instance["answer"])}
    if not seen:  # no archive at the question (cannot happen on the _s file; kept explicit)
        return {**row, "candidates": []}
    decision, items, task = seen["decision"], seen["items"], seen["task"]
    assert decision.n_active == 0
    row.update(
        goal=task.goal, question=task.observation.content,
        global_features=[float(x) for x in decision.global_features],
        candidates=[{"id": item.id, "created_at": item.created_at, "line": compact_line(item),
                     "tokens": item.token_count, "evidence": int(flag),
                     "features": [float(x) for x in decision.items[j]]}
                    for j, (item, flag) in enumerate(zip(items, decision.expert_retrieved))],
    )
    return row


def build_rows(out: Path, workers: int) -> None:
    from multiprocessing import Pool

    from memctl.envs.longmemeval import _index

    n = len(_index(PATH))
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with Pool(workers) as pool:
        rows = pool.map(capture_one, range(n), chunksize=4)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, out)
    sizes = [len(r["candidates"]) for r in rows]
    print(json.dumps({"questions": n, "answerable": sum(not r["unanswerable"] for r in rows),
                      "candidates_min": min(sizes), "candidates_max": max(sizes),
                      "seconds": round(time.time() - started, 1)}))


def read_rows(path: Path, keep=None) -> list[dict]:
    """Rows in file order; `keep(index)` filters BEFORE a row is parsed, so excluded questions are never read."""
    rows = []
    with gzip.open(path, "rt") as handle:
        for number, line in enumerate(handle):
            if keep is None or keep(number):
                row = json.loads(line)
                assert row["index"] == number
                rows.append(row)
    return rows


# ---- the fold guard -----------------------------------------------------------------------------------------------

def inner_split(index: list[dict], fold: int, part: str) -> list[int]:
    """The inner 6/2 split of fold `fold`'s training part: training questions are dealt stratum by stratum (question
    type, abstention apart; file order within), places 0-5 of every 8 to "inner_train", 6-7 to "inner_val"."""
    from memctl.splits import fold_indices

    if part not in ("inner_train", "inner_val"):
        raise ValueError(part)
    train = fold_indices(index, K, fold, "train")
    strata: dict = {}
    for i in train:
        strata.setdefault((index[i]["question_type"], str(index[i]["question_id"]).endswith("_abs")), []).append(i)
    dealt = [i for members in strata.values() for i in members]
    return sorted(i for place, i in enumerate(dealt) if (place % 8 < 6) == (part == "inner_train"))


def allowed_questions(fold: int, part: str) -> list[int]:
    """Question indices a fold-`fold` head may train or tune on. Never a test question of that fold."""
    from memctl.envs.longmemeval import _index
    from memctl.splits import fold_indices

    index = _index(PATH)
    test = set(fold_indices(index, K, fold, "test"))
    chosen = fold_indices(index, K, fold, "train") if part == "train" else inner_split(index, fold, part)
    if test & set(chosen):
        raise RuntimeError(f"fold guard: {len(test & set(chosen))} test questions of fold {fold} in part {part}")
    return chosen


def training_rows(rows_path: Path, utility_path: Path | None, fold: int, part: str) -> list[dict]:
    """The answerable questions of `part` of fold `fold`, with their utility attached. Rows of any other question,
    in particular fold `fold`'s test questions, are skipped unread in both files."""
    from memctl.envs.longmemeval import _index
    from memctl.splits import fold_indices

    chosen = set(allowed_questions(fold, part))
    rows = [r for r in read_rows(rows_path, chosen.__contains__) if not r["unanswerable"] and r["candidates"]]
    utility = {}
    if utility_path is not None:
        with gzip.open(utility_path, "rt") as handle:
            for number, line in enumerate(handle):
                if number in chosen:
                    record = json.loads(line)
                    assert record["index"] == number
                    utility[number] = record
    test_ids = {_index(PATH)[i]["question_id"] for i in fold_indices(_index(PATH), K, fold, "test")}
    for row in rows:
        if row["question_id"] in test_ids:  # belt and braces: refuse, never filter silently
            raise RuntimeError(f"fold guard: test question {row['question_id']} of fold {fold} reached training")
        if utility_path is not None:
            row["utility"] = utility[row["index"]]["utility"]
    return rows


# ---- scoring (pod) ------------------------------------------------------------------------------------------------

class Scorer:
    """log P(gold | prompt) from a vLLM 0.8.5 OpenAI-compatible server, cached on disk per (model, text).

    Request: POST {base_url}/completions {"prompt": chat_text(...), "max_tokens": 1, "temperature": 0, "echo": true,
    "logprobs": 1}. In 0.8.5, echo with logprobs sets prompt_logprobs and returns the prompt's tokens and their
    log-probabilities (the first is null) before the one generated token; usage.prompt_tokens marks the end of the
    prompt. The gold span is the last n prompt tokens, n = the server tokenizer's count for the gold alone
    (POST {root}/tokenize, add_special_tokens false). That holds because the gold follows "assistant\\n", and
    Qwen2's pre-tokenizer never merges a newline into the next word. Each record keeps a check that the span's
    decoded tokens spell the gold.
    `backend="stub"`: no server; a deterministic fake from word overlap, for the free rehearsal."""

    RETRIES = 5

    def __init__(self, base_url: str | None, cache_dir: str, backend: str = "openai", timeout_s: float = 300.0):
        self.base_url = (base_url or "").rstrip("/")
        self.root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.cache = Path(cache_dir)
        self.backend = backend
        self.timeout_s = timeout_s
        self.calls = self.hits = 0
        self.lock = threading.Lock()
        if backend not in ("openai", "stub"):
            raise ValueError(backend)

    def _post(self, url: str, body: dict) -> dict:
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "memctl"})
        for attempt in range(self.RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504, 520, 522, 524) or attempt == self.RETRIES:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                if attempt == self.RETRIES:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def _cached(self, kind: str, text: str, compute) -> dict:
        key = hashlib.sha256(json.dumps([kind, MODEL, text]).encode()).hexdigest()
        path = self.cache / key[:2] / f"{key}.json"
        try:
            record = json.loads(path.read_text())
            with self.lock:
                self.hits += 1
            return record
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        record = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        partial.write_text(json.dumps(record))
        os.replace(partial, path)
        with self.lock:
            self.calls += 1
        return record

    def gold_tokens(self, gold: str) -> int:
        def compute():
            if self.backend == "stub":
                return {"count": len(gold.split()) + 1}
            reply = self._post(f"{self.root}/tokenize", {"model": MODEL, "prompt": gold, "add_special_tokens": False})
            return {"count": int(reply["count"])}
        return int(self._cached("tokenize", gold, compute)["count"])

    def logprob(self, user: str, gold: str) -> dict:
        """{"logprob": sum over the gold tokens, "n": gold tokens, "prompt_tokens", "check": span spells the gold}."""
        n = self.gold_tokens(gold)
        text = chat_text(user, gold)

        def compute():
            if self.backend == "stub":
                return stub_logprob(user, gold, n)
            reply = self._post(f"{self.base_url}/completions", {
                "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0.0, "echo": True, "logprobs": 1})
            choice = reply["choices"][0]["logprobs"]
            p = int(reply["usage"]["prompt_tokens"])
            span = choice["token_logprobs"][p - n:p]
            if len(span) != n or any(v is None for v in span):
                raise RuntimeError(f"bad logprob span: n={n}, prompt_tokens={p}")
            return {"logprob": float(sum(span)), "n": n, "prompt_tokens": p,
                    "check": "".join(choice["tokens"][p - n:p]) == gold}
        return self._cached("logprob", text, compute)


def stub_logprob(user: str, gold: str, n: int) -> dict:
    """A fake reader: the more gold words the memory holds, the likelier the gold; plus fixed noise."""
    memory = user.split("## Memory", 1)[1].split("## Current input", 1)[0].lower()
    words = [w for w in gold.lower().split() if len(w) > 2] or gold.lower().split() or [""]
    overlap = sum(w in memory for w in words) / len(words)
    noise = int(hashlib.sha256((user + gold).encode()).hexdigest()[:8], 16) / 16**8 - 0.5
    return {"logprob": -3.0 * n + 2.5 * n * overlap + 0.3 * noise, "n": n,
            "prompt_tokens": int(1.3 * len(user.split())), "check": True}


def signal(rows_path: Path, out: Path, scorer: Scorer, workers: int) -> None:
    """Pod step 1: baseline and per-candidate log P(gold) for every answerable question; writes the utility rows
    (one line per question in file order; unanswerable questions get an empty line record)."""
    rows = read_rows(rows_path)
    jobs = []
    for row in rows:
        if row["unanswerable"]:
            continue
        jobs.append((row["index"], -1, reader_prompt(row["goal"], [], row["question"]), row["gold"]))
        for j, cand in enumerate(row["candidates"]):
            jobs.append((row["index"], j, reader_prompt(row["goal"], [cand["line"]], row["question"]), row["gold"]))
    print(json.dumps({"calls_planned": len(jobs), "answerable": sum(not r["unanswerable"] for r in rows)}), flush=True)
    started = time.time()
    done = [0]

    def run(job):
        result = scorer.logprob(job[2], job[3])
        with scorer.lock:
            done[0] += 1
            if done[0] % 1000 == 0:
                print(f"{done[0]}/{len(jobs)} in {time.time() - started:.0f}s", flush=True)
        return result

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run, jobs))
    by_question: dict[int, dict] = {}
    for (index, j, _, _), result in zip(jobs, results):
        by_question.setdefault(index, {})[j] = result
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    checks = prompt_tokens = 0
    with gzip.open(tmp, "wt") as handle:
        for row in rows:
            got = by_question.get(row["index"])
            if got is None:
                handle.write(json.dumps({"index": row["index"], "question_id": row["question_id"]}) + "\n")
                continue
            base = got[-1]["logprob"]
            logp = [got[j]["logprob"] for j in range(len(row["candidates"]))]
            checks += sum(not r["check"] for r in got.values())
            prompt_tokens += sum(r["prompt_tokens"] for r in got.values())
            handle.write(json.dumps({"index": row["index"], "question_id": row["question_id"], "baseline": base,
                                     "n_gold": got[-1]["n"], "logprob": logp,
                                     "utility": [v - base for v in logp]}) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"calls": len(jobs), "new_calls": scorer.calls, "cache_hits": scorer.hits,
                      "span_check_failures": checks, "prompt_tokens": prompt_tokens,
                      "seconds": round(time.time() - started, 1)}), flush=True)


# ---- heads (CPU) --------------------------------------------------------------------------------------------------

def new_policy():
    from memctl.features import GLOBAL_DIM, Featurizer
    from memctl.rl.policy import ItemPolicy

    return ItemPolicy(Featurizer(False, 0, version=1).item_dim, GLOBAL_DIM, 64, "deepsets")


def retrieve_logits(policy, row: dict) -> torch.Tensor:
    from memctl.rl.policy import RETRIEVE_COLUMN

    if "_tensors" not in row:  # built once per row
        row["_tensors"] = (torch.tensor([c["features"] for c in row["candidates"]], dtype=torch.float32),
                           torch.tensor(row["global_features"], dtype=torch.float32))
    logits, _ = policy(*row["_tensors"])
    return logits[:, RETRIEVE_COLUMN]


def question_loss(logits: torch.Tensor, row: dict, signal_name: str, w: float) -> torch.Tensor | None:
    """evidence: the §19 listsum loss on the evidence labels (questions with no evidence in the 32 are skipped);
    utility: cross-entropy against softmax(utility / TAU); blend: (1 - w) evidence + w utility."""
    terms = []
    gold = torch.tensor([c["evidence"] for c in row["candidates"]], dtype=torch.bool)
    if signal_name in ("evidence", "blend") and gold.any():
        listsum = (torch.logsumexp(logits, 0) - logits[gold]).sum()
        terms.append(listsum if signal_name == "evidence" else (1 - w) * listsum)
    if signal_name in ("utility", "blend"):
        target = torch.softmax(torch.tensor(row["utility"], dtype=torch.float32) / TAU, 0)
        ce = torch.logsumexp(logits, 0) - (target * logits).sum()
        terms.append(ce if signal_name == "utility" else w * ce)
    return sum(terms) if terms else None


def train_head(rows: list[dict], signal_name: str, w: float = 0.0, seed: int = SEED, epochs: int = EPOCHS):
    if signal_name not in ("evidence", "utility", "blend"):
        raise ValueError(signal_name)
    torch.set_num_threads(1)  # a 27,526-parameter network on 32 rows: threads only add overhead
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = new_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    last = 0.0
    for _ in range(epochs):
        order = rows[:]
        rng.shuffle(order)
        total = count = 0
        for start in range(0, len(order), BATCH):
            losses = [loss for row in order[start:start + BATCH]
                      if (loss := question_loss(retrieve_logits(policy, row), row, signal_name, w)) is not None]
            if not losses:
                continue
            batch = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += batch.item() * len(losses)
            count += len(losses)
        last = total / max(1, count)
    return policy, last


def save_head(policy, path: Path, info: dict) -> None:
    """RLController.save's checkpoint format, so `controller: {name: rl, checkpoint: ...}` loads it unchanged."""
    from memctl.rl.policy import N_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "policy": policy.config, "feature_version": 1,
                "use_embeddings": False, "embedding_dim": 0, "columns": N_COLUMNS}, path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))


def top_k(policy, row: dict, k: int = SHOW) -> list[int]:
    with torch.no_grad():
        logits = retrieve_logits(policy, row)
    return torch.topk(logits, min(k, len(logits))).indices.tolist()


def shown_prompt(row: dict, picks: list[int]) -> str:
    chosen = sorted((row["candidates"][j] for j in picks), key=lambda c: c["created_at"])
    return reader_prompt(row["goal"], [c["line"] for c in chosen], row["question"])


def all_found(row: dict, picks: list[int]) -> float:
    """Descriptive, training side only: all evidence candidates among the picks (1 when there are none)."""
    gold = {j for j, c in enumerate(row["candidates"]) if c["evidence"]}
    return float(gold <= set(picks))


def tune(rows_path: Path, utility_path: Path, scorer: Scorer, workers: int, fold: int) -> dict:
    """One fold (the five run as parallel processes): a blend head per grid weight on inner_train; mean reader
    log P(gold | question + its top 8) on inner_val; the best weight (ties: closest to 0.5, then the smaller).
    Training questions only. Written to tune_f{fold}.json; a fold already tuned is not redone."""
    out = OUT / f"tune_f{fold}.json"
    if out.exists():
        return json.loads(out.read_text())
    inner_train = training_rows(rows_path, utility_path, fold, "inner_train")
    inner_val = training_rows(rows_path, utility_path, fold, "inner_val")
    record = {"inner_train": len(inner_train), "inner_val": len(inner_val), "grid": {}}
    for w in GRID:
        started = time.time()
        policy, loss = train_head(inner_train, "blend", w)
        picks = [top_k(policy, row) for row in inner_val]
        with ThreadPoolExecutor(workers) as pool:
            scored = list(pool.map(lambda rp: scorer.logprob(shown_prompt(rp[0], rp[1]), rp[0]["gold"]),
                                   zip(inner_val, picks)))
        record["grid"][str(w)] = {
            "mean_logprob": sum(s["logprob"] for s in scored) / len(scored),
            "all_found@8": sum(all_found(r, p) for r, p in zip(inner_val, picks)) / len(inner_val),
            "train_loss": round(loss, 5), "seconds": round(time.time() - started, 1)}
    best = max(GRID, key=lambda w: (round(record["grid"][str(w)]["mean_logprob"], 9), -abs(w - 0.5), -w))
    record["chosen"] = best
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({"fold": fold, "chosen": best,
                      **{w: round(v["mean_logprob"], 4) for w, v in record["grid"].items()}}), flush=True)
    return record


def train_final(rows_path: Path, utility_path: Path, fold: int) -> None:
    tuned = json.loads((OUT / f"tune_f{fold}.json").read_text())
    rows = training_rows(rows_path, utility_path, fold, "train")
    w = float(tuned["chosen"])
    for name, signal_name, weight in (("utility", "utility", 1.0), ("blend", "blend", w),
                                      ("evidence", "evidence", 0.0)):
        started = time.time()
        policy, loss = train_head(rows, signal_name, weight)
        info = {"fold": fold, "signal": signal_name, "w": weight, "questions": len(rows), "epochs": EPOCHS,
                "lr": LR, "batch": BATCH, "tau": TAU, "seed": SEED, "final_loss": round(loss, 5),
                "parameters": sum(p.numel() for p in policy.parameters()),
                "seconds": round(time.time() - started, 1)}
        save_head(policy, HEADS / f"f{fold}" / f"{name}.pt", info)
        print(json.dumps(info), flush=True)


def check(rows_path: Path, fold: int, n: int) -> None:
    """The trained heads inside the real controller show the same 8 as `top_k` on the rows (n training questions)."""
    from memctl.controllers.rl import RLController
    from memctl.harness.runner import Experiment

    questions = allowed_questions(fold, "train")[:n]
    rows = {r["index"]: r for r in read_rows(rows_path, set(questions).__contains__)}
    for name in ("utility", "blend", "evidence"):
        path = HEADS / f"f{fold}" / f"{name}.pt"
        policy = new_policy()
        policy.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        for index in questions:
            config = capture_config(index)
            config["controller"]["checkpoint"] = str(path)
            experiment = Experiment(config)
            shown: list = []
            original = RLController.decide

            def decide(self, memory, task, original=original):
                actions = original(self, memory, task)
                for action in actions:
                    if action.parameters.get("method", "").startswith("rl+"):
                        shown.extend(action.target_ids)
                return actions

            RLController.decide = decide
            try:
                experiment.run_episode(seed=0, detail=False)
            finally:
                RLController.decide = original
            row = rows[index]
            expected = {row["candidates"][j]["id"] for j in top_k(policy, row)}
            if set(shown) != expected:
                raise AssertionError(f"{name} head, question {index}: controller and rows disagree")
    print(f"check: fold {fold}, {n} training questions x 3 heads: controller picks equal row picks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("rows", "signal", "tune", "train", "check"))
    parser.add_argument("--base-url")
    parser.add_argument("--backend", default="openai", choices=("openai", "stub"))
    parser.add_argument("--cache", default=os.environ.get("EXP29_CACHE", "cache/utility_exp29"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    rows_path, utility_path = OUT / "rows.jsonl.gz", OUT / "utility.jsonl.gz"
    if args.stage == "rows":
        build_rows(rows_path, args.workers)
        return
    if args.stage in ("signal", "tune") and args.backend == "openai" and not args.base_url:
        parser.error("--base-url is needed")
    scorer = Scorer(args.base_url, args.cache, args.backend)
    if args.stage == "signal":
        signal(rows_path, utility_path, scorer, args.workers)
    elif args.stage == "tune":
        tune(rows_path, utility_path, scorer, args.workers, args.fold)
        print(json.dumps({"new_calls": scorer.calls, "cache_hits": scorer.hits}))
    elif args.stage == "train":
        train_final(rows_path, utility_path, args.fold)
    else:
        check(rows_path, args.fold, args.n)


if __name__ == "__main__":
    main()
