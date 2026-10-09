"""Experiment 28: BM25 recall@32 of evidence turns, raw question against a rewritten query, and the gates.

    python -m memctl.rewrite_gate rule-tune --out runs/exp28_gate/rule.json
    python -m memctl.rewrite_gate rewrite --prompt-file configs/sweeps/exp28/prompts/v1.txt --base-url URL/v1
    python -m memctl.rewrite_gate inspect --version 1        (LoCoMo dev outputs only; LongMemEval is refused)
    python -m memctl.rewrite_gate gate --version 1
    python -m memctl.rewrite_gate final
    python -m memctl.rewrite_gate evaluate --rule runs/exp28_gate/rule.json     (once, at the evaluation)

Cross-fitting (as the §19 head, one checkpoint per fold): every choice for fold f (the rule filter's action and
slack; which prompt version arm (a) uses) is made on fold f's TRAINING part only, and applied to fold f's test
part. The labels of a fold's test part are read only by `evaluate`, which refuses to run before the gate is final.

recall@32 of a question: the share of its evidence requirements met by the 32 candidates (a marked turn; or,
for an answer session with no marked turn, any turn of it), as the harness counts them. Abstention questions
are left out (470 answerable in all; about 376 per training part).

Prompt design (design (i), the reviewer's ruling of 2026-10-09): every LongMemEval question is a test question in
some fold, so no person or agent reads LongMemEval question text or rewrite outputs to write v2-v4. Revisions are
written from a disjoint dev set, every fifth non-adversarial LoCoMo question (307), through `inspect`. LongMemEval
rewrites are read only by `gate` (training-part labels, aggregate output) and `final`.

The gate (pre-declared in EXPERIMENTS.md §28, before any recall was computed): a variant passes on fold f if,
on f's training part, mean recall@32 rises by at least +0.01 over the raw question, or rises by at least +0.02
on the temporal-reasoning questions while not falling overall. An arm goes to the reader only if it passes on
all five folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from memctl.envs.longmemeval import _index, _instance, parse_instance
from memctl.query_rewrite import (REWRITE_MAX_TOKENS, load_prompt, parse_rewrite, pool, rewrite_prompt, rule_range,
                                  split_question, widen)
from memctl.retrieval import LexicalRetriever
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
K = 32
FOLDS = 5
GATE_DIR = Path(os.environ.get("EXP28_GATE_DIR", "runs/exp28_gate"))  # overridden only by the stub rehearsal
REWRITE_CACHE = os.environ.get("EXP28_REWRITE_CACHE", "cache/rewrite_exp28")
MAX_VERSIONS = 4
ACTIONS = ("drop", "demote")
SLACKS = (0.5, 1.0, 2.0, 4.0)
LLM_ACTION, LLM_PAD_DAYS = "demote", 3  # arm (a)'s range handling, fixed before any rewrite


def passes(overall_gain: float, temporal_gain: float) -> bool:
    """The pre-declared gate (§28)."""
    return overall_gain >= 0.01 or (temporal_gain >= 0.02 and overall_gain >= 0.0)


class Question:
    """One LongMemEval question as the controller sees it at the question step (keep_none: every turn archived)."""

    def __init__(self, number: int) -> None:
        episode, fallback = parse_instance(_instance(PATH, number))
        item = episode.questions[0]
        self.number, self.id, self.type = number, item.id, item.category
        self.answerable = not item.unanswerable
        self.content = f"Question: {item.question}"  # the observation's text, as memctl/envs/qa.py writes it
        self.items = [SimpleNamespace(id=t.id, content=t.content, metadata=t.metadata, created_at=n)
                      for n, t in enumerate(episode.turns)]
        self._requirements = [(e,) for e in item.evidence_ids] + [tuple(ids) for ids in fallback]

    def recall(self, candidates) -> tuple[float, bool]:
        """(share of requirements met, all met). Reading this is reading a label."""
        found = {pair[0].id for pair in candidates}
        met = [any(i in found for i in requirement) for requirement in self._requirements]
        return (sum(met) / len(met), all(met)) if met else (1.0, True)


# ---- the prompt-design dev set: LoCoMo, disjoint from LongMemEval (§28, design (i)) -------------------------

LOCOMO_PATH = "data/locomo/locomo10.json"
DEV_EVERY = 5  # every fifth non-adversarial LoCoMo question, in file order (with evidence): 307 questions


def _lme_date(text: str) -> str:
    """A LoCoMo session time ("1:56 pm on 8 May, 2023") in LongMemEval's style ("2023/05/08 (Mon) 13:56")."""
    from datetime import datetime

    moment = datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y")
    return moment.strftime("%Y/%m/%d (%a) %H:%M")


class DevQuestion:
    """One LoCoMo dev question, shaped as a LongMemEval question step: session dates in LongMemEval's format, and
    the question date taken as the conversation's last session date (LoCoMo gives no question date)."""

    def __init__(self, sample: dict, episode, item) -> None:
        dates = [t.metadata["date"] for t in episode.turns]
        self.id, self.type = f"dev:{item.id}", ("temporal-reasoning" if item.category == "temporal" else item.category)
        self.answerable = not item.unanswerable
        self.content = f"Question: (asked on {_lme_date(dates[-1])}) {item.question}"
        self.items = [SimpleNamespace(id=t.id, content=t.content, metadata={**t.metadata, "date": _lme_date(t.metadata["date"])},
                                      created_at=n) for n, t in enumerate(episode.turns)]
        self._requirements = [(e,) for e in item.evidence_ids]

    recall = Question.recall


def dev_questions() -> list[DevQuestion]:
    from memctl.envs.locomo import parse_conversation

    found = []
    for sample in json.loads(Path(LOCOMO_PATH).read_text()):
        episode = parse_conversation(sample, include_adversarial=False)
        found += [DevQuestion(sample, episode, item) for item in episode.questions]
    return [q for q in found if q._requirements][::DEV_EVERY]


def _check_dev_only(ids: list[str]) -> None:
    """inspect reads LoCoMo dev outputs only: a LongMemEval question (a number, or anything not a dev id) is refused."""
    bad = [i for i in ids if not str(i).startswith("dev:")]
    if bad:
        raise PermissionError(f"inspect reads LoCoMo dev questions only; refused: {bad[:5]}")


def _training_numbers(fold: int) -> list[int]:
    return fold_indices(_index(PATH), FOLDS, fold, "train")


def _test_numbers(fold: int) -> list[int]:
    return fold_indices(_index(PATH), FOLDS, fold, "test")


def _summary(rows: list[dict], key: str) -> dict:
    """Mean recall and all-found over all answerable rows and over the temporal-reasoning ones."""
    def mean(values):
        return round(sum(values) / len(values), 4) if values else None
    temporal = [r for r in rows if r["type"] == "temporal-reasoning"]
    return {"n": len(rows), "recall": mean([r[key][0] for r in rows]), "all": mean([r[key][1] for r in rows]),
            "n_temporal": len(temporal), "recall_temporal": mean([r[key][0] for r in temporal]),
            "all_temporal": mean([r[key][1] for r in temporal])}


def _gain(summary: dict, base: dict) -> tuple[float, float]:
    return summary["recall"] - base["recall"], summary["recall_temporal"] - base["recall_temporal"]


# ---- arm (b): the rule filter, free -----------------------------------------------------------------------

def rule_rows(numbers: list[int], configs: list[tuple[str, float]]) -> dict[int, dict]:
    """Per question: recall@32 raw and under each (action, slack). One BM25 ranking per question."""
    retriever, rows = LexicalRetriever(), {}
    for number in numbers:
        q = Question(number)
        if not q.answerable:
            continue
        ranked = retriever.search(q.content, q.items, len(q.items))
        ranker = SimpleNamespace(search=lambda query, items, k, ranked=ranked: ranked[:k])
        asked, text = split_question(q.content)
        row = {"type": q.type, "raw": q.recall(ranked[:K]), "fired": {}}
        for action, slack in configs:
            span = rule_range(text, asked, slack)
            row["fired"][f"{action}:{slack}"] = span is not None
            row[f"{action}:{slack}"] = q.recall(pool(ranker, q.content, q.items, K, span, action))
        rows[number] = row
    return rows


def rule_tune(out: Path) -> dict:
    configs = [(a, s) for a in ACTIONS for s in SLACKS]
    train_union = sorted({n for f in range(FOLDS) for n in _training_numbers(f)})
    rows = rule_rows(train_union, configs)  # each question is in the training part of four folds
    result: dict = {"folds": {}}
    for fold in range(FOLDS):
        train = [rows[n] for n in _training_numbers(fold) if n in rows]
        base = _summary(train, "raw")
        scored = {}
        for action, slack in configs:
            key = f"{action}:{slack}"
            s = _summary(train, key)
            s["fired"] = sum(r["fired"][key] for r in train)
            s["fired_temporal"] = sum(r["fired"][key] for r in train if r["type"] == "temporal-reasoning")
            scored[key] = s
        # The choice: the highest overall training recall; ties go to demote, then the smaller slack.
        best = max(configs, key=lambda c: (scored[f"{c[0]}:{c[1]}"]["recall"], c[0] == "demote", -c[1]))
        key = f"{best[0]}:{best[1]}"
        overall, temporal = _gain(scored[key], base)
        result["folds"][str(fold)] = {"raw": base, "grid": scored, "chosen": {"action": best[0], "slack": best[1]},
                                      "gain": round(overall, 4), "gain_temporal": round(temporal, 4),
                                      "passes": passes(overall, temporal)}
    result["go"] = all(f["passes"] for f in result["folds"].values())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


# ---- arm (a): the LLM rewrite, on the pod -----------------------------------------------------------------

BACKEND = {"name": "openai"}  # "stub" only for the free pipeline rehearsal; it is part of the cache key


def _llm(base_url: str | None, cache_only: bool):
    """The rewriter. The settings match the controller's `query_rewrite.model`, so both share cache keys; a
    replay (cache_only) never calls a server."""
    from memctl.llm import build_llm

    config = {"backend": BACKEND["name"], "name": "qwen2.5-7b-instruct", "cache_dir": REWRITE_CACHE,
              "timeout_s": 300, "cache_only": cache_only, "base_url": base_url or "http://unused.invalid/v1"}
    return build_llm(config)


def _outputs(template: str, numbers: list, llm, workers: int = 16) -> dict:
    """Rewrites for LongMemEval question numbers, or for DevQuestion objects (keyed by their dev id)."""
    def one(entry) -> str:
        content = entry.content if isinstance(entry, DevQuestion) else Question(entry).content
        asked, text = split_question(content)
        return llm.generate(rewrite_prompt(template, asked, text), REWRITE_MAX_TOKENS)
    keys = [e.id if isinstance(e, DevQuestion) else e for e in numbers]
    with ThreadPoolExecutor(workers) as threads:
        return dict(zip(keys, threads.map(one, numbers)))


def _version_file(version: int) -> Path:
    return Path(os.environ.get("EXP28_PROMPT_DIR", "configs/sweeps/exp28/prompts")) / f"v{version}.txt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state() -> dict:
    path = GATE_DIR / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"versions": {}, "selected": {}, "final": None}


def _save_state(state: dict) -> None:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / "state.json").write_text(json.dumps(state, indent=1))


def rewrite(version: int, base_url: str | None) -> None:
    """Rewrites every question with prompt version N (label-free: no evidence is read). Every question is in
    some fold's training part, so the gate needs all 500; the rewrites are cached for the reader to replay."""
    if _state()["final"] is not None:
        sys.exit("the gate is final; no more rewriting")
    template = load_prompt(_version_file(version))
    dev = _outputs(template, dev_questions(), _llm(base_url, False))
    numbers = list(range(len(_index(PATH))))
    outputs = _outputs(template, numbers, _llm(base_url, False))  # never printed or shown before `final`
    print(f"v{version}: {len(dev)} LoCoMo dev and {len(outputs)} LongMemEval questions rewritten into {REWRITE_CACHE}")


def inspect(version: int, ids: list[str] | None = None) -> dict:
    """What a revision may use: LoCoMo dev outputs only (format, date style, and dev recall@32 from LoCoMo's own
    labels). LongMemEval rewrites are never read here; a LongMemEval question is refused (PermissionError)."""
    questions = dev_questions()
    if ids is not None:
        _check_dev_only(ids)
        wanted = set(ids)
        questions = [q for q in questions if q.id in wanted]
    template = load_prompt(_version_file(version))
    outputs = _outputs(template, questions, _llm(None, True))
    if "__CACHE_MISS__" in outputs.values():
        sys.exit(f"dev questions without a v{version} rewrite; run `rewrite` first")
    retriever, rows = LexicalRetriever(), []
    for q in questions:
        asked, _ = split_question(q.content)
        extra, span, ok = parse_rewrite(outputs[q.id])
        query = f"{q.content} {extra}" if extra else q.content
        rows.append({"type": q.type, "raw": q.recall(retriever.search(q.content, q.items, K)),
                     "llm": q.recall(pool(retriever, query, q.items, K, widen(span, LLM_PAD_DAYS, asked), LLM_ACTION)),
                     "ok": ok, "width": (span[1] - span[0]).days if span else None})
    widths = sorted(r["width"] for r in rows if r["width"] is not None)
    summary = {"dev_n": len(rows), "malformed": sum(not r["ok"] for r in rows), "with_range": len(widths),
               "median_width_days": widths[len(widths) // 2] if widths else None,
               "raw": _summary(rows, "raw"), "llm": _summary(rows, "llm")}
    print(json.dumps(summary, indent=1))
    for q in questions[:8]:
        print(f"DEV {q.id}: {q.content!r}\n  -> {outputs[q.id][:300]!r}")
    return summary


def llm_rows(version: int, numbers: list[int]) -> dict[int, dict]:
    template = load_prompt(_version_file(version))
    outputs = _outputs(template, numbers, _llm(None, True))
    retriever, rows = LexicalRetriever(), {}
    for number in numbers:
        q = Question(number)
        if not q.answerable:
            continue
        if outputs[number] == "__CACHE_MISS__":
            sys.exit(f"question {number} has no v{version} rewrite in {REWRITE_CACHE}; run `rewrite` first")
        asked, _ = split_question(q.content)
        extra, span, ok = parse_rewrite(outputs[number])
        query = f"{q.content} {extra}" if extra else q.content
        rows[number] = {"type": q.type, "raw": q.recall(retriever.search(q.content, q.items, K)),
                        "llm": q.recall(pool(retriever, query, q.items, K, widen(span, LLM_PAD_DAYS, asked), LLM_ACTION)),
                        "well_formed": ok, "range": span is not None}
    return rows


def gate(version: int) -> dict:
    """Scores prompt version N on the training part of every fold not yet decided, in order v1, v2, ..."""
    state = _state()
    if state["final"] is not None:
        sys.exit("the gate is final")
    if not 1 <= version <= MAX_VERSIONS:
        sys.exit(f"at most {MAX_VERSIONS} versions")
    if str(version) in state["versions"]:
        sys.exit(f"v{version} is already scored; a version is scored once")
    if version > 1 and str(version - 1) not in state["versions"]:
        sys.exit(f"score v{version - 1} first")
    for v, entry in state["versions"].items():
        if _sha(_version_file(int(v))) != entry["sha256"]:
            sys.exit(f"prompt v{v} changed after it was scored")
    open_folds = [f for f in range(FOLDS) if str(f) not in state["selected"]]
    if not open_folds:
        sys.exit("every fold already has its version")
    numbers = sorted({n for f in open_folds for n in _training_numbers(f)})
    rows = llm_rows(version, numbers)
    entry = {"sha256": _sha(_version_file(version)), "prompt": _version_file(version).read_text(), "folds": {}}
    for fold in open_folds:
        train = [rows[n] for n in _training_numbers(fold) if n in rows]
        base, arm = _summary(train, "raw"), _summary(train, "llm")
        overall, temporal = _gain(arm, base)
        ok = passes(overall, temporal)
        entry["folds"][str(fold)] = {"raw": base, "llm": arm, "gain": round(overall, 4),
                                     "gain_temporal": round(temporal, 4), "passes": ok}
        if ok:
            state["selected"][str(fold)] = version  # the first passing version is frozen for this fold
    state["versions"][str(version)] = entry
    _save_state(state)
    # Aggregate only: recall per fold, overall and temporal. No question text and no rewrite output is shown.
    print(json.dumps({f: {"raw": e["raw"]["recall"], "llm": e["llm"]["recall"], "gain": e["gain"],
                          "raw_temporal": e["raw"]["recall_temporal"], "llm_temporal": e["llm"]["recall_temporal"],
                          "gain_temporal": e["gain_temporal"], "passes": e["passes"]}
                      for f, e in entry["folds"].items()}, indent=1))
    return entry


def final() -> dict:
    """Closes the gate: GO if every fold has a passing version, else arm (a) is dropped. Nothing is carried forward
    as 'best of'. Writes runs/exp28_gate/selected.json, which the reader stage reads."""
    state = _state()
    if state["final"] is not None:
        return state["final"]
    scored = len(state["versions"])
    complete = all(str(f) in state["selected"] for f in range(FOLDS))
    if not complete and scored < MAX_VERSIONS:
        sys.exit(f"{scored} of {MAX_VERSIONS} versions scored and not every fold passes: revise, or score the rest")
    state["final"] = {"go": complete, "selected": state["selected"] if complete else {},
                      "reason": "every fold passes" if complete else
                      f"no passing version for folds {[f for f in range(FOLDS) if str(f) not in state['selected']]}"}
    _save_state(state)
    (GATE_DIR / "selected.json").write_text(json.dumps(state["final"], indent=1))
    return state["final"]


# ---- the evaluation (test parts; once) --------------------------------------------------------------------

def evaluate(rule_file: Path, out: Path) -> dict:
    """Reader-free numbers on the test folds, per arm: recall@32 and all evidence in the 32. Run once."""
    state = _state()
    if state["final"] is None:
        sys.exit("the arm (a) gate is not final; the test folds are not read before it is")
    if out.exists():
        sys.exit(f"{out} exists: the evaluation runs once")
    rule = json.loads(rule_file.read_text())
    result = {}
    for fold in range(FOLDS):
        numbers = _test_numbers(fold)
        chosen = rule["folds"][str(fold)]["chosen"]
        b = rule_rows(numbers, [(chosen["action"], chosen["slack"])])
        rows = [dict(r, rule=r[f"{chosen['action']}:{chosen['slack']}"]) for r in b.values()]
        if state["final"]["go"]:
            version = state["final"]["selected"][str(fold)]
            a = llm_rows(version, numbers)
            rows = [dict(r, llm=a[n]["llm"]) for n, r in zip(b.keys(), rows)]
        result[str(fold)] = {arm: _summary(rows, arm) for arm in ("raw", "rule", "llm") if arm in rows[0]}
    out.write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rule-tune"); p.add_argument("--out", type=Path, default=GATE_DIR / "rule.json")
    p = sub.add_parser("rewrite"); p.add_argument("--version", type=int, required=True)
    p.add_argument("--base-url")
    p = sub.add_parser("inspect"); p.add_argument("--version", type=int, required=True)
    p.add_argument("--ids", nargs="*", help="LoCoMo dev ids (dev:...); anything else is refused")
    p = sub.add_parser("gate"); p.add_argument("--version", type=int, required=True)
    sub.add_parser("final")
    p = sub.add_parser("evaluate"); p.add_argument("--rule", type=Path, default=GATE_DIR / "rule.json")
    p.add_argument("--out", type=Path, default=Path("runs/exp28_eval_recall.json"))
    parser.add_argument("--backend", default="openai", choices=("openai", "stub"))
    args = parser.parse_args(argv)
    BACKEND["name"] = args.backend
    if args.command == "rule-tune":
        result = rule_tune(args.out)
        for f, e in result["folds"].items():
            print(f"fold {f}: chosen {e['chosen']}, raw {e['raw']['recall']} -> gain {e['gain']:+.4f}, "
                  f"temporal raw {e['raw']['recall_temporal']} (n={e['raw']['n_temporal']}) gain "
                  f"{e['gain_temporal']:+.4f}, passes {e['passes']}")
        print("GO" if result["go"] else "NO GO")
    elif args.command == "rewrite":
        rewrite(args.version, args.base_url)
    elif args.command == "inspect":
        inspect(args.version, args.ids)
    elif args.command == "gate":
        gate(args.version)
    elif args.command == "final":
        print(json.dumps(final(), indent=1))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.rule, args.out), indent=1))


if __name__ == "__main__":
    main()
