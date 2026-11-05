"""§37 elenchus generator: E1 (grounds), E2 blind / in dialogue (derivation), E3 (confrontation), V-num, the 7B judge.

    python scripts/exp37_elenchus.py --reader qwen3b --folds 0,1,2,3,4 --backend vllm --out runs/exp37 --smoke
    python scripts/exp37_elenchus.py --reader qwen7b ... --judge          # the 7B's items, then the judge on the 3B's d
    python scripts/exp37_elenchus.py ... --backend stub --out runs/exp37_stub     # CPU, fake replies, end to end

Pre-registration: research/proposals/2026-10-16-exp37-elenchus.md at 714da06 (rev 2). Texts, parsers, normaliser and
scores: memctl/analysis/elenchus.py (its docstring lists the readings taken). The laptop development run goes through
scripts/exp37_dev.py (folds 0-1 only, transformers fp32 CPU, calibration and timing first).

Items (both readers use the 3B's candidates): V (intact prompt, §35 arm-B candidate), CutB (evidence-cut prompt, same
candidate), CutF (cut prompt, the 3B's §35 Fcut candidate), Pnum / Pname (intact prompt, a planted wrong value of the
gold's kind: every candidate row whose gold is a number / a name; the report keeps the rows judged correct, so the
generator never reads a label), Vdlg (E2 asked in dialogue after V's E1, then its own E3), Vnum (the §36b frame (a)
one-shot on the numbered prompt, one token). Turns per item: E1 (24 tokens), E2 (blind, 32), E3 (one token, top-20);
Vdlg: E2d, E3d. The 7B's own-candidate pass is cut (§4).

Engines: `vllm` (offline vllm.LLM 0.8.5, bf16, max_model_len 8192, prefix caching, gpu_memory_utilization 0.70,
max_num_batched_tokens 2048, greedy, seed 0), `hf` (transformers greedy with a KV cache, fp32 on CPU / bf16 on CUDA:
the laptop engine), `stub` (deterministic fake replies). Chat template with the generation prompt, no system message
of our own (the template's default), assistant turn empty. Each row records engine, model, sampling, wording sha256,
input sha256 (the messages), prompt sha256 (the rendered chat text) and intact-prompt sha256; a resume refuses rows
from another engine, input or wording.

Layout: <out>/<reader>/<item>/<turn>.fold<k>.jsonl (one row per question), <out>/<reader>/_smoke.jsonl,
<out>/<reader>/_calibration.json, <out>/qwen3b/judge/fold<k>.jsonl (verdicts on d and d'), <out>/<reader>/_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import elenchus as el  # noqa: E402
from memctl.analysis import verification as ver  # noqa: E402
from memctl.jacobian_data import load_records  # noqa: E402
from scripts.exp35_generate import (MODELS, READERS, STUB_TOKENIZER, JudgeCache, append_jsonl, judge_questions,  # noqa: E402
                                    read_jsonl, write_jsonl)

ALL_FOLDS = (0, 1, 2, 3, 4)
DEV = (0, 1)
CHUNK = 64
JUDGE_MAX_NEW_TOKENS = sel.JUDGE_MAX_NEW_TOKENS
ITEMS = {"qwen3b": ("V", "CutB", "CutF", "Pnum", "Pname", "Vnum", "Vdlg"), "qwen7b": ("V", "CutB", "Pnum", "Pname")}
# §6 triage read backwards: in-dialogue E2 dropped first, then V-num, then planted names; never: intact, Cut-B, Cut-F,
# planted numbers, the judge.
DEFAULT_PLAN = {"qwen3b": "V,CutB,CutF,Pnum|Pname@R3|Vnum@R2|Vdlg@R1", "qwen7b": "V,CutB,Pnum|Pname@RJ"}
TURNS = {"Vnum": ("Vnum",), "Vdlg": ("E2d", "E3d")}
SMOKE_N, SMOKE_NEED = 5, 4
SMOKE_FAILED, CALIBRATION_FAILED = 5, 6


def turns_of(item: str) -> tuple[str, ...]:
    return TURNS.get(item, ("E1", "E2", "E3"))


# --- engines --------------------------------------------------------------------------------------------------

def _lse(a: float | None, b: float) -> float:
    return b if a is None else max(a, b) + math.log1p(math.exp(-abs(a - b)))


class StubEngine:
    """Deterministic fake replies from a hash of the chat text; tokenises with a cached Qwen2.5 tokenizer if present."""

    def __init__(self, model: str) -> None:
        self.name, self.model = "stub", model
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(STUB_TOKENIZER, local_files_only=True)
        except Exception:  # noqa: BLE001
            self.tokenizer = None

    def chat_text(self, messages: list[dict]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def n_tokens(self, text: str) -> int | None:
        return None if self.tokenizer is None else len(self.tokenizer(text, add_special_tokens=False).input_ids)

    @staticmethod
    def _h(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)

    def generate(self, chats: list[list[dict]], max_tokens: int) -> list[tuple[str, dict]]:
        out = []
        for messages in chats:
            h = self._h(self.model + json.dumps(messages))
            last = messages[-1]["content"]
            if "Which memory lines" in last or "List the numbers" in last:
                text = ("[1] [3]", "NONE", "2, 5", "[2]")[h % 4]
            elif max_tokens == JUDGE_MAX_NEW_TOKENS:
                text = "Yes" if h % 3 == 0 else "No"
            else:
                cand = next((ln[len(el.YOU_ANSWERED):] for m in messages for ln in m["content"].splitlines()
                             if ln.startswith(el.YOU_ANSWERED)), None)
                text = ("not determined", cand or "Paris", "2")[h % 3]
            n_in = self.n_tokens(self.chat_text(messages)) or 1
            out.append((text, {"prefill_s": 1e-4 * n_in, "decode_s": 1e-3 * 3, "n_new": 4, "n_input": n_in}))
        return out

    def top20(self, chats: list[list[dict]]) -> list[tuple[dict, dict]]:
        out = []
        for messages in chats:
            h = self._h(self.model + json.dumps(messages))
            mass = 0.3 if h % 19 == 0 else 0.95
            if self.model in os.environ.get("EXP37_STUB_LOW_MASS_MODELS", "").split(","):  # smoke-failure tests
                mass = 0.1
            u = (h % 1000) / 1000
            lines = messages[-1]["content"].splitlines()
            pair = [ln.split(": ", 1)[1] for ln in lines if ln.startswith(("Answer 1: ", "Answer 2: "))]
            if len(pair) == 2:   # E3: a stub comparer, so the calibration path can be exercised
                u = 0.97 if el.same_answer(pair[0], pair[1]) else 0.03
            probs = {"A": u * mass * 0.8, " A": u * mass * 0.2, "B": 0.1 * mass, "C": (1 - u) * mass * 0.9, "The": 1 - mass}
            n_in = self.n_tokens(self.chat_text(messages)) or 1
            out.append(({k: math.log(v) for k, v in probs.items() if v > 0}, {"prefill_s": 1e-4 * n_in, "n_input": n_in}))
        return out


class HFEngine:
    """transformers, greedy with a KV cache, one input at a time; fp32 on CPU (the laptop: no bf16 units), bf16 on CUDA."""

    def __init__(self, model: str, threads: int = 8) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch, self.model = torch, model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.name = f"hf-transformers-{transformers.__version__}-{self.device}-{str(dtype).split('.')[-1]}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.net = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.stop = {self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id}

    def chat_text(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _ids(self, messages):
        return self.tokenizer(self.chat_text(messages), add_special_tokens=False).input_ids

    def generate(self, chats: list[list[dict]], max_tokens: int) -> list[tuple[str, dict]]:
        torch, out = self.torch, []
        for messages in chats:
            ids = self._ids(messages)
            t0 = time.perf_counter()
            with torch.no_grad():
                res = self.net(input_ids=torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1)
                t1 = time.perf_counter()
                cache, nxt, gen = res.past_key_values, int(res.logits[0, -1].argmax()), []
                for _ in range(max_tokens):
                    if nxt in self.stop:
                        break
                    gen.append(nxt)
                    if len(gen) == max_tokens:
                        break
                    step = self.net(input_ids=torch.tensor([[nxt]], device=self.device), past_key_values=cache, use_cache=True)
                    cache, nxt = step.past_key_values, int(step.logits[0, -1].argmax())
            t2 = time.perf_counter()
            out.append((self.tokenizer.decode(gen, skip_special_tokens=True),
                        {"prefill_s": t1 - t0, "decode_s": t2 - t1, "n_new": len(gen), "n_input": len(ids)}))
        return out

    def top20(self, chats: list[list[dict]]) -> list[tuple[dict, dict]]:
        torch, out = self.torch, []
        for messages in chats:
            ids = self._ids(messages)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.net(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1).logits[0, -1]
            logp = torch.log_softmax(logits.float(), -1)
            values, idx = logp.topk(ver.TOP_LOGPROBS)
            top: dict[str, float] = {}
            for v, i in zip(values.tolist(), idx.tolist()):
                tok = self.tokenizer.decode([i])
                top[tok] = _lse(top.get(tok), float(v))
            out.append((top, {"prefill_s": time.perf_counter() - t0, "n_input": len(ids)}))
        return out


class VLLMEngine:
    def __init__(self, model: str) -> None:
        import vllm
        from vllm import LLM

        self.model, self.name = model, f"vllm-{vllm.__version__}"
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, seed=0, gpu_memory_utilization=0.70,
                       enable_prefix_caching=True, max_num_batched_tokens=2048)
        self.tokenizer = self.llm.get_tokenizer()

    def chat_text(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _prompts(self, chats):
        from vllm.inputs import TokensPrompt
        return [TokensPrompt(prompt_token_ids=self.tokenizer(self.chat_text(m), add_special_tokens=False).input_ids)
                for m in chats]

    def generate(self, chats: list[list[dict]], max_tokens: int) -> list[tuple[str, dict]]:
        from vllm import SamplingParams
        t0 = time.perf_counter()
        outs = self.llm.generate(self._prompts(chats), SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                                                                      seed=0), use_tqdm=False)
        per = (time.perf_counter() - t0) / max(len(chats), 1)
        return [(o.outputs[0].text, {"batch_s_per_call": per, "n_new": len(o.outputs[0].token_ids)}) for o in outs]

    def top20(self, chats: list[list[dict]]) -> list[tuple[dict, dict]]:
        from vllm import SamplingParams
        t0 = time.perf_counter()
        outs = self.llm.generate(self._prompts(chats), SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                                                                      logprobs=ver.TOP_LOGPROBS, seed=0), use_tqdm=False)
        per = (time.perf_counter() - t0) / max(len(chats), 1)
        res = []
        for o in outs:
            top: dict[str, float] = {}
            for tid, lp in o.outputs[0].logprobs[0].items():
                tok = lp.decoded_token if lp.decoded_token is not None else self.tokenizer.decode([tid])
                top[tok] = _lse(top.get(tok), float(lp.logprob))
            res.append((top, {"batch_s_per_call": per}))
        return res


def build_engine(backend: str, model: str, threads: int = 8):
    if backend == "stub":
        return StubEngine(model)
    if backend == "hf":
        return HFEngine(model, threads)
    if backend == "vllm":
        return VLLMEngine(model)
    raise ValueError(f"unknown backend {backend!r}")


# --- tasks ----------------------------------------------------------------------------------------------------

@dataclass
class Task:
    item: str
    question_id: str
    fold: int
    prompt: str                 # the unnumbered view (intact or cut) of the cached §23 prompt
    memory_lines: tuple[str, ...]
    evidence_lines: tuple[str, ...]
    candidate: str
    candidate_source: str
    intact_prompt_sha256: str
    extra: dict = field(default_factory=dict)


class Sources:
    """§23 records (both readers, for the population; all folds so the name pool never depends on --folds) and the
    §35 candidates for the requested folds."""

    def __init__(self, folds, runs: str = "runs", exp35: Path = Path("runs/exp35")) -> None:
        records = {r: [x for f in ALL_FOLDS for x in load_records(r, f, runs_dir=runs)[0]] for r in READERS}
        self.qids, _ = sel.population(records)
        self.rec = {x.question_id: x for x in records["qwen3b"]}
        self.folds = tuple(folds)
        self.b = {r["question_id"]: r for f in self.folds for r in read_jsonl(exp35 / "qwen3b" / "B" / f"fold{f}.jsonl")}
        self.fcut = {r["question_id"]: r for f in self.folds for r in read_jsonl(exp35 / "qwen3b" / "Fcut" / f"fold{f}.jsonl")}
        self.name_pool = sorted({self.rec[q].gold.strip() for q in self.qids if ver.gold_kind(self.rec[q].gold) == "name"})

    def fold_qids(self, fold: int) -> list[str]:
        return [q for q in self.qids if self.rec[q].fold == fold]


def build_tasks(item: str, fold: int, src: Sources) -> list[Task]:
    tasks = []
    for q in src.fold_qids(fold):
        rec = src.rec[q]
        intact = ver.sha256(rec.prompt)
        cand = ver.candidate_of_row(src.b.get(q))
        if item in ("V", "Vnum", "Vdlg") and cand:
            tasks.append(Task(item, q, fold, rec.prompt, rec.memory_lines, rec.evidence_lines, cand, "qwen3b/B", intact))
        elif item == "CutB" and cand:
            cut, rem = el.cut_view(rec.prompt, rec.memory_lines, rec.evidence_lines)
            tasks.append(Task(item, q, fold, cut, rem, (), cand, "qwen3b/B", intact))
        elif item == "CutF":
            fc = ver.candidate_of_row(src.fcut.get(q))
            if fc:
                cut, rem = el.cut_view(rec.prompt, rec.memory_lines, rec.evidence_lines)
                tasks.append(Task(item, q, fold, cut, rem, (), fc, "qwen3b/Fcut", intact))
        elif item in ("Pnum", "Pname") and cand:
            kind = ver.gold_kind(rec.gold)
            if kind != {"Pnum": "number", "Pname": "name"}[item]:
                continue
            try:
                value = el.planted(rec.gold, src.name_pool, q, rec.prompt)
            except (ValueError, AssertionError):
                continue
            tasks.append(Task(item, q, fold, rec.prompt, rec.memory_lines, rec.evidence_lines, value, "planted", intact,
                              {"kind": kind}))
    return tasks


# --- rows -----------------------------------------------------------------------------------------------------

def turn_file(out: Path, reader: str, item: str, turn: str, fold: int) -> Path:
    return out / reader / item / f"{turn}.fold{fold}.jsonl"


def row_of(task: Task, reader: str, turn: str, engine, messages: list[dict], wording_sha: str, max_tokens: int,
           stats: dict, payload: dict) -> dict:
    text = engine.chat_text(messages)
    return {"question_id": task.question_id, "fold": task.fold, "reader": reader, "item": task.item, "turn": turn,
            "engine": engine.name, "model": engine.model, "sampling": {**el.SAMPLING, "max_tokens": max_tokens},
            "wording_sha256": wording_sha, "input_sha256": ver.sha256(json.dumps(messages)),
            "prompt_sha256": ver.sha256(text), "intact_prompt_sha256": task.intact_prompt_sha256,
            "n_input_tokens": engine.n_tokens(text), "candidate": task.candidate,
            "candidate_source": task.candidate_source, "n_items": len(task.memory_lines), "extra": task.extra,
            "normaliser_version": el.NORMALISER_VERSION,
            "stats": stats, **payload}


class StaleRowError(RuntimeError):
    pass


def check_existing(row: dict, messages: list[dict], wording_sha: str, engine) -> None:
    if row["input_sha256"] != ver.sha256(json.dumps(messages)) or row["wording_sha256"] != wording_sha:
        raise StaleRowError(f"{row['item']}/{row['turn']}/{row['question_id']}: different input or wording on disk")
    if row["engine"] != engine.name:
        raise StaleRowError(f"{row['item']}/{row['question_id']}: engine {row['engine']!r} on disk, this run {engine.name!r}")


@dataclass
class Opts:
    e1_fallback: bool = False
    e3_fallback: bool = False

    @property
    def sha(self) -> str:
        return el.wording_sha256(self.e1_fallback, self.e3_fallback)


def _done(path: Path) -> dict[str, dict]:
    return {r["question_id"]: r for r in read_jsonl(path)}


def run_turn(turn: str, reader: str, tasks: list[Task], engine, out: Path, opts: Opts, log, deadline: float = 0.0,
             dev_log: bool = True) -> int:
    """Run one turn for the tasks of one item and fold that need it; returns the number of new rows."""
    if not tasks:
        return 0
    fold, item = tasks[0].fold, tasks[0].item
    path = turn_file(out, reader, item, turn, fold)
    done = _done(path)
    e1_item = "V" if item == "Vdlg" else item
    prior = {}
    if turn in ("E2", "E3"):
        prior["E1"] = _done(turn_file(out, reader, e1_item, "E1", fold))
    if turn == "E3":
        prior["E2"] = _done(turn_file(out, reader, item, "E2", fold))
    if turn in ("E2d", "E3d"):
        prior["E1"] = _done(turn_file(out, reader, "V", "E1", fold))
    if turn == "E3d":
        prior["E2d"] = _done(turn_file(out, reader, item, "E2d", fold))
    todo: list[tuple[Task, list[dict]]] = []
    for t in tasks:
        e1 = el.e1_messages(t.prompt, t.memory_lines, t.candidate, opts.e1_fallback)
        if turn == "E1":
            messages = e1
        elif turn == "Vnum":
            messages = el.vnum_messages(t.prompt, t.memory_lines, t.candidate)
        else:
            r1 = prior["E1"].get(t.question_id)
            if r1 is None or not r1["g"]:
                continue
            if turn == "E2":
                messages = el.e2_blind_messages(t.prompt, t.memory_lines, r1["G"])
            elif turn == "E2d":
                messages = el.e2_dialogue_messages(e1, r1["reply"], r1["G"])
            else:
                r2 = prior["E2" if turn == "E3" else "E2d"].get(t.question_id)
                if r2 is None or not el.real_answer(r2["d"]):
                    continue
                messages = el.e3_messages(t.prompt, t.question_id, t.candidate, r2["d"], opts.e3_fallback)
        old = done.get(t.question_id)
        if old is not None:
            check_existing(old, messages, opts.sha, engine)
            continue
        todo.append((t, messages))
    new = 0
    for start in range(0, len(todo), CHUNK):
        if deadline and time.time() >= deadline:
            log(f"{reader} {item} {turn} fold{fold}: deadline reached, {len(todo) - start} left")
            break
        chunk = todo[start:start + CHUNK]
        chats = [m for _, m in chunk]
        rows = []
        if turn in ("E3", "E3d", "Vnum"):
            for (t, m), (top, stats) in zip(chunk, engine.top20(chats)):
                payload = {"top_logprobs": sorted(([k, round(v, 4)] for k, v in top.items()), key=lambda x: -x[1]),
                           "top1": ver.decode_token(max(top, key=top.get)), "scorer_version": ver.SCORER_VERSION,
                           **el.e3_score(top)}
                if turn != "Vnum":
                    payload["order_flip"] = el.order_flip(t.question_id)
                rows.append(row_of(t, reader, turn, engine, m, opts.sha, 1, stats, payload))
        else:
            for (t, m), (reply, stats) in zip(chunk, engine.generate(chats, el.MAX_TOKENS[turn])):
                if turn == "E1":
                    payload = {"reply": reply, **el.parse_e1(reply, len(t.memory_lines)),
                               "evidence_hit": None}
                    payload["evidence_hit"] = (el.evidence_hit(payload["G"], t.memory_lines, t.evidence_lines)
                                               if t.evidence_lines or item == "CutB" else None)
                else:
                    d = el.first_line(reply)
                    payload = {"reply": reply, "d": d, "real_answer": el.real_answer(d),
                               "m": el.same_answer(d, t.candidate)}
                rows.append(row_of(t, reader, turn, engine, m, opts.sha, el.MAX_TOKENS[turn], stats, payload))
        append_jsonl(path, rows)
        new += len(rows)
        if dev_log and fold in DEV:
            for r in rows:
                shown = r.get("reply", r.get("top1"))
                log(f"  [{reader} {item} {turn} f{fold}] {r['question_id']} {shown!r:.60}")
        log(f"{reader} {item} {turn} fold{fold}: {start + len(chunk)}/{len(todo)} new rows")
    return new


def run_item(item: str, reader: str, folds, src: Sources, engine, out: Path, opts: Opts, log, deadline: float = 0.0) -> dict:
    counts = {}
    for fold in folds:
        tasks = build_tasks(item, fold, src)
        for turn in turns_of(item):
            counts[f"{turn}.fold{fold}"] = run_turn(turn, reader, tasks, engine, out, opts, log, deadline)
    return counts


# --- smoke, calibration, judge ----------------------------------------------------------------------------------

def smoke(reader: str, src: Sources, engine, out: Path, opts: Opts, log) -> bool:
    """D-22: 5 development V rows through E1-E3 (E3 on (candidate, d), or (candidate, candidate) when d is not a real
    answer); pass iff >= 4 of 5 parse at E1 and >= 4 of 5 have E3 label mass >= 0.5."""
    tasks = [t for f in DEV for t in build_tasks("V", f, src)][:SMOKE_N]
    rows = []
    e1s = engine.generate([el.e1_messages(t.prompt, t.memory_lines, t.candidate, opts.e1_fallback) for t in tasks],
                          el.MAX_TOKENS["E1"])
    for t, (reply, _) in zip(tasks, e1s):
        p = el.parse_e1(reply, len(t.memory_lines))
        d = None
        if p["g"]:
            d = el.first_line(engine.generate([el.e2_blind_messages(t.prompt, t.memory_lines, p["G"])], el.MAX_TOKENS["E2"])[0][0])
        y = d if el.real_answer(d) else t.candidate
        top = engine.top20([el.e3_messages(t.prompt, t.question_id, t.candidate, y, opts.e3_fallback)])[0][0]
        rows.append({"question_id": t.question_id, "e1_reply": reply, "parses": p["parses"], "G": p["G"], "d": d,
                     "e3_pair_identical": y == t.candidate, **el.e3_score(top), "engine": engine.name,
                     "wording_sha256": opts.sha})
    write_jsonl(out / reader / "_smoke.jsonl", rows)
    parses = sum(r["parses"] for r in rows)
    mass = sum(r["label_mass"] >= 0.5 for r in rows)
    ok = len(rows) == SMOKE_N and parses >= SMOKE_NEED and mass >= SMOKE_NEED
    log(f"smoke {reader}: E1 parses {parses}/{len(rows)}, E3 mass >= 0.5 {mass}/{len(rows)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def calibrate(reader: str, src: Sources, engine, out: Path, opts: Opts, log) -> dict:
    """§2's E3 calibration: 20 development rows; identical pair -> A, another question's candidate -> C; >= 18/20 each."""
    rows = [(t.question_id, t.prompt, t.candidate) for f in DEV for t in build_tasks("V", f, src)]
    pairs = el.calibration_pairs(rows)
    tops = engine.top20([el.e3_pair_messages(p["prompt"], p["x"], p["y"], opts.e3_fallback) for p in pairs])
    results = [{k: v for k, v in p.items() if k != "prompt"} | el.e3_score(top) | {"seconds": st.get("prefill_s")}
               for p, (top, st) in zip(pairs, tops)]
    verdict = el.calibration_pass(results)
    record = {"e3_fallback": opts.e3_fallback, "wording_sha256": opts.sha, "engine": engine.name, "model": engine.model,
              "verdict": verdict, "rows": results}
    path = out / reader / ("_calibration_fallback.json" if opts.e3_fallback else "_calibration.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    log(f"E3 calibration ({'fallback' if opts.e3_fallback else 'primary'} wording): identical -> A "
        f"{verdict['identical']['right']}/{verdict['identical']['n']}, different -> C {verdict['different']['right']}/"
        f"{verdict['different']['n']}: {'PASS' if verdict['pass'] else 'FAIL'}")
    return verdict


def judge(src: Sources, engine, out: Path, folds, log) -> dict:
    """The §35 judge (memctl.judge LongMemEval prompts, Qwen2.5-7B, 4 tokens) on the 3B's d (V/E2) and d' (Vdlg/E2d),
    cached by text under cache/judge_exp37 (stub: cache/stub_judge_exp37)."""
    cache_dir = Path("cache/stub_judge_exp37" if engine.name == "stub" else "cache/judge_exp37")
    cache = JudgeCache(cache_dir, sel.judge_settings(engine.name), None, None)
    questions = judge_questions(src.qids)
    counts = {}
    for fold in folds:
        items = []
        for item, turn in (("V", "E2"), ("Vdlg", "E2d")):
            for r in read_jsonl(turn_file(out, "qwen3b", item, turn, fold)):
                if not el.real_answer(r["d"]):
                    continue
                rec = src.rec[r["question_id"]]
                items.append({"question_id": r["question_id"], "fold": fold, "item": f"{item}/{turn}", "answer": r["d"],
                              "judge_prompt": sel.judge_prompt(questions[r["question_id"]], rec.gold, r["d"], rec.question_type)})
        todo = list(dict.fromkeys(i["judge_prompt"] for i in items if cache.get(i["judge_prompt"], False)[0] is None))
        for start in range(0, len(todo), CHUNK * 4):
            chunk = todo[start:start + CHUNK * 4]
            replies = engine.generate([[{"role": "user", "content": p}] for p in chunk], JUDGE_MAX_NEW_TOKENS)
            for p, (reply, _) in zip(chunk, replies):
                cache.put(p, reply.strip())
        rows = []
        for i in items:
            reply, source = cache.get(i["judge_prompt"], False)
            rows.append({k: v for k, v in i.items() if k != "judge_prompt"} | {
                "judge_prompt_sha256": ver.sha256(i["judge_prompt"]), "verdict": sel.verdict_of(reply), "source": source,
                "judge_engine": engine.name, "judge_model": engine.model,
                "sampling": {**sel.SAMPLING, "max_new_tokens": JUDGE_MAX_NEW_TOKENS}})
        write_jsonl(out / "qwen3b" / "judge" / f"fold{fold}.jsonl", rows)
        counts[fold] = len(rows)
        log(f"judge fold{fold}: {len(rows)} verdicts ({len(todo)} new judge prompts)")
    return counts


# --- main -----------------------------------------------------------------------------------------------------

def parse_plan(plan: str, reader: str, reserves: dict[str, int]) -> list[tuple[list[str], int]]:
    groups = []
    for part in plan.split("|"):
        names, _, reserve = part.strip().partition("@")
        items = [x for x in names.split(",") if x]
        bad = [x for x in items if x not in ITEMS[reader]]
        if bad:
            raise ValueError(f"items {bad} not allowed for {reader} (allowed: {ITEMS[reader]})")
        groups.append((items, int(reserves.get(reserve, reserve or 0))))
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", required=True, choices=READERS)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--backend", default="vllm", choices=("vllm", "hf", "stub"))
    parser.add_argument("--out", default="runs/exp37")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--deadline", type=float, default=0.0)
    parser.add_argument("--reserves", default="R1=0,R2=0,R3=0,RJ=0", help="seconds kept free per reserve name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--judge", action="store_true", help="after the items, the 7B judge on the 3B's d and d'")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--e1-fallback", action="store_true")
    parser.add_argument("--e3-fallback", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    reserves = {k: int(v) for k, v in (x.split("=") for x in args.reserves.split(",") if x)}
    plan = parse_plan(args.plan or DEFAULT_PLAN[args.reader], args.reader, reserves)
    out = Path(args.out)
    opts = Opts(args.e1_fallback, args.e3_fallback)
    src = Sources(folds, args.runs, Path(args.exp35))
    engine = build_engine(args.backend, MODELS[args.reader], args.threads)
    manifest = {"reader": args.reader, "model": engine.model, "engine": engine.name, "folds": folds, "plan": plan,
                "wording": el.wording(opts.e1_fallback, opts.e3_fallback), "wording_sha256": opts.sha}
    (out / args.reader).mkdir(parents=True, exist_ok=True)
    (out / args.reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    log = lambda s: print(s, flush=True)  # noqa: E731
    if args.judge_only:
        log(f"judge: {judge(src, engine, out, folds, log)}")
        return 0
    if args.smoke and not smoke(args.reader, src, engine, out, opts, log):
        log(f"SMOKE FAILED for {args.reader} (exit {SMOKE_FAILED}); no item rows written")
        return SMOKE_FAILED
    counts = {}
    for items, reserve in plan:
        if args.deadline and args.deadline - time.time() <= reserve:
            log(f"{args.reader}: skipping {items} (less than {reserve} s left)")
            continue
        for item in items:
            counts[item] = run_item(item, args.reader, folds, src, engine, out, opts, log, args.deadline)
    manifest["counts"] = counts
    if args.judge:
        manifest["judge"] = judge(src, engine, out, folds, log)
    (out / args.reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"{args.reader}: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
