"""§35 generation and judging: arms N, A, B, F, Fcut, Ncut for one or both readers, then the 7B judge.

    python scripts/exp35_generate.py --reader qwen3b --folds 0,1,2,3,4 --arms N,A,B,F,Fcut,Ncut --backend vllm --out runs/exp35
    python scripts/exp35_generate.py --judge --reader qwen3b,qwen7b --folds 0,1,2,3,4 --backend vllm --out runs/exp35
    python scripts/exp35_generate.py ... --backend stub --out runs/exp35_stub      # CPU, canned outputs, end to end

Pre-registration: research/proposals/2026-10-13-exp35-answer-selection.md at commit 12c9168 (reviewer round 2, conditional ACCEPT);
arm logic in memctl/analysis/answer_selection.py.

Engine. `vllm` (offline `vllm.LLM`, pinned to 0.8.5 by scripts/exp35_pod.sh: the version that served every §23 answer and
verdict) is the pod engine; `hf` (transformers batched greedy generate, left padding, bf16) is the fallback the pod
script switches to, before any generation, if vLLM does not import; it then runs every arm of both readers and the
judge, so one run never mixes engines (a resume with another engine is refused per row, and the report refuses a
run directory holding more than one engine). Both build the raw input by id concatenation, ids(chat text) +
ids(continuation), as §34b's own-mode readout did, and decode greedily (temperature 0, seed 0). Each row records its
engine; a resume refuses rows from another engine or wording. Sampling flags (temperature 0, top_p 1, seed 0, greedy,
max_new_tokens) are stored in every row. `stub` is a deterministic fake for the CPU run; it tokenises with the
locally cached Qwen2.5 tokenizer when one is present (offline, no download), so token counts are real there too.

Log: per-question lines (answer, refusal flag) are printed for folds 0 and 1 only; folds 2-4 print chunk counts.

Layout under --out: <reader>/<arm>/fold<k>.jsonl (one row per question; appended in chunks, so a rerun resumes),
<reader>/<arm>/errors.fold<k>.jsonl (prompts whose replacement or cut failed), <reader>/judge/fold<k>.jsonl (verdict
rows, rewritten each judge pass), <reader>/_manifest.json. One file per fold so a development report never opens
folds 2-4.

Judging (--judge): the §23 judge prompt (official LongMemEval per-type prompts, Qwen2.5-7B, 4 new tokens) on every
generated final answer, on every cached note alone (D2), and on every cached final answer again ("cached", the judge
engine check against the §23 labels). Verdicts are cached by text under cache/judge_exp35 (TrackedLLM keying). For
arm answers and notes, an identical judge prompt already judged by §23 (cache/judge_exp23, read only) reuses that
verdict; the "cached" items never do. The stub backend reads and writes cache/stub_judge_exp35 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.jacobian_data import load_records  # noqa: E402

READERS = ("qwen3b", "qwen7b")
MODELS = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "data/longmemeval/longmemeval_s_cleaned.json"
CHUNK = 128
JUDGE_ITEMS = ("A", "B", "F", "Fcut", "Ncut", "D2", "cached")


# --- engines --------------------------------------------------------------------------------------------------

STUB_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"  # same tokenizer and chat template as the 3B and 7B


class StubEngine:
    """Deterministic canned outputs from a hash of the input; no model. Tokenises with a locally cached Qwen2.5
    tokenizer if one is on disk (never downloads), else token counts are None."""

    def __init__(self, model: str) -> None:
        self.name, self.model = "stub", model
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(STUB_TOKENIZER, local_files_only=True)
        except Exception:  # noqa: BLE001 - no transformers or no cached tokenizer: counts are None
            self.tokenizer = None

    def chat_text(self, prompt: str) -> str:
        return sel.QWEN25_CHAT.format(prompt=prompt)

    def encode(self, text: str) -> list[int] | None:
        return None if self.tokenizer is None else list(self.tokenizer(text, add_special_tokens=False).input_ids)

    @staticmethod
    def _h(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def generate(self, items: list[tuple[str, str]], max_new_tokens: int) -> list[str]:
        outs = []
        for prompt, continuation in items:
            h = self._h(self.chat_text(prompt) + continuation + str(max_new_tokens))
            if max_new_tokens == sel.JUDGE_MAX_NEW_TOKENS:
                outs.append("Yes" if h % 3 == 0 else "No")
                continue
            words = [w for w in prompt.rsplit("Question:", 1)[-1].split() if w.isalpha() and len(w) > 3] or ["thing"]
            word = words[h % len(words)]
            if continuation:
                outs.append(" unknown" if h % 4 == 0 else f" {word}")
            else:
                outs.append(("Stub note: the memory mentions it.\nAnswer: unknown", f"Stub note: {word}.\nAnswer: {word}",
                             word)[h % 3])
        return outs


class VLLMEngine:
    def __init__(self, model: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.85) -> None:
        import vllm
        from vllm import LLM

        self.model = model
        self.name = f"vllm-{vllm.__version__}"
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, seed=0,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
        self.tokenizer = self.llm.get_tokenizer()

    def chat_text(self, prompt: str) -> str:
        from memctl.analysis.jacobian import wrap_chat
        return wrap_chat(self.tokenizer, prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _ids(self, prompt: str, continuation: str) -> list[int]:
        return self.encode(self.chat_text(prompt)) + (self.encode(continuation) if continuation else [])

    def generate(self, items: list[tuple[str, str]], max_new_tokens: int) -> list[str]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=0)
        prompts = [TokensPrompt(prompt_token_ids=self._ids(p, c)) for p, c in items]
        return [out.outputs[0].text for out in self.llm.generate(prompts, params, use_tqdm=False)]


class HFEngine:
    """transformers batched greedy generation (left padding), bf16 on CUDA."""

    def __init__(self, model: str, batch_size: int = 16) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.model, self.batch_size = torch, model, batch_size
        self.name = f"hf-transformers-{transformers.__version__}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.net = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device).eval()
        self.device = device
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def chat_text(self, prompt: str) -> str:
        from memctl.analysis.jacobian import wrap_chat
        return wrap_chat(self.tokenizer, prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _ids(self, prompt: str, continuation: str) -> list[int]:
        return self.encode(self.chat_text(prompt)) + (self.encode(continuation) if continuation else [])

    def generate(self, items: list[tuple[str, str]], max_new_tokens: int) -> list[str]:
        torch = self.torch
        encoded = [self._ids(p, c) for p, c in items]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        outs: list[str] = [""] * len(encoded)
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            width = max(len(encoded[i]) for i in batch)
            ids = torch.tensor([[self.pad] * (width - len(encoded[i])) + encoded[i] for i in batch], device=self.device)
            mask = torch.tensor([[0] * (width - len(encoded[i])) + [1] * len(encoded[i]) for i in batch], device=self.device)
            with torch.no_grad():
                gen = self.net.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                        do_sample=False, pad_token_id=self.pad)
            for row, i in enumerate(batch):
                outs[i] = self.tokenizer.decode(gen[row][width:], skip_special_tokens=True)
        return outs


def build_engine(backend: str, model: str):
    if backend == "stub":
        return StubEngine(model)
    if backend == "vllm":
        return VLLMEngine(model)
    if backend == "hf":
        return HFEngine(model)
    raise ValueError(f"unknown backend {backend!r} (known: stub, vllm, hf)")


# --- files ----------------------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a killed process: dropped, regenerated on resume
                continue
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with open(path, "a") as handle:
        if torn:  # a line cut by a killed process: end it, so read_jsonl drops it alone
            handle.write("\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(partial, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(partial, path)


def arm_file(out: Path, reader: str, arm: str, fold: int) -> Path:
    return out / reader / arm / f"fold{fold}.jsonl"


def judge_file(out: Path, reader: str, fold: int) -> Path:
    return out / reader / "judge" / f"fold{fold}.jsonl"


# --- records --------------------------------------------------------------------------------------------------

def load_population(folds: list[int], runs_dir: str = "runs") -> tuple[dict[str, dict[str, object]], list[str], list[str]]:
    """{reader: {qid: PromptRecord}} over `folds`, the §35 population and the prompt mismatches."""
    records = {r: [x for f in folds for x in load_records(r, f, runs_dir=runs_dir)[0]] for r in READERS}
    qids, mismatched = sel.population(records)
    return {r: {x.question_id: x for x in rows} for r, rows in records.items()}, qids, mismatched


# --- generation -----------------------------------------------------------------------------------------------

DEV_LOG_FOLDS = (0, 1)


def token_counts(engine, item) -> tuple[str, tuple[int | None, int | None, int | None]]:
    """(raw text, (input tokens, teacher-forced tokens, chat + note tokens)) by the engine's own id concatenation."""
    chat = engine.chat_text(item.prompt)
    raw = chat + item.continuation
    chat_ids = engine.encode(chat)
    if chat_ids is None:
        return raw, (None, None, None)
    forced = len(engine.encode(item.continuation)) if item.continuation else 0
    prefix = len(chat_ids) + len(engine.encode(item.note)) if item.note is not None else None
    return raw, (len(chat_ids) + forced, forced, prefix)


def generate_reader(reader: str, folds: list[int], arms: list[str], engine, out: Path, by_q: dict, qids: list[str],
                    log=print) -> dict:
    """Generate every pending row for one reader; returns counts per arm."""
    counts: dict[str, dict] = {}
    for arm in arms:
        counts[arm] = {"rows": 0, "new": 0, "no_marker": 0, "errors": 0}
        for fold in folds:
            inputs, errors = [], []
            for q in qids:
                record = by_q[q]
                if record.fold != fold:
                    continue
                try:
                    item = sel.build_arm_input(record, arm)
                except ValueError as error:
                    errors.append({"question_id": q, "fold": fold, "arm": arm, "error": str(error)})
                    continue
                if item is None:
                    counts[arm]["no_marker"] += 1
                    continue
                inputs.append(item)
            write_jsonl(out / reader / arm / f"errors.fold{fold}.jsonl", errors)
            counts[arm]["errors"] += len(errors)
            path = arm_file(out, reader, arm, fold)
            todo = sel.pending_inputs(inputs, read_jsonl(path), engine.name)
            for start in range(0, len(todo), CHUNK):
                chunk = todo[start:start + CHUNK]
                began = time.perf_counter()
                texts = engine.generate([(i.prompt, i.continuation) for i in chunk], sel.MAX_NEW_TOKENS[arm])
                rows = []
                for i, t in zip(chunk, texts):
                    raw, counts_ = token_counts(engine, i)
                    rows.append(sel.make_row(i, t, engine.name, engine.model, raw, counts_).to_dict())
                append_jsonl(path, rows)
                if fold in DEV_LOG_FOLDS:
                    for row in rows:
                        log(f"  [{reader} {arm} f{fold}] {row['question_id']} refusal={row['refusal']} "
                            f"answer={row['final_answer'][:60]!r}")
                log(f"{reader} {arm} fold{fold}: {start + len(chunk)}/{len(todo)} new rows "
                    f"({(time.perf_counter() - began) / len(chunk):.3f} s/row)")
            counts[arm]["new"] += len(todo)
            counts[arm]["rows"] += len(read_jsonl(path))
    return counts


# --- judging --------------------------------------------------------------------------------------------------

class JudgeCache:
    def __init__(self, own_dir: Path, settings: dict, reuse_dir: Path | None, reuse_settings: dict | None) -> None:
        self.own_dir, self.settings = Path(own_dir), settings
        self.reuse_dir, self.reuse_settings = (Path(reuse_dir) if reuse_dir else None), reuse_settings

    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return json.loads(path.read_text())["output"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _path(self, root: Path, key: str) -> Path:
        return root / key[:2] / f"{key}.json"

    def get(self, prompt: str, allow_reuse: bool) -> tuple[str | None, str | None]:
        key = sel.judge_cache_key(prompt, self.settings)
        hit = self._read(self._path(self.own_dir, key))
        if hit is not None:
            return hit, "exp35_cache"
        if allow_reuse and self.reuse_dir is not None:
            key23 = sel.judge_cache_key(prompt, self.reuse_settings)
            hit = self._read(self._path(self.reuse_dir, key23))
            if hit is not None:
                return hit, "exp23_cache"
        return None, None

    def put(self, prompt: str, output: str) -> None:
        key = sel.judge_cache_key(prompt, self.settings)
        path = self._path(self.own_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(json.dumps({"prompt": prompt, "max_new_tokens": sel.JUDGE_MAX_NEW_TOKENS, **self.settings,
                                       "output": output}))
        os.replace(partial, path)


def judge_items(reader: str, fold: int, out: Path, by_q: dict, qids: list[str], questions: dict[str, str]) -> list[dict]:
    """Every (item, answer text) to judge for one reader and fold."""
    items = []
    for arm in sel.ARMS:
        for row in read_jsonl(arm_file(out, reader, arm, fold)):
            q = row["question_id"]
            if q not in by_q:
                continue
            items.append({"question_id": q, "fold": fold, "reader": reader, "item": arm, "answer": row["final_answer"]})
    for q in qids:
        record = by_q[q]
        if record.fold != fold:
            continue
        note, _ = sel.split_note(record.output)
        if note is not None:
            items.append({"question_id": q, "fold": fold, "reader": reader, "item": "D2", "answer": note.strip()})
        items.append({"question_id": q, "fold": fold, "reader": reader, "item": "cached",
                      "answer": sel.agent_answer(record.output)[0]})
    for it in items:
        record = by_q[it["question_id"]]
        it["judge_prompt"] = sel.judge_prompt(questions[it["question_id"]], record.gold, it["answer"], record.question_type)
    return items


def run_judge(readers: list[str], folds: list[int], engine, cache: JudgeCache, out: Path, by_reader: dict,
              qids: list[str], questions: dict[str, str], log=print) -> dict:
    all_items = []
    for reader in readers:
        for fold in folds:
            all_items += judge_items(reader, fold, out, by_reader[reader], qids, questions)
    todo: dict[str, None] = {}
    for it in all_items:
        reply, source = cache.get(it["judge_prompt"], allow_reuse=it["item"] != "cached")
        if reply is None:
            todo[it["judge_prompt"]] = None
    pending = list(todo)
    log(f"judge: {len(all_items)} items, {len(pending)} judge prompts not cached")
    for start in range(0, len(pending), CHUNK * 4):
        chunk = pending[start:start + CHUNK * 4]
        began = time.perf_counter()
        replies = engine.generate([(p, "") for p in chunk], sel.JUDGE_MAX_NEW_TOKENS)
        for prompt, reply in zip(chunk, replies):
            cache.put(prompt, reply.strip())
        log(f"judge: {start + len(chunk)}/{len(pending)} ({(time.perf_counter() - began) / len(chunk):.3f} s/call)")
    counts: dict[str, int] = {}
    for reader in readers:
        for fold in folds:
            rows = []
            for it in [i for i in all_items if i["reader"] == reader and i["fold"] == fold]:
                reply, source = cache.get(it["judge_prompt"], allow_reuse=it["item"] != "cached")
                rows.append({"question_id": it["question_id"], "fold": fold, "reader": reader, "item": it["item"],
                             "answer": it["answer"], "judge_prompt_sha256": sel.sha256(it["judge_prompt"]),
                             "verdict": sel.verdict_of(reply), "source": source, "judge_engine": engine.name,
                             "judge_model": engine.model,
                             "sampling": {**sel.SAMPLING, "max_new_tokens": sel.JUDGE_MAX_NEW_TOKENS}})
                counts[it["item"]] = counts.get(it["item"], 0) + 1
            write_jsonl(judge_file(out, reader, fold), rows)
    return counts


def judge_questions(qids: list[str]) -> dict[str, str]:
    from memctl.envs.longmemeval import _instance
    from memctl.jacobian_data import _instance_number

    number = _instance_number(DATA)
    return {q: sel.judge_question(_instance(DATA, number[q])) for q in qids}


# --- main -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", default="qwen3b", help="qwen3b, qwen7b or both comma-separated")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--arms", default=",".join(sel.ARMS))
    parser.add_argument("--backend", default="vllm", choices=("vllm", "hf", "stub"))
    parser.add_argument("--out", default="runs/exp35", help="root; rows go to <out>/<reader>/")
    parser.add_argument("--judge", action="store_true", help="judge stage instead of generation (loads the 7B judge)")
    parser.add_argument("--runs", default="runs")
    args = parser.parse_args(argv)
    readers = args.reader.split(",")
    folds = [int(f) for f in args.folds.split(",")]
    arms = args.arms.split(",")
    if any(r not in READERS for r in readers) or any(a not in sel.ARMS for a in arms):
        parser.error(f"readers from {READERS}, arms from {sel.ARMS}")
    out = Path(args.out)
    began = time.perf_counter()
    by_reader, qids, mismatched = load_population(folds, args.runs)
    print(f"population: {len(qids)} questions on folds {folds}; {len(mismatched)} prompt mismatches excluded")
    if args.judge:
        engine = build_engine(args.backend, JUDGE_MODEL)
        if args.backend == "stub":
            cache = JudgeCache(Path("cache/stub_judge_exp35"), sel.judge_settings("stub"), None, None)
        else:
            cache = JudgeCache(Path("cache/judge_exp35"), sel.judge_settings(engine.name), Path("cache/judge_exp23"),
                               sel.judge_settings("openai"))
        counts = run_judge(readers, folds, engine, cache, out, by_reader, qids, judge_questions(qids))
        print(f"judged: {counts} in {time.perf_counter() - began:.1f} s")
        return 0
    for reader in readers:
        engine = build_engine(args.backend, MODELS[reader])
        manifest = {"reader": reader, "model": MODELS[reader], "engine": engine.name, "wording": sel.wording(),
                    "wording_sha256": sel.wording_sha256(), "folds": folds, "population": len(qids),
                    "prompt_mismatches": mismatched}
        (out / reader).mkdir(parents=True, exist_ok=True)
        (out / reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
        counts = generate_reader(reader, folds, arms, engine, out, by_reader[reader], qids)
        print(f"{reader}: {json.dumps(counts)} in {time.perf_counter() - began:.1f} s")
        del engine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
