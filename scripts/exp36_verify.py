"""§36 verifier readouts for one reader: single-token A/B/C support labels and C4 prompt logprobs.

    python scripts/exp36_verify.py --reader qwen3b --folds 0,1,2,3,4 --backend vllm --out runs/exp36
    python scripts/exp36_verify.py --reader qwen7b --plan "V,CutB,CutF,Abs,CF|Vn,LC|V_own,C4|P2,P3" ...
    python scripts/exp36_verify.py ... --backend stub --out runs/exp36_stub      # CPU, fake logprobs, end to end

Pre-registration: research/proposals/2026-10-15-exp36-verification.md (frozen at d2a67f2 + a3a3f9c). Prompt builder,
score and counterfactual logic: memctl/analysis/verification.py (its docstring lists the readings taken where the
proposal is open).

Items. Both readers verify the SAME 3B candidates: V (no note), Vn (the 3B's frozen note shown), P2 and P3
(paraphrased instruction), CutB (the §35 arm-B candidate on the evidence-cut prompt), CutF (the §35 arm-Fcut
candidate on the cut prompt), Abs (the cached 3B answer on the 30 `*_abs` questions), CF (gold and substitute in the
intact and the counterfactual context, four variants) and LC (Cut-B with length-matched irrelevant lines in place of
the evidence). The 7B also verifies its own §35 B candidates (V_own). C4: two prompt-logprob passes on the reader's
own §35 arm-A prefix (chat text + cached note + "Answer:"), suffix " <candidate>" and " unknown"; the 3B on the 3B's
B candidates, the 7B on its own.

Engine. `vllm` (offline `vllm.LLM`, 0.8.5 as pinned by scripts/exp36_pod.sh: bf16, max_model_len 8192, prefix caching,
gpu_memory_utilization 0.70, max_num_batched_tokens 2048, the §35 prompt-logprob OOM lesson; greedy, max_tokens 1,
logprobs 20; prompt_logprobs 1 for C4, read only at the suffix positions). `hf` (transformers forward passes, bf16)
is the fallback the pod script switches to before any generation if vLLM does not import. `stub` returns
deterministic fake logprobs; it tokenises with the locally cached Qwen2.5 tokenizer when present. Raw input =
ids(chat text with generation prompt) + ids(continuation) [+ ids(suffix) for C4], as §35. Each row records engine,
model, sampling, wording sha256, prompt sha256, input sha256 and raw-input sha256; a resume refuses rows from another
engine, input or wording. A probe runs first: the longest label input and the longest C4 input of this run, timed,
not written (the first-minute real-shape call).

Plan. `--plan` lists item groups in priority order, separated by "|"; a group may end in "@SECONDS": it starts only
when more than SECONDS remain before `--deadline` (epoch seconds; 0 = none), so time is kept for the other reader's
higher-priority items. Chunks also stop at the deadline (rows are appended per chunk; rerun to resume).

Log: per-question lines for folds 0 and 1 only; folds 2-4 print chunk counts.

Layout under --out: <reader>/<item>/fold<k>.jsonl, <reader>/<item>/errors.fold<k>.jsonl, <reader>/_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import verification as ver  # noqa: E402
from memctl.jacobian_data import load_records  # noqa: E402
from scripts.exp35_generate import MODELS, READERS, STUB_TOKENIZER, append_jsonl, read_jsonl, write_jsonl  # noqa: E402

ALL_FOLDS = (0, 1, 2, 3, 4)
CHUNK = 128
DEV_LOG_FOLDS = (0, 1)
ITEMS_BY_READER = {"qwen3b": ("V", "Vn", "P2", "P3", "CutB", "CutF", "Abs", "CF", "LC", "C4"),
                   "qwen7b": ("V", "Vn", "P2", "P3", "CutB", "CutF", "Abs", "CF", "LC", "V_own", "C4")}
# Proposal §6 triage, read backwards (what is dropped first runs last): paraphrases, then V+n and the length control,
# then the 7B's own-candidate pass. Never dropped: the 3B's V, Cut-B, Cut-F and C4, the 7B's V on 3B candidates.
DEFAULT_PLAN = {"qwen3b": "V,CutB,CutF,C4,Abs,CF|Vn,LC|P2,P3",
                "qwen7b": "V,CutB,CutF,Abs,CF|Vn,LC|V_own,C4|P2,P3"}


# --- engines --------------------------------------------------------------------------------------------------

def _logaddexp(a: float | None, b: float) -> float:
    return b if a is None else max(a, b) + math.log1p(math.exp(-abs(a - b)))


class StubEngine:
    """Deterministic fake logprobs from a hash of the raw input; no model."""

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
        return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)

    def label_scores(self, raws: list[str]) -> list[dict[str, float]]:
        out = []
        for raw in raws:
            h = self._h(self.model + raw)
            mass = 0.3 if h % 23 == 0 else 0.9
            u = (h % 1000) / 1000
            probs = {"A": 0.9 * u * mass * 0.7, " A": 0.1 * u * mass * 0.7, "B": 0.15 * mass, " B": 0.0,
                     "C": (1 - u) * mass * 0.7 * 0.8, " C": (1 - u) * mass * 0.7 * 0.2}
            probs = {k: v for k, v in probs.items() if v > 0}
            probs["The"] = (1 - mass) * 0.6
            probs["Yes"] = (1 - mass) * 0.4
            out.append({k: math.log(v) for k, v in probs.items() if v > 0})
        return out

    def suffix_logprobs(self, raws: list[tuple[str, str]], n_suffix: list[int]) -> list[list[float]]:
        return [[-((self._h(self.model + prefix + suffix + str(j)) % 600) / 100) for j in range(max(n, 1))]
                for (prefix, suffix), n in zip(raws, n_suffix)]


class VLLMEngine:
    def __init__(self, model: str) -> None:
        import vllm
        from vllm import LLM

        self.model = model
        self.name = f"vllm-{vllm.__version__}"
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, seed=0, gpu_memory_utilization=0.70,
                       enable_prefix_caching=True, max_num_batched_tokens=2048)
        self.tokenizer = self.llm.get_tokenizer()

    def chat_text(self, prompt: str) -> str:
        from memctl.analysis.jacobian import wrap_chat
        return wrap_chat(self.tokenizer, prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def label_scores_ids(self, id_lists: list[list[int]]) -> list[dict[str, float]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=ver.TOP_LOGPROBS, seed=0)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in id_lists], params, use_tqdm=False)
        res = []
        for out in outs:
            top: dict[str, float] = {}
            for tid, lp in out.outputs[0].logprobs[0].items():
                token = lp.decoded_token if lp.decoded_token is not None else self.tokenizer.decode([tid])
                top[token] = _logaddexp(top.get(token), float(lp.logprob))
            res.append(top)
        return res

    def suffix_logprobs_ids(self, id_lists: list[list[int]], n_suffix: list[int]) -> list[list[float]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1, seed=0)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in id_lists], params, use_tqdm=False)
        res = []
        for out, ids, n in zip(outs, id_lists, n_suffix):
            pl = out.prompt_logprobs
            values = []
            for j in range(len(ids) - n, len(ids)):
                if pl is None or pl[j] is None or ids[j] not in pl[j]:
                    raise RuntimeError("prompt logprob missing at a suffix position (prefix cache served it?)")
                values.append(float(pl[j][ids[j]].logprob))
            res.append(values)
        return res


class HFEngine:
    """transformers forward passes, one input at a time, bf16 on CUDA."""

    def __init__(self, model: str) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.model = torch, model
        self.name = f"hf-transformers-{transformers.__version__}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.net = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()

    def chat_text(self, prompt: str) -> str:
        from memctl.analysis.jacobian import wrap_chat
        return wrap_chat(self.tokenizer, prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _logp(self, ids: list[int], keep: int):
        torch = self.torch
        with torch.no_grad():
            logits = self.net(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=keep).logits[0]
        return torch.log_softmax(logits.float(), -1)

    def label_scores_ids(self, id_lists: list[list[int]]) -> list[dict[str, float]]:
        res = []
        for ids in id_lists:
            logp = self._logp(ids, 1)[-1]
            values, idx = logp.topk(ver.TOP_LOGPROBS)
            top: dict[str, float] = {}
            for v, i in zip(values.tolist(), idx.tolist()):
                token = self.tokenizer.decode([i])
                top[token] = _logaddexp(top.get(token), float(v))
            res.append(top)
        return res

    def suffix_logprobs_ids(self, id_lists: list[list[int]], n_suffix: list[int]) -> list[list[float]]:
        res = []
        for ids, n in zip(id_lists, n_suffix):
            logp = self._logp(ids, n + 1)  # row r predicts ids[len - n - 1 + r + 1]
            res.append([float(logp[r, ids[len(ids) - n + r]]) for r in range(n)])
        return res


def build_engine(backend: str, model: str):
    if backend == "stub":
        return StubEngine(model)
    if backend == "vllm":
        return VLLMEngine(model)
    if backend == "hf":
        return HFEngine(model)
    raise ValueError(f"unknown backend {backend!r} (known: stub, vllm, hf)")


# --- sources --------------------------------------------------------------------------------------------------

class Sources:
    """§23 records of both readers (all folds: the length-control and name pools never depend on --folds), the §35
    rows the candidates come from, and the pools."""

    def __init__(self, runs: str = "runs", exp35: Path = Path("runs/exp35"), folds=ALL_FOLDS) -> None:
        records = {r: [x for f in ALL_FOLDS for x in load_records(r, f, runs_dir=runs)[0]] for r in READERS}
        qids, self.mismatched = sel.population(records)
        self.qids = qids
        self.rec = {r: {x.question_id: x for x in rows} for r, rows in records.items()}
        self.abs = sorted((x for x in records["qwen3b"] if not x.answerable), key=lambda x: x.question_id)
        self.folds = tuple(folds)
        self.rows35: dict[tuple[str, str], dict[str, dict]] = {}
        for reader, arm in (("qwen3b", "B"), ("qwen3b", "Fcut"), ("qwen7b", "B")):
            rows = {}
            for fold in self.folds:
                for row in read_jsonl(exp35 / reader / arm / f"fold{fold}.jsonl"):
                    rows[row["question_id"]] = row
            self.rows35[(reader, arm)] = rows
        engines = sorted({row["engine"] for rows in self.rows35.values() for row in rows.values()})
        self.exp35_engines = engines
        pop = set(qids)
        self.lc_pool: dict[str, list[str]] = {}
        lines_by_q = {q: [ln for ln in self.rec["qwen3b"][q].memory_lines if ln not in set(self.rec["qwen3b"][q].evidence_lines)]
                      for q in qids}
        all_lines = sorted({(ln, q) for q, lines in lines_by_q.items() for ln in lines})
        self._all_lines = all_lines
        self.name_pool = sorted({self.rec["qwen3b"][q].gold.strip() for q in pop
                                 if ver.gold_kind(self.rec["qwen3b"][q].gold) == "name"})

    def lc_pool_for(self, q: str) -> list[str]:
        return [ln for ln, owner in self._all_lines if owner != q]

    def fold_qids(self, fold: int) -> list[str]:
        return [q for q in self.qids if self.rec["qwen3b"][q].fold == fold]


def build_inputs(item: str, reader: str, fold: int, src: Sources) -> tuple[list[ver.VerifyInput], list[dict], dict]:
    """(inputs, errors, skip counts) for one item, reader and fold."""
    inputs, errors, skipped = [], [], {"no_candidate": 0, "not_qualifying": 0}
    b3, f3, b7 = src.rows35[("qwen3b", "B")], src.rows35[("qwen3b", "Fcut")], src.rows35[("qwen7b", "B")]

    def guard(q, fn):
        try:
            return fn()
        except (ValueError, AssertionError) as error:
            errors.append({"question_id": q, "fold": fold, "item": item, "error": str(error)})
            return None

    if item == "Abs":
        for rec in src.abs:
            if rec.fold != fold:
                continue
            cand = ver.candidate_of_output(rec.output)
            if cand is None:
                skipped["no_candidate"] += 1
                continue
            x = guard(rec.question_id, lambda: ver.label_input("Abs", rec, rec.prompt, cand, "qwen3b/cached",
                                                               extra={"marker": sel.split_note(rec.output)[0] is not None}))
            if x:
                inputs.append(x)
        return inputs, errors, skipped
    for q in src.fold_qids(fold):
        rec3, rec7 = src.rec["qwen3b"][q], src.rec["qwen7b"][q]
        if item == "CF":
            def cf_inputs():
                cf = ver.counterfactual(rec3.prompt, rec3.memory_lines, rec3.evidence_lines, rec3.gold, src.name_pool, q)
                if cf is None:
                    return None
                info = {k: cf[k] for k in ("original", "substitute", "kind", "start", "end", "outside")}
                return [ver.label_input("CF", rec3, ctx, cand, source, f"{name}@{where}", extra=info)
                        for where, ctx in (("intact", rec3.prompt), ("cf", cf["prompt"]))
                        for name, cand, source in (("orig", cf["original"], "gold"), ("sub", cf["substitute"], "substitute"))]
            made = guard(q, cf_inputs)
            if made is None:
                skipped["not_qualifying"] += 1
            else:
                inputs += made
            continue
        if item == "CutF":
            cand = ver.candidate_of_row(f3.get(q))
            if cand is None:
                skipped["no_candidate"] += 1
                continue
            x = guard(q, lambda: ver.label_input("CutF", rec3, ver.cut_prompt(rec3.prompt, rec3.memory_lines,
                                                                              rec3.evidence_lines), cand, "qwen3b/Fcut"))
            if x:
                inputs.append(x)
            continue
        if item == "V_own" or (item == "C4" and reader == "qwen7b"):
            cand = ver.candidate_of_row(b7.get(q))
            if cand is None:
                skipped["no_candidate"] += 1
                continue
            x = guard(q, (lambda: ver.label_input("V_own", rec7, rec7.prompt, cand, "qwen7b/B")) if item == "V_own"
                      else (lambda: ver.c4_input(rec7, cand, "qwen7b/B")))
            if x:
                inputs.append(x)
            continue
        cand = ver.candidate_of_row(b3.get(q))
        if cand is None:
            skipped["no_candidate"] += 1
            continue
        if item in ("V", "P2", "P3"):
            fn = lambda: ver.label_input(item, rec3, rec3.prompt, cand, "qwen3b/B")  # noqa: E731
        elif item == "Vn":
            fn = lambda: ver.label_input("Vn", rec3, rec3.prompt, cand, "qwen3b/B",  # noqa: E731
                                         note=sel.split_note(rec3.output)[0])
        elif item == "CutB":
            fn = lambda: ver.label_input("CutB", rec3, ver.cut_prompt(rec3.prompt, rec3.memory_lines,  # noqa: E731
                                                                      rec3.evidence_lines), cand, "qwen3b/B")
        elif item == "LC":
            def fn():
                lc, _, reps = ver.length_control(rec3.prompt, rec3.memory_lines, rec3.evidence_lines,
                                                 src.lc_pool_for(q), q, rec3.gold)
                lens = {"evidence_chars": [len(x) for x in rec3.evidence_lines], "replacement_chars": [len(x) for x in reps]}
                return ver.label_input("LC", rec3, lc, cand, "qwen3b/B", extra=lens)
        elif item == "C4":
            fn = lambda: ver.c4_input(rec3, cand, "qwen3b/B")  # noqa: E731
        else:
            raise ValueError(f"unknown item {item!r}")
        x = guard(q, fn)
        if x:
            inputs.append(x)
    return inputs, errors, skipped


# --- running --------------------------------------------------------------------------------------------------

def item_file(out: Path, reader: str, item: str, fold: int) -> Path:
    return out / reader / item / f"fold{fold}.jsonl"


def raw_and_ids(engine, x: ver.VerifyInput) -> tuple[str, list[int] | None, int | None]:
    """(raw text, ids or None, prefix token count for C4)."""
    chat = engine.chat_text(x.prompt)
    raw = chat + x.continuation + (x.suffixes[0] if x.suffixes else "")
    chat_ids = engine.encode(chat)
    if chat_ids is None:
        return raw, None, None
    prefix = chat_ids + engine.encode(x.continuation)
    return raw, prefix, len(prefix)


def run_chunk(engine, chunk: list[ver.VerifyInput]) -> list[tuple[dict, str, int | None]]:
    """[(result, raw text, n_input_tokens)] for a chunk of one kind."""
    prepared = [raw_and_ids(engine, x) for x in chunk]
    stub = isinstance(engine, StubEngine)
    if chunk[0].kind == "label":
        if stub:
            tops = engine.label_scores([raw for raw, _, _ in prepared])
        else:
            tops = engine.label_scores_ids([ids for _, ids, _ in prepared])
        return [({"top": t}, raw, None if ids is None else len(ids)) for t, (raw, ids, _) in zip(tops, prepared)]
    results = []
    lists, n_suffix, meta = [], [], []
    for x, (raw, ids, n_prefix) in zip(chunk, prepared):
        for suffix in x.suffixes:
            sids = engine.encode(suffix)
            lists.append(None if ids is None else ids + sids)
            n_suffix.append(None if sids is None else len(sids))
            meta.append((engine.chat_text(x.prompt) + x.continuation, suffix))
    if stub:
        lps = engine.suffix_logprobs(meta, [n if n is not None else 1 for n in n_suffix])
    else:
        lps = engine.suffix_logprobs_ids(lists, n_suffix)
    for i, (x, (raw, ids, n_prefix)) in enumerate(zip(chunk, prepared)):
        k = len(x.suffixes)
        pair = lps[i * k:(i + 1) * k]
        total = None if ids is None else sum(len(lists[i * k + j]) for j in range(k))
        results.append(({"suffix_logprobs": pair, "n_prefix_tokens": n_prefix}, raw, total))
    return results


def probe(engine, inputs: list[ver.VerifyInput], log=print) -> None:
    """The first-minute real-shape call: the longest label input and the longest C4 input, timed, not written."""
    for kind in ("label", "c4"):
        pool = [x for x in inputs if x.kind == kind]
        if not pool:
            continue
        longest = max(pool, key=lambda x: len(x.prompt) + len(x.continuation))
        began = time.perf_counter()
        res = run_chunk(engine, [longest])
        took = time.perf_counter() - began
        result, _, n_tokens = res[0]
        shape = (f"label mass {ver.label_probs(result['top'])['label_mass']:.3f}" if kind == "label"
                 else f"suffix tokens {[len(v) for v in result['suffix_logprobs']]}")
        log(f"probe {kind}: {longest.item}/{longest.question_id} {n_tokens} tokens, {took:.2f} s, {shape}")


def parse_plan(plan: str, allowed: tuple[str, ...]) -> list[tuple[list[str], int]]:
    groups = []
    for part in plan.split("|"):
        part = part.strip()
        if not part:
            continue
        names, _, reserve = part.partition("@")
        items = [x.strip() for x in names.split(",") if x.strip()]
        bad = [x for x in items if x not in allowed]
        if bad:
            raise ValueError(f"items {bad} not allowed for this reader (allowed: {allowed})")
        groups.append((items, int(reserve or 0)))
    return groups


def verify_reader(reader: str, folds: list[int], plan: list[tuple[list[str], int]], engine, out: Path, src: Sources,
                  deadline: float = 0.0, log=print, do_probe: bool = True) -> dict:
    counts: dict[str, dict] = {}
    staged: dict[tuple[str, int], tuple[list, list, dict]] = {}
    for items, _ in plan:
        for item in items:
            for fold in folds:
                staged[(item, fold)] = build_inputs(item, reader, fold, src)
    if do_probe:
        probe(engine, [x for inputs, _, _ in staged.values() for x in inputs], log)
    for items, reserve in plan:
        if deadline and deadline - time.time() <= reserve:
            log(f"{reader}: skipping group {items} (less than {reserve} s left before the deadline)")
            for item in items:
                counts[item] = {"skipped_group": True}
            continue
        for item in items:
            c = counts[item] = {"rows": 0, "new": 0, "no_candidate": 0, "not_qualifying": 0, "errors": 0, "cut": False}
            for fold in folds:
                inputs, errors, skipped = staged[(item, fold)]
                write_jsonl(out / reader / item / f"errors.fold{fold}.jsonl", errors)
                c["errors"] += len(errors)
                c["no_candidate"] += skipped["no_candidate"]
                c["not_qualifying"] += skipped["not_qualifying"]
                path = item_file(out, reader, item, fold)
                todo = ver.pending_inputs(inputs, read_jsonl(path), engine.name)
                for start in range(0, len(todo), CHUNK):
                    if deadline and time.time() >= deadline:
                        log(f"{reader} {item} fold{fold}: deadline reached, {len(todo) - start} rows left")
                        c["cut"] = True
                        break
                    chunk = todo[start:start + CHUNK]
                    began = time.perf_counter()
                    results = run_chunk(engine, chunk)
                    per = (time.perf_counter() - began) / len(chunk)
                    rows = [ver.make_row(x, reader, engine.name, engine.model, raw, n, per, res)
                            for x, (res, raw, n) in zip(chunk, results)]
                    append_jsonl(path, rows)
                    c["new"] += len(rows)
                    if fold in DEV_LOG_FOLDS:
                        for row in rows:
                            value = f"m={row['m']:+.3f}" if item == "C4" else (
                                f"s={row['s']:.3f} mass={row['label_mass']:.3f}" if row["s"] is not None else "s=None")
                            log(f"  [{reader} {item} f{fold}] {row['question_id']} {row['variant']} {value}")
                    log(f"{reader} {item} fold{fold}: {start + len(chunk)}/{len(todo)} new rows ({per:.3f} s/row)")
                c["rows"] += len(read_jsonl(path))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", required=True, choices=READERS)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--plan", default=None, help="item groups in priority order, e.g. 'V,CutB|Vn,LC@600|P2,P3@1200'")
    parser.add_argument("--backend", default="vllm", choices=("vllm", "hf", "stub"))
    parser.add_argument("--out", default="runs/exp36")
    parser.add_argument("--exp35", default="runs/exp35", help="§35 run directory (candidates)")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    plan = parse_plan(args.plan or DEFAULT_PLAN[args.reader], ITEMS_BY_READER[args.reader])
    out = Path(args.out)
    began = time.perf_counter()
    src = Sources(args.runs, Path(args.exp35), folds)
    if len(src.exp35_engines) > 1:
        raise SystemExit(f"REFUSED: the §35 candidates come from more than one engine: {src.exp35_engines}")
    print(f"population: {len(src.qids)} questions (all folds; {len(src.mismatched)} prompt mismatches excluded); "
          f"abs {len(src.abs)}; folds {folds}; plan {plan}")
    engine = build_engine(args.backend, MODELS[args.reader])
    manifest = {"reader": args.reader, "model": MODELS[args.reader], "engine": engine.name, "wording": ver.wording(),
                "wording_sha256": ver.wording_sha256(), "folds": folds, "plan": plan, "population": len(src.qids),
                "exp35": args.exp35, "exp35_engines": src.exp35_engines, "sampling": ver.SAMPLING,
                "c4_sampling": ver.C4_SAMPLING}
    (out / args.reader).mkdir(parents=True, exist_ok=True)
    (out / args.reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = verify_reader(args.reader, folds, plan, engine, out, src, args.deadline, do_probe=not args.no_probe)
    manifest["counts"] = counts
    (out / args.reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{args.reader}: {json.dumps(counts)} in {time.perf_counter() - began:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
