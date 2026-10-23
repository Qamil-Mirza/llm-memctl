"""§40 pod stage: the fixed layer-29 setting (alpha 2) on LoCoMo (A40, transformers bf16; the 3B with hooks and the 7B
judge resident in ONE process, so neither is loaded twice).

    python scripts/exp40_pod_run.py run --out runs/exp40_pod --deadline <epoch s> --judge-reserve 300
    python scripts/exp40_pod_run.py judge --out runs/exp40_pod --deadline <epoch s>     # resume a cut judge pass
    ... --backend stub [--stub-judge d1|silent|disagree]                               # CPU stub engine and judge

Pre-registration: research/proposals/2026-10-23-exp40-fixed-setting-validation.md (rev 0; rev 1 at 877f86d) and the
Advisor's ruling on R/C selection (2026-10-23); D-34. Logic: memctl/analysis/exp40.py (setting, selection, smoke and
agreement arithmetic), memctl/locomo_view.py (the set and its prompts), memctl/analysis/steering.py (hooks, prefix
cache, rows, classifier v2), memctl/analysis/conditional.py (the §41 signals). The judge code is §38's
(scripts/exp38_pod_run.run_judge, build_judge, the §35 per-type prompts via answer_selection.judge_prompt).

`run`, in order (every model stage checks the deadline; the arms stop at deadline - judge reserve):
  0. inputs: the LoCoMo set (600 answerable + 40 adversarial, seed 40; counts and shas to view.json), v_29 byte-exact
     (sha asserted), 2 v_29 (sha asserted), the bf16 cast and its cosine with the fp32 vector printed, the seed-40
     random vector; the §41 rule (runs/exp41_dev/qwen3b/rule.json) if present (its sha recorded; scored only if
     scored_on_40 is true; else signals are only logged). Only (layer 29, alpha 2) is accepted (exit 2 otherwise).
  1. the 3B and the 7B judge loaded.
  2. D-22 smoke on the first 5 rows: batched (one left-padded batch) against single-sequence notes (the engine's own
     greedy loop, no batching): same first token >= 4/5 and a marker >= 4/5; A on the 5; one Steer-all generation; the
     judge on the 5 A answers and the Steer output must all be yes/no verdicts. Any failure: exit 5.
  3. judge agreement on the 60 frozen cached §35 rows (runs/exp40_inputs/agreement60.jsonl): < 0.90 -> exit 7 before
     any arm (and before the note pass: the earliest point the judge is loaded).
  4. N pass: the §23-template note on all 640 rows, batched 32, left padding, greedy (repetition penalty 1, no
     sampling), 256 new tokens, EOS stop. note = split_note(output.strip()); no marker -> the row is dropped (counted).
  5. A pass on every marked row: note + 'Answer:', unsteered, 16 new tokens, newline/EOS stop, with the step-0 logits
     and the layer-29 residual at ':' captured from the same pass (§41 signals: margin over the refusal-start ids,
     projection on v_29; residuals saved).
  6. the judge on every marked A answer of the 600 (the adversarial block is never judged). Counts printed: marker
     rate, refusal rate, judged-correct rate. Selection: R = first 120 refusals, C = first 40 judged correct, shuffle
     order (selection.json). R < 40: the arms still run, the report marks the claims descriptive.
  7. arms (memctl/analysis/exp40.ARM_ORDER): R A-prefix Steer-all, Random, Steer-0 (one prefix cache); N-cut notes on
     R (batched); R cut-prefix A-cut (with the §41 signals) and Steer-all-cut; C Steer-all; R B; adversarial B and
     Steer-all; then descriptive N-cut + A-cut on C and the adversarial rows (§41's s_evid there).
  8. the judge on every R and C output (and anything not yet judged), until the deadline.
Rows are sha-stamped (input, wording, engine, vector) and the run resumes from its files; a row on disk from another
input, engine or vector is refused. Exit codes: 2 setting refused, 3 input missing or wrong, 5 smoke, 7 agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl import locomo_view as lv  # noqa: E402
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import conditional as cd  # noqa: E402
from memctl.analysis import exp40 as ex  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from memctl.analysis.jacobian import ablate_evidence  # noqa: E402
from scripts.exp35_generate import MODELS, append_jsonl, read_jsonl, write_jsonl  # noqa: E402

READER = "qwen3b"
DIRECTIONS = "runs/exp38_dev/qwen3b/directions.npz"
RULE = "runs/exp41_dev/qwen3b/rule.json"
AGREEMENT = "runs/exp40_inputs/agreement60.jsonl"


def log(s: str) -> None:
    print(s, flush=True)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the stub engine and judge ------------------------------------------------------------------------------------

class LocomoStub(st.StubEngine):
    """The §38 stub with canned outputs by a hash rule, so a CPU run exercises every stage (the outputs mean nothing):
    the N pass writes a note with a marker on 3 of 4 rows (answer 'unknown' or a word of the gold); A repeats the
    note's answer; B answers a gold word; a cut prompt's note answers 'unknown'."""

    def __init__(self) -> None:
        super().__init__("stub")
        self.gold_of: dict[str, str] = {}          # sha256(prompt) -> gold (set by the runner)

    @staticmethod
    def _h(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def _word(self, prompt: str) -> str:
        gold = self.gold_of.get(st.sha256(prompt), "")
        words = [w for w in "".join(c if c.isalnum() or c == " " else " " for c in gold.lower()).split() if w.isascii()]
        return words[0] if words else "thing"

    def canned(self, prompt: str, continuation: str) -> str:
        h = self._h(prompt)
        if not continuation:
            if st.sha256(prompt) not in self.gold_of:       # a cut prompt (not in the view)
                return "Stub note: nothing here.\nAnswer: unknown"
            return ("Stub note: the memory says little.\nAnswer: unknown", f"Stub note: {self._word(prompt)}.\nAnswer: "
                    f"{self._word(prompt)}", "Stub note: the memory says little.\nAnswer: unknown",
                    "Stub note cut off before the marker")[h % 4]
        if sel.B_INSTRUCTION in continuation:
            return f" {self._word(prompt)}"
        note_answer = sel.final_answer(self.canned(prompt, "")) or "unknown"     # A repeats the N output's answer
        return " unknown" if sel.is_refusal_answer(note_answer) else f" {note_answer}"

    def prepare(self, item: st.SteerInput) -> st.Prepared:
        prepared = st.HookedEngine.prepare(self, item)
        prepared.cache.forced = self.encode(self.canned(item.prompt, item.continuation))
        return prepared


def build_engine(backend: str, threads: int):
    if backend == "stub":
        import torch
        torch.set_num_threads(1)          # a tiny model; never compete with a laptop job
        return LocomoStub()
    return st.build_engine("hf", MODELS[READER], threads)


class Exp40StubJudge:
    """§38's stub judge (D1 on the prompt's fields) plus the frozen agreement rows: 'd1' answers them with their cached
    verdict (so the stub passes the bar), 'disagree' flips them (the stub must stop at exit 7), 'silent' answers ''."""

    def __init__(self, mode: str, known: dict[str, bool]) -> None:
        from scripts.exp38_pod_run import StubJudge
        self.inner = StubJudge("silent" if mode == "silent" else "d1")
        self.mode, self.known = mode, known
        self.model, self.name = "stub-judge", f"stub-judge-{mode}"

    def generate(self, items, max_new_tokens):
        out = self.inner.generate(items, max_new_tokens)
        for i, (prompt, _) in enumerate(items):
            if self.mode != "silent" and prompt in self.known:
                v = self.known[prompt] if self.mode == "d1" else not self.known[prompt]
                out[i] = "yes" if v else "no"
        return out


def build_judge(backend: str, stub_mode: str, known: dict[str, bool]):
    if backend == "stub":
        return Exp40StubJudge(stub_mode, known)
    from scripts.exp38_pod_run import build_judge as b38
    return b38("hf")


# --- notes: batched and single-sequence ---------------------------------------------------------------------------

def notes_single(engine, prompts: list[str], max_new: int = ex.NOTE_MAX_NEW) -> list[dict]:
    """The engine's own greedy loop, one sequence at a time (no padding): prefill all but the last prompt token, then
    argmax steps; stops at a stop id (never recorded) or max_new tokens. Newlines do not stop a note."""
    return _greedy_loop(engine, prompts, max_new)


def _greedy_loop(engine, prompts: list[str], max_new: int) -> list[dict]:
    import torch
    out = []
    for p in prompts:
        item = st.SteerInput("N", "", 0, p, "", "", "")
        prepared = engine.prepare(item)
        cache, gen = prepared.cache, []
        with torch.no_grad():
            logits = engine._step(prepared.last_id, cache)
            while True:
                nxt = int(logits.argmax())
                if nxt in engine.stop_ids:
                    break
                gen.append(nxt)
                if len(gen) >= max_new:
                    break
                logits = engine._step(nxt, cache)
        out.append({"generated": engine.decode(gen), "gen_ids": gen})
    return out


def notes_batched(engine, prompts: list[str], max_new: int = ex.NOTE_MAX_NEW, batch: int = ex.NOTE_BATCH) -> list[dict]:
    """HF: transformers generate on batches of `batch` (sorted by length, left padding, attention mask), greedy with no
    sampling and repetition penalty 1 (the model's generation_config is overridden), EOS = the engine's stop ids. The
    stub has no batched model: it falls back to `notes_single` (the D-30 test covers the plumbing only)."""
    if engine.name == "stub":
        return _greedy_loop(engine, prompts, max_new)
    torch = engine.torch
    tok = engine.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [engine.encode(engine.chat_text(p)) for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out: list[dict | None] = [None] * len(enc)
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        width = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[pad] * (width - len(enc[i])) + enc[i] for i in idx], device=engine.device)
        mask = torch.tensor([[0] * (width - len(enc[i])) + [1] * len(enc[i]) for i in idx], device=engine.device)
        with torch.no_grad():
            gen = engine.net.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                      repetition_penalty=1.0, pad_token_id=pad, eos_token_id=sorted(engine.stop_ids))
        for row, i in enumerate(idx):
            g = [int(t) for t in gen[row][width:].tolist()]
            cut = next((k for k, t in enumerate(g) if t in engine.stop_ids), len(g))
            g = g[:cut]
            out[i] = {"generated": engine.decode(g), "gen_ids": g}
    return out  # type: ignore[return-value]


def note_row(q: str, pop: str, kind: str, prompt: str, res: dict, engine, mode: str) -> dict:
    output = res["generated"].strip()
    note, _ = sel.split_note(output)
    return {"question_id": q, "pop": pop, "kind": kind, "engine": engine.name, "mode": mode,
            "prompt_sha256": st.sha256(prompt), "generated": res["generated"], "output": output, "note": note,
            "marker": note is not None, "gen_ids": res["gen_ids"],
            "first_id": res["gen_ids"][0] if res["gen_ids"] else None, "n_new": len(res["gen_ids"]),
            "max_new_tokens": ex.NOTE_MAX_NEW}


# --- the run ------------------------------------------------------------------------------------------------------

class Run:
    def __init__(self, engine, view: lv.LocomoSet, out: Path, direction: dict, rule: dict | None, deadline: float | None,
                 judge_reserve: float) -> None:
        self.engine, self.view, self.dir = engine, view, out / READER
        self.dir.mkdir(parents=True, exist_ok=True)
        self.by_q = view.by_q()
        self.order = [r.question_id for r in view.rows]
        self.adv = [r.question_id for r in view.adversarial]
        self.layer = st.map_layer(ex.LAYER, engine.n_layers)
        self.vec = {"steer_all": engine_vector(direction["steer"], engine),
                    "steer0": engine_vector(direction["steer"], engine),
                    "random": engine_vector(direction["random"], engine)}
        self.v29 = engine_vector(direction["v"], engine)
        self.rule = rule
        self.ids = list(rule["refusal_start_ids"]) if rule else list(ex.DEFAULT_REFUSAL_IDS)
        if engine.name == "stub":
            self.ids = [engine.encode(" unknown")[0]]
        self.deadline, self.reserve = deadline, judge_reserve
        self.wsha = st.sha256(json.dumps(ex.WORDING, sort_keys=True))
        self.plan: dict = {"arms": {}, "cut": [], "skipped": []}
        self.stage_s: dict[str, float] = {}
        self._notes_cache: dict = {}

    # files
    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=1, default=float))

    def time_left(self, reserve: bool = True) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - (self.reserve if reserve else 0.0) - time.time()

    # inputs
    def item(self, kind: str, q: str) -> st.SteerInput:
        rec = self.by_q[q]
        if kind in ("A", "B"):
            note = self.notes("notes.jsonl")[q]["note"]
            prompt, cont = rec.prompt, sel.frozen_continuation(kind, note)
            no_marker = False
        elif kind == "cut":
            prompt = self.cut_prompt(q)
            note = self.notes("notes_cut.jsonl")[q]["note"]
            cont, no_marker = (note or "") + sel.ANSWER_MARKER, note is None
        else:
            raise ValueError(kind)
        return st.SteerInput(kind, q, 0, prompt, cont, st.sha256(json.dumps([prompt, cont])), rec.prompt_sha256,
                             no_marker)

    def cut_prompt(self, q: str) -> str:
        rec = self.by_q[q]
        cut = ablate_evidence(rec.prompt, rec.memory_lines, rec.evidence_lines)
        if cut == rec.prompt:
            raise ValueError(f"{q}: the evidence cut changed nothing")
        return cut

    def notes(self, name: str) -> dict[str, dict]:
        """question id -> note row of `name` (re-read when the file changed)."""
        p = self.path(name)
        key = (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
        cached = self._notes_cache.get(name)
        if cached is None or cached[0] != key:
            self._notes_cache[name] = (key, {r["question_id"]: r for r in read_jsonl(p)})
        return self._notes_cache[name][1]

    def manifest(self, extra: dict) -> None:
        m = {"engine": self.engine.name, "model": self.engine.model, "layer": self.layer, "alpha": ex.ALPHA,
             "unit": ex.UNIT, "wording_sha256": self.wsha, "wording": ex.WORDING, "refusal_ids": self.ids, **extra}
        p = self.path("manifest.json")
        if p.exists():
            old = json.loads(p.read_text())
            for k in ("engine", "model", "layer", "alpha", "wording_sha256", "refusal_ids", "view_shas", "vectors"):
                if old.get(k) != m.get(k):
                    raise SystemExit(f"REFUSED: {p} has {k} = {old.get(k)!r}, this run {m.get(k)!r}; move it aside")
        self.write_json("manifest.json", m)

    # notes stage
    def note_pass(self, name: str, pop, kind: str, qs: list[str], prompts: dict[str, str]) -> dict:
        """Batched notes for `qs` (resumable); `pop` a label or {question id: label}."""
        pop_of = pop if isinstance(pop, dict) else {q: pop for q in qs}
        pop = "+".join(sorted(set(pop_of.values()))) if isinstance(pop, dict) else pop
        done = self.notes(name)
        for q in qs:
            if q in done and (done[q]["prompt_sha256"] != st.sha256(prompts[q]) or done[q]["engine"] != self.engine.name):
                raise st.StaleRowError(f"{name}/{q}: a note on disk from another prompt or engine")
        todo = [q for q in qs if q not in done]
        t0 = time.perf_counter()
        n_done = 0
        step = ex.NOTE_BATCH * 4
        for start in range(0, len(todo), step):
            if self.time_left() <= 0:
                self.plan["cut"].append({"stage": name, "pop": pop, "done": len(qs) - len(todo) + n_done, "of": len(qs)})
                log(f"  notes {name} {pop}: CUT by the deadline after {n_done}/{len(todo)}")
                break
            chunk = todo[start:start + step]
            res = notes_batched(self.engine, [prompts[q] for q in chunk])
            append_jsonl(self.path(name), [note_row(q, pop_of[q], kind, prompts[q], r, self.engine, "batched")
                                           for q, r in zip(chunk, res)])
            n_done += len(chunk)
            log(f"  notes {name} {pop}: {n_done}/{len(todo)} ({time.perf_counter() - t0:.0f} s)")
        secs = time.perf_counter() - t0
        rows = self.notes(name)
        got = [q for q in qs if q in rows]
        info = {"rows": len(got), "of": len(qs), "marked": sum(rows[q]["marker"] for q in got), "seconds": secs,
                "new": n_done}
        log(f"  notes {name} {pop}: {info['rows']}/{len(qs)} rows, {info['marked']} with a marker, {secs:.0f} s")
        return info

    # generation stage
    def rows(self, name: str) -> dict[tuple, dict]:
        return {st.row_key(r): r for r in read_jsonl(self.path(name))}

    def generate(self, name: str, pop: str, kind: str, q: str, settings: list[str], capture: bool = False,
                 done: dict | None = None) -> dict[str, dict]:
        """One prefill, then each setting from a copy of the prefix cache (A, cut: the unsteered pass first with the
        step-0 logits and the layer-29 residual). Returns {setting name: row}; rows already on disk are checked and kept."""
        item = self.item(kind, q)
        done = self.rows(name) if done is None else done
        out, todo = {}, []
        for s in settings:
            setting = ex.setting_of(s, self.layer)
            vec = None if s == "none" else self.vec[s]
            k = (pop, kind, setting.key, q)
            if k in done:
                st.check_row(done[k], item, self.engine.name, self.wsha, vec)
                out[s] = done[k]
            else:
                todo.append((s, setting, vec))
        if not todo:
            return out
        prepared = self.engine.prepare(item)
        new = []
        for s, setting, vec in todo:
            specs = [] if vec is None else [st.add_spec(self.layer, vec, setting.steps)]
            cap = capture and s == "none"
            gen = self.engine.run(prepared, specs, ex.MAX_NEW, capture=(self.layer,) if cap else (), return_logits=cap)
            row = st.make_row(name, pop, item, setting, gen, self.engine, prepared, vec, ex.MAX_NEW,
                              self.by_q[q].gold)
            row["wording_sha256"] = self.wsha
            row["category"] = self.by_q[q].category
            if cap:
                h = gen.captured[self.layer]
                np.save(self.resid_dir(kind) / f"{q}.npy", h.astype(np.float32))
                row["step0"] = cd.step0_summary(gen.logits0, self.ids)
                row["s_proj"] = cd.projection(h, self.v29)
            new.append(row)
            out[s] = row
        append_jsonl(self.path(name), new)
        done.update({st.row_key(r): r for r in new})
        return out

    def resid_dir(self, kind: str) -> Path:
        d = self.path("resid") / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def arm(self, name: str, pop: str, kind: str, qs: list[str], settings: tuple, file: str = "arms.jsonl") -> None:
        t0 = time.perf_counter()
        done_rows = self.rows(file)
        n = 0
        for q in qs:
            if self.time_left() <= 0:
                self.plan["cut"].append({"arm": name, "done": n, "of": len(qs)})
                log(f"  arm {name}: CUT by the deadline after {n}/{len(qs)} rows")
                break
            if kind == "cut" and self.notes("notes_cut.jsonl").get(q) is None:
                continue      # its N-cut note was cut by the deadline
            self.generate(file, pop, kind, q, list(settings), capture=(kind == "cut"), done=done_rows)
            n += 1
        secs = time.perf_counter() - t0
        self.plan["arms"][name] = {"rows": n, "of": len(qs), "seconds": secs, "settings": list(settings)}
        log(f"  arm {name}: {n}/{len(qs)} rows in {secs:.0f} s")
        self.write_json("plan.json", self.plan)

    def stage(self, name: str, t: float) -> None:
        self.stage_s[name] = time.perf_counter() - t
        self.write_json("timing_pod.json", {"stage_s": self.stage_s})


def engine_vector(v: np.ndarray, engine) -> np.ndarray:
    """The 3B adds the vector itself; the stub (hidden 16) its first `hidden` components (plumbing only)."""
    if engine.hidden == v.shape[0]:
        return v
    if engine.name != "stub":
        raise SystemExit(f"REFUSED: engine hidden size {engine.hidden} != {v.shape[0]} on a non-stub engine")
    return np.ascontiguousarray(v[:engine.hidden])


def load_direction(path: str = DIRECTIONS) -> dict:
    fsha = file_sha256(path)
    try:
        d = ex.check_direction(np.load(path)[f"L{ex.LAYER}"], fsha)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    d["npz_sha256"] = fsha
    return d


def bf16_report(direction: dict) -> dict:
    """The cast the pod's hook applies (torch.as_tensor(vector, dtype=bfloat16)), and its cosine with the fp32 vector."""
    import torch
    out = {}
    for name in ("steer", "random"):
        v = direction[name]
        b = torch.as_tensor(v, dtype=torch.bfloat16).float().numpy()
        out[name] = {"cosine_bf16_fp32": ex.cosine(b, v), "norm_fp32": float(np.linalg.norm(v.astype(np.float64))),
                     "norm_bf16": float(np.linalg.norm(b.astype(np.float64))),
                     "max_abs_diff": float(np.max(np.abs(b.astype(np.float64) - v.astype(np.float64))))}
    return out


def load_rule(path: str = RULE) -> tuple[dict | None, dict]:
    p = Path(path)
    if not p.exists():
        return None, {"path": path, "present": False, "scored": False, "why": "no rule file: signals logged only"}
    rule = json.loads(p.read_text())
    info = {"path": path, "present": True, "sha256": file_sha256(p), "signal": rule.get("signal"),
            "sign": rule.get("sign"), "tau": rule.get("tau"), "scored_on_40": rule.get("scored_on_40"),
            "refusal_start_ids": rule.get("refusal_start_ids")}
    bi = Path("bundle_info.json")
    if bi.exists():
        info["commit"] = json.loads(bi.read_text()).get("rule_commit")
    info["scored"] = bool(rule.get("scored_on_40"))
    if not info["scored"]:
        info["why"] = "the §41 development gate failed (scored_on_40 false): signals logged only"
    return rule, info


def load_agreement(path: str = AGREEMENT) -> list[dict]:
    rows = read_jsonl(Path(path))
    if len(rows) != ex.AGREEMENT_N or any(sel.sha256(r["judge_prompt"]) != r["judge_prompt_sha256"] for r in rows):
        raise SystemExit(f"input {path}: expected {ex.AGREEMENT_N} rows with matching prompt shas (exit {ex.EXIT_INPUT})")
    return rows


class JudgeShim:
    """What exp38_pod_run.run_judge reads per question: gold and question_type (the rev 1 iv map)."""

    def __init__(self, by_q: dict) -> None:
        self.by_q = {q: type("R", (), {"gold": r.gold, "question_type": r.judge_type})() for q, r in by_q.items()}


class JudgeCtx:
    """What the judge pass needs of a run (a `Run` has the same attributes)."""

    def __init__(self, out: Path, by_q: dict) -> None:
        self.dir, self.by_q = out / READER, by_q

    def path(self, name: str) -> Path:
        return self.dir / name


def judge_items(run, include_arms: bool) -> list[dict]:
    """A-pass rows of the 600 (marked), then the R and C arm rows present; the adversarial block is never judged. Every
    call includes every row on disk (run_judge rewrites judge.jsonl from its items), so a resume never drops verdicts."""
    items = []
    files = [("apass.jsonl", {"all"})] + ([("arms.jsonl", {"R", "C"})] if include_arms else [])
    for f, pops in files:
        for r in read_jsonl(run.path(f)):
            if r["pop"] in pops:
                items.append({"pop": r["pop"], "kind": r["kind"], "setting_key": r["setting"]["key"],
                              "question_id": r["question_id"], "fold": 0, "answer": r["final_answer"],
                              "row_engine": r["engine"], "row_input_sha256": r["input_sha256"],
                              "tier": "apass" if f == "apass.jsonl" else "arms"})
    return items


def do_judge(run, judge, include_arms: bool, deadline: float | None) -> dict:
    from scripts.exp38_pod_run import run_judge
    items = judge_items(run, include_arms)
    questions = {q: r.question for q, r in run.by_q.items()}
    return run_judge(run.dir, judge, JudgeShim(run.by_q), items, questions, deadline)


def verdicts(run) -> dict[tuple, bool | None]:
    return {(r["pop"], r["kind"], r["setting_key"], r["question_id"]): st.parse_verdict(r["reply"])
            for r in read_jsonl(run.path("judge.jsonl"))}


# --- the smoke ----------------------------------------------------------------------------------------------------

def smoke(run: Run, judge) -> int:
    out = run.dir / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    qs = run.order[:ex.SMOKE_N]
    prompts = [run.by_q[q].prompt for q in qs]
    t0 = time.perf_counter()
    batched = notes_batched(run.engine, prompts)
    t1 = time.perf_counter()
    single = notes_single(run.engine, prompts)
    t2 = time.perf_counter()
    b_rows = [note_row(q, "smoke", "N", p, r, run.engine, "batched") for q, p, r in zip(qs, prompts, batched)]
    s_rows = [note_row(q, "smoke", "N", p, r, run.engine, "single") for q, p, r in zip(qs, prompts, single)]
    check = ex.smoke_notes_check(b_rows, s_rows)
    problems = list(check["problems"])
    log(f"smoke notes: same first token {check['same_first']}/{check['n']}, marker batched {check['marker_batched']} "
        f"single {check['marker_single']}, identical text {check['exact_text']}/{check['n']}")
    a_rows, steer = [], None
    t3 = time.perf_counter()
    for q, b in zip(qs, b_rows):
        if b["note"] is None:
            continue
        lr = run.by_q[q]
        cont = sel.frozen_continuation("A", b["note"])
        item = st.SteerInput("A", q, 0, lr.prompt, cont, st.sha256(json.dumps([lr.prompt, cont])), lr.prompt_sha256)
        prepared = run.engine.prepare(item)
        gen = run.engine.run(prepared, [], ex.MAX_NEW, capture=(run.layer,), return_logits=True)
        row = st.make_row("smoke", "smoke", item, ex.NONE, gen, run.engine, prepared, None, ex.MAX_NEW, lr.gold)
        if row["last_token"] != ":":
            problems.append(f"{q}: last prompt token {row['last_token']!r}, not ':'")
        a_rows.append(row)
        if steer is None:
            s = ex.setting_of("steer_all", run.layer)
            g2 = run.engine.run(prepared, [st.add_spec(run.layer, run.vec["steer_all"], s.steps)], ex.MAX_NEW)
            steer = st.make_row("smoke", "smoke", item, s, g2, run.engine, prepared, run.vec["steer_all"], ex.MAX_NEW,
                                lr.gold)
    t4 = time.perf_counter()
    judged = a_rows + ([steer] if steer else [])
    jp = [sel.judge_prompt(run.by_q[r["question_id"]].question, run.by_q[r["question_id"]].gold, r["final_answer"],
                           run.by_q[r["question_id"]].judge_type) for r in judged]
    replies = [x.strip() for x in judge.generate([(p, "") for p in jp], sel.JUDGE_MAX_NEW_TOKENS)] if jp else []
    t5 = time.perf_counter()
    vs = [st.parse_verdict(x) for x in replies]
    for r, x, v in zip(judged, replies, vs):
        log(f"  smoke judge {r['setting']['key']}: reply {x!r} -> {v}")
        if v is None:
            problems.append(f"judge reply {x!r} on {r['setting']['key']}/{r['question_id']} is not a verdict")
    if len(a_rows) < ex.SMOKE_MIN:
        problems.append(f"only {len(a_rows)} smoke rows reached A (need {ex.SMOKE_MIN})")
    rec = {"rows": qs, "notes_check": check, "batched": b_rows, "single": s_rows, "a_rows": a_rows, "steer": steer,
           "replies": replies, "verdicts": vs, "problems": problems,
           "seconds": {"notes_batched": t1 - t0, "notes_single": t2 - t1, "a_and_steer": t4 - t3, "judge": t5 - t4},
           "projection": projection({"notes_batched": t1 - t0, "a_and_steer": t4 - t3, "judge": t5 - t4},
                                    len(qs), len(judged))}
    (out / "smoke.json").write_text(json.dumps(rec, indent=1, default=float))
    pj = rec["projection"]
    log(f"smoke projection (conservative): notes {pj['notes_min']:.1f} min, A pass {pj['apass_min']:.1f} min, arms "
        f"{pj['arms_min']:.1f} min, judge {pj['judge_min']:.1f} min, total {pj['total_min']:.1f} min")
    if problems:
        log(f"SMOKE FAILED: {problems} (exit {ex.EXIT_SMOKE}); the chain stops before the note pass")
        return ex.EXIT_SMOKE
    log("smoke PASS")
    return 0


def projection(secs: dict, n: int, n_judged: int) -> dict:
    """Minutes at the smoke's rates. Notes: the 5-row batch's seconds per batch of 32 (a batch costs about its longest
    row's decode, so 5 rows stand for 32); A and arms: seconds per generation (the smoke ran n A + 1 Steer)."""
    per_batch = secs["notes_batched"]
    n_batches = -(-(lv.N_ROWS + lv.N_ADVERSARIAL) // ex.NOTE_BATCH) + -(-ex.R_CAP // ex.NOTE_BATCH) \
        + -(-(ex.C_CAP + lv.N_ADVERSARIAL) // ex.NOTE_BATCH)
    per_gen = secs["a_and_steer"] / max(n + 1, 1)
    apass = (lv.N_ROWS + lv.N_ADVERSARIAL) * per_gen
    arm_gens = ex.R_CAP * 3 + ex.R_CAP * 2 + ex.C_CAP + ex.R_CAP + lv.N_ADVERSARIAL * 2 + ex.C_CAP + lv.N_ADVERSARIAL
    arms = arm_gens * per_gen * 1.5           # steered outputs run to the 16-token cap more often
    judge_n = lv.N_ROWS + ex.R_CAP * 6 + ex.C_CAP + ex.AGREEMENT_N
    judge = judge_n * secs["judge"] / max(n_judged, 1)
    notes = n_batches * per_batch
    return {"notes_min": notes / 60, "apass_min": apass / 60, "arms_min": arms / 60, "judge_min": judge / 60,
            "total_min": (notes + apass + arms + judge) / 60, "note_batches": n_batches, "arm_generations": arm_gens,
            "judge_prompts": judge_n}


def agreement_check(run: Run, judge, rows: list[dict]) -> dict:
    t0 = time.perf_counter()
    replies = [x.strip() for x in judge.generate([(r["judge_prompt"], "") for r in rows], sel.JUDGE_MAX_NEW_TOKENS)]
    res = ex.agreement(replies, [r["cached_verdict"] for r in rows])
    res.update({"seconds": time.perf_counter() - t0, "judge_engine": judge.name, "input": AGREEMENT,
                "input_sha256": file_sha256(AGREEMENT),
                "rows": [{"question_id": r["question_id"], "arm": r["arm"], "cached": r["cached_verdict"],
                          "reply": x} for r, x in zip(rows, replies)]})
    run.write_json("agreement.json", res)
    log(f"judge agreement on {res['n']} cached §35 rows: {res['agree']}/{res['n']} = {res['rate']:.3f} (bar "
        f"{res['bar']}); replies that are not verdicts {res['unparseable']}")
    return res


# --- commands -----------------------------------------------------------------------------------------------------

def setup(args):
    try:
        ex.check_setting(args.layer, args.alpha)
    except ValueError as exc:
        log(str(exc))
        raise SystemExit(ex.EXIT_SETTING) from None
    for p in (lv.LOCOMO_PATH, DIRECTIONS, AGREEMENT):
        if not Path(p).exists():
            log(f"input missing: {p} (exit {ex.EXIT_INPUT})")
            raise SystemExit(ex.EXIT_INPUT)
    if (args.n_rows, args.n_adv) != (lv.N_ROWS, lv.N_ADVERSARIAL) and args.backend != "stub":
        log(f"REFUSED: --n-rows / --n-adv are for stub tests only (exit {ex.EXIT_SETTING})")
        raise SystemExit(ex.EXIT_SETTING)
    t0 = time.perf_counter()
    view = lv.build_set(args.locomo, n_rows=args.n_rows, n_adv=args.n_adv)
    log(f"LoCoMo set: {view.counts['rows']} rows + {view.counts['adversarial']} adversarial in {time.perf_counter() - t0:.0f} s;"
        f" excluded {view.counts['excluded']}; eligible answerable {view.counts['eligible_answerable']}; by category "
        f"{view.counts['rows_by_category']}; zero-fill rows {view.counts['zero_fill_rows']}")
    direction = load_direction(DIRECTIONS)
    casts = bf16_report(direction)
    log(f"direction v_29 sha {direction['v_sha256'][:8]}..., ||v_29|| {direction['v_norm']:.2f}; steer 2 v_29 sha "
        f"{direction['steer_sha256'][:8]}...; random (rng [40, 29]) norm {direction['random_norm']:.2f}; bf16 cast cosine "
        f"steer {casts['steer']['cosine_bf16_fp32']:.6f}, random {casts['random']['cosine_bf16_fp32']:.6f}")
    rule, rule_info = load_rule(args.rule)
    log(f"§41 rule: {json.dumps(rule_info)}")
    return view, direction, casts, rule, rule_info


def cmd_run(args) -> int:
    t_start = time.perf_counter()
    view, direction, casts, rule, rule_info = setup(args)
    agreement_rows = load_agreement(AGREEMENT)
    known = {r["judge_prompt"]: r["cached_verdict"] for r in agreement_rows}
    t = time.perf_counter()
    engine = build_engine(args.backend, args.threads)
    load_3b = time.perf_counter() - t
    run = Run(engine, view, Path(args.out), direction, rule, args.deadline, args.judge_reserve)
    if args.backend == "stub":
        engine.gold_of = {r.prompt_sha256: r.gold for r in view.rows + view.adversarial}
    run.stage_s["load_3b"] = load_3b
    run.write_json("view.json", {"counts": view.counts, "shas": view.shas, "rows": run.order, "adversarial": run.adv})
    run.manifest({"view_shas": view.shas, "vectors": {k: direction[k] for k in ("v_sha256", "steer_sha256",
                  "random_sha256")}, "npz_sha256": direction["npz_sha256"], "bf16_cast": casts, "rule": rule_info,
                  "random_seed": [ex.RANDOM_SEED, ex.LAYER]})
    t = time.perf_counter()
    judge = build_judge(args.backend, args.stub_judge, known)
    run.stage_s["load_judge"] = time.perf_counter() - t
    log(f"engines: {engine.name} (3B, {load_3b:.0f} s), judge {judge.name} ({run.stage_s['load_judge']:.0f} s); layer "
        f"{run.layer}; {run.time_left():.0f} s left before the judge reserve")

    log("== D-22 smoke")
    t = time.perf_counter()
    rc = smoke(run, judge)
    run.stage("smoke", t)
    if rc:
        return rc

    log("== judge agreement (before any arm)")
    t = time.perf_counter()
    ag = agreement_check(run, judge, agreement_rows)
    run.stage("agreement", t)
    if not ag["pass"]:
        log(f"AGREEMENT BELOW {ag['bar']}: the run stops before any arm (exit {ex.EXIT_AGREEMENT})")
        return ex.EXIT_AGREEMENT

    log(f"== N pass on {len(run.order)} + {len(run.adv)} rows (batched {ex.NOTE_BATCH})")
    t = time.perf_counter()
    prompts = {q: run.by_q[q].prompt for q in run.order + run.adv}
    pop_of = {**{q: "all" for q in run.order}, **{q: "adv" for q in run.adv}}
    info_notes = run.note_pass("notes.jsonl", pop_of, "N", run.order + run.adv, prompts)
    run.stage("notes", t)

    log("== A pass (unsteered, step-0 logits and layer-29 residual)")
    t = time.perf_counter()
    notes = run.notes("notes.jsonl")
    done = run.rows("apass.jsonl")
    for pop, qs in (("all", run.order), ("adv", run.adv)):
        n = 0
        for q in qs:
            if run.time_left() <= 0:
                run.plan["cut"].append({"stage": "apass", "pop": pop, "done": n})
                log(f"  A pass {pop}: CUT by the deadline")
                break
            if q in notes and notes[q]["marker"]:
                run.generate("apass.jsonl", pop, "A", q, ["none"], capture=True, done=done)
                n += 1
        log(f"  A pass {pop}: {n} marked rows")
    run.stage("apass", t)

    log("== judge on the A answers of the 600")
    t = time.perf_counter()
    do_judge(run, judge, include_arms=True, deadline=args.deadline)
    run.stage("judge_apass", t)
    a_rows = {r["question_id"]: r for r in read_jsonl(run.path("apass.jsonl")) if r["pop"] == "all"}
    v = verdicts(run)
    verdict = {q: v.get(("all", "A", "none", q)) for q in a_rows}
    marked = {q for q in run.order if q in notes and notes[q]["marker"]}
    counts = ex.pass_counts(run.order, marked, a_rows, verdict)
    picked = ex.select(run.order, a_rows, verdict)
    R, C = picked["R"], picked["C"]
    adv_marked = [q for q in run.adv if q in notes and notes[q]["marker"]]
    run.write_json("selection.json", {"counts": counts, "R": R, "C": C, "adversarial_marked": adv_marked,
                                      "notes": info_notes,
                                      "rule": "R = first 120 marked A refusals (§23 rule), C = first 40 marked A "
                                              "answers judged correct, shuffle order (the Advisor's ruling)"})
    log(f"on the {counts['n']}: marker rate {counts['marker_rate']:.3f} ({counts['marked']}), refusal rate "
        f"{counts['refusal_rate']:.3f} ({counts['refusals']}), judged-correct rate {counts['judged_correct_rate']:.3f} "
        f"({counts['judged_correct']}); no verdict {counts['no_verdict']}")
    log(f"selected R {len(R)} (cap {ex.R_CAP}), C {len(C)} (cap {ex.C_CAP}); adversarial marked {len(adv_marked)}"
        + ("" if len(R) >= ex.R_MIN_FOR_CLAIMS else f"; R < {ex.R_MIN_FOR_CLAIMS}: the claims will be descriptive"))

    log(f"== arms; {run.time_left():.0f} s left before the judge reserve")
    t = time.perf_counter()
    pops = {"R": R, "C": C, "adv": adv_marked}
    for name, pop, kind, settings in ex.ARM_ORDER:
        qs = pops[pop]
        if kind == "ncut":
            t1 = time.perf_counter()
            cut_prompts = {q: run.cut_prompt(q) for q in qs}
            info = run.note_pass("notes_cut.jsonl", pop, "Ncut", qs, cut_prompts)
            run.plan["arms"][name] = {**info, "seconds": time.perf_counter() - t1}
            run.write_json("plan.json", run.plan)
            continue
        run.arm(name, pop, kind, qs, settings)
    run.stage("arms", t)

    log("== judge on every R and C output")
    t = time.perf_counter()
    do_judge(run, judge, include_arms=True, deadline=args.deadline)
    run.stage("judge_arms", t)
    run.stage_s["total"] = time.perf_counter() - t_start
    run.write_json("timing_pod.json", {"stage_s": run.stage_s})
    run.write_json("plan.json", run.plan)
    log(f"run done in {run.stage_s['total']:.0f} s; cut {run.plan['cut']}")
    return 0


def cmd_judge(args) -> int:
    view, _, _, _, _ = setup(args)
    known = {r["judge_prompt"]: r["cached_verdict"] for r in load_agreement(AGREEMENT)}
    judge = build_judge(args.backend, args.stub_judge, known)
    do_judge(JudgeCtx(Path(args.out), view.by_q()), judge, include_arms=True, deadline=args.deadline)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "judge"))
    ap.add_argument("--out", default="runs/exp40_pod")
    ap.add_argument("--locomo", default=lv.LOCOMO_PATH)
    ap.add_argument("--rule", default=RULE)
    ap.add_argument("--backend", default="hf", choices=("hf", "stub"))
    ap.add_argument("--stub-judge", default="d1", choices=("d1", "silent", "disagree"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layer", type=float, default=ex.LAYER, help="must be 29 (anything else is refused)")
    ap.add_argument("--alpha", type=float, default=ex.ALPHA, help="must be 2 (anything else is refused)")
    ap.add_argument("--n-rows", type=int, default=lv.N_ROWS, help="stub tests only (the pod refuses another value)")
    ap.add_argument("--n-adv", type=int, default=lv.N_ADVERSARIAL, help="stub tests only")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: all model work ends by then")
    ap.add_argument("--judge-reserve", type=float, default=300.0,
                    help="seconds kept free before --deadline for the final judge pass (the 7B is already loaded)")
    args = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "judge": cmd_judge}[args.command](args)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        raise


if __name__ == "__main__":
    raise SystemExit(main())
