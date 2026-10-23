"""§38 pod stage (A40, transformers bf16 for the 3B with hooks and for the 7B judge; one engine for every role).

    python scripts/exp38_pod_run.py smoke --out runs/exp38_pod          # D-22: 5 dev rows, one Steer, the judge
    python scripts/exp38_pod_run.py run   --out runs/exp38_pod --deadline <epoch s> --judge-reserve 420
    python scripts/exp38_pod_run.py judge --out runs/exp38_pod --deadline <epoch s>
    ... --backend stub [--stub-judge d1|silent]                          # CPU stub engine and stub judge (tests)

Pre-registration: research/proposals/2026-10-22-exp38-refusal-direction.md §6 with amendments 1 and 2; the dev review
research/reviews/2026-10-23-exp38-dev.md and its addendum (judge conditions). Laptop logic is reused: rows, hooks and
the prefix cache come from memctl/analysis/steering.py and scripts/exp38_dev.Run; scripts/exp38_dev.py itself still
refuses folds other than 0-1 (this script is the only one that writes confirmation-fold rows).

`run` (the 3B, folds 0-4, max 64 new tokens, greedy, newline/EOS stop, prefix KV cache before the ':' of Answer:):
  (i)   reproduction: unsteered A and B on every R row (A refuses in §35; 28 dev + 64 confirmation), A on every C row
        (A judged correct in §35; 39 dev + 55 confirmation). R rows: refusal flags (build review ruling 1, D1 verdict
        match a column). C rows: A answers here; whether the NEW judge calls it correct is applied by the report after
        the judge pass (the pod's C rule; on the laptop it was D1). Preliminary bar here: R flags + C 'answers' >= 0.8,
        else exit 3. Residuals at the ':' position are kept for the dev R rows only (the estimation needs them).
  (ii)  v_L from the reproduced dev pairs (fixed population: dev R rows where B answers, 13 in §35; never confirmation
        rows), at layers 25 and 29 on THIS engine; cosine with the laptop direction printed (descriptive).
  (iii) sanity S on those pairs: final-layer (35) overwrite with B's state >= 0.9 first-token match, zero add (L25, L29)
        and self-overwrite (L35, L7) byte-exact, 0.2 depth (L7) descriptive. Failure -> exit 4.
  (iv)  arms on the reproduced confirmation rows. Primary = the laptop gate.json's choice (read, never re-chosen):
        Steer, Random, Reverse on R (one A prefill per row), Steer-cut and A-cut (alpha 0, setting-independent, once)
        on R, Steer and Reverse (amendment 1.5, descriptive) on C (the C rows where A answered). Secondary (amendment 2, L29 alpha 2 vnorm): one arm at a time,
        Steer R, Steer-cut R, Random R, Steer C, Reverse R; an arm is skipped and logged when its estimate (the
        matching primary arm's measured seconds x 1.2) would end past deadline - judge reserve. The primary is cut
        (and logged) only if that point is reached inside it.
`judge` (Qwen2.5-7B-Instruct, HF bf16, the §35 judge prompt and settings: chat-wrapped, greedy, 4 new tokens) on every
  R and C output of the run (sanity rows are plumbing and are not judged) plus the cached §35 A and B final answers of
  the confirmation rows (pop 'cached35', the same-text judge-engine check; never a claim). Priority order: confirmation
  reproduction, primary arms, secondary arms, cached §35 texts, dev reproduction. Replies cached per prompt text in
  <run-dir>/judge_replies.jsonl; <run-dir>/judge.jsonl rewritten after every chunk.
`smoke` (D-22, into <out>/smoke/qwen3b): 5 dev pair rows through reproduction (A, B), one Steer generation at the primary
  layer and alpha from a 5-row direction, the 3B unloaded, the judge loaded, every smoke output judged; any reply that
  is not a yes/no verdict -> exit 5 (the chain stops before the full run). Measured rates and a projection printed.

Per-question log lines are printed for folds 0-1 only. One engine per role and run directory (manifest.json for the
steering engine, judge_manifest.json for the judge); a rerun resumes and refuses rows of another engine or wording.
Exit codes: 3 reproduction, 4 sanity, 5 smoke, 6 population counts differ from the §35 cache (an input is wrong).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import JUDGE_MODEL, MODELS, read_jsonl, write_jsonl  # noqa: E402
from scripts.exp38_dev import Run, load_source  # noqa: E402

READER = "qwen3b"
DEV = (0, 1)
CONFIRM = (2, 3, 4)
ALL_FOLDS = DEV + CONFIRM
EXPECTED = {"R_dev": 28, "R_confirm": 64, "pairs_dev": 13, "C_dev": 39, "C_confirm": 55}
EXIT_REPRO, EXIT_SANITY, EXIT_SMOKE, EXIT_POP = 3, 4, 5, 6
SMOKE_N = 5
ESTIMATE_MARGIN = 1.2
JUDGE_BATCH = 16
JUDGE_CHUNK = 64
DEFAULT_DESIGN = "runs/exp38_dev/qwen3b/gate.json"
LAPTOP_DIRECTIONS = "runs/exp38_dev/qwen3b/directions.npz"


def log(s: str) -> None:
    print(s, flush=True)


# --- population ---------------------------------------------------------------------------------------------------

def load_all(exp35: Path, runs: str, own_root: Path, check: bool = True):
    d, pop = load_source(list(ALL_FOLDS), exp35, runs, own_root)
    fold = lambda q: d.by_q[q].fold  # noqa: E731
    counts = {"R_dev": sum(fold(q) in DEV for q in pop.R), "R_confirm": sum(fold(q) in CONFIRM for q in pop.R),
              "pairs_dev": sum(fold(q) in DEV for q in pop.pairs), "C_dev": sum(fold(q) in DEV for q in pop.C),
              "C_confirm": sum(fold(q) in CONFIRM for q in pop.C)}
    log(f"populations: {counts}")
    if check and counts != EXPECTED:
        log(f"POPULATION MISMATCH: expected {EXPECTED} from the §35 cache; an input is missing or wrong (exit {EXIT_POP})")
        raise SystemExit(EXIT_POP)
    return d, pop, counts


# --- the steering run ---------------------------------------------------------------------------------------------

class PodRun(Run):
    """exp38_dev.Run on all folds, with fold-aware logging, an arms stage and the deadline."""

    def __init__(self, engine, d, pop, out: Path, primary: st.Setting, secondary: st.Setting, max_new: int,
                 deadline: float | None, judge_reserve: float) -> None:
        layers = sorted({primary.layer, secondary.layer})
        alphas = sorted({primary.alpha, secondary.alpha})
        super().__init__(engine, d, pop, out, layers, alphas, primary.unit, max_new, log)
        self.primary, self.secondary = primary, secondary
        self.gen_deadline = None if deadline is None else deadline - judge_reserve
        self.plan: dict = {"primary": primary.key, "secondary": secondary.key, "arms": {}, "skipped": [], "cut": []}
        self.stage_s: dict[str, float] = {}

    def log_rows(self, stage: str, rows: list[dict]) -> None:
        dev = [r for r in rows if r["fold"] in DEV]
        super().log_rows(stage, dev)

    def fold(self, q: str) -> int:
        return self.d.by_q[q].fold

    def check_manifest(self) -> None:
        manifest = {"engine": self.engine.name, "model": self.engine.model, "role": "steering", "layers": self.layers,
                    "alphas": self.alphas, "unit": self.unit, "max_new_tokens": self.max_new,
                    "wording_sha256": self.wsha, "wording": st.wording(self.max_new), "folds": list(ALL_FOLDS),
                    "primary": self.primary.key, "secondary": self.secondary.key}
        path = self.dir / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            for k in ("engine", "model", "layers", "alphas", "unit", "max_new_tokens", "wording_sha256", "primary",
                      "secondary"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"REFUSED: {path} has {k} = {old.get(k)!r}, this run {manifest[k]!r}; one engine "
                                     "per role and run directory (move it aside)")
        self.write_json("manifest.json", manifest)

    # --- (i) ---
    def reproduction(self) -> dict:
        none = st.Setting("none")
        result: dict = {"R": {}, "C": {}}
        for q in self.pop.R:
            cap = self.fold(q) in DEV
            a = self.do("repro", "R", "A", q, [none], capture=cap)["none"]
            b = self.do("repro", "R", "B", q, [none], capture=cap)["none"]
            ra = st.reproduces(a, self.d.rows["A"][q], self.d.by_q[q].gold)
            rb = st.reproduces(b, self.d.rows["B"][q], self.d.by_q[q].gold)
            result["R"][q] = {"A": ra, "B": rb, "ok": ra["ok"] and rb["ok"], "fold": self.fold(q)}
        for q in self.pop.C:
            a = self.do("repro", "C", "A", q, [none])["none"]
            result["C"][q] = {"answers": not a["refusal"], "fold": self.fold(q)}
        ok_r = [q for q, v in result["R"].items() if v["ok"]]
        ans_c = [q for q, v in result["C"].items() if v["answers"]]
        n = len(result["R"]) + len(result["C"])
        result.update({"R_ok": ok_r, "C_answers": ans_c, "rate_R": len(ok_r) / len(result["R"]),
                       "rate_preliminary": (len(ok_r) + len(ans_c)) / n,
                       "R_verdict_match_d1": sum(v["A"]["verdict_match"] and v["B"]["verdict_match"]
                                                 for v in result["R"].values()),
                       "note": "C rows: the report adds the new judge's verdict (A must be judged correct)"})
        result["pass_preliminary"] = result["rate_preliminary"] >= st.REPRO_GATE
        return result

    # --- (ii) ---
    def pod_directions(self, pairs: list[str]) -> dict:
        out = self.directions(pairs)
        lap = Path(LAPTOP_DIRECTIONS)
        if lap.exists() and self.engine.n_layers == st.N_LAYERS:
            old = np.load(lap)
            arrays = np.load(self.dir / "directions.npz")
            for L in self.layers:
                if f"L{L}" in old.files:
                    a, b = arrays[f"L{L}"].astype(np.float64), old[f"L{L}"].astype(np.float64)
                    out[str(L)]["cosine_with_laptop"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            self.write_json("directions.json", {"pairs": pairs, "layers": out})
        return out

    # --- (iv) ---
    def time_left(self) -> float:
        return float("inf") if self.gen_deadline is None else self.gen_deadline - time.time()

    def _arm(self, name: str, pop: str, kind: str, qs: list[str], settings: list[st.Setting]) -> dict:
        t0 = time.perf_counter()
        done = 0
        for q in qs:
            if self.time_left() <= 0:
                self.plan["cut"].append({"arm": name, "done": done, "of": len(qs)})
                log(f"  arm {name}: CUT by the deadline after {done}/{len(qs)} rows")
                break
            self.do("arms", pop, kind, q, settings)
            done += 1
        secs = time.perf_counter() - t0
        self.plan["arms"][name] = {"rows": done, "of": len(qs), "seconds": secs,
                                   "settings": [s.key for s in settings]}
        log(f"  arm {name}: {done}/{len(qs)} rows in {secs:.0f} s")
        self.write_json("plan.json", self.plan)
        return self.plan["arms"][name]

    def arms(self, R: list[str], C: list[str]) -> None:
        p, s = self.primary, self.secondary
        mk = lambda mode, x: st.Setting(mode, x.layer, x.alpha, x.unit)  # noqa: E731
        # primary: one A prefill per row for Steer, Random, Reverse; one cut prefill for A-cut and Steer-cut
        self._arm("primary:R:A", "R", "A", R, [p, mk("random", p), mk("reverse", p)])
        self._arm("primary:R:cut", "R", "cut", R, [p, st.Setting("none")])
        self._arm("primary:C:A", "C", "A", C, [p, mk("reverse", p)])   # Reverse on C: amendment 1.5, descriptive
        # per-arm seconds of the primary, to estimate the secondary's (prefill split evenly over the settings)
        prim = self.measured_arm_seconds()
        order = [("secondary:R:A:steer", "R", "A", R, [s], ("R", "A", p.key)),
                 ("secondary:R:cut:steer", "R", "cut", R, [s], ("R", "cut", p.key)),
                 ("secondary:R:A:random", "R", "A", R, [mk("random", s)], ("R", "A", mk("random", p).key)),
                 ("secondary:C:A:steer", "C", "A", C, [s], ("C", "A", p.key)),
                 ("secondary:R:A:reverse", "R", "A", R, [mk("reverse", s)], ("R", "A", mk("reverse", p).key))]
        for name, pop, kind, qs, settings, like in order:
            est = ESTIMATE_MARGIN * prim.get(like, 0.0)
            if est > self.time_left():
                self.plan["skipped"].append({"arm": name, "estimate_s": est, "left_s": self.time_left()})
                log(f"  arm {name}: SKIPPED (estimate {est:.0f} s > {self.time_left():.0f} s left before the judge "
                    "reserve)")
                self.write_json("plan.json", self.plan)
                continue
            self._arm(name, pop, kind, qs, settings)

    def measured_arm_seconds(self) -> dict:
        """(pop, kind, setting key) -> seconds of that arm's rows: copy + first step + decode + prefill / settings."""
        rows = self.rows("arms")
        per_prefill: dict[tuple, int] = {}
        for (pop, kind, key, q) in rows:
            per_prefill[(pop, kind, q)] = per_prefill.get((pop, kind, q), 0) + 1
        out: dict[tuple, float] = {}
        for (pop, kind, key, q), r in rows.items():
            t = r["timing"]
            secs = t["copy_s"] + t["first_step_s"] + t["decode_s"] + t["prefill_s"] / per_prefill[(pop, kind, q)]
            out[(pop, kind, key)] = out.get((pop, kind, key), 0.0) + secs
        return out


def stub_oracle(run: Run, qids_r: list[str], qids_c: list[str]) -> dict[str, str]:
    """Stub only: the cached §35 generations for A and B on R and A on C, so the stub reproduces the cache."""
    texts = {}
    for q in qids_r:
        for kind in ("A", "B"):
            it = run.item(kind, q)
            texts[st.sha256(run.engine.chat_text(it.prompt) + it.continuation)] = run.d.rows[kind][q]["generated"]
    for q in qids_c:
        it = run.item("A", q)
        texts[st.sha256(run.engine.chat_text(it.prompt) + it.continuation)] = run.d.rows["A"][q]["generated"]
    return texts


def settings_from_design(design: Path, n_layers: int) -> tuple[st.Setting, st.Setting, dict]:
    fixed = json.loads(design.read_text())
    if not fixed.get("chosen") or not fixed.get("pass"):
        raise SystemExit(f"REFUSED: {design} chose no setting or did not pass; nothing to confirm")
    chosen = st.setting_from_key(fixed["chosen"])
    L2, a2, u2 = st.SECONDARY
    primary = st.Setting("steer", st.map_layer(chosen.layer, n_layers), chosen.alpha, chosen.unit)
    secondary = st.Setting("steer", st.map_layer(L2, n_layers), a2, u2)
    info = {"design": str(design), "design_chosen": fixed["chosen"], "primary": primary.key,
            "secondary": secondary.key, "secondary_label": st.SECONDARY_LABEL,
            "layer_map": {str(chosen.layer): primary.layer, str(L2): secondary.layer}, "n_layers": n_layers}
    return primary, secondary, info


def write_settings(run_dir: Path, info: dict, backend: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(info, indent=1))
    if backend == "stub":   # the report reads the design's choice; the stub's layers differ, so a stub copy says so
        design = json.loads(Path(info["design"]).read_text())
        design["chosen"] = info["primary"]
        design["stub_note"] = f"stub copy of {info['design']}: {info['design_chosen']} mapped to {info['primary']}"
        (run_dir / "design_stub.json").write_text(json.dumps(design, indent=1))
        info_stub = dict(info, design_chosen=info["primary"])
        (run_dir / "settings.json").write_text(json.dumps(info_stub, indent=1))


def cmd_run(args) -> int:
    t_start = time.perf_counter()
    d, pop, counts = load_all(Path(args.exp35), args.runs, Path(args.own_root), check=not args.no_count_check)
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    load_s = time.perf_counter() - t0
    log(f"engine {engine.name}, 3B loaded in {load_s:.0f} s")
    primary, secondary, info = settings_from_design(Path(args.design), engine.n_layers)
    run = PodRun(engine, d, pop, Path(args.out), primary, secondary, args.max_new_tokens, args.deadline,
                 args.judge_reserve)
    write_settings(run.dir, info, args.backend)
    run.stage_s["load_3b"] = load_s
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run, pop.R, pop.C))
    run.check_manifest()
    log(f"primary {primary.key} (from {args.design}); secondary {secondary.key} ({st.SECONDARY_LABEL})")

    def stage(name, t):
        run.stage_s[name] = time.perf_counter() - t
        run.write_json("timing_pod.json", {"stage_s": run.stage_s, "forward_passes": run.forward_passes})

    log("== (i) reproduction (all folds)")
    t = time.perf_counter()
    rep = run.reproduction()
    run.write_json("repro.json", rep)
    stage("repro", t)
    log(f"reproduction: R flags {rep['rate_R']:.3f} ({len(rep['R_ok'])}/{len(rep['R'])}); C rows where A answers "
        f"{len(rep['C_answers'])}/{len(rep['C'])}; preliminary {rep['rate_preliminary']:.3f} (bar {st.REPRO_GATE})")
    if not rep["pass_preliminary"]:
        log(f"reproduction below {st.REPRO_GATE}: the engine is not fit, §38 stops here (exit {EXIT_REPRO})")
        return EXIT_REPRO
    pairs = [q for q in pop.pairs if run.fold(q) in DEV and q in rep["R_ok"]]
    R = [q for q in rep["R_ok"] if run.fold(q) in CONFIRM]
    C = [q for q in rep["C_answers"] if run.fold(q) in CONFIRM]

    log(f"== (ii) directions on {len(pairs)} reproduced dev pairs")
    t = time.perf_counter()
    for L, s in run.pod_directions(pairs).items():
        log(f"  L{L}: ||v|| {s['v_norm']:.3f}, mean||resid_A|| {s['mean_resid_A_norm']:.3f}, ratio "
            f"{s['v_over_resid']:.4f}" + (f", cosine with laptop {s['cosine_with_laptop']:.3f}" if "cosine_with_laptop" in s else ""))
    stage("directions", t)

    log("== (iii) sanity S")
    t = time.perf_counter()
    san = run.sanity(pairs)
    run.write_json("sanity.json", san)
    stage("sanity", t)
    log(f"final-layer overwrite {san['final_overwrite_rate']:.3f} (>= {st.SANITY_FINAL_GATE}); zero add exact "
        f"{san['zero_exact_all']}; self-overwrite exact {san['self_exact_all']}; 0.2 depth {san['shallow_overwrite_rate']:.3f}")
    if not san["pass"]:
        log(f"sanity S failed: the hooks are wrong, nothing is read (exit {EXIT_SANITY})")
        return EXIT_SANITY

    log(f"== (iv) arms on {len(R)} confirmation R rows and {len(C)} C rows; {run.time_left():.0f} s left before the "
        "judge reserve")
    t = time.perf_counter()
    run.arms(R, C)
    stage("arms", t)
    run.write_json("passes.json", run.forward_passes)
    run.stage_s["total"] = time.perf_counter() - t_start
    run.write_json("timing_pod.json", {"stage_s": run.stage_s, "forward_passes": run.forward_passes})
    log(f"run done in {run.stage_s['total']:.0f} s; skipped {[x['arm'] for x in run.plan['skipped']]}; "
        f"cut {run.plan['cut']}; forward passes {run.forward_passes}")
    return 0


# --- the judge ----------------------------------------------------------------------------------------------------

class StubJudge:
    """CPU stand-in for the 7B judge. 'd1': 'yes' when the response passes the §35 D1 string rule against the
    correct answer in the prompt, else 'no'. 'silent': an empty reply (no verdict; the smoke must abort)."""

    def __init__(self, mode: str = "d1") -> None:
        self.mode, self.model, self.name = mode, "stub-judge", f"stub-judge-{mode}"

    @staticmethod
    def _field(prompt: str, head: str, tail: str) -> str:
        return prompt.split(head, 1)[-1].split(tail, 1)[0]

    def generate(self, items: list[tuple[str, str]], max_new_tokens: int) -> list[str]:
        out = []
        for prompt, _ in items:
            if self.mode == "silent":
                out.append("")
                continue
            head = "\n\nRubric: " if "\n\nRubric: " in prompt else "\n\nCorrect Answer: "
            gold = self._field(prompt, head, "\n\nModel Response: ")
            answer = self._field(prompt, "\n\nModel Response: ", "\n\nIs the model response correct?")
            out.append("yes" if sel.d1(answer, gold) is True else "no")
        return out


def build_judge(backend: str, stub_mode: str = "d1"):
    if backend == "stub":
        return StubJudge(stub_mode)
    from scripts.exp35_generate import HFEngine
    eng = HFEngine(JUDGE_MODEL, batch_size=JUDGE_BATCH)
    dtype = str(eng.net.dtype).split(".")[-1]
    eng.name = f"{eng.name}-{eng.device}-{dtype}"     # e.g. hf-transformers-5.17.0-cuda-bfloat16
    return eng


def judge_settings_of(engine) -> dict:
    return {**sel.judge_settings(engine.name), "max_new_tokens": sel.JUDGE_MAX_NEW_TOKENS, "chat": "wrap_chat",
            "sampling": sel.SAMPLING}


PRIORITY = ("repro_confirm", "primary", "secondary", "cached35", "repro_dev")


def judge_items(run_dir: Path, d, primary_layer: int, secondary_layer: int, confirm_r: list[str],
                confirm_c: list[str]) -> list[dict]:
    """Every output to judge, in priority order."""
    items = []

    def tier(r: dict) -> str:
        if r["stage"] == "repro":
            return "repro_dev" if r["fold"] in DEV else "repro_confirm"
        return "primary" if r["setting"]["layer"] in (None, primary_layer) else "secondary"

    for stage in ("repro", "arms"):
        for r in read_jsonl(run_dir / f"{stage}.jsonl"):
            if r["pop"] not in ("R", "C"):
                continue
            items.append({"pop": r["pop"], "kind": r["kind"], "setting_key": r["setting"]["key"],
                          "question_id": r["question_id"], "fold": r["fold"], "answer": r["final_answer"],
                          "row_engine": r["engine"], "row_input_sha256": r["input_sha256"], "tier": tier(r)})
    for q in confirm_r:
        for kind in ("A", "B"):
            items.append({"pop": "cached35", "kind": kind, "setting_key": "cached", "question_id": q,
                          "fold": d.by_q[q].fold, "answer": d.rows[kind][q]["final_answer"], "row_engine": "§35 cache",
                          "row_input_sha256": None, "tier": "cached35"})
    for q in confirm_c:
        items.append({"pop": "cached35", "kind": "A", "setting_key": "cached", "question_id": q, "fold": d.by_q[q].fold,
                      "answer": d.rows["A"][q]["final_answer"], "row_engine": "§35 cache", "row_input_sha256": None,
                      "tier": "cached35"})
    items.sort(key=lambda it: PRIORITY.index(it["tier"]))
    return items


def run_judge(run_dir: Path, engine, d, items: list[dict], questions: dict[str, str], deadline: float | None) -> dict:
    """Judge `items` (resumable). Returns counts; writes judge.jsonl, judge_replies.jsonl, judge_manifest.json."""
    from scripts.exp35_generate import append_jsonl
    settings = judge_settings_of(engine)
    mpath = run_dir / "judge_manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old.get("judge_engine") != engine.name or old.get("judge_model") != engine.model:
            raise SystemExit(f"REFUSED: {mpath} has judge {old.get('judge_engine')!r}/{old.get('judge_model')!r}, this "
                             f"run {engine.name!r}/{engine.model!r}; one judge engine per run directory")
    rpath = run_dir / "judge_replies.jsonl"
    replies = {r["prompt_sha256"]: r["reply"] for r in read_jsonl(rpath) if r.get("judge_engine") == engine.name}
    for it in items:
        rec = d.by_q[it["question_id"]]
        it["judge_prompt"] = sel.judge_prompt(questions[it["question_id"]], rec.gold, it["answer"], rec.question_type)
        it["judge_prompt_sha256"] = sel.sha256(it["judge_prompt"])
    pending: dict[str, str] = {}
    for it in items:
        if it["judge_prompt_sha256"] not in replies:
            pending.setdefault(it["judge_prompt_sha256"], it["judge_prompt"])
    todo = list(pending.items())
    log(f"judge {engine.name}: {len(items)} items, {len(set(i['judge_prompt_sha256'] for i in items))} distinct prompts, "
        f"{len(todo)} not yet judged")
    t0 = time.perf_counter()
    cut = False

    def write_rows() -> list[dict]:
        rows = []
        for it in items:
            reply = replies.get(it["judge_prompt_sha256"])
            if reply is None:
                continue
            rows.append({k: it[k] for k in ("pop", "kind", "setting_key", "question_id", "fold", "answer", "row_engine",
                                             "row_input_sha256", "judge_prompt_sha256", "tier")}
                        | {"reply": reply, "parseable": st.parse_verdict(reply) is not None,
                           "verdict": sel.verdict_of(reply), "judge_engine": engine.name, "judge_model": engine.model,
                           "judge_settings": settings})
        write_jsonl(run_dir / "judge.jsonl", rows)
        return rows

    for start in range(0, len(todo), JUDGE_CHUNK):
        if deadline is not None and time.time() > deadline:
            cut = True
            log(f"judge: CUT by the deadline after {start}/{len(todo)} prompts")
            break
        chunk = todo[start:start + JUDGE_CHUNK]
        out = engine.generate([(p, "") for _, p in chunk], sel.JUDGE_MAX_NEW_TOKENS)
        new = [{"prompt_sha256": h, "reply": o.strip(), "judge_engine": engine.name} for (h, _), o in zip(chunk, out)]
        append_jsonl(rpath, new)
        replies.update({r["prompt_sha256"]: r["reply"] for r in new})
        write_rows()
        log(f"judge: {start + len(chunk)}/{len(todo)} prompts ({(time.perf_counter() - t0) / (start + len(chunk)):.3f} s/prompt)")
    rows = write_rows()
    secs = time.perf_counter() - t0
    unparseable = sum(not r["parseable"] for r in rows)
    manifest = {"judge_engine": engine.name, "judge_model": engine.model, "role": "judge", "settings": settings,
                "n_items": len(items), "n_judged": len(rows), "n_prompts_called": len(todo) if not cut else None,
                "seconds": secs, "unparseable": unparseable, "cut": cut,
                "not_judged": "sanity rows (pop 'pairs'): plumbing, never a gold figure"}
    mpath.write_text(json.dumps(manifest, indent=1))
    log(f"judge: {len(rows)}/{len(items)} items judged in {secs:.0f} s; replies that are not verdicts: {unparseable}")
    return manifest


def questions_for(qids: list[str], backend: str) -> dict[str, str]:
    from scripts.exp35_generate import judge_questions
    return judge_questions(sorted(set(qids)))


def cmd_judge(args) -> int:
    d, pop, _ = load_all(Path(args.exp35), args.runs, Path(args.own_root), check=not args.no_count_check)
    run_dir = Path(args.out) / READER
    info = json.loads((run_dir / "settings.json").read_text())
    p, s = st.setting_from_key(info["primary"]), st.setting_from_key(info["secondary"])
    conf_r = [q for q in pop.R if d.by_q[q].fold in CONFIRM]
    conf_c = [q for q in pop.C if d.by_q[q].fold in CONFIRM]
    items = judge_items(run_dir, d, p.layer, s.layer, conf_r, conf_c)
    t0 = time.perf_counter()
    engine = build_judge(args.backend, args.stub_judge)
    load_s = time.perf_counter() - t0
    log(f"judge engine {engine.name} ({engine.model}) loaded in {load_s:.0f} s")
    manifest = run_judge(run_dir, engine, d, items, questions_for([i["question_id"] for i in items], args.backend),
                         args.deadline)
    manifest["load_s"] = load_s
    (run_dir / "judge_manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


# --- the D-22 smoke -----------------------------------------------------------------------------------------------

def unload(engine) -> None:
    torch = getattr(engine, "torch", None)
    for attr in ("net", "layers", "tokenizer"):
        if hasattr(engine, attr):
            setattr(engine, attr, None)
    del engine
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_smoke(args) -> int:
    d, pop, _ = load_all(Path(args.exp35), args.runs, Path(args.own_root), check=not args.no_count_check)
    out = Path(args.out) / "smoke"
    qs = [q for q in pop.pairs if d.by_q[q].fold in DEV][:SMOKE_N]
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    load_s = time.perf_counter() - t0
    primary, secondary, info = settings_from_design(Path(args.design), engine.n_layers)
    run = PodRun(engine, d, pop, out, primary, secondary, args.max_new_tokens, None, 0.0)
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run, qs, []))
    run.check_manifest()
    problems = []
    none = st.Setting("none")
    t1 = time.perf_counter()
    flags = {}
    for q in qs:
        a = run.do("repro", "R", "A", q, [none], capture=True)["none"]
        b = run.do("repro", "R", "B", q, [none], capture=True)["none"]
        flags[q] = {"A": st.reproduces(a, d.rows["A"][q], d.by_q[q].gold)["ok"],
                    "B": st.reproduces(b, d.rows["B"][q], d.by_q[q].gold)["ok"]}
        for r in (a, b):
            if r["last_token"] != ":":
                problems.append(f"{q}/{r['kind']}: last prompt token {r['last_token']!r}, not ':'")
    repro_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    ra = {q: run.resid("A", q) for q in qs}
    rb = {q: run.resid("B", q) for q in qs}
    v = st.estimate_direction(ra, rb, qs, primary.layer)
    run.vectors[(primary.layer, primary.alpha, primary.unit)] = st.scaled_vector(v, primary.alpha, primary.unit)
    steer = run.do("arms", "R", "A", qs[0], [primary])[primary.key]
    steer_s = time.perf_counter() - t2
    log(f"smoke: reproduction flags {flags}; one Steer at {primary.key} on {qs[0]}: {steer['cls']} "
        f"({steer['timing']['n_new']} tokens) {steer['phrase']!r:.60}")
    rows = list(run.rows("repro").values()) + [steer]
    decode = [r["timing"] for r in rows if r["timing"]["n_new"] > 1]
    rate = {"prefill_s_per_token": sum(r["timing"]["prefill_s"] for r in rows) / sum(r["n_prefix_tokens"] for r in rows),
            "decode_s_per_token": (sum(t["decode_s"] for t in decode) / sum(t["n_new"] - 1 for t in decode)) if decode else None,
            "first_step_s": sum(r["timing"]["first_step_s"] for r in rows) / len(rows)}
    unload(engine)
    t3 = time.perf_counter()
    judge = build_judge(args.backend, args.stub_judge)
    judge_load_s = time.perf_counter() - t3
    t4 = time.perf_counter()
    questions = questions_for(qs, args.backend)
    prompts = [sel.judge_prompt(questions[r["question_id"]], d.by_q[r["question_id"]].gold, r["final_answer"],
                                d.by_q[r["question_id"]].question_type) for r in rows]
    replies = [x.strip() for x in judge.generate([(p, "") for p in prompts], sel.JUDGE_MAX_NEW_TOKENS)]
    judge_s = time.perf_counter() - t4
    verdicts = [st.parse_verdict(x) for x in replies]
    for r, x, v_ in zip(rows, replies, verdicts):
        log(f"  judge {r['kind']} {r['setting']['key']} {r['question_id']}: reply {x!r} -> {v_}")
        if v_ is None:
            problems.append(f"judge reply {x!r} on {r['kind']}/{r['setting']['key']}/{r['question_id']} is not a verdict")
    record = {"rows": qs, "flags": flags, "steer": {"key": primary.key, "cls": steer["cls"], "n_new": steer["timing"]["n_new"]},
              "replies": replies, "verdicts": verdicts, "problems": problems, "engine": run.engine.name,
              "judge_engine": judge.name, "load_3b_s": load_s, "repro_s": repro_s, "steer_s": steer_s,
              "load_judge_s": judge_load_s, "judge_s": judge_s, "judge_s_per_prompt": judge_s / max(len(prompts), 1),
              "rates": rate, "projection": projection(rate, judge_s / max(len(prompts), 1), d, pop)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(record, indent=1, default=float))
    pj = record["projection"]
    log(f"smoke rates: prefill {rate['prefill_s_per_token'] * 1000:.2f} ms/token, decode "
        f"{(rate['decode_s_per_token'] or 0) * 1000:.1f} ms/token; judge {record['judge_s_per_prompt']:.3f} s/prompt; "
        f"projected run (all steered outputs at the cap) {pj['minutes_cap']:.1f} min, judge {pj['judge_minutes']:.1f} min")
    if problems:
        log(f"SMOKE FAILED: {problems} (exit {EXIT_SMOKE}); the chain stops before the full run")
        return EXIT_SMOKE
    log("smoke PASS: every smoke output got a yes/no verdict")
    return 0


def projection(rate: dict, judge_s_per_prompt: float, d, pop, max_new: int = st.MAX_NEW_TOKENS_POD,
               prefill_tokens: int = 1100, unsteered_tokens: int = 6) -> dict:
    """Minutes for the run at measured rates, every steered output at the cap (conservative)."""
    rp = rate["prefill_s_per_token"] * prefill_tokens
    rd = rate["decode_s_per_token"] or rate["first_step_s"]
    gen = lambda n: rate["first_step_s"] + max(n - 1, 0) * rd  # noqa: E731
    nR, nC = len(pop.R), len(pop.C)
    cR = sum(d.by_q[q].fold in CONFIRM for q in pop.R)
    cC = sum(d.by_q[q].fold in CONFIRM for q in pop.C)
    repro = (2 * nR + nC) * (rp + gen(unsteered_tokens))
    sanity = 13 * rp + 13 * 6 * gen(unsteered_tokens)
    primary = 2 * cR * rp + cC * rp + (3 * cR + cR + cC) * gen(max_new) + cR * gen(unsteered_tokens)
    secondary = (4 * cR + cC) * rp + (4 * cR + cC) * gen(max_new)
    judge_items_n = 2 * nR + nC + (5 * cR + cC) + (4 * cR + cC) + 2 * cR + cC
    return {"minutes_repro": repro / 60, "minutes_sanity": sanity / 60, "minutes_primary": primary / 60,
            "minutes_secondary": secondary / 60, "minutes_cap": (repro + sanity + primary + secondary) / 60,
            "judge_items": judge_items_n, "judge_minutes": judge_items_n * judge_s_per_prompt / 60}


# --- main ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "run", "judge"))
    parser.add_argument("--out", default="runs/exp38_pod")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--design", default=DEFAULT_DESIGN, help="the laptop gate.json fixing the primary setting")
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    parser.add_argument("--stub-judge", default="d1", choices=("d1", "silent"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=st.MAX_NEW_TOKENS_POD)
    parser.add_argument("--deadline", type=float, default=None, help="epoch seconds: all model work ends by then")
    parser.add_argument("--judge-reserve", type=float, default=420.0,
                        help="seconds kept free before --deadline for the 7B load and the judge pass")
    parser.add_argument("--no-count-check", action="store_true", help="tests only: skip the population count check")
    args = parser.parse_args(argv)
    try:
        return {"smoke": cmd_smoke, "run": cmd_run, "judge": cmd_judge}[args.command](args)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        raise


if __name__ == "__main__":
    raise SystemExit(main())
