"""§38 laptop design run (CPU, transformers fp32, the 3B, folds 0-1 only).

    .venv/bin/python scripts/exp38_dev.py                                   # plan only: populations, token counts
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp38_dev.py --timing-only
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp38_dev.py --run   # after the build review
    .venv/bin/python scripts/exp38_dev.py --run --backend stub --out runs/exp38_stub      # CPU stub, end to end

Pre-registration: research/proposals/2026-10-22-exp38-refusal-direction.md (rev 1, bb758c3; amendment 1, c367d28), §5.
Logic in memctl/analysis/steering.py. Without --run and --timing-only the script loads the cache, prints the
populations and the exact input lengths and stops (no model). Stages with --run (each resumable, rows sha-stamped):

  (ii)  5-row timing first (unless <out>/qwen3b/timing.json exists): prefill s/token, per-step s/token, cache copy s;
        projection for the whole run. Over 5 h on the full grid -> the small grid (depths 0.6, 0.8, 0.9; alpha
        --alphas-small), §5.6. --timing-only stops here.
  (i)   reproduction: unsteered A and B on the R rows, A on the damage rows (all 36 layers' residuals at the ':'
        position saved); refusal flag (and the D1 string verdict when answering) against the §35 cache; failures are
        listed and dropped; below 0.8 -> stop (exit 3).
  (iii) direction v_L on the reproduced pairs at each sweep layer; norms and ||v_L|| / mean||resid_A|| printed.
  (iv)  sanity S on the pairs: overwrite with B's state at the final layer (first token == B's, >= 0.9) and at 0.2
        depth (descriptive); zero-vector add at every sweep layer and self-overwrite at both layers (byte-exact A).
        Failure -> stop (exit 4).
  (v)   Steer on R for every (L, alpha); Random at the candidate settings in gate order until one is eligible; then
        at the chosen setting Reverse on R, Steer-cut and A-cut on R, Steer / Random / Reverse on the damage rows.
        The design gate (gate.json) is the report's to read; the run prints its arithmetic.

Folds other than 0 and 1 are refused. Output: runs/exp38_dev/qwen3b/{manifest,timing,directions,gate}.json,
{repro,sanity,sweep,controls}.jsonl, resid/{A,B}/<qid>.npy. HF_TOKEN is never printed (the model is cached locally).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import MODELS, append_jsonl, read_jsonl  # noqa: E402

DEV = (0, 1)
READER = "qwen3b"
STAGES = ("repro", "sanity", "sweep", "controls")
EXIT_REPRO, EXIT_SANITY = 3, 4


def load_source(folds, root: Path, runs: str, own_root: Path):
    from scripts.exp35_report import load
    data, _, _ = load(set(folds), root, runs, own_root)
    d = data[READER]
    return d, st.populations(d)


class Run:
    def __init__(self, engine, d, pop, out: Path, layers, alphas, unit: str, max_new: int, log=print) -> None:
        self.engine, self.d, self.pop, self.log = engine, d, pop, log
        self.dir = out / READER
        self.layers, self.alphas, self.unit, self.max_new = list(layers), list(alphas), unit, max_new
        self.wsha = st.wording_sha256(max_new)
        self.final = st.final_layer(engine.n_layers)
        self.shallow = st.depth_layer(st.SHALLOW_DEPTH, engine.n_layers)
        self.vectors: dict = {}
        self.stats: dict = {}
        self.forward_passes = {"prefills": 0, "generations": 0, "tokens_generated": 0}
        self._rows: dict[str, dict] = {}     # stage -> {row key: row}, read once, kept in step with the file

    # --- files -----------------------------------------------------------------------------------------------
    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.jsonl"

    def rows(self, stage: str) -> dict:
        if stage not in self._rows:
            self._rows[stage] = {st.row_key(r): r for r in read_jsonl(self.path(stage))}
        return dict(self._rows[stage])

    def log_rows(self, stage: str, rows: list[dict]) -> None:
        for r in rows:
            self.log(f"  [{stage} {r['pop']} {r['kind']} {r['setting']['key']}] {r['question_id']} {r['cls']:10s} {r['phrase']!r:.50}")

    def resid_path(self, kind: str, q: str) -> Path:
        return self.dir / "resid" / kind / f"{q}.npy"

    def resid(self, kind: str, q: str) -> np.ndarray:
        return np.load(self.resid_path(kind, q))

    def write_json(self, name: str, obj) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=float))

    def check_manifest(self) -> None:
        manifest = {"engine": self.engine.name, "model": self.engine.model, "layers": self.layers, "alphas": self.alphas,
                    "unit": self.unit, "max_new_tokens": self.max_new, "wording_sha256": self.wsha,
                    "wording": st.wording(self.max_new), "folds": sorted({self.d.by_q[q].fold for q in self.pop.R})}
        path = self.dir / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            for k in ("engine", "model", "layers", "alphas", "unit", "max_new_tokens", "wording_sha256"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"REFUSED: {path} has {k} = {old.get(k)!r}, this run {manifest[k]!r}; one engine "
                                     "and one grid per run directory (move it aside)")
        self.write_json("manifest.json", manifest)

    # --- inputs ----------------------------------------------------------------------------------------------
    def item(self, kind: str, q: str) -> st.SteerInput:
        rec = self.d.by_q[q]
        if kind == "cut":
            return st.cut_input(rec, self.d.rows["Ncut"][q]["generated"])
        return st.arm_input(rec, kind)

    # --- one prompt, several settings -----------------------------------------------------------------------
    def do(self, stage: str, pop: str, kind: str, q: str, settings: list[st.Setting], capture: bool = False,
           own_state=None, b_state=None) -> dict:
        """Run the pending settings for one prompt from one shared prefix; returns {setting key: row}."""
        item = self.item(kind, q)
        done = self.rows(stage)
        out, todo = {}, []
        for s in settings:
            vec = st.hook_vector(s, self.vectors, own_state, b_state, self.engine.hidden)
            old = done.get((pop, kind, s.key, q))
            if old is not None and (not capture or self.resid_path(kind, q).exists()):
                st.check_row(old, item, self.engine.name, self.wsha, vec)
                out[s.key] = old
            else:
                todo.append((s, vec))
        if not todo:
            return out
        prepared = self.engine.prepare(item)
        self.forward_passes["prefills"] += 1
        if not self.engine.last_token_text(prepared).endswith(":"):
            self.log(f"  WARNING {kind}/{q}: last prompt token is {self.engine.last_token_text(prepared)!r}, not ':'")
        gold = self.d.by_q[q].gold
        new = []
        for s, vec in todo:
            specs = st.specs_for(s, self.vectors, own_state, b_state, self.engine.hidden)
            gen = self.engine.run(prepared, specs, self.max_new,
                                  capture=tuple(range(self.engine.n_layers)) if capture else ())
            self.forward_passes["generations"] += 1
            self.forward_passes["tokens_generated"] += gen.n_new
            if capture:
                arr = np.stack([gen.captured[L] for L in range(self.engine.n_layers)]).astype(np.float32)
                self.resid_path(kind, q).parent.mkdir(parents=True, exist_ok=True)
                np.save(self.resid_path(kind, q), arr)
            row = st.make_row(stage, pop, item, s, gen, self.engine, prepared, vec, self.max_new, gold, READER)
            new.append(row)
            out[s.key] = row
        append_jsonl(self.path(stage), new)
        self.rows(stage)
        self._rows[stage].update({st.row_key(r): r for r in new})
        self.log_rows(stage, new)
        return out

    # --- stage i ---------------------------------------------------------------------------------------------
    def reproduction(self) -> dict:
        none = st.Setting("none")
        result = {"R": {}, "damage": {}}
        for q in self.pop.R:
            a = self.do("repro", "R", "A", q, [none], capture=True)["none"]
            b = self.do("repro", "R", "B", q, [none], capture=True)["none"]
            ra = st.reproduces(a, self.d.rows["A"][q], self.d.by_q[q].gold)
            rb = st.reproduces(b, self.d.rows["B"][q], self.d.by_q[q].gold)
            result["R"][q] = {"A": ra, "B": rb, "ok": ra["ok"] and rb["ok"]}
        for q in self.pop.damage:
            a = self.do("repro", "C", "A", q, [none], capture=True)["none"]
            ra = st.reproduces_damage(a, self.d.rows["A"][q], self.d.by_q[q].gold)
            result["damage"][q] = {"A": ra, "ok": ra["ok"]}
        return summarise_repro(result)

    # --- stage iii -------------------------------------------------------------------------------------------
    def directions(self, pairs: list[str]) -> dict:
        ra = {q: self.resid("A", q) for q in pairs}
        rb = {q: self.resid("B", q) for q in pairs}
        out, arrays = {}, {}
        for L in self.layers:
            v = st.estimate_direction(ra, rb, pairs, L)
            stats = st.direction_stats(v, ra, pairs, L)
            out[str(L)] = stats
            arrays[f"L{L}"] = v
            for a in self.alphas:
                self.vectors[(L, float(a), self.unit)] = st.scaled_vector(v, float(a), self.unit, stats["mean_resid_A_norm"])
        self.stats = out
        np.savez(self.dir / "directions.npz", **arrays)
        self.write_json("directions.json", {"pairs": pairs, "layers": out})
        return out

    # --- stage iv --------------------------------------------------------------------------------------------
    def sanity(self, pairs: list[str]) -> dict:
        repro = self.rows("repro")
        fin, zero, self_ex, shallow = [], [], [], []
        per_row = {}
        for q in pairs:
            ra, rb = self.resid("A", q), self.resid("B", q)
            a0, b0 = repro[("R", "A", "none", q)], repro[("R", "B", "none", q)]
            settings = ([st.Setting("overwrite_B", self.final), st.Setting("overwrite_B", self.shallow),
                         st.Setting("overwrite_self", self.final), st.Setting("overwrite_self", self.shallow)]
                        + [st.Setting("zero", L) for L in self.layers])
            rows = self.do("sanity", "pairs", "A", q, settings, own_state=ra, b_state=rb)
            same = lambda r: r["generated"] == a0["generated"] and r["gen_ids"] == a0["gen_ids"]  # noqa: E731
            fin.append(rows[f"overwrite_B:L{self.final}"]["first_token"] == b0["first_token"])
            shallow.append(rows[f"overwrite_B:L{self.shallow}"]["first_token"] == b0["first_token"])
            self_ex += [same(rows[f"overwrite_self:L{self.final}"]), same(rows[f"overwrite_self:L{self.shallow}"])]
            zero += [same(rows[f"zero:L{L}"]) for L in self.layers]
            per_row[q] = {"final": fin[-1], "shallow": shallow[-1], "self": self_ex[-2:], "zero": zero[-len(self.layers):]}
        out = st.sanity_summary(fin, zero, self_ex, shallow)
        out["final_layer"], out["shallow_layer"], out["rows"] = self.final, self.shallow, per_row
        return out

    # --- stage v ---------------------------------------------------------------------------------------------
    def steer_stats(self, R: list[str]) -> dict:
        sweep = self.rows("sweep")
        out = {}
        for s in st.sweep_settings(self.layers, self.alphas, self.unit):
            rows = [sweep[("R", "A", s.key, q)] for q in R if ("R", "A", s.key, q) in sweep]
            out[s.key] = {"layer": s.layer, "alpha": s.alpha, **st.rates(rows)}
        return out

    def random_ans(self, R: list[str]) -> dict:
        sweep = self.rows("sweep")
        out = {}
        for s in st.sweep_settings(self.layers, self.alphas, self.unit):
            r = st.Setting("random", s.layer, s.alpha, s.unit)
            rows = [sweep[("R", "A", r.key, q)] for q in R if ("R", "A", r.key, q) in sweep]
            if len(rows) == len(R) and R:
                out[s.key] = st.rates(rows)["ans"]
        return out

    def sweep(self, R: list[str], sanity_pass: bool, repro_rate: float) -> dict:
        settings = st.sweep_settings(self.layers, self.alphas, self.unit)
        for q in R:
            self.do("sweep", "R", "A", q, settings)
        while True:
            gate = st.design_gate(self.steer_stats(R), self.random_ans(R), sanity_pass, repro_rate)
            if gate["status"] != "need_random":
                break
            s = next(x for x in settings if x.key == gate["need"])
            self.log(f"Random at candidate {s.key}")
            for q in R:
                self.do("sweep", "R", "A", q, [st.Setting("random", s.layer, s.alpha, s.unit)])
        return gate

    def controls(self, R: list[str], damage: list[str], chosen: st.Setting) -> None:
        rnd = st.Setting("random", chosen.layer, chosen.alpha, chosen.unit)
        rev = st.Setting("reverse", chosen.layer, chosen.alpha, chosen.unit)
        for q in R:
            self.do("controls", "R", "A", q, [rev])
            self.do("controls", "R", "cut", q, [st.Setting("none"), chosen])
        for q in damage:
            self.do("controls", "C", "A", q, [chosen, rnd, rev])


def stub_oracle(run: Run) -> dict[str, str]:
    """For the stub only: the cached §35 A and B generations, keyed by input text, so the stub reproduces the cache."""
    texts = {}
    for q in sorted(set(run.pop.R) | set(run.pop.damage)):
        for kind in ("A", "B"):
            item = run.item(kind, q)
            texts[st.sha256(run.engine.chat_text(item.prompt) + item.continuation)] = run.d.rows[kind][q]["generated"]
    return texts


def summarise_repro(result: dict) -> dict:
    ok_r = [q for q, v in result["R"].items() if v["ok"]]
    ok_c = [q for q, v in result["damage"].items() if v["ok"]]
    n = len(result["R"]) + len(result["damage"])
    result["R_ok"], result["damage_ok"] = ok_r, ok_c
    result["dropped"] = sorted(set(result["R"]) - set(ok_r)) + sorted(set(result["damage"]) - set(ok_c))
    result["rate"] = (len(ok_r) + len(ok_c)) / n if n else None
    result["rate_R"] = len(ok_r) / len(result["R"]) if result["R"] else None
    result["pass"] = result["rate"] is not None and result["rate"] >= st.REPRO_GATE
    # descriptive column (build review ruling 1): R rows whose A and B D1 verdicts also match the cache
    result["R_verdict_match"] = sum(v["A"]["verdict_match"] and v["B"]["verdict_match"] for v in result["R"].values())
    return result


# --- timing ---------------------------------------------------------------------------------------------------

def timing(run: Run) -> dict:
    """The first 5 R rows: prefill of A, unsteered A and a zero-add generation (a hooked run) from the same prefix."""
    recs = []
    for q in run.pop.R[:st.TIMING_N]:
        prepared = run.engine.prepare(run.item("A", q))
        g0 = run.engine.run(prepared, [], run.max_new)
        g1 = run.engine.run(prepared, [st.add_spec(run.layers[len(run.layers) // 2], np.zeros(run.engine.hidden, np.float32))],
                            run.max_new)
        for g in (g0, g1):
            recs.append({"n_prefix": prepared.n_prefix, "prefill_s": prepared.prefill_s, "copy_s": g.copy_s,
                         "first_step_s": g.first_step_s, "decode_s": g.decode_s, "n_new": g.n_new})
    pre = {(r["n_prefix"], r["prefill_s"]) for r in recs}
    rate_p = sum(s for _, s in pre) / sum(n for n, _ in pre)
    dec = [r for r in recs if r["n_new"] > 1]
    rate_d = (sum(r["decode_s"] for r in dec) / sum(r["n_new"] - 1 for r in dec)) if dec else \
        sum(r["first_step_s"] for r in recs) / len(recs)
    return {"rows": len(recs) // 2, "prefill_s_per_token": rate_p, "decode_s_per_token": rate_d,
            "first_step_s": sum(r["first_step_s"] for r in recs) / len(recs),
            "copy_s": sum(r["copy_s"] for r in recs) / len(recs),
            "mean_new_tokens_unsteered": sum(r["n_new"] for r in recs) / len(recs), "records": recs}


def projection(run: Run, t: dict, layers, alphas) -> dict:
    """Hours by stage from exact prefix lengths and the timed rates. A generation costs copy + first step + (n - 1)
    decode steps; unsteered and plumbing generations at the timed mean length, steered ones at max_new_tokens
    (conservative) and at the timed mean (optimistic). Random: one candidate (k = 1) and every setting (k = all)."""
    rp, rd, f, c = t["prefill_s_per_token"], t["decode_s_per_token"], t["first_step_s"], t["copy_s"]
    gen = lambda n: c + f + max(n - 1, 0) * rd  # noqa: E731
    g_mean, g_max = gen(t["mean_new_tokens_unsteered"]), gen(run.max_new)
    plen = lambda kind, qs: sum(len(run.engine.input_ids(run.item(kind, q))) - 1 for q in qs)  # noqa: E731
    R, pairs, dmg = run.pop.R, run.pop.pairs, run.pop.damage
    a_R, b_R, a_C, cut_R = plen("A", R) * rp, plen("B", R) * rp, plen("A", dmg) * rp, plen("cut", R) * rp
    a_P = plen("A", pairs) * rp
    n_set = len(layers) * len(alphas)
    secs = {
        "repro": a_R + b_R + a_C + (2 * len(R) + len(dmg)) * g_mean,
        "sanity": a_P + len(pairs) * (4 + len(layers)) * g_mean,
        "sweep": a_R + len(R) * n_set * g_max,
        "random_k1": a_R + len(R) * g_max,
        "controls": a_R + cut_R + a_C + len(R) * 2 * g_max + len(R) * g_mean + len(dmg) * 3 * g_max,
    }
    total = sum(secs.values())
    worst = total + (n_set - 1) * secs["random_k1"]
    optimistic = total - (len(R) * n_set + len(R) + len(R) * 2 + len(dmg) * 3) * (g_max - g_mean)
    return {"hours_by_stage": {k: v / 3600 for k, v in secs.items()}, "total_hours": total / 3600,
            "total_hours_random_all_settings": worst / 3600, "total_hours_optimistic": optimistic / 3600,
            "settings": n_set, "layers": list(layers), "alphas": list(alphas),
            "over_limit": total / 3600 > st.PROJECTION_LIMIT_H}


# --- main -----------------------------------------------------------------------------------------------------

def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp38_dev")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=st.MAX_NEW_TOKENS_LAPTOP)
    parser.add_argument("--alphas", default="1,2,4")
    parser.add_argument("--alphas-small", default="4", help="alpha grid of the §5.6 fallback")
    parser.add_argument("--unit", default=st.DEFAULT_UNIT, choices=st.UNITS)
    parser.add_argument("--grid-small", action="store_true", help="the §5.6 fallback grid from the start")
    parser.add_argument("--timing-only", action="store_true", help="5-row timing and projection, then stop")
    parser.add_argument("--run", action="store_true", help="run every stage (only after the Advisor's build review)")
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    if not set(folds) <= set(DEV):
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; the design run is folds 0-1 only")
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    d, pop = load_source(folds, Path(args.exp35), args.runs, Path(args.own_root))
    log(f"populations (folds {folds}): R {len(pop.R)} (B correct {sum(v == 'correct' for v in pop.b_outcome.values())}, "
        f"wrong {sum(v == 'wrong' for v in pop.b_outcome.values())}, refuses {sum(v == 'refusal' for v in pop.b_outcome.values())}); "
        f"pairs {len(pop.pairs)}; unseen {len(pop.unseen)}; C {len(pop.C)}; damage {len(pop.damage)}")
    full_layers = [st.depth_layer(x) for x in st.SWEEP_DEPTHS]
    small_layers = [st.depth_layer(x) for x in st.SMALL_DEPTHS]
    if not (args.run or args.timing_only):
        log("plan only (no model): pass --timing-only for the 5-row timing, --run for the design run")
        return 0
    out = Path(args.out)
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    log(f"engine {engine.name}, model loaded in {time.perf_counter() - t0:.0f} s")
    layers = small_layers if args.grid_small else full_layers
    alphas = parse_floats(args.alphas_small if args.grid_small else args.alphas)
    if args.backend == "stub":   # the stub has few layers: map the depths onto it
        layers = sorted({st.depth_layer(x, engine.n_layers) for x in (st.SMALL_DEPTHS if args.grid_small else st.SWEEP_DEPTHS)})
    run = Run(engine, d, pop, out, layers, alphas, args.unit, args.max_new_tokens, log)
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run))
    tpath = run.dir / "timing.json"
    if tpath.exists():
        record = json.loads(tpath.read_text())
        if record["engine"] != engine.name:
            raise SystemExit(f"REFUSED: {tpath} was timed on {record['engine']!r}, this run uses {engine.name!r}")
        log(f"timing from {tpath}")
    else:
        t = timing(run)
        record = {"engine": engine.name, "threads": args.threads, "timing": t,
                  "projection_full": projection(run, t, full_layers if args.backend == "hf" else layers, parse_floats(args.alphas)),
                  "projection_small": projection(run, t, small_layers if args.backend == "hf" else layers,
                                                 parse_floats(args.alphas_small))}
        run.write_json("timing.json", record)
    pf = record["projection_full"]
    log(f"timing: prefill {record['timing']['prefill_s_per_token'] * 1000:.1f} ms/token, decode "
        f"{record['timing']['decode_s_per_token']:.2f} s/token, cache copy {record['timing']['copy_s']:.2f} s")
    log(f"projection, full grid ({pf['settings']} settings): {pf['total_hours']:.1f} h conservative "
        f"({pf['total_hours_optimistic']:.1f} h optimistic, {pf['total_hours_random_all_settings']:.1f} h if Random runs "
        f"at every setting); small grid {record['projection_small']['total_hours']:.1f} h; by stage "
        + ", ".join(f"{k} {v:.2f} h" for k, v in pf["hours_by_stage"].items()))
    if args.timing_only:
        log("Stopping (--timing-only).")
        return 0
    if pf["over_limit"] and not args.grid_small:
        log(f"full grid projects over {st.PROJECTION_LIMIT_H} h: the §5.6 fallback grid applies (layers {small_layers}, "
            f"alpha {args.alphas_small})")
        run.layers = small_layers if args.backend == "hf" else run.layers
        run.alphas = parse_floats(args.alphas_small)
    run.check_manifest()

    log("== (i) reproduction")
    rep = run.reproduction()
    run.write_json("repro.json", rep)
    log(f"reproduction {rep['rate']:.3f} (R {rep['rate_R']:.3f}; D1 verdict match {rep['R_verdict_match']}/{len(rep['R'])}); dropped {rep['dropped']}")
    if not rep["pass"]:
        log(f"reproduction below {st.REPRO_GATE}: the engine is not fit, §38 stops here (exit {EXIT_REPRO})")
        return EXIT_REPRO
    pairs = [q for q in pop.pairs if q in rep["R_ok"]]
    R = rep["R_ok"]
    damage = rep["damage_ok"]

    log("== (iii) directions")
    for L, s in run.directions(pairs).items():
        log(f"  L{L}: ||v|| {s['v_norm']:.3f}, mean||resid_A|| {s['mean_resid_A_norm']:.3f}, ratio {s['v_over_resid']:.4f} "
            f"({s['n_pairs']} pairs)")

    log("== (iv) sanity S")
    san = run.sanity(pairs)
    run.write_json("sanity.json", san)
    log(f"final-layer overwrite first-token rate {san['final_overwrite_rate']:.3f} (>= {st.SANITY_FINAL_GATE}); zero add "
        f"byte-exact {san['zero_exact_all']}; self-overwrite byte-exact {san['self_exact_all']}; 0.2-depth rate "
        f"{san['shallow_overwrite_rate']:.3f} (descriptive)")
    if not san["pass"]:
        log(f"sanity S failed: the hooks are wrong, nothing is read (exit {EXIT_SANITY})")
        return EXIT_SANITY

    log("== (v) sweep")
    gate = run.sweep(R, san["pass"], rep["rate"])
    gate["steer"] = run.steer_stats(R)
    gate["n_R"], gate["n_damage"], gate["n_pairs"] = len(R), len(damage), len(pairs)
    run.write_json("gate.json", gate)
    for e in gate["order"]:
        log(f"  {e['setting']}: ans {e['ans']:.3f}, degenerate {e['degenerate']:.3f} -> {e.get('why')}")
    if gate["chosen"] is None:
        log("design gate: no eligible setting; §38 closes (no controls run)")
        return 0
    c = gate["chosen_setting"]
    chosen = st.Setting("steer", c["layer"], c["alpha"], run.unit)
    log(f"chosen {chosen.key}; gate pass {gate['pass']}; controls at the chosen setting")
    run.controls(R, damage, chosen)
    run.write_json("passes.json", run.forward_passes)
    log(f"done; forward passes {run.forward_passes}. Report: scripts/exp38_report.py --run-dir {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
