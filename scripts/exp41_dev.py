"""§41 laptop development run: the signals at ':' and the Steer-all outputs for conditional steering (CPU, fp32, the
3B, folds 0-1).

    .venv/bin/python scripts/exp41_dev.py                                   # plan only: rows, id set, shas
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp41_dev.py --timing-only
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp41_dev.py --run   # after the build review
    .venv/bin/python scripts/exp41_dev.py --run --backend stub --source <stub §38> --out <dir>   # tests only

Pre-registration: research/proposals/2026-10-23-exp41-conditional-steering.md (rev 0; rev 1, 877f86d). Logic in
memctl/analysis/conditional.py (signals, labels, utility, fit) and steering.py (engines, rows).

Rows: 67 distinct development rows, R 28 (A refuses) and C 39 (A judged correct in §35); the 8 damage rows are the
first 8 of C and are not run twice. Nothing is re-estimated:

  direction  v_29 from <source>/directions.npz (the §38 dev file), byte-exact (sha asserted); the Steer-all vector is
             2 * v_29 (sha e0000366...45d71c, the §38 Steer-all rows' sha), asserted.
  id set     the refusal-start ids = first generated ids of every refusal-class row of the §38 dev A and B rows
             (<source>/repro.jsonl), printed and stamped on the manifest and on every prefill row.
  reuse      Steer-all on R = the §38 rows steer:L29:a2:vnorm of <source>/sweep.jsonl, refused unless engine,
             classifier, wording sha, vector sha, max new tokens 16 and input sha match (the §39 guard); copied to
             <out>/qwen3b/steer_all.jsonl with reuse.json.
  timing     5 C rows (unless timing.json exists): intact prefill, the unsteered pass with capture and logits, a
             Steer-all generation; projection by stage. --timing-only stops here.
  (i)        intact prompt on R: one unsteered generation each (step-0 logits returned and the layer-29 residual at
             ':' captured from the same pass); must equal the §38 dev A row byte-exact (text and ids) on every R row,
             else stop before any steered generation (exit 6). The captured h_29 against §38's saved resid_A[29] is
             printed (descriptive).
  (ii)       intact prompt on C: the unsteered pass as (i) and Steer-all (layer 29, alpha 2, vnorm, every step) from
             the same prefix cache. The unsteered final answer against the §35 cached A answer and the 8 damage rows
             against the §38 dev C rows (byte-exact) are printed.
  (iii)      cut prompt (§38's builder: N-cut view + N-cut note, no marker -> no note) on all 67: the unsteered pass
             as (i).

Rows are sha-stamped (input, wording, engine, vector, the id set) and the run resumes. Folds other than 0 and 1 are
refused (exit 2). Output: <out>/qwen3b/{manifest,timing,reuse,repro,passes}.json, {prefill,steer,steer_all}.jsonl,
resid/{A,cut}/<qid>.npy (layer 29 only). Report and the frozen rule: scripts/exp41_report.py. HF_TOKEN never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import conditional as cd  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import MODELS, append_jsonl, read_jsonl  # noqa: E402
from scripts.exp38_dev import load_source, stub_oracle  # noqa: E402
from scripts.exp39_dev import (ALPHA, DEV, LAYER, READER, STEER_VECTOR_SHA256, UNIT, Run as Run39,  # noqa: E402
                               file_sha256, load_direction)

EXIT_BYTES = 6
MAX_NEW = st.MAX_NEW_TOKENS_LAPTOP


def engine_v29(v: np.ndarray, engine) -> np.ndarray:
    """v_29 as the engine's residual sees it (the stub: its first `hidden` components)."""
    if engine.hidden == v.shape[0]:
        return v
    if engine.name != "stub":
        raise SystemExit(f"REFUSED: engine hidden size {engine.hidden} != {v.shape[0]} on a non-stub engine")
    return np.ascontiguousarray(v[:engine.hidden])


class Run(Run39):
    """§39's Run (files, Steer-all reuse, sha-stamped rows, resume) with the §41 stages."""

    def __init__(self, engine, d, pop, out: Path, source: Path, direction: dict, log=print) -> None:
        super().__init__(engine, d, pop, out, source, direction, max_new=MAX_NEW, log=log)
        if engine.name != "stub" and st.vector_sha(self.vec) != STEER_VECTOR_SHA256:
            raise SystemExit("REFUSED: the Steer-all vector is not 2 * v_29 of the §38 dev file")
        self.ids = cd.refusal_start_ids(read_jsonl(source / "repro.jsonl"))
        self.v29 = engine_v29(direction["v"], engine)
        self.R, self.C = list(pop.R), list(pop.C)
        if pop.damage != self.C[:len(pop.damage)]:
            raise SystemExit("REFUSED: the damage rows are not the first rows of C")
        if set(self.R) & set(self.C):
            raise SystemExit("REFUSED: R and C overlap")

    def pop_of(self, q: str) -> str:
        return "R" if q in self.R else "C"

    def check_manifest(self) -> None:
        manifest = {"engine": self.engine.name, "model": self.engine.model, "layer": self.arms.layer, "alpha": ALPHA,
                    "unit": UNIT, "max_new_tokens": self.max_new, "wording_sha256": self.wsha,
                    "wording": st.wording(self.max_new), "source": str(self.source),
                    "source_repro_sha256": file_sha256(self.source / "repro.jsonl"),
                    "source_sweep_sha256": file_sha256(self.source / "sweep.jsonl"),
                    "npz_sha256": self.direction["npz_sha256"], "v_sha256": self.direction["v_sha256"],
                    "vector_sha256": st.vector_sha(self.vec), "refusal_start_ids": self.ids,
                    "capture_layer": self.arms.layer, "n_R": len(self.R), "n_C": len(self.C),
                    "damage": list(self.pop.damage), "folds": sorted({self.d.by_q[q].fold for q in self.R + self.C})}
        path = self.dir / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            for k in ("engine", "model", "layer", "alpha", "unit", "max_new_tokens", "wording_sha256", "npz_sha256",
                      "vector_sha256", "refusal_start_ids", "source_repro_sha256"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"REFUSED: {path} has {k} = {old.get(k)!r}, this run {manifest[k]!r}; one engine "
                                     "and one design per run directory (move it aside)")
        self.write_json("manifest.json", manifest)

    def h29_path(self, kind: str, q: str) -> Path:
        return self.dir / "resid" / kind / f"{q}.npy"

    # --- one prompt: the unsteered pass (signals) and, on the intact C prompt, Steer-all from the same prefix --
    def do_prompt(self, kind: str, q: str, steer: bool = False) -> dict:
        pop = self.pop_of(q)
        item = self.item(kind, q)
        none, sa = st.Setting("none"), self.arms.steer_all
        out, todo = {}, []
        old = self.rows("prefill").get((pop, kind, "none", q))
        if old is not None and self.h29_path(kind, q).exists():
            st.check_row(old, item, self.engine.name, self.wsha, None)
            if old["step0"]["refusal_ids"] != self.ids:
                raise st.StaleRowError(f"{pop}/{kind}/none/{q}: refusal-start ids on disk differ from this run's")
            out["none"] = old
        else:
            todo.append(("prefill", none, None))
        if steer:
            old = self.rows("steer").get((pop, kind, sa.key, q))
            if old is not None:
                st.check_row(old, item, self.engine.name, self.wsha, self.vec)
                out[sa.key] = old
            else:
                todo.append(("steer", sa, self.vec))
        if not todo:
            return out
        prepared = self.engine.prepare(item)
        self.forward_passes["prefills"] += 1
        if not self.engine.last_token_text(prepared).endswith(":"):
            self.log(f"  WARNING {kind}/{q}: last prompt token is {self.engine.last_token_text(prepared)!r}, not ':'")
        gold = self.d.by_q[q].gold
        for stage, s, vec in todo:
            if s.mode == "none":
                gen = self.engine.run(prepared, [], self.max_new, capture=(self.arms.layer,), return_logits=True)
            else:
                gen = self.engine.run(prepared, st.specs_for(s, self.vectors), self.max_new)
            self.forward_passes["generations"] += 1
            self.forward_passes["tokens_generated"] += gen.n_new
            row = st.make_row(stage, pop, item, s, gen, self.engine, prepared, vec, self.max_new, gold, READER)
            if s.mode == "none":
                h = np.asarray(gen.captured[self.arms.layer], np.float32)
                self.h29_path(kind, q).parent.mkdir(parents=True, exist_ok=True)
                np.save(self.h29_path(kind, q), h)
                row["step0"] = cd.step0_summary(gen.logits0, self.ids)
                row["capture_layer"] = self.arms.layer
                row["h29_sha256"] = st.vector_sha(h)
                row["s_proj"] = cd.projection(h, self.v29)
            append_jsonl(self.path(stage), [row])
            self.rows(stage)
            self._rows[stage][st.row_key(row)] = row
            self.log_rows(stage, [row])
            out[s.key] = row
        return out

    # --- stages -----------------------------------------------------------------------------------------------
    def stage_r(self) -> dict:
        """(i) intact R: byte-identity with the §38 dev A rows (the stop) and h_29 against §38's resid_A[29]."""
        src = self.src("repro")
        per, resid = {}, {}
        for q in self.R:
            row = self.do_prompt("A", q)["none"]
            old = src.get(("R", "A", "none", q))
            per[q] = old is not None and row["generated"] == old["generated"] and row["gen_ids"] == old["gen_ids"]
            resid[q] = self.resid_match(q)
        k = sum(per.values())
        return {"n": len(per), "k": k, "rate": k / len(per) if per else None, "pass": k == len(per),
                "rows": per, "resid_vs_38": resid, "resid_exact": sum(v is True for v in resid.values()),
                "resid_checked": sum(v is not None for v in resid.values())}

    def resid_match(self, q: str) -> bool | None:
        """h_29 of this run against §38's saved resid_A (36 layers) at layer 29, byte-exact; None if §38 has none."""
        p = self.source / "resid" / "A" / f"{q}.npy"
        if not p.exists():
            return None
        old = np.load(p)
        if old.shape[0] <= self.arms.layer:
            return None
        return bool(np.array_equal(np.load(self.h29_path("A", q)), old[self.arms.layer].astype(np.float32)))

    def stage_c(self) -> dict:
        """(ii) intact C: the unsteered pass and Steer-all from one prefix; agreement with the caches printed."""
        src = self.src("repro")
        vs35, dmg, resid = {}, {}, {}
        for q in self.C:
            row = self.do_prompt("A", q, steer=True)["none"]
            vs35[q] = row["final_answer"] == self.d.rows["A"][q]["final_answer"]
            old = src.get(("C", "A", "none", q))
            if q in self.pop.damage and old is not None:
                dmg[q] = row["generated"] == old["generated"] and row["gen_ids"] == old["gen_ids"]
                resid[q] = self.resid_match(q)
        return {"n": len(vs35), "vs35_final_answer_k": sum(vs35.values()), "vs35": vs35,
                "damage_bytes_vs_38_k": sum(dmg.values()), "damage_bytes_vs_38_n": len(dmg), "damage_bytes": dmg,
                "damage_resid_exact": sum(v is True for v in resid.values()), "damage_resid": resid}

    def stage_cut(self) -> dict:
        for q in self.R + self.C:
            self.do_prompt("cut", q)
        return {"n": len(self.R) + len(self.C)}


# --- timing ---------------------------------------------------------------------------------------------------

def timing(run: Run) -> dict:
    """The first 5 C rows: intact prefill, the unsteered pass with capture and logits, and a Steer-all generation
    from the same prefix. Only times and lengths are kept."""
    recs = []
    for q in run.C[:st.TIMING_N]:
        prepared = run.engine.prepare(run.item("A", q))
        g0 = run.engine.run(prepared, [], run.max_new, capture=(run.arms.layer,), return_logits=True)
        g1 = run.engine.run(prepared, st.specs_for(run.arms.steer_all, run.vectors), run.max_new)
        for kind, g in (("none", g0), ("steer", g1)):
            recs.append({"kind": kind, "n_prefix": prepared.n_prefix, "prefill_s": prepared.prefill_s,
                         "copy_s": g.copy_s, "first_step_s": g.first_step_s, "decode_s": g.decode_s, "n_new": g.n_new})
    pre = {(r["n_prefix"], r["prefill_s"]) for r in recs}
    rate_p = sum(s for _, s in pre) / sum(n for n, _ in pre)
    dec = [r for r in recs if r["n_new"] > 1]
    rate_d = (sum(r["decode_s"] for r in dec) / sum(r["n_new"] - 1 for r in dec)) if dec else \
        sum(r["first_step_s"] for r in recs) / len(recs)
    mean = lambda kind: sum(r["n_new"] for r in recs if r["kind"] == kind) / sum(r["kind"] == kind for r in recs)  # noqa: E731
    return {"rows": len(recs) // 2, "prefill_s_per_token": rate_p, "decode_s_per_token": rate_d,
            "first_step_s": sum(r["first_step_s"] for r in recs) / len(recs),
            "copy_s": sum(r["copy_s"] for r in recs) / len(recs),
            "mean_new_tokens_unsteered": mean("none"), "mean_new_tokens_steer": mean("steer"), "records": recs}


def projection(run: Run, t: dict) -> dict:
    """Hours by stage from the exact prefix lengths and the timed rates. A generation = cache copy + the ':' pass +
    (n - 1) decode steps. Unsteered passes (R intact, C intact, cut on all) at the timed mean unsteered length on C
    (conservative for R, whose A rows are ' unknown' + EOS); Steer-all on C at 16 tokens (conservative) or at the
    timed mean (optimistic). Prefills: intact on R and C, cut on R and C (67 + 67)."""
    rp, rd, f, c = t["prefill_s_per_token"], t["decode_s_per_token"], t["first_step_s"], t["copy_s"]
    gen = lambda n: c + f + max(n - 1, 0) * rd  # noqa: E731
    g_un, g_max, g_opt = gen(t["mean_new_tokens_unsteered"]), gen(run.max_new), gen(t["mean_new_tokens_steer"])
    plen = lambda kind, qs: sum(len(run.engine.input_ids(run.item(kind, q))) - 1 for q in qs)  # noqa: E731
    R, C = run.R, run.C
    secs = {"intact_R": plen("A", R) * rp + len(R) * g_un,
            "intact_C_and_steer": plen("A", C) * rp + len(C) * (g_un + g_max),
            "cut_all": plen("cut", R + C) * rp + len(R + C) * g_un}
    total = sum(secs.values())
    optimistic = total - len(C) * (g_max - g_opt)
    return {"hours_by_stage": {k: v / 3600 for k, v in secs.items()}, "total_hours": total / 3600,
            "total_hours_optimistic": optimistic / 3600, "n_prefills": 2 * len(R + C),
            "n_generations": 2 * len(R + C) + len(C), "rev1_estimate_min": 45}


# --- main -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp41_dev")
    parser.add_argument("--source", default="runs/exp38_dev/qwen3b", help="the §38 dev run directory")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timing-only", action="store_true", help="5-row timing and projection, then stop")
    parser.add_argument("--run", action="store_true", help="run every stage (only after the Advisor's build review)")
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    if not set(folds) <= set(DEV):
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; §41 is folds 0-1 only")
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    source = Path(args.source)
    direction = load_direction(source)
    log(f"direction: {source / 'directions.npz'} sha256 {direction['npz_sha256']}; v_{LAYER} sha256 "
        f"{direction['v_sha256']}; Steer-all vector 2 * v_{LAYER} sha256 {direction['steer_sha256']}")
    ids = cd.refusal_start_ids(read_jsonl(source / "repro.jsonl"))
    log(f"refusal-start ids (first ids of the refusal-class §38 dev A and B rows, {source / 'repro.jsonl'} sha256 "
        f"{file_sha256(source / 'repro.jsonl')}): {ids}")
    d, pop = load_source(folds, Path(args.exp35), args.runs, Path(args.own_root))
    log(f"rows (folds {folds}): R {len(pop.R)} + C {len(pop.C)} = {len(pop.R) + len(pop.C)} distinct; damage "
        f"{len(pop.damage)} = the first {len(pop.damage)} of C (not run twice)")
    if not (args.run or args.timing_only):
        log("plan only (no model): pass --timing-only for the 5-row timing, --run for the run")
        return 0
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    log(f"engine {engine.name}, model loaded in {time.perf_counter() - t0:.0f} s")
    run = Run(engine, d, pop, Path(args.out), source, direction, log=log)
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run))
    reuse = run.reuse_steer_all(pop.R)
    log(f"Steer-all on R reused: {reuse['n']} rows of {reuse['key']} from {reuse['source']} (sha256 "
        f"{reuse['source_sha256']}); engine, classifier, wording, vector sha, max tokens and input sha match")
    tpath = run.dir / "timing.json"
    if tpath.exists():
        record = json.loads(tpath.read_text())
        if record["engine"] != engine.name:
            raise SystemExit(f"REFUSED: {tpath} was timed on {record['engine']!r}, this run uses {engine.name!r}")
        log(f"timing from {tpath}")
    else:
        t = timing(run)
        record = {"engine": engine.name, "threads": args.threads, "timing": t, "projection": projection(run, t)}
        run.write_json("timing.json", record)
    p = record["projection"]
    log(f"timing: prefill {record['timing']['prefill_s_per_token'] * 1000:.1f} ms/token, decode "
        f"{record['timing']['decode_s_per_token']:.2f} s/token, cache copy {record['timing']['copy_s']:.2f} s")
    log(f"projection: {p['total_hours']:.2f} h conservative ({p['total_hours_optimistic']:.2f} h optimistic; rev 1 "
        f"estimate {p['rev1_estimate_min']} min); by stage " + ", ".join(f"{k} {v:.2f} h" for k, v in p["hours_by_stage"].items()))
    if args.timing_only:
        log("Stopping (--timing-only).")
        return 0
    run.check_manifest()

    log(f"== (i) intact prompt on R ({len(run.R)} rows): unsteered pass, step-0 logits, h_{run.arms.layer}")
    rep = {"R": run.stage_r()}
    r = rep["R"]
    log(f"byte-identity with the §38 dev A rows {r['k']}/{r['n']}; h_{run.arms.layer} equal to §38's resid_A "
        f"byte-exact on {r['resid_exact']}/{r['resid_checked']} (descriptive)")
    run.write_json("repro.json", rep)
    if not r["pass"]:
        log(f"the unsteered A rows do not reproduce §38 byte-exact on every R row: the engine is not the one the "
            f"Steer-all rows came from; stopping before any steered generation (exit {EXIT_BYTES})")
        return EXIT_BYTES

    log(f"== (ii) intact prompt on C ({len(run.C)} rows): unsteered pass and Steer-all ({run.arms.steer_all.key}) "
        "from one prefix")
    rep["C"] = run.stage_c()
    c = rep["C"]
    log(f"unsteered final answer equal to the §35 cached A answer {c['vs35_final_answer_k']}/{c['n']}; damage rows "
        f"byte-equal to the §38 dev C rows {c['damage_bytes_vs_38_k']}/{c['damage_bytes_vs_38_n']}; their h_"
        f"{run.arms.layer} equal to §38's {c['damage_resid_exact']}/{c['damage_bytes_vs_38_n']}")
    run.write_json("repro.json", rep)

    log(f"== (iii) cut prompt on all {len(run.R) + len(run.C)} rows: unsteered pass, step-0 logits")
    rep["cut"] = run.stage_cut()
    run.write_json("repro.json", rep)
    run.write_json("passes.json", run.forward_passes)
    log(f"done; forward passes {run.forward_passes}. Report and rule: scripts/exp41_report.py --run-dir {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
