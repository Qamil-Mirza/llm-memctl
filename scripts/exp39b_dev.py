"""§39b laptop development run: the first-token-only clean-cache control (CPU, fp32, the 3B, folds 0-1).

    .venv/bin/python scripts/exp39b_dev.py                                  # plan only: populations, sources, shas
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp39b_dev.py --timing-only
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp39b_dev.py --run   # after the build review
    .venv/bin/python scripts/exp39b_dev.py --run --backend stub --source <stub §38> --source39 <stub §39> --out <dir>

Pre-registration: research/proposals/2026-10-23-exp39-one-position.md, amendment 2 (3e7c349). §39's Steer-0 leaves
the pushed ':' K/V above layer 29 in the cache for every later token (amendment 1.1); here the pushed state is removed
entirely: the first generated token of §39's Steer-0 row is fed as a forced token on an UNPUSHED cache.

Forced path (`HookedEngine.run_forced`): unpushed prefix cache (every prompt token before ':'), ':' run unpushed (its
argmax is recorded as natural_first_token, not used), the forced id is gen_ids[0], greedy continuation with no hook at
any step (checked: refused if any forward hook is registered), EOS / newline stop as §38-§39 (a newline in the decoded
text stops after that token, so a forced token containing a newline ends the generation at once), at most 16 new
tokens counting the forced one.

Tokenisation of a forced text after "Answer:" (`continuation_first_id`): the first id of ids(continuation + text)
after the ids(continuation) prefix, the prefix asserted stable; the prompt's own ids end with ids(continuation), so
this is the id the model would emit there. ' unknown' -> asserted equal to the A row's gen_ids[0] on every row (on
the 3B: 9788, as A's cached first id). Gold-first: text = " " + gold; ids(" " + gold)[0] is recorded beside
(gold_first_alone_id) and the disagreements counted (none on the dev rows when checked at the build).

Stages, in order; each gate fires before any later output is generated:
  (i)   reproduction: unsteered A and B on R and A on the 8 damage rows, each from its own prefill; flag rule vs
        the §38 dev rows, bound 0.8 (exit 3); byte-identity with the §38 dev rows < 0.9 -> stop (exit 6).
  (ii)  Unknown-first on the reproduced R rows: ' unknown' forced; the id must equal the A row's gen_ids[0] and the
        text and ids must equal this run's A row byte-exact on every row, else stop (exit 4).
  (iii) step-0 check: §39 Steer-0 (layer 29, steps {0}, 2 * v_29 from the §38 directions.npz, byte-exact) run for
        ONE new token on every row; must equal the §39 Steer-0 row's gen_ids[0] on every row, else stop (exit 5).
  (iv)+(vi) Steer-first (forced = §39 Steer-0 gen_ids[0]) and Gold-first (forced = the gold's first id) on the A
        prompt, sharing one prefill per row.
  (v)   Steer-first-cut: forced = §39 Steer-0-cut gen_ids[0] on the §38 cut prompt (no marker -> no note).

Sources, verified before the model loads (REFUSED otherwise): the §39 manifest (engine, wording sha, vector sha, max
tokens, layer, npz sha) and every Steer-0 / Steer-0-cut row (engine, wording sha, vector sha, input sha, step set
{0}, gen_ids non-empty with first_token == gen_ids[0]); the rows are copied verbatim to <out>/qwen3b/steer0.jsonl
with reuse.json (the §39 file's sha256). Folds other than 0 and 1 are refused (exit 2).

Exit codes: 0 done / plan / timing; 2 non-dev folds; 3 reproduction flag gate; 4 Unknown-first; 5 step-0 check;
6 byte-identity stop. Output: <out>/qwen3b/{manifest,timing,repro,reuse,unknown_first,step0,passes}.json,
{repro,unknown_first,step0,forced,steer0}.jsonl. Report: scripts/exp39b_report.py. HF_TOKEN is never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import MODELS, append_jsonl, read_jsonl  # noqa: E402
from scripts.exp38_dev import load_source, stub_oracle  # noqa: E402
from scripts.exp39_dev import (ALPHA, DEV, LAYER, READER, UNIT, Run as Run39, engine_vector,  # noqa: E402
                               file_sha256, load_direction)

EXIT_REPRO, EXIT_UNKNOWN, EXIT_STEP0, EXIT_BYTES = 3, 4, 5, 6
UNKNOWN_TEXT = " unknown"
ARM_UNKNOWN = st.ForcedArm("unknown_first")
ARM_STEER = st.ForcedArm("steer_first")
ARM_STEER_CUT = st.ForcedArm("steer_first_cut")
ARM_GOLD = st.ForcedArm("gold_first")
FORCING = ("unpushed prefix cache through the token before ':'; ':' unpushed (argmax recorded, unused); forced id = "
           "gen_ids[0]; greedy, no hook at any step; EOS / newline stop (newline token kept); max new tokens counts "
           "the forced one")
TOKENISATION = "first id of ids(continuation + text) after the ids(continuation) prefix (prefix asserted stable)"


class Run(Run39):
    """§39's Run (files, reproduction, resume, sha-stamped rows) with the §39b stages."""

    def __init__(self, engine, d, pop, out: Path, source: Path, source39: Path, direction: dict,
                 max_new: int = st.MAX_NEW_TOKENS_LAPTOP, log=print) -> None:
        super().__init__(engine, d, pop, out, source, direction, max_new=max_new, log=log)
        self.source39 = source39
        self.steer0_rows: dict[tuple, dict] = {}

    def check_manifest(self) -> None:
        manifest = {"engine": self.engine.name, "model": self.engine.model, "layer": self.arms.layer, "alpha": ALPHA,
                    "unit": UNIT, "max_new_tokens": self.max_new, "wording_sha256": self.wsha,
                    "wording": st.wording(self.max_new), "source": str(self.source), "source39": str(self.source39),
                    "source39_arms_sha256": file_sha256(self.source39 / "arms.jsonl"),
                    "npz_sha256": self.direction["npz_sha256"], "v_sha256": self.direction["v_sha256"],
                    "vector_sha256": st.vector_sha(self.vec), "forcing": FORCING, "tokenisation": TOKENISATION,
                    "folds": sorted({self.d.by_q[q].fold for q in self.pop.R})}
        path = self.dir / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            for k in ("engine", "model", "layer", "alpha", "unit", "max_new_tokens", "wording_sha256", "npz_sha256",
                      "vector_sha256", "source39_arms_sha256"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"REFUSED: {path} has {k} = {old.get(k)!r}, this run {manifest[k]!r}; one engine "
                                     "and one design per run directory (move it aside)")
        self.write_json("manifest.json", manifest)

    # --- the §39 Steer-0 rows (verified, copied) ------------------------------------------------------------
    def load_steer0(self, R: list[str]) -> dict:
        refuse = lambda why: SystemExit(f"REFUSED (§39 Steer-0 rows): {why}")  # noqa: E731
        mpath = self.source39 / "manifest.json"
        if not mpath.exists():
            raise refuse(f"{mpath} missing")
        man = json.loads(mpath.read_text())
        want_vsha = st.vector_sha(self.vec)
        for k, theirs, ours in (("engine", man.get("engine"), self.engine.name),
                                ("wording_sha256", man.get("wording_sha256"), self.wsha),
                                ("vector_sha256", man.get("vector_sha256"), want_vsha),
                                ("max_new_tokens", man.get("max_new_tokens"), self.max_new),
                                ("layer", man.get("layer"), self.arms.layer),
                                ("npz_sha256", man.get("npz_sha256"), self.direction["npz_sha256"])):
            if theirs != ours:
                raise refuse(f"§39 manifest {k} = {theirs!r}, this run {ours!r}")
        key = self.arms.steer0.key
        src = {st.row_key(r): r for r in read_jsonl(self.source39 / "arms.jsonl")}
        rows = []
        for kind in ("A", "cut"):
            for q in R:
                r = src.get(("R", kind, key, q))
                if r is None:
                    raise refuse(f"no §39 {key} row R/{kind}/{q}")
                item = self.item(kind, q)
                for k, theirs, ours in (("engine", r["engine"], self.engine.name),
                                        ("wording_sha256", r["wording_sha256"], self.wsha),
                                        ("vector_sha256", r["vector_sha256"], want_vsha),
                                        ("input_sha256", r["input_sha256"], item.input_sha256),
                                        ("steps", r["setting"].get("steps"), [0])):
                    if theirs != ours:
                        raise refuse(f"{kind}/{q}: {k} = {theirs!r} on the §39 row, this run {ours!r}")
                if not r["gen_ids"] or r["first_token"] != r["gen_ids"][0]:
                    raise refuse(f"{kind}/{q}: empty gen_ids or first_token != gen_ids[0]")
                if r["gen_ids"][0] in self.engine.stop_ids:
                    raise refuse(f"{kind}/{q}: the first token is a stop id")
                rows.append(r)
                self.steer0_rows[(kind, q)] = r
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "steer0.jsonl", "w") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        info = {"source": str(self.source39 / "arms.jsonl"), "source_sha256": file_sha256(self.source39 / "arms.jsonl"),
                "key": key, "n": len(rows), "kinds": ["A", "cut"], "vector_sha256": want_vsha,
                "engine": self.engine.name, "wording_sha256": self.wsha, "max_new_tokens": self.max_new}
        self.write_json("reuse.json", info)
        return info

    # --- forced generations, one prefill per prompt ---------------------------------------------------------
    def do_forced(self, stage: str, pop: str, kind: str, q: str, arms: list[tuple]) -> dict:
        """`arms`: [(ForcedArm, forced id, source dict)]; the pending ones share one prefill. Returns {arm key: row}."""
        item = self.item(kind, q)
        done = self.rows(stage)
        out, todo = {}, []
        for arm, fid, source in arms:
            old = done.get((pop, kind, arm.key, q))
            if old is not None:
                st.check_row(old, item, self.engine.name, self.wsha, None)
                if old["forced_token"] != fid:
                    raise st.StaleRowError(f"{pop}/{kind}/{arm.key}/{q}: forced id {old['forced_token']} on disk, "
                                           f"this run {fid}")
                out[arm.key] = old
            else:
                todo.append((arm, fid, source))
        if not todo:
            return out
        prepared = self.engine.prepare(item)
        self.forward_passes["prefills"] += 1
        if not self.engine.last_token_text(prepared).endswith(":"):
            self.log(f"  WARNING {kind}/{q}: last prompt token is {self.engine.last_token_text(prepared)!r}, not ':'")
        gold = self.d.by_q[q].gold
        new = []
        for arm, fid, source in todo:
            gen = self.engine.run_forced(prepared, fid, self.max_new)
            self.forward_passes["generations"] += 1
            self.forward_passes["tokens_generated"] += gen.n_new
            row = st.make_forced_row(stage, pop, item, arm, gen, self.engine, prepared, self.max_new, gold, source,
                                     READER)
            new.append(row)
            out[arm.key] = row
        append_jsonl(self.path(stage), new)
        self.rows(stage)
        self._rows[stage].update({st.row_key(r): r for r in new})
        self.log_rows(stage, new)
        return out

    def do_step0(self, q: str) -> dict:
        """§39 Steer-0 for ONE new token on the A prompt (row stamped with the run's wording sha, max_new_tokens 1)."""
        stage, s = "step0", self.arms.steer0
        item = self.item("A", q)
        old = self.rows(stage).get(("R", "A", s.key, q))
        if old is not None:
            st.check_row(old, item, self.engine.name, self.wsha, self.vec)
            return old
        prepared = self.engine.prepare(item)
        self.forward_passes["prefills"] += 1
        gen = self.engine.run(prepared, st.specs_for(s, self.vectors), 1)
        self.forward_passes["generations"] += 1
        self.forward_passes["tokens_generated"] += gen.n_new
        row = st.make_row(stage, "R", item, s, gen, self.engine, prepared, self.vec, self.max_new,
                          self.d.by_q[q].gold, READER)
        row["max_new_tokens"] = 1
        append_jsonl(self.path(stage), [row])
        self.rows(stage)
        self._rows[stage][st.row_key(row)] = row
        self.log_rows(stage, [row])
        return row

    # --- forced ids -----------------------------------------------------------------------------------------
    def unknown_id(self, q: str) -> int:
        return st.continuation_first_id(self.engine, self.item("A", q).continuation, UNKNOWN_TEXT)

    def gold_ids(self, q: str) -> dict:
        gold = str(self.d.by_q[q].gold)
        ctx = st.continuation_first_id(self.engine, self.item("A", q).continuation, " " + gold)
        alone = self.engine.encode(" " + gold)[0]
        return {"id": ctx, "alone_id": alone, "agree": ctx == alone, "text": " " + gold}

    # --- stages ii-vi -------------------------------------------------------------------------------------------
    def unknown_first(self, R: list[str]) -> dict:
        repro = self.rows("repro")
        per = {}
        for q in R:
            a = repro[("R", "A", "none", q)]
            fid = self.unknown_id(q)
            if not a["gen_ids"] or fid != a["gen_ids"][0]:
                per[q] = {"token_ok": False, "forced": fid, "a_first": a["gen_ids"][:1], "bytes": False}
                return {"pass": False, "why": f"{q}: ' unknown' tokenises to {fid}, the A row's first id is "
                                              f"{a['gen_ids'][:1]}", "rows": per, "n": len(R), "k": 0}
            row = self.do_forced("unknown_first", "R", "A", q,
                                 [(ARM_UNKNOWN, fid, {"text": UNKNOWN_TEXT, "rule": TOKENISATION})])[ARM_UNKNOWN.key]
            same = row["generated"] == a["generated"] and row["gen_ids"] == a["gen_ids"]
            per[q] = {"token_ok": True, "forced": fid, "bytes": same, "natural_first": row["natural_first_token"]}
        k = sum(v["bytes"] for v in per.values())
        return {"pass": k == len(R), "n": len(R), "k": k, "rows": per,
                "why": None if k == len(R) else f"{len(R) - k} rows differ from the A row"}

    def step0_check(self, R: list[str]) -> dict:
        per = {}
        for q in R:
            row = self.do_step0(q)
            want = self.steer0_rows[("A", q)]["gen_ids"][0]
            per[q] = {"step0_first": row["first_token"], "steer0_first": want, "same": row["first_token"] == want}
        k = sum(v["same"] for v in per.values())
        return {"pass": k == len(R), "n": len(R), "k": k, "rows": per}

    def forced_arms(self, R: list[str]) -> dict:
        gold_tok = {}
        for q in R:
            s0 = self.steer0_rows[("A", q)]
            g = self.gold_ids(q)
            gold_tok[q] = g
            self.do_forced("forced", "R", "A", q, [
                (ARM_STEER, s0["gen_ids"][0], {"from": "§39 Steer-0 gen_ids[0]", "input_sha256": s0["input_sha256"]}),
                (ARM_GOLD, g["id"], {"from": "gold", "text": g["text"], "rule": TOKENISATION,
                                     "gold_first_alone_id": g["alone_id"], "agree": g["agree"]})])
        for q in R:
            sc = self.steer0_rows[("cut", q)]
            self.do_forced("forced", "R", "cut", q, [
                (ARM_STEER_CUT, sc["gen_ids"][0], {"from": "§39 Steer-0-cut gen_ids[0]",
                                                   "input_sha256": sc["input_sha256"]})])
        return {"gold_tokenisation_disagree": sorted(q for q, g in gold_tok.items() if not g["agree"])}


# --- timing ---------------------------------------------------------------------------------------------------

def timing(run: Run) -> dict:
    """The first 5 R rows: prefill of A, then from the same prefix the forced ' unknown' generation (Unknown-first)
    and a forced generation with the §39 Steer-0 first token (a long unhooked decode). Only times and lengths are
    kept; no text is written or printed."""
    recs = []
    for q in run.pop.R[:st.TIMING_N]:
        prepared = run.engine.prepare(run.item("A", q))
        for fid in (run.unknown_id(q), run.steer0_rows[("A", q)]["gen_ids"][0]):
            g = run.engine.run_forced(prepared, fid, run.max_new)
            recs.append({"n_prefix": prepared.n_prefix, "prefill_s": prepared.prefill_s, "copy_s": g.copy_s,
                         "first_step_s": g.first_step_s, "decode_s": g.decode_s, "n_new": g.n_new})
    pre = {(r["n_prefix"], r["prefill_s"]) for r in recs}
    rate_p = sum(s for _, s in pre) / sum(n for n, _ in pre)
    dec = [r for r in recs if r["n_new"] > 1]
    # forward passes after ':' in a forced generation: n_new - 1 (the last token is never fed back, or EOS ended it:
    # then n_new passes); per-token rate from the decode time over n_new - 1 passes, conservative.
    rate_d = (sum(r["decode_s"] for r in dec) / sum(r["n_new"] - 1 for r in dec)) if dec else \
        sum(r["first_step_s"] for r in recs) / len(recs)
    return {"rows": len(recs) // 2, "prefill_s_per_token": rate_p, "decode_s_per_token": rate_d,
            "first_step_s": sum(r["first_step_s"] for r in recs) / len(recs),
            "copy_s": sum(r["copy_s"] for r in recs) / len(recs),
            "mean_new_tokens_forced": sum(r["n_new"] for r in recs[1::2]) / len(recs[1::2]), "records": recs}


def projection(run: Run, t: dict, mean_unsteered: float = 1.0) -> dict:
    """Hours by stage from the exact prefix lengths and the timed rates. A generation = cache copy + the ':' pass +
    (passes after it) decode steps. Conservative: every forced arm of Steer-first / Gold-first / Steer-first-cut
    runs to max_new_tokens (max_new_tokens passes after ':'); Unknown-first = 1 pass after ':' (the EOS); step-0 =
    the ':' pass only; repro generations at `mean_unsteered` tokens (1: the A rows say ' unknown' then EOS).
    Optimistic: the forced arms at the timed mean forced length. Prefills: repro A, B on R and A on damage;
    Unknown-first A; step-0 A; Steer-first + Gold-first A (one shared); Steer-first-cut cut."""
    rp, rd, f, c = t["prefill_s_per_token"], t["decode_s_per_token"], t["first_step_s"], t["copy_s"]
    gen = lambda passes_after: c + f + max(passes_after, 0) * rd  # noqa: E731
    plen = lambda kind, qs: sum(len(run.engine.input_ids(run.item(kind, q))) - 1 for q in qs)  # noqa: E731
    R, dmg = run.pop.R, run.pop.damage
    pA, pB, pC, pcut = plen("A", R) * rp, plen("B", R) * rp, plen("A", dmg) * rp, plen("cut", R) * rp
    g_max = gen(run.max_new)
    g_opt = gen(t.get("mean_new_tokens_forced", run.max_new))
    g_rep = gen(mean_unsteered)
    secs = {"repro": pA + pB + pC + (2 * len(R) + len(dmg)) * g_rep,
            "unknown_first": pA + len(R) * gen(1),
            "step0_check": pA + len(R) * gen(0),
            "steer_gold_first": pA + 2 * len(R) * g_max,
            "steer_first_cut": pcut + len(R) * g_max}
    total = sum(secs.values())
    optimistic = total - 3 * len(R) * (g_max - g_opt)
    return {"hours_by_stage": {k: v / 3600 for k, v in secs.items()}, "total_hours": total / 3600,
            "total_hours_optimistic": optimistic / 3600, "n_prefills": 6 * len(R) + 2 * len(dmg),
            "n_generations": 2 * len(R) + len(dmg) + 5 * len(R),
            "over_limit": total / 3600 > st.PROJECTION_LIMIT_H}


# --- main -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp39b_dev")
    parser.add_argument("--source", default="runs/exp38_dev/qwen3b", help="the §38 dev run directory")
    parser.add_argument("--source39", default="runs/exp39_dev/qwen3b", help="the §39 dev run directory")
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
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; §39b is folds 0-1 only")
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    source, source39 = Path(args.source), Path(args.source39)
    direction = load_direction(source)
    log(f"direction: {source / 'directions.npz'} sha256 {direction['npz_sha256']}; v_{LAYER} sha256 "
        f"{direction['v_sha256']}; Steer-0 vector 2 * v_{LAYER} sha256 {direction['steer_sha256']} (step-0 check only)")
    d, pop = load_source(folds, Path(args.exp35), args.runs, Path(args.own_root))
    log(f"populations (folds {folds}): R {len(pop.R)}; pairs {len(pop.pairs)}; unseen {len(pop.unseen)}; "
        f"damage {len(pop.damage)}")
    log(f"§39 rows: {source39 / 'arms.jsonl'} sha256 {file_sha256(source39 / 'arms.jsonl')}")
    if not (args.run or args.timing_only):
        log("plan only (no model): pass --timing-only for the 5-row timing, --run for the run")
        return 0
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    log(f"engine {engine.name}, model loaded in {time.perf_counter() - t0:.0f} s")
    run = Run(engine, d, pop, Path(args.out), source, source39, direction, log=log)
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run))
    info = run.load_steer0(pop.R)
    log(f"§39 Steer-0 rows verified and copied: {info['n']} ({info['kinds']}) of {info['key']}, source sha256 "
        f"{info['source_sha256']}; engine, wording, vector and input shas match")
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
    log(f"projection: {p['total_hours']:.2f} h conservative ({p['total_hours_optimistic']:.2f} h optimistic); by stage "
        + ", ".join(f"{k} {v:.2f} h" for k, v in p["hours_by_stage"].items()))
    if args.timing_only:
        log("Stopping (--timing-only).")
        return 0
    run.check_manifest()

    log("== (i) reproduction")
    rep = run.reproduction()
    run.write_json("repro.json", rep)
    log(f"reproduction (flag rule vs §38 dev rows) {rep['rate']:.3f} (R {rep['rate_R']:.3f}); dropped {rep['dropped']}")
    log(f"byte-identity with the §38 dev rows {rep['byte_rate']:.3f} ({rep['byte_k']}/{rep['byte_n']}); flags match, "
        f"bytes differ (kept, marked): {rep['flags_ok_bytes_differ']}")
    if not rep["pass"]:
        log(f"reproduction below {st.REPRO_GATE}: the engine is not fit, §39b stops here (exit {EXIT_REPRO})")
        return EXIT_REPRO
    if not rep["byte_pass"]:
        log(f"byte-identity {rep['byte_rate']:.3f} < 0.9: engine is not the one the §39 rows came from; stopping "
            f"before any forced or steered generation (exit {EXIT_BYTES})")
        return EXIT_BYTES
    R = rep["R_ok"]

    log(f"== (ii) Unknown-first on {len(R)} R rows")
    unk = run.unknown_first(R)
    run.write_json("unknown_first.json", unk)
    log(f"Unknown-first equals the A row byte-exact on {unk['k']}/{unk['n']}")
    if not unk["pass"]:
        log(f"Unknown-first failed ({unk['why']}): the forcing path is wrong; stopping (exit {EXIT_UNKNOWN})")
        return EXIT_UNKNOWN

    log(f"== (iii) step-0 check (Steer-0, one new token) on {len(R)} R rows")
    s0 = run.step0_check(R)
    run.write_json("step0.json", s0)
    log(f"step-0 first token equals the §39 Steer-0 first token on {s0['k']}/{s0['n']}")
    if not s0["pass"]:
        log(f"step-0 check failed: the hooked path does not give §39's first token; stopping (exit {EXIT_STEP0})")
        return EXIT_STEP0

    log(f"== (iv)-(vi) Steer-first, Gold-first (A prompt), Steer-first-cut (cut prompt) on {len(R)} R rows")
    fa = run.forced_arms(R)
    log(f"Gold-first tokenisation: in-context and alone ids disagree on {fa['gold_tokenisation_disagree'] or 'no rows'}")
    run.write_json("passes.json", run.forward_passes)
    log(f"done; forward passes {run.forward_passes}. Report: scripts/exp39b_report.py --run-dir {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
