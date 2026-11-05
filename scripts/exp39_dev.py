"""§39 laptop development run: the §38 layer-29 direction added at the ':' position only (CPU, fp32, the 3B, folds 0-1).

    .venv/bin/python scripts/exp39_dev.py                                   # plan only: populations, direction shas
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp39_dev.py --timing-only
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp39_dev.py --run   # after the build review
    .venv/bin/python scripts/exp39_dev.py --run --backend stub --source <stub §38 dir> --out <dir>   # tests only

Pre-registration: research/proposals/2026-10-23-exp39-one-position.md (rev 1, 25d257f; amendment 1, 7db734d).
Logic in memctl/analysis/steering.py (`Setting.steps`, `HookSpec.steps`: step 0 is the forward pass at ':', which
produces the first generated token; step j >= 1 is the pass at the j-th generated token). Nothing is re-estimated:

  direction  v_29 from <source>/directions.npz (the §38 dev file), byte-exact: the file's sha256, v_29's sha and the
             steer vector 2 * v_29 (= `scaled_vector(v_29, 2, 'vnorm')`, the sha stamped on the §38 Steer-all rows)
             are asserted and printed; the seeded layer-29 random vector (rng [38, 29], norm ||2 v_29||) is rebuilt
             and its sha printed (it is not the pod's layer-29 random vector: bf16 re-estimate, other norm).
  reuse      Steer-all = the §38 rows `steer:L29:a2:vnorm` (pop R, kind A) of <source>/sweep.jsonl, refused unless the
             engine, classifier version, wording sha, vector sha and max new tokens all match this run (amendment
             1.7); copied verbatim to <out>/qwen3b/steer_all.jsonl with reuse.json (source file sha256).
  (ii) timing 5 R rows (unless timing.json exists): prefill, an unsteered and a zero-add generation; projection by
             stage; over 5 h -> Steer-k with k = 2 only (§4). --timing-only stops here.
  (i) repro  unsteered A and B on the R rows, A on the 8 damage rows, each from its own prefill; gate = the §38 flag
             rule against the §38 dev rows (R: A and B refusal flags; damage: A answers and passes D1), bound 0.8
             (exit 3); byte-identity of the text and ids with the §38 dev rows printed per row and as a rate, a
             flag-match without byte-match kept and marked; byte-identity below 0.9 -> stop before any steered
             generation (exit 5, "engine is not the one the Steer-all rows came from") (amendment 1.2).
  arms       one prefix cache per prompt shared by all its arms. A prompt (reproduced R rows): none (same-run
             repeat), Steer-0 (steps {0}), Steer-k (steps 0..k, k in {1, 2, 4}), Random-0 (steps {0}). Cut prompt
             (§38's builder; no marker -> no note): none, Steer-0-cut, Steer-all-cut. Damage-0: Steer-0 on every
             damage row (the report reads the reproduced ones and all 8).

Folds other than 0 and 1 are refused. Output: <out>/qwen3b/{manifest,timing,repro,reuse,passes}.json,
{repro,arms,steer_all}.jsonl. Report: scripts/exp39_report.py. HF_TOKEN is never printed.
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
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import MODELS, read_jsonl  # noqa: E402
from scripts.exp38_dev import Run as Run38, load_source, stub_oracle, summarise_repro  # noqa: E402

DEV = (0, 1)
READER = "qwen3b"
LAYER, ALPHA, UNIT = 29, 2.0, "vnorm"
DOSE_K = (1, 2, 4)
DOSE_K_SMALL = (2,)
BYTE_GATE = 0.9
EXIT_REPRO, EXIT_BYTES = 3, 5
# the §38 dev direction, byte-exact (runs/exp38_dev/qwen3b, 2263ca3 lineage)
SOURCE_NPZ_SHA256 = "be1e3a09bdb5157d24126ebe16b14c035d429ebd51409e9b615b347c5b29f921"
V29_SHA256 = "a7f967ecdf2f59a9e1f64d99e366d33011f27daa8b09b18f5ff33d1e1ba20c6c"
STEER_VECTOR_SHA256 = "e0000366813f3c94cf4875c5ba948fc7f4bac220099cc8848edb79196045d71c"   # 2 * v_29, §38 Steer-all rows
RANDOM_VECTOR_SHA256 = "633bc895c2af0dc7f5c88d235c557eaaa4fa7955d2e3c4ba7efdfe9fa2269513"   # random_vector(2 v_29, 29)
RANDOM_NOTE = ("seeded layer-29 random vector (rng [38, 29], norm ||2 v_29||), rebuilt deterministically; NOT the pod's "
               "layer-29 random vector (bf16 re-estimate, different norm)")
STEER_ALL_KEY = "steer:L29:a2:vnorm"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_direction(source: Path) -> dict:
    """v_29 from the §38 dev directions.npz, byte-exact; every sha asserted. Returns v, the steer vector, shas."""
    path = source / "directions.npz"
    fsha = file_sha256(path)
    if fsha != SOURCE_NPZ_SHA256:
        raise SystemExit(f"REFUSED: {path} sha256 {fsha}, expected the §38 dev file {SOURCE_NPZ_SHA256}")
    v = np.load(path)[f"L{LAYER}"]
    if v.dtype != np.float32 or st.vector_sha(v) != V29_SHA256:
        raise SystemExit(f"REFUSED: v_{LAYER} sha {st.vector_sha(v)} ({v.dtype}), expected {V29_SHA256}")
    steer = st.scaled_vector(v, ALPHA, UNIT)
    assert np.array_equal(steer, (2.0 * v.astype(np.float64)).astype(np.float32)), "steer vector != 2 * v_29"
    assert st.vector_sha(steer) == STEER_VECTOR_SHA256, "steer vector sha differs from the §38 Steer-all rows'"
    rnd = st.random_vector(steer, LAYER)
    assert st.vector_sha(rnd) == RANDOM_VECTOR_SHA256, "the seeded layer-29 random vector differs"
    return {"v": v, "steer": steer, "random": rnd, "npz_sha256": fsha, "v_sha256": V29_SHA256,
            "steer_sha256": STEER_VECTOR_SHA256, "random_sha256": RANDOM_VECTOR_SHA256,
            "v_norm": float(np.linalg.norm(v.astype(np.float64))),
            "steer_norm": float(np.linalg.norm(steer.astype(np.float64))),
            "random_norm": float(np.linalg.norm(rnd.astype(np.float64)))}


def engine_vector(steer: np.ndarray, engine) -> np.ndarray:
    """The vector the engine adds: 2 * v_29 itself on the 3B; on the stub (hidden 16) its first `hidden` components,
    so the CPU stub run exercises the real file (the stub's outputs mean nothing)."""
    if engine.hidden == steer.shape[0]:
        return steer
    if engine.name != "stub":
        raise SystemExit(f"REFUSED: engine hidden size {engine.hidden} != {steer.shape[0]} on a non-stub engine")
    return np.ascontiguousarray(steer[:engine.hidden])


class Arms:
    """The §39 settings at the engine's layer 29 (the stub maps it)."""

    def __init__(self, layer: int, ks=DOSE_K) -> None:
        self.layer, self.ks = layer, tuple(ks)
        self.steer_all = st.Setting("steer", layer, ALPHA, UNIT)
        self.steer0 = st.Setting("steer", layer, ALPHA, UNIT, st.steps_upto(0))
        self.steer_k = {k: st.Setting("steer", layer, ALPHA, UNIT, st.steps_upto(k)) for k in self.ks}
        self.random0 = st.Setting("random", layer, ALPHA, UNIT, st.steps_upto(0))
        self.none = st.Setting("none")

    def a_prompt(self) -> list[st.Setting]:
        return [self.none, self.steer0, *self.steer_k.values(), self.random0]

    def cut_prompt(self) -> list[st.Setting]:
        return [self.none, self.steer0, self.steer_all]

    def damage(self) -> list[st.Setting]:
        return [self.steer0]


class Run(Run38):
    """§38's Run (files, shared-prefix `do`, resume, sha-stamped rows) with the §39 stages."""

    def __init__(self, engine, d, pop, out: Path, source: Path, direction: dict, ks=DOSE_K,
                 max_new: int = st.MAX_NEW_TOKENS_LAPTOP, log=print) -> None:
        layer = st.map_layer(LAYER, engine.n_layers)
        super().__init__(engine, d, pop, out, [layer], [ALPHA], UNIT, max_new, log)
        self.source, self.direction = source, direction
        self.vec = engine_vector(direction["steer"], engine)
        self.vectors = {(layer, ALPHA, UNIT): self.vec}
        self.arms = Arms(layer, ks)
        self.source_rows: dict[str, dict] = {}

    def src(self, stage: str) -> dict:
        if stage not in self.source_rows:
            self.source_rows[stage] = {st.row_key(r): r for r in read_jsonl(self.source / f"{stage}.jsonl")}
        return self.source_rows[stage]

    def check_manifest(self) -> None:
        manifest = {"engine": self.engine.name, "model": self.engine.model, "layer": self.arms.layer, "alpha": ALPHA,
                    "unit": UNIT, "ks": list(self.arms.ks), "max_new_tokens": self.max_new,
                    "wording_sha256": self.wsha, "wording": st.wording(self.max_new), "source": str(self.source),
                    "npz_sha256": self.direction["npz_sha256"], "v_sha256": self.direction["v_sha256"],
                    "vector_sha256": st.vector_sha(self.vec), "random_note": RANDOM_NOTE,
                    "random_29_sha256": self.direction["random_sha256"],
                    "random_added_sha256": st.vector_sha(st.random_vector(self.vec, self.arms.layer)),
                    "folds": sorted({self.d.by_q[q].fold for q in self.pop.R})}
        path = self.dir / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            for k in ("engine", "model", "layer", "alpha", "unit", "ks", "max_new_tokens", "wording_sha256",
                      "npz_sha256", "vector_sha256"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"REFUSED: {path} has {k} = {old.get(k)!r}, this run {manifest[k]!r}; one engine "
                                     "and one design per run directory (move it aside)")
        self.write_json("manifest.json", manifest)

    # --- Steer-all reuse (amendment 1.7) ---------------------------------------------------------------------
    def reuse_steer_all(self, R: list[str]) -> dict:
        refuse = lambda why: SystemExit(f"REFUSED (Steer-all reuse): {why}")  # noqa: E731
        mpath = self.source / "manifest.json"
        if not mpath.exists():
            raise refuse(f"{mpath} missing")
        man = json.loads(mpath.read_text())
        checks = {"engine": (man.get("engine"), self.engine.name),
                  "classifier": ((man.get("wording") or {}).get("classifier"), st.CLASSIFIER_VERSION),
                  "wording_sha256": (man.get("wording_sha256"), self.wsha),
                  "max_new_tokens": (man.get("max_new_tokens"), self.max_new)}
        for k, (theirs, ours) in checks.items():
            if theirs != ours:
                raise refuse(f"§38 manifest {k} = {theirs!r}, this run {ours!r}")
        key = self.arms.steer_all.key
        sweep = self.src("sweep")
        want_vsha = st.vector_sha(self.vec)
        rows = []
        for q in R:
            r = sweep.get(("R", "A", key, q))
            if r is None:
                raise refuse(f"no §38 {key} row for {q}")
            item = self.item("A", q)
            for k, theirs, ours in (("engine", r["engine"], self.engine.name),
                                    ("wording_sha256", r["wording_sha256"], self.wsha),
                                    ("vector_sha256", r["vector_sha256"], want_vsha),
                                    ("max_new_tokens", r["max_new_tokens"], self.max_new),
                                    ("input_sha256", r["input_sha256"], item.input_sha256),
                                    ("classifier (re-classified)", r["cls"], st.classify(r["phrase"], r["gen_ids"]))):
                if theirs != ours:
                    raise refuse(f"{q}: {k} = {theirs!r} on the §38 row, this run {ours!r}")
            if r["setting"].get("steps") is not None:
                raise refuse(f"{q}: the §38 row is step-limited")
            rows.append(r)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "steer_all.jsonl", "w") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        info = {"source": str(self.source / "sweep.jsonl"), "source_sha256": file_sha256(self.source / "sweep.jsonl"),
                "key": key, "n": len(rows), "vector_sha256": want_vsha, "engine": self.engine.name,
                "classifier": st.CLASSIFIER_VERSION, "wording_sha256": self.wsha, "max_new_tokens": self.max_new}
        self.write_json("reuse.json", info)
        return info

    # --- stage i ---------------------------------------------------------------------------------------------
    def reproduction(self) -> dict:
        """Unsteered A and B on R, A on the damage rows (each from its own prefill), then `compare_repro`."""
        none = st.Setting("none")
        for q in self.pop.R:
            self.do("repro", "R", "A", q, [none])
            self.do("repro", "R", "B", q, [none])
        for q in self.pop.damage:
            self.do("repro", "C", "A", q, [none])
        return compare_repro(self.rows("repro"), self.src("repro"), self.d, self.pop, self.source)

    # --- arms ------------------------------------------------------------------------------------------------
    def run_arms(self, R: list[str]) -> None:
        for q in R:
            self.do("arms", "R", "A", q, self.arms.a_prompt())
            self.do("arms", "R", "cut", q, self.arms.cut_prompt())
        for q in self.pop.damage:
            self.do("arms", "C", "A", q, self.arms.damage())


def compare_repro(rows: dict, src: dict, d, pop, source: Path | str = "the §38 dev run") -> dict:
    """Gate: the §38 flag rule against the §38 dev rows (R: A and B refusal flags; damage: A answers and passes D1),
    bound 0.8 (`summarise_repro`). Byte-identity (text and ids) with the §38 dev rows per row and as a rate; a row
    whose flags match but bytes differ is kept and listed. The §35-cache flag check is printed beside."""
    result = {"R": {}, "damage": {}}
    per_row = {}

    def get(table, pop_, kind, q, what):
        r = table.get((pop_, kind, "none", q))
        if r is None:
            raise SystemExit(f"REFUSED: no {what} repro row {pop_}/{kind}/{q} ({source})")
        return r

    def bytes_eq(new, old):
        return new["generated"] == old["generated"] and new["gen_ids"] == old["gen_ids"]

    for q in pop.R:
        gold = d.by_q[q].gold
        a, b = get(rows, "R", "A", q, "§39"), get(rows, "R", "B", q, "§39")
        sa, sb = get(src, "R", "A", q, "§38 dev"), get(src, "R", "B", q, "§38 dev")
        ra, rb = st.reproduces(a, sa, gold), st.reproduces(b, sb, gold)
        ra["vs35"] = st.reproduces(a, d.rows["A"][q], gold)["ok"]
        rb["vs35"] = st.reproduces(b, d.rows["B"][q], gold)["ok"]
        ra["bytes"], rb["bytes"] = bytes_eq(a, sa), bytes_eq(b, sb)
        result["R"][q] = {"A": ra, "B": rb, "ok": ra["ok"] and rb["ok"]}
        per_row[f"R/A/{q}"], per_row[f"R/B/{q}"] = ra["bytes"], rb["bytes"]
    for q in pop.damage:
        gold = d.by_q[q].gold
        a, sa = get(rows, "C", "A", q, "§39"), get(src, "C", "A", q, "§38 dev")
        ra = st.reproduces_damage(a, sa, gold)
        ra["vs35"] = st.reproduces_damage(a, d.rows["A"][q], gold)["ok"]
        ra["bytes"] = bytes_eq(a, sa)
        result["damage"][q] = {"A": ra, "ok": ra["ok"]}
        per_row[f"C/A/{q}"] = ra["bytes"]
    out = summarise_repro(result)
    out["gate_against"] = "§38 dev rows (flag rule, bound 0.8)"
    out["byte_identity"] = per_row
    out["byte_n"], out["byte_k"] = len(per_row), sum(per_row.values())
    out["byte_rate"] = out["byte_k"] / out["byte_n"] if per_row else None
    out["byte_pass"] = out["byte_rate"] is not None and out["byte_rate"] >= BYTE_GATE
    out["flags_ok_bytes_differ"] = sorted(
        [f"R/{q}" for q, v in result["R"].items() if v["ok"] and not (v["A"]["bytes"] and v["B"]["bytes"])]
        + [f"C/{q}" for q, v in result["damage"].items() if v["ok"] and not v["A"]["bytes"]])
    n35 = [v["A"]["vs35"] and v["B"]["vs35"] for v in result["R"].values()] + \
          [v["A"]["vs35"] for v in result["damage"].values()]
    out["rate_vs35"] = sum(n35) / len(n35) if n35 else None
    return out


# --- timing ---------------------------------------------------------------------------------------------------

def timing(run: Run) -> dict:
    """The first 5 R rows: prefill of A, an unsteered and a zero-add (every step) generation from the same prefix."""
    recs = []
    for q in run.pop.R[:st.TIMING_N]:
        prepared = run.engine.prepare(run.item("A", q))
        g0 = run.engine.run(prepared, [], run.max_new)
        g1 = run.engine.run(prepared, [st.add_spec(run.arms.layer, np.zeros(run.engine.hidden, np.float32))], run.max_new)
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


def projection(run: Run, t: dict, ks) -> dict:
    """Hours by stage from the exact prefix lengths and the timed rates. A generation costs copy + first step +
    (n - 1) decode steps: unsteered ones at the timed mean length, steered ones at max_new_tokens (conservative) or
    at the timed mean (optimistic). Prefills: repro A, B on R and A on damage; arms A on R, cut on R, A on damage."""
    rp, rd, f, c = t["prefill_s_per_token"], t["decode_s_per_token"], t["first_step_s"], t["copy_s"]
    gen = lambda n: c + f + max(n - 1, 0) * rd  # noqa: E731
    g_mean, g_max = gen(t["mean_new_tokens_unsteered"]), gen(run.max_new)
    plen = lambda kind, qs: sum(len(run.engine.input_ids(run.item(kind, q))) - 1 for q in qs)  # noqa: E731
    R, dmg = run.pop.R, run.pop.damage
    pA, pB, pC, pcut = plen("A", R) * rp, plen("B", R) * rp, plen("A", dmg) * rp, plen("cut", R) * rp
    n_steer_a = 1 + len(ks) + 1          # Steer-0, Steer-k, Random-0
    secs = {"repro": pA + pB + pC + (2 * len(R) + len(dmg)) * g_mean,
            "arms_A": pA + len(R) * (g_mean + n_steer_a * g_max),
            "arms_cut": pcut + len(R) * (g_mean + 2 * g_max),
            "damage": pC + len(dmg) * g_max}
    total = sum(secs.values())
    n_steered = len(R) * (n_steer_a + 2) + len(dmg)
    optimistic = total - n_steered * (g_max - g_mean)
    return {"hours_by_stage": {k: v / 3600 for k, v in secs.items()}, "total_hours": total / 3600,
            "total_hours_optimistic": optimistic / 3600, "ks": list(ks), "n_steered_generations": n_steered,
            "n_generations": n_steered + len(R) * 4 + len(dmg),
            "n_prefills": 4 * len(R) + 2 * len(dmg),
            "over_limit": total / 3600 > st.PROJECTION_LIMIT_H}


# --- main -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp39_dev")
    parser.add_argument("--source", default="runs/exp38_dev/qwen3b", help="the §38 dev run directory")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=st.MAX_NEW_TOKENS_LAPTOP)
    parser.add_argument("--k-small", action="store_true", help="Steer-k with k = 2 only (the §4 fallback) from the start")
    parser.add_argument("--timing-only", action="store_true", help="5-row timing and projection, then stop")
    parser.add_argument("--run", action="store_true", help="run every stage (only after the Advisor's build review)")
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    if not set(folds) <= set(DEV):
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; §39 is folds 0-1 only")
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    source = Path(args.source)
    direction = load_direction(source)
    log(f"direction: {source / 'directions.npz'} sha256 {direction['npz_sha256']}")
    log(f"  v_{LAYER} sha256 {direction['v_sha256']} (||v|| {direction['v_norm']:.4f}); steer vector = 2 * v_{LAYER} "
        f"(scaled_vector a2 vnorm, byte-equal) sha256 {direction['steer_sha256']} (norm {direction['steer_norm']:.4f})")
    log(f"  Random-0: {RANDOM_NOTE}; sha256 {direction['random_sha256']} (norm {direction['random_norm']:.4f})")
    d, pop = load_source(folds, Path(args.exp35), args.runs, Path(args.own_root))
    log(f"populations (folds {folds}): R {len(pop.R)}; pairs {len(pop.pairs)}; unseen {len(pop.unseen)}; "
        f"damage {len(pop.damage)}")
    if not (args.run or args.timing_only):
        log("plan only (no model): pass --timing-only for the 5-row timing, --run for the run")
        return 0
    t0 = time.perf_counter()
    engine = st.build_engine(args.backend, MODELS[READER], args.threads)
    log(f"engine {engine.name}, model loaded in {time.perf_counter() - t0:.0f} s")
    ks = DOSE_K_SMALL if args.k_small else DOSE_K
    run = Run(engine, d, pop, Path(args.out), source, direction, ks, args.max_new_tokens, log)
    if args.backend == "stub":
        engine.set_oracle(stub_oracle(run))
    reuse = run.reuse_steer_all(pop.R)
    log(f"Steer-all reused: {reuse['n']} rows of {reuse['key']} from {reuse['source']} (sha256 {reuse['source_sha256']}); "
        f"engine, classifier, wording, vector sha and max tokens match")
    tpath = run.dir / "timing.json"
    if tpath.exists():
        record = json.loads(tpath.read_text())
        if record["engine"] != engine.name:
            raise SystemExit(f"REFUSED: {tpath} was timed on {record['engine']!r}, this run uses {engine.name!r}")
        log(f"timing from {tpath}")
    else:
        t = timing(run)
        record = {"engine": engine.name, "threads": args.threads, "timing": t,
                  "projection": projection(run, t, DOSE_K), "projection_small": projection(run, t, DOSE_K_SMALL)}
        run.write_json("timing.json", record)
    p = record["projection"]
    log(f"timing: prefill {record['timing']['prefill_s_per_token'] * 1000:.1f} ms/token, decode "
        f"{record['timing']['decode_s_per_token']:.2f} s/token, cache copy {record['timing']['copy_s']:.2f} s")
    log(f"projection (k {p['ks']}): {p['total_hours']:.2f} h conservative ({p['total_hours_optimistic']:.2f} h "
        f"optimistic); k = 2 only {record['projection_small']['total_hours']:.2f} h; by stage "
        + ", ".join(f"{k} {v:.2f} h" for k, v in p["hours_by_stage"].items()))
    if args.timing_only:
        log("Stopping (--timing-only).")
        return 0
    if p["over_limit"] and not args.k_small:
        log(f"projection over {st.PROJECTION_LIMIT_H} h: Steer-k with k = 2 only (§4)")
        run.arms = Arms(run.arms.layer, DOSE_K_SMALL)
    run.check_manifest()

    log("== (i) reproduction")
    rep = run.reproduction()
    run.write_json("repro.json", rep)
    log(f"reproduction (flag rule vs §38 dev rows) {rep['rate']:.3f} (R {rep['rate_R']:.3f}); vs §35 cache "
        f"{rep['rate_vs35']:.3f}; dropped {rep['dropped']}")
    log(f"byte-identity with the §38 dev rows {rep['byte_rate']:.3f} ({sum(rep['byte_identity'].values())}/"
        f"{len(rep['byte_identity'])}); flags match, bytes differ (kept, marked): {rep['flags_ok_bytes_differ']}")
    if not rep["pass"]:
        log(f"reproduction below {st.REPRO_GATE}: the engine is not fit, §39 stops here (exit {EXIT_REPRO})")
        return EXIT_REPRO
    if not rep["byte_pass"]:
        log(f"byte-identity {rep['byte_rate']:.3f} < {BYTE_GATE}: engine is not the one the Steer-all rows came from; "
            f"stopping before any steered generation (exit {EXIT_BYTES})")
        return EXIT_BYTES
    R = rep["R_ok"]

    log(f"== arms (layer {run.arms.layer}, a{ALPHA:g} {UNIT}, k {list(run.arms.ks)}) on {len(R)} R rows, "
        f"{len(pop.damage)} damage rows")
    run.run_arms(R)
    run.write_json("passes.json", run.forward_passes)
    log(f"done; forward passes {run.forward_passes}. Report: scripts/exp39_report.py --run-dir {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
