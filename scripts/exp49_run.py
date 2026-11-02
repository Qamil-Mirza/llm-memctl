#!/usr/bin/env python
"""§49 legible abstentions: baselines with captures, dev probes, dev α sweep, design, held arms. Resumable per row.

    .venv/bin/python scripts/exp49_run.py --stage timing --out runs/exp49_timing  # ONE unit of each kind
    .venv/bin/python scripts/exp49_run.py --smoke 5 --out runs/exp49_smoke/run    # D-22: the whole chain on dev bases
    python scripts/exp49_run.py --stage baselines --split dev                    # generations + residuals (both sites)
    python scripts/exp49_run.py --stage probe-dev                                # probes, ℓ* per site, cross-fit dev decodes
    python scripts/exp49_run.py --stage arms-dev                                 # gate-tune: R1 and LF at α 0.5/1/2, no-ops
    python scripts/exp49_run.py --stage design                                   # α, τ_c, τ_s -> design.json (+ sha)
    python scripts/exp49_run.py --stage mark-design --commit <hash>              # after the design commit
    python scripts/exp49_run.py --stage baselines --split held                   # refuses without the committed design
    python scripts/exp49_run.py --stage arms-held                                # R1-R4, D-gold, CF twins, R4 damage, F1 arm

Spec: research/proposals/2026-11-01-exp49-legible-abstentions.md rev 0a (D-63); helpers in memctl/analysis/abst49.py.
Engine: memctl.analysis.steering.HFEngine (fp32 CPU / any dtype on CUDA) with its `hooked` forward hooks; the residual
is the output of decoder layer l, captured by hooks on every layer in one teacher-forced pass over the model's own
tokens (prompt + generated prefix through "Answer:"), so a prefix never holds a token the model did not write.
Release arms run greedy from the frozen prefix (KV over everything before the P_ans token), the P_ans token is the
step-0 forward, at most 16 new tokens, stop at a newline. R1 / R3 / D-gold and R2 (one-position: the §38 / §40
layer-29 vector at α 2, rev 0b B5) add at step 0 (P_ans) only. R4 regenerates from the R4 prompt (160, the
baseline stop). Rev 0b: dev parts fit / select / gate-tune (B4), fallbacks (B3), M2b on H1 (B2), 20 no-op replays
per split. Nothing here prints an outcome except --smoke on dev rows. Exit 4: design guard. 9: deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import abst49 as A  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CASES = Path("runs/exp49/cases49.jsonl")
OUT = Path("runs/exp49/run")
EXIT_DESIGN, EXIT_DEADLINE = 4, 9


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def cities():
    import exp45_cases as C
    return C.CITIES


# ---------------------------------------------------------------------------------------------- engine

class Engine:
    """Generation and capture on a steering.HookedEngine (HF or the stub)."""

    def __init__(self, eng):
        self.eng = eng
        self.n_layers, self.hidden = eng.n_layers, eng.hidden

    def chat_ids(self, prompt: str) -> list[int]:
        return self.eng.encode(self.eng.chat_text(prompt))

    def decode(self, ids) -> str:
        return self.eng.decode(list(ids))

    def generate(self, ids: list[int], specs: list, max_new: int, stop) -> list[int]:
        """Greedy: KV over ids[:-1], ids[-1] is step 0 (hooks with steps {0} act there), then one token per step.
        EOS is not recorded; `stop(text)` ends after the token that satisfies it."""
        import torch
        cache = self.eng._prefill(ids[:-1])
        state = st.HookState()
        gen: list[int] = []
        with torch.no_grad(), st.hooked(self.eng.layers, specs, state):
            logits = self.eng._step(ids[-1], cache)
            while True:
                nxt = int(logits.argmax())
                if nxt in self.eng.stop_ids:
                    break
                gen.append(nxt)
                if len(gen) >= max_new or stop(self.decode(gen)):
                    break
                state.step += 1
                logits = self.eng._step(nxt, cache)
        return gen

    def capture(self, ids: list[int], positions: list[int]) -> np.ndarray:
        """[len(positions), n_layers, hidden] float32: the output of every decoder layer at `positions`, from one
        teacher-forced pass over `ids` (HF), or token by token (the stub)."""
        import torch
        net = getattr(self.eng, "net", None)
        if net is None:
            return self._capture_steps(ids, positions)
        saved = {}

        def mk(l):
            def fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                saved[l] = h[0, positions].detach().float().cpu().numpy()
            return fn
        handles = [layer.register_forward_hook(mk(l)) for l, layer in enumerate(self.eng.layers)]
        try:
            with torch.no_grad():
                net(input_ids=torch.tensor([ids], device=self.eng.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return np.stack([saved[l] for l in range(self.n_layers)], axis=1)

    def _capture_steps(self, ids, positions):
        import torch
        out = []
        for p in positions:
            cache = self.eng._prefill(ids[:p]) if p > 0 else None
            state = st.HookState()
            specs = [st.capture_spec(l) for l in range(self.n_layers)]
            with torch.no_grad(), st.hooked(self.eng.layers, specs, state):
                self.eng._step(ids[p], cache)
            out.append(np.stack([state.captured[l] for l in range(self.n_layers)]))
        return np.stack(out)


def load_engine(backend: str, threads: int, dtype: str | None) -> Engine:
    if backend == "stub":
        return Engine(st.StubEngine())
    return Engine(st.HFEngine(MODEL_ID, threads=threads, dtype=dtype))


def stop_fn(fmt: str):
    return A.answer_line_done if fmt == "F2" else (lambda text: False)


def newline(text: str) -> bool:
    return "\n" in text


# ---------------------------------------------------------------------------------------------- baselines

def baseline_row(e: Engine, r: dict, root: Path, prompt_key: str = "prompt") -> dict:
    """One baseline (or the R4 regeneration with prompt_key 'prompt_r4'): generation, outcome, the P_ans prefix and
    flags; with the plain prompt also the residuals at P_pre and P_ans (all layers) to resid/<case_id>.npy."""
    fmt = r["format"]
    ids = e.chat_ids(r[prompt_key])
    t0 = time.perf_counter()
    gen = e.generate(ids, [], A.MAX_NEW[fmt], stop_fn(fmt))
    gen_s = time.perf_counter() - t0
    text = e.decode(gen)
    ans = A.answer_text(text, fmt)
    cls = A.classify(ans, r["gold"], r["competitors"], cities())
    rec = {"case_id": r["case_id"], "prompt_key": prompt_key, "n_prompt": len(ids), "gen_ids": gen, "text": text,
           "answer": ans, **cls, "gen_s": round(gen_s, 3)}
    if prompt_key != "prompt":
        return rec
    j, ok, why = None, False, "F1"
    if fmt == "F2":
        px = A.ans_prefix(gen, e.decode)
        j, ok, why = px["j"], px["ok"], px["why"]
        if ok and len(gen) >= A.MAX_NEW["F2"] and not A.answer_line_done(text):
            ok, why = False, "truncated at 160"          # B3: P_ans undefined
    positions = [len(ids) - 1] + ([len(ids) + j - 1] if ok else [])
    t0 = time.perf_counter()
    resid = e.capture(ids + (gen[:j] if ok else []), positions)
    cap_s = time.perf_counter() - t0
    if not ok:
        resid = np.concatenate([resid, np.zeros_like(resid[:1])])
    np.save(root / "resid" / f"{r['case_id']}.npy", resid.astype(np.float32))
    note = e.decode(gen[:j]) if ok else (text if fmt == "F2" else "")
    rec.update({"j": j, "p_ans_ok": ok, "p_ans_why": why, "p_pre": len(ids) - 1,
                "p_ans": (len(ids) + j - 1) if ok else None, "cap_s": round(cap_s, 3),
                **(A.note_flags(note, r["gold"], cities()) if fmt == "F2" else {})})
    return rec


def run_baselines(e: Engine, rows: list[dict], root: Path, deadline: float | None) -> int:
    t0 = time.time()
    for r in rows:
        path = root / "baselines" / f"{r['case_id']}.json"
        if path.exists():
            continue
        if deadline and time.time() - t0 > deadline:
            return EXIT_DEADLINE
        write_json(path, baseline_row(e, r, root))
    return 0


def baseline(root: Path, case_id: str) -> dict | None:
    return read_json(root / "baselines" / f"{case_id}.json")


def resid(root: Path, case_id: str) -> np.ndarray:
    return np.load(root / "resid" / f"{case_id}.npy")          # [2 (P_pre, P_ans), L, d]


# ---------------------------------------------------------------------------------------------- probes (dev)

def probe_dev(rows: list[dict], root: Path) -> dict:
    """Rev 0b B3 / B4 on dev only: 12-way probes fit on the fit part, ℓ* per site by select-part accuracy (P_pre on
    ANS F1 + F2; ℓ*_F1 scored on select F1 rows; P_ans on ANS-F2 with a prefix); the sufficiency probe on H1 (ANS-F2-H1
    vs UNA-F2-H1) at P_pre, layer by select AUROC, refit on fit + select; centroids from fit + select; gate-tune
    decodes and scores (out of sample by construction). Pickles models + centroids; their sha256s go to the design."""
    dev = [r for r in rows if r["split"] == "dev"]
    B = {r["case_id"]: baseline(root, r["case_id"]) for r in dev}
    X = {r["case_id"]: resid(root, r["case_id"]) for r in dev}
    part = {r["case_id"]: A.dev_part(r["base_index"]) for r in dev}

    def stack(rs, site):
        return np.stack([X[r["case_id"]][site] for r in rs])

    def mask(rs, *parts):
        return np.array([part[r["case_id"]] in parts for r in rs])
    ans = [r for r in dev if r["set"] == "ANS"]
    ans_ok = [r for r in ans if r["format"] == "F2" and B[r["case_id"]]["p_ans_ok"]]
    Xpre, ypre = stack(ans, 0), [r["gold"] for r in ans]
    fit_pre, sel_pre = mask(ans, "fit"), mask(ans, "select")
    pre = A.choose_layer(Xpre, ypre, fit_pre, sel_pre)
    f1 = A.choose_layer(Xpre, ypre, fit_pre, sel_pre & np.array([r["format"] == "F1" for r in ans]))
    Xans, yans = stack(ans_ok, 1), [r["gold"] for r in ans_ok]
    fit_ans, sel_ans = mask(ans_ok, "fit"), mask(ans_ok, "select")
    pans = A.choose_layer(Xans, yans, fit_ans, sel_ans)
    suff_rows = [r for r in dev if r["format"] == "F2" and r["hardness"] == "H1" and r["set"] in ("ANS", "UNA")]
    Xs, ys = stack(suff_rows, 0), [int(r["set"] == "ANS") for r in suff_rows]
    suff = A.choose_layer(Xs, ys, mask(suff_rows, "fit"), mask(suff_rows, "select"), metric="auroc")
    lp, lf, la, ls = pre["layer"], f1["layer"], pans["layer"], suff["layer"]
    yp, ya = np.asarray(ypre), np.asarray(yans)
    fs = mask(suff_rows, "fit", "select")
    models = {"pre": A.probe_model().fit(Xpre[fit_pre, lp], yp[fit_pre]),
              "f1": A.probe_model().fit(Xpre[fit_pre, lf], yp[fit_pre]),
              "ans": A.probe_model().fit(Xans[fit_ans, la], ya[fit_ans]),
              "suff": A.probe_model().fit(Xs[fs, ls], np.asarray(ys)[fs])}
    kp = [r["base_key"] for r in ans]
    ka = [r["base_key"] for r in ans_ok]
    shuf = {"pre": A.probe_model().fit(Xpre[fit_pre, lp], A.shuffled_labels(ypre, kp)[fit_pre]),
            "ans": A.probe_model().fit(Xans[fit_ans, la], A.shuffled_labels(yans, ka)[fit_ans])}
    # centroids from fit + select
    classes = sorted(set(ypre))
    cs_ans = [r for r in ans_ok if part[r["case_id"]] != "tune"]
    g_ans = [r for r in cs_ans if B[r["case_id"]]["outcome"] == "gold"]
    ab = [r for r in cs_ans if B[r["case_id"]]["outcome"] == "abstain"]
    mu_ans = A.centroids(stack(g_ans, 1)[:, la], [r["gold"] for r in g_ans], ["F2"] * len(g_ans), classes)
    mu_abst = stack(ab, 1)[:, la].astype(np.float64).mean(0) if len(ab) >= A.MU_ABST_MIN else None
    g_pre = [r for r in ans if part[r["case_id"]] != "tune" and B[r["case_id"]]["outcome"] == "gold"]
    mu_f1 = A.centroids(stack(g_pre, 0)[:, lf], [r["gold"] for r in g_pre], [r["format"] for r in g_pre], classes)
    # gate-tune pool: H1 abstentions with a prefix (false: ANS-F2-H1, genuine: UNA-F2-H1)
    tune_pool = [r for r in dev if part[r["case_id"]] == "tune" and r["format"] == "F2" and r["hardness"] == "H1"
                 and r["set"] in ("ANS", "UNA") and B[r["case_id"]]["outcome"] == "abstain"
                 and B[r["case_id"]]["p_ans_ok"]]
    sm = models["suff"]
    pool = {}
    for r in tune_pool:
        h_ans, h_pre = X[r["case_id"]][1, la], X[r["case_id"]][0, ls]
        pr = models["ans"].predict_proba(h_ans[None])[0]
        pool[r["case_id"]] = {"set": r["set"], "c_hat": str(models["ans"].classes_[int(np.argmax(pr))]),
                              "content_max": float(pr.max()),
                              "suff": float(sm.predict_proba(h_pre[None])[0, list(sm.classes_).index(1)])}
    tune_f1 = {}
    for r in dev:
        if part[r["case_id"]] == "tune" and r["format"] == "F1":
            b = B[r["case_id"]]
            tune_f1[r["case_id"]] = {"outcome": b["outcome"],
                                     "c_hat": str(models["f1"].predict(X[r["case_id"]][0, lf][None])[0])}
    pdir = root / "probe"
    pdir.mkdir(parents=True, exist_ok=True)
    weights = pickle.dumps({"models": models, "shuffled": shuf})
    cent = {"mu_ans": mu_ans["mu"], "mu_abst": mu_abst, "mu_f1": mu_f1["mu"]}
    (pdir / "probes.pkl").write_bytes(weights)
    (pdir / "centroids.pkl").write_bytes(pickle.dumps(cent))
    arrays = {f"ans|{k}": v for k, v in mu_ans["mu"].items()} | {f"f1|{k}": v for k, v in mu_f1["mu"].items()}
    if mu_abst is not None:
        arrays["abst"] = mu_abst
    out = {"layer_pre": lp, "layer_f1": lf, "layer_ans": la, "layer_suff": ls,
           "select": {"pre": pre, "f1": f1, "ans": pans, "suff": suff},
           "mu_ans_source": mu_ans["source"], "mu_ans_n": mu_ans["n"], "mu_f1_source": mu_f1["source"],
           "mu_f1_n": mu_f1["n"], "n_abst_centroid": len(ab), "r1_estimable": mu_abst is not None,
           "n_tune_pool": len(pool), "n_tune_pool_false": sum(v["set"] == "ANS" for v in pool.values()),
           "label_counts_by_part": {pt: dict(sorted(Counter(r["gold"] for r in ans if part[r["case_id"]] == pt).items()))
                                    for pt in A.PARTS},
           "n_tune_f1": len(tune_f1), "parts": dict(Counter(part[r["case_id"]] for r in dev if r["set"] == "ANS"
                                                             and r["format"] == "F2" and r["hardness"] == "H1")),
           "weights_sha256": hashlib.sha256(weights).hexdigest(), "centroids_sha256": A.array_sha(arrays)}
    write_json(pdir / "probe_dev.json", out)
    write_json(pdir / "tune_pool.json", pool)
    write_json(pdir / "tune_f1.json", tune_f1)
    return out


def load_probes(root: Path) -> dict:
    P = pickle.loads((root / "probe" / "probes.pkl").read_bytes())
    P.update(pickle.loads((root / "probe" / "centroids.pkl").read_bytes()))
    return P


# ---------------------------------------------------------------------------------------------- arms

def prefix_ids(e: Engine, r: dict, b: dict) -> list[int]:
    ids = e.chat_ids(r["prompt"])
    if len(ids) != b["n_prompt"]:
        raise RuntimeError(f"{r['case_id']}: prompt re-tokenises to {len(ids)}, baseline {b['n_prompt']}")
    return ids + b["gen_ids"][:b["j"]]


def release(e: Engine, r: dict, b: dict, specs: list) -> dict:
    t0 = time.perf_counter()
    gen = e.generate(prefix_ids(e, r, b), specs, A.MAX_NEW_RELEASE, newline)
    text = e.decode(gen)
    cls = A.classify(text.split("\n")[0], r["gold"], r["competitors"], cities())
    return {"gen_ids": gen, "text": text, **cls, "s": round(time.perf_counter() - t0, 3)}


NO_RELEASE = {"outcome": "abstain", "item": "unknown", "text": None, "released": False}


def r1_trial(e: Engine, r: dict, b: dict, h: np.ndarray, P: dict, target: str, layer: int, alpha: float,
             random_arm: str | None = None) -> dict:
    """R1 (or D-gold with target = gold). B3: no μ_abst -> not estimable; no centroid for the target -> unestimable,
    scored as no release (the trial stays an abstention). With `random_arm` (R3): a seeded vector of norm α·‖h‖ on
    every trial, defined whether or not R1 is (build review ruling 8)."""
    rec = {"layer": layer, "alpha": alpha, "target": target}
    if random_arm:
        norm = alpha * float(np.linalg.norm(h.astype(np.float64)))
        v = A.random_like(norm, len(h), r["case_id"], random_arm)
        return {**rec, "status": "ok", "released": True, "vector_sha256": st.vector_sha(v),
                **release(e, r, b, [st.add_spec(layer, v, steps=(0,))])}
    if P["mu_abst"] is None:
        return {**rec, **NO_RELEASE, "status": "R1 not estimable"}
    if target not in P["mu_ans"]:
        return {**rec, **NO_RELEASE, "status": "R1 unestimable"}
    v = A.r1_vector(h, P["mu_ans"], P["mu_abst"], target, alpha)
    return {**rec, "status": "ok", "released": True, "vector_sha256": st.vector_sha(v),
            **release(e, r, b, [st.add_spec(layer, v, steps=(0,))])}


def noop_trial(e: Engine, r: dict, b: dict) -> dict:
    """B2 sanity: a zero vector at P_ans (layer 18) on the frozen prefix must reproduce the baseline continuation."""
    z = release(e, r, b, [st.add_spec(18, np.zeros(e.hidden, np.float32), steps=(0,))])
    want = b["gen_ids"][b["j"]:]
    return {**z, "reproduces": z["gen_ids"] == want[:len(z["gen_ids"])] and len(z["gen_ids"]) > 0}


def noop_rows(rows: list[dict], root: Path, split: str) -> list[dict]:
    """20 F2 rows of the split with a P_ans, by a seeded draw (seed 49)."""
    cand = sorted((r for r in rows if r["split"] == split and r["format"] == "F2"
                   and (baseline(root, r["case_id"]) or {}).get("p_ans_ok")), key=lambda r: r["case_id"])
    idx = np.random.default_rng(A.SEED).permutation(len(cand))[:A.NOOP_PER_SPLIT]
    return [cand[i] for i in sorted(idx)]


def f1_trial(e: Engine, r: dict, b: dict, root: Path, P: dict, lf: int, alpha: float, arm: str) -> dict:
    """Legible Failures' repair on F1 (labelled replication): fires iff the output is a pool city e with a centroid
    and the P_pre decode ĉ ≠ e (with a centroid); adds α_LF·(μ_ĉ − μ_e) (LF) or a matched-norm seeded vector (LFrand)
    at ℓ*_F1, step 0; LF = α·‖h‖·unit(μ_ĉ − μ_e) (build review fix 2). B3: no city -> not applied, stays in the F1-wrong denominator unrepaired."""
    h = resid(root, r["case_id"])[0, lf]
    c_hat = str(P["models"]["f1"].predict(h[None])[0])
    e_out = b["item"] if b["outcome"] in ("gold", "competitor", "wrong") else None
    fire = e_out is not None and c_hat != e_out and c_hat in P["mu_f1"] and e_out in P["mu_f1"]
    rec = {"c_hat": c_hat, "e_out": e_out, "fired": bool(fire), "layer": lf, "alpha": alpha}
    if not fire:
        return {**rec, "outcome": b["outcome"], "item": b["item"], "text": None}
    v = A.lf_vector(h, P["mu_f1"], c_hat, e_out, alpha)
    if arm == "LFrand":
        v = A.random_like(float(np.linalg.norm(v.astype(np.float64))), len(v), r["case_id"], "LFrand")
    gen = e.generate(e.chat_ids(r["prompt"]), [st.add_spec(lf, v, steps=(0,))], A.MAX_NEW["F1"], lambda t: False)
    text = e.decode(gen)
    return {**rec, "gen_ids": gen, "text": text, **A.classify(text, r["gold"], r["competitors"], cities()),
            "vector_sha256": st.vector_sha(v)}


def run_arms_dev(e: Engine, rows: list[dict], root: Path, deadline: float | None) -> int:
    """Gate-tune part only: R1 at every α on the H1 abstention pool, LF at every α on the F1 rows, 20 no-op replays."""
    t0 = time.time()
    P = load_probes(root)
    meta = read_json(root / "probe" / "probe_dev.json")
    by = {r["case_id"]: r for r in rows}
    la, lf = meta["layer_ans"], meta["layer_f1"]
    jobs = [(cid, "R1", a) for cid in sorted(read_json(root / "probe" / "tune_pool.json")) for a in A.ALPHAS]
    jobs += [(cid, "LF", a) for cid in sorted(read_json(root / "probe" / "tune_f1.json")) for a in A.ALPHAS]
    jobs += [(r["case_id"], "noop", None) for r in noop_rows(rows, root, "dev")]
    for cid, arm, alpha in jobs:
        path = root / "arms_dev" / (f"{cid}__{arm}" + ("" if alpha is None else f"_a{alpha:g}") + ".json")
        if path.exists():
            continue
        if deadline and time.time() - t0 > deadline:
            return EXIT_DEADLINE
        r, b = by[cid], baseline(root, cid)
        if arm == "R1":
            h = resid(root, cid)[1, la]
            c_hat = str(P["models"]["ans"].predict(h[None])[0])
            rec = r1_trial(e, r, b, h, P, c_hat, la, alpha)
        elif arm == "LF":
            rec = f1_trial(e, r, b, root, P, lf, alpha, "LF")
        else:
            rec = noop_trial(e, r, b)
        write_json(path, {"case_id": cid, "arm": arm, **rec})
    return 0


def dev_arm(root: Path, cid: str, arm: str, alpha) -> dict:
    return read_json(root / "arms_dev" / (f"{cid}__{arm}" + ("" if alpha is None else f"_a{alpha:g}") + ".json"))


def make_design(rows: list[dict], root: Path, cases: Path | None = None) -> dict:
    """B4: α (R1) by the false-gold minus genuine-unsupported objective, α_LF, τ_c and τ_s (U at λ 1, ties to the
    higher τ), all on the gate-tune part; every value and the weights / centroids sha256 go into design.json."""
    meta = read_json(root / "probe" / "probe_dev.json")
    pool = read_json(root / "probe" / "tune_pool.json")
    tf1 = read_json(root / "probe" / "tune_f1.json")
    ids = sorted(pool)
    out = {a: {c: dev_arm(root, c, "R1", a)["outcome"] for c in ids} for a in A.ALPHAS}
    al = A.choose_alpha({a: [out[a][c] for c in ids if pool[c]["set"] == "ANS"] for a in A.ALPHAS},
                        {a: [out[a][c] for c in ids if pool[c]["set"] == "UNA"] for a in A.ALPHAS})
    arm = [out[al["alpha"]][c] for c in ids]
    base = ["abstain"] * len(ids)
    tc = A.choose_tau([pool[c]["content_max"] for c in ids], base, arm)
    ts = A.choose_tau([pool[c]["suff"] for c in ids], base, arm)
    wrong = [c for c in sorted(tf1) if tf1[c]["outcome"] != "gold"]
    gold = [c for c in sorted(tf1) if tf1[c]["outcome"] == "gold"]
    lf = A.choose_alpha_lf({a: [dev_arm(root, c, "LF", a)["outcome"] for c in wrong] for a in A.ALPHAS},
                           {a: [dev_arm(root, c, "LF", a)["outcome"] for c in gold] for a in A.ALPHAS})
    noop = [dev_arm(root, r["case_id"], "noop", None)["reproduces"] for r in noop_rows(rows, root, "dev")]
    design = {"method": A.METHOD_VERSION, "proposal": "rev 0b", "cases49_sha256": hashlib.sha256((cases or CASES).read_bytes()).hexdigest(),
              "layer_pre": meta["layer_pre"], "layer_f1": meta["layer_f1"], "layer_ans": meta["layer_ans"],
              "layer_suff": meta["layer_suff"], "alpha": al["alpha"], "alpha_objective": al["objective_by_alpha"],
              "alpha_lf": lf["alpha"], "alpha_lf_objective": lf["objective_by_alpha"],
              "tau_c": tc["tau"], "tau_s": ts["tau"], "U_tune_tau_c": tc["U"], "U_tune_tau_s": ts["U"],
              "n_tune_pool": len(ids), "n_tune_f1_wrong": len(wrong), "n_tune_f1_gold": len(gold),
              "r1_estimable": meta["r1_estimable"], "mu_ans_source": meta["mu_ans_source"],
              "weights_sha256": meta["weights_sha256"], "centroids_sha256": meta["centroids_sha256"],
              "r2": {"label": "R2 (one-position)", "layer": A.R2_LAYER, "alpha": A.R2_ALPHA, "site": "P_ans only",
                     "steer_sha256": A.r2_vector()["steer_sha256"]},
              "dev_noop_reproduces": f"{sum(noop)}/{len(noop)}",
              "probe_C": A.PROBE_C, "max_new_release": A.MAX_NEW_RELEASE, "dev_parts": "k = index // 2; k mod 3"}
    write_json(root / "design.json", design)
    return {"design_sha256": A.design_sha(design), **design}


def mark_design(root: Path, commit: str) -> dict:
    """The marker the held stages check against git (abst49.require_design): commit, design sha, repo-relative path."""
    design = read_json(root / "design.json")
    try:
        rel = A.repo_relpath(root / "design.json")[1]
    except PermissionError:
        rel = None
    m = {"commit": commit, "design_sha256": A.design_sha(design), "path": rel}
    write_json(root / A.DESIGN_MARKER, m)
    return m


def held_plan(rows: list[dict], root: Path) -> list[tuple[str, str]]:
    """(case_id, arm) for the held arms, from the held baselines (read here only to route, never printed)."""
    held = [r for r in rows if r["split"] == "held"]
    B = {r["case_id"]: baseline(root, r["case_id"]) for r in held}
    plan = []
    for r in held:
        b = B[r["case_id"]]
        if r["format"] != "F2" or r["set"] == "CF":
            continue
        if b["outcome"] == "abstain" and b["p_ans_ok"]:
            arms = ["R1", "R2", "R3", "R4"] + (["Dgold"] if r["set"] == "ANS" else [])
            plan += [(r["case_id"], a) for a in arms]
        elif r["set"] == "ANS" and b["outcome"] == "gold":
            plan.append((r["case_id"], "R4"))
    for cid in m2b_cf(held, B):
        plan += [(cid, "R1"), (cid, "R2")]
    plan += [(r["case_id"], a) for r in held if r["format"] == "F1" for a in ("LF", "LFrand")]
    plan += [(r["case_id"], "noop") for r in noop_rows(rows, root, "held")]
    return plan


def m2b_cf(rows_split: list[dict], B: dict) -> list[str]:
    """B2: CF-H1 case ids of bases whose ANS-F2-H1 and CF-F2-H1 baselines both abstain (with a P_ans)."""
    ab = lambda r: B[r["case_id"]]["outcome"] == "abstain" and B[r["case_id"]]["p_ans_ok"]
    ans = {r["base_key"] for r in rows_split if r["set"] == "ANS" and r["format"] == "F2" and r["hardness"] == "H1"
           and ab(r)}
    return [r["case_id"] for r in rows_split if r["set"] == "CF" and r["base_key"] in ans and ab(r)]


def run_arms_held(e: Engine, rows: list[dict], root: Path, deadline: float | None) -> int:
    design = A.require_design(root)
    meta = read_json(root / "probe" / "probe_dev.json")
    if hashlib.sha256((root / "probe" / "probes.pkl").read_bytes()).hexdigest() != design["weights_sha256"] or \
            meta["centroids_sha256"] != design["centroids_sha256"]:
        raise PermissionError("probes / centroids differ from the committed design")
    P = load_probes(root)
    r2 = A.r2_vector()
    by = {r["case_id"]: r for r in rows}
    t0 = time.time()
    la, lf, alpha = design["layer_ans"], design["layer_f1"], design["alpha"]
    for cid, arm in held_plan(rows, root):
        path = root / "arms_held" / f"{cid}__{arm}.json"
        if path.exists():
            continue
        if deadline and time.time() - t0 > deadline:
            return EXIT_DEADLINE
        r, b = by[cid], baseline(root, cid)
        rec = {"case_id": cid, "arm": arm}
        if arm == "R4":
            g = baseline_row(e, r, root, prompt_key="prompt_r4")
            rec.update({k: g[k] for k in ("gen_ids", "text", "answer", "outcome", "item", "gen_s")})
        elif arm in ("LF", "LFrand"):
            rec.update(f1_trial(e, r, b, root, P, lf, design["alpha_lf"], arm))
        elif arm == "noop":
            rec.update(noop_trial(e, r, b))
        elif arm == "R2":
            rec.update({"label": "R2 (one-position)", "layer": A.R2_LAYER, "alpha": A.R2_ALPHA,
                        "vector_sha256": st.vector_sha(r2["steer"]),
                        **release(e, r, b, [st.add_spec(A.R2_LAYER, r2["steer"], steps=(0,))])})
        else:
            h = resid(root, cid)[1, la]
            c_hat = str(P["models"]["ans"].predict(h[None])[0])
            target = r["gold"] if arm == "Dgold" else c_hat
            rec.update({"c_hat": c_hat, **r1_trial(e, r, b, h, P, target, la, alpha,
                                                   random_arm="R3" if arm == "R3" else None)})
        write_json(path, rec)
    return 0


# ---------------------------------------------------------------------------------------------- timing / smoke

def timing(e: Engine, rows: list[dict], root: Path) -> dict:
    """ONE unit on the first dev ANS-F2 H1 row: its baseline (160 max, Answer-line stop), the captures at both sites,
    one release (a matched-norm random add at layer 18, step 0, 16 max). Prints seconds only."""
    r = next(x for x in rows if x["split"] == "dev" and x["set"] == "ANS" and x["format"] == "F2"
             and x["hardness"] == "H1")
    troot = root / "timing"
    (troot / "resid").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = baseline_row(e, r, troot)
    t_base = time.perf_counter() - t0
    t_rel = None
    if b["p_ans_ok"]:
        h = resid(troot, r["case_id"])[1, 18]
        v = A.random_like(float(np.linalg.norm(h)), len(h), r["case_id"], "timing")
        t1 = time.perf_counter()
        release(e, r, b, [st.add_spec(18, v, steps=(0,))])
        t_rel = time.perf_counter() - t1
    f1 = next(x for x in rows if x["split"] == "dev" and x["format"] == "F1")
    t2 = time.perf_counter()
    b1 = baseline_row(e, f1, troot)
    t_f1 = time.perf_counter() - t2
    out = {"case_id": r["case_id"], "n_prompt": b["n_prompt"], "n_generated": len(b["gen_ids"]),
           "generate_s": b["gen_s"], "capture_s": b["cap_s"], "baseline_unit_s": round(t_base, 3),
           "release_s": None if t_rel is None else round(t_rel, 3), "p_ans_ok": b["p_ans_ok"],
           "f1_case": f1["case_id"], "f1_unit_s": round(t_f1, 3), "f1_generated": len(b1["gen_ids"]),
           "s_per_token_f2": round(b["gen_s"] / max(1, len(b["gen_ids"])), 4)}
    write_json(troot / "timing.json", out)
    print(json.dumps(out))
    return out


def smoke_cases(rows: list[dict], n_per_task: int) -> list[dict]:
    """D-22 subset from DEV bases only: per task the first n dev bases (k = 0 .. n-1); k 0-2 stay dev (fit, select,
    gate-tune) and k >= 3 are relabelled split "held" (stand-ins, never real held rows)."""
    out = []
    for r in rows:
        if r["split"] != "dev":
            continue
        k = r["base_index"] // 2
        if k < n_per_task:
            out.append({**r, "split": "dev" if k < 3 else "held", "smoke_relabelled": k >= 3})
    return out


def smoke_commit(root: Path) -> str:
    """The smoke's design commit lives in a throwaway git repo inside the smoke dir (never the project repo), so the
    held guard runs its real git checks."""
    import subprocess
    g = ["git", "-C", str(root), "-c", "user.name=smoke", "-c", "user.email=smoke@localhost"]
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(g + ["add", "-f", "design.json"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "smoke design", "--allow-empty"], check=True)
    return subprocess.run(g + ["rev-parse", "HEAD"], check=True, capture_output=True).stdout.decode().strip()


def smoke(e: Engine, rows: list[dict], root: Path, n: int) -> int:
    """D-22: the whole chain on smoke_cases (dev bases only), centroid minimums relaxed to 1, a placeholder design
    marker (smoke dir only), then the report. Checks: every no-op replay reproduces its baseline continuation token for
    token; every arm file exists; the report is written. Prints timings and counts (dev-base outcomes only).
    Exit 5 on a failed check."""
    import exp49_report as REP
    sub = smoke_cases(rows, max(4, n))
    (root / "resid").mkdir(parents=True, exist_ok=True)
    cases = root / "cases_smoke.jsonl"
    cases.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sub))
    A.MU_MIN, A.MU_ABST_MIN, A.NOOP_PER_SPLIT = 1, 1, 4
    timing_s = {}
    t = time.time(); run_baselines(e, [r for r in sub if r["split"] == "dev"], root, None); timing_s["baselines_dev"] = time.time() - t
    t = time.time(); probe_dev(sub, root); timing_s["probe_dev"] = time.time() - t
    t = time.time(); run_arms_dev(e, sub, root, None); timing_s["arms_dev"] = time.time() - t
    design = make_design(sub, root, cases)
    mark_design(root, smoke_commit(root))
    t = time.time(); run_baselines(e, [r for r in sub if r["split"] == "held"], root, None); timing_s["baselines_held"] = time.time() - t
    t = time.time(); run_arms_held(e, sub, root, None); timing_s["arms_held"] = time.time() - t
    plan = held_plan(sub, root)
    missing = [f"{c}__{a}" for c, a in plan if not (root / "arms_held" / f"{c}__{a}.json").exists()]
    noops = [read_json(root / "arms_held" / f"{c}__noop.json")["reproduces"] for c, a in plan if a == "noop"]
    noops += [dev_arm(root, r["case_id"], "noop", None)["reproduces"] for r in noop_rows(sub, root, "dev")]
    rep = REP.held_report(sub, root)
    write_json(root / "report.json", rep)
    ok = not missing and bool(noops) and all(noops)
    n_rows = len(sub)
    payload = {"n_rows": n_rows, "timing_s": timing_s, "per_row_baseline_s": (timing_s["baselines_dev"] +
               timing_s["baselines_held"]) / n_rows, "plan_arms": dict(Counter(a for _, a in plan)),
               "missing": missing, "noop_reproduces": f"{sum(noops)}/{len(noops)}", "design_alpha": design["alpha"],
               "pass": ok}
    write_json(root / "smoke.json", payload)
    print(json.dumps(payload))
    return 0 if ok else 5


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("timing", "baselines", "probe-dev", "arms-dev", "design", "mark-design",
                                        "arms-held"))
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--split", choices=("dev", "held"), default="dev")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--backend", choices=("hf", "stub"), default="hf")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--limit", type=int, default=None, help="stub chain only")
    args = ap.parse_args(argv)
    rows = load_jsonl(args.cases)
    root = args.out
    (root / "resid").mkdir(parents=True, exist_ok=True)

    if args.stage == "probe-dev":
        out = probe_dev(rows, root)
        print(json.dumps({k: out[k] for k in ("layer_pre", "layer_f1", "layer_ans", "layer_suff", "n_dev_pool")}))
        return 0
    if args.stage == "design":
        d = make_design(rows, root, args.cases)
        print(json.dumps({"design_sha256": d["design_sha256"], "alpha": d["alpha"], "tau_c": d["tau_c"],
                          "tau_s": d["tau_s"]}))
        return 0
    if args.stage == "mark-design":
        if not args.commit:
            print("--commit is required", file=sys.stderr)
            return 2
        print(json.dumps(mark_design(root, args.commit)))
        return 0
    if args.stage == "arms-held" or (args.stage == "baselines" and args.split == "held"):
        try:
            A.require_design(root)
        except PermissionError as err:
            print(f"design guard: {err}", file=sys.stderr)
            return EXIT_DESIGN

    e = load_engine(args.backend, args.threads, args.dtype)
    if args.smoke:
        if root.resolve().is_relative_to((ROOT / "runs" / "exp49").resolve()):
            print("smoke outputs go outside runs/exp49/ (e.g. runs/exp49_smoke/)", file=sys.stderr)
            return 2
        return smoke(e, rows, root, args.smoke)
    if args.stage == "timing":
        timing(e, rows, root)
        return 0
    if args.stage == "baselines":
        todo = [r for r in rows if r["split"] == args.split]
        if args.limit:
            todo = todo[:args.limit]
        return run_baselines(e, todo, root, args.deadline_s)
    if args.stage == "arms-dev":
        return run_arms_dev(e, rows, root, args.deadline_s)
    if args.stage == "arms-held":
        return run_arms_held(e, rows, root, args.deadline_s)
    return 2


if __name__ == "__main__":
    sys.exit(main())
