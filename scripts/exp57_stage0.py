#!/usr/bin/env python
"""§57 stage 0 (proposal rev 0, 9e51a1b, §5 stage 0): memory/time of forward + backward, ā^C over the note, the
attribution-patching validity gate, and the batched-arm equality test. Laptop, $0, dev-A only.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage0.py timing [--grad-from L]
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage0.py cmean-note      # runs/exp57/c_mean_note.npy
    .venv/bin/python scripts/exp57_stage0.py freeze                           # runs/exp57/freeze.json
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage0.py validity         # runs/exp57/validity.json
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage0.py equality         # runs/exp57/equality.json

Fixed here (before the gate is read):
- Timing row: the longest dev-A class row (W-DE30_UNArel_H2, 721 ids). Gradients from the embedding (all 36 layers);
  if peak RSS > 21 GB, from layer 16 (recorded), and if that does not fit, stop.
- Validity rows: the dev-A FA rows in rows.jsonl order, positions 0, 28, 56, 84, 112 (5 rows).
  Per row and region (P, N): the 20 neurons with the largest |Δ̂| and 20 random neurons (uniform over the rest,
  numpy default_rng(5700 + position)); real ΔD for each single reset (alpha = 1). Set patch: the 50 neurons with the
  most negative Δ̂ on the row, reset together; predicted = their summed Δ̂.
  PASS per region: Spearman(Δ̂, real ΔD) over its 200 single pairs >= 0.6, and the set patch's real ΔD has the sign of
  its predicted value on >= 8 of the 10 set patches. NOTE: the proposal's "on >= 8 of 10" counts the 10 set patches of
  both regions together; the per-region count (of 5) is reported beside it.
- Equality rows: the dev-A FA rows at positions 7, 21, ..., 133 (10 rows, none in the validity set). Arms = the stage-3
  FA arms with stand-in sets: base; K_P* and K_N* = the top 50 by mean −Δ̂ over the 5 validity rows (per region), reset
  to ā^C; 20 random sets per region matched to the stand-in's per-layer counts (default_rng(5800)); B1 = −8 on the
  " unknown" first token at step 0. 44 arms per row.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp56_common as X56  # noqa: E402
import exp57_common as X  # noqa: E402

OUT = X.OUT
INTER, NL = 11008, 36
TIMING_ROW = "W-DE30_UNArel_H2"
VALID_POS = (0, 28, 56, 84, 112)
EQ_POS = tuple(range(7, 140, 14))
N_TOP, N_RAND, N_SET = 20, 20, 50
RSS_CAP = 21.0


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def dev_a(cls=None):
    import exp56_stage1 as S1
    rows = S1.dev_a_rows()
    return [r for r in rows if cls is None or r["cls"] == cls]


def case_rec(cid):
    cases = {c["case_id"]: c for c in X56.cases()}
    r = cases[cid]
    rec, _ = X56.record_for(r)
    return r, rec


def resets(need_n=True):
    rp = np.load(X.C_MEAN_P).reshape(NL, INTER)
    rn = np.load(X.C_MEAN_N).reshape(NL, INTER) if need_n else np.zeros((NL, INTER), np.float32)
    return rp, rn


def grad_from_frozen():
    return json.loads((OUT / "timing.json").read_text())["grad_from"]


def stage_timing(e, grad_from):
    r, rec = case_rec(TIMING_ROW)
    rp, _ = resets(need_n=False)
    at = X.Attributor(e, rp, np.zeros_like(rp), grad_from=grad_from)
    cids = X56.city_ids(e.encode, r)
    t0 = time.time()
    res = at.run(rec, cids)
    dt = time.time() - t0
    rss = EN.peak_rss_gb()
    out = {"row": TIMING_ROW, "n_ids": len(CM.prefix_ids(rec)), "grad_from": grad_from, "fwd_bwd_s": round(dt, 2),
           "peak_rss_gb": round(rss, 2), "fits": rss <= RSS_CAP, "D": res["D"]}
    log(json.dumps(out))
    p = OUT / "timing_runs.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(out) + "\n")
    if out["fits"]:
        (OUT / "timing.json").write_text(json.dumps(out, indent=1))


def stage_cmean_note(e):
    rows = dev_a("C")
    tot, n = np.zeros((NL, INTER), np.float64), 0
    t0 = time.time()
    for k, x in enumerate(rows):
        _, rec = case_rec(x["case_id"])
        s, c = X.note_sums(e, rec)
        tot += s; n += c
        if (k + 1) % 25 == 0 or k + 1 == len(rows):
            log(f"{k + 1}/{len(rows)} C rows, {(time.time() - t0) / (k + 1):.1f} s/row, peak RSS {EN.peak_rss_gb():.1f}")
    np.save(X.C_MEAN_N, (tot / n).astype(np.float32))
    (OUT / "c_mean_note.json").write_text(json.dumps({"n_rows": len(rows), "n_positions": n,
                                                      "sha256": X56.sha_file(X.C_MEAN_N)}, indent=1))


def stage_freeze():
    files = ["scripts/exp57_common.py", "scripts/exp57_arms.py", "scripts/exp57_stage0.py", "scripts/exp57_stage1.py",
             "tests/test_exp57.py"]
    data = {"rows_exp56": "runs/exp56/rows.jsonl", "freeze_exp56": "runs/exp56/freeze.json",
            "c_mean_P": "runs/exp56/c_mean_act.npy", "c_mean_N": "runs/exp57/c_mean_note.npy",
            "timing": "runs/exp57/timing.json"}
    fa = dev_a("FA")
    fz = {"proposal": "research/proposals/2026-11-10-exp57-differential-neurons.md rev 0 (9e51a1b)",
          "code": {f: X56.sha_file(ROOT / f) for f in files},
          "data": {k: {"path": v, "sha256": X56.sha_file(ROOT / v)} for k, v in data.items()},
          "grad_from": grad_from_frozen(),
          "validity_rows": [fa[i]["case_id"] for i in VALID_POS],
          "equality_rows": [fa[i]["case_id"] for i in EQ_POS],
          "validity_rule": {"n_top": N_TOP, "n_rand": N_RAND, "rand_seed": "5700 + position", "n_set": N_SET,
                            "spearman_min": 0.6, "set_sign_min": "8 of 10 (both regions pooled; per region of 5 "
                                                                "reported)"},
          "equality_rule": {"arms": "base, K_P*, K_N*, 20 random P, 20 random N, B1 -8", "rand_seed": 5800,
                            "same_min": 0.99, "dD_max": 1e-3},
          "regions": "P = {P_ans}; N = [n_prompt, P_ans)"}
    (OUT / "freeze.json").write_text(json.dumps(fz, indent=1))
    log(f"freeze written, sha {X56.sha_file(OUT / 'freeze.json')[:16]}")


def stage_validity(e):
    from scipy.stats import spearmanr, pearsonr
    gf = grad_from_frozen()
    rp, rn = resets()
    at = X.Attributor(e, rp, rn, grad_from=gf)
    fa = dev_a("FA")
    lo_flat = gf * INTER
    out_rows, pairs = [], {"P": [], "N": []}
    sets = []
    mean_neg = {"P": np.zeros(NL * INTER), "N": np.zeros(NL * INTER)}
    for pos in VALID_POS:
        cid = fa[pos]["case_id"]
        r, rec = case_rec(cid)
        t0 = time.time()
        res = at.run(rec, X56.city_ids(e.encode, r))
        t_grad = time.time() - t0
        pt = X.Patcher(e, rec, at.unk_id, res["t_star"])
        rng = np.random.default_rng(5700 + pos)
        row = {"case_id": cid, "D": res["D"], "D_clean_P": pt.D_clean_P, "D_clean_N": pt.D_clean_N,
               "t_grad_s": round(t_grad, 1), "n_note": res["p_ans"] - res["n_prompt"]}
        t1 = time.time()
        for reg, dh, rv in (("P", res["dhat_P"], rp), ("N", res["dhat_N"], rn)):
            f = dh.ravel().astype(np.float64)
            mean_neg[reg] += -f / len(VALID_POS)
            top = np.argsort(-np.abs(f))[:N_TOP]
            pool = np.setdiff1d(np.arange(lo_flat, NL * INTER), top)
            rnd = rng.choice(pool, N_RAND, replace=False)
            for kind, idxs in (("top", top), ("rand", rnd)):
                for k in idxs:
                    l, i = divmod(int(k), INTER)
                    real = pt.effect(reg, [(l, [i], [rv[l, i]])])
                    pairs[reg].append({"case_id": cid, "kind": kind, "layer": l, "neuron": i, "dhat": float(f[k]),
                                       "real": real})
            S = np.argsort(f)[:N_SET]                                      # most negative Δ̂
            spec = {}
            for k in S:
                l, i = divmod(int(k), INTER)
                spec.setdefault(l, []).append(i)
            real = pt.effect(reg, [(l, ii, rv[l, ii]) for l, ii in spec.items()])
            pred = float(f[S].sum())
            sets.append({"case_id": cid, "region": reg, "pred": pred, "real": real,
                         "sign_ok": bool(np.sign(pred) == np.sign(real))})
        row["t_patch_s"] = round(time.time() - t1, 1)
        out_rows.append(row)
        log(f"{cid}: D {res['D']:.3f}, grad {t_grad:.0f} s, patches {row['t_patch_s']:.0f} s; sets "
            + ", ".join(f"{s['region']} pred {s['pred']:.2f} real {s['real']:.2f}" for s in sets[-2:]))
    rep = {"grad_from": gf, "rows": out_rows, "regions": {}}
    for reg in ("P", "N"):
        dh = np.array([p["dhat"] for p in pairs[reg]]); rl = np.array([p["real"] for p in pairs[reg]])
        kt = np.array([p["kind"] for p in pairs[reg]])
        rho = float(spearmanr(dh, rl)[0])
        ss = [s for s in sets if s["region"] == reg]
        rep["regions"][reg] = {
            "n_pairs": len(dh), "spearman": rho, "pearson": float(pearsonr(dh, rl)[0]),
            "spearman_top": float(spearmanr(dh[kt == "top"], rl[kt == "top"])[0]),
            "spearman_rand": float(spearmanr(dh[kt == "rand"], rl[kt == "rand"])[0]),
            "sign_agree_top": float(np.mean(np.sign(dh[kt == "top"]) == np.sign(rl[kt == "top"]))),
            "median_ratio_real_over_dhat_top": float(np.median(rl[kt == "top"] / dh[kt == "top"])),
            "set_sign_ok": sum(s["sign_ok"] for s in ss), "n_sets": len(ss),
            "spearman_pass": rho >= 0.6}
    n_ok = sum(s["sign_ok"] for s in sets)
    rep["set_sign_ok_total"] = f"{n_ok}/{len(sets)}"
    for reg in ("P", "N"):
        rep["regions"][reg]["PASS"] = bool(rep["regions"][reg]["spearman_pass"] and n_ok >= 8)
    rep["sets"] = sets
    (OUT / "validity.json").write_text(json.dumps(rep, indent=1))
    (OUT / "validity_pairs.json").write_text(json.dumps(pairs, indent=1))
    np.save(OUT / "standin_mean_neg_dhat.npy", np.stack([mean_neg["P"], mean_neg["N"]]).astype(np.float32))
    log(json.dumps({k: v for k, v in rep.items() if k in ("regions", "set_sign_ok_total")}))


def standin_arms(rp, rn, unk_id):
    m = np.load(OUT / "standin_mean_neg_dhat.npy")
    rng = np.random.default_rng(5800)
    arms = [{"name": "base"}]
    for reg, mm, rv in (("P", m[0], rp), ("N", m[1], rn)):
        K = np.argsort(-mm)[:N_SET]
        lay = np.bincount(K // INTER, minlength=NL)
        key = f"reset_{reg}"
        spec = lambda ks: [(l, [int(k % INTER) for k in ks if k // INTER == l],
                            [float(rv[l, k % INTER]) for k in ks if k // INTER == l]) for l in sorted({int(k // INTER) for k in ks})]
        arms.append({"name": f"K_{reg}*", key: spec(K)})
        for j in range(20):
            ks = []
            for l in np.where(lay > 0)[0]:
                pool = np.setdiff1d(np.arange(INTER), K[K // INTER == l] % INTER)
                ks += [l * INTER + int(i) for i in rng.choice(pool, lay[l], replace=False)]
            arms.append({"name": f"R_{reg}{j:02d}", key: spec(np.array(ks))})
    arms.append({"name": "B1", "bias": {"ids": [unk_id], "value": -8.0}})
    return arms


def stage_equality(e):
    import exp57_arms as A
    rp, rn = resets()
    unk = e.encode(X56.UNK_TEXT)[0]
    arms = standin_arms(rp, rn, unk)
    fa = dev_a("FA")
    out, tb_all, tr_all = [], 0.0, 0.0
    for pos in EQ_POS:
        cid = fa[pos]["case_id"]
        r, rec = case_rec(cid)
        t_star = int(json.loads(str(np.load(ROOT / "runs/exp56/run/stage1" / f"{cid}.npz")["meta"]))["t_star"])
        rows, tb, tr = A.compare(e, rec, arms, unk, t_star)
        tb_all += tb; tr_all += tr
        for x in rows:
            x["case_id"] = cid
        out += rows
        ns = sum(x["same"] for x in rows)
        log(f"{cid}: {ns}/{len(rows)} same, max dD {max(x['dD'] for x in rows):.2e}, batched {tb:.0f} s, "
            f"batch-1 {tr:.0f} s; changed vs base: {sum(x['batched'] != rows[0]['batched'] for x in rows)}")
    same = np.mean([x["same"] for x in out])
    rep = {"n_row_arms": len(out), "same_frac": float(same), "max_dD": float(max(x["dD"] for x in out)),
           "PASS": bool(same >= 0.99 and max(x["dD"] for x in out) <= 1e-3),
           "differences": [x for x in out if not x["same"] or x["dD"] > 1e-3],
           "batched_s_per_row": tb_all / len(EQ_POS), "batch1_s_per_row": tr_all / len(EQ_POS),
           "n_arms": len(arms),
           "outputs_changed_vs_base": sum(x["batched"] != out[k - k % len(arms)]["batched"] for k, x in enumerate(out)),
           "peak_rss_gb": EN.peak_rss_gb()}
    (OUT / "equality.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (OUT / "equality_rows.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(json.dumps({k: v for k, v in rep.items() if k != "differences"}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["timing", "cmean-note", "freeze", "validity", "equality"])
    ap.add_argument("--grad-from", type=int, default=0)
    a = ap.parse_args(argv)
    if a.stage == "freeze":
        return stage_freeze()
    e = EN.load("hf")
    log(f"engine {e.name} loaded in {e.load_s:.0f} s")
    if a.stage == "timing":
        return stage_timing(e, a.grad_from)
    if a.stage == "cmean-note":
        return stage_cmean_note(e)
    if a.stage == "validity":
        return stage_validity(e)
    return stage_equality(e)


if __name__ == "__main__":
    sys.exit(main())
