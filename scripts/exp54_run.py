#!/usr/bin/env python
"""§54 runner: twin pairs and component patching for false "unknown" (Qwen2.5-3B, laptop, $0).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp54_run.py --stage b0        # B0 on the 900 W ANS rows
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp54_run.py --stage pairs     # flip pairs, forced note, D, gate, split

Spec: research/proposals/2026-11-07-exp54-native-switch.md rev 0; every rule comes from the stage-0 freeze
runs/exp54/freeze.json (66fcafe). Stages 2-4 are added after the stage-1 gate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_stage4 as S4  # noqa: E402

OUT = ROOT / "runs/exp54"
RUN = OUT / "run"
FREEZE = json.loads((OUT / "freeze.json").read_text())
SPLIT = FREEZE["data"]["split_name"]
UNK = FREEZE["definitions"]["unknown_token"]["id"]
CASES = OUT / "cases54.jsonl"
PAIRS = OUT / "pairs.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def all_rows() -> list[dict]:
    rows = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    assert all(r["split"] == SPLIT and r["seed"] == FREEZE["seeds"]["dev"] for r in rows)
    return rows


def b0(cid: str) -> dict | None:
    return CM.read_json(S4.b0_path(RUN, SPLIT, cid))


# ---------------------------------------------------------------------------------------------- stage 1

def stage_b0(e, limit: int | None = None) -> None:
    tc = CM.t_cities()
    rows = sorted([r for r in all_rows() if r["set"] == "ANS"], key=lambda r: r["case_id"])
    if limit:
        rows = rows[:limit]
    t0 = time.time()
    S4.run_b0(e, rows, RUN, SPLIT, tc)
    log(f"B0 on {len(rows)} ANS rows done in {time.time() - t0:.0f} s; peak RSS {S4.EN.peak_rss_gb():.1f} GB")


def d_at(e, ids: list[int], t_star: int) -> float:
    p = len(ids) - 1
    lg = e.read(ids, positions=[p], logit_positions=[p], logit_ids=[UNK, t_star])["logits"][0]
    return float(lg[0] - lg[1])


def flip_pairs(rows: list[dict]) -> tuple[list[dict], Counter]:
    by = {r["case_id"]: r for r in rows}
    out, why, stable = [], Counter(), []
    for r in rows:
        if r["set"] != "ANS" or r["rep"] != "a":
            continue
        a_id, b_id = r["case_id"], r["case_id"].replace("_ANSa_", "_ANSb_")
        ra, rb = b0(a_id), b0(b_id)
        if ra is None or rb is None:
            why["missing_b0"] += 1
            continue
        cls = {ra["cls"], rb["cls"]}
        if cls == {"gold"}:
            stable.append({"base_key": r["base_key"], "task": r["task"], "hardness": r["hardness"], "a": a_id,
                           "b": b_id})
            why["both_gold"] += 1
            continue
        if cls != {"gold", "abstention"}:
            why["no_flip"] += 1
            continue
        A, F = (a_id, b_id) if ra["cls"] == "gold" else (b_id, a_id)
        out.append({"base_key": r["base_key"], "task": r["task"], "hardness": r["hardness"], "A": A, "F": F,
                    "A_rep": A.split("_ANS")[1][0]})
    return out, why, stable


def stable_pairs(e, cands: list[dict], by: dict, tc) -> list[dict]:
    """freeze definitions.stable_pair: both gold under own B0, P_ans ok, equal prompt length; b gets rep a's note ids
    and must still grade gold (a's controlled prefix is its own B0 prefix)."""
    res = []
    for k, p in enumerate(cands):
        ra, rb = b0(p["a"]), b0(p["b"])
        q = dict(p)
        if not (ra["p_ans_ok"] and rb["p_ans_ok"]) or len(ra["prompt_ids"]) != len(rb["prompt_ids"]):
            q["stable"], q["why"] = False, "no P_ans or prompt lengths differ"
            res.append(q)
            continue
        b_ids = list(rb["prompt_ids"]) + list(ra["gen_ids"][:ra["j"]])
        g = e.generate(b_ids, mode="prefix")
        q["forced_b_text"], q["forced_b_cls"] = g.text, CM.grade(by[p["b"]], e.decode(ra["gen_ids"][:ra["j"]]) + g.text, tc)["cls"]
        q["stable"] = q["forced_b_cls"] == "gold"
        res.append(q)
        if (k + 1) % 20 == 0:
            log(f"stable candidate {k + 1}/{len(cands)}")
    return res


def stage_pairs(e) -> dict:
    tc = CM.t_cities()
    rows = all_rows()
    by = {r["case_id"]: r for r in rows}
    flips, why, stable_c = flip_pairs(rows)
    log(f"{len(flips)} flip pairs; {dict(why)}")
    res = []
    for k, p in enumerate(flips):
        rA, rF = b0(p["A"]), b0(p["F"])
        q = dict(p)
        if not (rA["p_ans_ok"] and rF["p_ans_ok"]):
            q["kind"], q["why"] = "other", "no P_ans"
            res.append(q)
            continue
        if len(rA["prompt_ids"]) != len(rF["prompt_ids"]):
            q["kind"], q["why"] = "other", "prompt lengths differ"
            res.append(q)
            continue
        t_star = rA["gen_ids"][rA["j"]] if len(rA["gen_ids"]) > rA["j"] else None
        q["t_star"] = t_star
        q["t_star_text"] = e.decode([t_star]) if t_star is not None else None
        a_ids = CM.prefix_ids(rA)
        f_ids = list(rF["prompt_ids"]) + list(rA["gen_ids"][:rA["j"]])
        g = e.generate(f_ids, mode="prefix")
        gr = CM.grade(by[p["F"]], e.decode(rA["gen_ids"][:rA["j"]]) + g.text, tc)   # head + tail, as §51
        q["forced_F_text"], q["forced_F_cls"] = g.text, gr["cls"]
        q["n_prefix"] = len(f_ids)
        q["D_A"], q["D_F"] = d_at(e, a_ids, t_star), d_at(e, f_ids, t_star)
        q["D_F_own"] = d_at(e, CM.prefix_ids(rF), t_star)
        q["secondary_ok"] = q["D_A"] < 0 < q["D_F_own"]
        if gr["cls"] == "gold":
            q["kind"] = "note_driven"
        elif gr["cls"] == "abstention" and q["D_A"] < 0 < q["D_F"]:
            q["kind"] = "primary"
        else:
            q["kind"] = "other"
            q["why"] = f"forced F {gr['cls']}; D_A {q['D_A']:.2f}, D_F {q['D_F']:.2f}"
        res.append(q)
        log(f"pair {k + 1}/{len(flips)} {p['A']} {q['kind']}  D_A {q['D_A']:.2f}  D_F {q['D_F']:.2f}")
    prim = [q for q in res if q["kind"] == "primary"]
    split = assign_halves(res, rows)
    for q in res:
        q["half"] = split[q["base_key"]]
    stab = stable_pairs(e, stable_c, by, tc)
    for q in stab:
        q["half"] = split[q["base_key"]]
    gate = {"n_primary": len(prim), "n_bases": len({q["base_key"] for q in prim}),
            "n_tasks": len({q["task"] for q in prim}),
            "primary_by_task": dict(Counter(q["task"] for q in prim)),
            "primary_by_hardness": dict(Counter(q["hardness"] for q in prim)),
            "primary_by_half": dict(Counter(q["half"] for q in prim)),
            "bases_by_half": dict(Counter(split[b] for b in {q["base_key"] for q in prim}))}
    gate["pass"] = gate["n_primary"] >= 40 and gate["n_bases"] >= 20 and gate["n_tasks"] >= 2
    n_fa = sum(1 for r in rows if r["set"] == "ANS" and (b0(r["case_id"]) or {}).get("cls") == "abstention")
    summ = {"n_twin_candidates": sum(1 for r in rows if r["set"] == "ANS" and r["rep"] == "a"),
            "no_flip": dict(why), "n_flip": len(flips), "kinds": dict(Counter(q["kind"] for q in res)),
            "n_secondary": sum(1 for q in res if q.get("secondary_ok")),
            "n_fa_rows": n_fa, "share_fa_in_flip_pairs": len(flips) / n_fa if n_fa else None,
            "n_stable_by_half": dict(Counter(q["half"] for q in stab if q["stable"])),
            "n_stable_candidates": len(stab),
            "kinds_by_half": {h: dict(Counter(q["kind"] for q in res if q["half"] == h)) for h in ("dev-A", "dev-B")},
            "secondary_by_half": dict(Counter(q["half"] for q in res if q.get("secondary_ok"))),
            "gate": gate, "pairs": res, "stable": stab, "split": split}
    CM.write_json(PAIRS, summ)
    log(f"gate {gate}")
    return summ


def assign_halves(pairs: list[dict], rows: list[dict]) -> dict[str, str]:
    """freeze split_rule: primary-pair bases, per task sorted and shuffled, concatenated DE, RC, TU, alternating
    dev-A / dev-B; the other bases by the same procedure separately."""
    prim_bases = {q["base_key"] for q in pairs if q["kind"] == "primary"}
    all_bases = {(r["task"], r["base_key"]) for r in rows}
    out = {}
    for group in (prim_bases, {b for _, b in all_bases} - prim_bases):
        seq = []
        for task in ("DE", "RC", "TU"):
            bs = sorted(b for t, b in all_bases if t == task and b in group)
            random.Random(f"54-split-{task}").shuffle(bs)
            seq += bs
        for i, b in enumerate(seq):
            out[b] = "dev-A" if i % 2 == 0 else "dev-B"
    return out


def stage_regrade() -> dict:
    """Bug fix (2026-11-08): stage 1 first graded the forced-prefix tail alone, so every row read 'no_answer_line'.
    §51 grades head (note + marker) + tail. The stored generations and D values are regraded; no model call."""
    from transformers import AutoTokenizer
    import exp51_engine as EN
    tok = AutoTokenizer.from_pretrained(str(EN.SNAPSHOT))
    dec = lambda ids: tok.decode(list(ids), skip_special_tokens=True)  # noqa: E731
    tc = CM.t_cities()
    rows = all_rows()
    by = {r["case_id"]: r for r in rows}
    d = CM.read_json(PAIRS)
    d.setdefault("regrade_note", stage_regrade.__doc__)
    for q in d["pairs"]:
        if "forced_F_text" not in q:
            continue
        rA = b0(q["A"])
        gr = CM.grade(by[q["F"]], dec(rA["gen_ids"][:rA["j"]]) + q["forced_F_text"], tc)
        q["forced_F_cls"] = gr["cls"]
        q.pop("why", None)
        if gr["cls"] == "gold":
            q["kind"] = "note_driven"
        elif gr["cls"] == "abstention" and q["D_A"] < 0 < q["D_F"]:
            q["kind"] = "primary"
        else:
            q["kind"] = "other"
            q["why"] = f"forced F {gr['cls']}; D_A {q['D_A']:.2f}, D_F {q['D_F']:.2f}"
    for q in d["stable"]:
        if "forced_b_text" not in q:
            continue
        ra = b0(q["a"])
        q["forced_b_cls"] = CM.grade(by[q["b"]], dec(ra["gen_ids"][:ra["j"]]) + q["forced_b_text"], tc)["cls"]
        q["stable"] = q["forced_b_cls"] == "gold"
    res, stab = d["pairs"], d["stable"]
    prim = [q for q in res if q["kind"] == "primary"]
    split = assign_halves(res, rows)
    for q in res + stab:
        q["half"] = split[q["base_key"]]
    gate = {"n_primary": len(prim), "n_bases": len({q["base_key"] for q in prim}),
            "n_tasks": len({q["task"] for q in prim}),
            "primary_by_task": dict(Counter(q["task"] for q in prim)),
            "primary_by_hardness": dict(Counter(q["hardness"] for q in prim)),
            "primary_by_half": dict(Counter(q["half"] for q in prim)),
            "bases_by_half": dict(Counter(split[b] for b in {q["base_key"] for q in prim}))}
    gate["pass"] = gate["n_primary"] >= 40 and gate["n_bases"] >= 20 and gate["n_tasks"] >= 2
    d.update({"kinds": dict(Counter(q["kind"] for q in res)),
              "kinds_by_half": {h: dict(Counter(q["kind"] for q in res if q["half"] == h)) for h in ("dev-A", "dev-B")},
              "secondary_by_half": dict(Counter(q["half"] for q in res if q.get("secondary_ok"))),
              "n_stable_by_half": dict(Counter(q["half"] for q in stab if q["stable"])),
              "forced_F_cls": dict(Counter(q.get("forced_F_cls") for q in res)),
              "forced_b_cls": dict(Counter(q.get("forced_b_cls") for q in stab)),
              "gate": gate, "split": split})
    CM.write_json(PAIRS, d)
    log(f"regraded: kinds {d['kinds']}; stable {d['n_stable_by_half']}; gate {gate}")
    return d


# ---------------------------------------------------------------------------------------------- stage 2

TIMING = OUT / "timing.json"
STAGE2 = OUT / "stage2.json"
KFILE = OUT / "K.json"
BUDGET_H = 24.0


def pairs_of(kind: str = "primary", half: str | None = None) -> list[dict]:
    d = CM.read_json(PAIRS)
    return [q for q in d["pairs"] if q["kind"] == kind and (half is None or q["half"] == half)]


def sites(e, q, layers):
    import exp54_patch as P
    rA, rF = b0(q["A"]), b0(q["F"])
    a_ids = CM.prefix_ids(rA)
    f_ids = list(rF["prompt_ids"]) + list(rA["gen_ids"][:rA["j"]])
    A = P.Site(e, a_ids, q["t_star"], UNK, layers)
    F = P.Site(e, f_ids, q["t_star"], UNK, layers)
    for s_, ref in ((A, q["D_A"]), (F, q["D_F"])):
        if abs(s_.base.D - ref) > 1e-2:
            raise RuntimeError(f"{q['A']}: cached-step D {s_.base.D:.4f} != read D {ref:.4f}")
    return A, F


def gap_fwd(A, F, D):
    return (F.base.D - D) / (F.base.D - A.base.D)


def gap_rev(A, F, D):
    return (D - A.base.D) / (F.base.D - A.base.D)


def atp(A, F, comps):
    """estimated forward / reverse gap closed per component (first order)."""
    import exp54_patch as P
    _, gF = F.step(grad=True)
    _, gA = A.step(grad=True)
    out = {}
    for c in comps:
        l, h = c
        if h == "mlp":
            dF = float(gF["mlp"][l] @ (A.base.mlp[l] - F.base.mlp[l]))
            dA = float(gA["mlp"][l] @ (F.base.mlp[l] - A.base.mlp[l]))
        else:
            sl = slice(h * F.hd, (h + 1) * F.hd)
            dF = float(gF["z"][l][sl] @ (A.base.z[l][sl] - F.base.z[l][sl]))
            dA = float(gA["z"][l][sl] @ (F.base.z[l][sl] - A.base.z[l][sl]))
        gap = F.base.D - A.base.D
        out[P.comp_name(c)] = {"fwd": -dF / gap, "rev": dA / gap}
    return out


def real_single(A, F, comps):
    import exp54_patch as P
    out = {}
    for c in comps:
        out[P.comp_name(c)] = {"fwd": gap_fwd(A, F, F.step(F.spec([c], A.base)).D),
                               "rev": gap_rev(A, F, A.step(A.spec([c], F.base)).D)}
    return out


def stage_timing(e) -> dict:
    """freeze time_rule + review ruling 3: time one real dev-A primary pair, project stages 2 + 3, pick the branch."""
    import exp54_patch as P
    d = CM.read_json(PAIRS)
    devA, devB = pairs_of("primary", "dev-A"), pairs_of("primary", "dev-B")
    n_stab_B = sum(1 for q in d["stable"] if q["stable"] and q["half"] == "dev-B")
    q = devA[0]
    comps = P.components(range(20, 36))
    t = {}
    t0 = time.perf_counter(); A, F = sites(e, q, range(20, 36)); t["sites"] = time.perf_counter() - t0
    t0 = time.perf_counter(); atp(A, F, comps); t["atp"] = time.perf_counter() - t0
    t0 = time.perf_counter(); real_single(A, F, comps[:20]); t["step"] = (time.perf_counter() - t0) / 40
    t0 = time.perf_counter(); e.generate(F.ids, arm=F.gen_arm([comps[0]], A.base), mode="prefix")
    t["gen"] = time.perf_counter() - t0
    t["peak_rss_gb"] = S4.EN.peak_rss_gb()
    nA, nB = len(devA), len(devB)
    n_una_B = 3 * sum(1 for b, h in d["split"].items() if h == "dev-B")

    def project(nc: int, pool: int) -> dict:
        s2 = nA * (t["sites"] + t["atp"] + 2 * pool * t["step"]) + sum(pool - i for i in range(5)) * nA * t["step"]
        # stage 3: K and two named sets: fwd, rev, 20 random, all, all-but-K, mean transplant -> ~26 steps each,
        # 3 sets x 2 directions + mean transplant generations, stable-pair sites (criterion 1), UNA-rel B0 and
        # mean-transplant generations on dev-B UNA-rel and non-twin FA rows (upper bound: all dev-B ANS rows)
        s3 = (nB * (t["sites"] + 3 * 26 * t["step"] + 7 * t["gen"]) + n_stab_B * t["sites"]
              + n_una_B * 18.1 + n_una_B * t["gen"] + 6 * sum(1 for b, h in d["split"].items() if h == "dev-B")
              * t["gen"] / 3)
        return {"n_components": nc, "pool": pool, "stage2_h": s2 / 3600, "stage3_h": s3 / 3600,
                "total_h": (s2 + s3) / 3600}
    full272 = project(272, 272)
    frozen272 = project(272, 72)
    full160 = project(160, 160)
    frozen160 = project(160, 72)
    if full272["total_h"] <= BUDGET_H:
        branch = {"layers": [20, 35], "pool": "full", "why": "full grid at 272 fits 24 h"}
    elif frozen272["total_h"] <= BUDGET_H:
        branch = {"layers": [20, 35], "pool": "frozen", "why": "full grid does not fit; frozen pool at 272 fits"}
    elif full160["total_h"] <= BUDGET_H:
        branch = {"layers": [26, 35], "pool": "full", "why": "time rule: layers 26-35; full grid at 160 fits"}
    else:
        branch = {"layers": [26, 35], "pool": "frozen", "why": "time rule: layers 26-35; frozen pool"}
    out = {"pair": q["A"], "n_devA_primary": nA, "n_devB_primary": nB, "n_devB_stable": n_stab_B,
           "measured_s": t, "projections": {"full272": full272, "frozen272": frozen272, "full160": full160,
                                            "frozen160": frozen160}, "branch": branch,
           "budget_h": BUDGET_H, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    CM.write_json(TIMING, out)
    log(json.dumps(out))
    return out


def stage2(e) -> dict:
    import exp54_patch as P
    tm = CM.read_json(TIMING)
    if tm is None:
        raise SystemExit("timing.json missing: run --stage timing and commit it first")
    lo, hi = tm["branch"]["layers"]
    layers = range(lo, hi + 1)
    comps = P.components(layers)
    names = [P.comp_name(c) for c in comps]
    always = FREEZE["stage2_screen"]["always_included"]
    devA = pairs_of("primary", "dev-A")
    S = []
    t0 = time.time()
    for k, q in enumerate(devA):
        A, F = sites(e, q, layers)
        S.append((q, A, F))
        log(f"dev-A pair {k + 1}/{len(devA)} sites built")
    est = {q["A"]: atp(A, F, comps) for q, A, F in S}
    log(f"AtP done at +{time.time() - t0:.0f} s")
    med_est = {n: float(np.median([est[q["A"]][n]["fwd"] for q, _, _ in S])) for n in names}
    if tm["branch"]["pool"] == "full":
        pool = names
        top30 = sorted(names, key=lambda n: -med_est[n])[:30]
        rand30 = []
    else:
        top30 = sorted(names, key=lambda n: -med_est[n])[:30]
        rest = sorted(n for n in names if n not in top30)
        rand30 = random.Random("54-atp-random30").sample(rest, 30)
        pool = sorted(set(top30) | set(rand30) | set(always), key=lambda n: names.index(n))
    real = {}
    for k, (q, A, F) in enumerate(S):
        real[q["A"]] = real_single(A, F, [P.parse(n) for n in pool])
        log(f"real singles pair {k + 1}/{len(S)} at +{time.time() - t0:.0f} s")
    med_real = {n: {"fwd": float(np.median([real[q["A"]][n]["fwd"] for q, _, _ in S])),
                    "rev": float(np.median([real[q["A"]][n]["rev"] for q, _, _ in S]))} for n in pool}
    xs = [med_est[n] for n in pool]
    ys = [med_real[n]["fwd"] for n in pool]
    from scipy.stats import pearsonr, spearmanr
    calib = {"pearson_median": float(pearsonr(xs, ys)[0]), "spearman_median": float(spearmanr(xs, ys)[0]),
             "pearson_all": float(pearsonr([est[q["A"]][n]["fwd"] for q, _, _ in S for n in pool],
                                           [real[q["A"]][n]["fwd"] for q, _, _ in S for n in pool])[0])}
    # greedy (freeze stage2_screen.greedy)
    order = {n: i for i, n in enumerate(names)}
    seq, path = [], []
    for step in range(5):
        best = None
        for n in pool:
            if n in seq:
                continue
            cs = [P.parse(x) for x in seq + [n]]
            m = float(np.median([gap_fwd(A, F, F.step(F.spec(cs, A.base)).D) for q, A, F in S]))
            key = (m, -order[n])
            if best is None or key > best[0]:
                best = (key, n, m)
        seq.append(best[1])
        path.append({"set": list(seq), "median_gap_fwd": best[2]})
        log(f"greedy step {step + 1}: {seq} median {best[2]:.3f} at +{time.time() - t0:.0f} s")
    K = None
    for p_ in path:
        if p_["median_gap_fwd"] >= 0.5:
            K = p_["set"]
            break
    out = {"branch": tm["branch"], "n_pairs": len(S), "pairs": [q["A"] for q, _, _ in S], "pool": pool,
           "top30_atp": top30, "random30": rand30, "median_atp_fwd": med_est, "median_real": med_real,
           "atp_per_pair": est, "real_per_pair": real, "atp_calibration": calib, "greedy_path": path, "K": K,
           "stop": None if K else "distributed: no greedy prefix of <= 5 components reaches 0.5 on dev-A",
           "wall_s": time.time() - t0}
    CM.write_json(STAGE2, out)
    if K:
        CM.write_json(KFILE, {"K": K, "median_gap_fwd_devA": next(p_["median_gap_fwd"] for p_ in path
                                                                  if p_["set"] == K),
                              "greedy_path": path, "branch": tm["branch"], "stage2_sha_note": "stage2.json"})
    log(f"stage 2 done: K = {K}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("b0", "pairs", "regrade", "timing", "stage2"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    if a.stage == "regrade":
        stage_regrade()
        return 0
    import exp51_engine as EN
    e = EN.load("hf", threads=a.threads)
    if a.stage == "b0":
        stage_b0(e, a.limit)
    elif a.stage == "pairs":
        stage_pairs(e)
    elif a.stage == "timing":
        stage_timing(e)
    else:
        stage2(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
