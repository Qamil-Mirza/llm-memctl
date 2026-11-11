#!/usr/bin/env python
"""§57 stage 3 (proposal rev 0 / 0a / 0b: 9e51a1b, 476a580, 02c7cab; K_P and K_N frozen at b284ce2): causal test and
selectivity on dev-B. Laptop, $0.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage3.py marker    # dev-A marker C means + K_N descriptive (rev 0b)
    .venv/bin/python scripts/exp57_stage3.py sets                        # freeze the 20+20 random sets
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage3.py equality  # 2 FA + 1 TA dev-A rows, all stage-3 arms
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage3.py calib     # dev-A FA: K_P, K_N, b ladder -> b_cal (frozen)
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage3.py run       # dev-B FA, TA, C rows (resumable)
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp57_stage3.py dhat      # Δ̂-predicted vs real ΔD0, dev-B FA (descriptive)
    .venv/bin/python scripts/exp57_stage3.py report                      # R3, R3', R4 (matched; dev-A-calibrated), Dam, U

Fixed here, before any stage-3 output exists:
- Marker (rev 0b): the marker positions of a row are the region-N positions whose text starts at or after the last
  "\\nAnswer" of the note-through-marker text (the stage-2 `tokens` rule); they end at P_ans − 1. Position matching is by
  token id: the K_N@marker reset value at a marker position holding token t is the mean of the neuron over all dev-A C
  rows' marker positions holding t. A token id with fewer than MARK_MIN (10) such C positions falls back to the most
  frequent C marker token's mean. Stored as runs/exp57/c_mean_marker.npz before any dev-B read.
- Descriptive (rev 0b): per K_N neuron, the frozen N reset vs the C and FA means at the main marker token, in units of the
  C-row SD there.
- Random sets: 20 per region, matched to the set's per-layer counts, drawn from the same layers excluding the set,
  numpy default_rng(SEED_SETS = 5701) (not 56/57); R_N@marker reuses the R_N draws with marker values.
- Arms. FA rows (dev-B): base, K_P, K_N, K_N@m, R_P×20, R_N×20, R_N@m×20, B1_cal, G1..G12 (9 grid values) = 74.
  TA rows (dev-B): base, K_P, K_N, K_N@m, B1_cal, G×9 = 14. C rows (dev-B): base, K_P, K_N, K_N@m, B1_cal = 5.
  Calibration rows (dev-A FA): base, K_P, K_N, L1..L16 (bias −b, b = 1..16) = 19. Bias = −b on " unknown" at P_ans
  (step 0 only), the stage-0-validated engine (exp57_arms, chunks of CHUNK = 40 arms sharing one prompt prefill).
- Grading: §51 grader on head + tail (head = the row's forced note through the marker; tail = the arm's ≤ 16 tokens).
- b_cal (§5 stage 3 step 3): target = the higher dev-A FA Rel of K_P and K_N; Rel_L(b) on the ladder b = 0 (base), 1..16;
  b_cal = linear interpolation between the first ladder pair that brackets the target from below; 0 if the target is at
  or below Rel(base); 16 with matched = false if above the ladder's max. Frozen in runs/exp57/b_cal.json before dev-B.
- Measures (dev-B): Rel = FA rows graded gold; Uns = TA rows graded in UNSUPPORTED (wrong, conflicting, other, mixed);
  Dam = C rows not graded gold; U(λ = 1) = mean over bases of grade51.base_mixture (FA and C rows as ANS, TA as UNA).
- R3 (per set): rank = 1 + the number of its 20 random sets with Rel ≥ the set's Rel (ties count against the set).
  SHOWN if rank 1 and Rel ≥ 0.20. R3' = the same for K_N@m against R_N@m (secondary, rev 0b).
- R4 matched (rev 0a, primary): B1 curve on dev-B over b ∈ GRID (b = 0 is base): Rel_B1(b) on FA rows, Uns_B1(b) on TA
  rows; Uns_B1 interpolated at Rel(set) between the first bracketing pair in b order (equal Rel at both ends → mean
  of the two Uns); statistic Uns(set) − Uns_B1@Rel(set); NOT READ if Rel(set) lies outside the curve's range. 95%
  interval: 2,000 base-clustered draws over dev-B bases (seed 56, resampling only), recomputing Rel(set) and the
  curve; draws where Rel(set) falls outside the curve are dropped and counted. Verdict per §6.
- R4 dev-A-calibrated (reported beside): Uns(set) − Uns(B1_cal), with Rel − Rel(B1_cal) and Dam, same bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp56_common as X56  # noqa: E402
import exp57_common as X  # noqa: E402
import exp57_stage0 as S0  # noqa: E402
import exp57_stage2 as S2  # noqa: E402
from memctl.analysis import grade51 as GR  # noqa: E402

OUT = X.OUT
RUN = OUT / "run/stage3"
INTER, NL = 11008, 36
SEED_SETS, SEED_BOOT, N_BOOT, N_RAND = 5701, 56, 2000, 20
GRID = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
LADDER = tuple(range(1, 17))
CHUNK = 40
MARK_MIN = 10
B_PLACEHOLDER = 8.0
MARKER_NPZ = OUT / "c_mean_marker.npz"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ---------------------------------------------------------------------------------------------- rows

def dev_b(cls):
    """dev-B rows of a class; FA/C from the frozen rows table, TA from the new P0 UNA generation (klass)."""
    rows = [json.loads(l) for l in (ROOT / "runs/exp56/rows.jsonl").read_text().splitlines()]
    cases = {c["case_id"]: c for c in X56.cases()}
    out = []
    for x in rows:
        if x["half"] != "B":
            continue
        if x["set"] == "ANS":
            if x["cls"] == cls:
                out.append(x)
        elif cls in ("TA", "UA"):
            r = cases[x["case_id"]]
            rec, _ = X56.record_for(r)
            k = X56.klass(r, rec)
            assert k != "PENDING", x["case_id"]
            if k == cls:
                out.append({**x, "cls": k})
    return sorted(out, key=lambda x: x["case_id"])


def head_text(e, rec):
    return e.decode(rec["gen_ids"][:rec["j"]])


# ---------------------------------------------------------------------------------------------- marker (rev 0b)

def marker_positions(e, rec):
    """Absolute marker positions (contiguous, ending at P_ans − 1) and their token ids."""
    n0, p, _ = X.regions(rec)
    gen = rec["gen_ids"][:rec["j"]]
    offs, text = S2._offsets(e, gen)
    assert text.endswith("Answer:"), text[-20:]
    mk = text.rfind("\nAnswer") if "\nAnswer" in text else text.rfind("Answer")
    ts = [t for t in range(p - n0) if offs[t][0] >= mk]
    assert ts and ts == list(range(ts[0], p - n0)), (ts, p - n0)
    return [n0 + t for t in ts], [int(gen[t]) for t in ts]


def marker_acts(e, rec, pos):
    """Activations of every neuron at the given absolute positions: [len(pos), NL, INTER] float64."""
    torch = e.torch
    ids = CM.prefix_ids(rec)
    acc = {}

    def mk(l):
        def fn(module, args):
            acc[l] = args[0][0, pos].to(torch.float64)
        return fn
    hs = [e.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in range(e.n_layers)]
    try:
        with torch.no_grad():
            e.net(input_ids=torch.tensor([ids[:max(pos) + 1]]), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return np.stack([acc[l].numpy() for l in range(e.n_layers)], 1)


def stage_marker(e):
    stats = {}
    for cls in ("C", "FA"):
        s, ss, n, lens = defaultdict(lambda: np.zeros((NL, INTER))), defaultdict(lambda: np.zeros((NL, INTER))), Counter(), Counter()
        rows = S0.dev_a(cls)
        t0 = time.time()
        for k, x in enumerate(rows):
            _, rec = S0.case_rec(x["case_id"])
            pos, toks = marker_positions(e, rec)
            A = marker_acts(e, rec, pos)
            lens[len(pos)] += 1
            for a, t in zip(A, toks):
                s[t] += a; ss[t] += a * a; n[t] += 1
            if (k + 1) % 50 == 0 or k + 1 == len(rows):
                log(f"marker {cls}: {k + 1}/{len(rows)}, {(time.time() - t0) / (k + 1):.1f} s/row, "
                    f"peak RSS {EN.peak_rss_gb():.1f} GB")
        stats[cls] = (s, ss, n, lens)
    sC, ssC, nC, lensC = stats["C"]
    main = nC.most_common(1)[0][0]
    keep = sorted(t for t in nC if nC[t] >= MARK_MIN)
    arrays = {f"mean_{t}": (sC[t] / nC[t]).astype(np.float32) for t in keep}
    arrays["main_token"] = np.array(main)
    np.savez(MARKER_NPZ, **arrays)
    # descriptive: K_N at the main marker token
    KN = json.loads((OUT / "K_N.json").read_text())["K"]
    sF, _, nF, lensF = stats["FA"]
    mC = sC[main] / nC[main]
    sdC = np.sqrt(np.maximum(ssC[main] / nC[main] - mC ** 2, 0) * nC[main] / max(nC[main] - 1, 1))
    mF = sF[main] / max(nF[main], 1)
    per = []
    for z in KN:
        l, i = z["layer"], z["neuron"]
        sd = sdC[l, i] if sdC[l, i] > 0 else np.nan
        per.append({"rank": z["rank"], "layer": l, "neuron": i, "reset_N": z["reset_value"], "C_marker": float(mC[l, i]),
                    "FA_marker": float(mF[l, i]), "C_marker_sd": float(sdC[l, i]),
                    "z_reset_vs_C": float((z["reset_value"] - mC[l, i]) / sd),
                    "z_FA_vs_C": float((mF[l, i] - mC[l, i]) / sd)})
    zr = np.array([abs(x["z_reset_vs_C"]) for x in per]); zf = np.array([abs(x["z_FA_vs_C"]) for x in per])
    rep = {"main_token": int(main), "main_token_text": e.decode([int(main)]),
           "C_positions_by_token": {e.decode([t]): nC[t] for t in nC}, "FA_positions_by_token": {e.decode([t]): nF[t] for t in nF},
           "C_marker_len_rows": dict(lensC), "FA_marker_len_rows": dict(lensF), "kept_tokens": [e.decode([t]) for t in keep],
           "mark_min": MARK_MIN, "npz_sha256": X56.sha_file(MARKER_NPZ),
           "K_N_descriptive": {"median_abs_z_reset_vs_C": float(np.nanmedian(zr)),
                               "share_abs_z_reset_gt_2": float(np.nanmean(zr > 2)),
                               "median_abs_z_FA_vs_C": float(np.nanmedian(zf)),
                               "share_abs_z_FA_gt_2": float(np.nanmean(zf > 2)),
                               "note": "z in units of the C-row SD at the main marker token; reset_N = frozen note-mean"},
           "K_N_per_neuron": per}
    (OUT / "marker.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    log(json.dumps({k: v for k, v in rep.items() if k != "K_N_per_neuron"}, ensure_ascii=False))


def marker_means():
    z = np.load(MARKER_NPZ)
    means = {int(k[5:]): z[k] for k in z.files if k.startswith("mean_")}
    return means, int(z["main_token"])


# ---------------------------------------------------------------------------------------------- sets and arms

def k_flat(reg):
    return np.array([x["flat"] for x in json.loads((OUT / f"K_{reg}.json").read_text())["K"]])


def stage_sets():
    rng = np.random.default_rng(SEED_SETS)
    out = {"seed": SEED_SETS, "n": N_RAND, "rule": "per-layer counts of the set, same layers, set excluded"}
    for reg in ("P", "N"):
        K = k_flat(reg)
        lay = np.bincount(K // INTER, minlength=NL)
        draws = []
        for _ in range(N_RAND):
            ks = []
            for l in np.where(lay > 0)[0]:
                pool = np.setdiff1d(np.arange(INTER), K[K // INTER == l] % INTER)
                ks += [int(l * INTER + i) for i in rng.choice(pool, lay[l], replace=False)]
            draws.append(sorted(ks))
        out[f"R_{reg}"] = draws
    (OUT / "stage3_sets.json").write_text(json.dumps(out, indent=1))
    log(f"sets frozen: {N_RAND} R_P, {N_RAND} R_N (seed {SEED_SETS})")


def _by_layer(ks):
    d = defaultdict(list)
    for k in ks:
        d[int(k // INTER)].append(int(k % INTER))
    return sorted(d.items())


def spec(ks, rv):
    return [(l, idx, [float(rv[l, i]) for i in idx]) for l, idx in _by_layer(ks)]


def spec_marker(ks, pos, toks, means, main):
    out = []
    for q, t in zip(pos, toks):
        m = means.get(t, means[main])
        out += [(q, q + 1, l, idx, [float(m[l, i]) for i in idx]) for l, idx in _by_layer(ks)]
    return out


class Arms:
    def __init__(self, e, b_cal=None):
        self.e = e
        self.unk = e.encode(X56.UNK_TEXT)[0]
        self.rp, self.rn = S0.resets()
        self.KP, self.KN = k_flat("P"), k_flat("N")
        st = json.loads((OUT / "stage3_sets.json").read_text())
        self.RP, self.RN = st["R_P"], st["R_N"]
        self.means, self.main = marker_means()
        self.b_cal = b_cal

    def bias(self, name, b):
        return {"name": name, "bias": {"ids": [self.unk], "value": -float(b)}}

    def build(self, kind, rec):
        pos, toks = marker_positions(self.e, rec)
        a = [{"name": "base"}, {"name": "K_P", "reset_P": spec(self.KP, self.rp)},
             {"name": "K_N", "reset_N": spec(self.KN, self.rn)},
             {"name": "K_N@m", "reset_at": spec_marker(self.KN, pos, toks, self.means, self.main)}]
        if kind == "CAL":
            return a[:3] + [self.bias(f"L{b}", b) for b in LADDER]
        if kind == "FA":
            a += [{"name": f"R_P{j:02d}", "reset_P": spec(s, self.rp)} for j, s in enumerate(self.RP)]
            a += [{"name": f"R_N{j:02d}", "reset_N": spec(s, self.rn)} for j, s in enumerate(self.RN)]
            a += [{"name": f"R_N@m{j:02d}", "reset_at": spec_marker(s, pos, toks, self.means, self.main)}
                  for j, s in enumerate(self.RN)]
        b = B_PLACEHOLDER if self.b_cal is None else self.b_cal
        a.append(self.bias("B1_cal", b))
        if kind in ("FA", "TA"):
            a += [self.bias(f"G{g}", g) for g in GRID if g > 0]
        return a


def run_row(e, arms_obj, kind, row):
    import exp57_arms as A
    r, rec = S0.case_rec(row["case_id"])
    arms = arms_obj.build(kind, rec)
    res = A.run_arms_chunked(e, rec, arms, arms_obj.unk, None, chunk=CHUNK, cand_ids=X56.city_ids(e.encode, r))
    t_star = res[0]["t_star"]
    head = head_text(e, rec)
    tc = CM.t_cities()
    out = {}
    for x in res:
        g = CM.grade(r, head + x["text"], tc)
        out[x["name"]] = {"text": x["text"], "stop": x["stop"], "D0": x["D0"], "cls": g["cls"]}
    return {"case_id": row["case_id"], "kind": kind, "base_key": row["base_key"], "set": row["set"], "t_star": t_star,
            "n_arms": len(arms), "arms": out}


def best_city(e, rec, cids):
    """t* = the best city first token of the row's pool at P_ans (unpatched), as in §56/§57."""
    torch = e.torch
    ids = CM.prefix_ids(rec)
    with torch.no_grad():
        lg = e.net(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0, -1]
    return int(cids[int(torch.argmax(lg[cids]))])


# ---------------------------------------------------------------------------------------------- equality

def stage_equality(e):
    import exp57_arms as A
    ao = Arms(e)
    fa, ta = S0.dev_a("FA"), S0.dev_a("TA")
    plan = [(fa[7]["case_id"], "FA"), (fa[21]["case_id"], "FA"), (ta[0]["case_id"], "TA")]
    out, times = [], []
    for cid, kind in plan:
        r, rec = S0.case_rec(cid)
        t_star = best_city(e, rec, X56.city_ids(e.encode, r))
        arms = ao.build(kind, rec)
        t0 = time.time()
        got = A.run_arms_chunked(e, rec, arms, ao.unk, None, chunk=CHUNK, cand_ids=X56.city_ids(e.encode, r))
        t_run = time.time() - t0
        assert got[0]["t_star"] == t_star, (got[0]["t_star"], t_star)
        rows, tb, tr = A.compare(e, rec, arms, ao.unk, t_star, chunk=CHUNK)
        for x in rows:
            x["case_id"] = cid
        out += rows
        times.append({"case_id": cid, "kind": kind, "n_arms": len(arms), "n_ids": len(CM.prefix_ids(rec)),
                      "batched_s": t_run, "batched_s_in_compare": tb, "batch1_s": tr})
        log(f"{cid} ({kind}): {sum(x['same'] for x in rows)}/{len(rows)} same, max dD "
            f"{max(x['dD'] for x in rows):.2e}, batched {t_run:.0f} s ({len(arms)} arms), batch-1 {tr:.0f} s, "
            f"peak RSS {EN.peak_rss_gb():.1f} GB")
    per_arm = float(np.sum([t["batched_s"] for t in times]) / np.sum([t["n_arms"] for t in times]))
    fa_s = float(np.mean([t["batched_s"] for t in times if t["kind"] == "FA"]))
    ta_s = float(np.mean([t["batched_s"] for t in times if t["kind"] == "TA"]))
    n = {"cal": len(fa), "FA": len(dev_b("FA")), "TA": len(dev_b("TA")), "C": len(dev_b("C"))}
    eta = {"calib_h": n["cal"] * 19 * per_arm / 3600, "FA_h": n["FA"] * fa_s / 3600, "TA_h": n["TA"] * ta_s / 3600,
           "C_h": n["C"] * 5 * per_arm / 3600, "dhat_h": n["FA"] * 19.1 / 3600}
    eta["total_h"] = sum(eta.values())
    rep = {"n_row_arms": len(out), "same_frac": float(np.mean([x["same"] for x in out])),
           "max_dD": float(max(x["dD"] for x in out)),
           "PASS": bool(all(x["same"] for x in out) and max(x["dD"] for x in out) <= 1e-3),
           "outputs_changed_vs_base": {c: sum(x["batched"] != next(y["batched"] for y in out if y["case_id"] == c and
                                                                    y["arm"] == "base") for x in out if x["case_id"] == c)
                                       for c, _ in plan},
           "differences": [x for x in out if not x["same"] or x["dD"] > 1e-3], "times": times,
           "per_arm_s": per_arm, "n_rows": n, "eta": eta, "peak_rss_gb": EN.peak_rss_gb()}
    (OUT / "stage3_equality.json").write_text(json.dumps(rep, indent=1))
    (OUT / "stage3_equality_rows.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(json.dumps({k: v for k, v in rep.items() if k != "differences"}))


# ---------------------------------------------------------------------------------------------- calibration

def interp_b(ladder_b, rel, target):
    """First ladder pair bracketing the target from below, linear in b."""
    if target <= rel[0]:
        return 0.0, True
    for i in range(1, len(ladder_b)):
        if rel[i] >= target:
            lo, hi = rel[i - 1], rel[i]
            f = 1.0 if hi == lo else (target - lo) / (hi - lo)
            return float(ladder_b[i - 1] + f * (ladder_b[i] - ladder_b[i - 1])), True
    return float(ladder_b[-1]), False


def stage_calib(e):
    ao = Arms(e)
    rows = S0.dev_a("FA")
    d = RUN / "calib"; d.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for k, x in enumerate(rows):
        p = d / f"{x['case_id']}.json"
        if p.exists():
            continue
        p.write_text(json.dumps(run_row(e, ao, "CAL", x), ensure_ascii=False))
        done += 1
        if done % 10 == 0:
            el = (time.time() - t0) / done
            log(f"calib {k + 1}/{len(rows)}, {el:.1f} s/row, peak RSS {EN.peak_rss_gb():.1f} GB")
    res = [json.loads((d / f"{x['case_id']}.json").read_text()) for x in rows]
    rel = lambda arm: float(np.mean([r["arms"][arm]["cls"] == "gold" for r in res]))
    target_arm = max(("K_P", "K_N"), key=rel)
    bs = [0] + list(LADDER)
    curve = [rel("base")] + [rel(f"L{b}") for b in LADDER]
    b_cal, matched = interp_b(bs, curve, rel(target_arm))
    out = {"b_cal": b_cal, "matched": matched, "target_arm": target_arm, "target_rel": rel(target_arm),
           "rel_K_P": rel("K_P"), "rel_K_N": rel("K_N"), "ladder": dict(zip(map(str, bs), curve)), "n_rows": len(res)}
    (OUT / "b_cal.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out))


# ---------------------------------------------------------------------------------------------- run (dev-B)

def stage_run(e):
    b = json.loads((OUT / "b_cal.json").read_text())["b_cal"]
    ao = Arms(e, b_cal=b)
    d = RUN / "devb"; d.mkdir(parents=True, exist_ok=True)
    plan = [("FA", x) for x in dev_b("FA")] + [("TA", x) for x in dev_b("TA")] + [("C", x) for x in dev_b("C")]
    todo = [(k, x) for k, x in plan if not (d / f"{x['case_id']}.json").exists()]
    log(f"stage 3 run: b_cal {b:.3f}; {len(plan)} rows ({Counter(k for k, _ in plan)}), {len(todo)} to do")
    t0 = time.time()
    for i, (kind, x) in enumerate(todo):
        (d / f"{x['case_id']}.json").write_text(json.dumps(run_row(e, ao, kind, x), ensure_ascii=False))
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            log(f"{i + 1}/{len(todo)} rows ({kind}), {el / (i + 1):.1f} s/row avg, elapsed {el / 3600:.2f} h, "
                f"peak RSS {EN.peak_rss_gb():.1f} GB")


# ---------------------------------------------------------------------------------------------- Δ̂ vs real (dev-B FA)

def stage_dhat(e):
    rp, rn = S0.resets()
    at = X.Attributor(e, rp, rn, grad_from=0)
    KP, KN = k_flat("P"), k_flat("N")
    out = []
    for x in dev_b("FA"):
        r, rec = S0.case_rec(x["case_id"])
        res = at.run(rec, X56.city_ids(e.encode, r))
        st = json.loads((RUN / "devb" / f"{x['case_id']}.json").read_text())
        assert st["t_star"] == res["t_star"], (st["t_star"], res["t_star"])
        a = st["arms"]
        out.append({"case_id": x["case_id"], "D": res["D"], "base_D0": a["base"]["D0"],
                    "pred_K_P": float(res["dhat_P"].reshape(-1)[KP].sum()), "real_K_P": a["K_P"]["D0"] - a["base"]["D0"],
                    "pred_K_N": float(res["dhat_N"].reshape(-1)[KN].sum()), "real_K_N": a["K_N"]["D0"] - a["base"]["D0"]})
    (OUT / "stage3_dhat_rows.json").write_text(json.dumps(out, indent=1))
    log(f"dhat: {len(out)} dev-B FA rows")


# ---------------------------------------------------------------------------------------------- report

def stage_report():
    from scipy.stats import spearmanr
    res = {k: [json.loads((RUN / "devb" / f"{x['case_id']}.json").read_text()) for x in dev_b(k)] for k in ("FA", "TA", "C")}
    bases = sorted({r["base_key"] for k in res for r in res[k]})
    bi = {b: i for i, b in enumerate(bases)}

    def mat(kind, arm, hit):
        v = np.zeros(len(bases)); n = np.zeros(len(bases))
        for r in res[kind]:
            v[bi[r["base_key"]]] += hit(r["arms"][arm]["cls"]); n[bi[r["base_key"]]] += 1
        return v, n
    gold = lambda c: c == "gold"
    uns = lambda c: c in GR.UNSUPPORTED
    dam = lambda c: c != "gold"
    fa_arms = list(res["FA"][0]["arms"])
    REL = {a: mat("FA", a, gold) for a in fa_arms}
    UNS = {a: mat("TA", a, uns) for a in res["TA"][0]["arms"]}
    DAM = {a: mat("C", a, dam) for a in res["C"][0]["arms"]}
    rate = lambda vn, w: float((vn[0] * w).sum() / (vn[1] * w).sum())
    ones = np.ones(len(bases))

    def curve(w):
        return [rate(REL["base" if g == 0 else f"G{g}"], w) for g in GRID], \
               [rate(UNS["base" if g == 0 else f"G{g}"], w) for g in GRID]

    def interp(rel_b, uns_b, target):
        if not (min(rel_b) <= target <= max(rel_b)):
            return None
        for i in range(len(GRID) - 1):
            lo, hi = sorted((rel_b[i], rel_b[i + 1]))
            if lo <= target <= hi:
                if rel_b[i + 1] == rel_b[i]:
                    return 0.5 * (uns_b[i] + uns_b[i + 1])
                f = (target - rel_b[i]) / (rel_b[i + 1] - rel_b[i])
                return uns_b[i] + f * (uns_b[i + 1] - uns_b[i])
        return None

    def U(arm):
        rows = [{"base_key": r["base_key"], "set": "ANS", "cls": r["arms"][arm]["cls"]} for r in res["FA"] + res["C"]]
        rows += [{"base_key": r["base_key"], "set": "UNA-rel", "cls": r["arms"][arm]["cls"]} for r in res["TA"]]
        m = GR.base_mixture(rows, 1.0)
        return float(np.mean(list(m.values()))) if m else None

    rng = np.random.default_rng(SEED_BOOT)
    W = [np.bincount(rng.choice(len(bases), len(bases)), minlength=len(bases)).astype(float) for _ in range(N_BOOT)]
    rep = {"n": {k: len(v) for k, v in res.items()}, "n_bases": len(bases),
           "b_cal": json.loads((OUT / "b_cal.json").read_text()), "curve_B1": dict(zip(map(str, GRID), zip(*curve(ones)))),
           "sets": {}}
    for name, rnd in (("K_P", "R_P"), ("K_N", "R_N"), ("K_N@m", "R_N@m")):
        rel = rate(REL[name], ones)
        rr = sorted(rate(REL[f"{rnd}{j:02d}"], ones) for j in range(N_RAND))
        rank = 1 + sum(x >= rel for x in rr)
        s = {"secondary": name == "K_N@m", "Rel": rel, "Uns": rate(UNS[name], ones), "Dam": rate(DAM[name], ones),
             "U_lambda1": U(name), "random_Rel": rr, "rank_of_21": rank,
             "R3": "SHOWN" if rank == 1 and rel >= 0.20 else "NOT SHOWN"}
        # R4 matched (primary)
        rb, ub = curve(ones)
        uref = interp(rb, ub, rel)
        if uref is None:
            s["R4_matched"] = {"verdict": "NOT READ", "why": "Rel(set) outside the B1 grid's release range"}
        else:
            st, drop = [], 0
            for w in W:
                rb_w, ub_w = curve(w)
                u = interp(rb_w, ub_w, rate(REL[name], w))
                if u is None:
                    drop += 1; continue
                st.append(rate(UNS[name], w) - u)
            lo, hi = np.percentile(st, [2.5, 97.5])
            s["R4_matched"] = {"stat": s["Uns"] - uref, "Uns_B1_at_Rel": uref, "ci95": [float(lo), float(hi)],
                               "draws_dropped": drop,
                               "verdict": "BETTER THAN THE DIAL" if hi < 0 else "WORSE" if lo > 0 else "NO DIFFERENCE SHOWN"}
        # R4 dev-A-calibrated (beside)
        d_u = [rate(UNS[name], w) - rate(UNS["B1_cal"], w) for w in W]
        d_r = [rate(REL[name], w) - rate(REL["B1_cal"], w) for w in W]
        lo, hi = np.percentile(d_u, [2.5, 97.5])
        s["R4_devA_calibrated"] = {"Uns_minus_B1cal": s["Uns"] - rate(UNS["B1_cal"], ones), "ci95": [float(lo), float(hi)],
                                   "Rel_minus_B1cal": rel - rate(REL["B1_cal"], ones),
                                   "Rel_ci95": np.percentile(d_r, [2.5, 97.5]).tolist(),
                                   "verdict": "BETTER THAN THE DIAL" if hi < 0 else "WORSE" if lo > 0 else "NO DIFFERENCE SHOWN"}
        s["R4_read"] = s["R3"] == "SHOWN"
        rep["sets"][name] = s
    rep["B1_cal"] = {"Rel": rate(REL["B1_cal"], ones), "Uns": rate(UNS["B1_cal"], ones), "Dam": rate(DAM["B1_cal"], ones),
                     "U_lambda1": U("B1_cal")}
    rep["base"] = {"Rel": rate(REL["base"], ones), "Uns": rate(UNS["base"], ones), "Dam": rate(DAM["base"], ones),
                   "U_lambda1": U("base")}
    p = OUT / "stage3_dhat_rows.json"
    if p.exists():
        dh = json.loads(p.read_text())
        rep["dhat_vs_real"] = {k: {"spearman": float(spearmanr([x[f"pred_{k}"] for x in dh], [x[f"real_{k}"] for x in dh])[0]),
                                   "mean_pred": float(np.mean([x[f"pred_{k}"] for x in dh])),
                                   "mean_real": float(np.mean([x[f"real_{k}"] for x in dh]))} for k in ("K_P", "K_N")}
    (OUT / "stage3_report.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps({k: rep["sets"][k]["R3"] for k in rep["sets"]}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["marker", "sets", "equality", "calib", "run", "dhat", "report"])
    a = ap.parse_args(argv)
    if a.stage == "sets":
        return stage_sets()
    if a.stage == "report":
        return stage_report()
    e = EN.load("hf")
    log(f"engine {e.name} loaded in {e.load_s:.0f} s")
    return {"marker": stage_marker, "equality": stage_equality, "calib": stage_calib, "run": stage_run,
            "dhat": stage_dhat}[a.stage](e)


if __name__ == "__main__":
    sys.exit(main())
