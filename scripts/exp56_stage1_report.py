#!/usr/bin/env python
"""§56 stage-1 readings from the stored dev-A reads (runs/exp56/run/stage1/*.npz). Proposal rev 0 §5 stage 1, §6 R1/R2
(6e6b075) + rev 0a (df5d028). Writes runs/exp56/stage1_report.json, runs/exp56/K.json (the frozen candidate set and
its mean-ablation values) and runs/exp56/c_mean_act.npy (dev-A C-row mean activation of every neuron, for the
stage-2 random sets).

Fixed here (before the report is run):
- R1: m = mean DLA over dev-A FA rows (396,288 neurons). P = sum of the positive means. Neurons ranked by m, descending;
  K50 = the smallest k whose top-k sum reaches P / 2. Interval: 2,000 bootstrap draws (seed 56) over the bases that
  have at least one FA row, all FA rows of a drawn base taken (clustered by base), K50 recomputed; 95% percentile.
  Verdict on the point: CONCENTRATED if K50 <= 50, DISTRIBUTED if K50 > 500, PARTLY CONCENTRATED otherwise.
- K = the top min(K50, 50) neurons by m. Mean-ablation value of each = its mean activation over dev-A C rows.
- R2: ratio = sum over K of mean DLA on TA / the same on FA. Interval: 2,000 draws (seed 56) over all 75 dev-A bases,
  FA and TA rows of a drawn base taken together, K held fixed; 95% percentile. SHARED CHANNEL if the interval is
  wholly >= 0.8, FA-LEANING if wholly <= 0.5, MIXED otherwise.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp56_common as X  # noqa: E402

OUT = X.OUT
RUN = OUT / "run/stage1"
N_BOOT, SEED, K_CAP = 2000, 56, 50
INTER = 11008


def load():
    import exp56_stage1 as S1
    rows = S1.dev_a_rows()
    metas, dla, act, mlp, attn = [], [], [], [], []
    for x in rows:
        z = np.load(RUN / f"{x['case_id']}.npz")
        m = json.loads(str(z["meta"]))
        assert m["case_id"] == x["case_id"] and m["cls"] == x["cls"]
        assert z["dla"].shape == (36, INTER)
        metas.append(m)
        dla.append(z["dla"].reshape(-1)); act.append(z["a"].reshape(-1))
        mlp.append(z["mlp_dla"]); attn.append(z["attn_dla"])
    return metas, np.stack(dla), np.stack(act), np.stack(mlp), np.stack(attn)


def k50(m: np.ndarray) -> tuple[int, np.ndarray]:
    pos = m[m > 0]
    if pos.size == 0:
        return 0, np.array([], np.int64)
    s = np.sort(pos)[::-1].astype(np.float64)
    c = np.cumsum(s)
    return int(np.searchsorted(c, c[-1] / 2) + 1), s


def verdict_r1(k):
    return "CONCENTRATED" if k <= 50 else "DISTRIBUTED" if k > 500 else "PARTLY CONCENTRATED"


def verdict_r2(lo, hi):
    return "SHARED CHANNEL" if lo >= 0.8 else "FA-LEANING" if hi <= 0.5 else "MIXED"


def boot_k50(D, bases, rng):
    ub = sorted(set(bases))
    idx = {b: [i for i, x in enumerate(bases) if x == b] for b in ub}
    out = []
    for _ in range(N_BOOT):
        w = np.zeros(len(bases), np.float32)
        for b in rng.choice(len(ub), len(ub)):
            w[idx[ub[b]]] += 1
        out.append(k50((w @ D) / w.sum())[0])
    return out


def main():
    metas, dla, act, mlp, attn = load()
    cls = np.array([m["cls"] for m in metas])
    base = [m["base_key"] for m in metas]
    rep: dict = {"n_rows": {c: int((cls == c).sum()) for c in X.CLASSES}}

    # sanity: per row, the neuron DLA sums to the all-layer MLP DLA, and D = its decomposition
    err = np.abs(dla.astype(np.float64).reshape(len(metas), 36, INTER).sum(2) - mlp).max()
    dec = np.array([m["emb_dla"] for m in metas]) + attn.sum(1) + mlp.sum(1)
    Dl = np.array([m["D_lin"] for m in metas]); Dv = np.array([m["D"] for m in metas])
    rep["sanity"] = {"max_layer_sum_err": float(err), "max_decomp_err": float(np.abs(dec - Dl).max()),
                     "max_D_vs_D_lin": float(np.abs(Dv - Dl).max())}

    fa, ta, cc, ua = (np.where(cls == c)[0] for c in ("FA", "TA", "C", "UA"))       # by name, not X.CLASSES order
    m_fa = dla[fa].mean(0, dtype=np.float64)
    K50, s = k50(m_fa)
    rng = np.random.default_rng(SEED)
    bk = boot_k50(dla[fa], [base[i] for i in fa], rng)
    lo, hi = np.percentile(bk, [2.5, 97.5])
    rep["R1"] = {"K50": K50, "ci95": [float(lo), float(hi)], "verdict": verdict_r1(K50),
                 "total_positive_push": float(s.sum()), "n_positive_neurons": int(s.size),
                 "net_mean_mlp_push_FA": float(m_fa.sum()),
                 "top_k_share": {str(k): float(s[:k].sum() / s.sum()) for k in (10, 50, 100, 500, 1000, 5000)},
                 "bootstrap": "2,000 draws, seed 56, over bases with >= 1 FA row, clustered by base"}

    nK = min(K50, K_CAP)
    order = np.argsort(-m_fa)
    K = order[:nK]
    c_mean = act[cc].mean(0, dtype=np.float64)
    np.save(OUT / "c_mean_act.npy", c_mean.astype(np.float32))
    m_ta, m_c = dla[ta].mean(0, dtype=np.float64), dla[cc].mean(0, dtype=np.float64)
    m_ua = dla[ua].mean(0, dtype=np.float64) if ua.size else np.zeros_like(m_fa)
    a_fa, a_ta, a_c = (act[i].mean(0, dtype=np.float64) for i in (fa, ta, cc))
    Kl = [{"rank": r + 1, "layer": int(k // INTER), "neuron": int(k % INTER), "flat": int(k),
           "mean_dla_FA": float(m_fa[k]), "mean_dla_TA": float(m_ta[k]), "mean_dla_C": float(m_c[k]),
           "mean_dla_UA": float(m_ua[k]), "mean_act_FA": float(a_fa[k]), "mean_act_TA": float(a_ta[k]),
           "mean_act_C": float(c_mean[k]), "ablation_value": float(c_mean[k])} for r, k in enumerate(K)]
    (OUT / "K.json").write_text(json.dumps({"n": int(nK), "rule": "top min(K50, 50) by mean dev-A FA DLA",
                                            "ablation": "mean activation over dev-A C rows, at P_ans only",
                                            "K": Kl}, indent=1))

    # R2 (K fixed), bootstrap over all bases jointly
    sK_fa, sK_ta = float(m_fa[K].sum()), float(m_ta[K].sum())
    ub = sorted(set(base)); bi = {b: [i for i, x in enumerate(base) if x == b] for b in ub}
    dK = dla[:, K].astype(np.float64).sum(1)                                        # per-row sum over K
    rr = []
    for _ in range(N_BOOT):
        pick = np.concatenate([bi[ub[b]] for b in rng.choice(len(ub), len(ub))])
        f, t = pick[cls[pick] == "FA"], pick[cls[pick] == "TA"]
        rr.append(dK[t].mean() / dK[f].mean())
    rlo, rhi = np.percentile(rr, [2.5, 97.5])
    rep["R2"] = {"ratio": sK_ta / sK_fa, "ci95": [float(rlo), float(rhi)], "verdict": verdict_r2(rlo, rhi),
                 "sum_K_mean_dla": {"FA": sK_fa, "TA": sK_ta, "C": float(m_c[K].sum()), "UA": float(m_ua[K].sum())},
                 "bootstrap": "2,000 draws, seed 56, over all 75 dev-A bases (FA and TA rows of a base together), K fixed"}

    # descriptives (§5 stage 1 step 5)
    k50_ta, _ = k50(m_ta); k50_c, _ = k50(m_c)
    pos_l = np.clip(m_fa, 0, None).reshape(36, INTER)
    topK50 = order[:K50]
    desc = {"layer_profile_FA": {"positive_push_share": (pos_l.sum(1) / pos_l.sum()).round(4).tolist(),
                                 "net_push": m_fa.reshape(36, INTER).sum(1).round(3).tolist(),
                                 "n_in_top_K50": np.bincount(topK50 // INTER, minlength=36).tolist(),
                                 "note": "per layer: share of the summed positive mean FA DLA; net mean FA DLA; "
                                         "count of the top-K50 neurons (the stop rule's layer profile)"},
            "K_layer_hist": dict(sorted(Counter(int(k // INTER) for k in K).items())),
            "K50_TA": k50_ta, "K50_C": k50_c,
            "ratio_C_over_FA_on_K": float(m_c[K].sum() / sK_fa),
            "top50_overlap_FA_TA": len(set(order[:50]) & set(np.argsort(-m_ta)[:50])),
            "top50_overlap_FA_C": len(set(order[:50]) & set(np.argsort(-m_c)[:50]))}
    # write-vector cosines (sign-corrected by the FA mean activation: the direction the neuron actually writes)
    W = _down_cols(sorted({int(k // INTER) for k in K}))
    u51 = np.load(ROOT / "runs/exp51/run/edits/8364bf135ac59a83.npz")["c0_U"][:, 0].astype(np.float64)
    v38 = np.load(ROOT / "runs/exp38_dev/qwen3b/directions.npz")["L29"].astype(np.float64)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    cs = []
    for x in Kl:
        w = W[x["layer"]][:, x["neuron"]].astype(np.float64) * np.sign(x["mean_act_FA"])
        cs.append({"rank": x["rank"], "cos_exp51_L35_write": cos(w, u51), "cos_exp38_L29": cos(w, v38)})
    rng2 = np.random.default_rng(SEED)
    ref = []
    for l in W:
        for n in rng2.choice(INTER, 200, replace=False):
            w = W[l][:, n].astype(np.float64)
            ref.append((abs(cos(w, u51)), abs(cos(w, v38))))
    ref = np.array(ref)
    desc["write_cosines"] = {"per_neuron": cs,
                             "mean_abs": {"exp51": float(np.mean([abs(c["cos_exp51_L35_write"]) for c in cs])),
                                          "exp38": float(np.mean([abs(c["cos_exp38_L29"]) for c in cs]))},
                             "random_same_layers_mean_abs": {"exp51": float(ref[:, 0].mean()),
                                                             "exp38": float(ref[:, 1].mean())},
                             "note": "write = down_proj column x sign(mean FA activation); §51 = submission 2 rank-1 U "
                                     "(8364bf13, anti-unknown / pro-city by design); §38 = v_29"}
    # where the push sits: per-layer mean MLP and attention DLA by class
    desc["per_layer_mean_dla"] = {c: {"mlp": mlp[i].mean(0).round(4).tolist(), "attn": attn[i].mean(0).round(4).tolist()}
                                  for c, i in (("FA", fa), ("TA", ta), ("C", cc))}
    desc["D_by_class"] = {c: {"mean": float(Dv[i].mean()), "median": float(np.median(Dv[i])),
                              "frac_pos": float((Dv[i] > 0).mean())} for c, i in (("FA", fa), ("TA", ta), ("C", cc),
                                                                                  ("UA", ua)) if i.size}
    # t* descriptives
    import exp56_stage0 as S0
    tok, _, _ = S0.tokenizer()
    t0 = Counter(); tg = Counter()
    for m in metas:
        t0[(m["cls"], tok.decode([m["t_star_rev0"]]))] += 1
        if m["gold_first_id"] is not None:
            tg[(m["cls"], m["t_star"] == m["gold_first_id"])] += 1
    desc["t_star_rev0_by_class"] = {c: [[t, n] for (k, t), n in t0.most_common() if k == c][:10] for c in X.CLASSES}
    desc["city_t_star_eq_gold"] = {c: [tg[(c, True)], tg[(c, True)] + tg[(c, False)]] for c in ("FA", "C")}
    desc["city_t_star_note"] = "75 of 603 C rows (both halves) answer name-first ('Rachel lives in Cambridge')"
    rep["descriptive"] = desc
    rep["K_top10"] = Kl[:10]
    (OUT / "stage1_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("n_rows", "sanity", "R1", "R2")}, indent=1))


def _down_cols(layers):
    """down_proj weights [hidden, inter] of the given layers from the snapshot (float32)."""
    from safetensors import safe_open
    import exp51_engine as EN
    out = {}
    for f in sorted(EN.SNAPSHOT.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for l in layers:
                k = f"model.layers.{l}.mlp.down_proj.weight"
                if k in h.keys():
                    out[l] = h.get_tensor(k).float().numpy()
    assert set(out) == set(layers)
    return out


if __name__ == "__main__":
    main()
