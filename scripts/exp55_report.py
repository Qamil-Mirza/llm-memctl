#!/usr/bin/env python
"""§55 report (rules from runs/exp55/freeze.json + rev 0a/0b). W: base-weighted FA / Gold / Uns / U(λ=1) / marker,
paired base-cluster bootstrap stratified by task (stats51, B 10,000, seed 5102, 95% two-sided); R1 and R2 readings;
controlled prefix with Dam; NtR among P1 abstentions; device check (pod check20 vs laptop §54 B0). LME: §52 rules,
paired question bootstrap, descriptive. Usage: exp55_report.py --root runs/exp55/pod/run"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp51_stage4 as S4  # noqa: E402
import exp55_run as R  # noqa: E402
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import grade51 as GR  # noqa: E402
from memctl.analysis import stats51 as S  # noqa: E402

ARMS = ("P0", "P1", "P2")


def ci(res):
    d = res["draws"]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))] if len(d) else [None, None]


def per_base(recs, meta, sets, hit):
    by = defaultdict(list)
    for cid, r in recs.items():
        if r is None or "cls" not in r or meta[cid]["set"] not in sets:
            continue
        by[meta[cid]["base_key"]].append(1.0 if hit(r) else 0.0)
    return {b: float(np.mean(v)) for b, v in by.items()}


def counts(recs, meta, sets, hit):
    xs = [hit(r) for c, r in recs.items() if r is not None and "cls" in r and meta[c]["set"] in sets]
    return [int(sum(xs)), len(xs)]


def note_of(text):
    i = text.rfind(GR.MARKER)
    return text[:i] if i >= 0 else None


def w_report(root: Path) -> dict:
    meta = {r["case_id"]: r for r in R.w_rows("P0")}
    bases = S.Bases({r["base_key"]: r["task"] for r in meta.values()})
    free = {a: {c: CM.read_json(S4.b0_path(root / a, "W", c)) for c in meta} for a in ARMS}
    pref = {"P0": free["P0"]}
    for a in ("P1", "P2"):
        pref[a] = {c: CM.read_json(root / a / "prefix" / "W" / f"{c}.json") for c in meta}
    out = {"n_rows": {a: sum(v is not None for v in free[a].values()) for a in ARMS}}
    M = {"FA": (("ANS",), lambda r: r["cls"] == "abstention"),
         "Gold": (("ANS",), lambda r: r["cls"] == "gold"),
         "Uns": (("UNA-rel", "UNA-swap"), lambda r: r["cls"] in GR.UNSUPPORTED),
         "Uns_rel": (("UNA-rel",), lambda r: r["cls"] in GR.UNSUPPORTED),
         "Uns_swap": (("UNA-swap",), lambda r: r["cls"] in GR.UNSUPPORTED),
         "NoMarker": (("ANS", "UNA-rel", "UNA-swap"), lambda r: r["cls"] == "no_answer_line")}
    for est, store in (("free", free), ("prefix", pref)):
        pb, tab = {}, {}
        for a in ARMS:
            recs = {c: r for c, r in store[a].items() if r is not None and "cls" in r}
            pb[a] = {m: per_base(recs, meta, s, h) for m, (s, h) in M.items()}
            pb[a]["U"] = GR.base_mixture([{"base_key": meta[c]["base_key"], "set": meta[c]["set"], "cls": r["cls"]}
                                          for c, r in recs.items()], 1.0)
            tab[a] = {m: {"bw": GR.mean(pb[a][m]), "k_n": counts(recs, meta, *M[m]) if m in M else None} for m in pb[a]}
        diffs = {}
        for x, y in (("P1", "P0"), ("P2", "P1"), ("P2", "P0")):
            for m in ("FA", "Gold", "Uns", "U"):
                res = S.bootstrap(S.diff_stat(bases, pb[x][m], pb[y][m]), bases.n_by_task())
                diffs[f"{m}:{x}-{y}"] = {"est": res["est"], "ci95": ci(res), "B_valid": res["B_valid"]}
        out[est] = {"arms": tab, "diffs": diffs}
        if est == "prefix":
            dam = {}
            for a in ("P1", "P2"):
                c_rows = [c for c, r in free["P0"].items() if r and meta[c]["set"] == "ANS" and r["cls"] == "gold"
                          and r.get("p_ans_ok") and pref[a].get(c) and "cls" in pref[a][c]]
                k = sum(pref[a][c]["cls"] != "gold" for c in c_rows)
                dpb = per_base({c: pref[a][c] for c in c_rows}, meta, ("ANS",), lambda r: r["cls"] != "gold")
                dam[a] = {"k_n": [k, len(c_rows)], "bw": GR.mean(dpb)}
            nm = {a: sum(1 for r in pref[a].values() if r and r.get("skipped") == "NM") for a in ("P1", "P2")}
            out[est]["Dam"], out[est]["NM"] = dam, nm
    fa0, fa1, fa2 = (out["free"]["arms"][a]["FA"]["bw"] for a in ARMS)
    r1 = "mostly created by the inconsistent wording" if fa1 <= fa0 / 3 else (
        "the inconsistency is not the main cause" if fa1 >= 2 * fa0 / 3 else "partly")
    du = out["free"]["diffs"]["Uns:P2-P1"]
    price = fa2 <= fa1 / 3 and du["ci95"][0] is not None and du["ci95"][0] > 0
    out["R1"] = {"FA_P0": fa0, "FA_P1": fa1, "ratio": fa1 / fa0 if fa0 else None, "reading": r1}
    u1, u2 = out["free"]["arms"]["P1"]["Uns"]["bw"], out["free"]["arms"]["P2"]["Uns"]["bw"]
    out["R2"] = {"FA_P1": fa1, "FA_P2": fa2, "Uns_P2_minus_P1": du, "price_reading_met": price,
                 "fix_allowed": abs(u2 - u1) <= 0.05,
                 "reading": ("the false abstention is largely the price of having an abstention option" if price
                             else "price reading not met; report the trade-off")}
    ntr = [d for c, r in free["P1"].items() if r and meta[c]["set"] == "ANS" and r["cls"] == "abstention"
           for d in [sel.d1(note_of(r["text"]), meta[c]["gold"])]]
    out["NtR_P1"] = {"k": sum(bool(x) for x in ntr), "n": len(ntr)}
    return out


def device_check(root: Path) -> dict:
    rows = []
    for p in sorted((root / "check20" / "b0" / "W").glob("*.json")):
        a = json.loads(p.read_text())
        b = CM.read_json(S4.b0_path(R.B054, "W", a["case_id"]))
        rows.append({"case_id": a["case_id"], "tokens": a["gen_ids"] == b["gen_ids"], "cls": a["cls"] == b["cls"]})
    return {"n": len(rows), "token_equal": sum(x["tokens"] for x in rows), "class_equal": sum(x["cls"] for x in rows),
            "mismatch": [x["case_id"] for x in rows if not (x["tokens"] and x["cls"])]}


def lme_report(root: Path) -> dict:
    rows = {x["question_id"]: x for x in map(json.loads, (R.OUT / "rows_LME.jsonl").read_text().splitlines())}
    from memctl import jacobian_data as jd
    q3 = {r.question_id: r for f in (0, 1) for r in jd.load_records("qwen3b", f)[0]}
    G = {}
    for a in ARMS:
        for q in rows:
            r = CM.read_json(root / "lme" / a / f"{q}.json")
            if r is None:
                continue
            ans, m = sel.agent_answer(r["text"])
            G.setdefault(a, {})[q] = {"marker": m, "unk": sel.is_refusal_answer(ans), "gold": bool(sel.d1(ans, rows[q]["gold"])),
                                      "ntr": bool(sel.d1(sel.split_note(r["text"])[0], rows[q]["gold"])) if m else False}
    groups = {"inview": [q for q, x in rows.items() if x["set52"] == "primary"],
              "notinview": [q for q, x in rows.items() if x["answerable"] and not x["all_in_view"]],
              "answerable": [q for q, x in rows.items() if x["answerable"]],
              "unanswerable": [q for q, x in rows.items() if not x["answerable"]]}
    out = {"n": {a: len(G.get(a, {})) for a in ARMS}, "arms": {}, "diffs": {}}
    for a in G:
        o = {}
        for gname, qs in groups.items():
            qs = [q for q in qs if q in G[a]]
            f = lambda k: [int(sum(G[a][q][k] for q in qs)), len(qs)]
            o[gname] = {"FA_or_unknown": f("unk"), "Gold_proxy": f("gold"), "marker": f("marker")}
        unk_ans = [q for q in groups["answerable"] if q in G[a] and G[a][q]["unk"]]
        o["NtR_among_unknown"] = [sum(G[a][q]["ntr"] for q in unk_ans), len(unk_ans)]
        o["Uns_unanswerable"] = [sum(not G[a][q]["unk"] for q in groups["unanswerable"] if q in G[a]),
                                 sum(q in G[a] for q in groups["unanswerable"])]
        out["arms"][a] = o
    rng = np.random.default_rng(5102)
    for x, y in (("P1", "P0"), ("P2", "P1"), ("P2", "P0")):
        if x not in G or y not in G:
            continue
        for gname in ("inview", "answerable"):
            qs = [q for q in groups[gname] if q in G[x] and q in G[y]]
            for k in ("unk", "gold"):
                d = np.array([G[x][q][k] - G[y][q][k] for q in qs], float)
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
                out["diffs"][f"{k}:{gname}:{x}-{y}"] = {"est": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "n": len(qs)}
    if "P0" in G:   # agreement of the laptop-engine... here pod-engine P0 with the §23 vLLM cache (descriptive)
        agree = [q for q in G["P0"] if q in q3]
        cu = [sel.is_refusal_answer(sel.agent_answer(q3[q].output)[0]) for q in agree]
        out["P0_vs_cache23"] = {"n": len(agree), "unknown_flag_equal": int(sum(G["P0"][q]["unk"] == c for q, c in zip(agree, cu))),
                                "cache_unknown_inview": int(sum(sel.is_refusal_answer(sel.agent_answer(q3[q].output)[0]) for q in groups["inview"]))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/exp55/pod/run")
    a = ap.parse_args(); root = ROOT / a.root
    rep = {"device_check": device_check(root), "W": w_report(root), "LME": lme_report(root)}
    (R.OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"device_check": rep["device_check"], "R1": rep["W"]["R1"], "R2": rep["W"]["R2"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
