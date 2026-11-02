#!/usr/bin/env python
"""§50 why false abstention: measure (baseline + teacher-forced logits/residuals + S / polarity / K_v), stage 2 on dev
(W layer sweep, ℓ, u_S, u_M, transfer checks, no-op replay), design, held stage 1, held stage 2. Resumable per row.

    .venv/bin/python scripts/exp50_run.py --stage timing --out runs/exp50_timing     # one unit of each kind
    python scripts/exp50_run.py --smoke 4 --backend fake --out runs/exp50_smoke/run   # D-22 (dev bases only)
    python scripts/exp50_run.py --stage measure --split dev
    python scripts/exp50_run.py --stage stage2-dev
    python scripts/exp50_run.py --stage dev-summary                                   # gate read + audit sample
    python scripts/exp50_run.py --stage design            # after the Advisor's "design OK"
    python scripts/exp50_run.py --stage mark-design --commit <hash>
    python scripts/exp50_run.py --stage measure --split held                          # guarded
    python scripts/exp50_run.py --stage stage2-held                                   # guarded; opening rule

Spec: research/proposals/2026-11-02-exp50-why-false-abstention.md rev 1 + rev 1a (D-70); helpers memctl/analysis/
why50.py; the §49 engine (scripts/exp49_run.Engine: greedy from a frozen prefix with steering hooks). Overwrites use
steering.overwrite_spec (the P_ans token, step 0); the residual is the output of decoder layer l. Nothing here prints
a held outcome. Exit 4: design guard. 9: deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp49_run as R49  # noqa: E402
from memctl.analysis import abst49 as A  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from memctl.analysis import why50 as W  # noqa: E402

CASES = Path("runs/exp50/cases50.jsonl")
OUT = Path("runs/exp50/run")
EXIT_DESIGN, EXIT_DEADLINE = 4, 9
write_json, read_json, load_jsonl = R49.write_json, R49.read_json, R49.load_jsonl


def cities():
    import exp45_cases as C
    return list(C.CITIES)


# ---------------------------------------------------------------------------------------------- engine

class Tok:
    """Token ids the measures need (single-token checked for cities / unknown)."""

    def __init__(self, e):
        enc = e.eng.encode if hasattr(e, "eng") else e.encode_text

        def first(t):
            ids = enc(t)
            return ids[0] if ids else None
        self.city = {c: first(" " + c) for c in cities()}
        self.unknown = first(" unknown")
        self.yes = sorted({first(t) for t in W.YES} - {None})
        self.no = sorted({first(t) for t in W.NO} - {None})
        self.digits = [sorted({first(str(d)), first(" " + str(d))} - {None}) for d in range(10)]


def tf_pass(e, ids: list[int], resid_pos: list[int], logit_pos: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """One teacher-forced pass: residuals [len(resid_pos), L, d] (every decoder layer's output) and logits
    [len(logit_pos), V] float32."""
    if hasattr(e, "tf_fake"):
        return e.tf_fake(ids, resid_pos, logit_pos)
    import torch
    saved = {}

    def mk(l):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            saved[l] = h[0, resid_pos].detach().float().cpu().numpy()
        return fn
    handles = [layer.register_forward_hook(mk(l)) for l, layer in enumerate(e.eng.layers)] if resid_pos else []
    try:
        with torch.no_grad():
            out = e.eng.net(input_ids=torch.tensor([ids], device=e.eng.device), use_cache=False, logits_to_keep=0)
        logits = out.logits[0, logit_pos].float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    res = np.stack([saved[l] for l in range(e.n_layers)], axis=1) if resid_pos else np.zeros((0, e.n_layers, e.hidden))
    return res.astype(np.float32), logits.astype(np.float32)


class FakeEngine50(R49.FakeEngine):
    """Canned §50 engine (tests, D-30): baseline notes name the gold; residuals carry a gold-city signal and an
    'answer' axis w (+3 gold, −3 abstain) at P_ans; an overwrite releases gold iff its vector's w-projection > 0;
    S logits favour yes on answerable rows; P_ans logits favour the gold or ' unknown' by the outcome."""

    def __init__(self, rows):
        super().__init__([{**r, "prompt_r4": None} for r in rows])
        self.rows = {r["prompt"]: r for r in rows}
        self.side = {}
        for r in rows:
            for k in ("s_prompt", "pol_prompt"):
                self.side[r[k]] = (k, r)
            if r["gold"]:
                self.side[r["kv_template"].format(city=r["gold"])] = ("kv", r)
        self.w = np.random.default_rng(7).standard_normal(self.D)
        self.w /= np.linalg.norm(self.w)
        self.V = 300

    def chat_ids(self, prompt):
        return list(prompt.encode())

    def encode_text(self, t):
        if not hasattr(self, "_vocab"):
            self._vocab = {}
        return [self._vocab.setdefault(t, 10 + len(self._vocab))]

    def _outcome(self, prompt):
        g = self.gold[prompt]
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 3
        return "unknown" if (g is None or h == 0) else (g if h == 1 else "Derby" if g != "Derby" else "York")

    def _baseline_text(self, prompt):
        r = self.rows.get(prompt)
        if r is None:
            return " x."
        if r["format"] == "F1":
            return f" {self._outcome(prompt)}."
        g = r["gold"] or r["base_gold"]
        return f"{r['A']} is in {g}.\nAnswer: {self._outcome(prompt)}\n"

    def generate(self, ids, specs, max_new, stop):
        if self.delay:
            time.sleep(self.delay)
        text = self.decode(ids)
        p = self._prompt_of(text)
        rest = self._baseline_text(p)[len(text) - len(p):]
        ov = [s for s in specs if s.mode == "overwrite"]
        if not ov:
            return list((rest if len(text) == len(p) else rest.split("\n")[0] + "\n").encode())
        g = self.gold[p]
        out = f" {g}\n" if (g and float(ov[0].vector[:self.D] @ self.w) > 0) else " unknown\n"
        return list(out.encode())

    def tf_fake(self, ids, resid_pos, logit_pos):
        text = self.decode(ids)
        if text in self.side:
            kind, r = self.side[text]
            rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
            ans = r["set"] == "ANS"
            res = np.stack([np.stack([rng.standard_normal(self.D) + (2.0 if ans else -2.0) * self.w
                                      for _ in range(self.L)]) for _ in resid_pos]) if resid_pos else \
                np.zeros((0, self.L, self.D))
            lg = rng.standard_normal((len(logit_pos), self.V))
            t = Tok(self)
            yes_good = ans if kind == "s_prompt" else (not ans)
            for i in range(len(logit_pos)):
                lg[i, t.yes] += 9.0 if yes_good else 7.0
                lg[i, t.no] += 7.0 if yes_good else 9.0
                lg[i, t.digits[7]] += 5.0
            return res.astype(np.float32), lg.astype(np.float32)
        p = self._prompt_of(text)
        r = self.rows[p]
        out = self._outcome(p)
        rng = np.random.default_rng(int(hashlib.sha256(p.encode()).hexdigest()[:8], 16))
        sig = self.basis[self.cities.index(r["gold"])] if r["gold"] else np.zeros(self.D)
        ax = 3.0 if out == r["gold"] else -3.0
        res = np.stack([np.stack([rng.standard_normal(self.D) + sig * (l + 1) / self.L + ax * self.w
                                  for l in range(self.L)]) for _ in resid_pos]) if resid_pos else \
            np.zeros((0, self.L, self.D))
        lg = rng.standard_normal((len(logit_pos), self.V))
        t = Tok(self)
        for i in range(len(logit_pos)):
            if r["gold"]:
                lg[i, t.city[r["gold"]]] += 2.0 + rng.standard_normal()
            lg[i, t.unknown] += 3.0 if out == "unknown" else 0.0
        return res.astype(np.float32), lg.astype(np.float32)


def load_engine(backend: str, threads: int, dtype: str | None, model: str, rows):
    if backend == "fake":
        return FakeEngine50(rows)
    return R49.load_engine(backend, threads, dtype, model, rows)


def encode_fn(e):
    return e.eng.encode if hasattr(e, "eng") else e.encode_text


# ---------------------------------------------------------------------------------------------- measure

def token_at_char(e, gen: list[int], char_start: int) -> int | None:
    """Index k of the generated token whose text span contains char `char_start` of decode(gen)."""
    for k in range(len(gen)):
        if len(e.decode(gen[:k + 1])) > char_start:
            return k
    return None


def measure_row(e, r: dict, root: Path, T: Tok) -> dict:
    """Baseline (§49 classifier), teacher-forced residuals / logits, S (+ polarity on dev-check, K_v on answer-bearing
    ANS rows). Residuals to resid/<case_id>.npz (P_ans; receiving position; the S position on dev rows)."""
    fmt = r["format"]
    ids = e.chat_ids(r["prompt"])
    t0 = time.perf_counter()
    gen = e.generate(ids, [], A.MAX_NEW[fmt], R49.stop_fn(fmt))
    gen_s = time.perf_counter() - t0
    text = e.decode(gen)
    ans = A.answer_text(text, fmt)
    cls = A.classify(ans, r["gold"], r["competitors"], cities())
    rec = {"case_id": r["case_id"], "n_prompt": len(ids), "gen_ids": gen, "text": text, "answer": ans, **cls,
           "gen_s": round(gen_s, 3)}
    arrays = {}
    t1 = time.perf_counter()
    if fmt == "F1":
        # build review fix 2: read at the position that produces the first pool city of the model's own output
        import re
        hits = sorted((m.start(), c) for c in cities() for m in re.finditer(r"\b" + re.escape(c) + r"\b", text))
        k = token_at_char(e, gen, hits[0][0]) if hits else None
        if k is None:
            rec["f1_margins"] = None
        else:
            _, lg = tf_pass(e, ids + gen[:k], [], [len(ids) + k - 1])
            rec["f1_margins"] = W.margins(lg[0], r["gold"], T.city, T.unknown) | {"first_city": hits[0][1], "token_k": k}
    else:
        px = A.ans_prefix(gen, e.decode)
        j, ok, why = px["j"], px["ok"], px["why"]
        if ok and len(gen) >= A.MAX_NEW["F2"] and not A.answer_line_done(text):
            ok, why = False, "truncated at 160"
        note = e.decode(gen[:j]) if ok else text
        ab = W.answer_bearing(note, r["gold"], r["task"], cities()) if r["set"] == "ANS" else False
        rec.update({"j": j, "p_ans_ok": ok, "p_ans_why": why, "note": note,
                    "note_gold": bool(r["gold"]) and r["gold"] in G_cities(note), "answer_bearing": ab})
        if ok:
            pa = len(ids) + j - 1
            rpos, lpos = [pa], [pa]
            k = None
            if ab:
                import re
                m = re.search(r"\b" + re.escape(r["gold"]) + r"\b", note)
                k = token_at_char(e, gen[:j], m.start()) if m else None
                if k is not None and k >= 1:
                    rpos.append(len(ids) + k)
                    lpos.append(len(ids) + k - 1)
            res, lg = tf_pass(e, ids + gen[:j], rpos, lpos)
            arrays["p_ans"] = res[0]
            g = r["gold"] or r["base_gold"]
            rec["margins"] = W.margins(lg[0], g, T.city, T.unknown)
            rec["margins_gold_is"] = "gold" if r["gold"] else "base_gold (UNA; descriptive only)"
            if len(rpos) > 1:
                arrays["recv"] = res[1]
                rec["q5"] = {"producing_m_city": W.margins(lg[1], r["gold"], T.city, T.unknown)["m_city"],
                             "recv_pos": rpos[1], "producing_pos": lpos[1]}
    rec["tf_s"] = round(time.perf_counter() - t1, 3)
    # S, polarity, K_v (F2 rows)
    if fmt == "F2":
        t2 = time.perf_counter()
        sids = e.chat_ids(r["s_prompt"])
        res, lg = tf_pass(e, sids, [len(sids) - 1] if r["split"] == "dev" else [], [len(sids) - 1])
        rec["S"] = W.yes_no(lg[0], T.yes, T.no)
        if r["split"] == "dev":
            arrays["s_pos"] = res[0]
        if r["part"] == "dev-check":
            pids = e.chat_ids(r["pol_prompt"])
            _, lg = tf_pass(e, pids, [], [len(pids) - 1])
            rec["POL"] = W.yes_no(lg[0], T.yes, T.no)
        if rec.get("answer_bearing"):
            kids = e.chat_ids(r["kv_template"].format(city=r["gold"]))
            _, lg = tf_pass(e, kids, [], [len(kids) - 1])
            rec["KV"] = W.expected_digit(lg[0], T.digits)
        rec["side_s"] = round(time.perf_counter() - t2, 3)
    if arrays:
        np.savez(root / "resid" / f"{r['case_id']}.npz", **arrays)
    return rec


def G_cities(text: str) -> list[str]:
    import exp49_cases as G
    return G.cities_in(text)


def run_measure(e, rows: list[dict], root: Path, deadline: float | None) -> int:
    T = Tok(e)
    t0 = time.time()
    for r in rows:
        path = root / "rows" / f"{r['case_id']}.json"
        if path.exists():
            continue
        if deadline and time.time() - t0 > deadline:
            return EXIT_DEADLINE
        write_json(path, measure_row(e, r, root, T))
    return 0


def rec_of(root: Path, cid: str) -> dict | None:
    return read_json(root / "rows" / f"{cid}.json")


def arr(root: Path, cid: str, key: str) -> np.ndarray:
    return np.load(root / "resid" / f"{cid}.npz")[key]


# ---------------------------------------------------------------------------------------------- pairs

def flip_pairs(rows: list[dict], root: Path, split: str) -> list[dict]:
    """Q-2 pairs: rep a and rep b of one base x hardness, both answer-bearing with a P_ans, one gold and one abstain."""
    by = {}
    for r in rows:
        if r["split"] == split and r["set"] == "ANS" and r["format"] == "F2":
            by.setdefault((r["base_key"], r["hardness"]), {})[r["rep"]] = r
    out = []
    for (bk, h), d in sorted(by.items()):
        if set(d) != {"a", "b"}:
            continue
        ra, rb = rec_of(root, d["a"]["case_id"]), rec_of(root, d["b"]["case_id"])
        if not (ra and rb and ra.get("answer_bearing") and rb.get("answer_bearing") and ra["p_ans_ok"] and rb["p_ans_ok"]):
            continue
        oc = {ra["outcome"], rb["outcome"]}
        if oc != {"gold", "abstain"}:
            continue
        ab, an = (d["a"], d["b"]) if ra["outcome"] == "abstain" else (d["b"], d["a"])
        out.append({"pair": f"{bk}_{h}", "base_key": bk, "task": d["a"]["task"], "hardness": h,
                    "abstained": ab["case_id"], "answered": an["case_id"], "gold": d["a"]["gold"]})
    return out


def patched_outcome(e, r: dict, b: dict, layer: int, vec: np.ndarray) -> dict:
    spec = st.overwrite_spec(layer, np.asarray(vec, np.float32))
    g = e.generate(R49.prefix_ids(e, r, b), [spec], A.MAX_NEW_RELEASE, R49.newline)
    text = e.decode(g)
    return A.classify(text.split("\n")[0], r["gold"], r["competitors"], cities()) | {"text": text}


# ---------------------------------------------------------------------------------------------- stage 2 (dev)

def fit_u_s(rows, root, layer, part="dev-fit", seed: int = 50) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    diffs = []
    for task in ("DE", "RC", "TU"):
        grp = {}
        for r in rows:
            if r["part"] != part or r["task"] != task or r["format"] != "F2":
                continue
            rec = rec_of(root, r["case_id"])
            if rec is None or not rec.get("S", {}).get("valid"):
                continue
            grp.setdefault(r["set"], []).append(r)
        if not all(k in grp for k in ("ANS", "UNA-rel", "UNA-swap")):
            continue
        means = {}
        for k, rs in grp.items():
            by_city = {}
            for r in rs:
                by_city.setdefault(r["gold"] or r["base_gold"], []).append(r)
            m = min(len(v) for v in by_city.values())
            pick = [x for v in by_city.values() for x in [v[i] for i in rng.permutation(len(v))[:m]]]
            means[k] = np.mean([arr(root, x["case_id"], "s_pos")[layer] for x in pick], axis=0)
        diffs.append(means["ANS"] - 0.5 * (means["UNA-rel"] + means["UNA-swap"]))
    return W.unit(np.mean(diffs, axis=0)) if diffs else None


def stable_answered(rows, root, part) -> list[dict]:
    """Answered answer-bearing ANS-F2 rows whose rep partner is also answered (decision did not vary)."""
    by = {}
    for r in rows:
        if r["part"] == part and r["set"] == "ANS" and r["format"] == "F2":
            by.setdefault((r["base_key"], r["hardness"]), []).append(r)
    out = []
    for rs in by.values():
        recs = [rec_of(root, x["case_id"]) for x in rs]
        if len(rs) == 2 and all(x and x["outcome"] == "gold" for x in recs):
            out += [x for x, y in zip(rs, recs) if y.get("answer_bearing") and y["p_ans_ok"]]
    return out


def fit_u_m(rows, root, layer, part="dev-fit") -> np.ndarray | None:
    cells = {}
    for r in stable_answered(rows, root, part):
        cells.setdefault((r["task"], r["gold"]), []).append(r)
    diffs = []
    for rs in cells.values():
        if len(rs) < 3:
            continue
        m = np.array([rec_of(root, x["case_id"])["margins"]["m_city"] for x in rs])
        lo, hi = np.percentile(m, [100 / 3, 200 / 3])
        top = [x for x, v in zip(rs, m) if v >= hi]
        bot = [x for x, v in zip(rs, m) if v <= lo]
        if not top or not bot or set(map(id, top)) == set(map(id, bot)):
            continue
        diffs.append(np.mean([arr(root, x["case_id"], "p_ans")[layer] for x in top], 0) -
                     np.mean([arr(root, x["case_id"], "p_ans")[layer] for x in bot], 0))
        FIT_U_M_INFO["cells"] = FIT_U_M_INFO.get("cells", 0) + 1
        FIT_U_M_INFO["rows"] = FIT_U_M_INFO.get("rows", 0) + len(rs)
    return W.unit(np.mean(diffs, axis=0)) if diffs else None


FIT_U_M_INFO: dict = {}


def stage2_dev(e, rows, root, deadline) -> dict:
    t0 = time.time()
    by = {r["case_id"]: r for r in rows}
    pairs = flip_pairs(rows, root, "dev")
    sweep = {}
    for p in pairs:
        r, b = by[p["abstained"]], rec_of(root, p["abstained"])
        donor = arr(root, p["answered"], "p_ans")
        for l in range(1, e.n_layers):                 # 1..35 on the 3B (embedding output excluded)
            path = root / "s2dev" / f"{p['pair']}__W_L{l}.json"
            if not path.exists():
                if deadline and time.time() - t0 > deadline:
                    return {"exit": EXIT_DEADLINE}
                write_json(path, {"pair": p["pair"], "layer": l, **patched_outcome(e, r, b, l, donor[l])})
            sweep.setdefault(l, []).append(read_json(path)["outcome"] == "gold")
    n_pairs, n_bases = len(pairs), len({p["base_key"] for p in pairs})
    rates = {l: float(np.mean(v)) for l, v in sweep.items()}
    feasible = n_pairs >= W.L_PAIRS_MIN and n_bases >= W.L_BASES_MIN
    layer = next((l for l in sorted(rates) if rates[l] >= W.L_RELEASE_MIN), None) if feasible else None
    out = {"n_dev_pairs": n_pairs, "n_dev_pair_bases": n_bases, "w_release_by_layer": rates, "feasible": feasible,
           "layer": layer, "layer_status": ("not feasible" if not feasible else
                                            "no qualifying single-site transfer was found under this protocol"
                                            if layer is None else "ok")}
    if layer is not None:
        noop = []
        for p in pairs:
            for cid in (p["abstained"], p["answered"]):
                path = root / "s2dev" / f"{cid}__noop_L{layer}.json"
                if not path.exists():
                    o = patched_outcome(e, by[cid], rec_of(root, cid), layer, arr(root, cid, "p_ans")[layer])
                    write_json(path, {"case_id": cid, **o})
                noop.append(read_json(path)["outcome"] == rec_of(root, cid)["outcome"])
        out["noop_reproduces"] = float(np.mean(noop)) if noop else None
        out["noop_n"] = len(noop)
        out["noop_ok"] = bool(noop) and float(np.mean(noop)) >= W.NOOP_MIN
        FIT_U_M_INFO.clear()
        u_s, u_m = fit_u_s(rows, root, layer), fit_u_m(rows, root, layer)
        out["u_m_fit"] = {"cells_used": FIT_U_M_INFO.get("cells", 0), "rows_used": FIT_U_M_INFO.get("rows", 0)}
        np.savez(root / "s2dev" / "directions.npz", **{k: v for k, v in (("u_s", u_s), ("u_m", u_m)) if v is not None})
        out["u_s_defined"], out["u_m_defined"] = u_s is not None, u_m is not None
        out["cos_s_m"] = float(u_s @ u_m) if (u_s is not None and u_m is not None) else None
        out["transfer"] = transfer_checks(rows, root, layer, u_s, u_m)
    write_json(root / "s2dev" / "stage2_dev.json", out)
    return out


def transfer_checks(rows, root, layer, u_s, u_m) -> dict:
    chk = [r for r in rows if r["part"] == "dev-check" and r["format"] == "F2"]
    out = {}
    if u_s is not None:
        xs = []
        for r in chk:
            rec = rec_of(root, r["case_id"])
            if r["set"] in ("ANS", "UNA-swap") and rec and rec["p_ans_ok"]:
                xs.append((r["base_key"], float(arr(root, r["case_id"], "p_ans")[layer] @ u_s), int(r["set"] == "ANS")))
        by_base = {}
        for k, v, y in xs:
            by_base.setdefault(k, []).append((v, y))
        pt = W.auroc([v for _, v, _ in xs], [y for *_, y in xs])
        ci = W.boot_bases(list(by_base), lambda s: (lambda z: W.auroc([a for a, _ in z], [b for _, b in z]))(
            [x for b in s for x in by_base[b]]), n=2000)
        out["u_s"] = {"auroc": pt, **ci, "n": len(xs), "pass": ci["lo95"] is not None and ci["lo95"] >= W.USA_AUROC_LO}
    if u_m is not None:
        ys = []
        for r in stable_answered(rows, root, "dev-check"):
            rec = rec_of(root, r["case_id"])
            ys.append((r["base_key"], float(arr(root, r["case_id"], "p_ans")[layer] @ u_m), rec["margins"]["m_city"]))
        by_base = {}
        for k, a, b in ys:
            by_base.setdefault(k, []).append((a, b))
        sp = lambda z: W.spearman(np.array([a for a, _ in z]), np.array([b for _, b in z])) if len(z) > 2 else None
        pt = sp([(a, b) for _, a, b in ys])
        ci = W.boot_bases(list(by_base), lambda s: sp([x for b in s for x in by_base[b]]), n=2000)
        out["u_m"] = {"spearman": pt, **ci, "n": len(ys),
                      "pass": pt is not None and pt >= W.UM_SPEARMAN and ci["lo95"] is not None and ci["lo95"] > 0}
    return out


# ---------------------------------------------------------------------------------------------- dev summary / design

def gate_inputs(rows, root, part: str) -> dict:
    """Per task, the S-gate quantities on `part` rows (§4.1)."""
    out = {}
    for task in ("DE", "RC", "TU"):
        rs = [r for r in rows if r["part"] == part and r["task"] == task and r["format"] == "F2"]
        recs = {r["case_id"]: rec_of(root, r["case_id"]) for r in rs}
        val = [r for r in rs if recs[r["case_id"]] and recs[r["case_id"]]["S"]["valid"]]
        ans = [r for r in val if r["set"] == "ANS" and recs[r["case_id"]]["outcome"] == "gold"]
        rel = [r for r in val if r["set"] == "UNA-rel"]
        swp = [r for r in val if r["set"] == "UNA-swap"]
        ans_h1 = [r for r in ans if r["hardness"] == "H1"]
        s = lambda r: recs[r["case_id"]]["S"]["s"]

        def auc_ci(pos, neg):
            by = {}
            for r in pos:
                by.setdefault(r["base_key"], []).append((s(r), 1))
            for r in neg:
                by.setdefault(r["base_key"], []).append((s(r), 0))
            f = lambda z: W.auroc([a for a, _ in z], [b for _, b in z])
            return {"auroc": f([x for v in by.values() for x in v]),
                    **W.boot_bases(list(by), lambda smp: f([x for b in smp for x in by[b]]), n=2000)}
        pol = [(recs[r["case_id"]]["S"]["yes"], recs[r["case_id"]]["POL"]["yes"]) for r in val
               if "POL" in recs[r["case_id"]] and recs[r["case_id"]]["POL"]["valid"]]
        g = {"n_valid": len(val), "n_invalid_mass": len(rs) - len(val), "n_answered": len(ans),
             "auroc_rel": auc_ci(ans, rel), "auroc_swap": auc_ci(ans_h1, swp),
             "yes_answered": float(np.mean([recs[r["case_id"]]["S"]["yes"] for r in ans])) if ans else None,
             "yes_rel": float(np.mean([recs[r["case_id"]]["S"]["yes"] for r in rel])) if rel else None,
             "yes_swap": float(np.mean([recs[r["case_id"]]["S"]["yes"] for r in swp])) if swp else None,
             "polarity_agreement": float(np.mean([a != b for a, b in pol])) if pol else None, "n_polarity": len(pol)}
        g["pass"] = bool(
            g["auroc_rel"]["lo95"] is not None and g["auroc_rel"]["lo95"] >= W.GATE_AUROC_LO
            and g["auroc_swap"]["lo95"] is not None and g["auroc_swap"]["lo95"] >= W.GATE_AUROC_LO
            and g["yes_answered"] is not None and g["yes_answered"] >= W.GATE_YES_ANS
            and g["yes_rel"] is not None and g["yes_rel"] <= W.GATE_YES_UNA
            and g["yes_swap"] is not None and g["yes_swap"] <= W.GATE_YES_UNA
            and g["polarity_agreement"] is not None and g["polarity_agreement"] >= W.GATE_POLARITY)
        out[task] = g
    return out


def dev_summary(rows, root) -> dict:
    dev = [r for r in rows if r["split"] == "dev"]
    oc = Counter((r["set"] + (r["rep"] if r["set"] == "ANS" else ""), r["format"], rec_of(root, r["case_id"])["outcome"])
                 for r in dev)
    chk = sorted((r for r in dev if r["part"] == "dev-check" and r["set"] == "ANS" and r["format"] == "F2"
                  and (rec_of(root, r["case_id"]) or {}).get("note_gold")), key=lambda r: r["case_id"])
    idx = np.random.default_rng(50).permutation(len(chk))[:40]
    audit = [{"case_id": chk[i]["case_id"], "task": chk[i]["task"], "gold": chk[i]["gold"],
              "rule_answer_bearing": rec_of(root, chk[i]["case_id"])["answer_bearing"],
              "note": rec_of(root, chk[i]["case_id"])["note"]} for i in sorted(idx)]
    out = {"outcomes": {"|".join(k): n for k, n in sorted(oc.items())}, "gate": gate_inputs(rows, root, "dev-check"),
           "stage2_dev": read_json(root / "s2dev" / "stage2_dev.json"), "n_audit": len(audit)}
    write_json(root / "dev_summary.json", out)
    write_json(root / "audit40.json", audit)
    return out


def make_design(rows, root, cases: Path) -> dict:
    s = read_json(root / "dev_summary.json")
    s2 = s["stage2_dev"] or {}
    dpath = root / "s2dev" / "directions.npz"
    design = {"method": W.METHOD_VERSION, "proposal": "rev 1 + rev 1a",
              "cases50_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
              "s_gate": {t: g["pass"] for t, g in s["gate"].items()}, "gate_detail": s["gate"],
              "layer": s2.get("layer"), "layer_status": s2.get("layer_status"),
              "noop_reproduces": s2.get("noop_reproduces"), "noop_ok": s2.get("noop_ok"),
              "directions_sha256": hashlib.sha256(dpath.read_bytes()).hexdigest() if dpath.exists() else None,
              "u_s_pass": (s2.get("transfer") or {}).get("u_s", {}).get("pass"),
              "u_m_pass": (s2.get("transfer") or {}).get("u_m", {}).get("pass"), "cos_s_m": s2.get("cos_s_m"),
              "rules": {"negators": W.NEGATORS, "window": W.NEG_WINDOW, "delta_m": W.DELTA_M,
                        "delta_s_frac": W.DELTA_S_FRAC, "r_s_min": W.R_S_MIN, "mass_min": W.MASS_MIN,
                        "feasibility": [W.FEAS_N, W.FEAS_BASES], "open": [W.OPEN_PAIRS, W.OPEN_BASES],
                        "rand_seeds": W.RAND_SEEDS, "cos_max": W.COS_MAX,
                        "same_outcome": "the arm's own recipient; donor from another base with the recipient's outcome "
                                        "(forward: abstained -> abstained, gold rate; reverse: answered -> answered, "
                                        "abstain rate); same task x hardness; same gold city preferred, then nearest "
                                        "|M_commit|; pairs without a same-city donor flagged",
                        "f1_margin_site": "the teacher-forced position producing the first pool city of the F1 output"}}
    write_json(root / "design.json", design)
    return {"design_sha256": A.design_sha(design), **design}


def mark_design(root: Path, commit: str) -> dict:
    return R49.mark_design(root, commit)


# ---------------------------------------------------------------------------------------------- stage 2 (held)

def same_outcome_donor(pool: list[dict], root: Path, rec_row: dict, outcome: str, exclude_base: str) -> tuple[str | None, bool]:
    """Build review fix 1: a donor from another base with the recipient's own outcome, same task x hardness; prefer the
    same gold city, then the nearest |M_commit|. Returns (case_id or None, same_city)."""
    target = rec_of(root, rec_row["case_id"])["margins"]["m_commit"]
    cands = []
    for r in pool:
        if r["base_key"] == exclude_base or r["task"] != rec_row["task"] or r["hardness"] != rec_row["hardness"]:
            continue
        x = rec_of(root, r["case_id"])
        if x and x["outcome"] == outcome and x.get("answer_bearing") and x["p_ans_ok"]:
            cands.append((r["gold"] != rec_row["gold"], abs(x["margins"]["m_commit"] - target), r["case_id"]))
    if not cands:
        return None, False
    best = min(cands)
    return best[2], not best[0]


def stage2_held(e, rows, root, deadline, stop_at: float | None = None, force: bool = False) -> int:
    """`force` (the D-22 smoke only): run the arms whether or not the opening rule and the transfer checks hold."""
    design = A.require_design(root)
    rep1 = read_json(root / "stage1_report.json")
    if not force and (not rep1 or not rep1.get("stage2_opens")):
        write_json(root / "s2held" / "status.json", {"run": False, "why": "opening rule not met"})
        return 0
    layer = design["layer"]
    d = np.load(root / "s2dev" / "directions.npz")
    u_s = d["u_s"] if ("u_s" in d.files and (design["u_s_pass"] or force)) else None
    u_m = d["u_m"] if ("u_m" in d.files and (design["u_m_pass"] or force)) else None
    by = {r["case_id"]: r for r in rows}
    pool = [r for r in rows if r["split"] == "held" and r["set"] == "ANS" and r["format"] == "F2"]
    t0 = time.time()
    for p in flip_pairs(rows, root, "held"):
        for direction, (rid, did) in (("forward", (p["abstained"], p["answered"])),
                                      ("reverse", (p["answered"], p["abstained"]))):
            r, b = by[rid], rec_of(root, rid)
            h_r, h_d = arr(root, rid, "p_ans")[layer], arr(root, did, "p_ans")[layer]
            arms = W.arm_vectors(h_r, h_d, u_s, u_m)
            jobs = [(k, [v["state"]], v["norm"]) for k, v in arms.items()]
            for x in ("S", "M", "span"):
                key = {"S": "P-S", "M": "P-M", "span": "P-span"}[x]
                if key in arms:
                    jobs.append((f"Rand-{x}", W.rand_states(h_r, arms[key]["disp"], p["pair"] + direction, x,
                                                            delta=np.asarray(h_d, np.float64) - np.asarray(h_r, np.float64)),
                                 arms[key]["norm"]))
            # same-outcome control (fix 1): the recipient is the row this direction's arms patch; the donor shares its
            # outcome (forward: an abstained donor into the abstained row; reverse: an answered donor into the answered row)
            so_rec = rid
            sod, so_same_city = same_outcome_donor(pool, root, by[rid], b["outcome"], p["base_key"])
            for arm, states, norm in jobs:
                path = root / "s2held" / f"{p['pair']}__{direction}__{arm}.json"
                if path.exists():
                    continue
                if (deadline and time.time() - t0 > deadline) or (stop_at and time.time() > stop_at):
                    write_json(root / "s2held" / "status.json",
                               {"run": False, "why": "stage 2 not completed (time budget); partial files not read"})
                    return EXIT_DEADLINE
                outs = [patched_outcome(e, r, b, layer, s_) for s_ in states]
                write_json(path, {"pair": p["pair"], "direction": direction, "arm": arm, "norm": norm,
                                  "outcomes": [o["outcome"] for o in outs]})
            path = root / "s2held" / f"{p['pair']}__{direction}__Same.json"
            if not path.exists() and sod is not None:
                rr = by[so_rec]
                o = patched_outcome(e, rr, rec_of(root, so_rec), layer, arr(root, sod, "p_ans")[layer])
                write_json(path, {"pair": p["pair"], "direction": direction, "arm": "Same", "donor": sod,
                                  "recipient": so_rec, "same_city_donor": so_same_city, "outcomes": [o["outcome"]]})
    write_json(root / "s2held" / "status.json", {"run": True})
    return 0


# ---------------------------------------------------------------------------------------------- timing / smoke

def timing(e, rows, root) -> dict:
    r = next(x for x in rows if x["split"] == "dev" and x["set"] == "ANS" and x["format"] == "F2" and x["hardness"] == "H1"
             and x["part"] == "dev-check")
    (root / "resid").mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    rec = measure_row(e, r, root, Tok(e))
    unit_s = time.perf_counter() - t
    rel = None
    if rec.get("p_ans_ok"):
        t = time.perf_counter()
        patched_outcome(e, r, rec, 18, arr(root, r["case_id"], "p_ans")[18])
        rel = time.perf_counter() - t
    out = {"case_id": r["case_id"], "n_generated": len(rec["gen_ids"]), "generate_s": rec["gen_s"],
           "tf_s": rec["tf_s"], "side_s": rec.get("side_s"), "measure_unit_s": round(unit_s, 3),
           "release_s": None if rel is None else round(rel, 3)}
    write_json(root / "timing.json", out)
    print(json.dumps(out))
    return out


def smoke(e, rows, root, n) -> int:
    """D-22: dev bases only (k < n per task; k >= n/2 relabelled stand-in held); every stage; checks: no-op replay
    rows exist and stage files are written; relaxed support rules (pairs >= 1)."""
    import exp50_report as REP
    sub = []
    for r in rows:
        if r["split"] != "dev":
            continue
        k = r["base_index"] // 2
        if k < n:
            hold = k >= n // 2
            sub.append({**r, "split": "held" if hold else "dev",
                        "part": "held" if hold else ("dev-fit" if k % 2 == 0 else "dev-check")})
    (root / "resid").mkdir(parents=True, exist_ok=True)
    cases = root / "cases_smoke.jsonl"
    cases.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sub))
    W.L_PAIRS_MIN, W.L_BASES_MIN, W.OPEN_PAIRS, W.OPEN_BASES, W.FEAS_N, W.FEAS_BASES = 1, 1, 1, 1, 1, 1
    W.L_RELEASE_MIN = 0.0
    tm = {}
    t = time.time(); run_measure(e, [r for r in sub if r["split"] == "dev"], root, None); tm["measure_dev"] = time.time() - t
    t = time.time(); s2 = stage2_dev(e, sub, root, None); tm["stage2_dev"] = time.time() - t
    dev_summary(sub, root)
    make_design(sub, root, cases)
    mark_design(root, R49.smoke_commit(root))
    t = time.time(); run_measure(e, [r for r in sub if r["split"] == "held"], root, None); tm["measure_held"] = time.time() - t
    REP.stage1(sub, root)
    t = time.time(); stage2_held(e, sub, root, None, force=True); tm["stage2_held"] = time.time() - t
    REP.final(sub, root)
    n_rows = len(sub)
    s2files = len(list((root / "s2held").glob("*.json"))) if (root / "s2held").exists() else 0
    n_s2dev = len(list((root / "s2dev").glob("*.json")))
    payload = {"n_rows": n_rows, "timing_s": tm, "per_row_measure_s": (tm["measure_dev"] + tm["measure_held"]) / n_rows,
               "per_release_s": (tm["stage2_dev"] + tm["stage2_held"]) / max(1, n_s2dev + s2files),
               "stage2_dev": {k: s2.get(k) for k in ("n_dev_pairs", "layer", "noop_reproduces", "noop_ok")},
               "s2held_files": s2files, "s2held_arms": dict(Counter(p.name.split("__")[-1][:-5] for p in
                                                                    (root / "s2held").glob("*__*__*.json"))),
               "pass": bool(s2.get("layer") is not None and s2.get("noop_ok") and s2files > 1 and
                            (root / "report.json").exists())}
    write_json(root / "smoke.json", payload)
    print(json.dumps(payload))
    return 0 if payload["pass"] else 5


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("timing", "measure", "stage2-dev", "dev-summary", "design", "mark-design",
                                        "stage1-report", "stage2-held", "report"))
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--split", choices=("dev", "held"), default="dev")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--backend", choices=("hf", "fake"), default="hf")
    ap.add_argument("--model", default=R49.MODEL_ID)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--stop-at-epoch", type=float, default=None, help="stage 2 wall-clock stop (rev 1a item 3)")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    rows = load_jsonl(a.cases)
    root = a.out
    for d in ("resid", "rows", "s2dev", "s2held"):
        (root / d).mkdir(parents=True, exist_ok=True)
    import exp50_report as REP
    if a.stage == "dev-summary":
        s = dev_summary(rows, root)
        print(json.dumps({"gate_pass": {t: g["pass"] for t, g in s["gate"].items()}, "n_audit": s["n_audit"]}))
        return 0
    if a.stage == "design":
        d = make_design(rows, root, a.cases)
        print(json.dumps({"design_sha256": d["design_sha256"], "layer": d["layer"], "s_gate": d["s_gate"]}))
        return 0
    if a.stage == "mark-design":
        print(json.dumps(mark_design(root, a.commit)))
        return 0
    held_stage = a.stage in ("stage1-report", "stage2-held", "report") or (a.stage == "measure" and a.split == "held")
    if held_stage:
        try:
            A.require_design(root)
        except PermissionError as err:
            print(f"design guard: {err}", file=sys.stderr)
            return EXIT_DESIGN
    if a.stage == "stage1-report":
        r = REP.stage1(rows, root)
        print(json.dumps({"stage2_opens": r["stage2_opens"]}))
        return 0
    if a.stage == "report":
        REP.final(rows, root)
        return 0
    e = load_engine(a.backend, a.threads, a.dtype, a.model, rows)
    if a.smoke:
        if root.resolve().is_relative_to((ROOT / "runs" / "exp50").resolve()):
            print("smoke outputs go outside runs/exp50/", file=sys.stderr)
            return 2
        return smoke(e, rows, root, a.smoke)
    if a.stage == "timing":
        timing(e, rows, root)
        return 0
    if a.stage == "measure":
        if a.backend == "fake":
            e.delay = float(os.environ.get("FAKE_DELAY_S", "0"))
        todo = [r for r in rows if r["split"] == a.split][:a.limit] if a.limit else [r for r in rows if r["split"] == a.split]
        return run_measure(e, todo, root, a.deadline_s)
    if a.stage == "stage2-dev":
        out = stage2_dev(e, rows, root, a.deadline_s)
        print(json.dumps({k: out.get(k) for k in ("n_dev_pairs", "layer", "layer_status", "noop_reproduces")}))
        return out.get("exit", 0)
    if a.stage == "stage2-held":
        return stage2_held(e, rows, root, a.deadline_s, a.stop_at_epoch)
    return 2


if __name__ == "__main__":
    sys.exit(main())
