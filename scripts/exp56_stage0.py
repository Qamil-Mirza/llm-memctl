#!/usr/bin/env python
"""§56 stage 0 (freeze, $0, laptop). Proposal research/proposals/2026-11-10-exp56-answer-neurons.md rev 0 (6e6b075).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp56_stage0.py rows      # row lists + classes (tokenizer only)
    ... dla-test     # the per-layer sum check on 2 dev-A rows (all 36 layers), plus the full decomposition of D
    ... timing       # one real row: the read storing layers 20-35 + one ablated short generation (dummy K)
    ... timing-kv    # stage-2 cost check: prefill once, copy the cache, time single steps
    ... freeze       # runs/exp56/freeze.json: shas of rows, code, grader, engine, rules
Outputs: runs/exp56/rows.jsonl, devb_una_todo.jsonl, dla_test.json, timing.json, freeze.json.
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
import exp56_common as X  # noqa: E402

OUT = X.OUT
TOL = 1e-4


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    from memctl.analysis.jacobian import wrap_chat
    import exp51_engine as EN
    tok = AutoTokenizer.from_pretrained(str(EN.SNAPSHOT))
    chat = lambda p: list(tok(wrap_chat(tok, p), add_special_tokens=False).input_ids)
    enc = lambda s: list(tok(s, add_special_tokens=False).input_ids)
    return tok, chat, enc


def stage_rows():
    tok, chat, enc = tokenizer()
    unk_id = enc(X.UNK_TEXT)[0]
    rows, todo = [], []
    gold_tok_ok = gold_tok_n = 0
    for r in X.cases():
        if r["set"] == "CF":
            continue
        half = X.half_of(r)
        rec, src = X.record_for(r)
        if rec is None:
            assert half == "B" and r["set"] != "ANS", r["case_id"]
            todo.append({"case_id": r["case_id"], "set": r["set"], "base_key": r["base_key"], "task": r["task"],
                         "base_index": r["base_index"]})
            k = "PENDING"
        else:
            assert rec["case_id"] == r["case_id"]
            assert rec["prompt_ids"] == chat(r["prompt"]), r["case_id"]       # the stored ids are this row's P0
            k = X.klass(r, rec)
        x = {"case_id": r["case_id"], "half": half, "set": r["set"], "base_key": r["base_key"], "task": r["task"],
             "base_index": r["base_index"], "cls": k, "source": src if rec is not None else "exp56_devb_p0 (to run)"}
        if rec is not None:
            x.update({"p_ans": rec.get("p_ans"), "n_prompt": rec["n_prompt"], "grade_cls": rec["cls"],
                      "rec_sha256": X.sha_file(_rec_path(r))})
        if r["set"] == "ANS":
            x["gold_first_id"] = enc(" " + r["gold"])[0]
            if k == "C":
                gold_tok_n += 1
                gold_tok_ok += rec["gen_ids"][rec["j"]] == x["gold_first_id"]
        if k in ("FA", "TA"):
            assert rec["gen_ids"][rec["j"]] == unk_id, r["case_id"]           # the token after ':' is " unknown"
        rows.append(x)
    rows.sort(key=lambda x: x["case_id"])
    todo.sort(key=lambda x: x["case_id"])
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (OUT / "devb_una_todo.jsonl").write_text("".join(json.dumps(x) + "\n" for x in todo))
    counts = {}
    for x in rows:
        counts.setdefault(x["half"], {}).setdefault(x["set"], {}).setdefault(x["cls"], 0)
        counts[x["half"]][x["set"]][x["cls"]] += 1
    summ = {"counts": counts, "devb_una_todo": len(todo), "unk_id": unk_id,
            "C_rows_gold_first_id_eq_generated": [gold_tok_ok, gold_tok_n]}
    (OUT / "rows_counts.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))


def _rec_path(r):
    if r["set"] == "ANS":
        return X.B054 / "b0/W" / f"{r['case_id']}.json"
    return (X.P055 if X.half_of(r) == "A" else X.DEVB_UNA) / "b0/W" / f"{r['case_id']}.json"


def load_rows():
    return [json.loads(l) for l in (OUT / "rows.jsonl").read_text().splitlines()]


def row_and_rec(cid):
    r = next(x for x in X.cases() if x["case_id"] == cid)
    return r, X.record_for(r)[0]


def stage_dla_test(e):
    rows = load_rows()
    pick = [next(x for x in rows if x["half"] == "A" and x["cls"] == c) for c in ("FA", "TA")]
    rd = X.Reader(e, layers=range(e.n_layers))
    out = []
    for x in pick:
        r, rec = row_and_rec(x["case_id"])
        ids = CM.prefix_ids(rec)
        assert len(ids) - 1 == rec["p_ans"]
        t0 = time.time()
        res = rd.read(ids)
        dsum = res["dla"].astype(np.float64).sum(1)
        err = np.abs(dsum - res["mlp_dla"])
        decomp = res["emb_dla"] + res["attn_dla"].sum() + res["mlp_dla"].sum()
        out.append({"case_id": x["case_id"], "cls": x["cls"], "t_star": res["t_star"],
                    "t_star_text": e.decode([res["t_star"]]), "D": res["D"], "D_lin": res["D_lin"],
                    "decomposition_sum": float(decomp), "max_abs_err_layer_sum": float(err.max()),
                    "worst_layer": int(err.argmax()), "mlp_dla_range": [float(res["mlp_dla"].min()),
                                                                        float(res["mlp_dla"].max())],
                    "pass": bool(err.max() <= TOL), "read_s": round(time.time() - t0, 2)})
        log(json.dumps(out[-1]))
    summ = {"tol": TOL, "rows": out, "pass": all(o["pass"] for o in out),
            "note": "layer sum check over all 36 layers; D_lin = <h_final, v>; decomposition = emb + sum attn + sum mlp DLA"}
    (OUT / "dla_test.json").write_text(json.dumps(summ, indent=1))
    log(f"DLA sum check pass={summ['pass']}")
    if not summ["pass"]:
        sys.exit(3)


def stage_timing(e):
    rows = load_rows()
    x = next(x for x in rows if x["half"] == "A" and x["cls"] == "C")
    r, rec = row_and_rec(x["case_id"])
    rd = X.Reader(e)
    ids = CM.prefix_ids(rec)
    t0 = time.time()
    res = rd.read(ids)
    t_read = time.time() - t0
    p = OUT / "stage0_timing" / f"{x['case_id']}.npz"
    t1 = time.time()
    X.save_read(p, res, {"case_id": x["case_id"]})
    t_save = time.time() - t1
    rng = np.random.default_rng(5600)
    spec = {}
    for _ in range(50):                                       # a dummy K: 50 neurons in layers 20-35, zero-ablated
        l = int(rng.choice(X.LAYERS)); spec.setdefault(l, set()).add(int(rng.integers(0, 11008)))
    spec = {l: (np.array(sorted(s)), np.zeros(len(s), np.float32)) for l, s in spec.items()}
    t2 = time.time()
    g = X.ablated_generate(e, rec, spec)
    t_gen = time.time() - t2
    t3 = time.time()
    g0 = e.generate(CM.prefix_ids(rec), mode="prefix")
    t_gen0 = time.time() - t3
    import exp51_engine as EN
    tim = {"case_id": x["case_id"], "n_tokens": len(ids), "read_s": round(t_read, 2), "save_s": round(t_save, 3),
           "npz_bytes": p.stat().st_size, "ablated_gen_s": round(t_gen, 2), "ablated_gen_tokens": len(g.gen_ids),
           "plain_gen_s": round(t_gen0, 2), "plain_gen_tokens": len(g0.gen_ids),
           "peak_rss_gb": round(EN.peak_rss_gb(), 2), "load_s": round(e.load_s, 1), "threads": e.threads,
           "engine": e.fingerprint(), "outputs": "discarded (dummy K; timing only)"}
    (OUT / "timing.json").write_text(json.dumps(tim, indent=1))
    log(json.dumps(tim))


def stage_timing_kv(e):
    """Stage-2 cost check: the prefix over positions < P_ans is the same for every arm (the ablation acts at P_ans
    only), so its KV can be computed once per row and copied per arm. Measures prefill, cache copy and one step."""
    import copy
    rows = load_rows()
    x = next(x for x in rows if x["half"] == "A" and x["cls"] == "FA")
    r, rec = row_and_rec(x["case_id"])
    ids = CM.prefix_ids(rec)
    t0 = time.time(); cache = e._prefill(ids[:-1]); t_pre = time.time() - t0
    t1 = time.time(); c2 = copy.deepcopy(cache); t_copy = time.time() - t1
    steps = []
    tok, pos = ids[-1], len(ids) - 1
    for _ in range(6):
        t2 = time.time(); lg = e._step(tok, c2, pos); steps.append(time.time() - t2)
        tok, pos = int(lg.argmax()), pos + 1
    t3 = time.time(); g = e.generate(ids, mode="prefix"); t_full = time.time() - t3
    import exp51_engine as EN
    kv = {"case_id": x["case_id"], "n_tokens": len(ids), "prefill_s": round(t_pre, 2), "cache_copy_s": round(t_copy, 3),
          "step_s": [round(s, 3) for s in steps], "full_prefix_gen_s": round(t_full, 2),
          "full_prefix_gen_tokens": len(g.gen_ids), "peak_rss_gb": round(EN.peak_rss_gb(), 2)}
    tim = json.loads((OUT / "timing.json").read_text())
    tim["kv_reuse"] = kv
    (OUT / "timing.json").write_text(json.dumps(tim, indent=1))
    log(json.dumps(kv))


def stage_freeze():
    import exp51_engine as EN
    rows = load_rows()
    fz = {
        "experiment": "§56 MLP neurons behind false 'unknown' at P_ans (Qwen 2.5 3B)",
        "proposal": "research/proposals/2026-11-10-exp56-answer-neurons.md rev 0 (6e6b075); D-83 (4f529ca)",
        "frozen_before_any_neuron_measured": "rows, classes, t* rule, DLA code (the 2-row sum test is part of stage 0)",
        "data": {"cases": {"file": "runs/exp54/cases54.jsonl", "sha256": X.sha_file(X.CASES), "seed": 55},
                 "rows": {"file": "runs/exp56/rows.jsonl", "sha256": X.sha_file(OUT / "rows.jsonl"),
                          "n": len(rows)},
                 "devb_una_todo": {"file": "runs/exp56/devb_una_todo.jsonl",
                                   "sha256": X.sha_file(OUT / "devb_una_todo.jsonl")},
                 "split_rule": "dev-A = even base_index, dev-B = odd base_index; CF rows not used",
                 "sources": {"ANS (both halves)": "§54 B0 runs/exp54/run/b0/W (laptop, max_new 160)",
                             "UNA dev-A": "§55 P0 runs/exp55/pod/run/P0/b0/W (pod, HF fp32 no-tf32, max_new 256; "
                                          "device check 20/20 byte-identical to laptop §54 B0, runs/exp55/check20.json)",
                             "UNA dev-B": "new: scripts/exp56_devb_una.py, §55 run_free (S4.run_b0 at max_new 256, "
                                          "layout 64, rows sorted by case id), laptop"},
                 "sealed": "seed 56 unused; seed 57 not generated"},
        "classes": {"FA": "ANS, cls abstention", "C": "ANS, cls gold", "TA": "UNA-rel/UNA-swap, cls abstention",
                    "UA": "UNA-rel/UNA-swap, any other graded class (descriptive only)",
                    "out": "NM (no usable P_ans), W (ANS answered wrong)",
                    "grader": "memctl/analysis/grade51.py via exp51_common.grade (head + tail of the P0 generation)"},
        "t_star_rule": ("t* = argmax of the unmodified logits at P_ans over the vocabulary minus the frozen §51 B1 set "
                        "(every id whose decoded, lower-cased, stripped text begins with 'unknown'); D = logit[UNK_ID] − "
                        "logit[t*], UNK_ID = first id of ' unknown'. Same for every class; for ANS rows agreement of t* "
                        "with gold_first_id (first id of ' ' + gold) is reported descriptively"),
        "dla": ("a_i = down_proj input at P_ans; DLA_i = a_i · <down_proj.weight[:, i], g ⊙ (u_unk − u_t*) / σ>, "
                "σ = sqrt(mean(h²)+eps) of the final-norm input at P_ans; stored layers 20–35 (a and DLA, float32), "
                "all-layer MLP / attention / embedding DLA; sum check 1e-4 on 2 dev-A rows (runs/exp56/dla_test.json)"),
        "ablation": "down_proj pre-hook at absolute position P_ans only, §51 prefix mode (MAX_NEW_RELEASE 16), "
                    "graded head + tail",
        "engine": {"script": "scripts/exp51_engine.py", "sha256": X.sha_file(ROOT / "scripts/exp51_engine.py"),
                   "snapshot": EN.SNAPSHOT.name, "model": EN.MODEL_ID, "dtype": "float32", "device": "cpu",
                   "threads": 16},
        "code": {f: X.sha_file(ROOT / f) for f in ("scripts/exp56_common.py", "scripts/exp56_stage0.py",
                                                    "scripts/exp51_common.py", "memctl/analysis/grade51.py")},
        "readings_and_stopping": "fixed in the proposal §6–§7; not restated here",
    }
    (OUT / "freeze.json").write_text(json.dumps(fz, indent=1, ensure_ascii=False))
    log(f"freeze written, rows sha {fz['data']['rows']['sha256'][:12]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rows", "dla-test", "timing", "timing-kv", "freeze"])
    ap.add_argument("--backend", default="hf")
    a = ap.parse_args(argv)
    if a.stage == "rows":
        return stage_rows()
    if a.stage == "freeze":
        return stage_freeze()
    import exp51_engine as EN
    e = EN.load(a.backend)
    log(f"engine {e.name} loaded in {e.load_s:.0f} s")
    return {"dla-test": stage_dla_test, "timing": stage_timing, "timing-kv": stage_timing_kv}[a.stage](e)


if __name__ == "__main__":
    sys.exit(main())
