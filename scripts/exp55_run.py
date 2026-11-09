#!/usr/bin/env python
"""§55 runner (Qwen 2.5 3B, §51 engine, laptop fp32, $0). Every rule comes from runs/exp55/freeze.json (3bfb13f).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp55_run.py --stage timing     # one W row + one LME row, batch 1
    ... --stage check20      # P0 rerun of 20 W ANS rows at 256 vs §54 B0 (byte identity), then the P0 ANS reuse
    ... --stage wfree        # W free generation: P0 UNA, P1, P2 (batched 64, rows sorted by case id)
    ... --stage wprefix      # W controlled prefix: P1, P2 on rows with a B0 P_ans
    ... --stage lme [--arms P0,P1,P2]   # LongMemEval free generation, batch 1, 256
Stores (gitignored, resumable per row): runs/exp55/run/<arm>/b0/W/<case>.json, runs/exp55/run/<arm>/prefix/W/<case>.json,
runs/exp55/run/lme/<arm>/<qid>.json.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp51_stage4 as S4  # noqa: E402
import exp55_prompts as P  # noqa: E402

OUT = ROOT / "runs/exp55"
RUN = OUT / "run"
FZ = json.loads((OUT / "freeze.json").read_text())
B054 = ROOT / "runs/exp54/run"
MAXNEW = 256


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def w_rows(arm: str) -> list[dict]:
    keep = {json.loads(l)["case_id"]: json.loads(l) for l in (OUT / "rows_W.jsonl").read_text().splitlines()}
    assert hashlib.sha256((OUT / "rows_W.jsonl").read_bytes()).hexdigest() == FZ["data"]["W"]["sha256"]
    out = []
    for l in (ROOT / "runs/exp54/cases54.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["case_id"] in keep:
            p = P.build(r["prompt"], arm)
            assert hashlib.sha256(p.encode()).hexdigest() == keep[r["case_id"]][f"sha_{arm}"]
            out.append({**r, "prompt": p})
    assert len(out) == 750
    return sorted(out, key=lambda r: r["case_id"])


def run_free(e, rows, root, tc):
    """S4.run_b0 at max_new 256 (same record, grade and population as §51 B0)."""
    orig = S4.EN.MAX_NEW_TOTAL
    eg = e.generate_batched

    def gb(ids, *a, **k):
        k.setdefault("max_new", MAXNEW)
        return eg(ids, *a, **k)
    e.generate_batched = gb
    try:
        S4.run_b0(e, rows, root, "W", tc)
    finally:
        e.generate_batched = eg
    assert S4.EN.MAX_NEW_TOTAL == orig


def stage_check20(e):
    tc = CM.t_cities()
    ans = [r for r in w_rows("P0") if r["set"] == "ANS"]
    pick = ans[::22][:20]
    root = RUN / "check20"
    run_free(e, pick, root, tc)
    res = []
    for r in pick:
        a = CM.read_json(S4.b0_path(root, "W", r["case_id"]))
        b = CM.read_json(S4.b0_path(B054, "W", r["case_id"]))
        res.append({"case_id": r["case_id"], "text_eq": a["text"] == b["text"], "ids_eq": a["gen_ids"] == b["gen_ids"],
                    "prompt_ids_eq": a["prompt_ids"] == b["prompt_ids"], "cls": [b["cls"], a["cls"]]})
    ok = all(x["text_eq"] and x["ids_eq"] and x["prompt_ids_eq"] for x in res)
    summ = {"n": len(res), "byte_identical": sum(x["text_eq"] and x["ids_eq"] for x in res), "pass": ok, "rows": res}
    (OUT / "check20.json").write_text(json.dumps(summ, indent=1))
    log(f"check20: {summ['byte_identical']}/20 byte-identical, pass={ok}")
    if not ok:
        log("STOP: P0 reuse not byte-identical; report to the Advisor before any reuse")
        sys.exit(7)
    dst = RUN / "P0" / "b0" / "W"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in ans:
        s = S4.b0_path(B054, "W", r["case_id"])
        if not (dst / s.name).exists():
            shutil.copyfile(s, dst / s.name); n += 1
    log(f"P0 ANS reuse: {n} §54 B0 records copied")


def stage_wfree(e, arms):
    tc = CM.t_cities()
    for arm in arms:
        rows = w_rows(arm)
        if arm == "P0":
            rows = [r for r in rows if r["set"] != "ANS"]
        t0 = time.time()
        run_free(e, rows, RUN / arm, tc)
        log(f"W free {arm}: {len(rows)} rows in {time.time() - t0:.0f} s; peak RSS {S4.EN.peak_rss_gb():.1f} GB")


def stage_wprefix(e, arms):
    tc = CM.t_cities()
    for arm in arms:
        rows = w_rows(arm)
        items = []
        for r in rows:
            out = RUN / arm / "prefix" / "W" / f"{r['case_id']}.json"
            if out.exists():
                continue
            rec = CM.read_json(S4.b0_path(RUN / "P0", "W", r["case_id"]))
            if not rec.get("p_ans_ok"):
                CM.write_json(out, {"case_id": r["case_id"], "skipped": "NM"}); continue
            ids = e.chat_ids(r["prompt"]) + list(rec["gen_ids"][:rec["j"]])
            if arm == "P2":
                assert ids == CM.b3_prefix(e, r, rec)
            items.append((r, rec, ids, e.decode(rec["gen_ids"][:rec["j"]]), out))
        t0 = time.time()
        for i in range(0, len(items), S4.LAYOUT):
            ch = items[i:i + S4.LAYOUT]
            gs = e.generate_batched([x[2] for x in ch], mode="prefix", max_new=S4.EN.MAX_NEW_RELEASE)
            for (r, rec, ids, head, out), g in zip(ch, gs):
                text = head + g.text
                CM.write_json(out, {"case_id": r["case_id"], "gen_ids": g.gen_ids, "text": text, "stop": g.stop,
                                    "arm": arm, "estimand": "prefix", **CM.grade(r, text, tc)})
        log(f"W prefix {arm}: {len(items)} rows in {time.time() - t0:.0f} s")


def lme_rows(arm):
    from memctl import jacobian_data as jd
    from memctl.analysis import families as F
    rec = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}
    rows = [json.loads(l) for l in (OUT / "rows_LME.jsonl").read_text().splitlines()]
    out = []
    for x in rows:
        p = P.build(rec[x["question_id"]].prompt, arm)
        assert hashlib.sha256(p.encode()).hexdigest() == x[f"sha_{arm}"]
        out.append({**x, "prompt": p})
    return out


def stage_lme(e, arms):
    for arm in arms:
        rows = lme_rows(arm)
        t0, n = time.time(), 0
        for x in rows:
            out = RUN / "lme" / arm / f"{x['question_id']}.json"
            if out.exists():
                continue
            ids = e.chat_ids(x["prompt"])
            t1 = time.time()
            g = e.generate_batched([ids], mode="total", max_new=MAXNEW)[0]
            CM.write_json(out, {"question_id": x["question_id"], "arm": arm, "n_prompt": len(ids), "gen_ids": g.gen_ids,
                                "text": g.text, "stop": g.stop, "row_s": round(time.time() - t1, 2)})
            n += 1
            if n % 20 == 0:
                log(f"LME {arm}: {n} rows, {(time.time() - t0) / n:.1f} s/row")
        log(f"LME {arm}: {n} rows in {time.time() - t0:.0f} s")


def stage_timing(e):
    w = [r for r in w_rows("P1") if r["set"] == "ANS"][0]
    l = lme_rows("P1")
    lens = sorted((len(e.chat_ids(x["prompt"])), x["question_id"]) for x in l)
    med = lens[len(lens) // 2]
    lx = next(x for x in l if x["question_id"] == med[1])
    e.generate_batched([e.chat_ids("warm up")], mode="total", max_new=2)
    res = {}
    for name, p in (("W", w["prompt"]), ("LME", lx["prompt"])):
        ids = e.chat_ids(p)
        t0 = time.time()
        g = e.generate_batched([ids], mode="total", max_new=MAXNEW)[0]
        res[name] = {"n_prompt": len(ids), "new_tokens": len(g.gen_ids), "stop": g.stop, "row_s": time.time() - t0}
    res["LME"]["median_prompt_tokens"], res["LME"]["max_prompt_tokens"] = med[0], lens[-1][0]
    res["peak_rss_gb"] = S4.EN.peak_rss_gb()
    res["note"] = "timing only (P1 prompts, batch 1); outputs discarded"
    (OUT / "timing.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("timing", "check20", "wfree", "wprefix", "lme"))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    import exp51_engine as EN
    e = EN.load("hf", threads=a.threads)
    arms = a.arms.split(",") if a.arms else None
    if a.stage == "timing":
        stage_timing(e)
    elif a.stage == "check20":
        stage_check20(e)
    elif a.stage == "wfree":
        stage_wfree(e, arms or ["P0", "P1", "P2"])
    elif a.stage == "wprefix":
        stage_wprefix(e, arms or ["P1", "P2"])
    else:
        stage_lme(e, arms or ["P0", "P1", "P2"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
