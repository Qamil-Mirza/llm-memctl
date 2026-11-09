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
RUN = OUT / "run"            # --root overrides (pod: runs/exp55/pod/run; stub: runs/exp55/stub/run)
POD = False                  # rev 0b (bc8e145): all arms generated on the device, check20 = device check (report only)
DEADLINE = None
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
        S4.run_b0(e, rows, root, "W", tc, deadline=DEADLINE)
    finally:
        e.generate_batched = eg
    assert S4.EN.MAX_NEW_TOTAL == orig


def stage_check20(e):
    tc = CM.t_cities()
    ans = [r for r in w_rows("P0") if r["set"] == "ANS"]
    pick = ans[::22][:20]
    root = RUN / "check20"
    run_free(e, pick, root, tc)
    if POD:
        log(f"check20 (device check): {len(pick)} rows generated; compared with the laptop §54 B0 after copy-back")
        return
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


def stage_smoke(e):
    """D-22 on the device: 8 W P0 ANS rows (the first check20 picks) against the laptop §54 B0 references bundled in
    runs/exp55/smoke_ref.json (graded class >= 7/8, the §51 62/64 rule scaled), plus 2 LME P0 rows that must run."""
    tc = CM.t_cities()
    ref = json.loads((OUT / "smoke_ref.json").read_text())
    rows = [r for r in w_rows("P0") if r["case_id"] in ref]
    assert len(rows) == 8
    run_free(e, rows, RUN, tc)
    agree = tok = 0
    for r in rows:
        a = CM.read_json(S4.b0_path(RUN, "W", r["case_id"]))
        agree += a["cls"] == ref[r["case_id"]]["cls"]; tok += a["gen_ids"] == ref[r["case_id"]]["gen_ids"]
    lx = lme_rows("P0")[:2]
    for x in lx:
        g = e.generate_batched([e.chat_ids(x["prompt"])], mode="total", max_new=MAXNEW)[0]
        assert g.gen_ids, x["question_id"]
    res = {"w_class_agree": agree, "w_token_agree": tok, "n": 8, "lme_ran": len(lx), "pass": agree >= 7}
    (RUN / "smoke.json").write_text(json.dumps(res))
    log(f"smoke: {res}")
    if not res["pass"]:
        sys.exit(5)


def stage_wfree(e, arms):
    tc = CM.t_cities()
    for arm in arms:
        rows = w_rows(arm)
        if arm == "P0" and not POD:
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
            if rec is None:
                continue                     # P0 not generated yet (stub / partial run): not NM, just pending
            if not rec.get("p_ans_ok"):
                CM.write_json(out, {"case_id": r["case_id"], "skipped": "NM"}); continue
            ids = e.chat_ids(r["prompt"]) + list(rec["gen_ids"][:rec["j"]])
            if arm == "P2":
                assert ids == CM.b3_prefix(e, r, rec)
            items.append((r, rec, ids, e.decode(rec["gen_ids"][:rec["j"]]), out))
        t0 = time.time()
        for i in range(0, len(items), S4.LAYOUT):
            if DEADLINE and time.time() > DEADLINE:
                raise S4.Deadline()
            ch = items[i:i + S4.LAYOUT]
            gs = e.generate_batched([x[2] for x in ch], mode="prefix", max_new=S4.EN.MAX_NEW_RELEASE)
            for (r, rec, ids, head, out), g in zip(ch, gs):
                text = head + g.text
                CM.write_json(out, {"case_id": r["case_id"], "gen_ids": g.gen_ids, "text": text, "stop": g.stop,
                                    "arm": arm, "estimand": "prefix", **CM.grade(r, text, tc)})
        log(f"W prefix {arm}: {len(items)} rows in {time.time() - t0:.0f} s")


def lme_p0() -> dict:
    """question_id -> P0 prompt: runs/exp55/lme_prompts.jsonl if present (the pod bundle), else the §24 records."""
    f = OUT / "lme_prompts.jsonl"
    if f.exists():
        return {x["question_id"]: x["prompt"] for x in map(json.loads, f.read_text().splitlines())}
    from memctl import jacobian_data as jd
    from memctl.analysis import families as F
    return {r.question_id: r.prompt for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}


def lme_rows(arm):
    rec = lme_p0()
    rows = [json.loads(l) for l in (OUT / "rows_LME.jsonl").read_text().splitlines()]
    out = []
    for x in rows:
        p = P.build(rec[x["question_id"]], arm)
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
            if DEADLINE and time.time() > DEADLINE:
                raise S4.Deadline()
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
    ap.add_argument("--stage", required=True, choices=("timing", "smoke", "check20", "wfree", "wprefix", "lme"))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="hf", choices=("hf", "dummy36"))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--pod", action="store_true", help="rev 0b mode")
    ap.add_argument("--deadline-s", type=float, default=None, help="seconds from now; stops between rows/batches (exit 9)")
    a = ap.parse_args(argv)
    global RUN, POD, DEADLINE
    if a.root:
        RUN = Path(a.root)
    POD = a.pod
    DEADLINE = time.time() + a.deadline_s if a.deadline_s else None
    import exp51_engine as EN
    e = EN.load(a.backend, threads=a.threads, device=a.device, snapshot=Path(a.snapshot) if a.snapshot else None)
    try:
        return _dispatch(e, a)
    except S4.Deadline:
        log(f"DEADLINE reached in stage {a.stage}; resumable")
        return 9


def _dispatch(e, a):
    arms = a.arms.split(",") if a.arms else None
    log(f"stage {a.stage} root {RUN} pod={POD} device={a.device} backend={a.backend} engine={e.fingerprint()['engine']}")
    if a.stage == "timing":
        stage_timing(e)
    elif a.stage == "smoke":
        stage_smoke(e)
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
