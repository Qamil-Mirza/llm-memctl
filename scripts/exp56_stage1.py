#!/usr/bin/env python
"""§56 stage 1 (dev-A only, laptop, $0): one read per dev-A class row, then the R1 / R2 readings and the frozen K.
Proposal rev 0 (6e6b075) + rev 0a (df5d028); freeze runs/exp56/freeze.json (3690b0f).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp56_stage1.py read      # resumable; runs/exp56/run/stage1/<case>.npz
    .venv/bin/python scripts/exp56_stage1.py report                     # runs/exp56/stage1_report.json + K.json
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
RUN = OUT / "run/stage1"
N_BOOT, SEED = 2000, 56
K_CAP = 50


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def dev_a_rows() -> list[dict]:
    rows = [json.loads(l) for l in (OUT / "rows.jsonl").read_text().splitlines()]
    fz = json.loads((OUT / "freeze.json").read_text())
    assert X.sha_file(OUT / "rows.jsonl") == fz["data"]["rows"]["sha256"]
    return [x for x in rows if x["half"] == "A" and x["cls"] in X.CLASSES]


def stage_read(e):
    cases = {r["case_id"]: r for r in X.cases()}
    rows = dev_a_rows()
    rd = X.Reader(e)
    todo = [x for x in rows if not (RUN / f"{x['case_id']}.npz").exists()]
    log(f"stage 1 read: {len(rows)} dev-A rows, {len(todo)} to do")
    t0 = time.time()
    for i, x in enumerate(todo):
        r = cases[x["case_id"]]
        rec, _ = X.record_for(r)
        ids = CM.prefix_ids(rec)
        assert len(ids) - 1 == rec["p_ans"] == x["p_ans"]
        cids = X.city_ids(e.encode, r)
        res = rd.read(ids, cids)
        X.save_read(RUN / f"{x['case_id']}.npz", res,
                    {"case_id": x["case_id"], "cls": x["cls"], "set": x["set"], "base_key": x["base_key"],
                     "task": x["task"], "n_tokens": len(ids), "city_ids": cids,
                     "gold_first_id": x.get("gold_first_id")})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            import exp51_engine as EN
            log(f"{i + 1}/{len(todo)} rows, {el / (i + 1):.1f} s/row, ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} "
                f"min, peak RSS {EN.peak_rss_gb():.1f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "report"])
    a = ap.parse_args(argv)
    if a.stage == "read":
        import exp51_engine as EN
        e = EN.load("hf")
        log(f"engine {e.name} loaded in {e.load_s:.0f} s")
        return stage_read(e)
    import exp56_stage1_report as REP
    return REP.main()


if __name__ == "__main__":
    sys.exit(main())
