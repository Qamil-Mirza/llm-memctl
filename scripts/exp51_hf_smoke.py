#!/usr/bin/env python
"""§51 F1 ruling 6 ($0, no API): a scripted DummyClient drives the real loop on the real model, on DRY- synthetic rows,
in its own run dir (never editor_run). It reaches one paired, one prefix and one read pass, save_edit, submit and
finish through the same launch path as the real run (service thread + file queue), and reports time per pass and
peak RSS.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_hf_smoke.py --run runs/exp51/hf_smoke2
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import exp51_common as CM  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp51_harness as H  # noqa: E402
import exp51_service as SV  # noqa: E402


def _tool(i: int, name: str, inp: dict) -> list[dict]:
    return [{"type": "tool_use", "id": f"s{i}", "name": name, "input": inp}]


def script(e, ids: list[str]) -> list:
    layer = e.n_layers - 2
    sh = e.edits.shapes()[ED.EDITABLE["down_proj"].format(l=layer)]
    code = (f"import numpy as np\nnp.savez('/work/edits_in/e1.npz', c0_layer={layer}, c0_tensor=np.array('down_proj'), "
            f"c0_kind=np.array('rows'), c0_idx=np.arange(2), c0_D=np.full((2, {sh[1]}), 1e-3))\n"
            f"np.save('/work/edits_in/u.npy', np.linspace(-1, 1, {e.hidden}))\nprint('ok')\n")
    got: dict = {}

    def eid(msgs):
        got.setdefault("eid", json.loads(msgs[-1]["content"][0]["content"])["edit_id"])
        return got["eid"]
    sub = {"kind": "repair", "component_and_computation": "c", "evidence": "e", "predictions_by_row_type": "p",
           "flips": "f", "activity_predictions": "a",
           "signed_endpoint": {"population": "FA", "direction": "increase", "outcome": "Rel"},
           "most_important_component": {"layer": layer, "tensor": "down_proj"},
           "f_site": {"layer": layer, "tensor": "resid"}, "f_direction": "/work/edits_in/u.npy", "f_int_sign": -1,
           "activity": [{"manipulation": m, "type": t, "sign": s} for m, t, s in
                        [("evidence_removed", "directional", -1), ("city_changed", "equivalence", 0),
                         ("distractors_added", "directional", 1), ("unrelated_vs_absent", "equivalence", 0)]]}
    return [_tool(1, "run_python", {"code": code}),
            _tool(2, "save_edit", {"spec": "/work/edits_in/e1.npz", "meta": {"note": "smoke"}}),
            lambda m: _tool(3, "forward", {"request": {"type": "paired", "rows": ids[:2],
                                                       "arms": [{"edit": None}, {"edit": eid(m), "dose": 1.0}]}}),
            lambda m: _tool(4, "forward", {"request": {"type": "prefix", "rows": ids[:2], "arm": {"edit": got["eid"]}}}),
            lambda m: _tool(5, "forward", {"request": {
                "type": "read", "items": [{"row": ids[0], "source": "b0_full"}], "positions": "p_ans",
                "layers": [0, e.n_layers - 1], "components": [[layer, "down_proj"]], "logits": False,
                "logit_ids": [1, 2], "arm": {"edit": got["eid"]}}}),
            lambda m: _tool(6, "submit", {"edit_id": got["eid"], **sub}),
            _tool(7, "finish", {"summary": "smoke"})]


def run(run_dir: Path, backend: str = "hf", threads: int = 8) -> dict:
    run_dir = Path(run_dir)
    if run_dir.resolve() == (H.ROOT / "runs/exp51/editor_run").resolve():
        raise SystemExit("the smoke run never uses editor_run")
    t0 = time.perf_counter()
    e = EN.load(backend, threads=threads, edit_dir=run_dir / "edits")
    load_s = time.perf_counter() - t0
    H.setup_dry(run_dir, engine=e)
    rows = CM.all_rows(run_dir / "workspace/data/E_rows.jsonl")
    if not all(r["case_id"].startswith("DRY-") and r.get("synthetic") for r in rows):
        raise SystemExit("the smoke run uses DRY- rows only")
    H._api_key = lambda: (_ for _ in ()).throw(AssertionError("key read in a $0 run"))
    svc = SV.Service(run_dir, e, tc=[] if backend.startswith("dummy") else None)
    H.record_pid(run_dir)
    th, ev = svc.start_thread()                    # the same launch path as the real run
    try:
        out = H.Harness(run_dir, H.DummyClient(script(e, [r["case_id"] for r in rows]))).loop()
    finally:
        ev.set()
        th.join(timeout=60)
    entries = svc.ledger.entries
    passes = {x["kind"]: {k: x.get(k) for k in ("ok", "wall_s", "input_tokens", "generated_tokens", "cached_tokens",
                                                "capture_bytes", "error")}
              for x in entries if x["kind"] in SV.PASS_TYPES or x["kind"] == "save_edit"}
    e.edits.assert_clean()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=H.ROOT, capture_output=True, text=True).stdout.strip()
    res = {"backend": backend, "commit": commit, "threads": threads, "rows": len(rows),
           "load_s": round(load_s, 2), "summary": out, "passes": passes,
           "submission_saved": (run_dir / "submissions/1.json").exists(),
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
           "ok": out["reason"] == "finish" and out["submissions"] == 1 and out["ledger"]["passes"] == 3
           and out["ledger"]["failed_requests"] == 0 and out["ledger"]["saved_edits"] == 1 and out["usd"] >= 0}
    CM.write_json(run_dir / "smoke_result.json", res)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--backend", choices=("hf", "dummy", "dummy36"), default="hf")
    a = p.parse_args(argv)
    r = run(Path(a.run), a.backend)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
