"""Stage 1 build checks ($0): arm texts (clauses cut from the authors' wrapper; the change text identical in every arm;
nothing forged outside the payload in I2a/I2b/I2d), the runner end to end on a fake engine (dev only; FULL reused
from the Stage 0 store, regenerated only when missing), scoring and readings on synthetic rows.

    .venv/bin/python -m pytest selfdoubt/tests/test_stage1.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import stage1 as S1  # noqa: E402

CH = "Use 12 apples instead of 30."


def test_arm_texts():
    k = S1.clauses()
    t = {a: S1.injected(a, CH) for a in S1.ARMS}
    assert t["N"] == "" and t["V0"] == C.injected_text("U0", CH) and t["VG"] == C.injected_text("UG", CH)
    for a in S1.ARMS[1:]:
        assert t[a].count(CH) == (2 if a == "I2c" else 1), a
    for a in ("I1", "I2a", "I2d", "I2b"):                 # payload arms: everything sits inside one tool block
        assert t[a].startswith(S1.TOOL_OPEN) and t[a].endswith(S1.TOOL_CLOSE) and t[a].count("<tool_output>") == 1
    assert k["prov"] in t["I2a"] and k["commit"] not in t["I2a"]
    assert k["commit"] in t["I2d"] and k["prov"] not in t["I2d"]
    assert all(k[x] in t["I2b"] for x in ("arrival", "prov", "commit")) and "<update>" in t["I2b"]
    assert k["prov"] in t["Vp"] and k["commit"] not in t["Vp"] and k["commit"] in t["Vc"] and k["prov"] not in t["Vc"]
    assert t["I2c"].startswith(t["I1"]) and t["I2c"].endswith(t["VG"].lstrip("\n"))   # harness wrapper outside
    assert "Reference note" in S1.injected("I1", CH, "doc")
    assert S1.system_prompt_s1().startswith(C.system_prompt()) and "tool output" in S1.system_prompt_s1()


@pytest.fixture(scope="module")
def tok():
    from selfdoubt.engines import load_tokenizer
    return load_tokenizer("Qwen/Qwen3-1.7B")


def test_runner_fake_engine_dev_only(tmp_path, tok):
    from selfdoubt import engines as E
    from selfdoubt import stage1_run as R
    out = tmp_path / "dev"
    eng = E.FakeEngine("Qwen/Qwen3-1.7B", tok)
    args = ["--engine", "fake", "--model", "Qwen/Qwen3-1.7B", "--split-name", "dev", "--source", "gsm8k",
            "--limit", "2", "--out-root", str(out), "--cut", "0.5"]
    assert R.main(args, engine=eng) == 0
    recs = [json.loads(x) for x in (out / "stage1/Qwen3-1.7B/gen.jsonl").read_text().splitlines()]
    assert len(recs) == 2 * len(S1.ARMS) and {r["arm"] for r in recs} == set(S1.ARMS)
    assert all(r["key"].startswith("S1|") and r["cut"] == 0.5 and r["payload"] == "plain" for r in recs)
    full = (out / "full/Qwen3-1.7B/gen.jsonl").read_text().splitlines()
    assert len(full) == 2                                   # FULL generated once for items without one
    n_calls = eng.calls
    assert R.main(args, engine=eng) == 0 and eng.calls == n_calls   # resume: nothing re-run
    with pytest.raises(SystemExit):
        R.main(args[:5] + ["heldout"] + args[6:], engine=eng)       # held-out needs --allow-heldout


def test_prompt_is_s1_system_plus_stage0_prefix(tmp_path, tok):
    items = C.load_items()
    split, _ = C.ensure_split(items)
    it = next(r for r in items if int(r["id"]) == split["dev"]["gsm8k"][0])
    full = {int(it["id"]): {"text": "<think>\nOkay, 3 plus 4 is 7, and 7 times 2 is 14. Done.\n</think>\n\n\\boxed{14}"}}
    reqs = S1.plan("Qwen/Qwen3-1.7B", tok, [it], full, 0.5, "plain")
    fp = S1.full_prompt_s1(tok, "Qwen/Qwen3-1.7B", it["revised_problem"])
    pre, _, _ = C.cut_trace(tok, "Qwen/Qwen3-1.7B", full[int(it["id"])]["text"], 0.5)
    for r in reqs:
        assert r.prompt == fp + pre + S1.injected(r.arm, it["update"]) and S1.RULE.strip() in r.prompt


def rows_for(acc):
    """acc: {arm: list of 0/1 per item}; N answers never gold."""
    rows = []
    for arm, v in acc.items():
        for i, x in enumerate(v):
            rows.append({"item_id": i, "arm": arm, "category": "new" if x else "stale", "accepted": bool(x)})
    return rows


def test_readings_gate_and_selectivity():
    n = 100
    acc = {"V0": [0] * 60 + [1] * 40, "VG": [1] * 90 + [0] * 10, "Vp": [1] * 50 + [0] * 50,
           "Vc": [1] * 80 + [0] * 20, "I1": [0] * n, "I2a": [0] * n, "I2d": [0] * n,
           "I2b": [1] * 40 + [0] * 60, "I2c": [1] * 70 + [0] * 30}
    r = S1.readings(rows_for(acc), n_boot=300)
    assert abs(r["deltas"]["dV"]["point"] - 0.5) < 1e-9 and r["gate"]["pass"]
    assert abs(r["deltas"]["dI2b"]["point"] - 0.4) < 1e-9 and r["not_selective"]["verdict"] == "SHOWN"
    assert abs(r["U_lambda1"]["point"] - 0.1) < 1e-9
    acc["VG"] = acc["V0"][:]
    r = S1.readings(rows_for(acc), n_boot=300)
    assert not r["gate"]["pass"] and r["not_selective"]["verdict"].startswith("UNDETERMINED")


def test_second_payload_reruns_only_tool_arms(tmp_path, tok):
    from selfdoubt import engines as E
    from selfdoubt import stage1_run as R
    out = tmp_path / "dev"
    eng = E.FakeEngine("Qwen/Qwen3-1.7B", tok)
    args = ["--engine", "fake", "--model", "Qwen/Qwen3-1.7B", "--split-name", "dev", "--source", "gsm8k",
            "--limit", "1", "--out-root", str(out), "--cut", "0.9"]
    assert R.main(args, engine=eng) == 0
    c0 = eng.calls
    assert R.main(args + ["--payload", "doc"], engine=eng) == 0
    recs = [json.loads(x) for x in (out / "stage1/Qwen3-1.7B/gen.jsonl").read_text().splitlines()]
    tool = [a for a in S1.ARMS if a.startswith("I")]
    assert eng.calls - c0 == len(tool) and len(recs) == len(S1.ARMS) + len(tool)
    assert {r["key"].split("|")[-1] for r in recs if r["arm"] in tool} == {"plain", "doc"}
