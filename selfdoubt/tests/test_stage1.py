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
    tool = [a for a in S1.ARMS if a.startswith("I")]
    assert len(recs) == 2 * (len(S1.ARMS) + len(tool)) and {r["arm"] for r in recs} == set(S1.ARMS)   # plain + doc
    assert all(r["key"].startswith("S1|") and r["cut"] == 0.5 for r in recs)
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
    assert abs(r["deltas"]["dI2b"]["point"] - 0.4) < 1e-9 and r["selectivity"]["verdict"] == "NOT SELECTIVE"
    assert abs(r["U_lambda1"]["point"] - 0.1) < 1e-9
    acc["VG"] = acc["V0"][:]
    r = S1.readings(rows_for(acc), n_boot=300)
    assert not r["gate"]["pass"] and r["selectivity"]["verdict"].startswith("UNDETERMINED")
    acc["VG"] = [1] * 90 + [0] * 10
    acc["I2b"] = [0] * n
    assert S1.readings(rows_for(acc), n_boot=300)["selectivity"]["verdict"] == "SELECTIVE"
    assert S1.payload_sanity({"plain": {"I1": 0, "I2c": 0}, "doc": {"I1": 0, "I2c": 0.1}})["pass"]
    assert not S1.payload_sanity({"plain": {"I1": 0, "I2c": 0}, "doc": {"I1": 0, "I2c": 0}})["pass"]


def test_second_payload_reruns_only_tool_arms(tmp_path, tok):
    from selfdoubt import engines as E
    from selfdoubt import stage1_run as R
    out = tmp_path / "dev"
    eng = E.FakeEngine("Qwen/Qwen3-1.7B", tok)
    args = ["--engine", "fake", "--model", "Qwen/Qwen3-1.7B", "--split-name", "dev", "--source", "gsm8k",
            "--limit", "1", "--out-root", str(out), "--cut", "0.9", "--payload", "plain"]
    assert R.main(args, engine=eng) == 0
    c0 = eng.calls
    assert R.main(args[:-1] + ["doc"], engine=eng) == 0
    recs = [json.loads(x) for x in (out / "stage1/Qwen3-1.7B/gen.jsonl").read_text().splitlines()]
    tool = [a for a in S1.ARMS if a.startswith("I")]
    assert eng.calls - c0 == len(tool) and len(recs) == len(S1.ARMS) + len(tool)
    assert {r["key"].split("|")[-1] for r in recs if r["arm"] in tool} == {"plain", "doc"}


# ---------------------------------------------------------------- Stage 1 v2 (PROPOSAL c69f28e)

def test_v2_system_prompts_and_keys(tok):
    assert S1.system_prompt_s1("s0") == C.system_prompt()
    assert S1.system_prompt_s1("s1t") == C.system_prompt() + S1.TOOL_RULE
    assert "Only the user's <update>" not in S1.system_prompt_s1("s1t") and "tool output" in S1.system_prompt_s1("s1t")
    assert S1.system_prompt_s1() == S1.system_prompt_s1("s1") == C.system_prompt() + S1.RULE
    # s1 keys/seeds are those of the original dev run; the other settings are tagged and seeded apart
    assert S1.key("M", 3, "I1", 0.9, "plain") == "S1|M|3|I1|0.90|plain" == S1.key("M", 3, "I1", 0.9, "plain", "s1")
    assert S1.key("M", 3, "V0", 0.9, "plain", "s0") == "S1|M|3|V0|0.90|sys=s0"
    assert len({S1.seed("M", 3, "V0", 0.9, "plain", s) for s in S1.SYSTEMS}) == 3
    assert S1.store_dir("s1") == "stage1" and S1.store_dir("s0") == "stage1_s0"
    items = C.load_items()
    split, _ = C.ensure_split(items)
    it = next(r for r in items if int(r["id"]) == split["dev"]["gsm8k"][0])
    # S0's FULL prompt is exactly Stage 0's FULL prompt (on-policy prefix)
    assert S1.full_prompt_s1(tok, "Qwen/Qwen3-1.7B", it["revised_problem"], "s0") == \
        C.full_prompt(tok, "Qwen/Qwen3-1.7B", it["revised_problem"])


def test_v2_runner_two_settings(tmp_path, tok):
    from selfdoubt import engines as E
    from selfdoubt import stage1_run as R
    out = tmp_path / "dev"
    eng = E.FakeEngine("Qwen/Qwen3-1.7B", tok)
    args = ["--engine", "fake", "--model", "Qwen/Qwen3-1.7B", "--split-name", "dev", "--source", "gsm8k",
            "--limit", "2", "--out-root", str(out), "--cut", "0.9", "--payload", "plain", "--sys", "s1t,s0"]
    assert R.main(args, engine=eng) == 0
    assert not (out / "stage1").exists()
    for sy in ("s1t", "s0"):
        recs = [json.loads(x) for x in (out / f"stage1_{sy}/Qwen3-1.7B/gen.jsonl").read_text().splitlines()]
        assert len(recs) == 2 * len(S1.ARMS) and {r["sys"] for r in recs} == {sy}
        assert all(r["key"].endswith(f"|sys={sy}") for r in recs)
        meta = json.loads((out / f"stage1_{sy}/Qwen3-1.7B/run_meta.json").read_text())
        assert meta["system_prompt"] == S1.system_prompt_s1(sy) and meta["sys"] == sy
    with pytest.raises(AssertionError):
        R.main(args[:-1] + ["s1t,s1t"], engine=eng)


def test_v2_primary_rule_and_r1b(tmp_path, monkeypatch):
    from selfdoubt import stage1_report as RP
    n = 100
    good = {"V0": [0] * 60 + [1] * 40, "VG": [1] * 90 + [0] * 10}
    flat = {"V0": [1] * 80 + [0] * 20, "VG": [1] * 82 + [0] * 18}
    rest = {a: [0] * n for a in ("Vp", "Vc", "I1", "I2a", "I2d", "I2b", "I2c")}
    stores = {"s1t": flat, "s0": good}
    monkeypatch.setattr(RP, "split_ids", lambda name: set(range(n)))
    monkeypatch.setattr(RP.C, "load_items", lambda: [])

    class St:
        def __init__(self, path):
            self.recs = {i: {"item_id": i, "sys": Path(path).name} for i in range(1)}
    monkeypatch.setattr(RP.C, "Store", St)
    monkeypatch.setattr(RP.S1, "score", lambda recs, items: rows_for({**rest, **stores[recs[0]["sys"]]}))
    out = RP.v2(Path("s1t"), Path("s0"), "dev", 200)
    assert out["primary"]["gates"] == {"s1t": False, "s0": True} and out["primary"]["setting"] == "s0"
    stores["s1t"] = good
    assert RP.v2(Path("s1t"), Path("s0"), "dev", 200)["primary"]["setting"] == "s1t"
    stores["s1t"] = stores["s0"] = flat
    o = RP.v2(Path("s1t"), Path("s0"), "dev", 200)["primary"]
    assert o["setting"] is None and "UNDETERMINED" in o["decision"]
    with pytest.raises(SystemExit):                       # a store must hold the setting it is passed as
        RP.v2(Path("s0"), Path("s0"), "dev", 200)


def test_paired_equivalence():
    from selfdoubt import stage1_report as RP
    from selfdoubt.stage0_report import verdict
    p = RP._paired({i: 0 for i in range(100)}, 200, 0.90)
    assert p["ci"] == [0.0, 0.0] and verdict(*p["ci"], RP.EQ_BAND, "within") == "SHOWN"
    p = RP._paired({i: (1 if i < 50 else 0) for i in range(100)}, 200, 0.90)
    assert verdict(*p["ci"], RP.EQ_BAND, "within") == "NOT SHOWN"
