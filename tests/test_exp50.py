"""§50 (rev 1 §7.2 build tests): nuisance balance, rep b = timestamps only, UNA-swap memory byte-identical, §49 overlap
report, the no-op replay (stub model and fake engine), projector algebra; plus the answer-bearing rule, S scoring,
cell estimands with the fallback, states / profiles, Holm, the held guard, and the whole chain on the fake engine."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp45_cases as C  # noqa: E402
import exp50_cases as G  # noqa: E402
import exp50_run as R  # noqa: E402
import exp50_report as REP  # noqa: E402
import exp49_run as R49  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from memctl.analysis import why50 as W  # noqa: E402


@pytest.fixture(scope="module")
def built():
    rows, info = G.build_cases(48)
    kept, ov = G.drop_overlap(rows)
    return rows, kept, info, ov


def test_checks_pass(built):
    _, kept, _, _ = built
    pr = G.check_cases(kept)
    assert not any(pr.values()), {k: v[:3] for k, v in pr.items() if v}


def test_render_shift_zero_is_s45_renderer(built):
    rows, *_ = built
    r = next(x for x in rows if x["set"] == "ANS" and x["rep"] == "a" and x["format"] == "F2")
    assert G.render_f2(r["facts"], r["question"], 0) == C.render_prompt("F2", r["facts"], r["question"]) == r["prompt"]


def test_rep_b_differs_in_timestamps_only(built):
    rows, *_ = built
    by = {r["case_id"]: r for r in rows}
    n = 0
    for r in rows:
        if r["set"] == "ANS" and r["rep"] == "b":
            a = by[r["case_id"].replace("_ANSb_", "_ANSa_")]
            assert a["facts"] == r["facts"] and a["question"] == r["question"]
            assert G.DATE_RE.sub("D", a["prompt"]) == G.DATE_RE.sub("D", r["prompt"])
            da, db = G.DATE_RE.findall(a["prompt"]), G.DATE_RE.findall(r["prompt"])
            assert len(da) == len(db) and all(x != y and x[-5:] == y[-5:] for x, y in zip(da, db))   # weekday kept
            n += 1
    assert n == 2 * 3 * 48 * 3


def test_una_swap_memory_byte_identical(built):
    rows, *_ = built
    by = {r["case_id"]: r for r in rows}
    sw = [r for r in rows if r["set"] == "UNA-swap"]
    assert sw
    for r in sw:
        a = by[r["case_id"].replace("_UNAswap_", "_ANSa_")]
        assert a["prompt"].split("## Current input")[0] == r["prompt"].split("## Current input")[0]
        d = r["swap_entity"]
        lines = [f for f in r["facts"] if re.search(r"\b" + d + r"\b", f)]
        assert len(lines) == 1 and " works as " in lines[0]
        assert d not in (r["A"], r["B"])


def test_nuisance_balance(built):
    rows, *_ = built
    for task in G.TASKS:
        c = Counter((r["set"], "pet" if " has a" in r["nuisance"] or " has an" in r["nuisance"] else "job")
                    for r in rows if r["task"] == task and r["format"] == "F2")
        kinds = {k for _, k in c}
        assert len(kinds) == 1                         # one nuisance kind per task, in ANS and UNA alike
        for r in rows:
            if r["task"] == task:
                assert r["nuisance"] in r["facts"]


def test_una_rel_replacements_never_repeat_a_line(built):
    rows, *_ = built
    for r in rows:
        if r["set"] == "UNA-rel":
            assert len(set(r["facts"])) == len(r["facts"])
            assert all(rep in r["facts"] for rep in r["una_replacements"])
            assert not any(G.G.cities_in(f) and r["query_entity"] in f for f in r["facts"])


def test_overlap_report_and_drop(built):
    rows, kept, _, ov = built
    assert "dropped_exact" in ov and "core_tuple_overlap_bases" in ov
    old = {json.loads(l)["prompt_sha256"] for l in G.CASES49.read_text().splitlines() if l.strip()}
    assert not any(r["prompt_sha256"] in old for r in kept)
    assert len(rows) - len(kept) == len(ov["dropped_exact"])


def test_parts_and_scale_down():
    rows, _ = G.build_cases(24)
    assert {r["part"] for r in rows} == {"dev-fit", "dev-check", "held"}
    assert all(r["part"] == ("held" if r["base_index"] % 2 else "dev-fit" if r["base_index"] % 4 == 0 else "dev-check")
               for r in rows)
    g = Counter(r["gold"] for r in rows if (r["split"], r["task"], r["set"], r["rep"], r["format"], r["hardness"]) == ("dev", "DE", "ANS", "a", "F2", "H1"))
    assert set(g.values()) == {2}


# ---------------------------------------------------------------------------------------------- measures

def test_answer_bearing_rule():
    cities = C.CITIES
    assert W.answer_bearing("Ben lives in York.", "York", "DE", cities)
    assert not W.answer_bearing("Ben does not live in York.", "York", "DE", cities)
    assert not W.answer_bearing("Ben moved from York to Leeds.", "York", "TU", cities)
    assert W.answer_bearing("Henry moved from Derby to Brighton.", "Brighton", "TU", cities)     # dev-audit fix
    assert W.answer_bearing("Paul moved from Liverpool to Derby", "Derby", "TU", cities)
    assert not W.answer_bearing("Paul moved from Derby.", "Derby", "DE", cities)
    assert W.answer_bearing("Ben lived in Leeds, later moved to York.", "York", "TU", cities)
    assert not W.answer_bearing("Ben is in York; before that Leeds.", "York", "TU", cities)      # not last city
    assert W.answer_bearing("not that one. Ben is known to live in York.", "York", "DE", cities)   # negator 7+ words back
    assert not W.answer_bearing("nothing", "York", "DE", cities)


def test_yes_no_and_digits():
    z = np.zeros(10)
    z[[1, 2]] = 5.0
    z[[3]] = 4.0
    out = W.yes_no(z, [1, 2], [3])
    assert out["valid"] and out["yes"] and 0.5 < out["s"] < 1
    d = W.expected_digit(np.array([0.0, 0, 0, 10]), [[0], [1], [2], [3]])
    assert abs(d["kv"] - 3) < 1e-3


def test_cell_delta_and_fallback():
    rows = [{"task": "DE", "hardness": "H0", "gold": "York", "cls": "fa", "m": 0.0},
            {"task": "DE", "hardness": "H0", "gold": "York", "cls": "ans", "m": 2.0},
            {"task": "DE", "hardness": "H0", "gold": "Leeds", "cls": "fa", "m": 1.0},
            {"task": "DE", "hardness": "H0", "gold": "Derby", "cls": "ans", "m": 5.0}]
    fine = W.cell_delta(rows, "m", coarse=False)
    assert fine["delta"] == -2.0 and fine["covered_fa"] == 1
    auto = W.cell_delta(rows, "m")                    # 1 of 2 false abstentions covered: 50% -> stays fine
    assert auto["coarse"] is False
    rows[0]["gold"] = "Cardiff"                        # 0 of 2 covered -> fallback to task x hardness
    auto = W.cell_delta(rows, "m")
    assert auto["coarse"] is True and auto["delta"] == pytest.approx(0.5 - 3.5)


def test_states_profiles_holm():
    ci = {"lo95": -0.9, "hi95": -0.6, "lo90": -0.85, "hi90": -0.65}
    assert W.state(-0.7, ci, 0.5) == "lower"
    assert W.state(0.0, {"lo95": -0.3, "hi95": 0.3, "lo90": -0.2, "hi90": 0.2}, 0.5) == "similar"
    assert W.state(0.0, {"lo95": -0.3, "hi95": 0.3, "lo90": -0.2, "hi90": 0.2}, None) == "inconclusive"
    assert W.state(-0.7, ci, 0.5, valid=False) == "invalid / underpowered"
    assert W.profile("lower", "similar") == "B-compatible" and W.profile("similar", "lower") == "C-compatible"
    assert W.profile("lower", "lower") == "Joint" and W.profile("similar", "similar") == "Neither separates"
    h = W.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)


def test_projector_algebra():
    rng = np.random.default_rng(0)
    u_s = W.unit(rng.standard_normal(16))
    u_m = W.unit(u_s + 0.7 * rng.standard_normal(16))
    assert abs(u_s @ u_m) > 0.1                                   # not orthogonal
    P = W.projector(np.stack([u_s, u_m], 1))
    assert np.allclose(P @ P, P) and np.allclose(P, P.T)
    h_r, h_d = rng.standard_normal(16), rng.standard_normal(16)
    arms = W.arm_vectors(h_r, h_d, u_s, u_m)
    delta = h_d - h_r
    assert np.allclose(arms["P-span"]["disp"] + arms["P-perp"]["disp"], delta)       # span + perp = W
    assert np.allclose(arms["W"]["state"], h_d, atol=1e-5)
    assert not np.allclose(arms["P-S"]["disp"] + arms["P-M"]["disp"], arms["P-span"]["disp"])   # handled by Π
    assert abs(arms["P-perp"]["disp"] @ u_s) < 1e-9 and abs(arms["P-perp"]["disp"] @ u_m) < 1e-9
    rs = W.rand_states(h_r, arms["P-S"]["disp"], "p", "S", delta=delta)
    assert len(rs) == 5 and all(np.isclose(np.linalg.norm(x - h_r), arms["P-S"]["norm"], rtol=1e-5) for x in rs)
    assert all(float((x - h_r) @ delta) >= 0 for x in rs)                 # fix 3: sign against the whole Δ
    neg = W.rand_states(h_r, arms["P-S"]["disp"], "p", "S", delta=-delta)
    assert all(np.allclose(a - h_r, -(b - h_r), atol=1e-5) for a, b in zip(rs, neg))
    assert np.array_equal(rs[0], W.rand_states(h_r, arms["P-S"]["disp"], "p", "S", delta=delta)[0])


def test_noop_replay_on_stub_model():
    """Overwriting the P_ans state with its own captured state reproduces the continuation token for token."""
    e = R49.Engine(st.StubEngine())
    ids = e.chat_ids("Where does Ben live now?")
    base = e.generate(ids, [], 20, lambda t: False)
    pre = ids + base[:6]
    own = e.capture(pre, [len(pre) - 1])[0]
    for layer in (3, 10):
        assert e.generate(pre, [st.overwrite_spec(layer, own[layer])], 14, lambda t: False) == base[6:20]
    other = own[10] + 50.0
    assert e.generate(pre, [st.overwrite_spec(10, other)], 14, lambda t: False) != base[6:20]


def test_held_guard(tmp_path):
    rows, _ = G.build_cases(24)
    cases = tmp_path / "c.jsonl"
    cases.write_text("".join(json.dumps(r) + "\n" for r in rows[:5]))
    for stage in (["--stage", "measure", "--split", "held"], ["--stage", "stage2-held"], ["--stage", "stage1-report"]):
        assert R.main(stage + ["--cases", str(cases), "--out", str(tmp_path / "o"), "--backend", "fake"]) == R.EXIT_DESIGN


def test_whole_chain_on_fake_engine(tmp_path):
    rows, _ = G.build_cases(24)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = tmp_path / "smoke"
    assert R.main(["--smoke", "8", "--backend", "fake", "--cases", str(cases), "--out", str(root)]) == 0
    sm = json.loads((root / "smoke.json").read_text())
    assert sm["pass"] and sm["stage2_dev"]["noop_reproduces"] == 1.0
    assert {"W", "P-S", "P-span", "P-perp", "Rand-S", "Same"} <= set(sm["s2held_arms"])
    same = [json.loads(p.read_text()) for p in (root / "s2held").glob("*__Same.json")]
    for x in same:                                                      # fix 1: the arm's own recipient
        rid = x["recipient"]
        assert ("_ANS" in rid) and x["donor"].split("_")[0] != rid.split("_")[0]
    rep = json.loads((root / "report.json").read_text())
    assert set(rep["stage1"]["tasks"]) == {"DE", "RC", "TU"}
    REP.markdown(rep)
