"""§41: the signals at ':' (margin arithmetic, projection, the step-0 logits returned by Engine.run), the label
sources, the utility with soft refusals and lambda, the C:R weighting, the tau fit and its tie-break, the gate,
apply_rule, the option (b) timing gate, fold refusal, resume, the byte-identity stop, an end-to-end stub run and the
refusal-start id set on the real §38 rows (scripts/exp41_dev.py, scripts/exp41_report.py). StubEngine only; no Qwen
weights are loaded."""
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import conditional as cd
from memctl.analysis import steering as st
from tests.test_exp39 import CACHE, SOURCE, _item, _lines, _make_stub_source

needs_data = pytest.mark.skipif(not (CACHE.exists() and (SOURCE / "directions.npz").exists()),
                                reason="§35 cache or §38 dev run not on disk")


# --- signals --------------------------------------------------------------------------------------------------

def test_margin_arithmetic():
    x = np.array([1.0, 5.0, 3.0, 4.5, -2.0])
    assert cd.margin(x, [1]) == pytest.approx(0.5)                 # 5 - max(1, 3, 4.5, -2)
    assert cd.margin(x, [2]) == pytest.approx(3.0 - 5.0)
    assert cd.margin(x, [2, 3]) == pytest.approx(4.5 - 5.0)        # max over the set
    with pytest.raises(ValueError):
        cd.margin(x, [])
    with pytest.raises(ValueError):
        cd.margin(x, range(5))
    s = cd.step0_summary(x, [2, 3], top=3)
    assert s["refusal_max"] == 4.5 and s["other_max"] == 5.0 and s["other_argmax"] == 1 and s["argmax"] == 1
    assert s["margin"] == pytest.approx(-0.5) and s["top"] == [[1, 5.0], [3, 4.5], [2, 3.0]] and s["vocab"] == 5
    assert s["refusal_ids"] == [2, 3]


def test_projection_and_signals():
    h, v = np.array([3.0, 4.0, 0.0]), np.array([0.0, 2.0, 0.0])
    assert cd.projection(h, v) == pytest.approx(4.0)                # unit direction: scale of v does not matter
    assert cd.projection(h, 10 * v) == pytest.approx(4.0)
    assert cd.projection(h, -v) == pytest.approx(-4.0)
    s = cd.signals_of(2.5, -1.0, h, v)
    assert s == {"s_margin": 2.5, "s_proj": pytest.approx(4.0), "s_evid": 3.5}


def test_engine_run_returns_step0_logits_without_changing_the_generation():
    eng = st.StubEngine()
    prepared = eng.prepare(_item())
    g0 = eng.run(prepared, [], 12)
    g1 = eng.run(prepared, [], 12, capture=(16,), return_logits=True)
    assert g0.logits0 is None and g1.logits0 is not None and g1.logits0.shape == (257,)
    assert g1.gen_ids == g0.gen_ids and g1.generated == g0.generated
    assert int(np.argmax(g1.logits0)) == g1.first_token
    assert g1.captured[16].shape == (16,)
    # the logits are the step-0 pass itself: replay it
    import copy
    import torch
    cache = copy.deepcopy(prepared.cache)
    with torch.no_grad():
        ref = eng._step(prepared.last_id, cache).numpy()
    assert np.array_equal(ref, g1.logits0)
    # under a hook the logits0 are those of the hooked pass
    g2 = eng.run(prepared, [st.add_spec(19, np.full(16, 5.0, np.float32))], 4, return_logits=True)
    assert not np.array_equal(g2.logits0, g1.logits0)


# --- labels, utility, weights ---------------------------------------------------------------------------------

def test_label_sources():
    args = dict(phrase="Luna", cls="answer", gold="Luna", cached_a="Luna", verdict_a=True, cached_b="Luna.",
                verdict_b=False)
    assert cd.label("Luna", **args) == (True, "cache:A")
    assert cd.label("Luna.", **args) == (False, "cache:B")
    a2 = dict(args, cached_b="Luna", verdict_b=False)            # equal to both: A first
    assert cd.label("Luna", **a2) == (True, "cache:A")
    assert cd.label(" Luna", **dict(args, phrase="Luna")) == (True, "D1")      # not byte-equal -> D1 on the phrase
    assert cd.label("Luna the cat", **dict(args, phrase="Luna the cat")) == (True, "D1")
    assert cd.label("Max", **dict(args, phrase="Max")) == (False, "D1")
    assert cd.label("Luna", **dict(args, verdict_a=None)) == (True, "D1")       # no cached verdict -> not a cache label
    assert cd.label("Luna", **dict(args, phrase="unknown", cls="refusal", verdict_a=None)) == (False, "D1")
    assert cd.label("Max", **dict(args, phrase="Max"), judge=lambda t: True) == (True, "judge7b")
    assert cd.label("Luna", **args, judge=lambda t: False) == (True, "cache:A")  # the cache wins over the judge


def test_utility_soft_refusals_and_lambda():
    assert cd.outcome("refusal", "unknown", True) == "refusal"
    assert cd.outcome("degenerate", "", True) == "degenerate"
    assert cd.outcome("answer", "Favorite rice NOT specified", True) == "soft"       # case-insensitive, beats a label
    assert cd.outcome("answer", "First event participation cannot be determined", False) == "soft"
    assert cd.outcome("answer", "The memory does not contain it", False) == "soft"
    assert cd.outcome("answer", "No information given", False) == "soft"
    assert cd.outcome("answer", "Serenity Yoga", True) == "correct"
    assert cd.outcome("answer", "Serenity Yoga", False) == "wrong"
    assert not cd.soft_refusal("not mentioned", "refusal")
    assert [cd.utility(o) for o in ("correct", "wrong", "soft", "refusal", "degenerate")] == [1, -1, 0, 0, 0]
    assert cd.utility("wrong", 2) == -2 and cd.utility("correct", 2) == 1
    outc = {"r1": "correct", "r2": "wrong", "c1": "correct", "c2": "wrong", "c3": "soft"}
    pops = {"r1": "R", "r2": "R", "c1": "C", "c2": "C", "c3": "C"}
    w = cd.weights(2, 3)
    assert w == {"R": 1, "C": Fraction(2, 9)}
    assert cd.total(outc, pops, None, 1) == 0 and cd.total(outc, pops, None, 2) == -2
    assert cd.total(outc, pops, w, 1) == 0 and cd.total(outc, pops, w, 2) == 1 - 2 + Fraction(2, 9) * (1 - 2)


def test_weighting_arithmetic_on_the_dev_counts():
    w = cd.weights(28, 39)
    assert w["C"] == Fraction(28, 117) == Fraction(28, 3) / 39 and w["R"] == 1
    assert w["C"] * 39 == Fraction(28, 3) and (w["C"] * 39) / (w["R"] * 28) == Fraction(1, 3)
    assert abs(float(w["C"]) - 0.239316) < 1e-6
    assert cd.weights(120, 40)["C"] == 1                          # §40's caps: 40 : 120 is already 1 : 3


# --- fit, tie-break, gate, rule -------------------------------------------------------------------------------

def _toy():
    """6 R rows (A refuses) and 3 C rows (A correct); Steer right on r1-r3, wrong on r4-r5, soft on r6; Steer damages
    c2. s_margin separates helpful from harmful rows (low margin = steer helps)."""
    a = {f"r{i}": "refusal" for i in range(1, 7)} | {"c1": "correct", "c2": "correct", "c3": "correct"}
    s = {"r1": "correct", "r2": "correct", "r3": "correct", "r4": "wrong", "r5": "wrong", "r6": "soft",
         "c1": "correct", "c2": "wrong", "c3": "correct"}
    pops = {q: "R" if q[0] == "r" else "C" for q in a}
    sig = {"s_margin": {"r1": 1.0, "r2": 2.0, "r3": 3.0, "r4": 7.0, "r5": 8.0, "r6": 9.0, "c1": 4.0, "c2": 10.0,
                        "c3": 5.0},
           "s_proj": {q: 0.0 for q in a},                         # constant: every tau gives Steer or A
           "s_evid": {q: -v for q, v in {"r1": 1.0, "r2": 2.0, "r3": 3.0, "r4": 7.0, "r5": 8.0, "r6": 9.0,
                                         "c1": 4.0, "c2": 10.0, "c3": 5.0}.items()}}
    return a, s, pops, sig


def test_fit_picks_the_best_rule_and_breaks_ties_deterministically():
    a, s, pops, sig = _toy()
    w = cd.weights(6, 3)
    f = cd.fit(sig, a, s, pops, w)
    c = f["chosen"]
    # best: steer iff s_margin < tau with tau in (3, 7]: rows r1-r3 (+3) and c1, c3 kept; s_evid '+' gives the same
    # utility (mirror image), the signal order picks s_margin; the taus tied for (s_margin, '-') are 4, 5, 7 -> 5
    assert (c["signal"], c["sign"]) == ("s_margin", "-")
    # utility: R +3 (r1-r3), C kept at A (3 x 2/3) -> 5; unweighted 3 + 3 = 6; fires r1-r3 and c1 (4 < 5)
    assert f["tied_taus"] == [4.0, 5.0, 7.0] and c["tau"] == 5.0
    assert c["weighted"] == 5 and c["unweighted"] == 6 and c["fires"] == 4
    assert cd.lower_median([1, 2, 3, 4]) == 2 and cd.lower_median([math.inf]) == math.inf
    # the same answer whatever the row order of the dicts
    rev = {n: dict(reversed(list(v.items()))) for n, v in sig.items()}
    f2 = cd.fit(rev, dict(reversed(list(a.items()))), dict(reversed(list(s.items()))), pops, w)
    assert f2["chosen"] == c
    # Steer and A are in the grid (+-inf); they are equal for every signal
    steer_u = cd.total(s, pops, w)
    assert any(x["tau"] == -math.inf and x["sign"] == "+" and x["weighted"] == steer_u for x in f["candidates"])
    assert len(f["candidates"]) == 2 * (len(set(sig["s_margin"].values())) + 2) + 2 * 3 + 2 * (9 + 2)
    best = cd.best_per_signal(f)
    assert best["s_proj"]["weighted"] == max(cd.total(a, pops, w), steer_u)
    assert best["s_evid"]["sign"] == "+" and best["s_evid"]["weighted"] == 5


def test_fit_unweighted_tie_break():
    # s_margin can isolate r1 (weighted +1, unweighted +1), s_proj c1-c3 (weighted 3 x 1/3 = 1, unweighted 3): equal
    # weighted utility, the higher unweighted one wins although s_proj comes later in the signal order
    a = {q: "refusal" for q in ("r1", "r2", "c1", "c2", "c3", "c4")}
    s = {"r1": "correct", "r2": "wrong", "c1": "correct", "c2": "correct", "c3": "correct", "c4": "wrong"}
    pops = {q: q[0].upper() for q in a}
    w = {"R": Fraction(1), "C": Fraction(1, 3)}
    sig = {"s_margin": {q: 1.0 if q == "r1" else 0.0 for q in a},
           "s_proj": {q: 1.0 if q in ("c1", "c2", "c3") else 0.0 for q in a},
           "s_evid": {q: 0.0 for q in a}}
    f = cd.fit(sig, a, s, pops, w)
    assert (f["chosen"]["signal"], f["chosen"]["sign"], f["chosen"]["tau"]) == ("s_proj", "+", 0.0)
    assert f["chosen"]["weighted"] == 1 and f["chosen"]["unweighted"] == 3 and f["chosen"]["fires"] == 3
    assert f["n_tied_at_top"] >= 2


def test_gate_arithmetic():
    assert cd.gate(Fraction(10), Fraction(7))["pass"]                       # exactly 3 weighted units
    assert not cd.gate(Fraction(10), Fraction(7) + Fraction(1, 117))["pass"]
    g = cd.gate(Fraction(5) + Fraction(28, 117), Fraction(2))
    assert g["pass"] and g["diff"] == Fraction(3) + Fraction(28, 117)
    assert not cd.gate(2, 0)["pass"]


def test_apply_rule():
    sig = {"s_margin": 1.5, "s_proj": -3.0, "s_evid": 0.2}
    assert cd.apply_rule({"signal": "s_margin", "sign": "+", "tau": 1.0}, sig)
    assert not cd.apply_rule({"signal": "s_margin", "sign": "+", "tau": 1.5}, sig)          # strict
    assert cd.apply_rule({"signal": "s_proj", "sign": "-", "tau": -2.0}, sig)
    assert cd.apply_rule({"signal": "s_evid", "sign": "+", "tau": "-inf"}, sig)             # always: Steer
    assert not cd.apply_rule({"signal": "s_evid", "sign": "+", "tau": "inf"}, sig)          # never: A
    rule = json.loads(json.dumps({"signal": "s_proj", "sign": "-", "tau": "inf"}))
    assert cd.apply_rule(rule, sig)
    with pytest.raises(ValueError):
        cd.apply_rule({"signal": "s_other", "sign": "+", "tau": 0}, sig)
    with pytest.raises(ValueError):
        cd.apply_rule({"signal": "s_margin", "sign": "+", "tau": 0}, {"s_margin": float("nan")})
    with pytest.raises(ValueError):
        cd.apply_rule({"signal": "s_margin", "sign": "*", "tau": 0}, sig)
    assert sig == {"s_margin": 1.5, "s_proj": -3.0, "s_evid": 0.2}                           # pure


def test_auroc_coverage_and_judge_gate():
    assert cd.auroc([3, 2, 1], [True, False, False]) == 1.0
    assert cd.auroc([1, 2, 3], [True, False, False]) == 0.0
    assert cd.auroc([1, 1], [True, False]) == 0.5
    assert cd.auroc([1, 2], [True, True]) is None
    cr = cd.coverage_risk({"a": "correct", "b": "wrong", "c": "soft", "d": "refusal"})
    assert (cr["answered"], cr["coverage"], cr["risk"]) == (2, 0.5, 0.5)
    g = cd.judge_timing_gate([12.0] * 5, 260)
    assert g["projected_min"] == pytest.approx(12 * 260 / 60) and not g["pass"]           # 52 min >= 45
    assert cd.judge_timing_gate([10.0] * 5, 270)["projected_min"] == 45.0
    assert not cd.judge_timing_gate([10.0] * 5, 270)["pass"]                                # under 45, strictly
    assert cd.judge_timing_gate([2.0] * 5, 260)["pass"]
    with pytest.raises(ValueError):
        cd.judge_timing_gate([1.0] * 4, 10)


# --- the real §38 rows ----------------------------------------------------------------------------------------

@needs_data
def test_refusal_start_ids_on_the_real_38_rows(capsys):
    from scripts.exp35_generate import read_jsonl
    rows = read_jsonl(SOURCE / "repro.jsonl")
    ids = cd.refusal_start_ids(rows)
    with capsys.disabled():
        print(f"\n§41 refusal-start ids on {SOURCE / 'repro.jsonl'}: {ids} "
              f"({sum(r['cls'] == 'refusal' for r in rows)} refusal-class A/B rows)")
    assert ids == [9788]                                  # ' unknown', as the §39b build found for the A rows


# --- the dev script -------------------------------------------------------------------------------------------

def test_dev_refuses_confirmation_folds(capsys):
    from scripts import exp41_dev
    assert exp41_dev.main(["--folds", "0,2"]) == 2
    assert "REFUSED" in capsys.readouterr().out


@pytest.fixture(scope="module")
def stub41(tmp_path_factory):
    if not (CACHE.exists() and (SOURCE / "directions.npz").exists()):
        pytest.skip("§35 cache or §38 dev run not on disk")
    from scripts import exp41_dev
    root = tmp_path_factory.mktemp("exp41")
    src = _make_stub_source(root / "src")
    out = root / "out"
    assert exp41_dev.main(["--run", "--backend", "stub", "--source", str(src), "--out", str(out)]) == 0
    return src, out / "qwen3b"


def test_stub_run_writes_every_row_sha_stamped(stub41):
    src, run_dir = stub41
    assert _lines(run_dir) == {"prefill.jsonl": 67 * 2, "steer.jsonl": 39, "steer_all.jsonl": 28}
    man = json.loads((run_dir / "manifest.json").read_text())
    ids = man["refusal_start_ids"]
    assert ids and man["n_R"] == 28 and man["n_C"] == 39 and len(man["damage"]) == 8
    pre = [json.loads(x) for x in (run_dir / "prefill.jsonl").read_text().splitlines()]
    assert {(r["pop"], r["kind"]) for r in pre} == {("R", "A"), ("R", "cut"), ("C", "A"), ("C", "cut")}
    for r in pre:
        assert r["setting"]["key"] == "none" and r["vector_sha256"] is None and r["engine"] == "stub"
        assert r["step0"]["refusal_ids"] == ids and r["last_token"] == ":" and r["wording_sha256"] == st.wording_sha256(16)
        assert r["step0"]["argmax"] == r["first_token"]
        h = np.load(run_dir / "resid" / r["kind"] / f"{r['question_id']}.npy")
        assert h.shape == (16,) and st.vector_sha(h) == r["h29_sha256"] and r["capture_layer"] == st.map_layer(29, 20)
    steer = [json.loads(x) for x in (run_dir / "steer.jsonl").read_text().splitlines()]
    from scripts import exp39_dev
    vec = exp39_dev.engine_vector(exp39_dev.load_direction(SOURCE)["steer"], st.StubEngine())
    assert all(r["pop"] == "C" and r["kind"] == "A" and r["vector_sha256"] == st.vector_sha(vec)
               and r["setting"]["key"] == f"steer:L{st.map_layer(29, 20)}:a2:vnorm" for r in steer)
    rep = json.loads((run_dir / "repro.json").read_text())
    assert rep["R"]["k"] == rep["R"]["n"] == 28 and rep["C"]["n"] == 39


def test_stub_run_resumes_without_new_rows(stub41):
    from scripts import exp41_dev
    src, run_dir = stub41
    before = _lines(run_dir)
    assert exp41_dev.main(["--run", "--backend", "stub", "--source", str(src), "--out", str(run_dir.parent)]) == 0
    assert _lines(run_dir) == before


def test_byte_identity_failure_stops_before_any_steered_generation(stub41, tmp_path, capsys):
    from scripts import exp41_dev
    src, _ = stub41
    bad = tmp_path / "src"
    shutil.copytree(src, bad)
    rows = [json.loads(x) for x in (bad / "repro.jsonl").read_text().splitlines()]
    for r in rows:
        if r["pop"] == "R" and r["kind"] == "A":
            r["generated"] += "x"
            break
    (bad / "repro.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    assert exp41_dev.main(["--run", "--backend", "stub", "--source", str(bad), "--out", str(out)]) == exp41_dev.EXIT_BYTES
    assert "stopping before any steered generation" in capsys.readouterr().out
    assert not (out / "qwen3b" / "steer.jsonl").exists()


# --- the report -----------------------------------------------------------------------------------------------

def test_report_fits_freezes_and_apply_rule_reproduces_the_fires(stub41, tmp_path):
    from scripts import exp41_report
    src, run_dir = stub41
    out, rule_path = tmp_path / "r.md", tmp_path / "rule.json"
    assert exp41_report.main(["--run-dir", str(run_dir), "--source", str(src), "--out", str(out),
                              "--rule", str(rule_path)]) == 0
    text = out.read_text()
    assert "DEVELOPMENT ONLY" in text and "PROVISIONAL" in text and "Frozen rule" in text and "28/117" in text
    rule = json.loads(rule_path.read_text())
    numbers = json.loads(out.with_suffix(".json").read_text())
    assert rule["signal"] in cd.SIGNALS and rule["sign"] in cd.SIGNS and rule["weights"]["C"] == "28/117"
    assert rule["refusal_start_ids"] == json.loads((run_dir / "manifest.json").read_text())["refusal_start_ids"]
    assert rule["lambda"] == 1 and rule["engine"] == "stub" and rule["folds"] == [0, 1]
    assert set(rule["input_sha256"]) >= {"prefill.jsonl", "steer.jsonl", "steer_all.jsonl", "source repro.jsonl",
                                         "source directions.npz", "exp35 A fold0", "exp35 judge fold1"}
    for pol, srcs in rule["label_sources"].items():
        assert sum(srcs.values()) == 67 and srcs["judge7b"] == 0, pol
    sig = numbers["signals"]
    fires = {q: cd.apply_rule(rule, {n: sig[n][q] for n in cd.SIGNALS}) for q in sig["s_margin"]}
    assert fires == numbers["fires"]
    outc = numbers["outcomes"]
    assert all(outc["Cond"][q] == (outc["Steer"][q] if fires[q] else outc["A"][q]) for q in fires)
    pol = numbers["policies"]
    assert pol["Cond"]["R"]["n"] == 28 and pol["Cond"]["C"]["n"] == 39
    g = numbers["gate"]
    assert g["pass"] == (Fraction(g["diff"]["exact"]) >= 3) and rule["scored_on_40"] == g["pass"]
    # Cond is at least as good as Steer and A on the fitted weighted utility (both are in the grid)
    val = lambda p: Fraction(pol[p]["weighted_l1"]["exact"])  # noqa: E731
    assert val("Cond") >= val("Steer") and val("Cond") >= val("A")
    assert set(numbers["auroc_gold_steer"]) == set(cd.SIGNALS) and len(numbers["curves"]) == 6


def test_report_refuses_mixed_engines_other_folds_and_option_b(stub41, tmp_path, capsys):
    from scripts import exp41_report
    src, run_dir = stub41
    for field, value in (("engine", "hf-other"), ("fold", 2)):
        bad = tmp_path / field
        shutil.copytree(run_dir, bad)
        rows = [json.loads(x) for x in (bad / "steer.jsonl").read_text().splitlines()]
        rows[0][field] = value
        (bad / "steer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert exp41_report.main(["--run-dir", str(bad), "--source", str(src), "--out", str(tmp_path / "x.md")]) == 2
        assert "REFUSED" in capsys.readouterr().out
        assert not (bad / "rule.json").exists()
    assert exp41_report.main(["--run-dir", str(run_dir), "--source", str(src), "--out", str(tmp_path / "y.md"),
                              "--judge-7b"]) == 2
    assert "option (b)" in capsys.readouterr().out


@needs_data
def test_run_guards_on_the_real_38_rows(tmp_path):
    from scripts import exp39_dev, exp41_dev
    from tests.test_exp39 import _NamedEngine, _dev_data
    d, pop = _dev_data()
    direction = exp39_dev.load_direction(SOURCE)
    run = exp41_dev.Run(_NamedEngine("hf-transformers-5.17.0-cpu-float32"), d, pop, tmp_path, SOURCE, direction)
    assert run.ids == [9788] and len(run.R) == 28 and len(run.C) == 39 and pop.damage == run.C[:8]
    assert st.vector_sha(run.vec) == exp39_dev.STEER_VECTOR_SHA256 and run.v29.shape == (2048,)
    assert run.reuse_steer_all(pop.R)["n"] == 28                  # the §39 guard accepts the real Steer-all rows
    other = exp41_dev.Run(_NamedEngine("hf-transformers-5.17.0-cuda-bfloat16"), d, pop, tmp_path / "o", SOURCE,
                          direction)
    with pytest.raises(SystemExit, match="REFUSED \\(Steer-all reuse\\).*engine"):
        other.reuse_steer_all(pop.R)
