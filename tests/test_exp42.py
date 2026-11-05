"""§42: selective steering by s_proj on fresh LoCoMo rows (proposal research/proposals/2026-10-25-exp42-selective-
steering.md, rev 1 at 8fd2613). The frozen rule (rule42.json) recomputed from §40's outputs, the fresh set (seed 42,
§40's 640 ids excluded, disjoint, pools 923 / 406), §40's set unchanged, the Cond firing logic (A refuses AND
s_proj >= tau; C rows never fire), the claims arithmetic on forged stub rows, the report end to end with --exp exp42,
the exp42 bundle. No Qwen weights; the stub engine and judge."""
import hashlib
import json
import random
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from memctl import locomo_view as lv
from memctl.analysis import conditional as cd
from memctl.analysis import exp40 as ex

LOCOMO = Path(lv.LOCOMO_PATH)
NPZ = Path("runs/exp38_dev/qwen3b/directions.npz")
AGREE = Path("runs/exp40_inputs/agreement60.jsonl")
VIEW40 = Path(ex.EXCLUDE_VIEW)
APASS40 = Path(ex.APASS40)
RULE42 = Path(ex.RULE42_PATH)
have40 = pytest.mark.skipif(not (VIEW40.exists() and APASS40.exists() and Path(ex.SELECTION40).exists()),
                            reason="§40's outputs not on disk")
needs_inputs = pytest.mark.skipif(not (LOCOMO.exists() and NPZ.exists() and AGREE.exists() and VIEW40.exists()
                                       and RULE42.exists()), reason="LoCoMo, §38/§40 inputs or rule42 not on disk")
N_ROWS, N_ADV = 60, 8
SHAS40 = {"rows": "8da240521cbd03320b5c63b839c481f3937902dfac5f0f3d96aa140ac4a836d0",
          "adversarial": "ea3d5581a00d7374d2c49f3a778f5f3a520f6e74367935a913c62f7cb0e6829e",
          "row_ids": "0d71d318e37e0466bd393994b7c63ec421cf28b3b8577d1833222b221ecddf71",
          "adversarial_ids": "1656c7421a7acb7acf2ba80af94cf77d650d1fe815b27acefc8bff557ce0448c"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the rule ------------------------------------------------------------------------------------------------------

@have40
def test_rule42_recomputes_from_40s_outputs():
    from scripts import exp42_rule as r42
    assert _sha(APASS40) == ex.APASS40_SHA256
    rule = r42.build()
    assert rule["tau"] == ex.TAU42 == -18.896762090918696
    assert RULE42.exists() and json.loads(RULE42.read_text()) == rule
    assert RULE42.read_text() == r42.text_of(rule)                      # byte-equal to the committed file
    assert rule["input_sha256"] == ex.APASS40_SHA256 and rule["fires_on_40_R"] == 36 and rule["n_R"] == 120
    assert rule["tau_method"] == ex.TAU_METHOD and "method='linear'" in rule["tau_method"]
    assert rule["rule"] == ex.RULE42_TEXT and rule["refusal_id"] == 9788 and rule["comparator"] == ">="
    assert rule["tau_curve"]["0.3"] == rule["tau"] and len(rule["tau_curve"]) == 9
    # independent recomputation from the files
    by = {r["question_id"]: r for r in _rows(APASS40) if r["pop"] == "all"}
    R = json.loads(Path(ex.SELECTION40).read_text())["R"]
    vals = np.array([by[q]["s_proj"] for q in R], dtype=np.float64)
    assert float(np.percentile(vals, 70, method="linear")) == ex.TAU42
    assert all(by[q]["step0"]["argmax"] == 9788 for q in R)            # every §40 R row refuses at step 0
    assert r42.main(["--check"]) == 0


def test_check_rule42_refuses_any_change():
    rule = {"tau": ex.TAU42, "input_sha256": ex.APASS40_SHA256, "refusal_id": 9788, "comparator": ">=",
            "signal": "s_proj", "layer": 29, "alpha": 2.0, "steer_setting": "steer_all", "rule": ex.RULE42_TEXT}
    ex.check_rule42(rule)
    for k, v in (("tau", ex.TAU42 + 1e-12), ("refusal_id", 9789), ("comparator", ">"), ("input_sha256", "0" * 64)):
        with pytest.raises(ValueError):
            ex.check_rule42({**rule, k: v})


def test_tau_at_is_numpy_linear():
    xs = [3.0, -1.0, 2.0, 10.0, 0.5]
    assert ex.tau_at(xs, 70) == float(np.percentile(np.array(xs), 70, method="linear"))
    assert ex.tau_at(xs, 70) != float(np.percentile(np.array(xs), 70, method="lower"))


# --- the Cond firing logic -----------------------------------------------------------------------------------------

def test_cond42_fires_only_when_A_refuses_and_s_proj_at_least_tau():
    tau, rid = -18.9, 9788
    assert ex.cond42_fires(9788, -18.9, tau, rid)                       # equality fires (>=)
    assert ex.cond42_fires(9788, -5.0, tau, rid)
    assert not ex.cond42_fires(9788, -18.900001, tau, rid)              # below tau: A's output
    assert not ex.cond42_fires(1234, 100.0, tau, rid)                   # A answers: never fires, whatever s_proj
    with pytest.raises(ValueError):
        ex.cond42_fires(9788, None, tau, rid)
    with pytest.raises(ValueError):
        ex.cond42_fires(9788, float("nan"), tau, rid)
    row = {"step0": {"argmax": 9788}, "s_proj": -10.0}
    assert ex.cond42_of(row, tau, rid) == {"argmax": 9788, "a_refuses": True, "s_proj": -10.0, "tau": tau,
                                           "refusal_id": rid, "fires": True, "source": "steer_all"}
    assert ex.cond42_of({"step0": {"argmax": 5}, "s_proj": -10.0}, tau, rid)["source"] == "A"


# --- the fresh set -------------------------------------------------------------------------------------------------

def test_check_fresh_refuses_overlap_duplicates_and_wrong_pools():
    counts = {"pool_answerable": 923, "pool_adversarial": 406}
    ex.check_fresh(["a", "b"], ["c"], counts, ["x", "y"])
    with pytest.raises(ValueError):
        ex.check_fresh(["a", "x"], ["c"], counts, ["x", "y"])            # a §40 row
    with pytest.raises(ValueError):
        ex.check_fresh(["a", "b"], ["y"], counts, ["x", "y"])            # a §40 adversarial row
    with pytest.raises(ValueError):
        ex.check_fresh(["a", "a"], ["c"], counts, ["x"])
    with pytest.raises(ValueError):
        ex.check_fresh(["a"], ["c"], {"pool_answerable": 922, "pool_adversarial": 406}, ["x"])
    with pytest.raises(ValueError):
        ex.exclusion_ids({"rows": ["a"] * 600, "adversarial": ["b"] * 40})


def test_exclusion_happens_before_the_shuffle_on_a_synthetic_set(tmp_path):
    from tests.test_exp40 import _sample
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps([_sample()]))
    s = lv.build_set(str(p), n_rows=2, n_adv=1, seed=42, exclude_ids=["conv-x_q0"])
    expect = ["conv-x_q1", "conv-x_q6"]
    random.Random(42).shuffle(expect)
    assert [r.question_id for r in s.rows] == expect
    assert s.counts["pool_answerable"] == 2 and s.counts["eligible_answerable"] == 3 and s.counts["seed"] == 42
    assert "exclude_ids" in s.shas
    plain = lv.build_set(str(p), n_rows=3, n_adv=1)
    assert "pool_answerable" not in plain.counts and "exclude_ids" not in plain.shas
    with pytest.raises(ValueError):
        lv.build_set(str(p), n_rows=3, n_adv=1, exclude_ids=["conv-x_q0"])


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo not on disk")
def test_40s_set_is_unchanged():
    for s in (lv.build_set(), lv.build_set(seed=40)):
        assert {k: s.shas[k] for k in SHAS40} == SHAS40 and set(s.shas) == set(SHAS40) | {"locomo10.json"}
        assert "pool_answerable" not in s.counts and "seed" not in s.counts
    if VIEW40.exists():
        v = json.loads(VIEW40.read_text())
        s = lv.build_set()
        assert v["shas"] == s.shas and v["counts"] == s.counts
        assert v["rows"] == [r.question_id for r in s.rows] and v["adversarial"] == [r.question_id for r in s.adversarial]


@pytest.mark.skipif(not (LOCOMO.exists() and VIEW40.exists()), reason="LoCoMo or §40's view not on disk")
def test_fresh_set_is_disjoint_with_pools_923_406_and_pinned_shas():
    assert _sha(VIEW40) == ex.VIEW40_SHA256
    excluded = ex.exclusion_ids(json.loads(VIEW40.read_text()))
    assert len(excluded) == 640
    s = lv.build_set(seed=42, exclude_ids=excluded)
    rows, adv = [r.question_id for r in s.rows], [r.question_id for r in s.adversarial]
    info = ex.check_fresh(rows, adv, s.counts, excluded)
    assert info["pools"] == [923, 406] and len(rows) == 600 and len(adv) == 40
    assert not (set(rows) | set(adv)) & set(excluded)
    assert s.counts["excluded_from_answerable"] == 600 and s.counts["excluded_from_adversarial"] == 40
    assert s.shas == ex.VIEW42_SHAS
    assert all(r.category in (1, 2, 3, 4) for r in s.rows) and all(r.category == 5 for r in s.adversarial)


# --- the runner on the stub ----------------------------------------------------------------------------------------

def _run(out: Path, *extra) -> int:
    from scripts import exp40_pod_run as pr
    return pr.main(["run", "--exp", "exp42", "--out", str(out), "--backend", "stub", "--n-rows", str(N_ROWS),
                    "--n-adv", str(N_ADV), *extra])


@pytest.fixture(scope="module")
def stub42(tmp_path_factory):
    if not (LOCOMO.exists() and NPZ.exists() and AGREE.exists() and VIEW40.exists() and RULE42.exists()):
        pytest.skip("inputs not on disk")
    out = tmp_path_factory.mktemp("exp42")
    assert _run(out) == 0
    return out / "qwen3b"


@needs_inputs
def test_stub_run_logs_cond_per_row_and_C_rows_never_fire(stub42):
    m = json.loads((stub42 / "manifest.json").read_text())
    assert m["exp"] == "exp42" and m["seed"] == 42 and m["rule42"]["sha256"] == _sha(RULE42)
    assert m["rule42"]["tau"] == ex.TAU42 and m["fresh"]["pools"] == [923, 406]
    view = json.loads((stub42 / "view.json").read_text())
    v40 = json.loads(VIEW40.read_text())
    assert not (set(view["rows"]) | set(view["adversarial"])) & (set(v40["rows"]) | set(v40["adversarial"]))
    assert view["counts"]["seed"] == 42
    rid = m["rule42"]["refusal_id_used"]
    apass = _rows(stub42 / "apass.jsonl")
    assert apass and all("cond42" in r for r in apass)
    for r in apass:
        c = r["cond42"]
        assert c["argmax"] == r["step0"]["argmax"] and c["a_refuses"] == (c["argmax"] == rid)
        assert c["fires"] == (c["a_refuses"] and r["s_proj"] >= ex.TAU42)
        assert c["source"] == ("steer_all" if c["fires"] else "A")
    logged = _rows(stub42 / "cond42.jsonl")
    assert {x["question_id"] for x in logged} == {r["question_id"] for r in apass}
    sel_ = json.loads((stub42 / "selection.json").read_text())
    by = {r["question_id"]: r for r in apass if r["pop"] == "all"}
    assert sel_["C"] and not any(by[q]["cond42"]["fires"] for q in sel_["C"])     # C rows never fire
    assert all(by[q]["cond42"]["a_refuses"] for q in sel_["R"])                   # stub: R refuses at step 0
    # §40's arms unchanged (Random kept)
    keys = {(r["pop"], r["kind"], r["setting"]["key"]) for r in _rows(stub42 / "arms.jsonl")}
    L = m["layer"]
    assert ("R", "A", ex.setting_of("random", L).key) in keys and ("R", "A", ex.setting_of("steer0", L).key) in keys
    assert not any(r.get("cond42") for r in _rows(stub42 / "arms.jsonl"))       # logged on A-pass rows only


@needs_inputs
def test_stub_run_resumes_without_new_rows(stub42):
    before = {p.name: len(p.read_text().splitlines()) for p in stub42.glob("*.jsonl")}
    assert _run(stub42.parent) == 0
    assert {p.name: len(p.read_text().splitlines()) for p in stub42.glob("*.jsonl")} == before


@needs_inputs
def test_runner_refuses_a_wrong_rule_or_exclusion_list(tmp_path):
    from scripts import exp40_pod_run as pr
    bad = tmp_path / "rule42.json"
    bad.write_text(json.dumps({**json.loads(RULE42.read_text()), "tau": -20.0}))
    # the pod (hf) accepts only the pinned rule: refused before any model is loaded
    assert pr.main(["run", "--exp", "exp42", "--out", str(tmp_path / "a"), "--backend", "hf",
                    "--rule42", str(bad)]) == ex.EXIT_INPUT
    assert not (tmp_path / "a" / "qwen3b").exists()
    assert _run(tmp_path / "b", "--rule42", str(tmp_path / "missing.json")) == ex.EXIT_INPUT
    v = tmp_path / "view.json"
    v.write_text(VIEW40.read_text() + " ")
    assert _run(tmp_path / "c", "--exclude-view", str(v)) == ex.EXIT_INPUT


# --- the report ----------------------------------------------------------------------------------------------------

def _forge42(d: Path, steer_yes: set, b_yes: set, tau: float) -> dict:
    """R: A refuses (unchanged), Steer-all answers on every R row (gold on `steer_yes`), B answers (gold on `b_yes`);
    the rule's tau and curve set to `tau`, the manifest's rule sha patched. Returns the rule."""
    m = json.loads((d / "manifest.json").read_text())
    L = m["layer"]
    k = {n: ex.setting_of(n, L).key for n in ("steer_all", "none")}
    verdict = {}
    arms = _rows(d / "arms.jsonl")
    for r in arms:
        q, pop, kind, key = r["question_id"], r["pop"], r["kind"], r["setting"]["key"]
        if pop == "R" and kind == "A" and key == k["steer_all"]:
            yes = q in steer_yes
        elif pop == "R" and kind == "B":
            yes = q in b_yes
        else:
            continue
        r["cls"], r["phrase"], r["final_answer"] = "answer", f"ans {q}", f"ans {q}"
        verdict[(pop, kind, key, q)] = yes
    _write(d / "arms.jsonl", arms)
    jr = _rows(d / "judge.jsonl")
    for j in jr:
        key = (j["pop"], j["kind"], j["setting_key"], j["question_id"])
        if key in verdict:
            j["reply"] = "yes" if verdict[key] else "no"
    _write(d / "judge.jsonl", jr)
    rule = json.loads(RULE42.read_text())
    rule["tau"] = tau
    rule["tau_curve"] = {**rule["tau_curve"], "0.3": tau}
    rp = d / "rule42.json"
    rp.write_text(json.dumps(rule))
    m["rule42"]["sha256"] = _sha(rp)
    (d / "manifest.json").write_text(json.dumps(m))
    return rule


def _report(d: Path, rule: Path, samples: int = 300):
    from scripts import exp40_report as rp
    return rp.build(d, True, samples=samples, rule=Path("runs/exp41_dev/qwen3b/rule.json"), exp="exp42", rule42=rule)


@needs_inputs
def test_report_claims_arithmetic_on_forged_stub_rows(stub42, tmp_path):
    d = Path(shutil.copytree(stub42, tmp_path / "qwen3b"))
    sel_ = json.loads((d / "selection.json").read_text())
    R, C = sel_["R"], sel_["C"]
    n = len(R)
    a = {r["question_id"]: r for r in _rows(d / "apass.jsonl") if r["pop"] == "all"}
    sp = sorted(a[q]["s_proj"] for q in R)
    tau = sp[len(sp) // 2]                                              # about half the R rows fire
    fired = {q for q in R if a[q]["s_proj"] >= tau}
    steer_yes, b_yes = set(R[::2]), set(R[:3])
    _forge42(d, steer_yes, b_yes, tau)
    text, num = _report(d, d / "rule42.json")
    s = num["s42"]
    assert s["n"] == n and s["fires"] == len(fired) and 0 < len(fired) < n
    for lam in (1, 2):
        b = s[f"lambda{lam}"]
        u_cond = sum(1 if q in steer_yes else -lam for q in fired)
        u_b = sum(1 if q in b_yes else -lam for q in R)
        u_s = sum(1 if q in steer_yes else -lam for q in R)
        assert b["U"] == {"A": 0, "B": u_b, "Steer": u_s, "Cond": u_cond}
        assert abs(b["claim1_cond_minus_B"]["mean"] - (u_cond - u_b) / n) < 1e-12
        assert abs(b["claim2_cond_minus_A"]["mean"] - u_cond / n) < 1e-12
        assert abs(b["claim3_cond_minus_steer"]["mean"] - (u_cond - u_s) / n) < 1e-12
        assert abs(b["steer_minus_A"]["mean"] - u_s / n) < 1e-12
        assert b["claim3_reading"] == ("expected under a net-negative steer; not evidence of a selective signal"
                                       if u_s <= 0 else "steer is net positive here")
        q3 = next(c for c in b["frozen_q_curve"] if c["verdict"])
        assert q3["q"] == 0.3 and q3["U"] == u_cond and q3["fires"] == len(fired)
        assert sum(c["verdict"] for c in b["frozen_q_curve"]) == 1 and len(b["frozen_q_curve"]) == 9
    gs = {q: q in steer_yes for q in R}
    assert s["claim4"]["auroc"] == cd.auroc([a[q]["s_proj"] for q in R], [gs[q] for q in R])
    assert s["fire_precision"]["fired"] == len(fired)
    assert s["fire_precision"]["gold_rate_fired"] == sum(q in steer_yes for q in fired) / len(fired)
    assert s["controlD"]["fires"] == 0 and s["controlD"]["mean"] == 0 and s["controlD"]["rows"] == len(C)
    assert not s["controlD"]["harmful"]
    cr = s["coverage_risk"]["Cond"]
    assert cr["answered"] == len(fired) and cr["wrong"] == len(fired - steer_yes)
    assert sum(b["n"] for b in s["by_type"].values()) == n
    # Control U: Cond on the marked adversarial rows = Steer-all where it fires, else A
    adv = sel_["adversarial_marked"]
    arows = {r["question_id"]: r for r in _rows(d / "apass.jsonl") if r["pop"] == "adv"}
    L = json.loads((d / "manifest.json").read_text())["layer"]
    srows = {r["question_id"]: r for r in _rows(d / "arms.jsonl")
             if r["pop"] == "adv" and r["kind"] == "A" and r["setting"]["key"] == ex.setting_of("steer_all", L).key}
    rid = s["refusal_id"]
    want = [srows[q] if ex.cond42_fires(arows[q]["step0"]["argmax"], arows[q]["s_proj"], tau, rid) else arows[q]
            for q in adv]
    assert s["controlU"]["Cond"]["n"] == len(adv)
    assert s["controlU"]["Cond"]["ans"] == sum(r["cls"] == "answer" for r in want) / len(adv)
    assert "§42: the frozen selective policy" in text and "only q = 0.3 carries a verdict" in text
    assert repr(tau) in text and "method='linear'" in text and "In distribution" in text
    if n < ex.R_MIN_FOR_CLAIMS:
        assert "descriptive" in s["lambda1"]["claim1_cond_minus_B"]["verdict"]


@needs_inputs
def test_report_end_to_end_with_exp42(stub42, tmp_path):
    from scripts import exp40_report as rp
    out = tmp_path / "r42.md"
    assert rp.main(["--exp", "exp42", "--run-dir", str(stub42), "--confirm", "--samples", "200",
                    "--out", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# §42: selective steering by s_proj on fresh LoCoMo rows")
    assert "(seed 42)" in text and "923 answerable / 406 adversarial" in text
    for s in ("**Claim 1**", "**Claim 2**", "**Claim 3**", "**Claim 4**", "**Control D**", "**Control U**",
              "Precision of fires", "Coverage / risk", "§40's readouts on the fresh rows"):
        assert s in text, s
    num = json.loads(out.with_suffix(".json").read_text())
    assert num["s42"]["tau"] == ex.TAU42 and num["s42"]["controlD"]["fires"] == 0
    assert num["s42"]["logged_mismatch"] == []
    assert rp.main(["--exp", "exp42", "--run-dir", str(stub42), "--out", str(tmp_path / "x.md")]) == 2   # no --confirm
    assert rp.main(["--exp", "exp40", "--run-dir", str(stub42), "--confirm", "--out", str(tmp_path / "y.md")]) == 2
    other = tmp_path / "other.json"
    other.write_text(RULE42.read_text() + " ")
    assert rp.main(["--exp", "exp42", "--run-dir", str(stub42), "--confirm", "--rule42", str(other),
                    "--out", str(tmp_path / "z.md")]) == 2                  # not the rule the run recorded


# --- the bundle ----------------------------------------------------------------------------------------------------

@needs_inputs
def test_exp42_bundle_adds_only_rule42_and_40s_view(tmp_path):
    out = tmp_path / "b.tar.gz"
    subprocess.run(["bash", "scripts/exp40_bundle.sh", str(out)], check=True, capture_output=True,
                   env={**__import__("os").environ, "EXP": "exp42"})
    with tarfile.open(out) as t:
        names = set(t.getnames())
        info = json.loads(t.extractfile("bundle_info.json").read())
    runs = {n for n in names if n.startswith("runs/")}
    assert {"runs/exp42_inputs/rule42.json", "runs/exp40_pod/qwen3b/view.json"} <= runs
    assert runs <= {"runs/exp38_dev/qwen3b/directions.npz", "runs/exp40_inputs/agreement60.jsonl",
                    "runs/exp41_dev/qwen3b/rule.json", "runs/exp42_inputs/rule42.json", "runs/exp40_pod/qwen3b/view.json"}
    assert "scripts/exp42_rule.py" in names and not any(n.startswith("tests/") or n.endswith(".env") for n in names)
    assert info["exp"] == "exp42" and info["rule42_sha256"] == _sha(RULE42) and info["view40_sha256"] == ex.VIEW40_SHA256


def test_pod_script_passes_the_exp_tag():
    sh = Path("scripts/exp40_pod.sh").read_text()
    assert 'EXP="${EXP:-exp40}"' in sh and '--exp "$EXP" --rule42 "$RULE42"' in sh
    assert 'runs/${EXP}_pod' in sh and 'INPUTS+=("$RULE42" runs/exp40_pod/qwen3b/view.json)' in sh
    d30 = Path("scripts/exp40_d30.sh").read_text()
    assert 'EXP="${EXP:-exp40}"' in d30 and "cond42.jsonl" in d30
