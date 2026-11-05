"""§38 pod stage: the pod runner end to end on the stub engine and stub judge (real §35 cache, no Qwen weights), the
judge rows read back by the report, the D-22 smoke abort, per-role engine refusal, the judge-agreement caveat, the
secondary-only wording and the deadline skip of secondary arms (scripts/exp38_pod_run.py, scripts/exp38_report.py)."""
import json
import shutil
import time
from pathlib import Path

import pytest

from memctl.analysis import steering as st

CACHE = Path("runs/exp35/qwen3b/A/fold0.jsonl")
DESIGN = Path("runs/exp38_dev/qwen3b/gate.json")
pytestmark = pytest.mark.skipif(not (CACHE.exists() and DESIGN.exists()), reason="§35 cache or §38 design not on disk")


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def pod(tmp_path_factory):
    from scripts import exp38_pod_run as pr
    out = tmp_path_factory.mktemp("exp38pod")
    common = ["--out", str(out), "--backend", "stub"]
    assert pr.main(["smoke", *common]) == 0
    assert pr.main(["run", *common, "--deadline", str(time.time() + 3600)]) == 0
    assert pr.main(["judge", *common]) == 0
    return out / "qwen3b"


def _report(run_dir: Path, samples: int = 200):
    from scripts import exp38_report
    return exp38_report.build(run_dir, True, samples=samples, design=run_dir / "design_stub.json")


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# --- the runner ---------------------------------------------------------------------------------------------------

def test_pod_run_writes_every_stage_on_all_folds(pod):
    repro = _rows(pod / "repro.jsonl")
    assert {r["fold"] for r in repro} == {0, 1, 2, 3, 4}
    assert sum(r["pop"] == "R" for r in repro) == 2 * 92 and sum(r["pop"] == "C" for r in repro) == 94
    info = json.loads((pod / "settings.json").read_text())
    p, s = info["primary"], info["secondary"]
    assert p == "steer:L14:a4:vnorm" and s == "steer:L16:a2:vnorm"      # the stub's 20 layers: L25 -> 14, L29 -> 16
    arms = _rows(pod / "arms.jsonl")
    assert {r["fold"] for r in arms} == {2, 3, 4}                       # arms on confirmation rows only
    keys = {(r["pop"], r["kind"], r["setting"]["key"]) for r in arms}
    for k in (p, s):
        for mode in ("steer", "random", "reverse"):
            assert ("R", "A", k.replace("steer:", f"{mode}:", 1)) in keys
        assert ("R", "cut", k) in keys and ("C", "A", k) in keys
    assert ("R", "cut", "none") in keys and ("C", "A", p.replace("steer:", "reverse:", 1)) in keys
    for r in arms + repro:
        assert r["max_new_tokens"] == st.MAX_NEW_TOKENS_POD and r["wording_sha256"] == st.wording_sha256(64)
        assert r["last_token"] == ":" and r["engine"] == "stub"
    d = json.loads((pod / "directions.json").read_text())
    assert d["pairs"] and len(d["pairs"]) == 13 and set(d["layers"]) == {"14", "16"}
    assert json.loads((pod / "sanity.json").read_text())["pass"]
    plan = json.loads((pod / "plan.json").read_text())
    assert not plan["skipped"] and not plan["cut"] and len(plan["arms"]) == 8
    smoke = json.loads((pod.parent / "smoke" / "smoke.json").read_text())
    assert not smoke["problems"] and all(v is not None for v in smoke["verdicts"]) and len(smoke["replies"]) == 11


def test_pod_run_resumes_without_new_rows(pod):
    from scripts import exp38_pod_run as pr
    before = {p.name: len(p.read_text().splitlines()) for p in pod.glob("*.jsonl")}
    assert pr.main(["run", "--out", str(pod.parent), "--backend", "stub"]) == 0
    assert {p.name: len(p.read_text().splitlines()) for p in pod.glob("*.jsonl")} == before


def test_judge_rows_round_trip_through_the_report(pod):
    from scripts import exp38_report
    jrows = _rows(pod / "judge.jsonl")
    need = {"pop", "kind", "setting_key", "question_id", "verdict", "reply", "parseable", "judge_engine", "judge_model"}
    assert jrows and all(need <= set(r) for r in jrows)
    assert {r["judge_engine"] for r in jrows} == {"stub-judge-d1"}
    assert json.loads((pod / "judge_manifest.json").read_text())["judge_engine"] == "stub-judge-d1"
    gold = exp38_report.Gold(pod)
    out_rows = [r for f in ("repro.jsonl", "arms.jsonl") for r in _rows(pod / f) if r["pop"] in ("R", "C")]
    for r in out_rows:          # every output of the run has this judge's verdict; gold = verdict and an answer
        v = gold(r)
        assert v is not None and v == (gold.judge[(r["pop"], r["kind"], r["setting"]["key"], r["question_id"])]
                                       and r["cls"] == "answer")
    assert not gold.missing and sum(r["pop"] == "cached35" for r in jrows) == 2 * 64 + 55
    text, n = _report(pod)
    assert n["gold_source"].startswith("judge") and "D1 string rule" not in n["gold_source"]
    assert n["claim1"]["n"] == n["populations"]["R_reproduced"] and n["secondary"]["claim1"]["n"] > 0
    assert n["b_gold"]["cached_35"] == {"k": 27, "n": 64}
    assert "27/64" in text and "Secondary setting" in text and st.SECONDARY_LABEL in text
    assert n["c_rule"].startswith("A answers and the new judge")
    assert "by_stage" in n["costs"] and "arms" in n["costs"]["by_stage"] and n["costs"]["judge"]["n_judged"] == len(jrows)


def test_c_rows_reproduce_by_the_new_judge_not_d1(pod, tmp_path):
    run = _copy(pod, tmp_path / "c")
    jrows = _rows(run / "judge.jsonl")
    c_none = [r for r in jrows if r["pop"] == "C" and r["setting_key"] == "none" and r["fold"] in (2, 3, 4)]
    flipped = c_none[0]["question_id"]
    for r in jrows:
        if r["pop"] == "C" and r["setting_key"] == "none" and r["question_id"] == flipped:
            r["verdict"] = not r["verdict"]
    _write(run / "judge.jsonl", jrows)
    _, before = _report(pod)
    _, after = _report(run)
    assert (flipped in before["repro"]["damage_ok"]) != (flipped in after["repro"]["damage_ok"])


def test_smoke_aborts_when_the_judge_gives_no_verdict(tmp_path):
    from scripts import exp38_pod_run as pr
    assert pr.main(["smoke", "--out", str(tmp_path), "--backend", "stub", "--stub-judge", "silent"]) == pr.EXIT_SMOKE
    smoke = json.loads((tmp_path / "smoke" / "smoke.json").read_text())
    assert smoke["problems"] and all(v is None for v in smoke["verdicts"])
    assert not (tmp_path / "qwen3b").exists()            # nothing of the full run was written


def test_population_mismatch_exits(tmp_path, monkeypatch):
    from scripts import exp38_pod_run as pr
    monkeypatch.setattr(pr, "EXPECTED", dict(pr.EXPECTED, R_confirm=63))
    assert pr.main(["smoke", "--out", str(tmp_path), "--backend", "stub"]) == pr.EXIT_POP


# --- engines per role ---------------------------------------------------------------------------------------------

def test_report_refuses_mixed_engines_per_role(pod, tmp_path):
    from scripts import exp38_report
    _report(pod)                      # steering 'stub' and judge 'stub-judge-d1' differ: allowed (separate roles)
    run = _copy(pod, tmp_path / "judge_mixed")
    jrows = _rows(run / "judge.jsonl")
    jrows[0]["judge_engine"] = "hf-transformers-5.17.0-cuda-bfloat16"
    _write(run / "judge.jsonl", jrows)
    with pytest.raises(exp38_report.Refused, match="mixes judge engines"):
        _report(run)
    run = _copy(pod, tmp_path / "steer_mixed")
    rows = _rows(run / "arms.jsonl")
    rows[0]["engine"] = "hf-transformers-5.17.0-cuda-bfloat16"
    _write(run / "arms.jsonl", rows)
    with pytest.raises(exp38_report.Refused, match="mixes steering engines"):
        _report(run)


def test_judge_runner_refuses_another_judge_engine(pod, tmp_path):
    from scripts import exp38_pod_run as pr
    run = _copy(pod.parent, tmp_path / "j")
    m = json.loads((run / "qwen3b" / "judge_manifest.json").read_text())
    m["judge_engine"] = "hf-other"
    (run / "qwen3b" / "judge_manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="REFUSED"):
        pr.main(["judge", "--out", str(run), "--backend", "stub"])


def test_parse_verdict():
    assert st.parse_verdict(" Yes.") is True and st.parse_verdict("no") is False
    assert st.parse_verdict("") is None and st.parse_verdict("The answer is 4") is None and st.parse_verdict(None) is None


# --- the agreement caveat and the secondary wording ---------------------------------------------------------------

def test_agreement_bar_caveat(pod, tmp_path):
    from scripts.exp38_dev import load_source
    text, n = _report(pod)
    ja = n["judge_agreement"]
    n_c_answers = sum(1 for r in _rows(pod / "repro.jsonl") if r["pop"] == "C" and r["fold"] in (2, 3, 4) and not r["refusal"])
    assert ja["bar"] == 0.90 and ja["reproduced_rows"]["n"] == 2 * n["populations"]["R_reproduced"] + n_c_answers
    if ja["caveat"]:
        assert f"Caveat: {st.JUDGE_CAVEAT}." in text
    # force agreement: the new judge's verdicts on the reproduced A/B rows set to the cached §35 ones
    d, _ = load_source([0, 1, 2, 3, 4], Path("runs/exp35"), "runs", Path("runs/exp34_jlens_own"))
    run = _copy(pod, tmp_path / "agree")
    jrows = _rows(run / "judge.jsonl")
    for r in jrows:
        if r["setting_key"] == "none" and r["pop"] in ("R", "C") and r["kind"] in ("A", "B"):
            r["verdict"] = bool(d.correct(r["kind"], r["question_id"]))
    _write(run / "judge.jsonl", jrows)
    text2, n2 = _report(run)
    assert n2["judge_agreement"]["reproduced_rows"]["p"] == 1.0 and not n2["judge_agreement"]["caveat"]
    assert st.JUDGE_CAVEAT not in text2.split("## Judge engine")[0] and f"Caveat: {st.JUDGE_CAVEAT}" not in text2
    # force disagreement: the R rows' A and B verdicts flipped (C rows kept, so they still reproduce) -> below the
    # bar, the caveat beside Claim 1; it does not void
    for r in jrows:
        if r["setting_key"] == "none" and r["pop"] == "R" and r["kind"] in ("A", "B"):
            r["verdict"] = not bool(d.correct(r["kind"], r["question_id"]))
    _write(run / "judge.jsonl", jrows)
    text3, n3 = _report(run)
    assert n3["judge_agreement"]["caveat"] and f"Caveat: {st.JUDGE_CAVEAT}." in text3
    assert n3["claim1"]["verdict"] in ("SHOWN", "NOT SHOWN")


def test_secondary_only_wording(pod, tmp_path):
    run = _copy(pod, tmp_path / "sec")
    info = json.loads((run / "settings.json").read_text())
    p, s = info["primary"], info["secondary"]
    arms = _rows(run / "arms.jsonl")
    for r in arms:          # the secondary Steer answers on every R row; Steer-cut, Random and the primary do not
        if r["pop"] == "R" and r["kind"] == "A" and r["setting"]["key"] == s:
            r["cls"], r["ans"] = "answer", True
    _write(run / "arms.jsonl", arms)
    jrows = _rows(run / "judge.jsonl")
    for r in jrows:
        if r["pop"] == "R" and r["kind"] == "A" and r["setting_key"] == s:
            r["verdict"] = True
        if r["pop"] == "R" and r["setting_key"] in (p, s) and r["kind"] == "cut":
            r["verdict"] = False
    _write(run / "judge.jsonl", jrows)
    text, n = _report(run)
    assert n["claim1"]["verdict"] != "SHOWN" and n["secondary"]["claim1"]["verdict"] == "SHOWN"
    assert n["record"] == st.SECONDARY_ONLY and st.SECONDARY_ONLY in text
    _, n0 = _report(pod)
    assert "record" not in n0


# --- the deadline -------------------------------------------------------------------------------------------------

def _primary_like_keys(run) -> list[tuple]:
    p = run.primary
    mk = lambda mode: st.Setting(mode, p.layer, p.alpha, p.unit).key  # noqa: E731
    return [("R", "A", p.key), ("R", "cut", p.key), ("R", "A", mk("random")), ("C", "A", p.key), ("R", "A", mk("reverse"))]


def test_deadline_skips_secondary_arms(tmp_path, monkeypatch):
    from scripts import exp38_pod_run as pr
    # the primary arms "measured" at 1e6 s each: every secondary arm's estimate passes the deadline -> skipped, logged
    monkeypatch.setattr(pr.PodRun, "measured_arm_seconds", lambda self: {k: 1e6 for k in _primary_like_keys(self)})
    assert pr.main(["run", "--out", str(tmp_path), "--backend", "stub", "--deadline", str(time.time() + 3600),
                    "--judge-reserve", "60"]) == 0
    run = tmp_path / "qwen3b"
    plan = json.loads((run / "plan.json").read_text())
    assert [x["arm"] for x in plan["skipped"]] == ["secondary:R:A:steer", "secondary:R:cut:steer", "secondary:R:A:random",
                                                   "secondary:C:A:steer", "secondary:R:A:reverse"]
    assert set(plan["arms"]) == {"primary:R:A", "primary:R:cut", "primary:C:A"} and not plan["cut"]
    s = json.loads((run / "settings.json").read_text())["secondary"]
    assert not any(r["setting"]["layer"] == st.setting_from_key(s).layer for r in _rows(run / "arms.jsonl"))
    assert pr.main(["judge", "--out", str(tmp_path), "--backend", "stub"]) == 0
    text, n = _report(run)
    assert n["secondary"].get("missing") and "No rows at this setting" in text and "secondary:R:A:steer" in text
    assert "claim1" in n


def test_primary_is_cut_at_the_deadline_and_logged(tmp_path):
    from scripts import exp38_pod_run as pr
    # the generation deadline (deadline - reserve) already passed: reproduction etc. run, every arm is cut at row 0
    assert pr.main(["run", "--out", str(tmp_path), "--backend", "stub", "--deadline", str(time.time() + 10),
                    "--judge-reserve", "3600"]) == 0
    plan = json.loads((tmp_path / "qwen3b" / "plan.json").read_text())
    assert plan["cut"] and plan["cut"][0]["arm"] == "primary:R:A" and plan["cut"][0]["done"] == 0
