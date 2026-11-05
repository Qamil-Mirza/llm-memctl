"""§43 stage 1, Llama 3.2 3B design script (scripts/exp43_llama_dev.py) and its report: populations on the real §24
cache, the rev 1 A2 gate arithmetic and choice rule, and a stub run end to end (resume, stamps, refusals). StubEngine
only; no Llama weights are loaded."""
import json
from pathlib import Path

import pytest

from memctl.analysis import steering as st
from scripts import exp43_llama_dev as dev

needs_exp24 = pytest.mark.skipif(not Path("runs/exp24_llama3b_f0").exists(), reason="§24 runs not on disk")


def test_refuses_confirmation_folds(capsys):
    assert dev.main(["--folds", "0,2"]) == 2
    assert "REFUSED" in capsys.readouterr().out


@needs_exp24
def test_populations_on_the_real_cache():
    pops = dev.populations(dev.load_records([0, 1]))
    assert len(pops.G) == 13 and len(pops.P0) == 30 and not set(pops.G) & set(pops.P0)
    assert len(pops.excluded_unanswerable) == 4 and all(q.endswith("_abs") for q in pops.excluded_unanswerable)
    for q in pops.G:
        r = pops.by_q[q]
        assert r.all_in_view and r.answerable and r.evidence_lines and r.correct is not None and dev.cached_refusal(r)
    for q in pops.P0:
        r = pops.by_q[q]
        assert not r.all_in_view and r.answerable and dev.is_marked(r) and dev.cached_refusal(r)
    assert {pops.by_q[q].fold for q in pops.G + pops.P0} == {0, 1}


def c(ans, degen=0, gold=0, n=13):
    return {"n": n, "ans": ans, "degenerate": degen, "refusal": n - ans - degen, "gold": gold}


def test_setting_gate_arithmetic():
    ok = dev.setting_gate(c(7, 1, 3), c(5, 0, 1), c(1))
    assert ok["pass"] is True and ok["gold_diff"] == 2 and ok["random_bound"] == pytest.approx(1.4)
    assert dev.setting_gate(c(6, 0, 3), c(5, 0, 1), c(0))["pass"] is False             # Steer answers < 7
    assert dev.setting_gate(c(7, 2, 3), c(5, 0, 1), c(0))["pass"] is False             # degenerate > 1
    assert dev.setting_gate(c(7, 0, 2), c(5, 0, 1), c(0))["pass"] is False             # gold diff 1 < 2
    assert dev.setting_gate(c(10, 0, 4), c(5, 0, 1), c(3))["pass"] is False            # Random 3 > 0.2 x 10
    assert dev.setting_gate(c(10, 0, 4), c(5, 0, 1), c(2))["pass"] is True             # Random 2 <= 2.0
    pending = dev.setting_gate(c(8, 0, 4), c(5, 0, 1), None)
    assert pending["pass"] is None and pending["random_ok"] is None
    assert dev.setting_gate(c(8, 0, 4), None, None)["pass"] is None
    assert dev.setting_gate(c(3), None, None)["pass"] is False
    assert dev.needs_cut(c(7, 1)) and not dev.needs_cut(c(6)) and not dev.needs_cut(c(9, 2))
    assert dev.ans_min(13) == 7 and dev.ans_min(12) == 6 and dev.ans_min(11) == 6 and dev.MIN_PAIRS == 5


def test_choice_rule_is_deeper_layer_then_smaller_alpha():
    settings = st.sweep_settings([17, 20, 22, 25], [1.0, 2.0, 4.0])
    order = [s.key for s in dev.gate_order(settings)]
    assert order[:4] == ["steer:L25:a1:vnorm", "steer:L25:a2:vnorm", "steer:L25:a4:vnorm", "steer:L22:a1:vnorm"]
    assert order[-1] == "steer:L17:a4:vnorm"
    alt = dev.alt_order(settings, {"steer:L17:a4:vnorm": 12, "steer:L20:a2:vnorm": 12, "steer:L25:a1:vnorm": 9})
    assert [s.key for s in alt[:3]] == ["steer:L20:a2:vnorm", "steer:L17:a4:vnorm", "steer:L25:a1:vnorm"]
    assert dev.FAMILY.sweep_layers == (17, 20, 22, 25)


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    """One stub design run on the real cache with the gate relaxed so every stage runs (the stub's byte tokenizer
    needs more than 256 'tokens' to reproduce a cached §24 output)."""
    if not Path("runs/exp24_llama3b_f0").exists():
        pytest.skip("§24 runs not on disk")
    out = tmp_path_factory.mktemp("exp43")
    with pytest.MonkeyPatch.context() as mp:
        for name, value in (("ANS_SHARE", 0), ("GOLD_DIFF_MIN", -100), ("DEGEN_MAX", 100), ("MAX_NEW_FULL", 8192)):
            mp.setattr(dev, name, value)
        assert dev.main(["--run", "--backend", "stub", "--out", str(out)]) == 0
    return out


def _lines(run_dir: Path) -> dict:
    return {p.name: len(p.read_text().splitlines()) for p in sorted(run_dir.glob("*.jsonl"))}


def test_stub_run_writes_every_stage_sha_stamped(stub_run):
    d = stub_run / "llama3b"
    lines = _lines(d)
    assert lines["repro.jsonl"] == 13 + 2 * 30 and lines["ncut.jsonl"] == 13 and lines["ncheck.jsonl"] == 5
    assert lines["sweep.jsonl"] == 13 * 12 and lines["random.jsonl"] == 13
    assert lines["cut.jsonl"] == 13 * 13                    # A-cut + Steer-cut at all 12 (gate relaxed)
    manifest = json.loads((d / "manifest.json").read_text())
    for p in d.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            assert row["wording_sha256"] == manifest["wording_sha256"] and row["family"] == "llama3b"
            assert row["fold"] in (0, 1) and len(row["input_sha256"]) == 64
    sweep = [json.loads(x) for x in (d / "sweep.jsonl").read_text().splitlines()]
    assert all(r["vector_sha256"] for r in sweep) and {r["pop"] for r in sweep} == {"G"}
    gate = json.loads((d / "gate.json").read_text())
    assert gate["chosen"] == "steer:L18:a1:vnorm" and not set(gate["pairs"]) & set(gate["G_ok"])
    assert manifest["family"]["template_kwargs"] == {"date_string": "09 Oct 2026"}


def test_stub_run_resumes_without_new_rows(stub_run):
    before = _lines(stub_run / "llama3b")
    with pytest.MonkeyPatch.context() as mp:
        for name, value in (("ANS_SHARE", 0), ("GOLD_DIFF_MIN", -100), ("DEGEN_MAX", 100), ("MAX_NEW_FULL", 8192)):
            mp.setattr(dev, name, value)
        assert dev.main(["--run", "--backend", "stub", "--out", str(stub_run)]) == 0
    assert _lines(stub_run / "llama3b") == before


def test_report_recomputes_the_gate_and_agrees(stub_run, monkeypatch, capsys):
    from scripts import exp43_llama_report as rep
    for name, value in (("ANS_SHARE", 0), ("GOLD_DIFF_MIN", -100), ("DEGEN_MAX", 100)):
        monkeypatch.setattr(dev, name, value)
    assert rep.main(["--run-dir", str(stub_run / "llama3b")]) == 0
    out = json.loads((stub_run / "llama3b" / "report.json").read_text())
    assert out["gate_json_agrees"] is True and out["chosen"] == "steer:L18:a1:vnorm" and len(out["table"]) == 12
    assert "| setting | L | alpha |" in capsys.readouterr().out


def test_stale_manifest_is_refused(stub_run, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(stub_run, d)
    m = json.loads((d / "llama3b" / "manifest.json").read_text())
    m["layers"] = [1, 2]
    (d / "llama3b" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="REFUSED"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dev, "MAX_NEW_FULL", 8192)
            dev.main(["--run", "--backend", "stub", "--out", str(d)])
