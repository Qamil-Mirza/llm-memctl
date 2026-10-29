"""§44 patching: recovery arithmetic, sanity bounds, filter, alignment by role, resume, fold guard (tiny model)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.analysis import patching as pt
from memctl.jacobian_data import PromptRecord

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nassistant, d1: turn b\n"
          "user, d2: I was born in Leeds.\nuser, d3: turn c\n\n## Current input\nQuestion: (asked on 2023/05/30) "
          "Where was I born?\n\nNote:")
LINES = ("user, d1: turn a", "assistant, d1: turn b", "user, d2: I was born in Leeds.", "user, d3: turn c")


def record(qid="q1", fold=0, gold="Leeds", evidence=(LINES[2],)):
    return PromptRecord(qid, fold, "qwen3b", PROMPT, "", gold, "single-session-user", True, LINES, evidence, True,
                        True, False)


@pytest.fixture(scope="module")
def engine():
    from memctl.analysis.steering import HFEngine

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    eng = HFEngine.__new__(HFEngine)
    eng.torch, eng.model, eng.family, eng.device = torch, "tiny", None, "cpu"
    eng.name, eng.tokenizer = "tiny-qwen2", tok
    eng.net = transformers.Qwen2ForCausalLM(config).eval()
    eng.layers = eng.net.model.layers
    eng.n_layers, eng.hidden = len(eng.layers), config.hidden_size
    eng.stop_ids = {tok.eos_token_id}
    return eng


# --- arithmetic -------------------------------------------------------------------------------------------------

def test_recovery_and_depth():
    r = pt.recovery([-10.0, -9.0, -6.0, -4.0], m_clean=-2.0, m_corrupt=-10.0)
    assert r == [0.0, 0.125, 0.5, 0.75]
    assert pt.causal_depth(r, 4) == 2 / 4                      # first l with R >= 0.5 (inclusive)
    assert pt.causal_depth([0.1, 0.49], 2) is None
    assert pt.recovery([1.0], -3.0, -3.0) is None and pt.causal_depth(None, 4) is None
    assert pt.recovery([-12.0], -2.0, -10.0) == [-0.25]          # unclipped for the statistics
    assert pt.clip_r(-3.0) == -0.25 and pt.clip_r(2.0) == 1.25 and pt.clip_r(0.3) == 0.3


def test_sanity_bounds():
    assert pt.sanity([0.049, 0.5, 1.0099])["pass"] is True
    assert pt.sanity([0.05, 0.5, 1.0])["zero_ok"] is False
    assert pt.sanity([0.0, 0.5, 0.99])["final_ok"] is False
    assert pt.sanity(None)["pass"] is None
    row = {"question_id": "q", "patched": True, "eligible": True, "sanity": pt.sanity([0.2, 1.0])}
    with pytest.raises(pt.SanityError):
        pt.check_sanity(row)
    pt.check_sanity({**row, "eligible": False})                  # --patch-all outside the filter: recorded only


def test_filter_and_relaxation():
    assert pt.evidence_dependent(10, 11) and not pt.evidence_dependent(11, 50) and not pt.evidence_dependent(1, 10)
    assert pt.patch_eligible(1, 4) and not pt.patch_eligible(1, 3)
    rows = [{"rank_clean": 1, "rank_corrupt": 50}] * 39 + [{"rank_clean": 1, "rank_corrupt": 5}] * 10
    f = pt.choose_filter(rows)
    assert f["n_strict"] == 39 and f["relaxed"] and f["k_corrupt"] == 3 and f["n_relaxed"] == 49
    f2 = pt.choose_filter(rows + [{"rank_clean": 2, "rank_corrupt": 11}])
    assert f2["n_strict"] == 40 and not f2["relaxed"] and f2["k_corrupt"] == 10


def test_readings():
    assert pt.c1_reading({"value": 0.5, "ci_low": 0.4, "ci_high": 0.64}) == "J-lens right"
    assert pt.c1_reading({"value": 0.8, "ci_low": 0.71, "ci_high": 0.9}) == "logit lens right"
    assert pt.c1_reading({"value": 0.7, "ci_low": 0.6, "ci_high": 0.8}) == "both partly right"
    assert pt.c2_reading({"value": 0.6, "ci_low": 0.51, "ci_high": 0.7}) == "carries most of the answer"
    assert pt.c2_reading({"value": 0.1, "ci_low": 0.0, "ci_high": 0.24}) == "carries little"
    assert pt.c3_reading({"value": 0.6, "ci_low": 0.5, "ci_high": 0.74}).startswith("lenses under-read")
    assert pt.c3_reading({"value": 0.89, "ci_low": 0.83, "ci_high": 0.94}) == "agrees with both lenses"
    assert pt.c3_overlaps_band({"value": 0.87, "ci_low": 0.86, "ci_high": 0.88})
    assert pt.c2_layer(28) == 13
    b = pt.bootstrap([0.5] * 10, samples=50)
    assert b["value"] == b["ci_low"] == b["ci_high"] == 0.5


# --- alignment by role ------------------------------------------------------------------------------------------

def test_inputs_align_by_role(engine):
    x = pt.build_inputs(engine.tokenizer, record())
    assert len(x.cut_ids) < len(x.clean_ids)
    assert x.clean_p1 == len(x.clean_ids) - 1 and x.cut_p1 == len(x.cut_ids) - 1
    assert x.clean_p1 != x.cut_p1                                 # not the same index ...
    assert x.clean_ids[x.clean_q:] == x.cut_ids[x.cut_q:]         # ... the same role
    assert "Leeds" not in x.cut_prompt and "turn c" in x.cut_prompt
    assert engine.tokenizer.decode(list(x.clean_ids[-2:])).endswith("Answer:")


def test_forced_whitespace_kept_on_both(engine):
    x = pt.build_inputs(engine.tokenizer, record(gold="4 hours"))
    assert x.n_leading_ws == 1 and engine.tokenizer.decode([x.gold_id]) == "4"
    assert x.clean_ids[-1] == x.cut_ids[-1] and engine.tokenizer.decode([x.clean_ids[-1]]).strip() == ""


def test_alignment_errors(engine):
    import dataclasses

    x = pt.build_inputs(engine.tokenizer, record())
    with pytest.raises(pt.AlignmentError):
        pt.check_alignment(dataclasses.replace(x, cut_p1=x.cut_p1 - 1))
    with pytest.raises(pt.AlignmentError):
        pt.check_alignment(dataclasses.replace(x, cut_ids=x.cut_ids[:-1] + (x.cut_ids[-1] + 1,)))
    with pytest.raises(pt.AlignmentError):
        pt.check_alignment(dataclasses.replace(x, cut_ids=x.clean_ids, cut_p1=x.clean_p1, cut_q=x.clean_q))


# --- the passes on a real (tiny) model --------------------------------------------------------------------------

def test_patch_row_sanity_holds_by_construction(engine):
    x = pt.build_inputs(engine.tokenizer, record())
    row = pt.patch_row(engine, x, patch=True)
    L = engine.n_layers
    assert row["patched"] and len(row["R"]) == L and len(row["lp_patched"]) == L
    assert abs(row["R"][-1] - 1.0) < 1e-4                         # the final residual fixes the logits
    assert abs(row["lp_patched"][-1] - row["lp_clean"]) < 1e-4
    assert row["d_c"] is not None and 0 <= row["d_c"] < 1
    # an unpatched pass on the corrupt cache reproduces lp_corrupt (the cache is copied, never consumed)
    again = pt.patch_row(engine, x, patch=False)
    assert again["lp_corrupt"] == pytest.approx(row["lp_corrupt"], abs=1e-6) and not again["patched"]


def test_patch_row_default_follows_eligibility(engine):
    x = pt.build_inputs(engine.tokenizer, record())
    row = pt.patch_row(engine, x)
    assert row["patched"] == pt.patch_eligible(row["rank_clean"], row["rank_corrupt"])


# --- script: resume and fold guard ------------------------------------------------------------------------------

def test_fold_guard():
    from scripts import exp44_patch as ex

    ex.check_folds([0, 1], confirm=False)
    with pytest.raises(SystemExit) as e:
        ex.check_folds([0, 2], confirm=False)
    assert e.value.code == ex.REFUSED
    with pytest.raises(SystemExit) as e:                           # not yet authorised even with --confirm
        ex.check_folds([3], confirm=True)
    assert e.value.code == ex.REFUSED


def test_resume_skips_done_rows_and_redoes_stale(engine, tmp_path, monkeypatch):
    from scripts import exp44_patch as ex

    recs = [record("qa"), record("qb", gold="4 hours")]
    calls = []
    real = pt.patch_row
    monkeypatch.setattr(pt, "patch_row", lambda e, x, patch=None: calls.append(x.question_id) or real(e, x, patch))
    ex.run(engine, recs, tmp_path, "qwen3b", "tiny", patch_all=True, log=lambda s: None)
    assert calls == ["qa", "qb"] and (tmp_path / "qa.json").exists()
    ex.run(engine, recs, tmp_path, "qwen3b", "tiny", patch_all=True, log=lambda s: None)
    assert calls == ["qa", "qb"]                                   # nothing redone
    row = json.loads((tmp_path / "qa.json").read_text())
    row["stamp"]["cut_ids_sha256"] = "stale"
    (tmp_path / "qa.json").write_text(json.dumps(row))
    ex.run(engine, recs, tmp_path, "qwen3b", "tiny", patch_all=True, log=lambda s: None)
    assert calls == ["qa", "qb", "qa"]                             # the stale row only
    fresh = json.loads((tmp_path / "qa.json").read_text())
    assert fresh["stamp"]["method_sha256"] == pt.method_sha256() and fresh["sanity_asserted"] == fresh["eligible"]


def test_sanity_failure_exits_after_writing(engine, tmp_path, monkeypatch):
    from scripts import exp44_patch as ex

    real = pt.patch_row

    def broken(e, x, patch=None):
        row = real(e, x, True)
        row["eligible"], row["sanity"] = True, pt.sanity([0.3] + row["R"][1:])
        return row

    monkeypatch.setattr(pt, "patch_row", broken)
    with pytest.raises(SystemExit) as e:
        ex.run(engine, [record("qa")], tmp_path, "qwen3b", "tiny", patch_all=False, log=lambda s: None)
    assert e.value.code == ex.SANITY and (tmp_path / "qa.json").exists()


# --- report -----------------------------------------------------------------------------------------------------

def _fake_row(q, fold, rc, rk, d_layer, L=28, correct=True, sha="s"):
    R = [0.0] * d_layer + [0.6] * (L - d_layer - 1) + [1.0]
    return {"question_id": q, "fold": fold, "n_layers": L, "rank_clean": rc, "rank_corrupt": rk, "patched": True,
            "sanity_asserted": True, "R": R, "d_c": pt.causal_depth(R, L), "sanity": pt.sanity(R),
            "lp_unknown_patched": [-1.0] * L, "lp_unknown_clean": -2.0, "lp_unknown_corrupt": -0.5,
            "top1_patched": [5] * (L - 1) + [7], "top1_corrupt": 5, "gold_id": 7, "correct": correct, "unknown": False,
            "stamp": {"clean_prompt_sha256": sha}}


def test_report_dev_and_confirm_guard(tmp_path):
    from scripts import exp44_report as rep

    root, exp34 = tmp_path / "patch", tmp_path / "exp34"
    for model in ("qwen7b",):
        (root / model).mkdir(parents=True)
        (exp34 / model).mkdir(parents=True)
        for i in range(45):
            row = _fake_row(f"q{i}", i % 2, 1, 50 if i < 41 else 5, 14 if i % 2 else 12)
            (root / model / f"q{i}.json").write_text(json.dumps(row))
            (exp34 / model / f"q{i}.json").write_text(json.dumps({"prompt_sha256": "s", "measures": {
                v: {"P1": {"gold": {"emergence_depth": 0.5 if v == "jlens" else 0.8}}} for v in ("jlens", "logit")}}))
        (root / model / "q99.json").write_text(json.dumps(_fake_row("q99", 3, 1, 50, 1)))   # confirmation fold
    text, n = rep.build(root, confirm=False, samples=200, exp34=exp34)
    s = n["models"]["qwen7b"]
    assert s["n_rows"] == 45 and s["filter_choice_dev"]["n_strict"] == 41 and s["k_corrupt"] == 10
    assert s["n_filter"] == 41 and n["claims"]["C1"]["stat"]["n"] == 41
    assert s["c2_layer"] == 13 and 0 < n["claims"]["C2"]["stat"]["value"] < 1
    assert s["dc_minus_lens"]["jlens"]["n"] == 41
    with pytest.raises(SystemExit) as e:
        rep.build(root, confirm=True, samples=10, exp34=exp34)
    assert e.value.code == rep.REFUSED
    bad = json.loads((exp34 / "qwen7b" / "q0.json").read_text())
    bad["prompt_sha256"] = "other"
    (exp34 / "qwen7b" / "q0.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as e:
        rep.build(root, confirm=False, samples=10, exp34=exp34)
    assert e.value.code == rep.MISMATCH
