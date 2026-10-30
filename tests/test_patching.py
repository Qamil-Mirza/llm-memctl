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
    base = {"question_id": "q", "patched": True, "rank_clean": 1, "rank_corrupt": 50, "lp_clean": -1.0,
            "lp_corrupt": -9.0}
    with pytest.raises(pt.SanityError):                          # R(L-1) off: still a hard assertion (exit 3)
        pt.check_sanity({**base, "sanity": pt.sanity([0.0, 0.98])})
    pt.check_sanity({**base, "sanity": pt.sanity([0.2, 1.0])})   # R(0) off alone: a flag, not a stop (rev 5)
    pt.check_sanity({**base, "rank_corrupt": 2, "sanity": pt.sanity([0.0, 0.5])})   # outside the filter: recorded
    pt.check_sanity({**base, "lp_corrupt": -1.5, "sanity": pt.sanity([0.0, 0.5])})  # gap < 1: not asserted
    pt.check_sanity({**base, "eligible": True, "lp_corrupt": -1.5,      # the stored field is never trusted
                     "sanity": pt.sanity([0.0, 0.5])})


def _r(rc, rk, lc=-1.0, lk=-9.0):
    return {"rank_clean": rc, "rank_corrupt": rk, "lp_clean": lc, "lp_corrupt": lk}


def test_filter_and_relaxation():
    assert pt.in_rank_filter(10, 11) and not pt.in_rank_filter(11, 50) and not pt.in_rank_filter(1, 10)
    assert pt.evidence_dependent(10, 11, -1.0, -2.0) and not pt.evidence_dependent(10, 11, -1.0, -1.99)
    assert pt.patch_eligible(1, 4, -1.0, -9.0) and not pt.patch_eligible(1, 3, -1.0, -9.0)
    assert pt.GAP_MIN == 1.0
    rows = [_r(1, 50)] * 39 + [_r(1, 5)] * 10
    f = pt.choose_filter(rows)
    assert f["n_strict"] == 39 and f["relaxed"] and f["k_corrupt"] == 3 and f["n_relaxed"] == 49
    f2 = pt.choose_filter(rows + [_r(2, 11)])
    assert f2["n_strict"] == 40 and not f2["relaxed"] and f2["k_corrupt"] == 10


def test_gap_condition_on_both_filters():
    # 37f165cf-like: strict by rank (8 / 22) but corrupt log p ABOVE clean (gap -1.70): out of both filters
    row = _r(8, 22, -16.58, -14.88)
    assert pt.in_rank_filter(8, 22, pt.K_CORRUPT)
    assert not pt.evidence_dependent(8, 22, -16.58, -14.88) and not pt.patch_eligible(8, 22, -16.58, -14.88)
    assert pt.row_gap_excluded(row, pt.K_CORRUPT) and pt.row_gap_excluded(row, pt.K_CORRUPT_RELAXED)
    relaxed_small = _r(7, 5, -2.0, -2.5)                         # relaxed by rank, gap 0.5
    assert not pt.patch_eligible(7, 5, -2.0, -2.5) and pt.row_gap_excluded(relaxed_small, pt.K_CORRUPT_RELAXED)
    assert not pt.row_gap_excluded(relaxed_small, pt.K_CORRUPT)  # not in the strict rank filter at all
    rows = [_r(1, 50)] * 40 + [row] * 3 + [relaxed_small] * 2 + [_r(1, 5)] * 4
    f = pt.choose_filter(rows)
    assert (f["n_strict_rank"], f["n_strict"], f["n_gap_excluded_strict"]) == (43, 40, 3)
    assert (f["n_relaxed_rank"], f["n_relaxed"], f["n_gap_excluded_relaxed"]) == (49, 44, 5)
    assert not f["relaxed"]
    f2 = pt.choose_filter(rows[1:])                              # 39 strict + gap (42 by rank) -> relaxed
    assert f2["n_strict"] == 39 and f2["relaxed"] and f2["n_strict_rank"] == 42
    d = pt.derive({**row, "patched": True, "R": [0.3, 1.0]})
    assert not d["eligible"] and not d["sanity_asserted"] and not d["r0_flag"] and d["gap_excluded_strict"]
    assert d["filter_version"] == pt.FILTER_VERSION == 2


def test_r0_flag_derived():
    ok = {**_r(1, 50), "patched": True, "R": [0.04, 0.5, 1.0]}
    bad = {**ok, "R": [-0.06, 0.5, 1.0]}
    assert not pt.derive(ok)["r0_flag"] and pt.derive(bad)["r0_flag"] and pt.derive(bad)["sanity_asserted"]
    assert not pt.derive({**bad, "rank_corrupt": 2})["r0_flag"]            # not eligible: no flag


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
    assert row["patched"] == pt.patch_eligible(row["rank_clean"], row["rank_corrupt"], row["lp_clean"],
                                               row["lp_corrupt"])
    assert row["filter_version"] == pt.FILTER_VERSION and row["eligible"] == row["patched"]


# --- script: resume and fold guard ------------------------------------------------------------------------------

def test_fold_guard():
    from scripts import exp44_patch as ex

    ex.check_folds([0, 1], confirm=False)
    with pytest.raises(SystemExit) as e:
        ex.check_folds([0, 2], confirm=False)
    assert e.value.code == ex.REFUSED
    assert ex.CONFIRM_AUTHORISED is True                           # rev 7 (6f457fd)
    ex.check_folds([2, 3, 4], confirm=True)
    saved = ex.CONFIRM_AUTHORISED
    try:                                                           # with the lock off, --confirm is still refused
        ex.CONFIRM_AUTHORISED = False
        with pytest.raises(SystemExit) as e:
            ex.check_folds([3], confirm=True)
        assert e.value.code == ex.REFUSED
    finally:
        ex.CONFIRM_AUTHORISED = saved


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
    assert fresh["stamp"]["method_sha256"] == pt.method_sha256() and fresh["filter_version"] == pt.FILTER_VERSION
    assert fresh["sanity_asserted"] == (fresh["patched"] and fresh["eligible"])


def _eligible_row(e, x, patch, real, r0=0.0, final=1.0):
    """The real passes, then lp / rank / R forced into the eligible region (a tiny model has no real cut effect)."""
    row = real(e, x, True)
    L = row["n_layers"]
    row.update({"rank_clean": 1, "rank_corrupt": 50, "lp_clean": -1.0, "lp_corrupt": -9.0,
                "R": [r0] + [0.6] * (L - 2) + [final]})
    row["sanity"] = pt.sanity(row["R"])
    row.update(pt.derive(row))
    return row


def test_resume_keeps_rev0_rows_and_does_not_redo_them(engine, tmp_path, monkeypatch):
    """A complete row written under the rev 0 rule (stale derived fields, no filter_version, even an R(0) failure on a
    gap<1 row like 37f165cf) is not redone: the stamp covers the passes only, and the filter version is not in it."""
    from scripts import exp44_patch as ex

    recs = [record("qa"), record("qb", gold="4 hours")]
    calls = []
    real = pt.patch_row
    monkeypatch.setattr(pt, "patch_row", lambda e, x, patch=None: calls.append(x.question_id) or real(e, x, patch))
    ex.run(engine, recs, tmp_path, "qwen7b", "tiny", patch_all=False, log=lambda s: None)
    n_first = len(calls)
    for q in ("qa", "qb"):                      # rewrite as rev 0 would have: rank-only derived fields
        path = tmp_path / f"{q}.json"
        row = json.loads(path.read_text())
        for key in ("filter_version", "gap", "gap_excluded_strict", "gap_excluded_relaxed", "r0_flag"):
            row.pop(key, None)
        if q == "qa":                           # a 37f165cf-like patched row with a negative gap and R(0) = -0.11
            row.update({"patched": True, "eligible": True, "rank_clean": 8, "rank_corrupt": 22, "lp_clean": -16.58,
                        "lp_corrupt": -14.88, "R": [-0.11, 1.0], "sanity": pt.sanity([-0.11, 1.0])})
        path.write_text(json.dumps(row))
    before = {q: (tmp_path / f"{q}.json").read_text() for q in ("qa", "qb")}
    ex.run(engine, recs, tmp_path, "qwen7b", "tiny", patch_all=False, log=lambda s: None)
    assert len(calls) == n_first                                    # nothing redone, nothing stopped
    assert {q: (tmp_path / f"{q}.json").read_text() for q in ("qa", "qb")} == before   # left as written
    assert not pt.derive(json.loads(before["qa"]))["eligible"]     # ... and re-derived out of the filter
    # an unpatched complete row that the current rule wants patched is redone
    row = json.loads(before["qb"])
    row.update({"patched": False, "rank_clean": 1, "rank_corrupt": 50, "lp_clean": -1.0, "lp_corrupt": -9.0})
    (tmp_path / "qb.json").write_text(json.dumps(row))
    assert not ex.is_done(tmp_path / "qb.json", row["stamp"], patch_all=False)
    row.update({"lp_corrupt": -1.5})                                # gap < 1: not eligible, done
    (tmp_path / "qb.json").write_text(json.dumps(row))
    assert ex.is_done(tmp_path / "qb.json", row["stamp"], patch_all=False)
    assert not ex.is_done(tmp_path / "qb.json", row["stamp"], patch_all=True)


def test_r0_flag_does_not_stop_the_run(engine, tmp_path, monkeypatch):
    from scripts import exp44_patch as ex

    real = pt.patch_row
    monkeypatch.setattr(pt, "patch_row", lambda e, x, patch=None: _eligible_row(e, x, patch, real, r0=0.11))
    logs = []
    result = ex.run(engine, [record("qa"), record("qb", gold="4 hours")], tmp_path, "qwen3b", "tiny",
                    patch_all=False, log=logs.append)
    assert result["r0_flagged"] == ["qa", "qb"]                     # both rows written, the run went on
    row = json.loads((tmp_path / "qb.json").read_text())
    assert row["r0_flag"] is True and row["sanity_asserted"] and row["sanity"]["zero_ok"] is False
    assert any("R0_FLAG" in line for line in logs)


def test_sanity_failure_exits_after_writing(engine, tmp_path, monkeypatch):
    from scripts import exp44_patch as ex

    real = pt.patch_row
    monkeypatch.setattr(pt, "patch_row", lambda e, x, patch=None: _eligible_row(e, x, patch, real, final=0.97))
    with pytest.raises(SystemExit) as e:
        ex.run(engine, [record("qa"), record("qb")], tmp_path, "qwen3b", "tiny", patch_all=False, log=lambda s: None)
    assert e.value.code == ex.SANITY and (tmp_path / "qa.json").exists() and not (tmp_path / "qb.json").exists()


# --- report -----------------------------------------------------------------------------------------------------

def _fake_row(q, fold, rc, rk, d_layer, L=28, correct=True, sha="s", lp=(-1.0, -9.0), r0=0.0):
    R = [r0] + [0.0] * (d_layer - 1) + [0.6] * (L - d_layer - 1) + [1.0]
    return {"question_id": q, "fold": fold, "n_layers": L, "rank_clean": rc, "rank_corrupt": rk, "patched": True,
            "lp_clean": lp[0], "lp_corrupt": lp[1],
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
    saved = rep.FIXED_FILTER
    try:                                                           # with the filter unset, --confirm is refused
        rep.FIXED_FILTER = None
        with pytest.raises(SystemExit) as e:
            rep.build(root, confirm=True, samples=10, exp34=exp34)
        assert e.value.code == rep.REFUSED
    finally:
        rep.FIXED_FILTER = saved
    assert rep.FIXED_FILTER == {"qwen3b": 3, "qwen7b": 3}         # rev 7 (6f457fd)
    bad = json.loads((exp34 / "qwen7b" / "q0.json").read_text())
    bad["prompt_sha256"] = "other"
    (exp34 / "qwen7b" / "q0.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as e:
        rep.build(root, confirm=False, samples=10, exp34=exp34)
    assert e.value.code == rep.MISMATCH


def _write_model(root, exp34, model, rows):
    (root / model).mkdir(parents=True, exist_ok=True)
    (exp34 / model).mkdir(parents=True, exist_ok=True)
    for row in rows:
        (root / model / f"{row['question_id']}.json").write_text(json.dumps(row))
        (exp34 / model / f"{row['question_id']}.json").write_text(json.dumps({"prompt_sha256": "s", "measures": {
            v: {"P1": {"gold": {"emergence_depth": 0.5}}} for v in ("jlens", "logit")}}))


def test_report_gap_exclusion_flags_and_robustness(tmp_path):
    from scripts import exp44_report as rep

    root, exp34 = tmp_path / "patch", tmp_path / "exp34"
    rows = [_fake_row(f"q{i}", i % 2, 1, 50, 14) for i in range(40)]
    rows += [_fake_row(f"f{i}", 0, 1, 50, 2, r0=0.2) for i in range(3)]          # R(0)-flagged, early d_c
    rows += [_fake_row("neg", 0, 8, 22, 14, lp=(-16.58, -14.88), r0=-0.11)]      # 37f165cf-like
    rows += [{**_fake_row("stale", 1, 1, 50, 14, lp=(-1.0, -1.5)), "eligible": True, "evidence_dependent": True}]
    _write_model(root, exp34, "qwen7b", rows)
    text, n = rep.build(root, confirm=False, samples=200, exp34=exp34, models=("qwen7b",))
    s = n["models"]["qwen7b"]
    c = s["filter_choice_dev"]
    assert (c["n_strict_rank"], c["n_strict"], c["n_gap_excluded_strict"]) == (45, 43, 2) and not c["relaxed"]
    assert s["n_rank_filter"] == 45 and s["n_gap_excluded"] == 2 and s["n_rank_filter_gap"] == 43
    assert s["n_r0_flagged"] == 3 and s["n_filter"] == 40 and n["claims"]["C1"]["stat"]["n"] == 40
    assert n["claims"]["C1"]["n_r0_flagged"] == 3 and n["claims"]["C1"]["robust"]["stat"]["n"] == 43
    assert n["claims"]["C2"]["robust"]["stat"]["n"] == 43
    assert {g["question_id"] for g in s["gap_excluded"]} == {"neg", "stale"}
    assert s["sanity"]["max_abs_r0_eligible"] == pytest.approx(0.2)            # the -0.11 row is not eligible
    assert not s["advisor_review_required"]                                     # 3 / 43 = 7% <= 10%
    assert "C1 (7B) robustness (flagged included)" in text and "ADVISOR REVIEW" not in text
    assert "neg (f0, ranks 8/22, gap -1.70)" in text


def test_report_warns_above_ten_percent_flagged(tmp_path):
    from scripts import exp44_report as rep

    root, exp34 = tmp_path / "patch", tmp_path / "exp34"
    rows = [_fake_row(f"q{i}", i % 2, 1, 50, 14) for i in range(36)]
    rows += [_fake_row(f"f{i}", 0, 1, 50, 14, r0=0.06) for i in range(5)]      # 5 / 41 = 12% > 10%
    _write_model(root, exp34, "qwen3b", rows)
    text, n = rep.build(root, confirm=False, samples=100, exp34=exp34, models=("qwen3b",))
    s = n["models"]["qwen3b"]
    assert s["advisor_review_required"] and n["claims"]["C3"]["advisor_review_required"]
    assert s["filter_choice_dev"]["relaxed"] is False and s["n_r0_flagged"] == 5
    assert text.count("ADVISOR REVIEW REQUIRED BEFORE ANY CONFIRMATION READ") >= 2
    assert "C3 (3B) robustness (flagged included)" in text


def test_report_cli_out_and_models(tmp_path, monkeypatch):
    from scripts import exp44_report as rep

    root, exp34 = tmp_path / "patch", tmp_path / "exp34"
    _write_model(root, exp34, "qwen3b", [_fake_row(f"q{i}", 0, 1, 50, 14) for i in range(5)])
    _write_model(root, exp34, "qwen7b", [_fake_row(f"q{i}", 0, 1, 50, 14) for i in range(5)])
    monkeypatch.setattr(rep, "EXP34", exp34)
    monkeypatch.setattr(rep.build, "__defaults__", (pt.BOOT_SAMPLES, exp34, rep.MODELS))
    out = tmp_path / "x" / "exp44_report_dev_qwen3b"
    rep.main(["--root", str(root), "--samples", "20", "--models", "qwen3b", "--out", str(out)])
    numbers = json.loads(out.with_name(out.name + ".json").read_text())
    assert list(numbers["models"]) == ["qwen3b"] and numbers["filter_version"] == pt.FILTER_VERSION
    assert out.with_name(out.name + ".md").exists()
