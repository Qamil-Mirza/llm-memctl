"""§44 rev 1 amendment §B-§C: Delta / F arithmetic, class rules, transfer arithmetic and bounds, controls, the U
prompt wording, claims arithmetic, fold guards, resume (tiny random Qwen2 as tests/test_patching.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memctl.analysis import patching as pt  # noqa: E402
from memctl.analysis import swap as sw  # noqa: E402
from memctl.jacobian_data import PromptRecord  # noqa: E402

LINES = ("user, d1: turn a", "assistant, d1: turn b", "user, d2: I was born in Leeds.", "user, d3: turn c")
PROMPT = ("Use only the memory.\n\n## Memory\n" + "\n".join(LINES) + "\n\n## Current input\nQuestion: (asked on "
          "2023/05/30) Where was I born?\n\nNote:")
OUTPUT = "The memory says I was born in a northern city.\nAnswer: unknown"


def record(qid="q1", fold=0, gold="Leeds", correct=False, unknown=True):
    lines = tuple(l.replace("Leeds", gold) for l in LINES)
    return PromptRecord(qid, fold, "qwen3b", PROMPT.replace("Leeds", gold), OUTPUT, gold, "single-session-user", True,
                        lines, (lines[2],), True, correct, unknown)


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


def swap_inputs(engine, rec=None, sub="Paris"):
    rec = rec or record()
    new = LINES[2].replace("Leeds", sub)
    return sw.build_swap_inputs(engine.tokenizer, rec, PROMPT.replace(LINES[2], new), LINES[2], new, sub,
                                (9788, 21693))


# --- arithmetic -------------------------------------------------------------------------------------------------

def test_readout_delta_and_unknown():
    logits = np.zeros(10, dtype=np.float32)
    logits[[2, 5, 7, 8]] = [3.0, 1.0, 0.5, 0.5]
    r = sw.readout(logits, g=2, s=5, refusal_ids=(7, 8))
    assert r["delta"] == pytest.approx(2.0) and r["lp_g"] - r["lp_s"] == pytest.approx(2.0)
    assert r["rank_g"] == 1 and r["rank_s"] == 2 and r["top1"] == 2
    assert r["lp_unknown"] == pytest.approx(np.logaddexp(r["lp_refusal_ids"][0], r["lp_refusal_ids"][1]))


def test_follows_and_classes():
    assert sw.follows(0.6, -0.6) and not sw.follows(0.4, -0.4)          # F = 1.2 vs 0.8 at 1 nat
    assert not sw.follows(3.0, 0.5)                                     # F >= 1 without the sign flip
    assert not sw.follows(-0.5, -2.0)
    assert sw.failure_class(0.6, -0.6) == "answer_production"
    assert sw.failure_class(2.0, 1.5) == "evidence_use"                 # |F| = 0.5 < 1
    assert sw.failure_class(-1.0, -0.2) == "evidence_use"               # |F| = 0.8, negative F
    assert sw.failure_class(3.0, 0.5) == "indeterminate"                # F >= 1, no flip
    assert sw.failure_class(-2.0, 0.5) == "indeterminate"               # F <= -1: neither
    assert sw.failure_class(0.3, -0.3, margin=0.5) == "answer_production"
    assert sw.failure_class(0.3, -0.3, margin=2) == "evidence_use"


def test_transfer_arithmetic_and_bounds():
    # forward: from clean (Delta 4) to swap (Delta -2)
    t = sw.transfer([4.0, 3.0, 1.0, -0.5, -2.0], d_from=4.0, d_to=-2.0)
    assert t == pytest.approx([0.0, 1 / 6, 0.5, 0.75, 1.0])
    assert sw.transfer_depth(t, 5) == 2 / 5
    # reverse: from swap to clean
    assert sw.transfer([-2.0, 1.0, 4.0], d_from=-2.0, d_to=4.0) == pytest.approx([0.0, 0.5, 1.0])
    assert sw.transfer([1.0], 2.0, 2.0) is None and sw.transfer_depth(None, 4) is None
    assert sw.transfer_sanity([0.049, 0.5, 1.0099])["pass"] is True
    assert sw.transfer_sanity([0.05, 1.0])["zero_ok"] is False
    assert sw.transfer_sanity([0.0, 0.98])["final_ok"] is False
    assert sw.transfer_sanity(None)["pass"] is None


def test_controls_arithmetic():
    assert sw.shift_toward_s([4.0, 3.5, 5.0], d_clean=4.0) == [0.0, 0.5, -1.0]
    v = sw.random_overwrite(7.0, 16, row_index=3, layer=2)
    assert np.linalg.norm(v) == pytest.approx(7.0, rel=1e-5)
    assert np.array_equal(v, sw.random_overwrite(7.0, 16, 3, 2))
    assert not np.array_equal(v, sw.random_overwrite(7.0, 16, 4, 2))
    assert not np.array_equal(v, sw.random_overwrite(7.0, 16, 3, 1))
    order = [{"gold_first_id": 1, "substitute_first_id": 2}, {"gold_first_id": 2, "substitute_first_id": 9},
             {"gold_first_id": 5, "substitute_first_id": 6}, {"gold_first_id": 7, "substitute_first_id": 1}]
    assert sw.choose_donor(order, 3) == 2            # previous row, no shared first id
    assert sw.choose_donor(order, 1) == 3            # row 0's S1 (2) is row 1's G1: skipped
    assert sw.choose_donor(order, 0) == 2            # 3 has S1 = 1 = row 0's G1, skipped; then 2
    assert sw.choose_donor(order[:1], 0) is None
    assert sw.self_layers(36) == (0, 18, 35)


# --- the passes on a real (tiny) model --------------------------------------------------------------------------

def test_inputs_align_and_differ_only_in_the_line(engine):
    x = swap_inputs(engine)
    assert x.clean_ids[-1] == x.swap_ids[-1] == x.cut_ids[-1]
    assert x.g_id != x.s_id and engine.tokenizer.decode([x.s_id]).strip() == "Paris"
    assert "Paris" in x.swap_prompt and "Leeds" not in x.swap_prompt and "Leeds" not in x.cut_prompt
    assert x.swap_prompt.replace("Paris", "Leeds") == x.clean_prompt
    with pytest.raises(pt.AlignmentError):           # same first token as the gold is refused
        swap_inputs(engine, sub="Leeds United")


def test_base_and_patch_passes(engine):
    x = swap_inputs(engine)
    row, state = sw.base_passes(engine, x)
    L = engine.n_layers
    assert row["F"] == pytest.approx(row["clean"]["delta"] - row["swap"]["delta"])
    assert row["cls"]["1"] == sw.failure_class(row["clean"]["delta"], row["swap"]["delta"])
    assert set(row["follows"]) == {"0.5", "1", "2"} and state["h_clean"].shape == (L, engine.hidden)
    donor = state["h_swap"] * 0.5
    row = sw.patch_passes(engine, x, row, state, donor, row_index=0)
    assert row["patched"] and len(row["T_fwd"]) == L and len(row["T_rev"]) == L
    assert abs(row["T_fwd"][-1] - 1) < 1e-4 and abs(row["T_rev"][-1] - 1) < 1e-4     # final residual fixes logits
    assert row["fwd"]["delta"][-1] == pytest.approx(row["swap"]["delta"], abs=1e-4)
    assert row["control_self"]["pass"] and all(abs(v) < 1e-5 for v in row["control_self"]["shift"].values())
    assert all(v == pytest.approx(0, abs=1e-5) for v in row["control_self"]["T"].values())
    assert len(row["control_mismatch"]["shift"]) == L and len(row["control_random"]["shift"]) == L
    assert row["control_mismatch"]["shift"][0] == pytest.approx(row["clean"]["delta"]
                                                                 - row["control_mismatch"]["delta"][0])
    assert row["timing"]["n_patch_passes"] == 4 * L + 3
    # the caches are copied, not consumed: a second base pass reproduces the clean Delta
    again, _ = sw.base_passes(engine, x)
    assert again["clean"]["delta"] == pytest.approx(row["clean"]["delta"], abs=1e-6)


def test_patch_sanity_check():
    good = {"question_id": "q", "patched": True, "sanity_fwd": sw.transfer_sanity([0.0, 1.0]),
            "sanity_rev": sw.transfer_sanity([0.01, 1.0]), "control_self": {"pass": True, "shift": {}}}
    sw.check_patch_sanity(good)
    sw.check_patch_sanity({"patched": False})
    with pytest.raises(sw.SanityError):
        sw.check_patch_sanity({**good, "sanity_rev": sw.transfer_sanity([0.2, 1.0])})
    with pytest.raises(sw.SanityError):
        sw.check_patch_sanity({**good, "control_self": {"pass": False, "shift": {"0": 0.1}}})


# --- condition U --------------------------------------------------------------------------------------------------

def test_u_wording_is_the_35_frozen_note():
    from memctl.analysis import answer_selection as sel
    note = "The memory says X.\n"
    assert sel.B_INSTRUCTION == "Based on the note above, give the answer as a short phrase. Do not reply unknown."
    assert sw.u_continuation(note, "U") == note + "\n" + sel.B_INSTRUCTION + "\nAnswer:"
    assert sw.u_continuation(note, "U_steer") == sw.u_continuation(note, "U")
    assert sw.u_continuation(note, "A") == note + "Answer:" == sw.u_continuation(note, "A_steer")
    w = sw.u_wording("qwen3b")
    assert w["max_new_tokens"] == 64 and w["steer"]["layer"] == 29 and w["steer"]["alpha"] == 2.0
    assert sw.u_wording("llama3b")["steer"]["layer"] == 17
    assert sw.u_wording_sha256("qwen3b") != sw.u_wording_sha256("llama3b")


def test_u_generate(engine):
    from memctl.analysis import steering as st
    vec = np.full(engine.hidden, 0.1, np.float32)
    out = sw.u_generate(engine, record(), sw.U_KINDS, vec, layer=3, max_new=4)
    assert not out["no_marker"] and set(sw.U_KINDS) <= set(out)
    assert out["U"]["vector_sha256"] is None and out["U_steer"]["vector_sha256"] == st.vector_sha(vec)
    assert out["U"]["last_token"].endswith(":") and len(out["U"]["gen_ids"]) <= 4
    assert out["U"]["input_sha256"] != out["A"]["input_sha256"]
    import dataclasses
    unmarked = dataclasses.replace(record(), output="no marker here")
    assert sw.u_generate(engine, unmarked, ("U",), vec, 3) == {"no_marker": True}


def test_steer_vectors_match_their_directions_files():
    for model in ("qwen3b", "llama3b"):
        layer, alpha, unit, path = sw.STEER[model]
        if not Path(path).exists():
            pytest.skip(f"{path} not on disk")
        v = sw.steer_vector(model)
        raw = np.load(path)[f"L{layer}"]
        assert np.allclose(v, 2.0 * raw)


# --- claims (§C) ------------------------------------------------------------------------------------------------

def _row(label, dc, ds):
    return {"label": label, "follows": {f"{m:g}": sw.follows(dc, ds, m) for m in sw.MARGINS},
            "cls": {f"{m:g}": sw.failure_class(dc, ds, m) for m in sw.MARGINS}}


def test_s0():
    rows = [_row("success", 3, -3)] * 45 + [_row("success", 2, 1.8)] * 5          # 0.90
    s = sw.s0(rows, samples=500)
    assert s["value"] == pytest.approx(0.9) and s["pass"] and s["reading"] == "valid"
    bad = [_row("success", 3, -3)] * 7 + [_row("success", 2, 1.8)] * 3           # 0.70
    assert not sw.s0(bad, samples=500)["pass"] and sw.s0(bad, 500)["reading"] == "swap not valid"
    assert sw.s0([], samples=10)["reading"] is None


def test_s1_criteria_and_readings():
    ap = [_row("refusal", 3, -3)] * 60          # follows at every margin
    eu = [_row("wrong", 1.0, 0.8)] * 40         # |F| = 0.2: evidence-use at every margin
    s = sw.s1(ap + eu, samples=500)
    assert s["shares"]["1"]["answer_production"] == pytest.approx(0.6) and s["indeterminate"] == 0
    assert s["a"] and s["b"] and s["c"] and s["established"]
    assert s["reading"] == "answer-production failures are the majority"
    # (b) fails: rows that follow at 0.5 but not at 2 nats
    marginal = [_row("refusal", 0.4, -0.4)] * 30            # F = 0.8: AP at 0.5, EU at 1 and 2
    s = sw.s1(ap[:30] + marginal + eu, samples=500)
    assert not s["b"] and s["reading"] == "not reliably distinguishable on these rows"
    # (a) fails: many indeterminate
    ind = [_row("wrong", 3.0, 0.5)] * 30
    assert not sw.s1(ap + ind, samples=500)["a"]
    # (c) fails on thin rows
    s = sw.s1(ap[:5] + eu[:5], samples=500)
    assert not s["c"] and not s["established"]
    s = sw.s1(ap[:20] + eu, samples=500)
    assert s["established"] in (True, False) and (not s["established"] or s["reading"] in (
        "evidence-use failures are the majority", "mixed"))
    assert sw.s1([], samples=10)["reading"] is None


def test_s2_threshold_and_difference():
    ap = [{"cls": {"1": "answer_production"}, "steer_gold": g} for g in [True] * 7 + [False] * 1]
    eu = [{"cls": {"1": "evidence_use"}, "steer_gold": g} for g in [True] * 1 + [False] * 7]
    s = sw.s2(ap + eu, samples=500)
    assert s["readable"] and s["value"] == pytest.approx(0.75) and s["shown"] is True
    s = sw.s2(ap[:7] + eu, samples=500)
    assert not s["readable"] and s["shown"] is None and s["reading"].startswith("descriptive")


def test_s3_donor_bound():
    def prow(d, shift):
        return {"d_T_fwd": d, "d_T_rev": d + 0.1, "control_mismatch": {"shift": shift},
                "control_random": {"shift": [0.0, 0.0]}}
    s = sw.s3([prow(0.5, [0.1, 0.4]), prow(0.7, [-0.1, 0.5])], samples=200)
    assert s["read"] and s["mismatch_max_abs_mean_shift"] == pytest.approx(0.45)
    assert s["d_T_fwd"]["value"] == pytest.approx(0.6) and s["asymmetry_median"] == pytest.approx(-0.1)
    s = sw.s3([prow(0.5, [0.1, 0.9]), prow(0.7, [0.1, 0.5])], samples=200)
    assert not s["read"] and s["mismatch_within_bound"] is False


def test_u1_readings():
    def urow(g, gs):
        return {"U": {"gold": g, "cls": "answer"}, "U_steer": {"gold": gs, "cls": "answer"}}
    high = [urow(True, True)] * 70 + [urow(False, True)] * 30
    s = sw.u1(high, samples=500)
    assert s["value"] == pytest.approx(0.7) and s["reading"] == "the refusal disposition is mostly a prompt artefact"
    assert s["steer_minus_u"]["value"] == pytest.approx(0.3)
    low = [urow(True, False)] * 10 + [urow(False, False)] * 90
    assert sw.u1(low, samples=500)["reading"] == "not the format"
    mid = [urow(True, False)] * 40 + [urow(False, False)] * 60
    assert sw.u1(mid + [{"no_marker": True}], samples=500)["reading"] == "partial"
    assert sw.u1(mid + [{"no_marker": True}], samples=10)["n_no_marker"] == 1


# --- scripts: guards, resume, report ----------------------------------------------------------------------------

def test_fold_guards():
    import exp44_swap as ex
    import exp44_swap_report as rep
    ex.check_folds([0, 1], pooled=False)
    for folds, pooled in (([0, 2], False), ([0, 1, 2, 3, 4], True)):
        with pytest.raises(SystemExit) as e:
            ex.check_folds(folds, pooled)
        assert e.value.code == ex.REFUSED
        with pytest.raises(SystemExit) as e:
            rep.check_scope(folds, pooled)
        assert e.value.code == rep.REFUSED
    assert ex.POOL_AUTHORISED is False and rep.POOL_AUTHORISED is False


def _eligible(engine, rec, sub):
    import exp44_swap_gate as G
    _, place = G.place_gold(rec, G.Checker())
    prompt, _, _ = G.swap_prompt(rec, place, sub)
    gi, si = G.token_info(engine.tokenizer, rec.gold), G.token_info(engine.tokenizer, sub)
    return {"question_id": rec.question_id, "fold": rec.fold, "substitute": sub, "surface_type": "name",
            "gold_first_id": gi["first_id"], "substitute_first_id": si["first_id"],
            "evidence_line_index": place["line_index"], "clean_prompt_sha256": pt.sha256_text(rec.prompt),
            "swap_prompt_sha256": pt.sha256_text(prompt)}


def test_srun_resume_and_forced_patch(engine, tmp_path, monkeypatch):
    import exp44_swap as ex
    recs = [record("qa", gold="Leeds", correct=True, unknown=False), record("qb", gold="Madrid"),
            record("qc", fold=1, gold="Oslo", correct=False, unknown=False)]
    subs = ["Paris", "Berlin", "Tokyo"]
    order = [_eligible(engine, r, s) for r, s in zip(recs, subs)]
    real_base = sw.base_passes

    def base_follow(e, x):                       # force "follows" so the patch path runs on the tiny model
        row, state = real_base(e, x)
        row["follows"]["1"] = True
        return row, state
    monkeypatch.setattr(sw, "base_passes", base_follow)
    calls = []
    real_patch = sw.patch_passes
    monkeypatch.setattr(sw, "patch_passes", lambda *a, **k: calls.append(a[1].question_id) or real_patch(*a, **k))
    monkeypatch.setattr(sw, "check_patch_sanity", lambda row: None)     # T(0) on a random tiny model is not bounded
    run = ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None)
    run.run()
    assert sorted(calls) == ["qa", "qb", "qc"] and calls[0] == "qb"     # qa's donor (qc, cyclic) is deferred
    rows = {q: json.loads((tmp_path / "S" / f"{q}.json").read_text()) for q in ("qa", "qb", "qc")}
    assert rows["qa"]["donor"] == "qc" and rows["qb"]["donor"] == "qa" and rows["qc"]["donor"] == "qb"
    assert all(r["patched"] and r["control_mismatch"] for r in rows.values())
    assert rows["qa"]["label"] == "success" and rows["qb"]["label"] == "refusal" and rows["qc"]["label"] == "wrong"
    run2 = ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None)
    run2.run()
    assert len(calls) == 3                                              # nothing redone
    stale = rows["qb"]
    stale["stamp"]["swap_ids_sha256"] = "stale"
    (tmp_path / "S" / "qb.json").write_text(json.dumps(stale))
    ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None).run()
    assert calls[3:] == ["qb"]
    bad = dict(order[0], swap_prompt_sha256="x")
    with pytest.raises(SystemExit) as e:
        ex.rebuild_swap(recs[0], bad)
    assert e.value.code == ex.GATE_MISMATCH


def test_report_build(tmp_path):
    import exp44_swap_report as rep
    S = tmp_path / "qwen3b" / "S"
    U = tmp_path / "qwen3b" / "U"
    S.mkdir(parents=True)
    U.mkdir(parents=True)
    rows = ([("success", 3, -3)] * 9 + [("success", 1, 0.9)] + [("refusal", 3, -3)] * 4 + [("wrong", 1, 0.9)] * 2)
    for i, (lab, dc, ds) in enumerate(rows):
        r = {"question_id": f"q{i}", "fold": i % 2, **_row(lab, dc, ds), "patched": sw.follows(dc, ds),
             "d_T_fwd": 0.5, "d_T_rev": 0.6, "donor": "q0", "control_mismatch": {"shift": [0.1, 0.0]},
             "control_random": {"shift": [0.0, 0.0]}}
        (S / f"q{i}.json").write_text(json.dumps(r))
        if lab == "refusal":
            (U / f"q{i}.json").write_text(json.dumps({"question_id": f"q{i}", "fold": i % 2, "in_swap_subset": True,
                                                      "U": {"gold": True, "cls": "answer"},
                                                      "U_steer": {"gold": True, "cls": "answer"},
                                                      "A": {"gold": False}, "A_steer": {"gold": True}}))
    (S / "q99.json").write_text(json.dumps({"question_id": "q99", "fold": 3, "label": "success"}))   # never opened
    text, s = rep.build(tmp_path, "qwen3b", (0, 1), samples=200, patch_root=tmp_path / "none",
                        exp34=tmp_path / "none")
    assert s["n_swap_rows"] == 16 and s["claims"]["S0"]["value"] == pytest.approx(0.9)
    assert s["claims"]["S1"]["shares"]["1"]["answer_production"] == pytest.approx(4 / 6)
    assert s["claims"]["U1"]["value"] == 1.0 and s["claims"]["S2"]["n_a"] == 4
    assert s["status"]["S1"] == "descriptive" and "S0" in text
    unpatched = json.loads((S / "q10.json").read_text())
    unpatched["patched"] = False
    (S / "q10.json").write_text(json.dumps(unpatched))
    with pytest.raises(SystemExit) as e:
        rep.build(tmp_path, "qwen3b", (0, 1), samples=10)
    assert e.value.code == rep.UNPATCHED
