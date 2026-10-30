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


def test_rev6_difference_controls_arithmetic():
    rng = np.random.default_rng(0)
    hc, hs = rng.standard_normal(16).astype(np.float32), rng.standard_normal(16).astype(np.float32)
    dc, ds = rng.standard_normal(16).astype(np.float32), rng.standard_normal(16).astype(np.float32)
    assert np.allclose(sw.difference_patch(hc, dc, ds), hc + (ds - dc), atol=1e-6)
    assert np.array_equal(sw.difference_patch(hc, dc, dc), hc)              # zero donor difference: h_clean unchanged
    v, n = sw.difference_perturbation(hc, hs, row_index=3, layer=2)
    assert n == pytest.approx(float(np.linalg.norm(hs.astype(np.float64) - hc)))
    assert np.linalg.norm(v.astype(np.float64) - hc) == pytest.approx(n, rel=1e-5)   # eps added, norm = diff norm
    direction = sw.random_overwrite(1.0, 16, 3, 2)                          # same seeded [44, row, layer] direction
    assert np.allclose((v - hc) / n, direction, atol=1e-5)
    v0, n0 = sw.difference_perturbation(hc, hc, 3, 2)
    assert n0 == 0 and np.array_equal(v0, hc)


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
    donor = (state["h_clean"] * 0.9, state["h_swap"] * 0.5)
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
    assert row["control_mismatch"]["kind"] == "difference_patch"
    assert row["control_random"]["kind"] == "difference_norm_perturbation"
    diff = np.linalg.norm(state["h_swap"].astype(np.float64) - state["h_clean"], axis=1)
    assert row["control_random"]["eps_norm"] == pytest.approx(list(diff), rel=1e-6)
    assert row["control_mismatch"]["donor_diff_norm"] == pytest.approx(
        list(np.linalg.norm(donor[1].astype(np.float64) - donor[0], axis=1)), rel=1e-5)
    # the caches are copied, not consumed: a second base pass reproduces the clean Delta
    again, _ = sw.base_passes(engine, x)
    assert again["clean"]["delta"] == pytest.approx(row["clean"]["delta"], abs=1e-6)


def test_rev6_zero_donor_difference_gives_zero_shift(engine):
    x = swap_inputs(engine)
    row, state = sw.base_passes(engine, x)
    self_donor = (state["h_clean"], state["h_clean"])                  # donor difference of zero
    row = sw.patch_passes(engine, x, row, state, self_donor, row_index=0)
    assert all(abs(v) < 1e-5 for v in row["control_mismatch"]["shift"])
    assert all(n == pytest.approx(0, abs=1e-6) for n in row["control_mismatch"]["donor_diff_norm"])


def test_rev6_pass_count_per_patched_row_unchanged(engine, monkeypatch):
    """Rev 6 replaces the controls in place: 4L + 3 forward passes with a donor, 3L + 3 without, as before."""
    x = swap_inputs(engine)
    L = engine.n_layers
    for donor_on, want in ((True, 4 * L + 3), (False, 3 * L + 3)):
        row, state = sw.base_passes(engine, x)
        donor = (state["h_clean"] * 0.9, state["h_swap"]) if donor_on else None
        n = []
        real = pt._p1_logits
        monkeypatch.setattr(pt, "_p1_logits", lambda *a, **k: n.append(1) or real(*a, **k))
        row = sw.patch_passes(engine, x, row, state, donor, row_index=0)
        monkeypatch.setattr(pt, "_p1_logits", real)
        assert len(n) == want == row["timing"]["n_patch_passes"]


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


def _prow(d, shift, fwd_shift=None, dc=4.0):
    fwd_shift = fwd_shift or [0.0] * len(shift)
    return {"d_T_fwd": d, "d_T_rev": d + 0.1, "control_mismatch": {"shift": shift}, "clean": {"delta": dc},
            "fwd": {"delta": [dc - v for v in fwd_shift]}, "control_random": {"shift": [0.0] * len(shift)}}


def test_s3_donor_bound():
    s = sw.s3([_prow(0.5, [0.1, 0.4]), _prow(0.7, [-0.1, 0.5])], samples=200)
    assert s["read"] and s["mismatch_max_abs_mean_shift"] == pytest.approx(0.45)
    assert s["validity"].startswith("primary") and not s["fallback_used"]
    assert s["d_T_fwd"]["value"] == pytest.approx(0.6) and s["asymmetry_median"] == pytest.approx(-0.1)
    s = sw.s3([_prow(0.5, [0.1, 0.9]), _prow(0.7, [0.1, 0.5])], samples=200)
    assert not s["read"] and s["mismatch_within_bound"] is False and s["validity"].startswith("not read")


def test_s3_fallback_specificity_ratio():
    # L = 4; median d_T = 0.5 -> l0 = ceil(0.5 * 4) = 2: layers 2, 3 checked
    # donor mean shift per layer [3, 3, 1, 2]: primary fails (|3| > 0.5)
    # forward mean shift per layer [0, 1, 8, 10]: 1 <= 0.25*8 = 2, 2 <= 0.25*10 = 2.5 -> fallback passes
    a = _prow(0.25, [4.0, 4.0, 0.0, 2.0], [0.0, 1.0, 6.0, 10.0])
    b = _prow(0.75, [2.0, 2.0, 2.0, 2.0], [0.0, 1.0, 10.0, 10.0])
    s = sw.s3([a, b], samples=200)
    assert s["mismatch_mean_shift_per_layer"] == pytest.approx([3, 3, 1, 2])
    assert s["forward_mean_shift_per_layer"] == pytest.approx([0, 1, 8, 10])
    assert s["mismatch_within_bound"] is False and s["mismatch_max_abs_mean_shift"] == pytest.approx(3)
    assert s["fallback"]["l0"] == 2 and s["fallback"]["pass"] and s["fallback_used"] and s["read"]
    assert s["validity"].startswith("fallback")
    assert s["fallback"]["donor_over_forward"] == pytest.approx([1 / 8, 2 / 10])
    # layer 3 donor 3.0 > 0.25 * 10: fallback fails -> not read (layers before l0 never count)
    b2 = _prow(0.75, [2.0, 2.0, 2.0, 4.0], [0.0, 1.0, 10.0, 10.0])
    s = sw.s3([a, b2], samples=200)
    assert s["fallback"]["pass"] is False and s["fallback"]["failing_layers"] == [3] and not s["read"]
    assert s["validity"].startswith("not read")
    # l0 rounding: median 0.6 * 4 = 2.4 -> l0 = 3
    assert sw.s3_fallback([9, 9, 9, 1], [0, 0, 0, 4], 0.6)["l0"] == 3
    assert sw.s3_fallback([9, 9, 9, 1], [0, 0, 0, 4], 0.6)["pass"] is True
    assert sw.s3_fallback([], [], None)["pass"] is None


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
    # a confirmation fold without --pooled is always refused
    with pytest.raises(SystemExit) as e:
        ex.check_folds([0, 2], False)
    assert e.value.code == ex.REFUSED
    with pytest.raises(SystemExit) as e:
        rep.check_scope([0, 2], False)
    assert e.value.code == rep.REFUSED
    # rev 4 (f9952b2) authorised the pooled read: all 330 rows with --pooled pass; with the lock off it is refused
    assert ex.POOL_AUTHORISED is True and rep.POOL_AUTHORISED is True
    ex.check_folds([0, 1, 2, 3, 4], True)
    rep.check_scope([0, 1, 2, 3, 4], True)
    old_ex, old_rep = ex.POOL_AUTHORISED, rep.POOL_AUTHORISED
    try:
        ex.POOL_AUTHORISED = rep.POOL_AUTHORISED = False
        with pytest.raises(SystemExit) as e:
            ex.check_folds([0, 1, 2, 3, 4], True)
        assert e.value.code == ex.REFUSED
        with pytest.raises(SystemExit) as e:
            rep.check_scope([0, 1, 2, 3, 4], True)
        assert e.value.code == rep.REFUSED
    finally:
        ex.POOL_AUTHORISED, rep.POOL_AUTHORISED = old_ex, old_rep


def _eligible(engine, rec, sub):
    import exp44_swap_gate as G
    _, place = G.place_gold(rec, G.Checker())
    prompt, _, _ = G.swap_prompt(rec, place, sub)
    gi, si = G.token_info(engine.tokenizer, rec.gold), G.token_info(engine.tokenizer, sub)
    prefix, gd, sd = sw.shared_prefix(gi["content"], si["content"])
    return {"question_id": rec.question_id, "fold": rec.fold, "substitute": sub, "surface_type": "name",
            "gold_first_id": gi["first_id"], "substitute_first_id": si["first_id"], "shared_prefix_ids": list(prefix),
            "gold_read_id": gd, "substitute_read_id": sd,
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
    assert all(r["control_mismatch"]["kind"] == "difference_patch" for r in rows.values())
    old = json.loads((tmp_path / "S" / "qa.json").read_text())         # rev 6: an old-method row is refused, not redone
    old["stamp"]["method_sha256"] = "rev1"
    (tmp_path / "S" / "qa.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit) as e:
        ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None).run()
    assert e.value.code == ex.OLD_METHOD and len(calls) == 4
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
             "control_random": {"shift": [0.0, 0.0]}, "fwd": {"delta": [dc, ds]}, "clean": {"delta": dc},
             "stamp": {"method_sha256": sw.method_sha256()}}
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
    assert s["claims"]["S3"]["validity"].startswith("primary") and "Absolute check (primary)" in text
    old = json.loads((S / "q0.json").read_text())
    (S / "q0.json").write_text(json.dumps({**old, "stamp": {"method_sha256": "rev1"}}))
    with pytest.raises(SystemExit) as e:                               # rev 6: old-method rows are never pooled
        rep.build(tmp_path, "qwen3b", (0, 1), samples=10)
    assert e.value.code == rep.OLD_METHOD
    (S / "q0.json").write_text(json.dumps(old))
    unpatched = json.loads((S / "q10.json").read_text())
    unpatched["patched"] = False
    (S / "q10.json").write_text(json.dumps(unpatched))
    with pytest.raises(SystemExit) as e:
        rep.build(tmp_path, "qwen3b", (0, 1), samples=10)
    assert e.value.code == rep.UNPATCHED


# --- rev 3: the shared-prefix rule ------------------------------------------------------------------------------

def money_record(qid="m1", gold="$800"):
    lines = ("user, d1: turn a", f"user, d2: The sofa cost me {gold} last week.", "user, d3: turn c")
    prompt = ("Use only the memory.\n\n## Memory\n" + "\n".join(lines) + "\n\n## Current input\nQuestion: (asked on "
              "2023/05/30) How much did the sofa cost?\n\nNote:")
    return PromptRecord(qid, 0, "qwen3b", prompt, OUTPUT, gold, "single-session-user", True, lines, (lines[1],),
                        True, False, True)


def test_shared_prefix_arithmetic():
    assert sw.shared_prefix((5, 8, 0), (5, 2, 9, 0)) == ((5,), 8, 2)
    assert sw.shared_prefix((7,), (9,)) == ((), 7, 9)                  # empty prefix: the first tokens
    assert sw.shared_prefix((7,), (7, 3)) == ((7,), None, 3)           # gold is a token-prefix of S: no divergence
    assert sw.shared_prefix((7, 1), (7, 1)) == ((7, 1), None, None)


def test_dollar_prefix_is_forced_and_read_after(engine):
    tok = engine.tokenizer
    rec = money_record()
    old = rec.memory_lines[1]
    new = old.replace("$800", "$2,000")
    x = sw.build_swap_inputs(tok, rec, rec.prompt.replace(old, new), old, new, "$2,000", (9788,))
    assert tok.decode(list(x.prefix_ids)) == " $" and x.g_first_id == x.prefix_ids[0]
    assert tok.decode([x.g_id]) == "8" and tok.decode([x.s_id]) == "2"
    for ids in (x.clean_ids, x.swap_ids, x.cut_ids):                   # the prefix ends every sequence
        assert ids[-len(x.prefix_ids):] == x.prefix_ids
        assert tok.decode(list(ids[-3:])).endswith("Answer: $")
    from memctl.analysis.jlens_readout import readout_for
    plain = readout_for(tok, rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold)
    assert x.clean_ids == tuple(plain.input_ids) + x.prefix_ids
    row, state = sw.base_passes(engine, x)
    assert row["prefix_ids"] == list(x.prefix_ids) and row["g_id"] == x.g_id
    row = sw.patch_passes(engine, x, row, state, None, 0)
    assert abs(row["T_fwd"][-1] - 1) < 1e-4 and abs(row["T_rev"][-1] - 1) < 1e-4


def test_empty_prefix_reproduces_rev2_inputs(engine):
    from memctl.analysis.jlens_readout import readout_for, tracked_token_ids
    tok = engine.tokenizer
    rec = record()
    x = swap_inputs(engine)
    assert x.prefix_ids == () and x.g_id == x.g_first_id
    clean = readout_for(tok, rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold)
    assert x.clean_ids == tuple(clean.input_ids) and x.g_id == clean.tracked["gold"]
    assert x.s_id == tracked_token_ids(tok, "Paris")[0]["gold"]
    new = LINES[2].replace("Leeds", "Paris")
    swap = readout_for(tok, x.swap_prompt, tuple(new if l == LINES[2] else l for l in LINES), (new,), rec.gold)
    assert x.swap_ids == tuple(swap.input_ids)


def test_gate_rev3_first_token_after_prefix(engine):
    import exp44_swap_gate as G
    tok = engine.tokenizer
    r = money_record("a", "$800")
    pool = [r, money_record("b", "$2,000")]
    tinfo = {x.question_id: G.token_info(tok, x.gold) for x in pool}
    types = {x.question_id: G.type_key(x.gold) for x in pool}
    _, place = G.place_gold(r, G.Checker())
    assert G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), uses=G.Counter(), rev=2)[1] == \
        "first_token_equal"
    cand, why, _, _ = G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), uses=G.Counter(), rev=3)
    assert why is None and cand.question_id == "b"
    # a candidate whose content tokens extend the gold's has no divergent token: first_token_equal
    lr = record("c", gold="Leeds")
    pool2 = [lr, record("d", gold="Paris")]
    ti = {x.question_id: G.token_info(tok, x.gold) for x in pool2}
    ti["d"] = dict(ti["d"], content=ti["c"]["content"] + (5,), first_id=ti["c"]["first_id"])
    ty = {x.question_id: G.type_key(x.gold) for x in pool2}
    _, pl = G.place_gold(lr, G.Checker())
    assert G.choose_substitute(lr, pl, pool2, ti, ty, G.Checker(), uses=G.Counter(), rev=3)[1] == "first_token_equal"
    del ti["d"]["content"], ti["d"]["first_id"]
    ti["d"] = G.token_info(tok, "Paris")
    assert G.choose_substitute(lr, pl, pool2, ti, ty, G.Checker(), uses=G.Counter(), rev=3)[0].question_id == "d"


# --- rev 8 (83fe60a, D-47) ----------------------------------------------------------------------------------------

def test_rev8_gate_excludes_other_and_matches_determiners(engine):
    import exp44_swap_gate as G
    recs = [record("qa", gold="Leeds"), record("qb", gold="Paris", correct=True, unknown=False),
            record("qc", gold="in the attic"), record("qd", gold="The Grand Hall"),
            record("qe", gold="The Old Mill", correct=True, unknown=False), record("qf", gold="Madrid")]
    res = G.evaluate(recs, engine.tokenizer, rev=8)
    removed = {x["question_id"]: x["exclusion"] for x in res["removed_rows"]}
    assert removed == {"qc": "type_other_rev8"}
    sub = {e["question_id"]: e["substitute"] for e in res["eligible"]}
    assert sub["qd"] == "The Old Mill" and sub["qe"] == "The Grand Hall"
    assert all(G.leading_determiner(sub[q]) is None for q in ("qa", "qb", "qf"))
    t = G.tables(res)
    assert list(t["exclusions_in_order"]) == list(G.EXCLUSIONS_REV8) and t["exclusions_in_order"]["type_other_rev8"] == 1
    # rev 3 on the same rows: "other" is kept (no other "other" row to swap with: no_substitute), determiners free
    res3 = G.evaluate(recs, engine.tokenizer, rev=3)
    assert {x["question_id"]: x["exclusion"] for x in res3["removed_rows"]}.get("qc") == "no_substitute"


def test_rev8_srun_stamp_is_distinct_and_never_mixed(engine, tmp_path):
    import exp44_swap as ex
    recs = [record("qa", gold="Leeds", correct=True, unknown=False), record("qb", gold="Madrid")]
    order = [_eligible(engine, r, s) for r, s in zip(recs, ["Paris", "Berlin"])]
    assert sw.method_sha256(8) != sw.method_sha256() == sw.method_sha256(3)
    ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None,
            gate_rev=8).run()
    row = json.loads((tmp_path / "S" / "qa.json").read_text())
    assert row["stamp"]["method_sha256"] == sw.method_sha256(8) and row["gate_rev"] == 8
    with pytest.raises(SystemExit) as e:                               # a rev 3 run over rev 8 rows: refused
        ex.SRun(engine, "qwen3b", {r.question_id: r for r in recs}, order, tmp_path, "sha", log=lambda s: None).run()
    assert e.value.code == ex.OLD_METHOD


def test_rev8_report_honesty_split_and_wording(tmp_path):
    import exp44_swap_report as rep
    S = tmp_path / rep.REV8_DIR / "S"
    U = tmp_path / "qwen3b" / "U"
    S.mkdir(parents=True)
    U.mkdir(parents=True)
    # 10 unchanged successes all follow; 5 changed successes, 2 follow; 45 failures, all answer-production
    rows = ([("u", "success", 3, -3)] * 10 + [("c", "success", 3, -3)] * 2 + [("c", "success", 1, 0.9)] * 3
            + [("u", "wrong", 3, -3)] * 45)
    changed = set()
    for i, (kind, lab, dc, ds) in enumerate(rows):
        q = f"q{i}"
        if kind == "c":
            changed.add(q)
        r = {"question_id": q, "fold": i % 5, **_row(lab, dc, ds), "patched": sw.follows(dc, ds), "d_T_fwd": 0.5,
             "d_T_rev": 0.6, "donor": "q0", "control_mismatch": {"shift": [0.1, 0.0]},
             "control_random": {"shift": [0.0, 0.0]}, "fwd": {"delta": [dc, ds]}, "clean": {"delta": dc},
             "stamp": {"method_sha256": sw.method_sha256(8)}}
        (S / f"{q}.json").write_text(json.dumps(r))
    text, s = rep.build(tmp_path, "qwen3b", tuple(range(5)), samples=300, patch_root=tmp_path / "none",
                        exp34=tmp_path / "none", gate_rev=8, changed=changed)
    r8 = s["rev8"]
    assert s["claims"]["S0"]["value"] == pytest.approx(12 / 15)
    assert r8["S0_changed"]["value"] == pytest.approx(2 / 5) and r8["S0_changed"]["n"] == 5
    assert r8["S0_unchanged"]["value"] == 1.0 and r8["S0_unchanged"]["n"] == 10
    assert r8["n_changed_rows"] == 5 and r8["changed_labels"]["success"] == 5 and r8["confirmatory"] is False
    assert "a repaired design tested on the same rows after the fault was seen" in text and "No claim" in text
    assert "S0 on the changed-substitute rows" in text and "Power, stated in advance" in text
    if s["claims"]["S1"]["established"]:
        assert s["claims"]["S1"]["reading"].startswith(rep.REV8_PREFIX)
    assert all("not confirmatory" in v for v in s["status"].values() if v.startswith("read"))
    # rev 3 rows in the rev 8 directory are refused (never pooled)
    old = json.loads((S / "q0.json").read_text())
    (S / "q0.json").write_text(json.dumps({**old, "stamp": {"method_sha256": sw.method_sha256()}}))
    with pytest.raises(SystemExit) as e:
        rep.build(tmp_path, "qwen3b", tuple(range(5)), samples=10, gate_rev=8, changed=changed)
    assert e.value.code == rep.OLD_METHOD


def test_rev8_report_s1_wording_when_established():
    import exp44_swap_report as rep
    fails = [_row("wrong", 3, -3)] * 60
    claims = {"S1": sw.s1(fails, samples=300)}
    assert claims["S1"]["established"]
    rep.rev8_reads([], claims, set(), 10)
    assert claims["S1"]["reading"] == rep.REV8_PREFIX + "answer-production failures are the majority"
    few = {"S1": sw.s1([_row("wrong", 3, -3)] * 10 + [_row("wrong", 0.2, 0.1)] * 10, samples=300)}
    out = rep.rev8_reads([], few, set(), 10)
    assert out["underpowered"] and not few["S1"]["established"]
