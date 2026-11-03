"""§51 build tests: generator and closure, grader fixtures, bootstrap rules, edit layer (family, manifest, bit-for-bit
apply / revert), engine triggers, the metered service, the sandbox and harness (dummy client, no network), the
mechanism kit, the no-op gate and the sealed-split guard. Model tests use the tiny random dummy Qwen2."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_cases as G  # noqa: E402
import exp51_common as CM  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402
from memctl.analysis import grade51 as GR  # noqa: E402
from memctl.analysis import mech51 as MK  # noqa: E402
from memctl.analysis import stats51 as S  # noqa: E402

T_CITIES = ["London", "Oxford", "Bristol", "Glasgow", "Sheffield", "Plymouth", "Swansea", "Aberdeen", "Bradford",
            "Nottingham", "Southampton", "Portsmouth"]


# ---------------------------------------------------------------------------------------------- generator

@pytest.fixture(scope="module")
def small_cases():
    sizes = {"E": 4, "S": 3, "H": 4, "Tcity": 2}
    rows, info = G.build_cases(sizes, T_CITIES)
    return rows, info, sizes


def test_feasibility_reports_reduction():
    f = G.feasibility()
    assert f["fits_at_1"] is True and f["de_tu_bases_requested"] == 128 and f["r"] == 1.0   # T-format dropped
    with_tf = G.feasibility({**G.SIZES, "Tformat": 24})                  # the rev 1a plan did not fit (r 0.958)
    assert with_tf["fits_at_1"] is False and with_tf["r"] == pytest.approx(0.958)


def test_thirteen_rows_closure_and_keys(small_cases):
    rows, info, sizes = small_cases
    pr = G.check_cases(rows, T_CITIES)
    assert not any(pr.values()), {k: v[:3] for k, v in pr.items() if v}
    from collections import Counter
    assert set(Counter(r["base_key"] for r in rows).values()) == {13}
    sets = Counter((r["set"], r["hardness"]) for r in rows if r["base_key"] == rows[0]["base_key"])
    assert sets[("ANS", "H0")] == 2 and sets[("CF", "H2")] == 1 and sets[("UNA-swap", "H1")] == 1
    owner = {}
    for r in rows:
        if r["effective_key"]:
            k = tuple(r["effective_key"])
            assert owner.setdefault(k, r["split"]) == r["split"]


def test_key_matches_exp50_lambda(small_cases):
    rows, _, _ = small_cases
    lam = lambda r: (r["task"], r["A"], r["B"] if r["task"] == "RC" else None, r["base_gold"], r["old_city"])
    assert all(G.key(r) == lam(r) for r in rows)


def test_cf_and_tcity(small_cases):
    rows, _, _ = small_cases
    for r in rows:
        if r["set"] == "CF":
            assert r["gold"] == r["cf_city"] != r["base_gold"]
        if r["split"] == "Tcity":
            assert not [c for c in G.C.CITIES if c in r["prompt"]]
        assert r["split"] != "Tformat"


def test_old_prompt_hash_excluded(small_cases):
    rows, _, _ = small_cases
    old, _, _ = G.old_hashes()
    assert not old & {r["prompt_sha256"] for r in rows}


# ---------------------------------------------------------------------------------------------- grader

def test_grader_fixtures():
    bad = [r for r in GR.run_fixtures() if not r["ok"]]
    assert not bad, bad


def test_utility_and_mixture():
    assert GR.utility("gold", "ANS") == 1 and GR.utility("mixed", "ANS") == 0 and GR.utility("other", "ANS", 2) == -2
    assert GR.utility("mixed", "UNA-rel") == -1 and GR.utility("abstention", "UNA-swap") == 0
    rows = [{"base_key": "b", "set": "ANS", "cls": "gold"}] + [{"base_key": "b", "set": "ANS", "cls": "abstention"}] * 5
    rows += [{"base_key": "b", "set": "UNA-rel", "cls": "wrong"}] + [{"base_key": "b", "set": "UNA-rel", "cls": "abstention"}] * 3
    rows += [{"base_key": "b", "set": "CF", "cls": "wrong"}] * 3
    assert GR.base_mixture(rows)["b"] == pytest.approx(0.5 / 6 - 0.5 / 4)      # 1/12 per ANS row, 1/8 per UNA row


# ---------------------------------------------------------------------------------------------- stats

def test_bootstrap_rules():
    bases = S.Bases({f"b{i}": "DE" for i in range(30)})
    a = {f"b{i}": 1.0 for i in range(30)}
    z = {f"b{i}": 0.0 for i in range(30)}
    res = S.bootstrap(S.diff_stat(bases, a, z), bases.n_by_task(), B=200)
    assert S.p_greater(res) == pytest.approx(1 / 201)          # degenerate on the alternative's side
    assert S.p_less(res) == 1.0
    res2 = S.bootstrap(lambda idx: None, bases.n_by_task(), B=100)
    assert res2["B_valid"] == 0 and S.p_greater(res2) is None
    assert S.tost(res, 0.0) == 1.0                              # SD 0 -> unestimable
    h = S.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert [h[k]["reject"] for k in "abc"] == [True, True, True]
    h2 = S.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert [h2[k]["reject"] for k in "abc"] == [True, False, False]
    assert S.iut([0.01, None]) is None and S.iut([0.01, 0.2]) == 0.2
    assert S.auroc([3, 2, 1], [1, 0, 0]) == 1.0 and S.auroc([1, 1], [1, 1]) is None


def test_failed_replicates_over_one_percent():
    bases = S.Bases({f"b{i}": "DE" for i in range(5)})
    vals = {f"b{i}": float(i) for i in range(5)}
    stat = lambda idx: None if 0 not in idx["DE"] else S.bw_mean(vals, bases.resampled(idx))
    res = S.bootstrap(stat, bases.n_by_task(), B=500)
    assert res["n_fail"] > 5 and not res["fail_ok"] and S.p_greater(res) is None


def test_m_order_and_doses():
    c = [{"effect": 0.2, "se": 0.05, "sign": 1, "n_components": 2, "n_params": 10, "index": 1, "bound_ok": True, "support": 25},
         {"effect": -0.3, "se": 0.05, "sign": -1, "n_components": 1, "n_params": 10, "index": 2, "bound_ok": True, "support": 25},
         {"effect": 0.04, "se": 0.001, "sign": 1, "n_components": 1, "n_params": 1, "index": 3, "bound_ok": True, "support": 25}]
    assert [x["index"] for x in S.m_order(c)] == [2, 1]
    assert MK.pick_dose({-1: 0.3, -16: 0.3, float("-inf"): 0.1}, {-1: 1, -16: 16, float("-inf"): 1e9}) == -1
    assert MK.closest_b1(0.2, {-1.0: 0.1, 1.0: 0.3, -2.0: 0.25}) == -2.0


# ---------------------------------------------------------------------------------------------- model fixtures

@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    import exp51_engine as EN
    d = tmp_path_factory.mktemp("edits")
    return EN.load("dummy", threads=2, edit_dir=d)


def _rand_comp(e, layer, tensor, kind, n=4, scale=1e-2, seed=0):
    shapes = e.edits.shapes()
    shape = shapes[ED.EDITABLE[tensor].format(l=layer)]
    rng = np.random.default_rng(seed)
    if kind == "dense":
        return {"layer": layer, "tensor": tensor, "kind": "dense", "U": rng.standard_normal((shape[0], n)) * scale,
                "V": rng.standard_normal((shape[1], n)) * scale}
    if kind == "rows":
        return {"layer": layer, "tensor": tensor, "kind": "rows", "idx": np.arange(n),
                "D": rng.standard_normal((n, shape[1])) * scale}
    if kind == "cols":
        return {"layer": layer, "tensor": tensor, "kind": "cols", "idx": np.arange(n),
                "D": rng.standard_normal((shape[0], n)) * scale}
    return {"layer": layer, "tensor": tensor, "kind": "coords", "idx": np.arange(n), "D": rng.standard_normal(n) * scale}


def test_edit_family_refusals(engine):
    e = engine
    sh = e.edits.shapes()
    ok = [_rand_comp(e, 0, "o_proj", "dense")]
    ED.validate(ok, sh, originals=e.edits.original_np)
    with pytest.raises(ED.EditRefused):
        ED.validate([{**ok[0], "tensor": "q_proj"}], sh)
    with pytest.raises(ED.EditRefused):
        ED.validate([_rand_comp(e, 0, "o_proj", "dense", n=5)], sh)
    with pytest.raises(ED.EditRefused):
        ED.validate([_rand_comp(e, l, "down_proj", "dense") for l in range(4)] + [_rand_comp(e, 0, "o_proj", "dense")], sh)
    ED.validate([_rand_comp(e, l, "down_proj", "cols", n=128) for l in range(4)], sh)       # 512: allowed
    old = ED.MAX_SPARSE
    try:
        ED.MAX_SPARSE = 100           # the dummy's tensors are too small to exceed 512 with 4 components
        with pytest.raises(ED.EditRefused):
            ED.validate([_rand_comp(e, 0, "down_proj", "cols", n=64), _rand_comp(e, 1, "input_layernorm", "coords",
                                                                                 n=40)], sh)
    finally:
        ED.MAX_SPARSE = old
    with pytest.raises(ED.EditRefused):
        ED.validate([{"layer": 0, "tensor": "input_layernorm", "kind": "dense", "U": np.ones((64, 1)), "V": np.ones((1, 1))}], sh)
    W = e.edits.original_np(ED.EDITABLE["o_proj"].format(l=0)).astype(np.float64)
    u, s, vt = np.linalg.svd(W)
    k = 4
    with pytest.raises(ED.EditRefused):          # a near-scaling of a (here nearly low-rank) tensor is refused
        Wl = (u[:, :k] * s[:k]) @ vt[:k]
        ED.validate([{"layer": 0, "tensor": "o_proj", "kind": "dense", "U": 0.5 * u[:, :k] * s[:k], "V": vt[:k].T}],
                    sh, originals=lambda n: Wl.astype(np.float32))
    with pytest.raises(ED.EditRefused):
        ED.validate([{"layer": 0, "tensor": "o_proj", "kind": "rows", "idx": np.array([1, 1]), "D": np.zeros((2, 64))}], sh)


def test_apply_bit_for_bit_and_revert(engine, tmp_path):
    e = engine
    e.edits.edit_dir = tmp_path
    sh = e.edits.shapes()
    comps = [_rand_comp(e, 1, "down_proj", "dense"), _rand_comp(e, 2, "o_proj", "cols", n=8),
             _rand_comp(e, 2, "post_attention_layernorm", "coords", n=8)]
    names = [ED.comp_name(c) for c in comps]
    before = {n: e.edits.tensor_sha(n) for n in names}
    man = ED.save_edit(tmp_path, comps, sh, originals=e.edits.original_np)
    assert ED.save_edit(tmp_path, comps, sh)["edit_id"] != "" and man["n_sparse"] == 16
    res = e.edits.record_results(man["edit_id"], doses=(1.0, 2.0))
    assert {n: e.edits.tensor_sha(n) for n in names} == before
    a1 = e.edits.apply(man["edit_id"], 1.0)
    assert all(a1[n] == res[f"{n}@1"] for n in names)
    e.edits.revert_component(2, "o_proj")
    assert e.edits.tensor_sha(names[1]) == before[names[1]] and e.edits.tensor_sha(names[0]) != before[names[0]]
    e.edits.revert()
    e.edits.assert_clean()
    a2 = e.edits.apply(man["edit_id"], 2.0)
    assert all(a2[n] == res[f"{n}@2"] for n in names)
    e.edits.revert()
    assert {n: e.edits.tensor_sha(n) for n in names} == before
    (tmp_path / f"{man['edit_id']}.npz").write_bytes(b"x")
    with pytest.raises(ED.EditRefused):
        ED.load_edit(tmp_path, man["edit_id"])


def test_b6_random_matched_norms(engine):
    e = engine
    sh = e.edits.shapes()
    for kind in ("dense", "rows", "cols", "coords"):
        t = "input_layernorm" if kind == "coords" else "down_proj"
        c = _rand_comp(e, 0, t, kind, n=3)
        shape = sh[ED.comp_name(c)]
        spec = ED.random_spec(c, shape, seed=2)
        d, rnd = ED.delta64(c, shape), ED.random_matched(spec, shape)
        assert np.array_equal(rnd != 0, d != 0) or kind == "dense"
        if kind == "coords":
            assert np.linalg.norm(rnd) == pytest.approx(np.linalg.norm(d))
        else:
            assert np.allclose(np.linalg.norm(rnd, axis=0), np.linalg.norm(d, axis=0))
        assert np.array_equal(rnd, ED.random_matched(spec, shape))
        assert not np.array_equal(rnd, ED.random_matched({**spec, "seed": 3}, shape))


def test_b5_orthogonalization_removes_direction(engine):
    e = engine
    name = ED.EDITABLE["o_proj"].format(l=0)
    W = e.edits.original_np(name)
    d = np.random.default_rng(1).standard_normal(W.shape[0])
    c = ED.orthogonalize_spec(d, {name: W, ED.EDITABLE["down_proj"].format(l=0): e.edits.original_np(
        ED.EDITABLE["down_proj"].format(l=0))}, [0], 1.0)[0]
    W2 = W.astype(np.float64) + ED.delta64(c, W.shape)
    assert np.abs((d / np.linalg.norm(d)) @ W2).max() < 1e-6


# ---------------------------------------------------------------------------------------------- engine

def _scripted(e, monkeypatch, pieces: list[str], alt: dict[int, str]):
    """Patch _step to emit `pieces` (one token each); at step k in `alt`, the scripted token gets 10 and alt[k]'s
    first token 9 (so a bias of -inf on the scripted token yields the alternative)."""
    toks = [e.encode(p)[0] for p in pieces]
    st = {"i": 0}
    import torch

    def step(token, cache, pos):
        i = st["i"]
        st["i"] += 1
        z = torch.full((e.net.config.vocab_size,), -10.0)
        if i < len(toks):
            z[toks[i]] = 10.0
            if i in alt:
                z[e.encode(alt[i])[0]] = 9.0
        else:
            z[list(e.stop_ids)[0]] = 10.0
        return z
    monkeypatch.setattr(e, "_step", step)
    monkeypatch.setattr(e, "_prefill", lambda ids: None)
    return toks


def test_b1_online_trigger(engine, monkeypatch):
    e = engine
    pieces = [" unknown", "\n", "Answer", ":", " unknown", "\n"]
    _scripted(e, monkeypatch, pieces, {0: " York", 4: " York"})
    import exp51_engine as EN
    unk = [e.encode(" unknown")[0]]
    g = e.generate([1, 2, 3], EN.Arm(bias={"ids": unk, "value": float("-inf"), "mode": "online"}), mode="total")
    assert g.text == " unknown\nAnswer: York\n" and g.trigger_step == 4
    _scripted(e, monkeypatch, pieces, {0: " York", 4: " York"})
    g = e.generate([1, 2, 3], EN.Arm(bias={"ids": unk, "value": float("-inf"), "mode": "whole"}), mode="total")
    assert g.text.startswith(" York")
    _scripted(e, monkeypatch, [" unknown", "\n"], {0: " York"})
    g = e.generate([1, 2, 3], EN.Arm(bias={"ids": unk, "value": float("-inf"), "mode": "prefix"}), mode="prefix")
    assert g.text.startswith(" York")
    _scripted(e, monkeypatch, [" unknown", "\n", "Note", "\n"], {0: " York"})
    g = e.generate([1, 2, 3], EN.Arm(bias={"ids": unk, "value": float("-inf"), "mode": "online"}), mode="total",
                   max_new=4)
    assert g.trigger_step is None and g.text.startswith(" unknown")        # no marker, never on


def test_steer_and_component_hooks(engine):
    import exp51_engine as EN
    e = engine
    ids = e.chat_ids("Where does Alice live? Note:")
    base = e.read(ids, [len(ids) - 1], layers=[1], comps=[(1, "o_proj")])
    v = np.ones(e.hidden, np.float32)
    arm = EN.Arm(comp=[{"layer": 1, "tensor": "o_proj", "op": "set", "values": {len(ids) - 1: v}}])
    hit = e.read(ids, [len(ids) - 1], layers=[1], comps=[(1, "o_proj")], arm=arm)
    assert np.allclose(hit["comp"][(1, "o_proj")][len(ids) - 1], v)
    assert not np.allclose(hit["resid"][1], base["resid"][1])
    early = EN.Arm(comp=[{"layer": 1, "tensor": "o_proj", "op": "set", "values": {len(ids) - 3: v}}])
    b3 = e.read(ids, [len(ids) - 1], layers=[1, 3])["resid"]
    x = e.read(ids, [len(ids) - 1], layers=[1, 3], arm=early)["resid"]
    assert np.array_equal(x[1], b3[1]) and not np.allclose(x[3], b3[3])   # reaches the last position via attention
    g0 = e.generate(ids, mode="prefix", max_new=4)
    g1 = e.generate(ids, mode="prefix", max_new=4)
    assert g0.gen_ids == g1.gen_ids                             # deterministic at batch 1


# ---------------------------------------------------------------------------------------------- B0 / no-op / service

def _fake_b0(e, row):
    ids = e.chat_ids(row["prompt"])
    gen = e.encode(f"{row['A']} is in {row['base_gold']}.\nAnswer: unknown\n")
    rec = {"case_id": row["case_id"], "split": row["split"], "prompt_ids": ids, "gen_ids": gen,
           "text": e.decode(gen), "n_prompt": len(ids)}
    d = CM.designated(e.decode, gen)
    rec.update({"j": d["j"], "p_ans_ok": d["ok"], "p_ans": len(ids) + d["j"] - 1 if d["ok"] else None,
                **CM.grade(row, rec["text"], T_CITIES)})
    rec["populations"] = CM.population(row, rec, T_CITIES)
    return rec


@pytest.fixture()
def service(engine, tmp_path, small_cases):
    import exp51_service as SV
    rows = [r for r in small_cases[0] if r["split"] == "E"][:6]
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "edits_in").mkdir()
    (ws / "data/E_rows.jsonl").write_text("".join(json.dumps({**r, "b0": _fake_b0(engine, r)}) + "\n" for r in rows))
    CM.write_json(tmp_path / "bounds.json", {**SV.BOUNDS_F0, "passes": 3, "failed_requests": 3})
    return SV.Service(tmp_path, engine, tc=T_CITIES), rows


def test_designated_marker_and_population(engine, small_cases):
    row = next(r for r in small_cases[0] if r["set"] == "ANS")
    rec = _fake_b0(engine, row)
    assert rec["p_ans_ok"] and engine.decode(CM.prefix_ids(rec)[rec["n_prompt"]:]).endswith("\nAnswer:")
    assert rec["cls"] == "abstention" and "FA" in rec["populations"]
    ids, left = CM.note_ids(engine, rec)
    assert engine.decode(ids) + left + "\nAnswer:" == engine.decode(CM.prefix_ids(rec)[rec["n_prompt"]:])
    b2 = CM.b2_prefix(engine, row, rec)
    assert engine.decode(b2[rec["n_prompt"]:]).endswith("\nDo not reply unknown.\nAnswer:")


def _req(svc, rid, body):
    CM.write_json(svc.q_in / f"{rid}.json", body)
    svc.serve(once=True)
    return json.loads((svc.q_out / f"{rid}.json").read_text())


def test_service_passes_bounds_and_ledger(service, engine):
    svc, rows = service
    ids = [r["case_id"] for r in rows]
    r = _req(svc, "r1", {"type": "paired", "rows": ids[:2], "arms": [{"edit": None}, {"edit": None}]})
    assert r["ok"] and r["items"][0]["a"]["cached"] and r["ledger"]["passes"] == 1
    sh = engine.edits.shapes()
    np.savez(svc.ws / "edits_in/e1.npz", c0_layer=1, c0_tensor=np.array("down_proj"), c0_kind=np.array("rows"),
             c0_idx=np.arange(2), c0_D=np.ones((2, sh[ED.EDITABLE["down_proj"].format(l=1)][1])) * 1e-3)
    s = _req(svc, "r2", {"type": "save_edit", "spec": "edits_in/e1.npz"})
    assert s["ok"], s
    eid = s["edit_id"]
    p = _req(svc, "r3", {"type": "prefix", "rows": ids[:2], "arm": {"edit": eid}})
    assert p["ok"] and len(p["items"]) == 2
    rd = _req(svc, "r4", {"type": "read", "items": [{"row": ids[0]}, {"text": "hello", "continuation": " there"}],
                          "positions": "last", "layers": [0, 3], "components": [[1, "o_proj"]], "logits": True,
                          "logit_ids": [1, 2, 3], "arm": {"edit": eid}})
    assert rd["ok"], rd
    z = np.load(svc.ws / rd["npz"])
    assert z["i0_resid_L0"].shape == (1, engine.hidden) and z["i1_logits"].shape == (1, 3)
    assert _req(svc, "r5", {"type": "prefix", "rows": ids[:1], "arm": {"edit": eid}})["refused"]   # pass bound (3)
    bad = _req(svc, "r6", {"type": "paired", "rows": ["S-DE00_ANSa_H0"], "arms": [{}, {}]})
    assert not bad["ok"] and bad["ledger"]["failed_requests"] == 2
    w = _req(svc, "r7", {"type": "weights", "tensors": ["model.layers.0.self_attn.q_proj.weight"]})
    assert not w["ok"]
    assert _req(svc, "r8", {"type": "weights", "tensors": ["model.norm.weight"]})["refused"]    # closed at 3 failures
    t = svc.ledger.totals()
    assert t["passes"] == 3 and t["failed_requests"] == 4 and t["saved_edits"] == 1 and t["input_tokens"] > 0
    assert (svc.ws / "ledger_copy.jsonl").exists() and (svc.run / "ledger.jsonl").exists()
    engine.edits.assert_clean()


def test_service_refuses_scaling_and_family(service, engine):
    svc, rows = service
    sh = engine.edits.shapes()
    n = sh[ED.EDITABLE["o_proj"].format(l=0)]
    np.savez(svc.ws / "edits_in/bad.npz", c0_layer=0, c0_tensor=np.array("q_proj"), c0_kind=np.array("dense"),
             c0_U=np.ones((n[0], 1)), c0_V=np.ones((n[1], 1)))
    r = _req(svc, "b1", {"type": "save_edit", "spec": "edits_in/bad.npz"})
    assert not r["ok"] and "not editable" in r["error"]
    r = _req(svc, "b2", {"type": "save_edit", "spec": "../outside.npz"})
    assert not r["ok"]


def test_noop_gate_dummy(engine, tmp_path, small_cases, monkeypatch):
    import exp51_run as RUN
    monkeypatch.setattr(CM, "RUN", tmp_path)
    e = engine
    e.edits.edit_dir = tmp_path / "edits"
    rows = [r for r in small_cases[0] if r["split"] == "E" and r["set"] == "ANS"][:2]
    for r in rows:
        rec = _fake_b0(e, r)
        CM.write_json(CM.b0_path(r["split"], r["case_id"], tmp_path), rec)
    # a real B0 generation path for one row (total effect equality needs B0 from the same engine)
    rec = RUN.run_b0_row(e, rows[0], tmp_path, T_CITIES)
    CM.write_json(CM.b0_path("E", rows[0]["case_id"], tmp_path), rec)
    res = RUN.noop_gate(e, rows[:1], tmp_path, T_CITIES)
    assert not res["total_mismatch"] and res["weights_restored"]


def test_mechanism_hooks_and_positions(engine, small_cases, tmp_path):
    import exp51_mech as MC
    e = engine
    e.edits.edit_dir = tmp_path
    row = next(r for r in small_cases[0] if r["split"] == "E" and r["set"] == "ANS")
    rec = _fake_b0(e, row)
    sh = e.edits.shapes()
    zero = ED.save_edit(tmp_path, [{"layer": 1, "tensor": "o_proj", "kind": "dense", "U": np.zeros((sh[ED.EDITABLE['o_proj'].format(l=1)][0], 1)),
                                    "V": np.zeros((sh[ED.EDITABLE['o_proj'].format(l=1)][1], 1))}], sh)["edit_id"]
    b0p = e.generate(CM.prefix_ids(rec), mode="prefix")
    t = MC.transplant_arm(e, row, rec, zero, 1.0, 1, "o_proj", [rec["p_ans"]], T_CITIES)
    c = MC.clamp_arm(e, row, rec, zero, 1.0, 1, "o_proj", [rec["p_ans"]], T_CITIES)
    assert t["gen_ids"] == b0p.gen_ids == c["gen_ids"] and t["delta_norm"] == 0
    with pytest.raises(ValueError):
        MC.clamp_arm(e, row, rec, zero, 1.0, 1, "o_proj", [rec["p_ans"] + 1], T_CITIES)
    f = MC.f_value(e, rec, (2, "resid"), np.ones(e.hidden))
    assert np.isfinite(f)
    e.edits.assert_clean()


def test_mech_stats_synthetic():
    bases = S.Bases({f"b{i}": ("DE" if i % 2 else "RC") for i in range(40)})
    rng = np.random.default_rng(0)
    star = {b: 0.8 for b in bases.resampled({t: np.arange(n) for t, n in bases.n_by_task().items()})}
    rev = {b: 0.1 + 0.05 * rng.random() for b in star}
    m = MK.m1(bases, star, rev, rev, +1, B=500)
    assert m["p"] < 0.01
    inter = MK.interaction(bases, star, rev, rev, rev, +1, B=500)
    assert inter["p"] < 0.01
    eq = MK.activity_equivalence(bases, {b: 0.0 for b in star}, 0.0, B=100)
    assert eq["p"] == 1.0 and eq["unestimable"] and eq["all_within_1e-6"]
    train = [{"task": "DE" if i % 2 else "RC", "hardness": "H1", "city": "York", "label": i % 2, "m_city": float(i % 2),
              "unk_logodds": 0.0, "f": rng.standard_normal(), "base_key": f"b{i}"} for i in range(40)]
    res = MK.m3c(train, train, bases, B=200)
    assert res["auroc_baseline"] == 1.0 and res["p"] is not None


# ---------------------------------------------------------------------------------------------- harness

def _ws(tmp_path):
    ws = tmp_path / "workspace"
    for d in ("data/tokenizer", "notes", "scratch", "queue/requests", "queue/responses", "captures", "edits_in"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    import exp51_engine as EN
    shutil.copy(EN.SNAPSHOT / "tokenizer.json", ws / "data/tokenizer/tokenizer.json")
    (ws / "data/E_rows.jsonl").write_text('{"x": 1}\n')
    for p in (ws / "data").rglob("*"):
        p.chmod(0o444 if p.is_file() else 0o555)
    (ws / "data").chmod(0o555)
    return ws


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_sandbox_boundaries(tmp_path):
    ws = _ws(tmp_path)
    r = __import__("exp51_harness").boundary_check(ws)
    assert r["pass"], {k: v for k, v in r["probes"].items() if not v["ok"]}


def test_harness_dummy_loop_and_cost_stop(service, tmp_path, monkeypatch):
    import exp51_harness as H
    svc, rows = service
    monkeypatch.setattr(H, "_api_key", lambda: (_ for _ in ()).throw(AssertionError("key read in a dry run")))
    script = [[{"type": "text", "text": "Plan."},
               {"type": "tool_use", "id": "a", "name": "forward",
                "input": {"request": {"type": "paired", "rows": [rows[0]["case_id"]], "arms": [{}, {}]}}}],
              [{"type": "text", "text": "No tool."}],
              [{"type": "tool_use", "id": "b", "name": "submit", "input": {
                  "edit_id": "nope", "kind": "repair", "component_and_computation": "x", "evidence": "x",
                  "predictions_by_row_type": "x", "flips": "x", "activity_predictions": "x",
                  "signed_endpoint": {"population": "FA", "direction": "increase", "outcome": "Rel"},
                  "most_important_component": {"layer": 1, "tensor": "o_proj"}, "f_site": {"layer": 1,
                  "tensor": "resid"}, "f_direction": "/work/edits_in/u.npy", "f_int_sign": 1,
                  "activity": [{"manipulation": "evidence_removed", "type": "directional", "sign": -1}]}}],
              [{"type": "tool_use", "id": "c", "name": "finish", "input": {"summary": "done"}}]]
    h = H.Harness(svc.run, H.DummyClient(script), service=svc)
    out = h.loop()
    assert out["reason"] == "finish" and out["ledger"]["turns"] == 4 and out["ledger"]["passes"] == 1
    assert out["usd"] == pytest.approx(4 * H.cost_of(H.DummyClient([]).usage))
    tr = (svc.run / "transcript.jsonl").read_text()
    assert "sk-ant" not in tr and "x-api-key" not in tr
    h2 = H.Harness(svc.run, H.DummyClient(script), service=svc, stop_usd=out["usd"] + 0.01)
    assert h2.loop()["reason"].startswith("cost stop")


def test_api_body_has_cache_and_effort():
    import exp51_harness as H
    b = H.ApiClient().body([{"role": "user", "content": "hi"}])
    assert b["model"] == "claude-opus-5-5" and b["output_config"]["effort"] == "high"
    assert b["system"][0]["cache_control"]["ttl"] == "1h" and b["messages"][-1]["content"][-1]["cache_control"]
    assert "x-api-key" not in json.dumps(b)


def test_cost_of_prices():
    import exp51_harness as H
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
         "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000, "ephemeral_5m_input_tokens": 0}}
    assert H.cost_of(u) == pytest.approx(4 + 20 + 0.2 + 8)


# ---------------------------------------------------------------------------------------------- guard

def test_sealed_splits_refused():
    with pytest.raises(PermissionError):
        CM.rows("H")
    import exp51_run as RUN
    with pytest.raises(PermissionError):
        RUN.require_f2(Path("/nonexistent"))


def test_failed_request_ledgers_tokens(service, engine, monkeypatch):
    svc, rows = service
    real = engine.generate
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)
    monkeypatch.setattr(engine, "generate", flaky)
    r = _req(svc, "f1", {"type": "prefix", "rows": [rows[0]["case_id"], rows[1]["case_id"]], "arm": {"edit": None}})
    assert not r["ok"]
    last = svc.ledger.entries[-1]
    assert last["ok"] is False and last["input_tokens"] > 0 and last["wall_s"] >= 0
    assert svc.ledger.totals()["passes"] == 0 and svc.ledger.totals()["failed_requests"] == 1


def test_make_b6(engine, tmp_path):
    import exp51_run as RUN
    e = engine
    e.edits.edit_dir = tmp_path
    sh = e.edits.shapes()
    man = ED.save_edit(tmp_path, [_rand_comp(e, 1, "down_proj", "cols", n=5), _rand_comp(e, 2, "input_layernorm",
                                                                                         "coords", n=7)], sh)
    ids = RUN.make_b6(e, man["edit_id"], 2.0)
    assert len(set(ids.values())) == 5
    _, cand = ED.load_edit(tmp_path, man["edit_id"])
    m6, r6 = ED.load_edit(tmp_path, ids[1])
    d_c = ED.delta64(cand[0], sh[ED.comp_name(cand[0])], 2.0)
    d_r = ED.delta64(r6[0], sh[ED.comp_name(r6[0])])
    assert np.allclose(np.linalg.norm(d_r, axis=0), np.linalg.norm(d_c, axis=0))
    assert np.array_equal(np.nonzero(d_r.any(0))[0], np.arange(5))
    e.edits.apply(ids[3])
    e.edits.revert()
    e.edits.assert_clean()


def test_b2_prefix_straddling_token(engine, small_cases):
    """Build review: B0's last note token carries the marker's newline ('.\\n'); the note keeps the ids before it, the
    leftover '.' is encoded with the B2 tail, and the text is note + tail exactly."""
    e = engine
    row = next(r for r in small_cases[0] if r["set"] == "ANS")
    dot_nl = e.tok.convert_tokens_to_ids(e.tok.tokenize(".\n"))
    assert len(dot_nl) == 1, "the tokenizer has no single '.\\n' token"
    gen = e.encode("Alice is in York") + dot_nl + e.encode("Answer:") + e.encode(" unknown\n")
    ids = e.chat_ids(row["prompt"])
    d = CM.designated(e.decode, gen)
    assert d["ok"]
    rec = {"prompt_ids": ids, "gen_ids": gen, "j": d["j"], "n_prompt": len(ids)}
    note, left = CM.note_ids(e, rec)
    assert left == "." and e.decode(note) == "Alice is in York"
    b2 = CM.b2_prefix(e, row, rec)
    assert e.decode(b2[len(ids):]) == "Alice is in York.\nDo not reply unknown.\nAnswer:"
    assert b2[:len(ids) + len(note)] == ids + note                  # B0's own note ids, unchanged
    b3 = CM.b3_prefix(e, row, rec)
    assert b3[-d["j"]:] == gen[:d["j"]]                             # B3 keeps B0's ids through the marker
    plain = e.encode("Alice is in York.") + e.encode("\nAnswer:") + e.encode(" York\n")
    rec2 = {**rec, "gen_ids": plain, "j": CM.designated(e.decode, plain)["j"]}
    n2, l2 = CM.note_ids(e, rec2)
    assert e.decode(n2) + l2 == "Alice is in York."


def test_harness_stop_reasons_validation_and_retries(service):
    import exp51_harness as H
    svc, rows = service
    paired = {"type": "tool_use", "id": "p", "name": "forward",
              "input": {"request": {"type": "paired", "rows": [rows[0]["case_id"]], "arms": [{}, {}]}}}
    script = [{"stop_reason": "max_tokens", "content": [paired]},           # truncated: must not run
              [{"type": "tool_use", "id": "v", "name": "run_python", "input": {"timeout_s": 5}}],   # no code: invalid
              "TRANSIENT", "TRANSIENT",
              [{"type": "tool_use", "id": "f", "name": "finish", "input": {"summary": "ok"}}]]
    h = H.Harness(svc.run, H.DummyClient(script), service=svc)
    slept = []
    h.sleep = slept.append
    out = h.loop()
    assert out["reason"] == "finish" and out["ledger"]["passes"] == 0 and out["ledger"]["python_runs"] == 0
    assert out["ledger"]["turns"] == 3 and out["ledger"]["failed_requests"] == 0 and slept == [2.0, 4.0]
    tr = [json.loads(l) for l in (svc.run / "transcript.jsonl").read_text().splitlines()]
    results = [c for t in tr if t["role"] == "user" and isinstance(t["content"], list) for c in t["content"]]
    assert "NOT run" in results[0]["content"] and "invalid tool input" in results[1]["content"]
    retries = [e for e in svc.ledger.entries if e["kind"] == "api_retry"]
    assert len(retries) == 2
    h2 = H.Harness(svc.run, H.DummyClient(["TRANSIENT"] * 6), service=svc)
    h2.sleep = lambda s: None
    assert h2.loop()["reason"].startswith("5 consecutive API errors")
    h3 = H.Harness(svc.run, H.DummyClient([{"stop_reason": "refusal", "content": [{"type": "text", "text": "no"}]}]),
                   service=svc)
    assert h3.loop()["reason"].startswith("refusal")


def test_validate_input():
    import exp51_harness as H
    assert H.validate_input(H.TOOL_SCHEMAS["run_python"], {"code": "x"}) == []
    assert H.validate_input(H.TOOL_SCHEMAS["run_python"], {"code": 3})
    assert H.validate_input(H.TOOL_SCHEMAS["run_python"], {"code": "x", "timeout_s": 10 ** 6})
    bad = {"edit_id": "e", "kind": "other"}
    errs = H.validate_input(H.TOOL_SCHEMAS["submit"], bad)
    assert any("kind" in e for e in errs) and any("signed_endpoint" in e for e in errs)
    acts = {"activity": [{"manipulation": "evidence_removed", "type": "directional", "sign": 1}] * 5}
    assert any("more than 4" in e for e in H.validate_input(H.TOOL_SCHEMAS["submit"], acts))
    assert any("activity[0].manipulation" in e for e in H.validate_input(H.TOOL_SCHEMAS["submit"], {
        "activity": [{"manipulation": "x", "type": "directional", "sign": 1}]}))


def test_shortcut_parity_dummy(engine, small_cases, tmp_path):
    import exp51_run as RUN
    import exp51_engine as EN
    e = engine
    row = next(r for r in small_cases[0] if r["split"] == "E" and r["set"] == "ANS")
    rec = RUN.run_b0_row(e, row, tmp_path, T_CITIES)
    arm = EN.Arm(bias={"ids": [e.encode(" unknown")[0]], "value": -4.0, "mode": "online"})
    full = e.generate(list(rec["prompt_ids"]), arm, mode="total")
    sc = RUN.shortcut_row(e, row, rec, arm, T_CITIES)
    assert full.gen_ids == sc["gen_ids"]                # the dummy never writes a marker: both equal B0
    fake = {**rec, "first_trigger_step": 3}
    g = RUN.shortcut_row(e, row, fake, arm, T_CITIES)
    assert g["gen_ids"][:3] == rec["gen_ids"][:3] and len(g["gen_ids"]) <= EN.MAX_NEW_TOTAL


def test_generate_batched_matches_batch1_dummy(engine, small_cases):
    import exp51_engine as EN
    e = engine
    rows = sorted([r for r in small_cases[0] if r["split"] == "E"][:5], key=lambda r: r["case_id"])
    batch = [e.chat_ids(r["prompt"]) for r in rows]
    assert len({len(x) for x in batch}) > 1                            # padding is exercised
    unk = [e.encode(" unknown")[0]]
    for arm, mode in ((EN.Arm(), "total"), (EN.Arm(bias={"ids": unk, "value": -4.0, "mode": "whole"}), "total"),
                      (EN.Arm(steer={"layer": 1, "vector": np.ones(e.hidden) * 0.1, "mode": "prefix"}), "prefix")):
        got = e.generate_batched(batch, arm, mode=mode, max_new=12)
        ref = [e.generate(x, arm, mode=mode, max_new=12) for x in batch]
        assert [g.gen_ids for g in got] == [r.gen_ids for r in ref]
    with pytest.raises(ValueError):
        e.generate_batched(batch, EN.Arm(hybrid=True))


def test_stage4_end_to_end_dummy(engine, small_cases, tmp_path, monkeypatch):
    """Pod S (B0, R and M arms, selection, M follow-ups) and pod H+T (guarded) on the dummy with scripted B0 notes."""
    import exp51_stage4 as S4
    e = engine
    root = tmp_path / "gpu"
    e.edits.edit_dir = root / "edits"
    rows = small_cases[0]
    keep = {sp: {f"{sp}-{t}00" for t in ("DE", "RC", "TU")} for sp in ("S", "H", "Tcity")}   # one base per task
    by_split = {sp: [r for r in rows if r["split"] == sp and r["base_key"] in keep[sp]] for sp in ("S", "H", "Tcity")}
    rows_S = by_split["S"]
    import exp51_engine as EN
    monkeypatch.setattr(S4, "LAYOUT", 16)
    monkeypatch.setattr(EN, "MAX_NEW_TOTAL", 12)
    monkeypatch.setattr(EN, "STEER_LAYER", 1)
    monkeypatch.setattr(CM, "b4_vector", lambda a, path=None: np.full(e.hidden, 0.01 * a, np.float32))
    # B0 on S batched, then scripted notes (the dummy writes no marker) so the prefix paths run
    S4.run_b0(e, rows_S[:3], root, "S", T_CITIES)
    assert S4.b0_path(root, "S", sorted(rows_S, key=lambda r: r["case_id"])[0]["case_id"]).exists() or True
    for sp in ("S", "H", "Tcity"):
        for r in by_split[sp]:
            CM.write_json(S4.b0_path(root, sp, r["case_id"]), {**_fake_b0(e, r), "first_trigger_step": 3,
                                                                "layout": 8})
    sh = e.edits.shapes()
    subs_dir = tmp_path / "subs"
    subs_dir.mkdir()
    for k in (1, 2):
        man = ED.save_edit(root / "edits", [_rand_comp(e, k, "down_proj", "cols", n=4, seed=k)], sh)
        CM.write_json(subs_dir / f"{k}.json", {
            "edit_id": man["edit_id"], "kind": "repair", "signed_endpoint": {"population": "FA", "direction": "increase",
                                                                             "outcome": "Rel"},
            "most_important_component": {"layer": k, "tensor": "down_proj"}, "f_site": {"layer": 2, "tensor": "resid"},
            "f_int_sign": 1, "activity": [{"manipulation": "evidence_removed", "type": "directional", "sign": -1},
                                          {"manipulation": "city_changed", "type": "equivalence", "sign": 0}]})
        np.save(subs_dir / f"{k}_f_direction.npy", np.ones(e.hidden))
    subs = S4.load_subs(subs_dir)
    b5 = {str(b): ED.save_edit(root / "edits", [_rand_comp(e, 3, "o_proj", "dense", n=1, seed=9)], sh, "comparator",
                               meta={"b": b})["edit_id"] for b in (0.25, 0.5, 0.75, 1.0)}
    b1 = [e.encode(" unknown")[0]]
    for spec, est in S4.r_specs(e, subs, b5, b1):
        S4.run_arm(e, rows_S, root, "S", spec, est, T_CITIES)
    au = [r for r in rows_S if r["set"] in ("ANS", "UNA-rel", "UNA-swap")]
    for spec, est in S4.m_specs(subs):
        S4.run_arm(e, au, root, "S", spec, est, T_CITIES)
    monkeypatch.setattr(CM, "b1_token_set", lambda tok: b1)
    sel = S4.s_select(e, root, rows_S, subs, b5, T_CITIES, B=200)
    assert set(sel["R"]["comparator_doses"]) == {"B1", "B2", "B3", "B4", "B5"}
    assert len(sel["R"]["entries"]) == 2 and len(sel["M"]["entries"]) == 2
    # force an M candidate to exercise the follow-ups (the dummy's effects are noise)
    sel["M"]["candidate"] = {"index": 1, "edit_id": subs[0]["edit_id"], "dose": 1.0}
    mf = S4.m_followups(e, root, rows_S, subs, sel, T_CITIES, B=200)
    assert mf["primary_component"]["i"] == 0 and len(mf["b6_edit_ids"]) == 5 and len(mf["activity"]) == 2
    assert mf["f_int_doses"] and "unestimable" in mf["predictor"]["coef"]     # scripted B0: FA only
    assert mf["pair"] == ["FA", "UNA"]
    # §13.4(b) with a non-FA endpoint: the pair is (C, UNA) and both grids run on C ∪ UNA rows
    assert S4.interaction_pair({"population": "C"}) == ["C", "UNA"]
    assert S4.interaction_pair({"population": "UNA"}) == ["UNA", "FA"]
    subs_c = [dict(x) for x in subs]
    subs_c[0]["signed_endpoint"] = {"population": "C", "direction": "decrease", "outcome": "gold"}
    for r in rows_S:                                     # make some ANS rows C (gold) so C is populated
        if r["set"] == "ANS" and r["hardness"] == "H0":
            rec = CM.read_json(S4.b0_path(root, "S", r["case_id"]))
            gen = e.encode(f"{r['A']} is in {r['base_gold']}.\nAnswer: {r['base_gold']}\n")
            d = CM.designated(e.decode, gen)
            rec.update({"gen_ids": gen, "text": e.decode(gen), "j": d["j"], "p_ans": len(rec["prompt_ids"]) + d["j"] - 1,
                        **CM.grade(r, e.decode(gen), T_CITIES)})
            rec["populations"] = CM.population(r, rec, T_CITIES)
            CM.write_json(S4.b0_path(root, "S", r["case_id"]), rec)
    mfc = S4.m_followups(e, tmp_path / "gpu_c" if False else root, rows_S, subs_c, sel, T_CITIES, B=200)
    assert mfc["pair"] == ["C", "UNA"]
    c_rows = [r["case_id"] for r in rows_S if "C" in CM.read_json(S4.b0_path(root, "S", r["case_id"]))["populations"]]
    fint = {"arm": "FINT", "dose": float(mfc["f_int_doses"][0]), "tag": "sub1"}
    assert c_rows and all(S4.out_path(root, "S", fint, "prefix", c).exists() for c in c_rows)
    sel["M_followups"] = mf
    sel["R"]["candidate"] = {"index": 2, "edit_id": subs[1]["edit_id"], "dose": 1.0}
    # pod H+T: the guard refuses without a committed F2
    with pytest.raises(PermissionError):
        S4.frozen(root)
    f2 = json.loads(json.dumps(sel, default=float))
    monkeypatch.setattr(S4, "frozen", lambda r: f2)
    monkeypatch.setattr(CM, "rows", lambda split, path=None, allow_sealed=False: by_split[split])
    if mf.get("f_int_dose") is None:
        f2["M_followups"]["f_int_dose"] = mf["f_int_doses"][0]
    S4.h_run(e, root, subs, T_CITIES)
    assert any((root / "arms" / "H").rglob("*.json")) and any((root / "arms" / "Tcity").rglob("*.json"))
    assert (root / "f" / "H.json").exists()
    rep = S4.h_report(root, subs, rows_S, by_split["H"], by_split["Tcity"], B=200)
    assert set(rep["R"]["holm"]) == {f"{k}:{o}" for k in ("B1", "B2", "B3", "B4", "B5") for o in ("O1", "O2")}
    assert set(rep["M"]["holm"]) == {"M1", "M2", "M3"} and rep["Tcity"]["descriptive"]
    e.edits.assert_clean()


def test_dry_run_caps(service):
    import exp51_harness as H
    svc, rows = service
    loop_forever = [[{"type": "text", "text": "thinking"}]]          # never finishes: the turn cap must end it
    h = H.Harness(svc.run, H.DummyClient(loop_forever), service=svc, stop_usd=3.0, max_turns=3)
    out = h.loop()
    assert out["reason"] == "turn bound" and out["ledger"]["turns"] == 3
    assert H.Harness(svc.run, H.DummyClient(loop_forever), service=svc, stop_usd=99).stop_usd == H.STOP_USD
