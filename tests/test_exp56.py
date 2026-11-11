"""§56 build checks on the tiny dummy Qwen2 (runs/exp51/dummy_model): the per-layer neuron DLA sums to the layer's MLP
DLA, D equals its linear form and its full decomposition, and the ablation acts at P_ans only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

EN = pytest.importorskip("exp51_engine")
import exp56_common as X  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    if not (EN.SNAPSHOT / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer not on this machine")
    return EN.load("dummy", threads=2)


def fake_rec(e):
    p = e.chat_ids("Below is your memory. Paul lives in Leeds. Where does Paul live?")
    g = e.encode("user mentioned that Paul lives in Leeds.\nAnswer:")
    return {"prompt_ids": p, "gen_ids": g + e.encode(" Leeds"), "j": len(g), "p_ans": len(p) + len(g) - 1}


def test_layer_sum_and_decomposition(eng):
    rd = X.Reader(eng, layers=range(eng.n_layers))
    rec = fake_rec(eng)
    cities = [eng.encode(" Leeds")[0], eng.encode(" York")[0]]
    res = rd.read(X.CM.prefix_ids(rec), cities)
    assert res["t_star"] in cities
    assert np.allclose(res["dla"].astype(np.float64).sum(1), res["mlp_dla"], atol=1e-4)
    assert abs(res["D"] - res["D_lin"]) < 1e-3
    dec = res["emb_dla"] + res["attn_dla"].sum() + res["mlp_dla"].sum()
    assert abs(dec - res["D_lin"]) < 1e-3
    assert res["t_star_rev0"] not in rd.unk_set and rd.unk_id == eng.encode(" unknown")[0]


def test_ablation_no_op_and_p_ans_only(eng):
    rec = fake_rec(eng)
    rd = X.Reader(eng, layers=range(eng.n_layers))
    res = rd.read(X.CM.prefix_ids(rec), [eng.encode(" Leeds")[0]])
    l = eng.n_layers - 1
    idx = np.arange(5)
    same = {l: (idx, res["a"][l][idx])}                       # set to its own value: nothing changes
    g0 = eng.generate(X.CM.prefix_ids(rec), mode="prefix")
    g1 = X.ablated_generate(eng, rec, same)
    assert g0.gen_ids == g1.gen_ids
    fired = []
    hs = X.install_ablation(eng, rec["p_ans"], {l: (idx, np.zeros(5, np.float32))})
    hs.append(eng.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: fired.append((eng._offset, a[0].shape[1]))))
    try:
        eng.generate(X.CM.prefix_ids(rec), mode="prefix", max_new=3)
    finally:
        for h in hs:
            h.remove()
    hits = [(lo, T) for lo, T in fired if lo <= rec["p_ans"] < lo + T]
    assert len(hits) == 1 and hits[0] == (rec["p_ans"], 1)    # the prefill stops before P_ans; step 0 is P_ans


def test_city_ids_single_token_main_pool():
    from transformers import AutoTokenizer
    if not (EN.SNAPSHOT / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer not on this machine")
    tok = AutoTokenizer.from_pretrained(str(EN.SNAPSHOT))
    enc = lambda s: list(tok(s, add_special_tokens=False).input_ids)
    ids = X.city_ids(enc, {"city_pool": "main"})
    assert len(ids) == len(set(ids)) == 12 and X.Reader.__init__ is not None
    assert enc(" Leeds")[0] in ids


def test_batched_arms_equal_batch1(eng):
    import exp56_arms as A
    rec = fake_rec(eng)
    rng = np.random.default_rng(0)
    inter = eng.layers[0].mlp.down_proj.weight.shape[1]
    unk = eng.encode(" unknown")[0]; ts = eng.encode(" Leeds")[0]
    specs = [{"name": "base", "ablate": None, "bias": None}]
    for k in range(4):
        l = int(rng.integers(0, eng.n_layers))
        idx = rng.choice(inter, 6, replace=False)
        specs.append({"name": f"r{k}", "ablate": {l: (idx, rng.normal(0, 3, 6).astype(np.float32))}, "bias": None})
    specs.append({"name": "bias", "ablate": None, "bias": {"ids": [unk], "value": -8.0}})
    rows, _, _ = A.compare(eng, rec, specs, unk, ts)
    assert all(r["same"] for r in rows), rows
    assert max(r["dD"] for r in rows) <= 1e-3
