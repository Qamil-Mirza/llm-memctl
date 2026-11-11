"""§57 build checks: the attribution-patching formula is exact on a linear toy; on the tiny dummy Qwen2 the predicted
effect Δ̂ matches the real patch in the linear regime (small alpha) in both regions; the batched arm engine matches
batch 1 with region-N and region-P resets and a bias."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

EN = pytest.importorskip("exp51_engine")
import exp57_common as X  # noqa: E402


def test_attrib_exact_on_linear_toy():
    import torch
    g = torch.Generator().manual_seed(0)
    T, I = 9, 13
    W = torch.randn(T, I, generator=g, dtype=torch.float64)
    a = torch.randn(T, I, generator=g, dtype=torch.float64, requires_grad=True)
    D = (W * a).sum() + 0.7                                   # linear in every a_i(t)
    (grad,) = torch.autograd.grad(D, [a])
    reset = torch.randn(I, generator=g, dtype=torch.float64)
    for lo, hi in ((8, 9), (2, 8)):                           # one position (P) and a span (N)
        pred = X.attrib(a.detach(), grad, reset, lo, hi)
        for i in range(I):
            b = a.detach().clone(); b[lo:hi, i] = reset[i]
            exact = float((W * b).sum() + 0.7 - D.detach())
            assert abs(float(pred[i]) - exact) < 1e-10


@pytest.fixture(scope="module")
def eng():
    if not (EN.SNAPSHOT / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer not on this machine")
    return EN.load("dummy", threads=2)


def fake_rec(e):
    p = e.chat_ids("Below is your memory. Paul lives in Leeds. Where does Paul live?")
    g = e.encode("user mentioned that Paul lives in Leeds.\nAnswer:")
    return {"prompt_ids": p, "gen_ids": g + e.encode(" Leeds"), "j": len(g), "p_ans": len(p) + len(g) - 1}


def _resets(e, seed=0):
    rng = np.random.default_rng(seed)
    I = e.layers[0].mlp.down_proj.weight.shape[1]
    return (rng.normal(0, 1, (e.n_layers, I)).astype(np.float32), rng.normal(0, 1, (e.n_layers, I)).astype(np.float32))


@pytest.mark.parametrize("grad_from", [0, 2])
def test_dhat_matches_patch_linear_regime(eng, grad_from):
    rec = fake_rec(eng)
    rp, rn = _resets(eng)
    at = X.Attributor(eng, rp, rn, grad_from=grad_from)
    cities = [eng.encode(" Leeds")[0], eng.encode(" York")[0]]
    res = at.run(rec, cities)
    pt = X.Patcher(eng, rec, at.unk_id, res["t_star"])
    assert abs(pt.D_clean_P - res["D"]) < 1e-4 and abs(pt.D_clean_N - res["D"]) < 1e-4
    alpha = 1e-2
    for region, dh, rv in (("P", res["dhat_P"], rp), ("N", res["dhat_N"], rn)):
        assert not dh[:grad_from].any()                        # nothing below the graph start
        flat = np.argsort(-np.abs(dh).ravel())[:6]
        for f in flat:
            l, i = divmod(int(f), dh.shape[1])
            real = pt.effect(region, [(l, [i], [rv[l, i]])], alpha=alpha) / alpha
            assert abs(real - dh[l, i]) <= 0.05 * abs(dh[l, i]) + 1e-4, (region, l, i, real, dh[l, i])


def test_batched_arms_equal_batch1(eng):
    import exp57_arms as A
    rec = fake_rec(eng)
    rng = np.random.default_rng(1)
    I = eng.layers[0].mlp.down_proj.weight.shape[1]
    unk = eng.encode(" unknown")[0]; ts = eng.encode(" Leeds")[0]
    rs = lambda: [(int(rng.integers(0, eng.n_layers)), rng.choice(I, 5, replace=False), rng.normal(0, 3, 5))
                  for _ in range(2)]
    arms = [{"name": "base"}]
    arms += [{"name": f"N{k}", "reset_N": rs()} for k in range(3)]
    arms += [{"name": f"P{k}", "reset_P": rs()} for k in range(3)]
    arms += [{"name": "NP", "reset_N": rs(), "reset_P": rs()}, {"name": "bias", "bias": {"ids": [unk], "value": -8.0}}]
    rows, _, _ = A.compare(eng, rec, arms, unk, ts)
    assert all(r["same"] for r in rows), rows
    assert max(r["dD"] for r in rows) <= 1e-3
    # resets move D (the hooks fire), the bias does not move D0
    ref = {r["arm"]: r for r in rows}
    out = {a["name"]: x for a, x in zip(arms, A.run_arms(eng, rec, arms, unk, ts))}
    assert any(abs(out[f"N{k}"]["D0"] - out["base"]["D0"]) > 1e-6 for k in range(3))
    assert any(abs(out[f"P{k}"]["D0"] - out["base"]["D0"]) > 1e-6 for k in range(3))
    assert abs(out["bias"]["D0"] - out["base"]["D0"]) < 1e-5 and ref


def test_reset_at_and_chunked_equal_batch1(eng):
    """rev 0b: position-specific resets (K_N@marker) and the chunked runner match batch 1; reset_at over the whole
    region N equals reset_N."""
    import exp57_arms as A
    rec = fake_rec(eng)
    n0, p, _ = X.regions(rec)
    rng = np.random.default_rng(2)
    I = eng.layers[0].mlp.down_proj.weight.shape[1]
    unk = eng.encode(" unknown")[0]; ts = eng.encode(" Leeds")[0]
    l, idx = 1, rng.choice(I, 5, replace=False)
    v1, v2 = rng.normal(0, 3, 5), rng.normal(0, 3, 5)
    arms = [{"name": "base"},
            {"name": "at_last", "reset_at": [(p - 1, p, l, idx, v1)]},
            {"name": "at_two", "reset_at": [(p - 2, p - 1, l, idx, v2), (p - 1, p, l, idx, v1)]},
            {"name": "at_all", "reset_at": [(n0, p, l, idx, v1)]},
            {"name": "N_all", "reset_N": [(l, idx, v1)]},
            {"name": "bias", "bias": {"ids": [unk], "value": -4.0}}]
    rows, _, _ = A.compare(eng, rec, arms, unk, ts, chunk=2)
    assert all(r["same"] for r in rows), rows
    assert max(r["dD"] for r in rows) <= 1e-3
    out = {x["name"]: x for x in A.run_arms_chunked(eng, rec, arms, unk, ts, chunk=4)}
    assert abs(out["at_all"]["D0"] - out["N_all"]["D0"]) < 1e-5
    assert abs(out["at_last"]["D0"] - out["base"]["D0"]) > 1e-6


def test_t_star_resolved_from_base(eng):
    import exp57_arms as A
    rec = fake_rec(eng)
    unk = eng.encode(" unknown")[0]
    cities = [eng.encode(" Leeds")[0], eng.encode(" York")[0]]
    arms = [{"name": "base"}, {"name": "bias", "bias": {"ids": [unk], "value": -3.0}},
            {"name": "N", "reset_N": [(1, [0, 1], [0.5, -0.5])]}]
    got = A.run_arms_chunked(eng, rec, arms, unk, None, chunk=2, cand_ids=cities)
    ts = got[0]["t_star"]
    assert ts in cities and all(x["t_star"] == ts for x in got)
    ref = A.run_arms(eng, rec, arms, unk, ts)
    assert [x["gen_ids"] for x in got] == [x["gen_ids"] for x in ref]
    assert max(abs(a["D0"] - b["D0"]) for a, b in zip(got, ref)) < 1e-4
