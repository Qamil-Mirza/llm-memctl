"""§47: sites, k1, transfer, generated outcome, L* and readings; the rows file; the patched-suffix runner on a tiny
model (fork = plain forward, k3 = the swap prompt, last-layer patch inert, S2 = overwrite_spec, batch independence)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import bridge47 as B  # noqa: E402

ROWS = ROOT / "runs/exp47_bridge/rows47.jsonl"


def test_diff_positions_and_k1():
    assert B.diff_positions([1, 2, 3, 4, 5], [1, 9, 3, 8, 5]) == [1, 3]
    with pytest.raises(ValueError):
        B.diff_positions([1, 2], [1, 2, 3])
    hc = np.zeros((2, 8), np.float32)
    hs = np.stack([np.full(8, 1.0), np.full(8, 3.0)]).astype(np.float32)
    v = B.k1_vectors(hc, hs, "q", 4, [10, 11])
    assert np.allclose(np.linalg.norm(v - hc, axis=1), np.linalg.norm(hs - hc, axis=1), rtol=1e-5)
    assert np.array_equal(v, B.k1_vectors(hc, hs, "q", 4, [10, 11]))
    assert not np.array_equal(v, B.k1_vectors(hc, hs, "q", 5, [10, 11]))


def test_transfer_and_readings():
    assert B.transfer(0.0, 2.0, -2.0) == 0.5 and B.transfer(-2.0, 2.0, -2.0) == 1.0 and B.transfer(2.0, 2.0, -2.0) == 0
    assert B.transfer(1.0, 1.0, 1.0) is None
    good = {l: [True] * 12 + [False] for l in range(20)}
    t = B.share_profile(good)
    k1 = B.share_profile({l: [False] * 13 for l in range(20)})
    assert t[0]["k"] == 12 and B.l_star(t, k1) == 19
    t[11] = B.share_profile({0: [True] * 10 + [False] * 3})[0]          # 10/13: LB 0.497 < 0.55
    assert B.l_star(t, k1) == 10
    assert B.l_star(B.share_profile({0: [True] * 11 + [False] * 2}), k1) == 0    # 11/13 passes
    assert B.class_status(7) == "descriptive" and B.class_status(11) == "thin" and B.class_status(13) == "read"
    assert B.row_l_star([0.9, 0.6, 0.4, 0.9]) == 1 and B.row_l_star([0.1, 0.9]) is None
    assert B.b1_reading(12, "read").startswith("yes") and B.b1_reading(5, "read").startswith("no")
    assert "downstream" in B.b2_reading(12, 11, "read", True)
    assert "not carried" in B.b2_reading(12, 4, "read", True)
    assert "no reference" in B.b2_reading(5, 12, "read", False)
    assert B.wilson(7, 8)[1] < 0.55 <= B.wilson(8, 8)[1]


def test_generated_outcome():
    assert B.answer_phrase("note here\nAnswer: 25 years\n") == "25 years"
    assert B.answer_phrase("25 years old") == "25 years old"
    assert B.generated_outcome("note\nAnswer: 25\n", "25", "33") == "G"
    assert B.generated_outcome("note\nAnswer: 33\n", "25", "33") == "S"
    assert B.generated_outcome("note\nAnswer: unknown\n", "25", "33") == "unknown"
    assert B.generated_outcome("note\nAnswer: 25 or 33\n", "25", "33") == "other"
    assert B.generated_outcome("note\nAnswer: Paris\n", "25", "33") == "other"


def test_rows_file_if_built():
    if not ROWS.exists():
        pytest.skip("rows47 not built")
    rows = [json.loads(l) for l in ROWS.read_text().splitlines()]
    assert len(rows) == 36
    ent = [r for r in rows if r["entered"]]
    assert sum(r["label"] == "success" for r in ent) == 13 and sum(r["label"] != "success" for r in ent) == 13
    assert sorted(r["question_id"] for r in rows if r["multi_occurrence"]) == ["3e321797", "8b9d4367", "c7cf7dfd"]
    for r in rows:
        assert B.diff_positions(r["chat_clean"], r["chat_swap"]) == r["diff_positions"]
        assert r["p1"] == r["n_chat"] + len(r["forced_tail"]) - 1


# --- the runner on a tiny model ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    net = transformers.Qwen2ForCausalLM(cfg).eval()
    import exp47_patch as E
    return E, E.Runner(net, tok, "cpu", 5), tok


def pair(tok):
    from memctl.analysis.jacobian import wrap_chat
    c = tok(wrap_chat(tok, "Memory: I am 25 years old and my friend is 25 too.\nQuestion: how old am I?"),
            add_special_tokens=False).input_ids
    s = tok(wrap_chat(tok, "Memory: I am 33 years old and my friend is 33 too.\nQuestion: how old am I?"),
            add_special_tokens=False).input_ids
    tail = tok("Answer: ", add_special_tokens=False).input_ids
    return c, s, tail


def test_fork_readout_equals_plain_forward(tiny):
    E, run, tok = tiny
    c, s, tail = pair(tok)
    sites = B.diff_positions(c, s)
    assert len(sites) == 4                                            # two occurrences x two digit tokens
    got = run.run(c, sites, [("none", None, None)], tail, True, max_new=4)[0]
    ref, _ = run.capture(c + tail, [0])
    assert np.allclose(got["logits_p1"], ref, atol=1e-4)
    alone = run.run(c, sites, [("none", None, None)], None, True, max_new=4)[0]
    assert alone["ids"] == got["ids"]                                 # the fork does not touch the generation


def test_k3_is_the_swap_prompt_and_last_layer_inert(tiny):
    E, run, tok = tiny
    c, s, tail = pair(tok)
    sites = B.diff_positions(c, s)
    emb = run.embeddings([s[p] for p in sites])
    k3 = run.run(c, sites, [("embed", None, emb)], tail, True, max_new=5)[0]
    swap_ref, _ = run.capture(s + tail, [0])
    assert np.allclose(k3["logits_p1"], swap_ref, atol=1e-4)
    assert k3["ids"] == run.run(s, sites, [("none", None, None)], None, True, max_new=5)[0]["ids"]
    _, hs = run.capture(s + tail, sites)
    last = run.run(c, sites, [("resid", run.n_layers - 1, hs[:, -1])], tail, False)[0]
    clean_ref, _ = run.capture(c + tail, [0])
    assert np.allclose(last["logits_p1"], clean_ref, atol=1e-4)     # no layer after the last reads the sites


def test_batch_independence(tiny):
    E, _, tok = tiny
    run1 = E.Runner(tiny[1].hf, tok, "cpu", 1)
    c, s, tail = pair(tok)
    sites = B.diff_positions(c, s)
    _, hs = run1.capture(s + tail, sites)
    v = [("resid", l, hs[:, l]) for l in range(run1.n_layers)]
    a = tiny[1].run(c, sites, v, tail, True, max_new=4)
    b = run1.run(c, sites, v, tail, True, max_new=4)
    assert [x["ids"] for x in a] == [x["ids"] for x in b]
    assert all(np.allclose(x["logits_p1"], y["logits_p1"], atol=1e-4) for x, y in zip(a, b))


def test_s2_matches_overwrite_spec(tiny):
    """S2 (site = P1 of the forced sequence) reproduces steering.overwrite_spec's step-0 logits."""
    from memctl.analysis.steering import HFEngine, overwrite_spec
    import torch
    E, run, tok = tiny
    c, s, tail = pair(tok)
    forced = c + tail
    eng = HFEngine.__new__(HFEngine)
    eng.torch, eng.model, eng.family, eng.device = torch, "tiny", None, "cpu"
    eng.name, eng.tokenizer, eng.net = "tiny", tok, run.hf
    eng.layers = run.hf.model.layers
    eng.n_layers, eng.hidden = len(eng.layers), run.hf.config.hidden_size
    eng.stop_ids = {tok.eos_token_id}
    from memctl.analysis import patching as pt
    vec = np.random.default_rng(1).standard_normal(run.hf.config.hidden_size).astype(np.float32) * 4
    for layer in (0, 3, 5):
        prep = pt._prepare(eng, forced)
        ref, _ = pt._p1_logits(eng, prep, [overwrite_spec(layer, vec)])
        got = run.s2(forced, [("resid", layer, vec)])[0]
        assert np.allclose(got, ref, atol=1e-4)
