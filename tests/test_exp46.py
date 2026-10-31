"""§46: triples (distinct golds, alignment, H2 policy), readout, statistics and verdict, probe shapes and CV folds,
the held guards, and the batched position-indexed hook on a tiny model (k3 floor; equivalence with overwrite_spec)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import patch46 as T  # noqa: E402
from memctl.analysis import probe46 as P  # noqa: E402

CASES = ROOT / "runs/exp45_maps/cases.jsonl"


def cases():
    if not CASES.exists():
        pytest.skip("runs/exp45_maps/cases.jsonl not present")
    return [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]


def rc_source():
    return [r for r in cases() if r["task"] == "RC" and r["variant"] in ("v0", "v3")]


# --- triples ----------------------------------------------------------------------------------------------------

def test_triples_distinct_golds_and_shared_text():
    import exp45_cases as C
    src = rc_source()
    assert len(src) == 144
    for c in src:
        tr = {r["role"]: r for r in T.build_triple_rows(c, C.NAMES, C.CITIES, C.render_prompt)}
        p, pp, ppp = tr["P"], tr["Pp"], tr["Ppp"]
        assert p["gold"] == c["gold"] == ppp["gold"] != pp["gold"]               # P / P' differ in the gold
        assert len({p["supervisor"], pp["supervisor"], ppp["supervisor"]}) == 3
        assert p["facts"][:] != pp["facts"] and len(p["facts"]) == len(c["facts"]) + 2
        # only the supervisor sentence differs between the three
        diff = [i for i, (a, b) in enumerate(zip(p["facts"], pp["facts"])) if a != b]
        assert diff == [p["facts"].index(f"{c['entities'][0]}'s supervisor is {c['entities'][1]}.")]
        assert p["facts"] == [f if i not in diff else f for i, f in enumerate(p["facts"])]
        assert f"{pp['supervisor']} works in {pp['gold']}." in p["prompt"]
        assert f"{ppp['supervisor']} works in {c['gold']}." in p["prompt"]
        assert p["addition"] == T.ADDITION
        if c["hardness"] != "H2":
            assert not p["b2_reuses_distractor"]
            assert not T._present(p["Bp"], c["prompt"]) and not T._present(p["Bpp"], c["prompt"])
        else:                                                                 # rev 2 ruling 1
            assert p["b2_reuses_distractor"] and not T._present(p["Bp"], c["prompt"])
            rel = {d["relation"] for d in c["distractors"] if d["name"] == p["Bpp"]}
            assert rel and "works_in" not in rel
        assert not T._present(p["C1p"], c["prompt"])


def test_triple_seeded_and_drop_policy():
    import exp45_cases as C
    c = next(r for r in rc_source() if r["hardness"] == "H2")
    a = T.triple_choice(c, C.NAMES, C.CITIES)
    assert a == T.triple_choice(c, C.NAMES, C.CITIES)
    with pytest.raises(T.TripleAlignmentError):
        T.triple_choice(c, C.NAMES, C.CITIES, h2_policy="drop")


def test_alignment_check():
    ids = {"P": [1, 2, 3, 4], "Pp": [1, 2, 9, 4], "Ppp": [1, 2, 8, 4]}
    assert T.check_triple_alignment(ids, {"P": 2, "Pp": 2, "Ppp": 2}) == 2
    with pytest.raises(T.TripleAlignmentError):
        T.check_triple_alignment({**ids, "Pp": [1, 7, 9, 4]}, {"P": 2, "Pp": 2, "Ppp": 2})
    with pytest.raises(T.TripleAlignmentError):
        T.check_triple_alignment({**ids, "Pp": [1, 2, 9]}, {"P": 2, "Pp": 2, "Ppp": 2})
    with pytest.raises(T.TripleAlignmentError):
        T.check_triple_alignment(ids, {"P": 2, "Pp": 1, "Ppp": 2})


def test_cases46_file_if_built():
    path = ROOT / "runs/exp46_probe/cases46.jsonl"
    if not path.exists():
        pytest.skip("cases46 not built")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 432 and all(r["addition"] == T.ADDITION for r in rows)
    tids = {r["triple_id"] for r in rows}
    assert len(tids) == 144
    by = {}
    for r in rows:
        by.setdefault(r["triple_id"], set()).add(r["gold"])
    assert all(len(g) == 2 for g in by.values())


# --- readout and statistics -------------------------------------------------------------------------------------

CITIES = ("York", "Durham", "Chester", "Leeds")


def test_first_city_readout():
    assert T.classify_first("Kate's supervisor is Laura, who works in Durham.", "Durham", "Chester", CITIES) == "C1"
    assert T.classify_first("She works in Chester. Durham", "Durham", "Chester", CITIES) == "C1p"
    assert T.classify_first("in York", "Durham", "Chester", CITIES) == "other"
    assert T.classify_first("Answer: unknown", "Durham", "Chester", CITIES) == "abstain"
    assert T.classify_first("no city here", "Durham", "Chester", CITIES) == "other"
    assert T.stop_now("works in Leeds", CITIES) and not T.stop_now("works in", CITIES)
    assert T.names_before_city("Laura works in Durham", ["Laura", "Alice"], CITIES) == ["Laura"]


def test_conclusion_readout_rev4():
    restate = ("Based on the information provided:\n\n- Alice works in Chester.\n- Sarah works in Durham.\n"
               "- Kate's supervisor is Laura.\n- Laura works in Durham.\n\nTherefore, Kate's supervisor, Laura, works in Durham.")
    assert T.classify_first(restate, "Durham", "Chester", CITIES) == "C1p"            # the rev 1 misread
    assert T.classify_conclusion(restate, "F1", "Durham", "Chester", CITIES) == "C1"
    note = "Kate's supervisor is Laura; Laura works in Durham, Alice in Chester.\nAnswer: Chester\n"
    assert T.classify_conclusion(note, "F2", "Durham", "Chester", CITIES) == "C1p"
    assert T.classify_conclusion("Answer: unknown\n", "F2", "Durham", "Chester", CITIES) == "abstain"
    assert T.classify_conclusion("Durham, no marker", "F2", "Durham", "Chester", CITIES) == "other"
    assert T.classify_conclusion("Answer: Leeds or Durham", "F2", "Durham", "Chester", CITIES) == "other"   # first on the line
    assert T.classify_conclusion("nothing", "F1", "Durham", "Chester", CITIES) == "other"
    assert T.classify_conclusion("Durham? I do not know: unknown", "F1", "Durham", "Chester", CITIES) == "abstain"
    assert T.names_before_city(restate, ["Laura", "Alice", "Sarah"], CITIES, "F1") == ["Laura", "Alice", "Sarah"]
    assert T.names_before_city(note, ["Laura", "Alice"], CITIES, "F2") == ["Laura", "Alice"]
    assert T.MAX_NEW_CONCLUSION == {"F1": 96, "F2": 160}


def test_attributed_supervisor():
    names = ("Laura", "Alice", "Sarah", "Kate", "Ben")
    f = lambda t, fmt="F1": T.attributed_supervisor(t, fmt, "Kate", names, "Laura", "Alice", "Sarah", CITIES)
    assert f("Kate's supervisor is Alice, and Alice works in Chester.") == "Bp"
    assert f("Kate's supervisor works in Chester.") == "none"                 # a verb is not a name
    assert f("- Kate's supervisor is Laura.\nTherefore, Kate's supervisor, Alice, works in Chester.") == "Bp"  # last
    assert f("Kate's supervisor is Ben, who works in York.") == "none"         # a pool name outside the triple
    assert f("The supervisor Sarah works in Durham.") == "Bpp"
    assert f("Kate's supervisor is Laura.\nAnswer: unknown\n", "F2") == "B"


def test_k1_vector_norm_matched():
    rng = np.random.default_rng(0)
    h, hp = np.zeros(16, np.float32), np.full(16, 2.0, np.float32)
    v = T.k1_vector(h, hp, rng)
    assert np.isclose(np.linalg.norm(v - h), np.linalg.norm(hp - h), rtol=1e-5)
    assert T.k1_seed("t", "S1", 3) == T.k1_seed("t", "S1", 3) != T.k1_seed("t", "S1", 4)


def test_transfer_and_lstar_and_verdict():
    good = {l: ["C1p"] * 9 + ["C1"] for l in range(12)}
    bad = {l: ["C1"] * 10 for l in range(12)}
    t, z = T.transfer_profile(good), T.transfer_profile(bad)
    assert t[0]["share"] == 0.9 and z[0]["share"] == 0.0
    assert T.carried_through(t, z, z) == 11
    late = dict(t)
    late[4] = z[4]
    assert T.carried_through(late, z, z) == 3
    assert T.carried_through(t, t, z) is None                                  # k1 transfers -> fails at 0
    v = T.verdict(20, 12, 40)
    assert v["verdict"] == "instrument blind spot" and not v["under_powered"]
    assert T.verdict(None, 3, 40)["verdict"] == "route late, lenses adequate"
    assert T.verdict(25, None, 40)["verdict"] == "represented, not used"
    assert T.verdict(None, 15, 10)["under_powered"]
    assert T.verdict(30, 15, 40)["probe_decodable_le_29"] is False


# --- probe ------------------------------------------------------------------------------------------------------

def synthetic(n_per=12, labels=("a", "b", "c"), d=8, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = [lab for lab in labels for _ in range(n_per)]
    X = rng.standard_normal((len(y), d)).astype(np.float32)
    for i, lab in enumerate(y):
        X[i, labels.index(lab)] += signal
    groups = [f"g{i % 6}" for i in range(len(y))]
    return X, y, groups


def test_probe_shapes_and_grouped_cv():
    X, y, g = synthetic()
    fit = P.fit_probe(X, y, groups=g)
    assert fit["C"] in P.C_GRID and set(fit["cv_scores"]) == {str(c) for c in P.C_GRID}
    assert fit["model"].predict(X).shape == (len(y),)
    cv = P.cv_accuracy(X, y, fit["C"], groups=g)
    assert cv["folds"] == 6 and cv["n"] == len(y) and cv["acc"] > 0.8
    noise = P.cv_accuracy(np.random.default_rng(1).standard_normal(X.shape), y, 1.0, groups=g)
    assert noise["acc"] < 0.7
    assert P.cv_group({"format": "F1", "hardness": "H2"}) == "F1|H2"


def test_dev_conditions_and_decodable():
    hi = {"acc": 0.9, "lower": 0.8}
    assert P.dev_conditions(hi, {"acc": 0.5}, {"acc": 0.3})
    assert not P.dev_conditions(hi, {"acc": 0.7}, None)                         # selectivity 0.2 < 0.30
    assert not P.dev_conditions(hi, {"acc": 0.1}, {"acc": 0.85})                # presence >= probe lower bound
    assert P.dev_conditions(hi, {"acc": 0.1}, None)                             # TU: no presence clause
    assert P.decodable({"acc": 0.75, "lower": 0.6})                            # rev 5: held numbers only
    assert not P.decodable({"acc": 0.75, "lower": 0.5})
    assert not P.decodable({"acc": 0.65, "lower": 0.6})
    assert not P.decodable({"acc": None, "lower": None})
    assert P.first_decodable({0: False, 3: True, 7: True}) == 3 and P.first_decodable({0: False}) is None


def test_positions_on_real_cases():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    from memctl.analysis.jacobian import wrap_chat
    for c in [r for r in cases() if r["task"] in ("RC", "TU")][::37]:
        chat = wrap_chat(tok, c["prompt"])
        ids = tok(chat, add_special_tokens=False).input_ids
        pos = P.positions(tok, chat, c)
        assert pos["P_last"] == len(ids) - 1
        want = {"P_B": c["entities"][1], "P_q": c["entities"][0], "P_cur": c["gold"],
                "P_old": (c["competitors"] or [""])[0]}
        for k, v in pos.items():
            if k != "P_last":
                assert want[k].startswith(tok.decode([ids[v]]).strip())


# --- held guards ------------------------------------------------------------------------------------------------

def test_held_probe_refuses_without_marker(tmp_path, monkeypatch):
    import exp46_probe as E
    cfg = {"x": 1}
    with pytest.raises(PermissionError):
        P.require_design(tmp_path, cfg)
    (tmp_path / P.DESIGN_MARKER).write_text(json.dumps({"commit": "", "design_sha256": P.design_stamp(cfg)}))
    with pytest.raises(PermissionError):
        P.require_design(tmp_path, cfg)
    (tmp_path / P.DESIGN_MARKER).write_text(json.dumps({"commit": "abc", "design_sha256": "other"}))
    with pytest.raises(PermissionError):
        P.require_design(tmp_path, cfg)
    (tmp_path / P.DESIGN_MARKER).write_text(json.dumps({"commit": "abc", "design_sha256": P.design_stamp(cfg)}))
    assert P.require_design(tmp_path, cfg)["commit"] == "abc"

    # the script: a fake capture of one prompt, --held exits 4 with no marker
    cap = tmp_path / "cap"
    cap.mkdir()
    c = next(r for r in cases() if r["task"] == "RC")
    meta = {"case_id": c["case_id"], "task": "RC", "split": c["split"], "base_index": c["base_index"],
            "hardness": c["hardness"], "capture_version": "v", "device": "cpu", "dtype": "float32", "model": "m"}
    (cap / f"{c['case_id']}.json").write_text(json.dumps(meta))
    np.savez(cap / f"{c['case_id']}.npz", resid=np.zeros((3, 36, 4), np.float32),
             sites=np.array(["P_last", "P_B", "P_q"]))
    (cap / "probe_dev.json").write_text("{}")                 # probe outputs beside the capture are not prompts
    (cap / "design.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["exp46_probe.py", "--capture", str(cap), "--held"])
    assert E.main() == E.EXIT_DESIGN
    assert not (cap / "probe_held.json").exists()


def test_held_report_refuses_without_marker(tmp_path, monkeypatch):
    import exp46_report as R
    (tmp_path / "c.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["exp46_report.py", "--probe", str(tmp_path), "--patch", str(tmp_path),
                                      "--cases46", str(tmp_path / "c.jsonl"), "--held"])
    assert R.main() == R.EXIT_DESIGN


def test_pools_match_exp45():
    import exp45_cases as C
    import exp46_probe as E
    assert E.NAMES == C.NAMES and E.CITIES == C.CITIES


# --- the batched hook on a tiny model ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    net = transformers.Qwen2ForCausalLM(config).eval()
    return torch, transformers, tok, net


def runner(tiny, batch=7):
    import exp46_patch as E
    torch, _, tok, net = tiny
    return E.Runner(net, tok, "cpu", batch)


def three_pairs(tok):
    out = []
    for a, b, bp in (("Kate", "Laura", "Alice"), ("Ben", "Emma", "Paul"), ("Sarah", "James", "Henry")):
        p = f"Alice works in Leeds.\n{a}'s supervisor is {b}.\n{b} works in York.\n\nIn which city does {a}'s supervisor work?"
        pp = p.replace(f"supervisor is {b}.", f"supervisor is {bp}.")
        out.append((p, pp))
    return out


def test_k3_embedding_floor_reproduces_pprime(tiny):
    """k3 on 3 pairs: overwriting the input embedding at P_B with P''s token reproduces P''s greedy output."""
    run = runner(tiny)
    never = lambda text: False
    for p, pp in three_pairs(run.tok):
        ip, ipp = run.ids(p), run.ids(pp)
        assert len(ip) == len(ipp)
        diff = [i for i, (a, b) in enumerate(zip(ip, ipp)) if a != b]
        assert len(diff) == 1
        pb = diff[0]
        ref = run.generate(ipp, len(ipp) - 1, [("none", None, None)], 6, never)[0]
        got = run.generate(ip, pb, [("embed", None, run.embedding(ipp[pb])), ("none", None, None)], 6, never)
        assert got[0]["ids"] == ref["ids"]
        own = run.generate(ip, len(ip) - 1, [("none", None, None)], 6, never)[0]
        assert got[1]["ids"] == own["ids"]                                    # the unpatched row is untouched


def test_batched_rows_independent_and_prefix_cache(tiny):
    """A batch of layer variants equals the same variants run one at a time (batch 1)."""
    import exp46_patch as E
    torch, _, tok, net = tiny
    run7, run1 = runner(tiny, 7), runner(tiny, 1)
    p, pp = three_pairs(tok)[0]
    ip, ipp = run7.ids(p), run7.ids(pp)
    pb = next(i for i, (a, b) in enumerate(zip(ip, ipp)) if a != b)
    h = run7.residuals(ipp, [pb])[0]
    variants = [("resid", l, h[l]) for l in range(run7.n_layers)]
    a = run7.generate(ip, pb, variants, 5, lambda t: False)
    b = run1.generate(ip, pb, variants, 5, lambda t: False)
    assert [x["ids"] for x in a] == [x["ids"] for x in b]
    # patching every layer >= the last at P_B with P''s own residual: the last layer's patch equals P''s first logits path
    full = run7.generate(ip, pb, [("resid", run7.n_layers - 1, h[-1])], 5, lambda t: False)[0]
    assert isinstance(full["ids"], list) and len(full["ids"]) == 5


def test_hook_matches_overwrite_spec_at_last_position(tiny):
    """Rev 2 ruling 2: with the site at the last prompt token, the position-indexed hook reproduces
    steering.overwrite_spec (HookedEngine.run) on one prompt."""
    from memctl.analysis.steering import HFEngine, overwrite_spec
    torch, transformers, tok, net = tiny
    eng = HFEngine.__new__(HFEngine)
    eng.torch, eng.model, eng.family, eng.device = torch, "tiny", None, "cpu"
    eng.name, eng.tokenizer = "tiny-qwen2", tok
    eng.net = net
    eng.layers = net.model.layers
    eng.n_layers, eng.hidden = len(eng.layers), net.config.hidden_size
    eng.stop_ids = {tok.eos_token_id}
    run = runner(tiny)
    p, _ = three_pairs(tok)[0]
    ids = run.ids(p)
    vec = np.random.default_rng(3).standard_normal(net.config.hidden_size).astype(np.float32) * 5
    for layer in (0, 3, 5):
        prepared = eng.prepare(type("I", (), {"prompt": p, "continuation": ""})())
        assert prepared.ids == ids
        ref = eng.run(prepared, [overwrite_spec(layer, vec)], max_new_tokens=6)
        stop = lambda text: "\n" in text
        got = run.generate(ids, len(ids) - 1, [("resid", layer, vec)], 6, stop)[0]
        got_ids = [t for t in got["ids"] if t not in run.eos]
        assert got_ids[:len(ref.gen_ids)] == ref.gen_ids
