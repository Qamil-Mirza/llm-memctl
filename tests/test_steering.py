"""§38 steering: hooks, prefix cache, scaling, classification, populations on the real cache, the design gate, resume
and the report's refusals (memctl/analysis/steering.py, scripts/exp38_dev.py, scripts/exp38_report.py). StubEngine only;
no Qwen weights are loaded."""
import json
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import answer_selection as sel
from memctl.analysis import steering as st
from memctl.jacobian_data import PromptRecord
from tests.test_elenchus import LINES, make_prompt

CACHE = Path("runs/exp35/qwen3b/A/fold0.jsonl")
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="§35 cache not on disk")


def item(prompt="Question: what? \n\nNote:", continuation="the note\nAnswer:"):
    return st.SteerInput("A", "q1", 0, prompt, continuation, st.sha256(json.dumps([prompt, continuation])), "x")


# --- prefix cache and position ------------------------------------------------------------------------------------

def test_prefix_cache_excludes_the_last_token_and_last_is_colon():
    eng = st.StubEngine()
    it = item()
    prepared = eng.prepare(it)
    ids = eng.input_ids(it)
    assert prepared.ids == ids and prepared.last_id == ids[-1]
    assert prepared.cache.length == len(ids) - 1 == prepared.n_prefix
    assert eng.last_token_text(prepared) == ":"
    gen = eng.run(prepared, [], 4)
    assert prepared.cache.length == len(ids) - 1      # the shared prefix is never mutated by a run (deep copy)
    assert gen.n_new <= 4


def test_generation_stops_at_newline_and_eos():
    eng = st.StubEngine()
    it = item()
    eng.set_oracle({st.sha256(eng.chat_text(it.prompt) + it.continuation): " Luna\nmore text"})
    gen = eng.run(eng.prepare(it), [], 16)
    assert gen.generated == " Luna\n" and st.phrase_of(gen.generated) == "Luna"
    eng.set_oracle({st.sha256(eng.chat_text(it.prompt) + it.continuation): " unknown"})
    gen = eng.run(eng.prepare(it), [], 16)
    assert gen.generated == " unknown" and gen.n_new == len(" unknown")


# --- hooks -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tuple_output", [False, True])
def test_add_and_overwrite_act_on_the_layer_output_and_are_removed(tuple_output):
    eng = st.StubEngine(tuple_output=tuple_output)
    prepared = eng.prepare(item())
    base = eng.run(prepared, [], 3, capture=tuple(range(eng.n_layers)))
    vec = np.arange(eng.hidden, dtype=np.float32) / 10
    added = eng.run(prepared, [st.add_spec(5, vec)], 3, capture=(4, 5))
    assert np.allclose(added.captured[5], base.captured[5] + vec, atol=1e-5)
    assert np.array_equal(added.captured[4], base.captured[4])     # below the hook: untouched
    target = np.ones(eng.hidden, dtype=np.float32)
    over = eng.run(prepared, [st.overwrite_spec(5, target)], 3, capture=(5,))
    assert np.allclose(over.captured[5], target)
    assert all(len(layer._forward_hooks) == 0 for layer in eng.layers)


def test_hooks_removed_on_exception():
    eng = st.StubEngine()
    state = st.HookState()
    with pytest.raises(RuntimeError):
        with st.hooked(eng.layers, [st.add_spec(2, np.zeros(eng.hidden, np.float32))], state):
            assert len(eng.layers[2]._forward_hooks) == 1
            raise RuntimeError("boom")
    assert len(eng.layers[2]._forward_hooks) == 0


def test_add_applies_every_step_overwrite_only_step0():
    eng = st.StubEngine()
    prepared = eng.prepare(item())
    calls = {}
    for name, spec in (("add", st.add_spec(3, np.zeros(eng.hidden, np.float32))),
                       ("over", st.overwrite_spec(3, np.zeros(eng.hidden, np.float32)))):
        state = st.HookState()
        import torch
        cache = __import__("copy").deepcopy(prepared.cache)
        with torch.no_grad(), st.hooked(eng.layers, [spec], state):
            for step in range(3):
                state.step = step
                eng._step(prepared.last_id, cache)
        calls[name] = state.calls
    assert calls == {"add": 3, "over": 1}


def test_zero_add_and_self_overwrite_are_byte_exact():
    eng = st.StubEngine()
    prepared = eng.prepare(item())
    base = eng.run(prepared, [], 8, capture=tuple(range(eng.n_layers)))
    own = np.stack([base.captured[L] for L in range(eng.n_layers)])
    for spec in (st.add_spec(6, np.zeros(eng.hidden, np.float32)), st.overwrite_spec(eng.n_layers - 1, own[-1]),
                 st.overwrite_spec(4, own[4])):
        g = eng.run(prepared, [spec], 8)
        assert g.generated == base.generated and g.gen_ids == base.gen_ids


# --- vectors -----------------------------------------------------------------------------------------------------

def test_direction_and_scaling_units():
    a = {"p": np.zeros((3, 4), np.float32), "q": np.ones((3, 4), np.float32)}
    b = {"p": np.full((3, 4), 2.0, np.float32), "q": np.full((3, 4), 3.0, np.float32)}
    v = st.estimate_direction(a, b, ["p", "q"], 1)
    assert np.allclose(v, 2.0)
    assert np.allclose(st.scaled_vector(v, 2.0, "vnorm"), 4.0)
    r = st.scaled_vector(v, 2.0, "residnorm", mean_resid=3.0)
    assert np.isclose(np.linalg.norm(r), 6.0) and np.allclose(r / np.linalg.norm(r), v / np.linalg.norm(v))
    stats = st.direction_stats(v, a, ["p", "q"], 1)
    assert np.isclose(stats["v_norm"], 4.0) and np.isclose(stats["mean_resid_A_norm"], 1.0)
    assert np.isclose(stats["v_over_resid"], 4.0)
    with pytest.raises(ValueError):
        st.scaled_vector(v, 1.0, "bogus")


def test_random_matches_norm_is_seeded_and_reverse_negates():
    v = np.random.default_rng(1).standard_normal(64).astype(np.float32)
    r1, r2 = st.random_vector(v, 22), st.random_vector(v, 22)
    assert np.array_equal(r1, r2) and np.isclose(np.linalg.norm(r1), np.linalg.norm(v), rtol=1e-5)
    assert not np.allclose(r1, st.random_vector(v, 25))
    vectors = {(22, 2.0, "vnorm"): 2 * v}
    rnd = st.setting_vector(st.Setting("random", 22, 2.0, "vnorm"), vectors)
    assert np.isclose(np.linalg.norm(rnd), np.linalg.norm(2 * v), rtol=1e-5)
    assert np.allclose(st.setting_vector(st.Setting("reverse", 22, 2.0, "vnorm"), vectors), -2 * v)
    assert st.vector_sha(r1) == st.vector_sha(r2) != st.vector_sha(-r1)


def test_depths_map_to_the_preregistered_layers():
    assert [st.depth_layer(d) for d in st.SWEEP_DEPTHS] == [18, 22, 25, 29, 32]
    assert [st.depth_layer(d) for d in st.SMALL_DEPTHS] == [22, 29, 32]
    assert st.depth_layer(st.SHALLOW_DEPTH) == 7 and st.final_layer() == 35
    assert st.ALPHAS == (1.0, 2.0, 4.0) and st.DEFAULT_UNIT == "vnorm" and st.MAX_NEW_TOKENS_LAPTOP == 16


# --- classification ----------------------------------------------------------------------------------------------

def test_classification():
    assert st.classify("unknown") == "refusal" and st.classify("Unknown.") == "refusal"
    assert st.classify("") == "degenerate" and st.classify("  ") == "degenerate"
    assert st.classify("?!...") == "degenerate"
    assert st.classify("ha ha", [5] * 8) == "degenerate"
    assert st.classify("ha ha", [5] * 7 + [6]) == "answer"
    # a list answer reuses the comma id 8 times but never 8 in a row (build review ruling 2)
    assert st.classify("1, 2, 3, 4, 5, 6, 7, 8, 9", [1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8, 9, 10]) == "answer"
    assert st.classify("ha ha", [6] + [5] * 8 + [6]) == "degenerate"
    assert st.CLASSIFIER_VERSION.startswith("v2")
    assert st.classify("Luna") == "answer"
    assert st.provisional_gold("Luna the cat", "Luna", "answer") is True
    assert st.provisional_gold("unknown", "unknown", "refusal") is False
    assert st.phrase_of(" 4 hours\nNote") == "4 hours"


def test_reproduces():
    cached_ref = {"refusal": True, "final_answer": "unknown"}
    assert st.reproduces({"refusal": True, "final_answer": "unknown"}, cached_ref, "Luna")["ok"]
    assert not st.reproduces({"refusal": False, "final_answer": "Luna"}, cached_ref, "Luna")["ok"]
    cached_ans = {"refusal": False, "final_answer": "Luna"}
    assert st.reproduces({"refusal": False, "final_answer": "it is Luna"}, cached_ans, "Luna")["ok"]
    # R rows and pairs: refusal flags only; the D1 verdict match is a column (build review ruling 1)
    r = st.reproduces({"refusal": False, "final_answer": "Max"}, cached_ans, "Luna")
    assert r["ok"] and not r["verdict_match"]
    # damage rows: A must answer and pass D1 on the new engine
    assert st.reproduces_damage({"refusal": False, "final_answer": "Luna"}, cached_ans, "Luna")["ok"]
    assert not st.reproduces_damage({"refusal": False, "final_answer": "Max"}, cached_ans, "Luna")["ok"]
    assert not st.reproduces_damage({"refusal": True, "final_answer": "unknown"}, cached_ans, "Luna")["ok"]


def test_cut_input_without_marker_has_no_note():
    rec = PromptRecord("q1", 0, "qwen3b", make_prompt(), "note\nAnswer: Luna", "Luna", "single-session-user", True,
                       LINES, (LINES[0],), True, True, False)
    with_note = st.cut_input(rec, "the cut note\nAnswer: unknown")
    assert with_note.continuation == "the cut note\nAnswer:" and not with_note.cut_no_marker
    assert LINES[0] not in with_note.prompt
    bare = st.cut_input(rec, "no marker here")
    assert bare.continuation == "Answer:" and bare.cut_no_marker
    a = st.arm_input(rec, "A")
    assert a.continuation == "note\nAnswer:" and a.prompt == rec.prompt
    b = st.arm_input(rec, "B")
    assert b.continuation == "note\n\n" + sel.B_INSTRUCTION + "\nAnswer:"


# --- the design gate ---------------------------------------------------------------------------------------------

def _s(layer, alpha, ans, deg=0.0):
    return {"layer": layer, "alpha": alpha, "ans": ans, "degenerate": deg, "n": 28}


def test_gate_tie_break_shallower_layer_then_smaller_alpha():
    steer = {"steer:L29:a2": _s(29, 2.0, 0.7), "steer:L22:a4": _s(22, 4.0, 0.7), "steer:L22:a2": _s(22, 2.0, 0.7)}
    g = st.design_gate(steer, {}, True, 0.9)
    assert g["status"] == "need_random" and g["need"] == "steer:L22:a2" and g["pass"] is None
    g = st.design_gate(steer, {"steer:L22:a2": 0.1}, True, 0.9)
    assert g["chosen"] == "steer:L22:a2" and g["pass"] is True


def test_gate_random_and_degenerate_eligibility_and_arithmetic():
    steer = {"a": _s(18, 4.0, 0.9, deg=0.25), "b": _s(25, 2.0, 0.8), "c": _s(32, 1.0, 0.6), "d": _s(18, 1.0, 0.3)}
    g = st.design_gate(steer, {"b": 0.17, "c": 0.12}, True, 0.85)
    assert [e["setting"] for e in g["order"]] == ["a", "b", "c"]
    assert "degenerate" in g["order"][0]["why"]
    assert g["order"][1]["eligible"] is False and np.isclose(g["order"][1]["random_bound"], 0.16)
    assert g["chosen"] == "c" and g["pass"] is True and g["chosen_setting"]["layer"] == 32
    # below 0.5: no Random asked, gate fails
    g = st.design_gate({"d": _s(18, 1.0, 0.3)}, {}, True, 0.9)
    assert g["status"] == "none" and g["pass"] is False
    # sanity or reproduction failing fails the gate even with a chosen setting
    assert st.design_gate({"c": _s(32, 1.0, 0.6)}, {"c": 0.0}, False, 0.9)["pass"] is False
    assert st.design_gate({"c": _s(32, 1.0, 0.6)}, {"c": 0.0}, True, 0.7)["pass"] is False


def test_sanity_summary():
    s = st.sanity_summary([True] * 9 + [False], [True] * 5, [True, True], [False, True])
    assert s["final_overwrite_rate"] == 0.9 and s["pass"] and s["shallow_overwrite_rate"] == 0.5
    assert not st.sanity_summary([True], [True, False], [True], [])["pass"]
    assert not st.sanity_summary([True], [True], [False], [])["pass"]


# --- scripts -------------------------------------------------------------------------------------------------------

def test_dev_refuses_confirmation_folds(capsys):
    from scripts import exp38_dev
    assert exp38_dev.main(["--folds", "0,2"]) == 2
    assert "REFUSED" in capsys.readouterr().out


@needs_cache
def test_population_counts_on_the_real_cache():
    from scripts.exp38_dev import load_source
    _, pop = load_source([0, 1], Path("runs/exp35"), "runs", Path("runs/exp34_jlens_own"))
    assert len(pop.R) == 28 and len(pop.pairs) == 13 and len(pop.unseen) == 15 and len(pop.C) == 39
    counts = {k: sum(v == k for v in pop.b_outcome.values()) for k in ("correct", "wrong", "refusal")}
    assert counts == {"correct": 8, "wrong": 5, "refusal": 15}
    assert pop.damage == sorted(pop.C)[:8] and len(pop.damage) == 8
    assert set(pop.pairs) | set(pop.unseen) == set(pop.R) and not set(pop.pairs) & set(pop.unseen)


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    """One stub design run on the real cache with the gates relaxed so every stage, the controls included, runs."""
    if not CACHE.exists():
        pytest.skip("§35 cache not on disk")
    from scripts import exp38_dev
    out = tmp_path_factory.mktemp("exp38")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "ANS_GATE", 0.0)
        assert exp38_dev.main(["--run", "--backend", "stub", "--out", str(out)]) == 0
    return out / "qwen3b"


def _lines(run_dir: Path) -> dict:
    return {p.name: len(p.read_text().splitlines()) for p in sorted(run_dir.glob("*.jsonl"))}


def test_stub_run_writes_every_stage_sha_stamped(stub_run):
    lines = _lines(stub_run)
    assert set(lines) == {"repro.jsonl", "sanity.jsonl", "sweep.jsonl", "controls.jsonl"}
    gate = json.loads((stub_run / "gate.json").read_text())
    assert gate["chosen"] is not None
    rows = [json.loads(x) for x in (stub_run / "controls.jsonl").read_text().splitlines()]
    kinds = {(r["pop"], r["kind"], r["setting"]["mode"]) for r in rows}
    assert {("R", "A", "reverse"), ("R", "cut", "none"), ("R", "cut", "steer"), ("C", "A", "steer"),
            ("C", "A", "random"), ("C", "A", "reverse")} <= kinds
    for r in rows:
        assert r["engine"] == "stub" and r["wording_sha256"] == st.wording_sha256(16) and len(r["input_sha256"]) == 64
        assert r["last_token"] == ":" and r["n_prefix_tokens"] == r["n_input_tokens"] - 1
        if r["setting"]["mode"] != "none":
            assert r["vector_sha256"]
    assert any(r["cut_no_marker"] for r in rows if r["kind"] == "cut")
    assert (stub_run / "resid" / "A").is_dir() and (stub_run / "directions.json").exists()


def test_stub_run_resumes_without_new_rows(stub_run):
    from scripts import exp38_dev
    before = _lines(stub_run)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "ANS_GATE", 0.0)
        assert exp38_dev.main(["--run", "--backend", "stub", "--out", str(stub_run.parent)]) == 0
    assert _lines(stub_run) == before


def test_stale_rows_and_manifest_are_refused(stub_run):
    row = json.loads((stub_run / "sweep.jsonl").read_text().splitlines()[0])
    it = st.SteerInput("A", row["question_id"], row["fold"], "p", "c", row["input_sha256"], "x")
    with pytest.raises(st.StaleRowError):
        st.check_row(row, it, "hf-other", row["wording_sha256"], None)
    with pytest.raises(st.StaleRowError):
        st.check_row(row, it, row["engine"], "different-wording", None)
    from scripts import exp38_dev
    with pytest.raises(SystemExit, match="REFUSED"):
        exp38_dev.main(["--run", "--backend", "stub", "--out", str(stub_run.parent), "--alphas", "3"])


def test_report_dev_writes_md_and_json(stub_run, tmp_path, monkeypatch):
    from scripts import exp38_report
    monkeypatch.setattr(st, "ANS_GATE", 0.0)   # as in the stub run, so the report re-derives the same choice
    out = tmp_path / "rep.txt"
    assert exp38_report.main(["--run-dir", str(stub_run), "--out", str(out), "--samples", "200"]) == 0
    md, js = tmp_path / "rep.md", tmp_path / "rep.json"
    assert md.exists() and js.exists()
    numbers = json.loads(js.read_text())
    assert numbers["sanity"]["pass"] and numbers["gate"]["chosen"] == json.loads((stub_run / "gate.json").read_text())["chosen"]
    assert "Steer-cut" in numbers["controls"] and "A-cut" in numbers["controls"]
    assert "PROVISIONAL" in md.read_text()
    assert exp38_report.out_paths(Path("a/b.md")) == (Path("a/b.md"), Path("a/b.json"))


def _copy_run(src: Path, dst: Path) -> Path:
    import shutil
    shutil.copytree(src, dst)
    return dst


def test_report_refuses_mixed_engines(stub_run, tmp_path, capsys):
    from scripts import exp38_report
    run = _copy_run(stub_run, tmp_path / "mixed")
    lines = (run / "controls.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["engine"] = "hf-transformers-x-cpu-float32"
    (run / "controls.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    assert exp38_report.main(["--run-dir", str(run), "--out", str(tmp_path / "r.md"), "--samples", "50"]) == 2
    assert "mixes engines" in capsys.readouterr().out


def test_report_confirm_guard(stub_run, tmp_path, capsys):
    from scripts import exp38_report
    assert exp38_report.main(["--run-dir", str(stub_run), "--confirm", "--out", str(tmp_path / "r.md")]) == 2
    assert "without confirmation-fold rows" in capsys.readouterr().out
    run = _copy_run(stub_run, tmp_path / "fold2")
    lines = (run / "controls.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["fold"] = 2
    (run / "controls.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    assert exp38_report.main(["--run-dir", str(run), "--out", str(tmp_path / "r.md")]) == 2
    assert "confirmation folds" in capsys.readouterr().out


# --- the real HF engine's plumbing on a tiny random Qwen2 (no weights, no download) -------------------------------

def _tiny_hf_engine():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("no cached Qwen2.5 tokenizer")
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    eng = st.HFEngine.__new__(st.HFEngine)
    eng.torch, eng.model, eng.device, eng.name = torch, "tiny", "cpu", "hf-tiny"
    eng.tokenizer = tok
    eng.net = transformers.Qwen2ForCausalLM(cfg).eval()
    eng.layers, eng.n_layers, eng.hidden = eng.net.model.layers, 4, 32
    eng.stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return eng


def test_hf_engine_prefix_plus_last_step_matches_a_full_forward():
    eng = _tiny_hf_engine()
    torch = eng.torch
    it = item("Question: what is my cat's name?\n\nNote:", "She is Luna.\nAnswer:")
    prepared = eng.prepare(it)
    assert eng.last_token_text(prepared) == ":" and prepared.n_prefix == len(prepared.ids) - 1
    with torch.no_grad():
        full = eng.net(input_ids=torch.tensor([prepared.ids])).logits[0, -1]
        cache = __import__("copy").deepcopy(prepared.cache)
        stepped = eng._step(prepared.last_id, cache)
    assert torch.allclose(full, stepped, atol=1e-4)
    base = eng.run(prepared, [], 6, capture=tuple(range(4)))
    again = eng.run(prepared, [], 6)
    assert again.gen_ids == base.gen_ids               # the deep-copied prefix is reusable
    own = np.stack([base.captured[L] for L in range(4)])
    for spec in (st.add_spec(2, np.zeros(32, np.float32)), st.overwrite_spec(3, own[3]), st.overwrite_spec(1, own[1])):
        g = eng.run(prepared, [spec], 6)
        assert g.generated == base.generated and g.gen_ids == base.gen_ids
    pushed = eng.run(prepared, [st.add_spec(1, np.full(32, 5.0, np.float32))], 6, capture=(1,))
    assert np.allclose(pushed.captured[1], own[1] + 5.0, atol=1e-4)
    assert all(len(layer._forward_hooks) == 0 for layer in eng.layers)


def test_report_confirm_path_runs_the_claims(stub_run, tmp_path, monkeypatch):
    """The confirm branch exercised on DEVELOPMENT rows only (the fold sets are relabelled for the test; no
    confirmation-fold file is opened)."""
    from scripts import exp38_report
    monkeypatch.setattr(st, "ANS_GATE", 0.0)
    monkeypatch.setattr(exp38_report, "CONFIRM_FOLDS", {0, 1})
    text, numbers = exp38_report.build(stub_run, True, samples=200, design=stub_run / "gate.json")
    c1 = numbers["claim1"]
    assert c1["n"] == numbers["populations"]["R_reproduced"] and c1["underpowered"] is True
    assert "verdict" in c1 and "n" in numbers["claim1_prime"] and numbers["claim1_prime"]["n"] <= c1["n"]
    assert set(numbers["control_T"]) and "Control X" in text and "Claim 1′" in text
    assert exp38_report.main(["--run-dir", str(stub_run), "--confirm", "--out", str(tmp_path / "c.md")]) == 2
