"""§39b: the forced-first-token path (no hook at any step), the Unknown-first and step-0 stops, the reading arithmetic,
fold refusal, the §39 source guard on the real rows, resume and an end-to-end stub run (scripts/exp39b_dev.py,
scripts/exp39b_report.py). StubEngine only; no Qwen weights are loaded."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import steering as st
from tests.test_exp39 import CACHE, SOURCE, _dev_data, _item, _lines, _make_stub_source, _NamedEngine

SOURCE39 = Path("runs/exp39_dev/qwen3b")
needs_data = pytest.mark.skipif(not (CACHE.exists() and (SOURCE / "directions.npz").exists()
                                     and (SOURCE39 / "arms.jsonl").exists()),
                                reason="§35 cache, §38 or §39 dev run not on disk")


# --- the forced path ------------------------------------------------------------------------------------------

def test_forced_token_is_gen_ids_0_and_continuation_is_greedy():
    eng = st.StubEngine()
    prepared = eng.prepare(_item())
    natural = eng.run(prepared, [], 12)
    forced_id = ord("q") if natural.first_token != ord("q") else ord("z")
    g = eng.run_forced(prepared, forced_id, 12)
    assert g.gen_ids[0] == forced_id and g.first_token == forced_id and g.forced == forced_id
    assert g.natural_first == natural.first_token and g.generated == eng.decode(g.gen_ids)
    assert 1 <= g.n_new <= 12 and g.captured == {}
    # the continuation is the greedy one from the forced token: replay it step by step without hooks
    import copy
    import torch
    cache = copy.deepcopy(prepared.cache)
    with torch.no_grad():
        eng._step(prepared.last_id, cache)
        ids = [forced_id]
        while "\n" not in eng.decode(ids) and len(ids) < 12:
            nxt = int(eng._step(ids[-1], cache).argmax())
            if nxt in eng.stop_ids:
                break
            ids.append(nxt)
    assert ids == g.gen_ids


def test_forced_equals_unforced_when_forced_is_the_natural_first_token():
    eng = st.StubEngine()
    for p, c in (("Question: what? \n\nNote:", "the note\nAnswer:"), ("Another one", "x\nAnswer:")):
        item = st.SteerInput("A", "q", 0, p, c, st.sha256(json.dumps([p, c])), "x")
        prepared = eng.prepare(item)
        for max_new in (1, 3, 16):
            natural = eng.run(prepared, [], max_new)
            forced = eng.run_forced(prepared, natural.first_token, max_new)
            assert forced.gen_ids == natural.gen_ids and forced.generated == natural.generated
            assert forced.natural_first == natural.first_token


def test_no_hook_fires_at_any_step(monkeypatch):
    eng = st.StubEngine()
    prepared = eng.prepare(_item())

    def boom(*a, **k):
        raise AssertionError("hooked() called on the forced path")
    monkeypatch.setattr(st, "hooked", boom)
    g = eng.run_forced(prepared, ord("a"), 16)          # works without the hook machinery
    monkeypatch.undo()
    # a registered hook (from anywhere) is refused, not silently applied
    calls = []
    handle = eng.layers[3].register_forward_hook(lambda m, i, o: calls.append(1))
    try:
        with pytest.raises(RuntimeError, match="forward hooks registered"):
            eng.run_forced(prepared, ord("a"), 16)
    finally:
        handle.remove()
    assert calls == []
    assert eng.run_forced(prepared, ord("a"), 16).gen_ids == g.gen_ids
    # hooks used by an earlier hooked run are removed, so a forced run after it is clean and unchanged
    eng.run(prepared, [st.add_spec(5, np.full(16, 3.0, np.float32))], 6)
    assert eng.run_forced(prepared, ord("a"), 16).gen_ids == g.gen_ids


def test_newline_in_the_forced_token_stops_at_once_and_stop_ids_cannot_be_forced():
    eng = st.StubEngine()
    prepared = eng.prepare(_item())
    g = eng.run_forced(prepared, ord("\n"), 16)
    assert g.gen_ids == [ord("\n")] and g.generated == "\n"
    assert eng.run_forced(prepared, ord("a"), 1).gen_ids == [ord("a")]          # max counts the forced token
    with pytest.raises(ValueError):
        eng.run_forced(prepared, st.StubEngine.EOS, 16)
    with pytest.raises(ValueError):
        eng.run_forced(prepared, ord("a"), 0)


def test_continuation_first_id_and_forced_row():
    eng = st.StubEngine()
    assert st.continuation_first_id(eng, "x\nAnswer:", " unknown") == ord(" ")

    class Tok:                                   # a tokenizer that re-tokenises the continuation: refused
        def encode(self, s):
            return [len(s)]
    with pytest.raises(ValueError):
        st.continuation_first_id(Tok(), "Answer:", " x")
    item = _item()
    prepared = eng.prepare(item)
    g = eng.run_forced(prepared, ord("b"), 16)
    row = st.make_forced_row("forced", "R", item, st.ForcedArm("steer_first"), g, eng, prepared, 16, "gold",
                             {"from": "test"})
    assert row["setting"]["key"] == "forced:steer_first" and row["vector_sha256"] is None
    assert row["forced_token"] == ord("b") and row["forced_text"] == "b" and row["gen_ids"][0] == ord("b")
    assert row["natural_first_token"] == g.natural_first and row["forced_source"] == {"from": "test"}


# --- the reading ----------------------------------------------------------------------------------------------

def test_reading_arithmetic_and_boundaries():
    from scripts.exp39b_report import reading, reading_line, thresholds
    assert thresholds(26, 7) == {"carries_aF_min": 21, "carries_gF_min": 6, "state_aF_max": 12, "split_gF_max": 3}
    v = lambda aF, gF: reading(28, aF, gF, 26, 7)["verdict"]  # noqa: E731
    assert v(21, 6) == "token carries it"                 # both boundaries met exactly (105 >= 104, 30 >= 28)
    assert v(28, 7) == "token carries it"
    assert v(20, 7) == "undetermined on 28 rows"          # 100 < 104
    assert v(21, 5) == "undetermined on 28 rows"          # 25 < 28, 10 >= 7: between
    assert v(21, 4) == "undetermined on 28 rows"          # 8 >= 7: not a split
    assert v(21, 3) == "split"                            # 6 < 7
    assert v(26, 0) == "split"
    assert v(12, 7) == "state needed"                     # 24 < 26, checked first
    assert v(12, 0) == "state needed"
    assert v(13, 0) == "undetermined on 28 rows"          # 26 = a0, not below
    assert reading(28, 21, 3, 26, 7)["meaning"].startswith("the token keeps the model answering")
    line = reading_line(reading(28, 21, 3, 26, 7))
    assert "aF = 21/28" in line and "a0 = 26/28" in line and "aF <= 12/28" in line and "gF >= 6/28" in line \
        and "gF <= 3/28" in line


def test_soft_refusal_and_divergence():
    from scripts.exp39b_report import SOFT_REFUSAL_PATTERN, divergence, soft_refusal
    assert SOFT_REFUSAL_PATTERN == (r"not (specified|mentioned|provided|determined|explicitly|contain|named|available|stated|given)"
                                    r"|does not contain|cannot be determined|no (information|mention)")
    assert soft_refusal({"cls": "answer", "phrase": "First event participation cannot be determined"})
    assert soft_refusal({"cls": "answer", "phrase": "Favorite rice type NOT specified"})
    assert soft_refusal({"cls": "answer", "phrase": "The memory does not contain information about it"})
    assert not soft_refusal({"cls": "refusal", "phrase": "not mentioned"})
    assert not soft_refusal({"cls": "answer", "phrase": "Serenity Yoga"})
    assert divergence([1, 2, 3], [1, 2, 3]) is None
    assert divergence([1, 2, 3], [1, 5, 3]) == 1
    assert divergence([1, 2], [1, 2, 3]) == 2
    assert divergence([4], [1]) == 0


# --- folds, sources -------------------------------------------------------------------------------------------

def test_dev_refuses_confirmation_folds(capsys):
    from scripts import exp39b_dev
    assert exp39b_dev.main(["--folds", "0,2"]) == 2
    assert "REFUSED" in capsys.readouterr().out


@needs_data
def test_source39_guard_accepts_the_real_rows_and_refuses_another_engine(tmp_path):
    from scripts import exp39_dev, exp39b_dev
    d, pop = _dev_data()
    direction = exp39_dev.load_direction(SOURCE)
    run = exp39b_dev.Run(_NamedEngine("hf-transformers-5.17.0-cpu-float32"), d, pop, tmp_path, SOURCE, SOURCE39,
                         direction)
    run.engine.stop_ids = {151645, 151643}
    info = run.load_steer0(pop.R)
    assert info["n"] == 56 and len((tmp_path / "qwen3b" / "steer0.jsonl").read_text().splitlines()) == 56
    assert run.steer0_rows[("A", "4388e9dd")]["gen_ids"][0] == 24832
    other = exp39b_dev.Run(_NamedEngine("hf-transformers-5.17.0-cuda-bfloat16"), d, pop, tmp_path / "o", SOURCE,
                           SOURCE39, direction)
    with pytest.raises(SystemExit, match="REFUSED \\(§39 Steer-0 rows\\).*engine"):
        other.load_steer0(pop.R)


@needs_data
def test_real_39_rows_give_the_preregistered_a0_g0():
    from scripts.exp39b_report import EXPECTED_A0, EXPECTED_G0, block
    from scripts.exp35_generate import read_jsonl
    _, pop = _dev_data()
    rows = {r["question_id"]: r for r in read_jsonl(SOURCE39 / "arms.jsonl")
            if r["pop"] == "R" and r["kind"] == "A" and r["setting"]["key"] == "steer:L29:a2:vnorm:s0"}
    b = block(rows, pop.R)
    assert (b["n"], b["ans_k"], b["gold_k"]) == (28, EXPECTED_A0, EXPECTED_G0) == (28, 26, 7)
    assert b["soft_k"] == 4                              # the §39 review's count


# --- end to end on the stub -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stub_sources(tmp_path_factory):
    if not (CACHE.exists() and (SOURCE / "directions.npz").exists()):
        pytest.skip("§35 cache or §38 dev run not on disk")
    from scripts import exp39_dev
    root = tmp_path_factory.mktemp("exp39b_src")
    src38 = _make_stub_source(root / "src")
    assert exp39_dev.main(["--run", "--backend", "stub", "--source", str(src38), "--out", str(root / "s39")]) == 0
    return src38, root / "s39" / "qwen3b"


def _main39b(src38, src39, out, extra=()):
    from scripts import exp39b_dev
    return exp39b_dev.main(["--run", "--backend", "stub", "--source", str(src38), "--source39", str(src39),
                            "--out", str(out), *extra])


@pytest.fixture(scope="module")
def stub39b(stub_sources, tmp_path_factory):
    src38, src39 = stub_sources
    out = tmp_path_factory.mktemp("exp39b") / "out"
    assert _main39b(src38, src39, out) == 0
    return src38, src39, out / "qwen3b"


def test_stub_run_writes_every_arm_sha_stamped(stub39b):
    src38, src39, run_dir = stub39b
    assert _lines(run_dir) == {"forced.jsonl": 28 * 3, "repro.jsonl": 28 * 2 + 8, "step0.jsonl": 28,
                               "steer0.jsonl": 56, "unknown_first.jsonl": 28}
    steer0 = {(r["kind"], r["question_id"]): r for r in map(json.loads, (run_dir / "steer0.jsonl").read_text().splitlines())}
    src_rows = [json.loads(x) for x in (src39 / "arms.jsonl").read_text().splitlines()]
    s0key = f"steer:L{st.map_layer(29, 20)}:a2:vnorm:s0"
    picked = [r for r in src_rows if r["setting"]["key"] == s0key and r["pop"] == "R"]
    assert len(picked) == 56 and all(steer0[(r["kind"], r["question_id"])] == r for r in picked)   # verbatim
    forced = [json.loads(x) for x in (run_dir / "forced.jsonl").read_text().splitlines()]
    assert {(r["kind"], r["setting"]["key"]) for r in forced} == {("A", "forced:steer_first"), ("A", "forced:gold_first"),
                                                                  ("cut", "forced:steer_first_cut")}
    for r in forced:
        assert r["engine"] == "stub" and r["wording_sha256"] == st.wording_sha256(16) and r["vector_sha256"] is None
        assert r["gen_ids"][0] == r["forced_token"] and r["last_token"] == ":" and r["max_new_tokens"] == 16
        assert len(r["gen_ids"]) <= 16
        if r["setting"]["key"] == "forced:steer_first":
            assert r["forced_token"] == steer0[("A", r["question_id"])]["gen_ids"][0]
        if r["setting"]["key"] == "forced:steer_first_cut":
            assert r["forced_token"] == steer0[("cut", r["question_id"])]["gen_ids"][0]
        if r["setting"]["key"] == "forced:gold_first":
            assert r["forced_source"]["agree"] and r["forced_text"] == " "
    step0 = [json.loads(x) for x in (run_dir / "step0.jsonl").read_text().splitlines()]
    assert all(r["max_new_tokens"] == 1 and r["setting"]["steps"] == [0] and r["n_input_tokens"] for r in step0)
    assert all(r["first_token"] == steer0[("A", r["question_id"])]["gen_ids"][0] for r in step0)
    assert json.loads((run_dir / "unknown_first.json").read_text())["k"] == 28
    man = json.loads((run_dir / "manifest.json").read_text())
    from scripts import exp39b_dev
    assert man["source39_arms_sha256"] == exp39b_dev.file_sha256(src39 / "arms.jsonl")


def test_stub_run_resumes_without_new_rows(stub39b):
    src38, src39, run_dir = stub39b
    before = _lines(run_dir)
    assert _main39b(src38, src39, run_dir.parent) == 0
    assert _lines(run_dir) == before


def test_unknown_first_mismatch_exits_4_before_any_other_arm(stub_sources, tmp_path, monkeypatch):
    from scripts import exp39b_dev
    src38, src39 = stub_sources
    real = st.StubEngine.run_forced

    def wrong(self, prepared, forced, max_new_tokens):          # a forcing path that drops the continuation
        g = real(self, prepared, forced, max_new_tokens)
        g.gen_ids, g.generated = g.gen_ids[:1], self.decode(g.gen_ids[:1])
        return g
    monkeypatch.setattr(st.StubEngine, "run_forced", wrong)
    out = tmp_path / "out"
    assert _main39b(src38, src39, out) == exp39b_dev.EXIT_UNKNOWN == 4
    assert not (out / "qwen3b" / "step0.jsonl").exists() and not (out / "qwen3b" / "forced.jsonl").exists()
    monkeypatch.undo()
    monkeypatch.setattr(exp39b_dev.Run, "unknown_id", lambda self, q: ord("x"))   # a tokenisation that is not A's
    out2 = tmp_path / "out2"
    assert _main39b(src38, src39, out2) == 4
    assert not (out2 / "qwen3b" / "unknown_first.jsonl").exists()
    assert "tokenises" in json.loads((out2 / "qwen3b" / "unknown_first.json").read_text())["why"]


def test_step0_mismatch_exits_5_before_any_forced_arm(stub_sources, tmp_path):
    from scripts import exp39b_dev
    src38, src39 = stub_sources
    bad = tmp_path / "s39"
    shutil.copytree(src39, bad)
    rows = [json.loads(x) for x in (bad / "arms.jsonl").read_text().splitlines()]
    hit = next(r for r in rows if r["pop"] == "R" and r["kind"] == "A" and r["setting"]["key"].startswith("steer:")
               and r["setting"]["key"].endswith(":s0"))
    t = (hit["gen_ids"][0] + 1) % 256
    hit["gen_ids"][0] = hit["first_token"] = t
    (bad / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    assert _main39b(src38, bad, out) == exp39b_dev.EXIT_STEP0 == 5
    assert (out / "qwen3b" / "unknown_first.jsonl").exists() and not (out / "qwen3b" / "forced.jsonl").exists()


def test_byte_stop_exits_6_before_any_forced_generation(stub_sources, tmp_path):
    from scripts import exp39b_dev
    src38, src39 = stub_sources
    bad = tmp_path / "src"
    shutil.copytree(src38, bad)
    rows = [json.loads(x) for x in (bad / "repro.jsonl").read_text().splitlines()]
    for r in rows[:10]:
        r["gen_ids"] = r["gen_ids"] + [0]
    (bad / "repro.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    assert _main39b(bad, src39, out) == exp39b_dev.EXIT_BYTES == 6
    assert not (out / "qwen3b" / "unknown_first.jsonl").exists()


def test_source39_row_mismatch_is_refused(stub_sources, tmp_path):
    src38, src39 = stub_sources
    bad = tmp_path / "s39"
    shutil.copytree(src39, bad)
    rows = [json.loads(x) for x in (bad / "arms.jsonl").read_text().splitlines()]
    hit = next(r for r in rows if r["pop"] == "R" and r["kind"] == "cut" and r["setting"]["key"].startswith("steer:")
               and r["setting"]["key"].endswith(":s0"))
    hit["input_sha256"] = "0" * 64
    (bad / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="REFUSED \\(§39 Steer-0 rows\\).*input_sha256"):
        _main39b(src38, bad, tmp_path / "out")


def test_report_writes_md_and_json(stub39b, tmp_path):
    from scripts import exp39b_report
    _, _, run_dir = stub39b
    out = tmp_path / "r.md"
    assert exp39b_report.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    text = out.read_text()
    numbers = json.loads(out.with_suffix(".json").read_text())
    assert "DEVELOPMENT ONLY" in text and "PROVISIONAL" in text and "skipped on the stub" in text
    assert numbers["unknown_check"] == {"n": 28, "k": 28, "natural_first_is_forced_k": 28}
    assert numbers["step0_check"] == {"n": 28, "k": 28}
    r = numbers["reading"]
    assert r["n"] == 28 and (r["verdict"] in ("token carries it", "state needed", "split")
                             or r["verdict"].startswith("undetermined"))
    assert set(numbers["soft_refusals"]) == {"Steer-first", "Steer-0 (§39)", "Steer-first-cut", "Gold-first"}
    assert numbers["identity"]["n"] == 28 and len(numbers["identity"]["rows"]) == 28
    assert numbers["n_pairs"] + numbers["n_unseen"] == numbers["n_R"] == 28
    assert "| question | forced | Steer-first | Steer-0 | same text | divergence | Gold-first |" in text


def test_report_refuses_mixed_engines_and_other_folds(stub39b, tmp_path, capsys):
    from scripts import exp39b_report
    _, _, run_dir = stub39b
    for field, value in (("engine", "hf-other"), ("fold", 2)):
        bad = tmp_path / field
        shutil.copytree(run_dir, bad)
        rows = [json.loads(x) for x in (bad / "forced.jsonl").read_text().splitlines()]
        rows[0][field] = value
        (bad / "forced.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert exp39b_report.main(["--run-dir", str(bad)]) == 2
        assert "REFUSED" in capsys.readouterr().out
