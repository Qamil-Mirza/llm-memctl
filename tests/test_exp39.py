"""§39: step-limited hooks, setting keys, the Steer-all reuse guard, the §3 reading arithmetic, fold refusal, resume,
the byte-identity stop and an end-to-end stub run reading the real §38 dev directions.npz (scripts/exp39_dev.py,
scripts/exp39_report.py). StubEngine only; no Qwen weights are loaded."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import steering as st

CACHE = Path("runs/exp35/qwen3b/A/fold0.jsonl")
SOURCE = Path("runs/exp38_dev/qwen3b")
needs_data = pytest.mark.skipif(not (CACHE.exists() and (SOURCE / "directions.npz").exists()),
                                reason="§35 cache or §38 dev run not on disk")


# --- hooks and keys -------------------------------------------------------------------------------------------

def _item():
    p, c = "Question: what? \n\nNote:", "the note\nAnswer:"
    return st.SteerInput("A", "q1", 0, p, c, st.sha256(json.dumps([p, c])), "x")


def _residuals(eng, specs, layer, n_steps=4):
    """Residual at `layer` per step, feeding the same token each step (so only the hook differs)."""
    import torch
    prepared = eng.prepare(_item())
    cache = copy.deepcopy(prepared.cache)
    state = st.HookState()
    out = []
    with torch.no_grad(), st.hooked(eng.layers, list(specs) + [st.HookSpec(layer, "capture", None, None)], state):
        for step in range(n_steps):
            state.step = step
            eng._step(prepared.last_id, cache)
            out.append(state.captured[layer].copy())
    return out


@pytest.mark.parametrize("steps", [(0,), (0, 1), (0, 1, 2)])
def test_step_limited_add_acts_only_on_the_listed_steps(steps):
    eng = st.StubEngine()
    vec = np.linspace(-1, 1, eng.hidden).astype(np.float32)
    base = _residuals(eng, [], 5)
    pushed = _residuals(eng, [st.add_spec(5, vec, steps)], 5)
    for step in range(4):
        diff = pushed[step] - base[step]
        if step in steps:
            assert np.allclose(diff, vec, atol=1e-5)
        else:
            assert np.array_equal(pushed[step], base[step])


def test_setting_steps_reach_the_hook_and_every_step_is_unchanged():
    s0 = st.Setting("steer", 5, 2.0, "vnorm", st.steps_upto(0))
    vec = np.ones(16, np.float32)
    vectors = {(5, 2.0, "vnorm"): vec}
    (spec,) = st.specs_for(s0, vectors)
    assert spec.steps == frozenset({0}) and spec.mode == "add"
    (spec_all,) = st.specs_for(st.Setting("steer", 5, 2.0, "vnorm"), vectors)
    assert spec_all.steps is None
    (rnd,) = st.specs_for(st.Setting("random", 5, 2.0, "vnorm", (0,)), vectors)
    assert rnd.steps == frozenset({0}) and np.isclose(np.linalg.norm(rnd.vector), np.linalg.norm(vec))
    eng = st.StubEngine()
    prepared = eng.prepare(_item())
    g_all = eng.run(prepared, [st.add_spec(5, np.full(16, 3.0, np.float32))], 6)
    g_0 = eng.run(prepared, [st.add_spec(5, np.full(16, 3.0, np.float32), (0,))], 6)
    assert g_all.first_token == g_0.first_token           # step 0 is the same forward pass in both


def test_setting_keys():
    S = st.Setting
    assert S("steer", 29, 2.0, "vnorm").key == "steer:L29:a2:vnorm"                  # §38 key unchanged
    assert "steps" not in S("steer", 29, 2.0, "vnorm").to_dict()                      # §38 dict unchanged
    assert S("steer", 29, 2.0, "vnorm", (0,)).key == "steer:L29:a2:vnorm:s0"
    assert S("steer", 29, 2.0, "vnorm", (0, 1, 2)).key == "steer:L29:a2:vnorm:s0-2"
    assert S("steer", 29, 2.0, "vnorm", (0, 1, 2, 3, 4)).key == "steer:L29:a2:vnorm:s0-4"
    assert S("random", 29, 2.0, "vnorm", (0,)).key == "random:L29:a2:vnorm:s0"
    assert S("steer", 29, 2.0, "vnorm", (0, 3)).key == "steer:L29:a2:vnorm:s0,3"
    assert S("steer", 29, 2.0, "vnorm", (0, 1)).to_dict()["steps"] == [0, 1]
    for key in ("steer:L29:a2:vnorm", "steer:L29:a2:vnorm:s0", "steer:L29:a2:vnorm:s0-4", "random:L29:a2:vnorm:s0,3"):
        assert st.setting_from_key(key).key == key
    assert st.setting_from_key("steer:L29:a2:vnorm:s0-2").steps == (0, 1, 2)
    with pytest.raises(ValueError):
        st.steps_label(())
    from scripts.exp39_dev import Arms
    arms = Arms(29)
    assert [s.key for s in arms.a_prompt()] == ["none", "steer:L29:a2:vnorm:s0", "steer:L29:a2:vnorm:s0-1",
                                               "steer:L29:a2:vnorm:s0-2", "steer:L29:a2:vnorm:s0-4",
                                               "random:L29:a2:vnorm:s0"]
    assert [s.key for s in arms.cut_prompt()] == ["none", "steer:L29:a2:vnorm:s0", "steer:L29:a2:vnorm"]
    assert [s.key for s in Arms(29, (2,)).a_prompt()][2:4] == ["steer:L29:a2:vnorm:s0-2", "random:L29:a2:vnorm:s0"]


# --- the §3 reading -------------------------------------------------------------------------------------------

def test_reading_arithmetic_and_boundaries():
    from scripts.exp39_report import reading, thresholds
    t = thresholds(26, 8, 21)
    assert t == {"m1_a0_min": 21, "m1_g0_min": 7, "m1_r0_max": 4, "m2_a0_max": 12, "split_g0_max": 3}
    assert reading(28, 21, 26, 7, 8, 4)["verdict"] == "M1 supported"        # every boundary met exactly
    assert reading(28, 21, 26, 6, 8, 0)["verdict"] == "undetermined on 28 rows"   # g0 = 6 < 6.4
    assert reading(28, 21, 26, 7, 8, 5)["verdict"] == "undetermined on 28 rows"   # Random 5 > 4.2
    assert reading(28, 20, 26, 7, 8, 0)["verdict"] == "undetermined on 28 rows"   # a0 = 20 < 20.8
    assert reading(28, 21, 26, 3, 8, 0)["verdict"] == "split"                      # g0 = 3 < 4
    assert reading(28, 21, 26, 4, 8, 0)["verdict"] == "undetermined on 28 rows"   # g0 = 4 = 0.5 gA, not below
    assert reading(28, 12, 26, 0, 8, 0)["verdict"] == "M2 supported"               # 12 < 13
    assert reading(28, 13, 26, 0, 8, 0)["verdict"] == "undetermined on 28 rows"   # 13 = 0.5 aA, not below
    assert reading(28, 0, 0, 0, 0, 0)["verdict"] == "M1 supported"                 # degenerate edge: all zero
    assert reading(28, 5, 5, 0, 0, 1)["verdict"] == "M1 supported"                 # r0 = 1 <= 0.2 * 5
    assert reading(28, 21, 26, 7, 8, 4)["meaning"].startswith("the pushed ':' state carries the answer path")
    from scripts.exp39_report import reading_line
    line = reading_line(reading(28, 21, 26, 3, 8, 0))
    assert "g0 = 3/28" in line and "gA = 8/28" in line and "g0 >= 7/28" in line and "g0 <= 3/28" in line


# --- direction ------------------------------------------------------------------------------------------------

@needs_data
def test_direction_is_the_38_file_byte_exact(tmp_path):
    from scripts import exp39_dev
    d = exp39_dev.load_direction(SOURCE)
    assert d["npz_sha256"] == exp39_dev.SOURCE_NPZ_SHA256 and st.vector_sha(d["v"]) == exp39_dev.V29_SHA256
    assert np.array_equal(d["steer"], (2.0 * d["v"].astype(np.float64)).astype(np.float32))
    assert st.vector_sha(d["steer"]) == exp39_dev.STEER_VECTOR_SHA256
    assert st.vector_sha(st.random_vector(d["steer"], 29)) == exp39_dev.RANDOM_VECTOR_SHA256
    z = dict(np.load(SOURCE / "directions.npz"))
    z["L29"] = z["L29"] * 1.0001
    np.savez(tmp_path / "directions.npz", **z)
    with pytest.raises(SystemExit, match="REFUSED"):
        exp39_dev.load_direction(tmp_path)


class _NamedEngine:
    """Enough of an engine for the reuse guard and the inputs (no generation)."""

    def __init__(self, name, hidden=2048, n_layers=36):
        self.name, self.model, self.hidden, self.n_layers = name, "Qwen/Qwen2.5-3B-Instruct", hidden, n_layers


def _dev_data():
    from scripts.exp38_dev import load_source
    return load_source([0, 1], Path("runs/exp35"), "runs", Path("runs/exp34_jlens_own"))


@needs_data
def test_reuse_guard_accepts_the_real_38_rows_and_refuses_another_engine(tmp_path):
    from scripts import exp39_dev
    d, pop = _dev_data()
    direction = exp39_dev.load_direction(SOURCE)
    run = exp39_dev.Run(_NamedEngine("hf-transformers-5.17.0-cpu-float32"), d, pop, tmp_path, SOURCE, direction)
    info = run.reuse_steer_all(pop.R)
    assert info["n"] == 28 and info["vector_sha256"] == exp39_dev.STEER_VECTOR_SHA256
    assert len((tmp_path / "qwen3b" / "steer_all.jsonl").read_text().splitlines()) == 28
    other = exp39_dev.Run(_NamedEngine("hf-transformers-5.17.0-cuda-bfloat16"), d, pop, tmp_path / "o", SOURCE,
                          direction)
    with pytest.raises(SystemExit, match="REFUSED \\(Steer-all reuse\\).*engine"):
        other.reuse_steer_all(pop.R)
    short = exp39_dev.Run(_NamedEngine("hf-transformers-5.17.0-cpu-float32"), d, pop, tmp_path / "s", SOURCE,
                          direction, max_new=64)
    with pytest.raises(SystemExit, match="REFUSED \\(Steer-all reuse\\)"):
        short.reuse_steer_all(pop.R)


@needs_data
def test_real_38_repro_rows_against_themselves():
    from scripts import exp39_dev
    from scripts.exp35_generate import read_jsonl
    d, pop = _dev_data()
    src = {st.row_key(r): r for r in read_jsonl(SOURCE / "repro.jsonl")}
    rep = exp39_dev.compare_repro(src, src, d, pop)
    assert rep["byte_rate"] == 1.0 and rep["byte_n"] == 64
    assert len(rep["R_ok"]) == 28 and len(rep["damage_ok"]) == 4 and abs(rep["rate"] - 32 / 36) < 1e-12


# --- end to end on the stub -----------------------------------------------------------------------------------

def _make_stub_source(root: Path) -> Path:
    """A §38-shaped stub source: the real directions.npz (bytes), stub repro rows, stub Steer-all sweep rows at the
    stub's layer 29 with the vector exp39 adds on the stub, the unsteered cut rows and a manifest."""
    from scripts import exp38_dev, exp39_dev
    d, pop = _dev_data()
    eng = st.StubEngine()
    layer = st.map_layer(29, eng.n_layers)
    run = exp38_dev.Run(eng, d, pop, root, [layer], [2.0], "vnorm", 16, log=lambda s: None)
    run.dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / "directions.npz", run.dir / "directions.npz")
    direction = exp39_dev.load_direction(run.dir)
    run.vectors[(layer, 2.0, "vnorm")] = exp39_dev.engine_vector(direction["steer"], eng)
    eng.set_oracle(exp38_dev.stub_oracle(run))
    run.check_manifest()
    none, steer = st.Setting("none"), st.Setting("steer", layer, 2.0, "vnorm")
    for q in pop.R:
        run.do("repro", "R", "A", q, [none])
        run.do("repro", "R", "B", q, [none])
        run.do("sweep", "R", "A", q, [steer])
        run.do("controls", "R", "cut", q, [none])
    for q in pop.damage:
        run.do("repro", "C", "A", q, [none])
    return run.dir


@pytest.fixture(scope="module")
def stub39(tmp_path_factory):
    if not (CACHE.exists() and (SOURCE / "directions.npz").exists()):
        pytest.skip("§35 cache or §38 dev run not on disk")
    from scripts import exp39_dev
    root = tmp_path_factory.mktemp("exp39")
    src = _make_stub_source(root / "src")
    out = root / "out"
    assert exp39_dev.main(["--run", "--backend", "stub", "--source", str(src), "--out", str(out)]) == 0
    return src, out / "qwen3b"


def _lines(run_dir: Path) -> dict:
    return {p.name: len(p.read_text().splitlines()) for p in sorted(run_dir.glob("*.jsonl"))}


def test_dev_refuses_confirmation_folds(capsys):
    from scripts import exp39_dev
    assert exp39_dev.main(["--folds", "0,3"]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_stub_run_writes_every_arm_sha_stamped(stub39):
    src, run_dir = stub39
    assert _lines(run_dir) == {"arms.jsonl": 28 * 6 + 28 * 3 + 8, "repro.jsonl": 28 * 2 + 8, "steer_all.jsonl": 28}
    rows = [json.loads(x) for x in (run_dir / "arms.jsonl").read_text().splitlines()]
    keys = {(r["pop"], r["kind"], r["setting"]["key"]) for r in rows}
    L = st.map_layer(29, 20)
    assert keys == ({("R", "A", k) for k in ("none", f"steer:L{L}:a2:vnorm:s0", f"steer:L{L}:a2:vnorm:s0-1",
                                             f"steer:L{L}:a2:vnorm:s0-2", f"steer:L{L}:a2:vnorm:s0-4",
                                             f"random:L{L}:a2:vnorm:s0")}
                    | {("R", "cut", k) for k in ("none", f"steer:L{L}:a2:vnorm:s0", f"steer:L{L}:a2:vnorm")}
                    | {("C", "A", f"steer:L{L}:a2:vnorm:s0")})
    for r in rows:
        assert r["engine"] == "stub" and r["wording_sha256"] == st.wording_sha256(16) and len(r["input_sha256"]) == 64
        assert r["last_token"] == ":"
        if r["setting"]["mode"] != "none":
            assert r["vector_sha256"]
        if r["setting"]["key"].endswith(":s0"):
            assert r["setting"]["steps"] == [0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    from scripts import exp39_dev
    assert manifest["npz_sha256"] == exp39_dev.SOURCE_NPZ_SHA256 and manifest["v_sha256"] == exp39_dev.V29_SHA256
    rep = json.loads((run_dir / "repro.json").read_text())
    assert rep["byte_rate"] == 1.0 and rep["pass"]


def test_stub_run_resumes_without_new_rows(stub39):
    from scripts import exp39_dev
    src, run_dir = stub39
    before = _lines(run_dir)
    assert exp39_dev.main(["--run", "--backend", "stub", "--source", str(src), "--out", str(run_dir.parent)]) == 0
    assert _lines(run_dir) == before
    with pytest.raises(SystemExit, match="REFUSED"):     # another design in the same directory
        exp39_dev.main(["--run", "--backend", "stub", "--source", str(src), "--out", str(run_dir.parent), "--k-small"])


def test_reuse_refused_on_a_mismatched_vector_sha(stub39, tmp_path):
    from scripts import exp39_dev
    src, _ = stub39
    bad = tmp_path / "src"
    shutil.copytree(src, bad)
    rows = [json.loads(x) for x in (bad / "sweep.jsonl").read_text().splitlines()]
    rows[3]["vector_sha256"] = "0" * 64
    (bad / "sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="REFUSED \\(Steer-all reuse\\).*vector_sha256"):
        exp39_dev.main(["--run", "--backend", "stub", "--source", str(bad), "--out", str(tmp_path / "out")])
    man = json.loads((src / "manifest.json").read_text())
    man["engine"] = "hf-transformers-5.17.0-cpu-float32"
    shutil.copyfile(src / "sweep.jsonl", bad / "sweep.jsonl")
    (bad / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="REFUSED \\(Steer-all reuse\\).*engine"):
        exp39_dev.main(["--run", "--backend", "stub", "--source", str(bad), "--out", str(tmp_path / "out2")])


def test_byte_identity_below_090_stops_before_any_steered_generation(stub39, tmp_path, capsys):
    from scripts import exp39_dev
    src, _ = stub39
    bad = tmp_path / "src"
    shutil.copytree(src, bad)
    rows = [json.loads(x) for x in (bad / "repro.jsonl").read_text().splitlines()]
    for r in rows[:10]:                    # 10 of 64 rows differ in bytes, flags kept -> 54/64 = 0.84 < 0.9
        r["gen_ids"] = r["gen_ids"] + [0]
    (bad / "repro.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    assert exp39_dev.main(["--run", "--backend", "stub", "--source", str(bad), "--out", str(out)]) == exp39_dev.EXIT_BYTES
    assert "engine is not the one the Steer-all rows came from" in capsys.readouterr().out
    assert not (out / "qwen3b" / "arms.jsonl").exists()


def test_report_writes_md_and_json(stub39, tmp_path):
    from scripts import exp39_report
    _, run_dir = stub39
    out = tmp_path / "r.md"
    assert exp39_report.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    text = out.read_text()
    numbers = json.loads(out.with_suffix(".json").read_text())
    assert "DEVELOPMENT ONLY" in text and "PROVISIONAL" in text and "Dose curve" in text
    from scripts import exp39_dev
    assert "NOT the pod's" in text and exp39_dev.RANDOM_VECTOR_SHA256 in text
    r = numbers["reading"]
    assert r["n"] == numbers["n_R"] and r["verdict"] in ("M1 supported", "M2 supported", "split") or \
        r["verdict"].startswith("undetermined")
    assert [e["steps"] for e in numbers["dose"]] == ["steps 0", "steps 0-1", "steps 0-2", "steps 0-4", "all steps"]
    assert numbers["n_pairs"] + numbers["n_unseen"] == numbers["n_R"]
    assert set(numbers["damage"]) == {"reproduced", "all"} and len(numbers["damage"]["all"]["qids"]) == 8
    assert numbers["identity"]["n"] == numbers["n_R"]
    assert numbers["repeats"]["A_none_vs_repro"]["k"] == numbers["repeats"]["A_none_vs_repro"]["n"] == numbers["n_R"]


def test_report_refuses_mixed_engines_and_other_folds(stub39, tmp_path, capsys):
    from scripts import exp39_report
    _, run_dir = stub39
    for field, value in (("engine", "hf-other"), ("fold", 2)):
        bad = tmp_path / field
        shutil.copytree(run_dir, bad)
        rows = [json.loads(x) for x in (bad / "arms.jsonl").read_text().splitlines()]
        rows[0][field] = value
        (bad / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert exp39_report.main(["--run-dir", str(bad)]) == 2
        assert "REFUSED" in capsys.readouterr().out
