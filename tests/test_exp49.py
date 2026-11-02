"""§49 legible abstentions (rev 0b): the case list (counts, label balance, split integrity, UNA without a queried city,
nuisance fact and line count, R4 prompt), outcomes, the P_ans prefix, selections and their tie rules, centroid
fallbacks, the clustered bootstrap, the design guard, the R2 vector, zero-vector replays on the stub model, and the
whole chain (dev probes -> design -> held arms -> report) on a fake engine with canned outputs."""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp45_cases as C  # noqa: E402
import exp49_cases as G  # noqa: E402
import exp49_run as R  # noqa: E402
import exp49_report as REP  # noqa: E402
from memctl.analysis import abst49 as A  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return G.build_cases()


# ---------------------------------------------------------------------------------------------- cases

def test_cases_pass_all_checks(rows):
    pr = G.check_cases(rows)
    assert not any(pr.values()), {k: v[:3] for k, v in pr.items() if v}
    assert len(rows) == 1584


def test_label_balance(rows):
    for split in ("dev", "held"):
        for task in G.TASKS:
            for kind, fmt, hards in (("ANS", "F2", ("H0", "H1", "H2")), ("CF", "F2", ("H1",)), ("ANS", "F1", ("H1",))):
                for h in hards:
                    g = [r["gold"] for r in rows if (r["split"], r["task"], r["set"], r["format"], r["hardness"])
                         == (split, task, kind, fmt, h)]
                    counts = {c: g.count(c) for c in C.CITIES}
                    assert len(set(counts.values())) == 1, (split, task, kind, fmt, h, counts)
    assert all(r["gold"] is None for r in rows if r["set"] == "UNA")
    assert all(r["cf_city"] != r["base_gold"] for r in rows)


def test_split_integrity(rows):
    split = {}
    for r in rows:
        assert split.setdefault(r["base_key"], r["split"]) == r["split"]
        assert r["split"] == ("dev" if r["base_index"] % 2 == 0 else "held")
    prompts = {}
    for r in rows:
        prompts.setdefault(r["prompt"], set()).add(r["split"])
    assert all(len(s) == 1 for s in prompts.values())


def test_una_has_no_queried_relation_city(rows):
    for r in (x for x in rows if x["set"] == "UNA"):
        q = r["B"] if r["task"] == "RC" else r["A"]
        for f in r["facts"]:
            if re.search(r"\b" + re.escape(q) + r"\b", f):
                assert not G.cities_in(f), (r["case_id"], f)
        for c in (r["base_gold"], r["cf_city"], r["old_city"]):
            if c:
                assert c not in r["cities_in_prompt"], r["case_id"]
        assert "lives in" not in " ".join(f for f in r["facts"] if r["A"] in f and r["task"] != "RC")


def test_nuisance_fact_and_line_count(rows):
    by = {}
    for r in rows:
        if r["format"] == "F2" and r["hardness"] == "H1":
            by.setdefault(r["base_key"], {})[r["set"]] = r
    for key, d in by.items():
        assert len({len(x["facts"]) for x in d.values()}) == 1, key
        nuis = G.nuisance(d["ANS"]["task"], {"A": d["ANS"]["A"], "B": d["ANS"]["B"], "J": d["ANS"]["nuisance_job"],
                                             "pet": d["ANS"]["nuisance_pet"]})
        assert all(nuis in x["facts"] for x in d.values()), key
        diff = [i for i, (a, b) in enumerate(zip(d["ANS"]["facts"], d["UNA"]["facts"])) if a != b]
        assert len(diff) == (2 if d["ANS"]["task"] == "TU" else 1)
        assert all(i in d["ANS"]["core_fact_indices"] for i in diff)


def test_r4_prompt(rows):
    r = next(x for x in rows if x["format"] == "F2")
    assert G.R4_SENTENCE in r["prompt"] and G.R4_SENTENCE not in r["prompt_r4"]
    assert r["prompt_r4"] == r["prompt"].replace(G.R4_SENTENCE, "")
    assert all(x["prompt_r4"] is None for x in rows if x["format"] == "F1")


# ---------------------------------------------------------------------------------------------- outcomes, prefix

def test_classify_and_utility():
    cities = C.CITIES
    assert A.classify(" York", "York", [], cities)["outcome"] == "gold"
    assert A.classify(" unknown", "York", [], cities)["outcome"] == "abstain"
    assert A.classify(" Leeds", "York", ["Leeds"], cities)["outcome"] == "competitor"
    assert A.classify(" Derby", "York", [], cities)["outcome"] == "wrong"
    assert A.classify(" Derby", None, [], cities)["outcome"] == "wrong"           # UNA: any city is unsupported
    assert A.classify(" the moon", "York", [], cities)["outcome"] == "degenerate"
    assert A.classify(None, "York", [], cities)["outcome"] == "degenerate"
    assert A.answer_text("note York\nAnswer: Leeds\nmore", "F2") == " Leeds"
    assert A.answer_text("no marker", "F2") is None
    assert [A.utility(o, 2) for o in ("gold", "abstain", "wrong", "degenerate")] == [1, 0, -2, -2]


def _bytes_decode(ids):
    return bytes(ids).decode()


def test_ans_prefix_holds_only_model_tokens():
    text = "Ben lives in Leeds.\nAnswer: Leeds\n"
    gen = list(text.encode())
    px = A.ans_prefix(gen, _bytes_decode)
    assert px["ok"] and _bytes_decode(gen[:px["j"]]).endswith("Answer:")
    assert gen[:px["j"]] == list("Ben lives in Leeds.\nAnswer:".encode())       # a slice of the model's own ids
    assert not A.ans_prefix(list(b"no marker here"), _bytes_decode)["ok"]
    flags = A.note_flags("Ben lives in Leeds.", "Leeds", C.CITIES)
    assert flags["note_gold"] and not flags["note_substitute"]
    assert A.note_flags("Ben lives in York.", "Leeds", C.CITIES)["note_substitute"]


def test_prefix_ids_is_prompt_plus_own_generation():
    e = R.Engine(st.StubEngine())
    r = {"case_id": "x", "prompt": "Where does Ben live?"}
    ids = e.chat_ids(r["prompt"])
    gen = list(b"note\nAnswer: York\n")
    b = {"n_prompt": len(ids), "gen_ids": gen, "j": len(b"note\nAnswer:")}
    p = R.prefix_ids(e, r, b)
    assert p[:len(ids)] == ids and p[len(ids):] == gen[:b["j"]]
    assert "York" not in _bytes_decode(p[len(ids):])
    with pytest.raises(RuntimeError):
        R.prefix_ids(e, r, {**b, "n_prompt": len(ids) + 1})


def test_zero_vector_replays_reproduce_baseline_on_stub():
    e = R.Engine(st.StubEngine())
    ids = e.chat_ids("Where does Ben live now?")
    base = e.generate(ids, [], 24, lambda t: False)
    zero = [st.add_spec(5, np.zeros(e.hidden, np.float32), steps=(0,))]
    assert e.generate(ids, zero, 24, lambda t: False) == base               # same prefix: bitwise identical
    for j in (1, 5, 10):
        if j < len(base):
            assert e.generate(ids + base[:j], zero, 24 - j, lambda t: False) == base[j:]   # frozen-prefix replay
    big = [st.add_spec(5, np.full(e.hidden, 50.0, np.float32), steps=(0,))]
    assert e.generate(ids, big, 24, lambda t: False) != base


# ---------------------------------------------------------------------------------------------- selections

def test_dev_parts():
    assert [A.dev_part(i) for i in (0, 2, 4, 6, 1, 3)] == ["fit", "select", "tune", "fit", "fit", "select"]


def test_choose_tau_and_alpha_ties():
    t = A.choose_tau([0.2, 0.9], ["abstain", "abstain"], ["wrong", "gold"])
    assert t["tau"] == 0.9 and t["U"] == 0.5
    t = A.choose_tau([0.5, 0.5], ["abstain"] * 2, ["wrong", "gold"])       # U(0.5) = 0 = U(never): higher τ wins
    assert t["tau"] == float("inf")
    a = A.choose_alpha({0.5: ["gold"], 1.0: ["gold"], 2.0: ["abstain"]}, {0.5: ["abstain"], 1.0: ["abstain"],
                                                                          2.0: ["abstain"]})
    assert a["alpha"] == 0.5
    a = A.choose_alpha({0.5: ["abstain"], 1.0: ["gold"], 2.0: ["gold"]}, {0.5: ["abstain"], 1.0: ["wrong"],
                                                                         2.0: ["abstain"]})
    assert a["alpha"] == 2.0


def test_choose_layer_ties_to_lower():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    X = rng.standard_normal((40, 3, 4))
    X[:, 1, 0] += 5 * y
    X[:, 2, 0] += 5 * y
    fit = np.arange(40) < 20
    out = A.choose_layer(X, y, fit, ~fit, metric="auroc")
    assert out["layer"] == 1


def test_centroid_fallbacks():
    X = np.arange(12, dtype=float).reshape(12, 1)
    labels = ["a"] * 6 + ["b"] * 5 + ["c"]
    fmts = ["F2"] * 6 + ["F1"] * 3 + ["F2"] * 2 + ["F2"]
    c = A.centroids(X, labels, fmts, ["a", "b", "c", "d"])
    assert c["source"] == {"a": "F2", "b": "pooled", "c": None, "d": None}
    assert np.isclose(c["mu"]["b"][0], np.mean(range(6, 11)))


def test_r1_unestimable_is_no_release():
    P = {"mu_abst": np.zeros(4), "mu_ans": {"York": np.ones(4)}}
    out = R.r1_trial(None, {"case_id": "x"}, {}, np.ones(4, np.float32), P, "Leeds", 1, 1.0)
    assert out["status"] == "R1 unestimable" and out["outcome"] == "abstain" and not out["released"]
    out = R.r1_trial(None, {"case_id": "x"}, {}, np.ones(4, np.float32), {**P, "mu_abst": None}, "York", 1, 1.0)
    assert out["status"] == "R1 not estimable"
    v = A.r1_vector(np.full(4, 2.0), P["mu_ans"], P["mu_abst"], "York", 0.5)
    assert np.isclose(np.linalg.norm(v), 0.5 * 4.0)


def test_cluster_boot():
    keys = ["a", "a", "b", "b"]
    out = A.cluster_boot(keys, [1, 1, 0, 0], n=2000)
    assert out["est"] == 0.5 and out["n_bases"] == 2 and out["lo"] == 0.0 and out["hi"] == 1.0
    same = A.cluster_boot(keys, [1, 0, 1, 0], [1, 0, 1, 0], n=500)
    assert same["est"] == 0 and same["lo"] == 0 and same["hi"] == 0


def test_readings():
    assert REP.reading({"lo": -0.1, "est": 0.1}, {"est": 0}) == "no recovery detected at this site and strength"
    assert REP.reading({"lo": 0.05, "est": 0.2}, {"est": 0.1, "lo": 0, "hi": 0.2}).startswith("direction-specific")
    assert REP.q6_reading({"lo": 0.0}) == "contrast not interpretable: calibration did not reproduce"
    assert REP.q6_reading({"lo": 0.01}).startswith("interpretable")


def test_design_guard(tmp_path):
    with pytest.raises(PermissionError):
        A.require_design(tmp_path)
    (tmp_path / "design.json").write_text(json.dumps({"alpha": 1.0}))
    (tmp_path / A.DESIGN_MARKER).write_text(json.dumps({"commit": "abc", "design_sha256": "wrong"}))
    with pytest.raises(PermissionError):
        A.require_design(tmp_path)
    R.mark_design(tmp_path, "abc")
    assert A.require_design(tmp_path) == {"alpha": 1.0}
    assert R.main(["--stage", "arms-held", "--out", str(tmp_path / "none")]) == R.EXIT_DESIGN


def test_r2_vector_is_the_s40_file():
    d = A.r2_vector()
    assert np.array_equal(d["steer"], (2.0 * d["v"].astype(np.float64)).astype(np.float32))
    assert d["v"].shape == (2048,)


def test_shuffled_labels_keep_one_label_per_base():
    y = ["a", "a", "b", "c", "c"]
    k = ["1", "1", "2", "3", "3"]
    s = A.shuffled_labels(y, k)
    assert s[0] == s[1] and s[3] == s[4] and sorted(set(s)) == ["a", "b", "c"]


# ---------------------------------------------------------------------------------------------- whole chain (fake)

class FakeEngine(R.Engine):
    """Canned baselines per prompt (gold, unknown or a wrong city by a hash), residuals = noise + a gold-city signal;
    a non-zero added vector releases to the decoded target if it is a pool city in the vector's sha, else 'Derby'."""

    L, D = 4, 24

    def __init__(self, rows):
        self.n_layers, self.hidden = self.L, self.D
        self.gold = {r["prompt"]: r["gold"] for r in rows}
        self.gold.update({r["prompt_r4"]: r["gold"] for r in rows if r["prompt_r4"]})
        self.cities = list(C.CITIES)
        self.basis = np.random.default_rng(1).standard_normal((12, self.D)) * 3

    def chat_ids(self, prompt):
        return list(prompt.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def _prompt_of(self, text):
        return max((p for p in self.gold if text.startswith(p)), key=len)

    def _baseline_text(self, prompt):
        g = self.gold[prompt]
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 3
        item = "unknown" if (g is None or h == 0) else (g if h == 1 else "Derby" if g != "Derby" else "York")
        return f"Note on it.\nAnswer: {item}\n" if "Note:" in prompt else f" {item}."

    def generate(self, ids, specs, max_new, stop):
        text = self.decode(ids)
        p = self._prompt_of(text)
        base = self._baseline_text(p)
        rest = base[len(text) - len(p):]
        nonzero = [s for s in specs if s.vector is not None and np.any(s.vector)]
        if not nonzero:
            out = rest if len(text) == len(p) else rest.split("\n")[0] + "\n"
            return list(out.encode())
        g = self.gold[p]
        out = f" {g}\n" if (g and int(st.vector_sha(nonzero[0].vector), 16) % 2 == 0) else " Derby\n"
        return list(out.encode())

    def capture(self, ids, positions):
        p = self._prompt_of(self.decode(ids))
        g = self.gold[p]
        rng = np.random.default_rng(int(hashlib.sha256(p.encode()).hexdigest()[:8], 16))
        sig = self.basis[self.cities.index(g)] if g else np.zeros(self.D)
        return np.stack([np.stack([rng.standard_normal(self.D) + sig * (l + 1) / self.L for l in range(self.L)])
                         for _ in positions]).astype(np.float32)


def test_whole_chain_on_fake_engine(rows, tmp_path):
    e = FakeEngine(rows)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = tmp_path / "run"
    (root / "resid").mkdir(parents=True)
    assert R.run_baselines(e, [r for r in rows if r["split"] == "dev"], root, None) == 0
    assert not (root / "baselines" / f"{next(r for r in rows if r['split'] == 'held')['case_id']}.json").exists()
    meta = R.probe_dev(rows, root)
    assert meta["r1_estimable"] and meta["n_tune_pool"] > 0
    assert R.run_arms_dev(e, rows, root, None) == 0
    d = R.make_design(rows, root, cases)
    assert d["alpha"] in A.ALPHAS and d["dev_noop_reproduces"] == "20/20"
    with pytest.raises(PermissionError):
        R.run_arms_held(e, rows, root, None)
    R.mark_design(root, "test")
    assert R.run_baselines(e, [r for r in rows if r["split"] == "held"], root, None) == 0
    assert R.run_arms_held(e, rows, root, None) == 0
    rep = REP.held_report(rows, root)
    assert rep["noop_held"] == "20/20"
    assert rep["n_false_abstentions"] > 0 and rep["n_genuine_abstentions"] > 0
    assert set(rep["Q-P"]) == {"gold_R1_minus_R3_false", "unsupported_R1_minus_R3_genuine", "reading"}
    assert "R2 (one-position)" in rep["Q-3"]
    assert rep["Q-4"]["n_false_h1"] <= rep["n_false_abstentions"]
    REP.markdown(rep)
