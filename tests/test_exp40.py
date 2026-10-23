"""§40: the LoCoMo view builder, the batched note path, the smoke and agreement aborts, the direction and setting
asserts, the arms end to end on the stub, the report arithmetic, the §41 rule path and the bundle contents
(memctl/locomo_view.py, memctl/analysis/exp40.py, scripts/exp40_pod_run.py, scripts/exp40_report.py,
scripts/exp40_bundle.sh). No Qwen weights; the stub engine and judge."""
import json
import random
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from memctl import locomo_view as lv
from memctl.analysis import exp40 as ex
from memctl.analysis import steering as st

LOCOMO = Path(lv.LOCOMO_PATH)
NPZ = Path("runs/exp38_dev/qwen3b/directions.npz")
AGREE = Path("runs/exp40_inputs/agreement60.jsonl")
needs_inputs = pytest.mark.skipif(not (LOCOMO.exists() and NPZ.exists() and AGREE.exists()),
                                  reason="LoCoMo, the §38 directions or the agreement file not on disk")
N_ROWS, N_ADV = 60, 8


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- the view builder on a synthetic conversation -----------------------------------------------------------------

def _sample():
    turns1 = [{"speaker": "Ann", "dia_id": f"D1:{i}", "text": f"turn {i} about apples and pears number {i}"}
              for i in range(1, 8)]
    turns1[2]["blip_caption"] = "a photo of a red bicycle"
    turns1[2]["img_url"] = ["x"]
    turns2 = [{"speaker": "Bob", "dia_id": f"D2:{i}", "text": f"Bob talks about bicycles and trains {i}"}
              for i in range(1, 8)]
    qa = [{"question": "What bicycle did Ann show?", "answer": "red bicycle", "evidence": ["D1:3"], "category": 4},
          {"question": "Which trains?", "answer": "x", "evidence": ["D2:2; D2:3"], "category": 1},
          {"question": "unresolved one", "answer": "x", "evidence": ["D9:1"], "category": 2},
          {"question": "no evidence", "answer": "x", "evidence": [], "category": 3},
          {"question": "garbled evidence", "answer": "x", "evidence": ["not an id"], "category": 4},
          {"question": "too many apples", "answer": "x", "evidence": [f"D1:{i}" for i in range(1, 8)] + ["D2:1", "D2:2"],
           "category": 1},
          {"question": "When did Bob talk about trains?", "answer": "8 May", "evidence": ["D2:5"], "category": 2},
          {"question": "adversarial bicycle?", "adversarial_answer": "none", "evidence": ["D1:3"], "category": 5}]
    return {"sample_id": "conv-x", "qa": qa, "conversation": {
        "speaker_a": "Ann", "speaker_b": "Bob", "session_1_date_time": "1:56 pm on 8 May, 2023", "session_1": turns1,
        "session_2_date_time": "3:00 pm on 9 May, 2023", "session_2": turns2}}


@pytest.fixture()
def synth(tmp_path):
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps([_sample()]))
    return p


def test_exclusions_and_evidence_resolution(synth):
    s = lv.build_set(str(synth), n_rows=3, n_adv=1)
    assert s.counts["excluded"] == {"none:cat3": 1, "too_many:cat1": 1, "unresolved:cat2": 1, "unresolved:cat4": 1}
    assert s.counts["eligible_answerable"] == 3 and s.counts["eligible_adversarial"] == 1
    by = {r.question: r for r in s.rows}
    assert by["Which trains?"].evidence_ids == ("D2:2", "D2:3")       # 'D2:2; D2:3' split by the id pattern
    assert all(1 <= len(r.evidence_ids) <= 8 for r in s.rows + s.adversarial)
    with pytest.raises(ValueError):
        lv.build_set(str(synth), n_rows=4, n_adv=1)


def test_view_has_8_items_evidence_arrival_order_and_bm25_determinism(synth):
    a = lv.build_set(str(synth), n_rows=3, n_adv=1)
    b = lv.build_set(str(synth), n_rows=3, n_adv=1)
    assert a.shas == b.shas and [r.to_dict() for r in a.rows] == [r.to_dict() for r in b.rows]
    order = [f"D1:{i}" for i in range(1, 8)] + [f"D2:{i}" for i in range(1, 8)]
    for r in a.rows + a.adversarial:
        assert len(r.view_ids) == lv.VIEW_K == len(r.memory_lines)
        assert set(r.evidence_ids) <= set(r.view_ids) and set(r.evidence_lines) <= set(r.memory_lines)
        assert list(r.view_ids) == sorted(r.view_ids, key=order.index)          # arrival order
        assert r.n_bm25_fill + r.n_zero_fill + len(r.evidence_ids) == lv.VIEW_K
    q = next(r for r in a.rows if r.question == "Which trains?")
    # BM25 (question words 'trains') prefers Bob's turns over Ann's; ties go to the earlier turn
    assert sum(i.startswith("D2:") for i in q.view_ids) == 7


def test_caption_and_line_format(synth):
    s = lv.build_set(str(synth), n_rows=3, n_adv=1)
    r = next(x for x in s.rows if x.question == "What bicycle did Ann show?")
    assert r.evidence_lines == ("Ann, 1:56 pm on 8 May, 2023: turn 3 about apples and pears number 3 "
                                "[shares a photo: a photo of a red bicycle]",)
    assert r.judge_type == "single-session-user"
    assert {x.category: x.judge_type for x in s.rows} == {4: "single-session-user", 1: "multi-session",
                                                          2: "temporal-reasoning"}
    assert lv.JUDGE_TYPE[3] == "default"
    assert r.prompt.endswith("## Current input\nQuestion: What bicycle did Ann show?\n\nNote:")


def test_seeded_shuffle_is_random_40_over_file_order(synth):
    s = lv.build_set(str(synth), n_rows=3, n_adv=1)
    file_order = ["conv-x_q0", "conv-x_q1", "conv-x_q6"]
    expect = list(file_order)
    random.Random(40).shuffle(expect)
    assert [r.question_id for r in s.rows] == expect


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo not on disk")
def test_real_set_counts_and_shas():
    s = lv.build_set()
    assert s.shas["locomo10.json"] == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    assert s.shas["rows"] == "8da240521cbd03320b5c63b839c481f3937902dfac5f0f3d96aa140ac4a836d0"
    assert s.shas["adversarial"] == "ea3d5581a00d7374d2c49f3a778f5f3a520f6e74367935a913c62f7cb0e6829e"
    assert s.counts["eligible_answerable"] == 1523 and len(s.rows) == 600 and len(s.adversarial) == 40
    assert all(r.category in (1, 2, 3, 4) for r in s.rows) and all(r.category == 5 for r in s.adversarial)
    assert all(len(r.memory_lines) == 8 and set(r.evidence_lines) <= set(r.memory_lines) for r in s.rows)
    from memctl.analysis.jacobian import ablate_evidence
    for r in s.rows[:50]:
        cut = ablate_evidence(r.prompt, r.memory_lines, r.evidence_lines)
        assert cut != r.prompt and not any(line in cut for line in r.evidence_lines)


def _cached_prompt():
    for p in sorted(Path("cache/generations_exp23_qwen3b").glob("*/*.json"))[:200]:
        d = json.loads(p.read_text())
        if d.get("max_new_tokens") == 256 and "## Memory\n" in d["prompt"] and d["prompt"].endswith("\n\nNote:"):
            return d["prompt"]
    return None


def _skeleton(prompt: str) -> str:
    head, rest = prompt.split("## Memory\n", 1)
    _, tail = rest.split("\n\n## Current input\n", 1)
    assert tail.startswith("Question: ") and tail.endswith("\n\nNote:")
    return head + "## Memory\n<MEMORY>\n\n## Current input\nQuestion: <QUESTION>\n\nNote:"


@pytest.mark.skipif(not Path("cache/generations_exp23_qwen3b").exists(), reason="§23 cache not on disk")
def test_template_skeleton_matches_a_cached_23_prompt(synth):
    cached = _cached_prompt()
    assert cached is not None
    s = lv.build_set(str(synth), n_rows=3, n_adv=1)
    for r in s.rows:
        assert _skeleton(r.prompt) == _skeleton(cached)              # byte-for-byte except memory and question
        mem = r.prompt.split("## Memory\n", 1)[1].split("\n\n## Current input\n", 1)[0]
        assert mem == "\n".join(r.memory_lines)


# --- arithmetic of the pure module ---------------------------------------------------------------------------------

def test_setting_refusal():
    ex.check_setting(29, 2.0)
    for layer, alpha in ((25, 2.0), (29, 4.0), (29, 1.0), (29.5, 2.0), (32, 2.0)):
        with pytest.raises(ValueError):
            ex.check_setting(layer, alpha)
    assert ex.setting_of("steer_all").key == "steer:L29:a2:vnorm"
    assert ex.setting_of("steer0").key == "steer:L29:a2:vnorm:s0"
    assert ex.setting_of("random").key == "random:L29:a2:vnorm"


@pytest.mark.skipif(not NPZ.exists(), reason="§38 directions not on disk")
def test_direction_asserts():
    from scripts import exp40_pod_run as pr
    d = pr.load_direction(str(NPZ))
    assert d["v_sha256"].startswith("a7f967ec") and st.vector_sha(d["steer"]) == ex.STEER_VECTOR_SHA256
    assert abs(d["steer_norm"] - 64.47) < 0.01 and abs(d["random_norm"] - d["steer_norm"]) < 1e-3
    assert st.vector_sha(d["random"]) == st.vector_sha(st.random_vector(d["steer"], 29, seed=40))
    assert st.vector_sha(d["random"]) != st.vector_sha(st.random_vector(d["steer"], 29, seed=38))
    casts = pr.bf16_report(d)
    assert casts["steer"]["cosine_bf16_fp32"] > 0.9999
    bad = d["v"].copy()
    bad[0] += 1e-3
    with pytest.raises(ValueError):
        ex.check_direction(bad)
    with pytest.raises(ValueError):
        ex.check_direction(d["v"], npz_sha="0" * 64)


def test_selection_and_counts():
    order = [f"q{i}" for i in range(10)]
    a = {f"q{i}": {"refusal": i % 2 == 0, "cls": "refusal" if i % 2 == 0 else "answer"} for i in range(9)}  # q9 unmarked
    v = {f"q{i}": i in (1, 5, 7) for i in range(9)}
    s = ex.select(order, a, v, r_cap=3, c_cap=2)
    assert s == {"R": ["q0", "q2", "q4"], "C": ["q1", "q5"]}
    c = ex.pass_counts(order, set(a), a, v)
    assert c["marked"] == 9 and c["marker_rate"] == 0.9 and c["refusals"] == 5 and c["judged_correct"] == 3


def test_smoke_check_and_agreement_arithmetic():
    b = [{"first_id": i, "marker": True, "output": "x"} for i in range(5)]
    assert ex.smoke_notes_check(b, b)["pass"]
    s = [dict(r, first_id=99) if i < 2 else r for i, r in enumerate(b)]
    assert not ex.smoke_notes_check(b, s)["pass"]                          # 3/5 same first token
    m = [dict(r, marker=i > 1) for i, r in enumerate(b)]
    assert not ex.smoke_notes_check(m, b)["pass"]                          # 3/5 markers
    one = [dict(r, first_id=99) if i == 0 else r for i, r in enumerate(b)]
    assert ex.smoke_notes_check(b, one)["pass"]                            # 4/5 passes
    assert ex.agreement(["yes"] * 54 + ["no"] * 6, [True] * 60)["rate"] == 0.9
    assert ex.agreement(["yes"] * 54 + ["no"] * 6, [True] * 60)["pass"]
    r = ex.agreement(["yes"] * 53 + ["maybe"] + ["no"] * 6, [True] * 54 + [False] * 6)
    assert r["agree"] == 59 and r["unparseable"] == 1


# --- batched vs single notes ---------------------------------------------------------------------------------------

def test_batched_equals_single_on_the_stub():
    from scripts import exp40_pod_run as pr
    eng = pr.build_engine("stub", 1)
    prompts = ["Question: a?\n\nNote:", "Question: bb?\n\nNote:", "x" * 50]
    eng.gold_of = {st.sha256(p): "red bicycle" for p in prompts}
    assert pr.notes_batched(eng, prompts) == pr.notes_single(eng, prompts)


def _tiny_hf():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("no cached Qwen2.5 tokenizer")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    net = transformers.Qwen2ForCausalLM(cfg).eval()
    net.generation_config.repetition_penalty = 1.05          # as the 3B's generation_config.json; must be overridden
    net.generation_config.do_sample = True
    eng = object.__new__(st.HFEngine)
    eng.torch, eng.model, eng.device, eng.name = torch, "tiny-qwen2", "cpu", "hf-tiny"
    eng.tokenizer, eng.net, eng.layers = tok, net, net.model.layers
    eng.n_layers, eng.hidden = 2, 32
    eng.stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return eng


def test_batched_left_padded_generate_equals_the_single_sequence_loop_on_a_tiny_hf_model():
    from scripts import exp40_pod_run as pr
    eng = _tiny_hf()
    prompts = ["Hi.", "A longer prompt with several more words in it, to force left padding.", "Question: x?\n\nNote:"]
    b = pr.notes_batched(eng, prompts, max_new=12, batch=3)
    s = pr.notes_single(eng, prompts, max_new=12)
    assert [x["gen_ids"] for x in b] == [x["gen_ids"] for x in s]


# --- the runner end to end on the stub -----------------------------------------------------------------------------

def _run(out: Path, *extra) -> int:
    from scripts import exp40_pod_run as pr
    return pr.main(["run", "--out", str(out), "--backend", "stub", "--n-rows", str(N_ROWS), "--n-adv", str(N_ADV),
                    *extra])


@pytest.fixture(scope="module")
def stub40(tmp_path_factory):
    if not (LOCOMO.exists() and NPZ.exists() and AGREE.exists()):
        pytest.skip("inputs not on disk")
    out = tmp_path_factory.mktemp("exp40")
    assert _run(out, "--rule", str(out / "no_rule.json")) == 0
    return out / "qwen3b"


@needs_inputs
def test_stub_run_writes_every_stage(stub40):
    sel_ = json.loads((stub40 / "selection.json").read_text())
    R, C = sel_["R"], sel_["C"]
    assert 0 < len(R) <= ex.R_CAP and len(C) <= ex.C_CAP and not set(R) & set(C)
    assert sel_["counts"]["n"] == N_ROWS
    apass = _rows(stub40 / "apass.jsonl")
    notes = {r["question_id"]: r for r in _rows(stub40 / "notes.jsonl")}
    assert len(notes) == N_ROWS + N_ADV
    assert {r["question_id"] for r in apass} == {q for q, r in notes.items() if r["marker"]}
    for r in apass:
        assert r["last_token"] == ":" and "step0" in r and "s_proj" in r and r["max_new_tokens"] == ex.MAX_NEW
        assert (stub40 / "resid" / "A" / f"{r['question_id']}.npy").exists()
    arms = _rows(stub40 / "arms.jsonl")
    keys = {(r["pop"], r["kind"], r["setting"]["key"]) for r in arms}
    L = json.loads((stub40 / "manifest.json").read_text())["layer"]
    for pop, kind, names in [("R", "A", ("steer_all", "random", "steer0")), ("R", "cut", ("none", "steer_all")),
                             ("C", "A", ("steer_all",)), ("R", "B", ("none",)), ("adv", "B", ("none",)),
                             ("adv", "A", ("steer_all",)), ("adv", "cut", ("none",))]:
        for n in names:
            if (pop == "C" and not C):
                continue
            assert (pop, kind, ex.setting_of(n, L).key) in keys, (pop, kind, n)
    m = json.loads((stub40 / "manifest.json").read_text())
    rnd = {r["vector_sha256"] for r in arms if r["setting"]["mode"] == "random"}
    assert len(rnd) == 1                                                   # one random vector for every row
    assert m["random_seed"] == [40, 29] and m["vectors"]["steer_sha256"] == ex.STEER_VECTOR_SHA256
    # the selection uses only the A outputs and the judge, recomputed
    from scripts import exp40_pod_run as pr
    a_rows = {r["question_id"]: r for r in apass if r["pop"] == "all"}
    v = {(r["pop"], r["kind"], r["setting_key"], r["question_id"]): st.parse_verdict(r["reply"])
         for r in _rows(stub40 / "judge.jsonl")}
    view = json.loads((stub40 / "view.json").read_text())
    assert ex.select(view["rows"], a_rows, {q: v.get(("all", "A", "none", q)) for q in a_rows}) == {"R": R, "C": C}
    assert not any(k[0] == "adv" for k in v)                               # the adversarial block is never judged
    assert pr.READER == "qwen3b"


@needs_inputs
def test_stub_run_resumes_without_new_rows(stub40):
    before = {p.name: len(p.read_text().splitlines()) for p in stub40.glob("*.jsonl")}
    assert _run(stub40.parent, "--rule", str(stub40.parent / "no_rule.json")) == 0
    assert {p.name: len(p.read_text().splitlines()) for p in stub40.glob("*.jsonl")} == before


@needs_inputs
def test_other_settings_are_refused_before_anything_runs(tmp_path):
    assert _run(tmp_path, "--layer", "25") == ex.EXIT_SETTING
    assert _run(tmp_path, "--alpha", "4") == ex.EXIT_SETTING
    from scripts import exp40_pod_run as pr
    assert pr.main(["run", "--out", str(tmp_path), "--backend", "hf", "--n-rows", "10"]) == ex.EXIT_SETTING
    assert not (tmp_path / "qwen3b").exists()


@needs_inputs
def test_smoke_aborts_on_a_silent_judge_and_on_batched_single_mismatch(tmp_path, monkeypatch):
    assert _run(tmp_path / "a", "--stub-judge", "silent") == ex.EXIT_SMOKE
    assert not (tmp_path / "a" / "qwen3b" / "notes.jsonl").exists()
    from scripts import exp40_pod_run as pr
    real = pr.notes_single

    def shifted(engine, prompts, max_new=ex.NOTE_MAX_NEW):
        out = real(engine, prompts, max_new)
        return [{"generated": "Z" + o["generated"], "gen_ids": [90] + o["gen_ids"]} for o in out]
    monkeypatch.setattr(pr, "notes_single", shifted)
    assert _run(tmp_path / "b") == ex.EXIT_SMOKE
    smoke = json.loads((tmp_path / "b" / "qwen3b" / "smoke" / "smoke.json").read_text())
    assert smoke["notes_check"]["same_first"] == 0 and smoke["problems"]


@needs_inputs
def test_agreement_below_the_bar_stops_before_any_arm(tmp_path):
    assert _run(tmp_path, "--stub-judge", "disagree") == ex.EXIT_AGREEMENT
    d = tmp_path / "qwen3b"
    ag = json.loads((d / "agreement.json").read_text())
    assert ag["rate"] == 0.0 and not ag["pass"] and ag["n"] == 60
    assert not (d / "notes.jsonl").exists() and not (d / "arms.jsonl").exists()


# --- the report ----------------------------------------------------------------------------------------------------

def _report(run_dir: Path, rule: Path | None = None, samples: int = 300):
    from scripts import exp40_report as rp
    return rp.build(run_dir, True, samples=samples, rule=rule)


@needs_inputs
def test_report_refuses_without_confirm(stub40):
    from scripts import exp40_report as rp
    with pytest.raises(rp.Refused):
        rp.build(stub40, False)
    assert rp.main(["--run-dir", str(stub40), "--out", str(stub40.parent / "r.md")]) == 2


def _forge(run_dir: Path, R: list[str], C: list[str], steer_yes: set, cut_yes: set, random_ans: set, c_yes: set,
           b_yes: set) -> None:
    """Rewrite the stub rows into a known pattern: Steer-all answers on every R row, gold on `steer_yes`; Steer-all-cut
    answers, gold on `cut_yes`; Random answers on `random_ans`; C Steer-all gold on `c_yes`; B gold on `b_yes`."""
    m = json.loads((run_dir / "manifest.json").read_text())
    L = m["layer"]
    k = {n: ex.setting_of(n, L).key for n in ("steer_all", "random", "none")}
    arms = _rows(run_dir / "arms.jsonl")
    verdict = {}
    for r in arms:
        q, pop, kind, key = r["question_id"], r["pop"], r["kind"], r["setting"]["key"]
        yes = None
        if pop == "R" and kind == "A" and key == k["steer_all"]:
            yes = q in steer_yes
        elif pop == "R" and kind == "cut" and key == k["steer_all"]:
            yes = q in cut_yes
        elif pop == "R" and kind == "A" and key == k["random"]:
            r["cls"] = "answer" if q in random_ans else "refusal"
            yes = False
        elif pop == "C" and kind == "A" and key == k["steer_all"]:
            yes = q in c_yes
        elif pop == "R" and kind == "B":
            yes = q in b_yes
        if yes is not None:
            if key != k["random"]:
                r["cls"], r["phrase"], r["final_answer"] = "answer", f"ans {q}", f"ans {q}"
            verdict[(pop, kind, key, q)] = yes
    _write(run_dir / "arms.jsonl", arms)
    jr = _rows(run_dir / "judge.jsonl")
    for j in jr:
        key = (j["pop"], j["kind"], j["setting_key"], j["question_id"])
        if key in verdict:
            j["reply"] = "yes" if verdict[key] else "no"
    _write(run_dir / "judge.jsonl", jr)


@needs_inputs
def test_report_claim_arithmetic(stub40, tmp_path):
    d = Path(shutil.copytree(stub40, tmp_path / "qwen3b"))
    sel_ = json.loads((d / "selection.json").read_text())
    R, C = sel_["R"], sel_["C"]
    n = len(R)
    steer_yes, cut_yes = set(R[: n // 2]), set(R[: n // 4])
    _forge(d, R, C, steer_yes, cut_yes, random_ans=set(R[:1]), c_yes=set(C[1:]), b_yes=set(R[:3]))
    text, num = _report(d)
    c1 = num["claim1"]
    assert c1["n"] == n and abs(c1["mean"] - (len(steer_yes) - len(cut_yes)) / n) < 1e-12
    assert num["R_table"]["Steer-all"]["gold"]["p"] == len(steer_yes) / n
    assert c1["meaningful"] == (len(steer_yes) / n >= 0.30)
    assert num["descriptive"] == (n < ex.R_MIN_FOR_CLAIMS)
    if n < ex.R_MIN_FOR_CLAIMS:
        assert "descriptive" in c1["verdict"] and "descriptive, no verdict" in text
    assert num["controlX"]["ans_random"] == 1 / n and num["controlX"]["ans_steer"] == 1.0
    assert num["controlX"]["pass"] == (1 / n <= 0.2)
    if C:
        cc = num["controlC"]
        assert abs(cc["mean"] - (-(1 / len(C)))) < 1e-12 and cc["p_gold_A"] == 1.0
    assert abs(num["claim2"]["mean"] - (len(steer_yes) - 3) / n) < 1e-12
    assert sum(b["n"] for b in num["by_type"].values()) == n
    assert num["selection_recomputed_equal"] and not num["partial"]


@needs_inputs
def test_report_marks_partial_rows(stub40, tmp_path):
    d = Path(shutil.copytree(stub40, tmp_path / "qwen3b"))
    plan = json.loads((d / "plan.json").read_text())
    plan["cut"] = [{"arm": "R:B:none", "done": 3, "of": 10}]
    (d / "plan.json").write_text(json.dumps(plan))
    arms = [r for r in _rows(d / "arms.jsonl") if not (r["pop"] == "R" and r["kind"] == "B")]
    _write(d / "arms.jsonl", arms)
    text, num = _report(d)
    assert num["partial"] and "PARTIAL" in text and num["R_table"]["B"]["rows"] == 0


@needs_inputs
def test_rule_path_scores_cond(tmp_path):
    rule = tmp_path / "rule.json"
    out = tmp_path / "run"
    rule.write_text(json.dumps({"signal": "s_margin", "sign": "+", "tau": "-inf", "scored_on_40": True,
                                "refusal_start_ids": [9788]}))
    assert _run(out, "--rule", str(rule)) == 0
    d = out / "qwen3b"
    m = json.loads((d / "manifest.json").read_text())
    assert m["rule"]["present"] and m["rule"]["scored"] and len(m["rule"]["sha256"]) == 64
    _, num = _report(d, rule)
    s = num["s41"]
    assert s["scored"] and s["fires"] == s["n"] > 0                      # tau -inf: the rule fires everywhere
    for lam in (1, 2):
        assert s[f"lambda{lam}"]["U"]["Cond"] == s[f"lambda{lam}"]["U"]["Steer"]
        assert s[f"lambda{lam}"]["cond_minus_steer"]["mean"] == 0
    rule.write_text(json.dumps({"signal": "s_margin", "sign": "+", "tau": "inf", "scored_on_40": True,
                                "refusal_start_ids": [9788]}))
    from scripts import exp40_report as rp
    with pytest.raises(rp.Refused):                                         # the rule changed after the run
        _report(d, rule)
    m["rule"]["sha256"] = rp.ex_file_sha(rule)
    (d / "manifest.json").write_text(json.dumps(m))
    _, num = _report(d, rule)
    s = num["s41"]
    assert s["fires"] == 0 and s["lambda1"]["U"]["Cond"] == s["lambda1"]["U"]["A"]
    rule.write_text(json.dumps({"signal": "s_evid", "sign": "-", "tau": 0.0, "scored_on_40": False,
                                "refusal_start_ids": [9788]}))
    m["rule"]["sha256"] = rp.ex_file_sha(rule)
    m["rule"]["scored"] = False
    (d / "manifest.json").write_text(json.dumps(m))
    _, num = _report(d, rule)
    assert not num["s41"]["scored"] and set(num["s41"]["signals"]) == {"s_margin", "s_proj", "s_evid"}


def test_apply_rule_on_logged_signals():
    from memctl.analysis import conditional as cd
    rule = {"signal": "s_evid", "sign": "-", "tau": 1.5}
    assert cd.apply_rule(rule, {"s_evid": 1.0}) and not cd.apply_rule(rule, {"s_evid": 2.0})
    with pytest.raises(ValueError):
        cd.apply_rule(rule, {"s_evid": None})


# --- the bundle ----------------------------------------------------------------------------------------------------

@needs_inputs
def test_bundle_contents(tmp_path):
    out = tmp_path / "b.tar.gz"
    subprocess.run(["bash", "scripts/exp40_bundle.sh", str(out)], check=True, capture_output=True)
    with tarfile.open(out) as t:
        names = set(t.getnames())
        info = json.loads(t.extractfile("bundle_info.json").read())
    for f in ("data/locomo/locomo10.json", "runs/exp38_dev/qwen3b/directions.npz", "runs/exp40_inputs/agreement60.jsonl",
              "scripts/exp40_pod_run.py", "scripts/exp40_pod.sh", "scripts/exp40_report.py", "memctl/locomo_view.py",
              "memctl/analysis/exp40.py", "pyproject.toml", "bundle_info.json"):
        assert f in names, f
    runs = {n for n in names if n.startswith("runs/")}
    allowed = {"runs/exp38_dev/qwen3b/directions.npz", "runs/exp40_inputs/agreement60.jsonl",
               "runs/exp41_dev/qwen3b/rule.json"}
    assert runs <= allowed
    assert not any(n.startswith(("tests/", "cache/", ".venv")) or n.endswith(".env") or "__pycache__" in n for n in names)
    assert {n for n in names if n.startswith("data/")} == {"data/locomo/locomo10.json"}
    assert info["directions_sha256"] == ex.SOURCE_NPZ_SHA256 and len(info["head_commit"]) == 40
    assert ("runs/exp41_dev/qwen3b/rule.json" in names) == Path("runs/exp41_dev/qwen3b/rule.json").exists()


def test_pod_script_install_line_and_preflight_cover_the_imports():
    sh = Path("scripts/exp40_pod.sh").read_text()
    line = next(x for x in sh.splitlines() if "pip install" in x)
    for pkg in ("-e .", "transformers==", "accelerate", "scipy"):
        assert pkg in line
    assert re.search(r"HARD_STOP_S:-2700", sh) and re.search(r"SELF_STOP_S:-3300", sh) and "JUDGE_RESERVE_S" in sh
    assert "stop_pod_now" in sh and '"$RC" -eq 7' in sh
    src = Path("scripts/exp40_pod_run.py").read_text() + Path("scripts/exp40_report.py").read_text()
    third = {m for m in re.findall(r"^\s*(?:from|import) (\w+)", src, re.M)} - {
        "argparse", "hashlib", "json", "sys", "time", "pathlib", "collections", "__future__", "memctl", "scripts"}
    assert third <= {"numpy", "torch"}, third
