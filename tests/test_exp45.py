"""§45 maps: case list, pinned tokens, the §23 match, readability, the map pass on a tiny random Qwen2, the runner's
smoke / resume / deadline, the report and the render. No real model is loaded (tokenizer + tiny random Qwen2)."""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import maps as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _script(name):
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {error}")


@pytest.fixture(scope="module")
def cases(tok):
    return _script("exp45_cases").build_cases(tok, 45)


# ---------------------------------------------------------------------------------------------- the case list

def test_cases_instantiate_864_without_duplicate_golds(cases, tok):
    C = _script("exp45_cases")
    problems = C.check_cases(cases, tok)
    assert not any(problems.values()), {k: v[:3] for k, v in problems.items() if v}
    assert len(cases) == 864 and len({c["case_id"] for c in cases}) == 864
    assert sum(c["split"] == "dev" for c in cases) == 432
    for task in C.TASKS:
        for variant in C.VARIANTS:
            golds = [c["gold"] for c in cases if (c["task"], c["variant"], c["format"], c["hardness"]) == (task, variant, "F1", "H0")]
            assert len(golds) == 12 and len(set(golds)) == 12


def test_cases_are_deterministic_and_matched_across_variants(cases, tok):
    C = _script("exp45_cases")
    again = C.build_cases(tok, 45)
    assert [c["prompt_sha256"] for c in again] == [c["prompt_sha256"] for c in cases]
    other = C.build_cases(tok, 46)
    assert [c["prompt_sha256"] for c in other] != [c["prompt_sha256"] for c in cases]
    by = {c["case_id"]: c for c in cases}
    # same distractors in every variant of a base case; H1 is a subset of H2
    assert by["RC03_v0_F1_H2"]["distractors"] == by["RC03_v1_F1_H2"]["distractors"]
    assert by["RC03_v0_F1_H1"]["distractors"] == by["RC03_v0_F1_H2"]["distractors"][:3]
    tu = by["TU00_v1_F1_H0"]
    assert tu["gold"] == by["TU00_v0_F1_H0"]["competitors"][0] and tu["competitors"] == [by["TU00_v0_F1_H0"]["gold"]]
    assert by["RC00_v2_F1_H0"]["gold"] == by["RC00_v0_F1_H0"]["intermediate"] and by["RC00_v2_F1_H0"]["intermediate"] is None
    assert by["DE00_v0_F1_H0"]["prompt"].count("\n") == 2 and len(by["DE00_v0_F1_H2"]["facts"]) == 11


def test_f2_is_the_23_memory_note_wrapper(cases):
    c = next(x for x in cases if x["case_id"] == "TU00_v0_F2_H0")
    assert c["prompt"].endswith("\n\nNote:") and "\n\n## Memory\nuser, 2023/03/01 (Wed) 10:00: " in c["prompt"]
    assert "\n\n## Current input\nQuestion: (asked on 2023/03/03 (Fri) 12:00) Where does" in c["prompt"]
    cached = sorted((ROOT / "cache/generations_exp23_qwen3b").rglob("*.json"))[:1]
    if not cached:
        pytest.skip("no §23 cache on this machine")
    real = json.loads(cached[0].read_text())["prompt"]
    head = c["prompt"][: c["prompt"].index("## Memory")]
    assert real.startswith(head)


# ---------------------------------------------------------------------------------------------- pinned tokens

def test_first_content_token_on_digits_and_determiners(tok):
    four, text = M.first_content_token(tok, "4 hours", "sp")
    assert text == "4" and tok.decode(tok(" 4 hours", add_special_tokens=False).input_ids[:1]) == " "
    assert M.first_content_token(tok, "4 hours", "bare")[0] == four            # digits: the two forms coincide
    song, text = M.first_content_token(tok, "The Song of Achilles", "sp")
    assert text == " Song"
    assert M.first_content_token(tok, "the doctor", "sp")[1] == " doctor"
    assert M.first_content_token(tok, "The", "sp")[1] == " The"                # a lone determiner is the content
    assert M.first_content_token(tok, "Leeds", "bare")[1] == "Le" and M.first_content_token(tok, "Leeds", "sp")[1] == " Leeds"
    with pytest.raises(ValueError):
        M.first_content_token(tok, "x", "other")


def test_pin_tokens_merges_roles_and_refuses_a_shared_first_token(tok):
    pins = M.pin_tokens(tok, [("gold", "York"), ("entity", "York"), ("entity", "Alice")])
    york = [p for p in pins if p["surface"] == "York"]
    assert len(york) == 2 and all(set(p["roles"]) == {"gold", "entity"} for p in york)
    assert {p["surface"] for p in pins} == {"York", "Alice", "unknown"} and len(pins) == 6
    with pytest.raises(ValueError):
        M.pin_tokens(tok, [("gold", "Leeds"), ("entity", "Leicester")])         # bare "Le" shared
    dropped = []
    pins = M.pin_tokens(tok, [("gold", "Leeds"), ("substitute", "Leicester")], strict=False, dropped=dropped)
    assert dropped and dropped[0]["surface"] == "Leicester" and dropped[0]["form"] == "bare"
    fixed = M.pin_tokens(tok, [("gold", "York")], fixed={("York", "sp"): 12345})
    assert fixed[0]["id"] == 12345


# ---------------------------------------------------------------------------------------------- outcome

@pytest.mark.parametrize("output, want", [
    ("Leeds.", "gold"), ("Alice lives in Leeds.", "gold"), ("Note: Alice moved.\nAnswer: Leeds", "gold"),
    ("York", "competitor"), ("Answer: York", "competitor"),
    ("unknown", "abstain"), ("The memory does not contain the answer; reply: unknown", "abstain"),
    ("Paris", "other"), ("Alice lived in York and moved to Leeds.", "other"), ("", "other"),
])
def test_classify_four_outcomes(output, want):
    got = M.classify(output, "Leeds", ["York"])
    assert got["outcome"] == want, got
    if output.startswith("Alice lived in York"):
        assert got["ambiguous"]


def test_classify_grades_after_the_last_answer_marker():
    got = M.classify("Note: York, then Leeds.\nAnswer: York", "Leeds", ["York"])
    assert got["outcome"] == "competitor" and got["answer_marker"] and got["answer"] == "York"
    assert not M.classify("Leeds", "Leeds")["answer_marker"]


# ---------------------------------------------------------------------------------------------- readability

def _table():
    # L=4 layers (0..3, 3 final), P=5 positions (index 2 = last prompt token), T=3 pinned columns
    r = np.full((4, 5, 3), 500, dtype=np.int32)
    r[1, 0, 0] = 1            # before the last prompt token: never counts
    r[2, 3, 0] = 7            # column 0 readable at layer 2, position 3 (generated)
    r[3, 2, 1] = 2            # column 1 readable only at the final layer
    r[0, 4, 2] = 11           # column 2 just outside the top-10
    return r, [0, 1, 2, 3]


def test_readability_on_a_synthetic_rank_table():
    r, layers = _table()
    a = M.readability(r, [0], layers, 2, exclude_final=True)
    assert a == {"readable": True, "first_layer": 2, "first_depth": 0.5, "first_pos": 3, "best_rank": 7}
    assert M.readability(r, [1], layers, 2, exclude_final=True)["readable"] is False
    assert M.readability(r, [1], layers, 2)["first_layer"] == 3
    assert M.readability(r, [2], layers, 2)["readable"] is False and M.readability(r, [2], layers, 2, k=11)["readable"]
    assert M.readability(r, [1, 2], layers, 2)["readable"] and M.readability(r, [0], layers, 0)["first_layer"] == 1
    assert M.readability(r, [], layers, 2)["readable"] is None
    assert M.gold_trajectory(r, [0], 2).tolist() == [500, 500, 7, 500]
    assert M.departure_layer(np.array([500, 500, 500, 3]), np.array([500, 9, 2, 1]), layers) == 1


def test_wilson_verdict_spearman_quartiles():
    p, lo, hi = M.wilson(8, 10)
    assert p == 0.8 and 0.44 < lo < 0.50 and 0.94 < hi < 0.97
    assert M.wilson(0, 0) == (None, None, None)
    assert M.criterion_verdict(19, 20) == "recovers" and M.criterion_verdict(7, 10) == "not recovered"
    assert M.criterion_verdict(0, 0) == "no data"
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert M.spearman([1, 1, 1], [1, 2, 3]) is None
    assert M.quartiles([0.0, 1.0, None]) == [0.25, 0.5, 0.75]


# ---------------------------------------------------------------------------------------------- the map pass

@pytest.fixture(scope="module")
def tiny(tok, tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    jlens = pytest.importorskip("jlens")
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    d = tmp_path_factory.mktemp("stub")
    model.save_pretrained(d / "qwen3b")
    tok.save_pretrained(d / "qwen3b")
    lm = jlens.from_hf(model, tok)
    lens = jlens.fit(lm, [" ".join(f"w{i}" for i in range(200))] * 2, source_layers=[0, 1, 2], dim_batch=8,
                     max_seq_len=64, skip_first=4)
    lens.save(str(d / "lens.pt"))
    return model, lm, lens, d


def test_map_prompt_shapes_final_layer_and_lens_apply(tok, tiny, cases):
    from memctl.analysis.jacobian import wrap_chat
    from memctl.analysis.jlens_readout import ranks_of

    model, lm, lens, _ = tiny
    c = next(x for x in cases if x["case_id"] == "RC00_v0_F1_H0")
    x = M.MapInputs(c["case_id"], wrap_chat(tok, c["prompt"]), [p["id"] for p in c["pinned"]], None, 4)
    payload, a = M.map_prompt(model, lm, tok, x, {"j56": lens, "logit": None}, k=5)
    L, P, T = 4, payload["n_tokens"], len(c["pinned"])
    assert a["ranks"].shape == (2, L, P, T) and a["top_ids"].shape == (2, L, P, 5) and a["layers"].tolist() == [0, 1, 2, 3]
    assert payload["n_generated"] <= 4 and payload["n_tokens"] == payload["n_prompt_tokens"] + payload["n_generated"]
    assert payload["last_prompt_index"] == payload["n_prompt_tokens"] - 1 and payload["n_bos"] == 0
    assert payload["position_tokens"][0] == "<|im_start|>" and payload["position_tokens"][payload["last_prompt_index"]] == "\n"
    assert (a["ranks"][0, -1] == a["ranks"][1, -1]).all() and (a["ranks"] >= 1).all()
    assert a["input_ids"][: payload["n_prompt_tokens"]].tolist() == tok(x.chat_text, add_special_tokens=False).input_ids
    x0 = M.MapInputs(c["case_id"], x.chat_text, x.pinned_ids, None, 0)      # no generation: ids round-trip exactly
    payload, a = M.map_prompt(model, lm, tok, x0, {"j56": lens, "logit": None}, k=5)
    text = tok.decode(a["input_ids"].tolist())
    for layer in (0, 2):
        for pos in (0, payload["last_prompt_index"]):
            for li, use_j in ((0, True), (1, False)):
                lg, _, used = lens.apply(lm, text, layers=[layer], positions=[pos], max_seq_len=4096, use_jacobian=use_j)
                assert used[0].tolist() == a["input_ids"].tolist()
                row = lg[layer].numpy()
                # batched vs single-position matmuls round differently: near-tied logits of the random model may
                # swap, so ranks agree to within a few places
                want, got = ranks_of(row, x.pinned_ids)[0], a["ranks"][li, layer, pos]
                assert np.all(np.abs(want - got) <= np.maximum(3, 1e-4 * want)), (want, got)
                assert np.argmax(row[0]) == a["top_ids"][li, layer, pos, 0]


def test_map_prompt_keeps_chosen_positions_and_every_generated_one(tok, tiny):
    from memctl.analysis.jacobian import wrap_chat

    model, lm, lens, _ = tiny
    x = M.MapInputs("k", wrap_chat(tok, "Alice lives in York.\n\nWhere does Alice live?"), [100, 200], [3, 4, 9999], 3)
    payload, a = M.map_prompt(model, lm, tok, x, {"j56": lens, "jrefit": lens, "logit": None})
    lp = payload["n_prompt_tokens"] - 1
    assert a["positions"].tolist() == [3, 4] + list(range(lp, payload["n_tokens"]))
    assert payload["last_prompt_index"] == 2 and payload["lens_names"] == ["j56", "jrefit", "logit"]
    assert (a["ranks"][0] == a["ranks"][1]).all()


def test_torch_ranks_refuses_non_finite_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 2.0, 1.0, 2.0]])
    assert M.torch_ranks(logits, [0, 1, 2]).tolist() == [[4, 1, 3]]
    with pytest.raises(ValueError):
        M.torch_ranks(torch.tensor([[0.0, float("nan")]]), [0])


def test_row44_inputs_positions_cover_evidence_and_question(tok):
    E = _script("exp45_maps")
    prompt = "Intro.\n\n## Memory\nuser, d1: hi\nuser, d2: I paid $800 for it.\n\n## Current input\nQuestion: How much?\n\nNote:"
    ev = prompt.index("user, d2"); q = prompt.index("Question:")
    row = {"key": "q1_clean", "question_id": "q1", "fold": 0, "kind": "clean", "prompt": prompt, "prompt_sha256": "s",
           "gold": "$800", "substitute": "$900", "gold_read_id": tok("8", add_special_tokens=False).input_ids[0],
           "substitute_read_id": tok("9", add_special_tokens=False).input_ids[0], "label23": "wrong",
           "question_type": "t", "output23": "o", "evidence_span": [ev, ev + len("user, d2: I paid $800 for it.")],
           "question_span": [q, q + len("Question: How much?")]}
    x, meta = E.row44_inputs(tok, row, 16)
    chosen = "".join(tok.decode([tok(x.chat_text, add_special_tokens=False).input_ids[p]]) for p in x.positions)
    assert "I paid $800 for it." in chosen and "Question: How much?" in chosen and "Intro" not in chosen
    assert meta["competitors"] == ["$900"] and {p["surface"] for p in meta["pinned"]} >= {"$800", "$900", "unknown"}
    assert [d["surface"] for d in meta["pinned_dropped"]] == ["$900"]          # bare "$" is $800's: left out


# ---------------------------------------------------------------------------------------------- runner, report, render

@pytest.fixture(scope="module")
def runs(tok, tiny, cases, tmp_path_factory):
    """The runner end to end on the tiny model: smoke, stage 1 (4 prompts), resume, deadline, stage 3 (2 prompts)."""
    E = _script("exp45_maps")
    C = _script("exp45_cases")
    _, _, _, d = tiny
    work = tmp_path_factory.mktemp("work")
    C.write_cases(cases, work, C.check_cases(cases, tok), 45)
    prompt = "Intro.\n\n## Memory\nuser, d1: I live in York.\n\n## Current input\nQuestion: Where?\n\nNote:"
    rows = [{"key": f"q{i}_{k}", "question_id": f"q{i}", "fold": 0, "kind": k, "prompt": prompt, "prompt_sha256": "s",
             "gold": "York", "substitute": "Leeds", "gold_read_id": tok(" York", add_special_tokens=False).input_ids[0],
             "substitute_read_id": tok(" Leeds", add_special_tokens=False).input_ids[0], "label23": "success",
             "question_type": "t", "output23": None, "evidence_span": [prompt.index("user, d1"), prompt.index("York.") + 5],
             "question_span": [prompt.index("Question:"), prompt.index("\n\nNote:")]} for i, k in ((0, "clean"), (0, "swap"))]
    (work / "rows44.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    base = ["--lens56", str(d / "lens.pt"), "--allow-lens-sha", "--weights", str(d / "qwen3b"), "--device", "cpu",
            "--cases", str(work / "cases.jsonl"), "--rows44", str(work / "rows44.jsonl"), "--max-new", "3", "--max-new-note", "12"]
    res = {"work": work, "d": d}
    res["smoke"] = E.main(base + ["--smoke", "3", "--out", str(work / "smoke")])
    res["smoke_bad"] = E.main(base + ["--smoke", "1", "--out", str(work / "smoke_bad"), "--break-bos"])
    res["ref"] = E.main(base + ["--write-reference", "--out", str(work / "ref")])
    res["smoke_ref"] = E.main(base + ["--smoke", "3", "--out", str(work / "smoke2"), "--reference", str(work / "ref")])
    res["s1"] = E.main(base + ["--stage", "1", "--limit", "4", "--root", str(work / "maps")])
    res["s1_again"] = E.main(base + ["--stage", "1", "--limit", "4", "--root", str(work / "maps")])
    res["s1_deadline"] = E.main(base + ["--stage", "1", "--limit", "7", "--root", str(work / "maps"),
                                        "--deadline", str(time.time() + 0.5)])
    res["s3"] = E.main(base + ["--stage", "3", "--root", str(work / "maps")])
    return res


def test_runner_smoke_resume_deadline_and_stage3(runs):
    work = runs["work"]
    assert runs["smoke"] == 0 and json.loads((work / "smoke/_smoke.json").read_text())["pass"]
    assert runs["smoke_bad"] == 5 and not json.loads((work / "smoke_bad/_smoke.json").read_text())["pass"]
    assert runs["ref"] == 0 and len(json.loads((work / "ref/reference.json").read_text())["cases"]) == 3
    assert runs["smoke_ref"] == 0                                    # same weights, same device: top-1 equal
    s1 = work / "maps/j56-logit/stage1"
    timing = json.loads((s1 / "_timing.json").read_text())["runs"]
    assert runs["s1"] == 0 and timing[0]["prompts"] == 4 and runs["s1_again"] == 0 and timing[1]["prompts"] == 0
    assert runs["s1_deadline"] == 9 and 1 <= timing[2]["prompts"] < 3
    names = sorted(p.name for p in s1.glob("*.json") if not p.name.startswith("_"))
    assert names[0].startswith("DE00_v0_F1_H0") and all(json.loads((s1 / n).read_text())["split"] == "dev" for n in names)
    payload = json.loads((s1 / names[0]).read_text())
    assert payload["outcome"] in M.OUTCOMES and payload["stamp"]["maps_version"] == "exp45_maps_v2"
    f2 = json.loads((s1 / "DE00_v0_F2_H0.json").read_text())
    assert f2["answer_span_rule"] == "after_marker" and f2["n_generated"] <= 12 and f2["max_new"] == 12
    assert payload["answer_span_rule"] == "all" and payload["max_new"] == 3
    assert payload["counted_positions"] == list(range(payload["last_prompt_index"], payload["n_tokens"]))
    assert set(payload["lens_sha256"]) == {"j56", "logit"}
    assert runs["s3"] == 0 and len(list((work / "maps/j56-logit/stage3").glob("q0_*.npz"))) == 2


def test_refit_is_used_only_above_its_minimum(tmp_path):
    E = _script("exp45_maps")
    pt = tmp_path / "qwen3b.pt"
    assert E.refit_usable(None)[0] is False and E.refit_usable(str(pt))[0] is False
    pt.write_bytes(b"x")
    (tmp_path / "qwen3b.json").write_text(json.dumps({"n_prompts": 90, "n_minimum": 96, "below_minimum": True}))
    assert E.refit_usable(str(pt))[0] is False
    (tmp_path / "qwen3b.json").write_text(json.dumps({"n_prompts": 128, "n_minimum": 96, "below_minimum": False}))
    assert E.refit_usable(str(pt))[0] is True
    R = _script("exp45_pod_run")
    assert R.refit_box(14000, 75, 4200, 140) == 75 and R.refit_box(5000, 75, 4200, 140) == 11 and R.refit_box(4500, 75, 4200, 140) == 0


def test_report_and_render_run_on_the_runner_outputs(runs, monkeypatch):
    work = runs["work"]
    rep = _script("exp45_report")
    assert rep.main(["--root", str(work / "maps"), "--half", "all", "--swap-dir", str(work / "nothing")]) == 0
    out = json.loads((work / "maps/j56-logit/report_all.json").read_text())
    assert out["stage1"]["n_prompts"] >= 4 and len(out["stage3"]) == 2
    assert out["stage2"]["too_few"] and "too few failures" in (work / "maps/j56-logit/report_all.md").read_text()
    assert rep.main(["--root", str(work / "maps"), "--half", "dev"]) == 0
    ren = _script("exp45_render")
    assert ren.main(["--dir", str(work / "maps/j56-logit/stage1"), "--limit", "2"]) == 0
    pages = list((work / "maps/j56-logit/stage1/html").glob("*.html"))
    assert len(pages) == 3 and "data:image/png;base64," in next(p for p in pages if p.name != "index.html").read_text()


def test_report_calls_criteria_by_task(tok):
    rep = _script("exp45_report")
    pinned = M.pin_tokens(tok, [("gold", "York"), ("intermediate", "Ben"), ("entity", "Alice")])
    L, P, T = 4, 3, len(pinned)
    ranks = np.full((2, L, P, T), 900, dtype=np.int32)
    gold = M.entity_columns(pinned, "York")
    ranks[0, 1, 1, gold[0]] = 3                     # j56: gold readable at layer 1, last prompt token
    ranks[:, 3, 2, M.entity_columns(pinned, "Ben")[0]] = 1
    payload = {"pinned": pinned, "gold": "York", "intermediate": "Ben", "competitors": [], "task": "RC",
               "lens_names": ["j56", "logit"], "last_prompt_index": 1}
    c = rep.calls(payload, {"ranks": ranks, "layers": np.arange(L)})
    assert c["j56"]["a_gold"]["readable"] and not c["logit"]["a_gold"]["readable"]
    assert c["j56"]["b_intermediate"]["first_layer"] == 3 and c["j56"]["c_competitor"] is None
    payload["task"] = "DE"
    assert rep.calls(payload, {"ranks": ranks, "layers": np.arange(L)})["j56"]["b_intermediate"] is None


# ---------------------------------------------------------------------------------------------- rev 1 rulings 1-3

def test_ruling1_space_form_everywhere_bare_form_only_at_the_last_prompt_token():
    # T columns: 0 = " York" (sp), 1 = "York" (bare); P = 5, last prompt token at index 2
    r = np.full((4, 5, 2), 500, dtype=np.int32)
    layers = [0, 1, 2, 3]
    r[1, 4, 1] = 1                                   # bare form hit at a generated position: does not count
    got = M.readability_rule(r, [0], [1], layers, [2, 3, 4], [2])
    assert got["readable"] is False
    r[2, 2, 1] = 4                                   # bare form at the last prompt token: counts
    got = M.readability_rule(r, [0], [1], layers, [2, 3, 4], [2])
    assert got == {"readable": True, "first_layer": 2, "first_depth": 0.5, "first_pos": 2, "best_rank": 4}
    r[0, 3, 0] = 9                                   # space form at a counted generated position: earlier layer wins
    assert M.readability_rule(r, [0], [1], layers, [2, 3, 4], [2])["first_layer"] == 0
    assert M.readability(r, [0, 1], layers, 2)["first_layer"] == 0     # "either form anywhere" beside


def test_ruling3_counted_positions_exclude_note_tokens():
    r = np.full((3, 6, 1), 500, dtype=np.int32)
    layers = [0, 1, 2]
    r[1, 3, 0] = 2                                   # a note token (index 3), before the answer span (index 5)
    assert M.readability_rule(r, [0], [], layers, [1, 5], [1])["readable"] is False          # fixed
    assert M.readability_rule(r, [0], [], layers, [1], [1])["readable"] is False             # last prompt only
    assert M.readability_rule(r, [0], [], layers, list(range(1, 6)), [1])["readable"] is True  # any generated
    assert M.readability_rule(r, [], [], layers, [1], [1])["readable"] is None


def test_answer_line_stop_rule_and_answer_start(tok):
    assert not M.answer_line_done("Alice lives in Leeds. Answer: Leeds")
    assert M.answer_line_done("Alice lives in Leeds.\nAnswer: Leeds\n")
    assert not M.answer_line_done("Answer:\n") and M.answer_line_done("Answer:\nLeeds\n")
    ids = tok("Alice lived in York, then Leeds.\nAnswer: Leeds\n", add_special_tokens=False).input_ids
    j0 = M.answer_start_token(tok, ids)
    assert tok.decode(ids[j0:]).strip() == "Leeds" and tok.decode(ids[:j0]).endswith("Answer:")
    assert M.answer_start_token(tok, tok("no marker here", add_special_tokens=False).input_ids) is None


@pytest.mark.parametrize("text, found", [("Note: Alice moved.\nAnswer: York\n", True), ("Note: Alice moved to", False)])
def test_map_prompt_counts_the_answer_span_after_the_marker(tok, tiny, monkeypatch, text, found):
    from memctl.analysis.jacobian import wrap_chat

    model, lm, lens, _ = tiny
    gen = tok(text, add_special_tokens=False).input_ids
    seen = {}
    def fake(hf, ids, max_new, eos, tokenizer=None, stop_after_answer=False):
        seen.update(max_new=max_new, stop=stop_after_answer)
        return list(gen)
    monkeypatch.setattr(M, "generate_greedy", fake)
    x = M.MapInputs("k", wrap_chat(tok, "Alice lived in Leeds.\n\nWhere?"), [100], None, 160, "after_marker")
    payload, a = M.map_prompt(model, lm, tok, x, {"j56": lens, "logit": None})
    lp = payload["last_prompt_index"]
    assert seen == {"max_new": 160, "stop": True} and payload["answer_span_found"] is found
    if found:
        j0 = payload["answer_start_token"]
        assert payload["counted_positions"] == [lp] + list(range(lp + j0, payload["n_tokens"]))
        assert payload["position_tokens"][lp + j0].endswith(":")      # the marker's last token predicts the answer
    else:
        assert payload["counted_positions"] == [lp]


def test_case_and_row_inputs_set_generation_by_format(tok, cases):
    E = _script("exp45_maps")
    by = {c["case_id"]: c for c in cases}
    x1, _ = E.case_inputs(tok, by["DE00_v0_F1_H0"], 16)
    x2, _ = E.case_inputs(tok, by["DE00_v0_F2_H0"], 16)
    assert (x1.max_new, x1.answer_span) == (16, "all") and (x2.max_new, x2.answer_span) == (160, "after_marker")
    prompt = "Intro.\n\n## Memory\nuser, d1: I live in York.\n\n## Current input\nQuestion: Where?\n\nNote:"
    row = {"key": "q_clean", "question_id": "q", "fold": 0, "kind": "clean", "prompt": prompt, "prompt_sha256": "s",
           "gold": "York", "substitute": "Leeds", "gold_read_id": 4261, "substitute_read_id": 32371, "label23": "x",
           "question_type": "t", "output23": None, "evidence_span": [20, 40], "question_span": [60, 75]}
    x3, _ = E.row44_inputs(tok, row, 16)
    assert (x3.max_new, x3.answer_span) == (160, "after_marker")


def test_report_variants_follow_the_rulings(tok):
    rep = _script("exp45_report")
    pinned = M.pin_tokens(tok, [("gold", "York")])
    sp, bare = M.form_columns(pinned, "York")
    ranks = np.full((4, 6, len(pinned)), 900, dtype=np.int32)
    ranks[1, 3, sp[0]] = 2                           # note token (not counted)
    ranks[2, 1, bare[0]] = 5                          # bare at a counted answer position that is not the last prompt
    payload = {"pinned": pinned, "gold": "York", "last_prompt_index": 1, "counted_positions": [1, 5]}
    layers = [0, 1, 2, 3]
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload)["first_layer"] == 2      # bare at lp counts
    ranks[2, 1, bare[0]] = 900
    ranks[2, 5, bare[0]] = 5
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload)["readable"] is False
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "any generated token")["first_layer"] == 1
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "either form anywhere")["readable"] is True
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "last prompt token only")["readable"] is False


# ---------------------------------------------------------------------------------------------- post hoc echo rules

def test_echo_rule_a_drops_only_the_position_fed_the_target():
    # stored positions 0..5, lp = 1; X = ids {7, 8} (two forms); X is fed in at position 3
    input_at = [50, 51, 52, 7, 53, 54]
    counted = [1, 2, 3, 4, 5]
    assert M.echo_counted(counted, input_at, [7, 8], "A", 1) == [1, 2, 4, 5]
    assert M.echo_counted(counted, [50, 51, 52, 8, 53, 7], [7, 8], "A", 1) == [1, 2, 4]   # either form
    r = np.full((3, 6, 2), 500, dtype=np.int32)      # columns: 0 = sp, 1 = bare of X
    r[0, 3, 0] = 1                                   # the echo: X ranked 1 at layer 0 where X is the input
    layers = [0, 1, 2]
    assert M.readability_rule(r, [0], [1], layers, counted, [1], exclude_final=True)["first_layer"] == 0   # fixed
    a = M.readability_rule(r, [0], [1], layers, M.echo_counted(counted, input_at, [7, 8], "A", 1), [1],
                           exclude_final=True)
    assert a["readable"] is False
    r[1, 4, 0] = 3                                   # a hit at a later, non-echo position still counts under A
    a = M.readability_rule(r, [0], [1], layers, M.echo_counted(counted, input_at, [7, 8], "A", 1), [1],
                           exclude_final=True)
    assert a["readable"] and a["first_pos"] == 4


def test_echo_rule_b_keeps_positions_up_to_the_one_producing_the_first_occurrence():
    input_at = [50, 51, 52, 7, 53, 7]                # X first fed in at 3, so position 2 produced it
    counted = [1, 2, 3, 4, 5]
    assert M.echo_counted(counted, input_at, [7], "B", 1) == [1, 2]
    assert M.echo_counted(counted, [50, 51, 52, 53, 54, 55], [7], "B", 1) == counted      # never produced
    assert M.echo_counted([1, 4, 5], input_at, [7], "B", 1) == [1]                         # F2: note held X
    # X fed in at the last prompt token itself does not drop it (only lp+1.. are searched)
    assert M.echo_counted(counted, [50, 7, 52, 53, 54, 55], [7], "B", 1) == counted
    with pytest.raises(ValueError):
        M.echo_counted(counted, input_at, [7], "C", 1)


def test_echo_rules_leave_the_bare_form_at_the_last_prompt_token(tok):
    rep = _script("exp45_report")
    pinned = M.pin_tokens(tok, [("gold", "York")])
    sp, bare = M.form_columns(pinned, "York")
    sp_id = pinned[sp[0]]["id"]
    ranks = np.full((4, 6, len(pinned)), 900, dtype=np.int32)
    ranks[2, 1, bare[0]] = 5                         # bare form at the last prompt token (index 1)
    ranks[0, 4, sp[0]] = 1                           # echo of " York" fed in at position 4
    payload = {"pinned": pinned, "gold": "York", "last_prompt_index": 1, "counted_positions": [1, 2, 3, 4, 5]}
    input_at = np.array([9, 10, 11, sp_id, sp_id, 12])            # " York" first fed in at 3 (produced at 2)
    layers = [0, 1, 2, 3]
    fixed = rep.entity_call(ranks, pinned, ["York"], layers, payload, exclude_final=True)
    assert fixed["first_layer"] == 0                 # inflated by the echo
    for rule in ("rule A", "rule B"):
        got = rep.entity_call(ranks, pinned, ["York"], layers, payload, rule, exclude_final=True, input_at=input_at)
        assert got["readable"] and got["first_layer"] == 2 and got["first_pos"] == 1      # the bare hit at lp
    ranks[1, 3, sp[0]] = 4                           # position 3 is fed " York": excluded by A and B
    ranks[1, 2, sp[0]] = 900
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "rule A", exclude_final=True,
                           input_at=input_at)["first_layer"] == 2
    ranks[1, 5, sp[0]] = 4                           # position 5: not an echo (A counts), after production (B not)
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "rule A", exclude_final=True,
                           input_at=input_at)["first_layer"] == 1
    assert rep.entity_call(ranks, pinned, ["York"], layers, payload, "rule B", exclude_final=True,
                           input_at=input_at)["first_layer"] == 2
    with pytest.raises(ValueError):
        rep.entity_call(ranks, pinned, ["York"], layers, payload, "rule A")


def test_report_rules_read_input_tokens_at_the_stored_positions(tok):
    # stage 3 keeps a subset of prompt positions: the input token at stored index p is input_ids[positions[p]]
    rep = _script("exp45_report")
    pinned = M.pin_tokens(tok, [("gold", "York")])
    sp, _ = M.form_columns(pinned, "York")
    sp_id = pinned[sp[0]]["id"]
    full = np.array([1, 2, sp_id, 3, 4, 5, 6, 7], dtype=np.int32)    # " York" in the prompt at sequence index 2
    positions = np.array([0, 5, 6, 7], dtype=np.int32)              # lp = stored 1 (sequence 5), then generated
    ranks = np.full((2, 4, 4, len(pinned)), 900, dtype=np.int32)
    ranks[:, 2, 2, sp[0]] = 3                                       # hit at stored 2 (fed token 6, not " York")
    payload = {"pinned": pinned, "gold": "York", "intermediate": None, "competitors": [], "task": "DE",
               "lens_names": ["j56", "logit"], "last_prompt_index": 1, "counted_positions": [1, 2, 3]}
    c = rep.calls(payload, {"ranks": ranks, "layers": np.arange(4), "input_ids": full, "positions": positions})
    assert c["j56"]["a_gold | rule B"]["readable"] and c["j56"]["a_gold | rule A"]["readable"]
    assert c["j56"]["a_gold | rule B"]["first_pos"] == 2
    assert c["j56"]["b_intermediate | rule A"] is None and c["j56"]["a_gold | rule A, incl. final layer"] is None
