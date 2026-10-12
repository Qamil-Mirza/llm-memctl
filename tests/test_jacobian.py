import gzip, json
from pathlib import Path

import pytest

from memctl.jacobian_data import PromptRecord, load_records, memory_line

RUNS = Path("runs")
CACHE = Path("cache/generations_exp23_qwen3b")
needs_exp23 = pytest.mark.skipif(not (RUNS / "exp23_qwen3b_f0").exists() or not CACHE.exists(),
                                 reason="§23 traces or cache not present")


def _item(ep, item_id, content, date="2023/05/20 (Sat) 02:21", speaker="user", created_at=1):
    return {"experiment_id": "fixed8__fraction0.05", "episode_id": ep, "id": item_id, "content": content,
            "created_at": created_at, "source_type": speaker, "metadata": {"speaker": speaker, "date": date}}


def test_memory_line_matches_compact_reader_format():
    item = _item("ep00000", "s1:0", "I was born in Leeds.")
    assert memory_line(item) == "user, 2023/05/20 (Sat) 02:21: I was born in Leeds."
    item["metadata"] = {"speaker": "assistant"}
    assert memory_line(item) == "assistant: I was born in Leeds."


def _write_fake_run(tmp_path, prompt_lines, question="(asked on 2023/05/30 (Tue) 23:40) Where was I born?",
                    output="Note\nAnswer: Leeds", qid="q_abc", complete=1.0):
    """A one-episode fixed8 run folder plus a cache folder whose prompt is rebuilt from the lines."""
    run = tmp_path / "runs" / "exp23_fake_f0" / "fixed8__fraction0.05"
    run.mkdir(parents=True)
    items = [_item("ep00000", f"s1:{n}", content, created_at=n + 1) for n, content in enumerate(prompt_lines)]
    obs = _item("ep00000", "q:0", f"Question: {question}", created_at=99)
    with gzip.open(run / "items.jsonl.gz", "wt") as fh:
        for it in items + [obs]:
            fh.write(json.dumps(it) + "\n")
    step = {"episode_id": "ep00000", "step": 5, "observation_id": "q:0", "scored": True,
            "memory": {"active_ids": [it["id"] for it in items] + ["q:0"]}, "agent_output": output}
    with gzip.open(run / "steps.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"episode_id": "ep00000", "step": 1, "scored": False, "memory": {"active_ids": []}}) + "\n")
        fh.write(json.dumps(step) + "\n")
    (run / "episodes.jsonl").write_text(json.dumps(
        {"episode_id": "ep00000", "seed": 0, "evidence_complete_rate": complete, "env_stats": {"episode": qid}}) + "\n")
    memory = "\n".join(memory_line(it) for it in items)
    prompt = f"Instructions here.\n\n## Memory\n{memory}\n\n## Current input\nQuestion: {question}\n\nNote:"
    cache = tmp_path / "cache"
    (cache / "00").mkdir(parents=True)
    (cache / "00" / "deadbeef.json").write_text(json.dumps({"prompt": prompt, "output": output}))
    verdicts = [{"reader": "fake", "arm": "fixed8", "question_id": qid, "type": "single-session-user",
                 "question": question, "gold": "Leeds", "answer": "Leeds", "unknown": False, "correct": True}]
    (tmp_path / "runs" / "exp23_verdicts.json").write_text(json.dumps(verdicts))
    return run, cache, prompt


def test_load_records_rebuilds_prompt_and_marks_evidence(tmp_path, monkeypatch):
    lines = ["Hello, how are you?", "I was born in Leeds, by the way.\nSecond line of the same turn.", "Nice weather."]
    run, cache, prompt = _write_fake_run(tmp_path, lines)
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids",
                        lambda data_path, question_id: (frozenset({"s1:1"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta",
                        lambda data_path, question_id: ("Leeds", "single-session-user", True))
    records, dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert dropped == {"no_cached_prompt": 0, "evidence_not_found": 0}
    [r] = records
    assert isinstance(r, PromptRecord)
    assert r.prompt == prompt and r.output == "Note\nAnswer: Leeds"
    assert r.memory_lines == tuple(memory_line(_item("ep00000", f"s1:{n}", c)) for n, c in enumerate(lines))
    assert r.evidence_lines == (r.memory_lines[1],)
    assert r.all_in_view is True and r.correct is True and r.unknown is False and r.gold == "Leeds"


def test_load_records_fallback_session_marks_every_in_view_turn(tmp_path, monkeypatch):
    lines = ["turn a", "turn b", "turn c"]
    run, cache, _ = _write_fake_run(tmp_path, lines)
    # a fallback session: any one of s1:0 / s1:2 counts, so both in-view lines are evidence
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids",
                        lambda data_path, question_id: (frozenset({"s1:0", "s1:2"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta",
                        lambda data_path, question_id: ("x", "multi-session", True))
    [r], _ = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert r.evidence_lines == (r.memory_lines[0], r.memory_lines[2])


def test_load_records_drops_prompt_without_cache_entry(tmp_path, monkeypatch):
    run, cache, _ = _write_fake_run(tmp_path, ["turn a"])
    (cache / "00" / "deadbeef.json").unlink()
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids", lambda d, q: (frozenset({"s1:0"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta", lambda d, q: ("x", "t", True))
    records, dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert records == [] and dropped["no_cached_prompt"] == 1


def test_load_records_keeps_partial_view_record_with_all_in_view_false(tmp_path, monkeypatch):
    run, cache, _ = _write_fake_run(tmp_path, ["turn a"], complete=0.0)
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids", lambda d, q: (frozenset({"zzz:9"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta", lambda d, q: ("x", "t", True))
    [r], dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert r.evidence_lines == () and r.all_in_view is False and dropped["evidence_not_found"] == 0


def test_load_records_drops_complete_answerable_prompt_whose_evidence_is_missing(tmp_path, monkeypatch):
    run, cache, _ = _write_fake_run(tmp_path, ["turn a"], complete=1.0)
    monkeypatch.setattr("memctl.jacobian_data.evidence_item_ids", lambda d, q: (frozenset({"zzz:9"}),))
    monkeypatch.setattr("memctl.jacobian_data.question_meta", lambda d, q: ("x", "t", True))
    records, dropped = load_records("fake", 0, runs_dir=str(tmp_path / "runs"), cache_dir=str(cache), data_path="unused")
    assert records == [] and dropped["evidence_not_found"] == 1 and dropped["no_cached_prompt"] == 0


@needs_exp23
def test_real_fold0_qwen3b_records_all_match():
    records, dropped = load_records("qwen3b", 0)
    assert dropped["no_cached_prompt"] == 0
    assert len(records) == 100
    in_view = [r for r in records if r.all_in_view and r.answerable]
    assert all(r.evidence_lines for r in in_view)
    assert all(line in r.prompt for r in records for line in r.memory_lines)


from memctl.analysis.jacobian import REGIONS, segment_prompt, token_regions

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nassistant, d1: turn b\nhas newline\n"
          "user, d2: turn c\n\n## Current input\nQuestion: what?\n\nNote:")
LINES = ("user, d1: turn a", "assistant, d1: turn b\nhas newline", "user, d2: turn c")


def test_segment_prompt_tiles_the_prompt_in_order():
    segments = segment_prompt(PROMPT, LINES, evidence_lines=(LINES[1],))
    assert segments[0][1] == 0 and segments[-1][2] == len(PROMPT)
    assert all(a[2] == b[1] for a, b in zip(segments, segments[1:]))
    assert [s[0] for s in segments] == ["instruction", "distractor", "evidence", "distractor", "question"]
    region_text = {s[0]: PROMPT[s[1]:s[2]] for s in segments}
    assert region_text["evidence"].startswith("assistant, d1: turn b\nhas newline")
    assert region_text["question"].startswith("\n\n## Current input")
    assert region_text["instruction"] == "Use only the memory.\n\n## Memory\n"


def test_segment_prompt_handles_a_line_with_a_newline_by_exact_search():
    # a naive split on "\n" would see four memory lines; exact search sees three
    segments = segment_prompt(PROMPT, LINES, evidence_lines=())
    assert sum(1 for s in segments if s[0] == "distractor") == 3


def test_segment_prompt_rejects_missing_line():
    with pytest.raises(ValueError):
        segment_prompt(PROMPT, LINES + ("user, d9: not there",), evidence_lines=())


def test_token_regions_assigns_by_token_start_and_marks_template():
    segments = [("instruction", 0, 5), ("evidence", 5, 9), ("question", 9, 12)]
    # chat text = "<s>" + prompt(12 chars) + "<e>"; offsets in chat-text coordinates
    offsets = [(0, 3), (3, 5), (5, 8), (8, 10), (10, 14), (14, 15), (15, 18)]
    regions = token_regions(offsets, prompt_start=3, prompt_len=12, segments=segments)
    assert regions == ["template", "instruction", "instruction", "evidence", "evidence", "question", "template"]
    assert set(regions) <= set(REGIONS)


import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.analysis.jacobian import (JacobianResult, TARGETS, continuation_ids, jacobian_pass, target_value,
                                      unknown_id, wrap_chat)


@pytest.fixture(scope="module")
def tiny():
    """The Qwen2.5 tokenizer (chat template) with a 2-layer random Qwen2 model of the same vocabulary."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:  # offline
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    return model, tok


def test_wrap_chat_contains_prompt_verbatim_and_generation_prompt(tiny):
    model, tok = tiny
    text = wrap_chat(tok, PROMPT)
    assert PROMPT in text
    assert text.rstrip().endswith("<|im_start|>assistant")


def test_continuation_ids_gold_scores_exactly_the_gold_tokens(tiny):
    _, tok = tiny
    ids, start = continuation_ids(tok, "gold", gold="4 hours", output="")
    assert ids[:start] == tok("Answer:", add_special_tokens=False).input_ids
    assert ids[start:] == tok(" 4 hours", add_special_tokens=False).input_ids
    m_ids, m_start = continuation_ids(tok, "margin", gold="4 hours", output="")
    assert m_ids == ids[: start + 1] and m_start == start
    o_ids, o_start = continuation_ids(tok, "own", gold="", output="Note\nAnswer: unknown")
    assert o_ids == tok("Note\nAnswer: unknown", add_special_tokens=False).input_ids and o_start == 0
    assert unknown_id(tok) == tok(" unknown", add_special_tokens=False).input_ids[0]


def test_jacobian_pass_shapes_and_target_values(tiny):
    model, tok = tiny
    out = jacobian_pass(model, tok, PROMPT, LINES, (LINES[1],), gold="4 hours", output="Answer: nope", targets=TARGETS)
    assert set(out) == set(TARGETS)
    for result in out.values():
        assert isinstance(result, JacobianResult)
        assert len(result.regions) == result.n_tokens == len(result.g_l2) == len(result.g_gxi)
        assert np.isfinite(result.g_l2).all() and (result.g_l2 >= 0).all()
        assert "evidence" in result.regions and "template" in result.regions
    # the gold target equals a direct teacher-forced log-prob sum
    text = wrap_chat(tok, PROMPT)
    ids, start = continuation_ids(tok, "gold", "4 hours", "")
    full = tok(text, add_special_tokens=False).input_ids + ids
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([full])).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    n = len(full) - len(ids)
    expected = sum(logp[n + j - 1, ids[j]].item() for j in range(start, len(ids)))
    assert out["gold"].target_value == pytest.approx(expected, rel=1e-4, abs=1e-4)
    assert target_value(model, tok, PROMPT, "4 hours", "", "gold") == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_gradient_is_nonzero_only_where_it_should_be(tiny):
    model, tok = tiny
    out = jacobian_pass(model, tok, PROMPT, LINES, (LINES[1],), gold="4 hours", output="", targets=("gold",))
    assert out["gold"].g_l2.sum() > 0


from memctl.analysis.jacobian import ablate_evidence, evidence_lift, measures, region_counts, region_shares, topk_fraction


def test_region_shares_sum_to_one_and_ignore_template():
    regions = ["template", "instruction", "evidence", "evidence", "distractor", "question", "template"]
    g = np.array([100.0, 1.0, 2.0, 2.0, 4.0, 1.0, 100.0])
    shares = region_shares(g, regions)
    assert shares == pytest.approx({"instruction": 0.1, "evidence": 0.4, "distractor": 0.4, "question": 0.1})
    assert region_counts(regions) == {"template": 2, "instruction": 1, "evidence": 2, "distractor": 1, "question": 1}
    assert evidence_lift(shares, region_counts(regions)) == pytest.approx(0.4 / (2 / 5))
    assert evidence_lift({"evidence": 0.0}, {"evidence": 0, "instruction": 3}) is None


def test_topk_fraction_counts_evidence_among_largest():
    regions = ["template", "instruction", "evidence", "evidence", "distractor", "question"]
    g = np.array([9.0, 1.0, 5.0, 4.0, 3.0, 0.5])
    assert topk_fraction(g, regions, k=2) == 1.0
    assert topk_fraction(g, regions, k=4) == 0.5
    assert topk_fraction(g, regions, k=50) == pytest.approx(2 / 5)


def test_measures_bundles_both_variants():
    result = JacobianResult(["template", "evidence", "distractor"], np.array([1.0, 3.0, 1.0]), np.array([1.0, 1.0, 3.0]), -2.5, 3)
    m = measures(result, k=1)
    assert m["l2"]["shares"]["evidence"] == 0.75 and m["gxi"]["shares"]["evidence"] == 0.25
    assert m["l2"]["topk"] == 1.0 and m["gxi"]["topk"] == 0.0
    assert m["target_value"] == -2.5 and m["n_tokens"] == 3 and m["counts"]["evidence"] == 1


def test_ablate_evidence_removes_the_lines_and_keeps_the_rest_intact():
    ablated = ablate_evidence(PROMPT, LINES, (LINES[1],))
    assert LINES[1] not in ablated
    assert ablated == ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nuser, d2: turn c\n\n## Current input\n"
                       "Question: what?\n\nNote:")
    assert ablate_evidence(PROMPT, LINES, ()) == PROMPT


def test_ablate_evidence_removes_the_first_memory_line_with_the_newline_after_it():
    assert ablate_evidence(PROMPT, LINES, (LINES[0],)) == (
        "Use only the memory.\n\n## Memory\nassistant, d1: turn b\nhas newline\nuser, d2: turn c\n\n"
        "## Current input\nQuestion: what?\n\nNote:")


def test_ablate_evidence_removes_two_lines_at_once():
    expected = "Use only the memory.\n\n## Memory\nuser, d2: turn c\n\n## Current input\nQuestion: what?\n\nNote:"
    assert ablate_evidence(PROMPT, LINES, (LINES[0], LINES[1])) == expected
    assert ablate_evidence(PROMPT, LINES, (LINES[1], LINES[2])) == (
        "Use only the memory.\n\n## Memory\nuser, d1: turn a\n\n## Current input\nQuestion: what?\n\nNote:")


def _memory_prompt(lines):
    return "Intro.\n\n## Memory\n" + "\n".join(lines) + "\n\n## Current input\nQuestion: q?\n\nNote:"


def test_ablate_evidence_removes_only_the_exact_line_on_a_prefix_collision():
    lines = ("user, d1: Yes, but later", "user, d1: Yes", "user, d2: other")
    assert ablate_evidence(_memory_prompt(lines), lines, ("user, d1: Yes",)) == _memory_prompt((lines[0], lines[2]))
    flipped = (lines[1], lines[0], lines[2])
    assert ablate_evidence(_memory_prompt(flipped), flipped, ("user, d1: Yes",)) == _memory_prompt((lines[0], lines[2]))


def test_ablate_evidence_treats_an_identical_copy_as_evidence_too():
    # evidence is marked by content, so both copies of a duplicated line are evidence and both go
    lines = ("user, d1: same", "user, d2: keep", "user, d1: same")
    prompt = _memory_prompt(lines)
    assert [s[0] for s in segment_prompt(prompt, lines, (lines[2],))][1:-1] == ["evidence", "distractor", "evidence"]
    assert ablate_evidence(prompt, lines, (lines[2],)) == _memory_prompt(("user, d2: keep",))


def test_ablate_evidence_rejects_an_evidence_line_not_in_memory():
    with pytest.raises(ValueError):
        ablate_evidence(PROMPT, LINES, ("user, d9: not there",))
    with pytest.raises(ValueError):  # present in the prompt text but not passed as a memory line
        ablate_evidence(PROMPT, LINES[:2], (LINES[2],))


from memctl.analysis.jacobian_stats import auc, bootstrap_auc, bootstrap_spearman, paired_claim


def test_auc_is_mann_whitney_with_half_credit_for_ties():
    assert auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_bootstrap_auc_interval_covers_point_and_is_in_unit_range():
    rng = np.random.default_rng(0)
    pos = list(rng.normal(1.0, 1.0, 60)); neg = list(rng.normal(0.0, 1.0, 60))
    r = bootstrap_auc(pos, neg, samples=500)
    assert r["n_pos"] == 60 and r["n_neg"] == 60
    assert 0 <= r["ci_low"] <= r["auc"] <= r["ci_high"] <= 1 and r["ci_low"] > 0.5


def test_bootstrap_spearman_detects_a_monotone_relation():
    x = list(range(50)); y = [v ** 2 + (1 if v % 2 else -1) for v in x]
    r = bootstrap_spearman(x, y, samples=500)
    assert r["n"] == 50 and r["rho"] > 0.9 and r["ci_low"] > 0.5


def test_paired_claim_uses_shared_keys_and_gives_a_verdict():
    a = {"q1": 0.5, "q2": 0.6, "q3": 0.7, "q4": 0.8, "zz": 9.0}
    b = {"q1": 0.1, "q2": 0.2, "q3": 0.3, "q4": 0.4}
    r = paired_claim(a, b, samples=500)
    assert r["n"] == 4 and r["mean"] == pytest.approx(0.4) and r["verdict"] == "SHOWN"
    assert paired_claim(b, a, samples=500)["verdict"] == "NOT SHOWN"


import importlib.util, sys


def _script():
    spec = importlib.util.spec_from_file_location("jacobian_lens", "scripts/jacobian_lens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_paths_and_resume_skip(tmp_path):
    s = _script()
    out = tmp_path / "o"
    assert s.output_path(out, "q1", "gold").name == "q1.gold.json"
    assert not s.is_done(out, "q1", ("gold", "margin"))
    out.mkdir()
    s.write_json(s.output_path(out, "q1", "gold"), {"ok": 1})
    s.write_json(s.output_path(out, "q1", "margin"), {"ok": 1})
    assert s.is_done(out, "q1", ("gold", "margin"))
    (out / "q1.margin.json").write_text("{not json")  # a partial file from a crash
    assert not s.is_done(out, "q1", ("gold", "margin"))


def test_write_json_is_atomic(tmp_path):
    s = _script()
    path = tmp_path / "x.json"
    s.write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_run_records_writes_measures_npz_and_ablation(tiny, tmp_path):
    s = _script()
    model, tok = tiny
    record = PromptRecord("qX", 0, "fake", PROMPT, "Answer: nope", "4 hours", "single-session-user", True,
                          LINES, (LINES[1],), True, False, False)
    timing = s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=True, k=5, model_id="tiny")
    assert set(p.name for p in tmp_path.glob("qX.*")) == {"qX.gold.json", "qX.gold.npz", "qX.margin.json", "qX.margin.npz",
                                                          "qX.own.json", "qX.own.npz", "qX.ablation.json"}
    gold = json.loads((tmp_path / "qX.gold.json").read_text())
    assert gold["measures"]["l2"]["shares"]["evidence"] >= 0 and gold["target"] == "gold" and gold["model_id"] == "tiny"
    assert "own_skipped" not in gold
    arrays = np.load(tmp_path / "qX.gold.npz")
    assert arrays["g_l2"].dtype == np.float32 and arrays["regions"].dtype == np.int8
    ablation = json.loads((tmp_path / "qX.ablation.json").read_text())
    assert ablation["delta"] == pytest.approx(ablation["gold_full"] - ablation["gold_ablated"])
    assert timing["qX"] > 0
    # a second call skips everything
    assert s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=True, k=5, model_id="tiny") == {}


def test_run_records_skips_own_on_empty_output_and_resumes(tiny, tmp_path):
    s = _script()
    model, tok = tiny
    record = PromptRecord("qE", 0, "fake", PROMPT, "   ", "4 hours", "single-session-user", True,
                          LINES, (LINES[1],), True, False, False)
    timing = s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=False, k=5, model_id="tiny")
    assert "qE" in timing
    assert set(p.name for p in tmp_path.glob("qE.*")) == {"qE.gold.json", "qE.gold.npz", "qE.margin.json", "qE.margin.npz"}
    assert json.loads((tmp_path / "qE.gold.json").read_text())["own_skipped"] is True
    # resume does not wait for the own file that will never exist
    assert s.run_records(model, tok, [record], tmp_path, targets=("gold", "margin", "own"), ablate=False, k=5, model_id="tiny") == {}


def test_regen_check_counts_matches(tiny, tmp_path):
    s = _script()
    model, tok = tiny
    record = PromptRecord("qX", 0, "fake", PROMPT, "whatever the cache said", "4 hours", "t", True, LINES, (), True, None, None)
    result = s.regen_check(model, tok, [record], n=1, max_new_tokens=4)
    assert result["n"] == 1 and result["matches"] in (0, 1) and len(result["pairs"]) == 1


def _fake_outputs(root, reader, rows, folds=(0,), target="gold"):
    """rows: list of (question_id, fold, evidence_share, delta)."""
    d = root / reader
    d.mkdir(parents=True, exist_ok=True)
    for qid, fold, share, delta in rows:
        m = {"l2": {"shares": {"instruction": 0.2, "evidence": share, "distractor": 0.7 - share, "question": 0.1},
                    "lift": share / 0.2, "topk": share},
             "gxi": {"shares": {"instruction": 0.2, "evidence": share / 2, "distractor": 0.7 - share / 2, "question": 0.1},
                     "lift": share / 0.4, "topk": share / 2},
             "counts": {"evidence": 20, "distractor": 60, "instruction": 15, "question": 5}, "target_value": -3.0, "n_tokens": 100}
        (d / f"{qid}.{target}.json").write_text(json.dumps({"question_id": qid, "fold": fold, "reader": reader, "target": target,
                                                            "model_id": "x", "measures": m}))
        (d / f"{qid}.ablation.json").write_text(json.dumps({"question_id": qid, "gold_full": 0.0, "gold_ablated": -delta, "delta": delta}))


def test_report_builds_claims_from_outputs(tmp_path, monkeypatch):
    from scripts import exp33_report as rep
    qids = [f"q{i}" for i in range(40)]
    small = [(q, 0, 0.1 + 0.002 * i, 0.5 + 0.01 * i) for i, q in enumerate(qids)]
    large = [(q, 0, 0.3 + 0.002 * i, 1.0 + 0.01 * i) for i, q in enumerate(qids)]
    _fake_outputs(tmp_path, "qwen3b", small); _fake_outputs(tmp_path, "qwen7b", large)

    calls = []

    def records(reader, fold, **kw):
        calls.append(reader)
        rs = [PromptRecord(q, 0, reader, "p", "o", "g", "single-session-user", True, ("l",), ("l",), True,
                           correct=(i % 2 == 0) if reader == "qwen3b" else True, unknown=False) for i, q in enumerate(qids)]
        return rs, {}
    monkeypatch.setattr(rep, "load_records", records)
    markdown, numbers = rep.build_report(("qwen3b", "qwen7b"), {0}, "l2", "k50", tmp_path, samples=300)
    assert numbers["claim1"]["gold"]["verdict"] == "SHOWN" and numbers["claim1"]["gold"]["n"] == 40
    assert 0 <= numbers["claim2"]["qwen3b"]["auc"] <= 1
    assert numbers["ablation"]["qwen3b"]["rho"] > 0.9
    assert "Claim 1" in markdown and "qwen7b" in markdown and "folds: 0" in markdown
    assert set(calls) == {"qwen3b", "qwen7b"}
    calls.clear()
    rep.build_report(("qwen3b", "qwen7b"), {0}, "l2", "k50", tmp_path, samples=300, records_from="qwen3b")
    assert calls and all(name == "qwen3b" for name in calls)
