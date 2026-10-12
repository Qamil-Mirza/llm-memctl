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
