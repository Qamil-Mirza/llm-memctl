import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.analysis.jlens_readout import ANSWER_PREFIX, Readout, readout_for, tracked_token_ids

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nassistant, d1: turn b\nhas newline\n"
          "user, d2: I was born in Leeds.\n\n## Current input\nQuestion: (asked on 2023/05/30) Where was I born?\n\nNote:")
LINES = ("user, d1: turn a", "assistant, d1: turn b\nhas newline", "user, d2: I was born in Leeds.")


@pytest.fixture(scope="module")
def tiny():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as error:
        pytest.skip(f"tokenizer unavailable: {error}")
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    return model, tok


def test_tracked_token_ids_skip_a_bare_space(tiny):
    _, tok = tiny
    ids, text = tracked_token_ids(tok, "4 hours")
    assert tok.decode([ids["gold"]]) == "4" and text == "4"
    ids2, text2 = tracked_token_ids(tok, "Leeds")
    assert tok.decode([ids2["gold"]]) == " Leeds" and text2 == " Leeds"
    assert tok.decode([ids["unknown_sp"]]) == " unknown" and tok.decode([ids["unknown"]]) == "unknown"


def test_readout_positions_land_on_the_right_tokens(tiny):
    _, tok = tiny
    r = readout_for(tok, PROMPT, LINES, (LINES[2],), gold="Leeds")
    assert isinstance(r, Readout) and r.text.endswith(ANSWER_PREFIX)
    ids = list(r.input_ids)
    assert ids == tok(r.text, add_special_tokens=False).input_ids
    (p1,) = r.positions["P1"]; (p0,) = r.positions["P0"]; (e,) = r.positions["E"]; (q,) = r.positions["Q"]
    assert p1 == len(ids) - 1 and p0 < p1
    assert tok.decode(ids[p0 + 1:p1 + 1]) == ANSWER_PREFIX          # P0 is the token right before "Answer:"
    assert tok.decode(ids[e]).strip().endswith(".") and "Leeds" in tok.decode(ids[e - 3:e + 1])
    assert tok.decode(ids[q]).strip().endswith("?")
    assert r.tracked["gold"] == tracked_token_ids(tok, "Leeds")[0]["gold"] and r.gold_token_text == " Leeds"


def test_readout_with_two_evidence_lines_and_no_note(tiny):
    _, tok = tiny
    prompt = PROMPT.replace("\n\nNote:", "")
    r = readout_for(tok, prompt, LINES, (LINES[0], LINES[2]), gold="Leeds")
    assert len(r.positions["E"]) == 2 and r.positions["E"][0] < r.positions["E"][1]
    (q,) = r.positions["Q"]
    assert tok.decode(list(r.input_ids)[q]).strip().endswith("?")


def test_readout_rejects_an_evidence_line_not_in_memory(tiny):
    _, tok = tiny
    with pytest.raises(ValueError):
        readout_for(tok, PROMPT, LINES, ("user, d9: not there",), gold="Leeds")


RUNS = Path("runs")
needs_exp23 = pytest.mark.skipif(not (RUNS / "exp23_qwen3b_f0").exists(), reason="§23 traces not present")


@needs_exp23
def test_readout_on_a_real_cached_prompt(tiny):
    from memctl.jacobian_data import load_records
    _, tok = tiny
    records, _ = load_records("qwen3b", 0)
    record = next(r for r in records if r.answerable and r.all_in_view and r.evidence_lines)
    r = readout_for(tok, record.prompt, record.memory_lines, record.evidence_lines, record.gold)
    assert len(r.input_ids) > 512                                   # longer than apply's default truncation
    assert len(r.positions["E"]) == len(record.evidence_lines)
    assert all(0 <= p < len(r.input_ids) for ps in r.positions.values() for p in ps)


from memctl.analysis.jlens_readout import (OUTCOMES, best_rank, emergence_depth, outcome, question_measures,
                                           rank_table, ranks_of, topk_table)


def test_ranks_of_counts_strictly_greater_plus_one():
    logits = np.array([[0.1, 0.9, 0.9, 0.5], [3.0, 2.0, 1.0, 0.0]])
    assert ranks_of(logits, [1, 3, 0]).tolist() == [[1, 3, 4], [2, 4, 1]]


def test_rank_table_takes_the_final_layer_from_model_logits():
    vocab = 6
    lens = {0: np.zeros((1, vocab)), 2: np.zeros((1, vocab))}
    lens[0][0, 4] = 5.0           # gold top at layer 0
    model = np.zeros((1, vocab)); model[0, 1] = 9.0; model[0, 4] = 1.0   # at the final layer, token 1 is top, gold second
    ranks, layers = rank_table(lens, model, n_layers=4, token_ids=[4, 1])
    assert layers == [0, 2, 3] and ranks.shape == (3, 1, 2)
    assert ranks[0, 0].tolist() == [1, 2] and ranks[2, 0].tolist() == [2, 1]
    ids, lg, layers2 = topk_table(lens, model, n_layers=4, k=2)
    assert layers2 == layers and ids.shape == (3, 1, 2) and ids[2, 0, 0] == 1 and lg.dtype == np.float32


def test_outcome_and_depth_rules():
    assert outcome(np.array([50, 5, 3, 40]), k=10) == "B"
    assert outcome(np.array([50, 5, 3, 7]), k=10) == "C"
    assert outcome(np.array([50, 30, 20, 40]), k=10) == "A"
    assert outcome(np.array([1, 1, 1, 1]), k=10) == "C"          # always-top token is C, never B (review focus 1)
    assert best_rank(np.array([50, 5, 3, 40])) == 3
    assert emergence_depth(np.array([50, 5, 3, 40]), layers=[0, 8, 16, 23], n_layers=24, k=10) == pytest.approx(8 / 24)
    assert emergence_depth(np.array([50, 30, 20, 40]), layers=[0, 8, 16, 23], n_layers=24, k=10) is None
    assert set(OUTCOMES) == {"A", "B", "C"}


def test_question_measures_shapes_every_position():
    layers = [0, 1, 2, 3]
    ranks = np.full((4, 4, 3), 99)             # 4 layers, 4 positions (P1, P0, E, Q), 3 tracked
    ranks[1:, 0, 0] = [4, 2, 50]               # gold at P1: emerges at layer 1, lost at the end -> B
    ranks[:, 0, 1] = [99, 99, 3, 1]            # " unknown" at P1 reaches top-1 at the last layer
    m = question_measures(ranks, layers, n_layers=4, tracked_names=["gold", "unknown_sp", "unknown"],
                          position_names={"P1": [0], "P0": [1], "E": [2], "Q": [3]}, k=10)
    assert m["k"] == 10 and m["P1"]["gold"] == {"best_rank": 2, "final_rank": 50, "emergence_depth": pytest.approx(0.25), "outcome": "B"}
    assert m["P1"]["unknown_sp"]["first_top1_depth"] == pytest.approx(0.75)
    assert m["P0"]["gold"]["outcome"] == "A" and len(m["E"]) == 1 and m["Q"]["gold"]["best_rank"] == 99


import importlib.util, time


def _script(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_prompts_is_deterministic_and_filters_short_paragraphs(tmp_path):
    s = _script("jlens_fit")
    import pyarrow as pa, pyarrow.parquet as pq
    rows = ["short one", " ".join(["word"] * 160), " ".join(["other"] * 200), " ".join(["third"] * 151)]
    pq.write_table(pa.table({"text": rows}), tmp_path / "c.parquet")
    paragraphs = s.load_corpus(tmp_path / "c.parquet", min_words=150)
    assert len(paragraphs) == 3
    a = s.select_prompts(paragraphs, 2, seed=0); b = s.select_prompts(paragraphs, 2, seed=0)
    assert a == b and len(a) == 2 and all(len(x.split()) >= 150 for x in a)


def test_fit_under_time_box_stops_early_and_resumes(tiny, tmp_path):
    jlens = pytest.importorskip("jlens")
    s = _script("jlens_fit")
    model, tok = tiny
    lens_model = jlens.from_hf(model, tok)
    prompts = [" ".join(f"w{i}" for i in range(200))] * 6
    ckpt = tmp_path / "ckpt.pt"
    lens, n = s.fit_under_time_box(lens_model, prompts, max_minutes=0.0, chunk=2, checkpoint_path=str(ckpt),
                                   source_layers=[1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    assert n == 2 and lens.n_prompts == 2                        # one chunk, then the box is over
    lens2, n2 = s.fit_under_time_box(lens_model, prompts, max_minutes=10.0, chunk=2, checkpoint_path=str(ckpt),
                                     source_layers=[1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    assert n2 == 6 and lens2.n_prompts == 6                      # resumed from the checkpoint, finished the list


def test_sidecar_marks_below_minimum(tmp_path):
    s = _script("jlens_fit")
    side = s.sidecar("qwen05b", "m", n_prompts=3, n_target=8, n_minimum=4, settings={"a": 1}, wall_s=1.0,
                     corpus_sha256="x", prompt_sha256s=["a", "b", "c"])
    assert side["below_minimum"] is True and side["n_prompts"] == 3 and side["jlens_commit"]


def _manifest(**over):
    m = {"model_id": "m", "seed": 0, "corpus_sha256": "c", "max_seq_len": 128, "dim_batch": 8, "source_layers": "all",
         "skip_first": 16, "prompt_sha256s": ["a", "b"]}
    m.update(over)
    return m


def _fake_ckpt(out, reader="r", next_idx=2):
    torch.save({"jacobian_sum": {}, "n_done": next_idx, "next_idx": next_idx, "source_layers": [0], "target_layer": 1,
                "skip_first": 16}, out / f"{reader}.ckpt.pt")


def test_check_manifest_refuses_a_checkpoint_from_another_seed(tmp_path):
    s = _script("jlens_fit")
    s.check_manifest(tmp_path, "r", _manifest())                 # fresh: writes the manifest
    _fake_ckpt(tmp_path)
    with pytest.raises(SystemExit) as e:
        s.check_manifest(tmp_path, "r", _manifest(seed=1))
    assert e.value.code == 6
    with pytest.raises(SystemExit) as e:
        s.check_manifest(tmp_path, "r", _manifest(prompt_sha256s=["a", "x", "y"]))
    assert e.value.code == 6
    with pytest.raises(SystemExit) as e:                         # checkpoint already past a shorter list
        s.check_manifest(tmp_path, "r", _manifest(prompt_sha256s=["a"]))
    assert e.value.code == 6


def test_check_manifest_accepts_a_longer_list_with_the_same_prefix(tmp_path):
    s = _script("jlens_fit")
    s.check_manifest(tmp_path, "r", _manifest())
    _fake_ckpt(tmp_path)
    s.check_manifest(tmp_path, "r", _manifest(prompt_sha256s=["a", "b", "c", "d"]))
    assert json.loads((tmp_path / "r.ckpt.json").read_text())["prompt_sha256s"] == ["a", "b", "c", "d"]


def test_check_manifest_refuses_a_checkpoint_without_a_manifest(tmp_path):
    s = _script("jlens_fit")
    _fake_ckpt(tmp_path)
    with pytest.raises(SystemExit) as e:
        s.check_manifest(tmp_path, "r", _manifest())
    assert e.value.code == 6 and not (tmp_path / "r.ckpt.json").exists()


def test_total_wall_adds_the_previous_sidecar(tmp_path):
    s = _script("jlens_fit")
    assert s.total_wall(tmp_path, "r", 5.0) == 5.0
    (tmp_path / "r.json").write_text(json.dumps({"wall_s": 100.0, "wall_s_total": 300.0}))
    assert s.total_wall(tmp_path, "r", 5.0) == 305.0
    (tmp_path / "r.json").write_text(json.dumps({"wall_s": 100.0}))   # sidecar from before wall_s_total existed
    assert s.total_wall(tmp_path, "r", 5.0) == 105.0


# --- Task 4: the read script ---------------------------------------------------------------------
from memctl.jacobian_data import PromptRecord  # noqa: E402


@pytest.fixture(scope="module")
def tiny_lens(tiny, tmp_path_factory):
    jlens = pytest.importorskip("jlens")
    model, tok = tiny
    lens_model = jlens.from_hf(model, tok)
    prompts = [" ".join(f"w{i}" for i in range(200))] * 2
    lens = jlens.fit(lens_model, prompts, source_layers=[0, 1, 2], dim_batch=8, max_seq_len=64, skip_first=4)
    path = tmp_path_factory.mktemp("lens") / "tiny.pt"
    lens.save(str(path))
    return lens, lens_model, tok, path


def test_read_record_produces_both_variants_from_one_forward(tiny_lens):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_read")
    record = PromptRecord("qX", 0, "fake", PROMPT, "Answer: Leeds", "Leeds", "single-session-user", True, LINES, (LINES[2],), True, True, False)
    payload, arrays = s.read_record(lens, lens_model, tok, record, k=5)
    L, P, T = arrays["ranks_jlens"].shape
    assert arrays["layers"].tolist() == [0, 1, 2, 3] and P == 4 and T == 3
    assert arrays["ranks_logit"].shape == (L, P, T)
    assert (arrays["ranks_jlens"][-1] == arrays["ranks_logit"][-1]).all()          # final layer identical (model logits)
    assert arrays["top_ids_jlens"].shape == (L, P, 5)
    assert payload["measures"]["jlens"]["P1"]["gold"]["outcome"] in ("A", "B", "C")
    assert payload["positions"]["P1"] == [payload["n_tokens"] - 1] and payload["gold_token_text"] == " Leeds"
    assert len(payload["prompt_sha256"]) == 64


def test_read_record_rejects_truncated_ids(tiny_lens, monkeypatch):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_read")
    record = PromptRecord("qX", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    monkeypatch.setattr(s, "MAX_SEQ_LEN", 8)                      # simulate apply-style truncation
    with pytest.raises(ValueError):
        s.read_record(lens, lens_model, tok, record, k=5)


def test_run_records_writes_resumes_and_isolates_failures(tiny_lens, tmp_path):
    lens, lens_model, tok, path = tiny_lens
    s = _script("jlens_read")
    good = PromptRecord("q1", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    bad = PromptRecord("q2", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, ("not a memory line",), True, True, False)
    timing = s.run_records(lens, lens_model, tok, [good, bad], tmp_path, k=5, model_id="tiny", lens_path=str(path))
    assert set(p.name for p in tmp_path.glob("q1.*")) == {"q1.npz", "q1.json"} and not list(tmp_path.glob("q2.*"))
    failed = json.loads((tmp_path / "_failed.json").read_text())
    assert "q2" in failed and "q1" in timing
    assert s.run_records(lens, lens_model, tok, [good], tmp_path, k=5, model_id="tiny", lens_path=str(path)) == {}
