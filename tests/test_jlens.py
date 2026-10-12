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
    assert ids[:r.n_chat] == tok(r.chat_text, add_special_tokens=False).input_ids     # the chat text, then "Answer:"
    assert ids[r.n_chat:] == tok(ANSWER_PREFIX, add_special_tokens=False).input_ids and r.n_leading_ws == 0
    (p1,) = r.positions["P1"]; (p0,) = r.positions["P0"]; (e,) = r.positions["E"]; (q,) = r.positions["Q"]
    assert p1 == len(ids) - 1 and p0 < p1 and r.positions["P1_raw"] == (p1,)          # content-first gold: P1 == P1_raw
    assert tok.decode(ids[p0 + 1:p1 + 1]) == ANSWER_PREFIX          # P0 is the token right before "Answer:"
    assert tok.decode(ids[e]).strip().endswith(".") and "Leeds" in tok.decode(ids[e - 3:e + 1])
    assert tok.decode(ids[q]).strip().endswith("?")
    assert r.tracked["gold"] == tracked_token_ids(tok, "Leeds")[0]["gold"] and r.gold_token_text == " Leeds"


def test_readout_teacher_forces_the_leading_space_of_a_numeric_gold(tiny):
    _, tok = tiny
    r = readout_for(tok, PROMPT, LINES, (LINES[2],), gold="4 hours")
    ids = list(r.input_ids)
    (p1,) = r.positions["P1"]; (p1_raw,) = r.positions["P1_raw"]
    assert tok.decode([ids[-1]]) == " " and r.n_leading_ws == 1        # the bare space is teacher-forced
    assert p1 == len(ids) - 1 and p1_raw == p1 - 1                     # P1 predicts the digit; P1_raw follows "Answer:"
    assert tok.decode(ids[p1_raw - 1:p1_raw + 1]).endswith(ANSWER_PREFIX)
    assert tok.decode([r.tracked["gold"]]) == "4" and r.gold_token_text == "4"
    leeds = readout_for(tok, PROMPT, LINES, (LINES[2],), gold="Leeds")
    assert r.positions["P0"] == leeds.positions["P0"] and r.positions["E"] == leeds.positions["E"]
    assert r.positions["Q"] == leeds.positions["Q"] and ids[:-1] == list(leeds.input_ids)


def test_position_names_end_with_p1_raw():
    from memctl.analysis.jlens_readout import POSITION_NAMES
    assert POSITION_NAMES == ("P1", "P0", "E", "Q", "P1_raw")


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
    ranks = np.full((4, 5, 3), 99)             # 4 layers, 5 positions (P1, P0, E, Q, P1_raw), 3 tracked
    ranks[1:, 0, 0] = [4, 2, 50]               # gold at P1: emerges at layer 1, lost at the end -> B
    ranks[:, 0, 1] = [99, 99, 3, 1]            # " unknown" at P1 reaches top-1 at the last layer
    ranks[:, 4, 0] = [7, 7, 7, 7]              # gold at P1_raw: present throughout -> C
    m = question_measures(ranks, layers, n_layers=4, tracked_names=["gold", "unknown_sp", "unknown"],
                          position_names={"P1": [0], "P0": [1], "E": [2], "Q": [3], "P1_raw": [4]}, k=10)
    assert m["P1_raw"]["gold"]["outcome"] == "C" and m["P1_raw"]["gold"]["final_rank"] == 7
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
    assert arrays["layers"].tolist() == [0, 1, 2, 3] and P == 5 and T == 3
    assert arrays["ranks_logit"].shape == (L, P, T)
    assert (arrays["ranks_jlens"][-1] == arrays["ranks_logit"][-1]).all()          # final layer identical (model logits)
    assert arrays["top_ids_jlens"].shape == (L, P, 5)
    assert payload["measures"]["jlens"]["P1"]["gold"]["outcome"] in ("A", "B", "C")
    assert payload["positions"]["P1"] == [payload["n_tokens"] - 1] and payload["gold_token_text"] == " Leeds"
    assert len(payload["prompt_sha256"]) == 64
    assert payload["positions"]["P1_raw"] == payload["positions"]["P1"] and payload["n_leading_ws"] == 0


def test_read_record_forwards_the_teacher_forced_ids(tiny_lens, monkeypatch):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_read")
    record = PromptRecord("qN", 0, "fake", PROMPT, "o", "4 hours", "t", True, LINES, (LINES[2],), True, True, False)
    seen = []
    original = lens_model.forward
    monkeypatch.setattr(lens_model, "forward", lambda ids: seen.append(ids[0].tolist()) or original(ids))
    payload, arrays = s.read_record(lens, lens_model, tok, record, k=5)
    expected = readout_for(tok, PROMPT, LINES, (LINES[2],), "4 hours")
    assert seen == [list(expected.input_ids)] and tok.decode([seen[0][-1]]) == " "
    assert payload["n_tokens"] == len(expected.input_ids) and payload["n_leading_ws"] == 1
    assert payload["positions"]["P1"] == [payload["n_tokens"] - 1] and payload["positions"]["P1_raw"] == [payload["n_tokens"] - 2]
    assert arrays["position_index"].tolist()[-1] == payload["n_tokens"] - 2


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
    timing = s.run_records(lens, lens_model, tok, [good, bad], tmp_path, k=5, model_id="tiny", lens_path=str(path),
                           lens_sha256="h1")
    assert set(p.name for p in tmp_path.glob("q1.*")) == {"q1.npz", "q1.json"} and not list(tmp_path.glob("q2.*"))
    failed = json.loads((tmp_path / "_failed.json").read_text())
    assert "q2" in failed and "q1" in timing
    assert json.loads((tmp_path / "q1.json").read_text())["lens_sha256"] == "h1"
    assert s.is_done(tmp_path, "q1", "h1") and not s.is_done(tmp_path, "q1", "h2")
    assert s.run_records(lens, lens_model, tok, [good], tmp_path, k=5, model_id="tiny", lens_path=str(path),
                         lens_sha256="h1") == {}


def test_run_records_prints_outcomes_only_for_development_folds(tiny_lens, tmp_path, capsys):
    lens, lens_model, tok, path = tiny_lens
    s = _script("jlens_read")
    dev = PromptRecord("qd", 1, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    conf = PromptRecord("qc", 2, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    s.run_records(lens, lens_model, tok, [dev, conf], tmp_path, k=5, model_id="tiny", lens_path=str(path), lens_sha256="h")
    lines = capsys.readouterr().out.splitlines()
    dev_line = next(l for l in lines if " qd " in l); conf_line = next(l for l in lines if " qc " in l)
    assert "final_rank" in dev_line and "jlens=" in dev_line
    assert "n_tokens=" in conf_line and not any(w in conf_line for w in ("final_rank", "jlens=", "logit=", "P1", "rank"))


def test_run_records_redoes_a_question_read_with_another_lens(tiny_lens, tmp_path):
    lens, lens_model, tok, path = tiny_lens
    s = _script("jlens_read")
    good = PromptRecord("q1", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    (tmp_path / "q1.json").write_text(json.dumps({"question_id": "q1", "lens_sha256": "old"}))
    (tmp_path / "q3.json").write_text(json.dumps({"question_id": "q3"}))          # no hash: an older output
    current = s.lens_sha256(path)
    assert len(current) == 64 and current != "old"
    assert not s.is_done(tmp_path, "q1", current) and not s.is_done(tmp_path, "q3", current)
    timing = s.run_records(lens, lens_model, tok, [good], tmp_path, k=5, model_id="tiny", lens_path=str(path),
                           lens_sha256=current)
    assert "q1" in timing                                                          # redone with the current lens
    assert json.loads((tmp_path / "q1.json").read_text())["lens_sha256"] == current
    assert s.is_done(tmp_path, "q1", current)


def test_main_hashes_the_lens_before_deciding_nothing_is_left(tmp_path, monkeypatch, capsys):
    s = _script("jlens_read")
    lens = tmp_path / "l.pt"
    lens.write_bytes(b"new lens")
    rec = PromptRecord("q1", 0, "fake", PROMPT, "o", "Leeds", "t", True, LINES, (LINES[2],), True, True, False)
    monkeypatch.setattr(s, "load_records", lambda reader, fold: ([rec], 0))
    out = tmp_path / "out"
    out.mkdir()
    (out / "q1.json").write_text(json.dumps({"question_id": "q1", "lens_sha256": s.lens_sha256(lens)}))
    assert s.main(["--reader", "qwen05b", "--lens", str(lens), "--folds", "0", "--out", str(out)]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_eval_readout_position_before_target_and_poetry_newline(tiny):
    _, tok = tiny
    s = _script("jlens_eval")
    item = {"prompt": "The capital of France is ", "target": "Paris", "intermediates": ["France"]}
    text, pos = s.readout_position(tok, item, "multihop")
    ids0 = tok(text, add_special_tokens=False).input_ids
    assert text == "The capital of France is" and pos == len(ids0) - 1 and tok.decode([ids0[pos]]) == " is"
    natural = tok(text + " " + item["target"].lstrip(), add_special_tokens=False).input_ids
    assert natural[:len(ids0)] == ids0                              # the space joins the target, not the prompt
    text2, pos2 = s.readout_position(tok, {"prompt": "Roses are red, violets are blue\nSugar is sweet", "intermediates": ["blue"]}, "poetry")
    ids = tok(text2, add_special_tokens=False).input_ids
    assert "\n" in tok.decode([ids[pos2]])
    text3, _ = s.readout_position(tok, {"prompt": [{"role": "user", "content": "hi"}], "intermediates": ["x"]}, "association")
    assert "hi" in text3 and "<|im_start|>" in text3
    text4, _ = s.readout_position(tok, {"prompt": [{"role": "user", "content": "hi  "}], "intermediates": ["x"]}, "association")
    assert text4 == text3                                           # last turn's trailing space stripped


def test_pass_at_k_uses_only_the_band():
    s = _script("jlens_eval")
    ranks = np.array([[1, 50], [20, 50], [30, 3], [1, 1]])      # 4 layers, 2 intermediates
    band = np.array([False, True, True, False])
    assert s.pass_at_k(ranks, band, k=10) == pytest.approx(0.5)  # only the second intermediate hits inside the band
    assert s.pass_at_k(ranks, np.ones(4, bool), k=10) == 1.0


def test_score_set_all_layers_metric_matches_an_all_true_mask(tiny_lens, monkeypatch):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_eval")
    item = {"name": "a", "prompt": "The capital of France is ", "target": "Paris", "intermediates": ["France", "Paris"]}
    text, pos = s.readout_position(tok, item, "multihop")
    from memctl.analysis.jlens_readout import rank_table
    ids = [tracked_token_ids(tok, w)[0]["gold"] for w in item["intermediates"]]
    expected = {}
    for v, lg in zip(("jlens", "logit"), s.logits_at(lens, lens_model, text, pos)[:2]):
        ranks, layers = rank_table({l: x[None] for l, x in lg.items()}, s.logits_at(lens, lens_model, text, pos)[2][None], 4, ids)
        expected[v] = s.pass_at_k(ranks[:, 0, :], np.ones(len(layers), bool), k=50)
    out = s.score_set(lens, lens_model, tok, [item], "multihop", band=(1, 2), k=50)
    assert out["pass_at_k_all"] == pytest.approx(expected)
    assert out["pass_at_k_all"]["jlens"] >= out["pass_at_k"]["jlens"]


def test_score_set_scores_both_variants_and_skips_long_items(tiny_lens, monkeypatch):
    lens, lens_model, tok, _ = tiny_lens
    s = _script("jlens_eval")
    items = [{"name": "a", "prompt": "The capital of France is", "target": " Paris", "intermediates": ["France", "Paris"]},
             {"name": "b", "prompt": " ".join(["word"] * 40), "intermediates": ["word"]}]
    out = s.score_set(lens, lens_model, tok, items, "multihop", band=(0, 4), k=10)
    assert out["n_items"] == 2 and out["n_skipped"] == 0
    assert set(out["pass_at_k"]) == set(out["pass_at_k_all"]) == {"jlens", "logit"}
    assert all(0.0 <= v <= 1.0 for v in [*out["pass_at_k"].values(), *out["pass_at_k_all"].values()])
    monkeypatch.setattr(s, "MAX_SEQ_LEN", 20)
    out2 = s.score_set(lens, lens_model, tok, items, "multihop", band=(0, 4), k=10)
    assert out2["n_items"] == 1 and out2["n_skipped"] == 1


# --- Task 6: the report --------------------------------------------------------------------------
from memctl.analysis.jacobian_stats import bootstrap_proportion  # noqa: E402

N_LAYERS = 24


def _measure(best, outcome, depth):
    gold = {"best_rank": best, "final_rank": best if outcome == "C" else best * 20, "emergence_depth": depth, "outcome": outcome}
    unk = {"best_rank": 5, "final_rank": 5, "first_top1_depth": None}
    pos = {"gold": gold, "unknown_sp": dict(unk), "unknown": dict(unk)}
    return {"k": 10, "layers": list(range(N_LAYERS)), "P1": pos, "P0": pos, "E": [pos], "Q": pos}


def _jlens_outputs(root, reader, rows, sha="h0"):
    """rows: (question_id, fold, J-lens gold best rank at P1, outcome). The logit lens is ten times worse."""
    d = root / reader
    d.mkdir(parents=True, exist_ok=True)
    for qid, fold, best, outcome in rows:
        depth = None if outcome == "A" else 0.5
        entry = {"question_id": qid, "fold": fold, "reader": reader, "model_id": "x", "lens_path": "l.pt", "lens_n_prompts": 128,
                 "n_tokens": 100, "positions": {"P1": [99], "P0": [97], "E": [40], "Q": [90]},
                 "tracked": {"gold": 1, "unknown_sp": 2, "unknown": 3}, "gold_token_text": "x",
                 "measures": {"jlens": _measure(best, outcome, depth), "logit": _measure(best * 10, outcome, depth)}}
        if sha is not None:
            entry["prompt_sha256"] = sha
        ranks = np.full((N_LAYERS, 4, 3), 1000, np.int32)
        ranks_logit = ranks.copy()
        ranks[:, 0, 0], ranks_logit[:, 0, 0] = best, best * 10
        np.savez(d / f"{qid}.npz", ranks_jlens=ranks, ranks_logit=ranks_logit, layers=np.arange(N_LAYERS, dtype=np.int32),
                 position_index=np.array([99, 97, 40, 90], np.int32))
        (d / f"{qid}.json").write_text(json.dumps(entry))


def _jlens_side_files(root, reader, jlens_pass=0.3, logit_pass=0.1):
    (root / "eval").mkdir(exist_ok=True)
    (root / "lenses").mkdir(exist_ok=True)
    sets = {name: {"n_items": 20, "n_skipped": 0, "pass_at_k": {"jlens": jlens_pass, "logit": logit_pass},
                   "pass_at_k_all": {"jlens": jlens_pass, "logit": logit_pass}} for name in ("multihop", "association")}
    (root / "eval" / f"{reader}.json").write_text(json.dumps({
        "reader": reader, "k": 10, "band": [8, 16], "sets": sets, "mean_pass_at_k": {"jlens": jlens_pass, "logit": logit_pass},
        "mean_pass_at_k_all": {"jlens": jlens_pass, "logit": logit_pass}}))
    (root / "lenses" / f"{reader}.json").write_text(json.dumps({"reader": reader, "n_prompts": 128, "n_target": 128,
                                                                 "n_minimum": 32, "below_minimum": False}))


def _jlens_records(qids):
    def records(reader, fold, **kw):
        rs = [PromptRecord(q, 0, reader, "p", "o", "g", "multi-session" if i % 2 else "single-session-user", True, ("l",),
                           ("l",), True,
                           correct=(i % 2 == 0) if reader == "qwen3b" else (i % 5 != 0),
                           unknown=(i % 4 == 1) if reader == "qwen3b" else (i % 10 == 5)) for i, q in enumerate(qids)]
        return rs, {"no_cached_prompt": 0, "evidence_not_found": 2}
    return records


def test_report_builds_gate_claims_and_decomposition(tmp_path, monkeypatch):
    rep = _script("exp34_report")
    qids = [f"q{i}" for i in range(40)]
    small = [(q, 0, 50 + i + (200 if i % 2 else 0), "A" if i % 3 else "B") for i, q in enumerate(qids)]
    large = [(q, 0, 2 + i % 7 + (5 if i % 5 == 0 else 0), "C" if i % 4 else "A") for i, q in enumerate(qids)]
    _jlens_outputs(tmp_path, "qwen3b", small); _jlens_outputs(tmp_path, "qwen7b", large)
    for reader in ("qwen3b", "qwen7b"):
        _jlens_side_files(tmp_path, reader)
    monkeypatch.setattr(rep, "load_records", _jlens_records(qids))
    markdown, numbers = rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=300)
    for reader in ("qwen3b", "qwen7b"):
        gate = numbers["gate"][reader]
        assert gate["verdict"] == "PASS" and gate["a"]["mean"] == pytest.approx(1.0) and gate["b"]["verdict"] == "PASS"
        assert gate["band"] == [8, 16] and gate["n_prompts"] == 128 and gate["below_minimum"] is False
    assert numbers["claim1"]["best_logrank"]["verdict"] == "SHOWN" and numbers["claim1"]["best_logrank"]["n"] == 40
    assert numbers["claim1"]["outcome_A"]["mean"] > 0
    assert numbers["claim2"]["qwen3b"]["auc"] > 0.5 and numbers["claim2"]["qwen3b"]["verdict"] == "SHOWN"
    for reader in ("qwen3b", "qwen7b"):
        for cls, shares in numbers["decomposition"][reader].items():
            assert sum(shares[o]["p"] for o in "ABC") == pytest.approx(1.0), (reader, cls)
    assert numbers["decomposition"]["qwen3b"]["refusal"]["A"]["n"] == 10
    assert numbers["descriptive"]["question_type"]["qwen3b"]["multi-session"]["n"] == 20
    assert numbers["dropped"]["qwen7b"] == {"0": {"no_cached_prompt": 0, "evidence_not_found": 2}}
    for heading in ("Lens gate", "Claim 1", "Claim 2", "Decomposition", "Descriptive", "Dropped"):
        assert heading in markdown
    assert "folds: 0" in markdown and "community" not in numbers["descriptive"]
    json.dumps(numbers)  # the numbers file must serialise


def test_report_gate_fails_when_the_logit_lens_wins(tmp_path, monkeypatch):
    rep = _script("exp34_report")
    qids = [f"q{i}" for i in range(10)]
    _jlens_outputs(tmp_path, "qwen3b", [(q, 0, 5, "C") for q in qids]); _jlens_outputs(tmp_path, "qwen7b", [(q, 0, 5, "C") for q in qids])
    _jlens_side_files(tmp_path, "qwen3b"); _jlens_side_files(tmp_path, "qwen7b", jlens_pass=0.0, logit_pass=0.2)
    monkeypatch.setattr(rep, "load_records", _jlens_records(qids))
    _, numbers = rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=100)
    assert numbers["gate"]["qwen3b"]["verdict"] == "PASS" and numbers["gate"]["qwen7b"]["verdict"] == "FAIL"
    assert numbers["gate"]["qwen7b"]["b"]["verdict"] == "FAIL"


def test_band_best_logrank_reads_only_the_band(tmp_path):
    rep = _script("exp34_report")
    ranks = np.full((24, 2, 3), 100, np.int32)
    ranks[2, 0, 0] = 1           # outside the band: ignored
    ranks[10, 0, 0] = 10
    np.savez(tmp_path / "q.npz", ranks_jlens=ranks, ranks_logit=ranks, layers=np.arange(24, dtype=np.int32))
    assert rep.band_best_logrank(tmp_path / "q.npz", "jlens", 0, 0, (8, 16)) == pytest.approx(1.0)
    assert rep.layer_band(np.arange(24)) == (8, 16) and rep.layer_band(np.arange(36)) == (12, 24)


def test_bootstrap_proportion_interval_contains_p():
    flags = [True] * 30 + [False] * 70
    r = bootstrap_proportion(flags, samples=500, seed=0)
    assert r["n"] == 100 and r["p"] == pytest.approx(0.3) and r["ci_low"] <= 0.3 <= r["ci_high"] and r["ci_low"] < r["ci_high"]
    assert bootstrap_proportion([], samples=10) == {"n": 0, "p": None, "ci_low": None, "ci_high": None}


def test_report_refuses_confirmation_folds_without_confirm(tmp_path, monkeypatch):
    rep = _script("exp34_report")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    called = []
    monkeypatch.setattr(rep, "build_report", lambda *a, **k: called.append(a) or ("x", {}))
    assert rep.main(["--folds", "0,1,2", "--root", str(tmp_path)]) == 2
    assert not called and list(work.rglob("*")) == []
    assert rep.main(["--folds", "0,1,2", "--root", str(tmp_path), "--confirm"]) == 0
    written = sorted(p.name for p in work.rglob("*") if p.is_file())
    assert written == ["exp34_report_confirm.json", "exp34_report_confirm.md"]
    assert called[0][1] == {0, 1, 2}


def test_report_exit_4_on_prompt_mismatch(tmp_path, monkeypatch):
    rep = _script("exp34_report")
    qids = [f"q{i}" for i in range(6)]
    _jlens_outputs(tmp_path, "qwen3b", [(q, 0, 5, "C") for q in qids])
    _jlens_outputs(tmp_path, "qwen7b", [(q, 0, 5, "C") for q in qids[:5]])
    _jlens_outputs(tmp_path, "qwen7b", [("q5", 0, 5, "C")], sha="other")
    for reader in ("qwen3b", "qwen7b"):
        _jlens_side_files(tmp_path, reader)
    monkeypatch.setattr(rep, "load_records", _jlens_records(qids))
    with pytest.raises(rep.PromptIdentityError, match="q5"):
        rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=50)
    _jlens_outputs(tmp_path, "qwen7b", [("q5", 0, 5, "C")], sha=None)
    with pytest.raises(rep.PromptIdentityError, match="q5.*missing"):
        rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=50)
    monkeypatch.chdir(tmp_path)
    assert rep.main(["--folds", "0", "--root", str(tmp_path), "--samples", "50", "--out", str(tmp_path / "r.md")]) == 4
    assert not (tmp_path / "r.md").exists() and not (tmp_path / "r.json").exists()


def test_report_uses_only_questions_present_for_both_readers(tmp_path, monkeypatch):
    rep = _script("exp34_report")
    qids = [f"q{i}" for i in range(40)]
    _jlens_outputs(tmp_path, "qwen3b", [(q, 0, 50 + i, "A" if i % 3 else "B") for i, q in enumerate(qids)])
    _jlens_outputs(tmp_path, "qwen7b", [(q, 0, 3, "C") for q in qids if q not in ("q3", "q8")])  # 3b has two extra
    for reader in ("qwen3b", "qwen7b"):
        _jlens_side_files(tmp_path, reader)
    base = _jlens_records(qids)

    def records(reader, fold, **kw):
        rs, d = base(reader, fold)
        return [r if r.question_id != "q6" else PromptRecord(*[*r.__dict__.values()][:11], None, None) for r in rs], d
    monkeypatch.setattr(rep, "load_records", records)
    markdown, numbers = rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=100)
    assert numbers["paired"] == {"n": 38, "missing": {"qwen3b": 0, "qwen7b": 2}}
    assert numbers["dropped_unlabelled"] == {"qwen3b": 1, "qwen7b": 1}
    c2 = numbers["claim2"]
    assert c2["qwen3b"]["n_pos"] + c2["qwen3b"]["n_neg"] == c2["qwen7b"]["n_pos"] + c2["qwen7b"]["n_neg"] == 37
    d = numbers["decomposition"]
    for reader in ("qwen3b", "qwen7b"):   # each reader's own labels: the classes cover the 37 paired, labelled questions
        assert sum(d[reader][c]["A"]["n"] for c in ("refusal", "answered wrong", "success") if c in d[reader]) == 37
    _, same = rep.build_report(("qwen3b", "qwen7b"), {0}, "k10", tmp_path, samples=100, records_from="qwen3b")
    d = same["decomposition"]           # the same labels for both readers: identical class sizes
    assert set(d["qwen3b"]) == set(d["qwen7b"])
    for cls in d["qwen3b"]:
        assert d["qwen3b"][cls]["A"]["n"] == d["qwen7b"][cls]["A"]["n"], cls
    assert numbers["gate"]["qwen3b"]["a"]["n"] == numbers["gate"]["qwen7b"]["a"]["n"] == 38
    assert numbers["descriptive"]["emergence"]["qwen3b"]["n_read"] == 38
    assert "label" in markdown


def test_top_jaccard_matches_layers_by_value(tmp_path):
    rep = _script("exp34_report")
    layers = np.array([4, 6, 8, 10, 12], np.int32)          # a lens fitted on a subset of layers
    ours = np.zeros((5, 1, 10), np.int32); theirs = np.zeros((5, 1, 10), np.int32)
    for i in range(5):
        ours[i, 0] = np.arange(10)
        theirs[i, 0] = np.arange(10) if layers[i] < 8 else np.arange(100, 110)
    np.savez(tmp_path / "a.npz", top_ids_jlens=ours, layers=layers)
    np.savez(tmp_path / "b.npz", top_ids_jlens=theirs, layers=layers)
    assert rep.top_jaccard(tmp_path / "a.npz", tmp_path / "b.npz", (0, 8)) == pytest.approx(1.0)   # layers 4, 6
    assert rep.top_jaccard(tmp_path / "a.npz", tmp_path / "b.npz", (8, 16)) == pytest.approx(0.0)  # layers 8, 10, 12
    assert rep.top_jaccard(tmp_path / "a.npz", tmp_path / "b.npz", (5, 7)) == pytest.approx(1.0)   # layer 6 only
    assert rep.top_jaccard(tmp_path / "a.npz", tmp_path / "b.npz", (0, 4)) is None


# --- §34 pod runbook (python stubbed on a temp PATH, as in the §33 pod-script tests) ---
import shutil, subprocess  # noqa: E402

_POD_STUB = """#!/bin/sh
echo "fake python $*" >> "$POD_CALLS"
echo "fake python $*"
if [ "$1" = "scripts/jlens_fit.py" ]; then
  reader=""; prev=""
  for a in "$@"; do [ "$prev" = "--reader" ] && reader="$a"; prev="$a"; done
  mkdir -p runs/exp34_jlens/lenses
  if [ "$POD_BELOW" = MISSING ]; then
    printf '{\\n "reader": "%s",\\n "n_prompts": 3\\n}\\n' "$reader" > "runs/exp34_jlens/lenses/$reader.json"
  else
    printf '{\\n "reader": "%s",\\n "n_prompts": 3,\\n "n_target": 128,\\n "n_minimum": 32,\\n "below_minimum": %s\\n}\\n' \\
      "$reader" "$POD_BELOW" > "runs/exp34_jlens/lenses/$reader.json"
  fi
fi
"""


def _pod34_tree(tmp_path, below="false"):
    (tmp_path / "scripts").mkdir()
    for name in ("exp34_pod.sh", "exp33_pod.sh"):
        shutil.copy(f"scripts/{name}", tmp_path / "scripts" / name)
    (tmp_path / "data" / "longmemeval").mkdir(parents=True)
    (tmp_path / "data" / "longmemeval" / "longmemeval_s_cleaned.json").write_text("[]")
    (tmp_path / "data" / "jlens_corpus").mkdir()
    (tmp_path / "data" / "jlens_corpus" / "wikitext-103-raw-v1-test.parquet").write_text("")
    (tmp_path / "external" / "jacobian-lens").mkdir(parents=True)
    (tmp_path / "external" / "jacobian-lens" / "pyproject.toml").write_text("")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "exp23_verdicts.json").write_text("[]")
    for reader in ("qwen3b", "qwen7b"):
        for fold in range(5):
            run = tmp_path / "runs" / f"exp23_{reader}_f{fold}" / "fixed8__fraction0.05"
            run.mkdir(parents=True)
            for name in ("items.jsonl.gz", "steps.jsonl.gz", "episodes.jsonl"):
                (run / name).write_text("")
        (tmp_path / "cache" / f"generations_exp23_{reader}" / "00").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(_POD_STUB)
    (bin_dir / "python").chmod(0o755)
    calls = tmp_path / "calls.log"
    return tmp_path, {"PATH": f"{bin_dir}:/usr/bin:/bin", "POD_CALLS": str(calls), "POD_BELOW": below}, calls


def _run_pod34(root, env, *args):
    return subprocess.run(["bash", str(root / "scripts" / "exp34_pod.sh"), *args], env=env, capture_output=True, text=True)


def _calls(path):
    return path.read_text().splitlines() if path.exists() else []


@pytest.mark.parametrize("gone", ["data/jlens_corpus/wikitext-103-raw-v1-test.parquet",
                                  "external/jacobian-lens/pyproject.toml",
                                  "runs/exp23_qwen7b_f3/fixed8__fraction0.05/steps.jsonl.gz"])
def test_pod34_preflight_exits_3_naming_the_missing_path_before_any_python(tmp_path, gone):
    root, env, calls = _pod34_tree(tmp_path)
    (root / gone).unlink()
    done = _run_pod34(root, env)
    assert done.returncode == 3, done.stdout + done.stderr
    assert gone in done.stderr
    assert _calls(calls) == []                                    # nothing installed, downloaded or run


def test_pod34_preflight_counts_an_empty_cache_folder_as_missing(tmp_path):
    root, env, calls = _pod34_tree(tmp_path)
    shutil.rmtree(root / "cache" / "generations_exp23_qwen7b" / "00")
    done = _run_pod34(root, env)
    assert done.returncode == 3 and "cache/generations_exp23_qwen7b" in done.stderr
    assert _calls(calls) == []


@pytest.mark.parametrize("below", ["true", "MISSING"])
def test_pod34_stops_with_exit_5_when_a_lens_is_below_the_minimum(tmp_path, below):
    root, env, calls = _pod34_tree(tmp_path, below=below)     # MISSING: the stub writes a sidecar without the key
    done = _run_pod34(root, env)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "below_minimum" in done.stdout + done.stderr
    made = _calls(calls)
    assert any("scripts/jlens_fit.py --reader qwen3b" in c for c in made)
    assert not any("jlens_read.py" in c or "jlens_eval.py" in c for c in made)   # never read with a thin lens
    assert not any("jlens_fit.py --reader qwen7b" in c for c in made)


def test_pod34_runs_every_stage_in_order_with_the_brief_settings(tmp_path):
    root, env, calls = _pod34_tree(tmp_path)
    done = _run_pod34(root, env, "90", "40", "128")
    assert done.returncode == 0, done.stdout + done.stderr
    made = [c.removeprefix("fake python ") for c in _calls(calls)]
    fit3 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_fit.py --reader qwen3b"))
    fit7 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_fit.py --reader qwen7b"))
    read3 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_read.py --reader qwen3b"))
    read7 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_read.py --reader qwen7b")
                 and "community" not in c)
    eval3 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_eval.py --reader qwen3b"))
    eval7 = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_eval.py --reader qwen7b"))
    hub = next(i for i, c in enumerate(made) if "andyx10/jacobian-lens-qwen2.5-7b-instruct" in c)
    comm = next(i for i, c in enumerate(made) if c.startswith("scripts/jlens_read.py") and "community_qwen7b.pt" in c)
    j33 = next(i for i, c in enumerate(made) if c.startswith("scripts/jacobian_lens.py --reader qwen3b"))
    assert fit3 < fit7 < read3 < read7 < eval3 < eval7 < hub < comm < j33
    assert "--n-prompts 128 --min-prompts 32 --max-minutes 40 --dim-batch 32 --device cuda" in made[fit3]
    assert "--n-prompts 128 --min-prompts 32 --max-minutes 90 --dim-batch 32 --device cuda" in made[fit7]
    assert "--lens runs/exp34_jlens/lenses/qwen3b.pt --folds 0,1,2,3,4 --device cuda" in made[read3]
    assert "--lens runs/exp34_jlens/lenses/qwen7b.pt" in made[eval7]
    assert "--folds 0,1 " in made[comm] + " " and "--out runs/exp34_jlens/community_qwen7b/qwen7b" in made[comm]
    assert "shutil.copyfile" in made[hub] and "runs/exp34_jlens/lenses/community_qwen7b.pt" in made[hub]
    assert "preflight ok" in done.stdout and "elapsed" in done.stdout


def test_pod34_pins_the_local_transformers_and_installs_the_vendored_jlens(tmp_path):
    root, env, calls = _pod34_tree(tmp_path)
    done = _run_pod34(root, env)
    assert done.returncode == 0, done.stdout + done.stderr
    install = next(c for c in _calls(calls) if "pip install" in c)
    assert f"transformers=={transformers.__version__}" in install    # the pod pin is the tested local version
    assert "-e external/jacobian-lens" in install
    assert Path("scripts/exp34_pod.sh").stat().st_mode & 0o111      # executable
