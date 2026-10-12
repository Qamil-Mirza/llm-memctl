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
