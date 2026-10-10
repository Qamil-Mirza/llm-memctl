"""§29, the reader-utility signal: prompt shape, fold guard, losses, the logprob client against a fake vLLM 0.8.5
server, and that a trained head loads into the real controller."""

import gzip
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from memctl import reader_utility as ru
from memctl.agents.llm import LLMAgent, compact_line
from memctl.memory.items import MemoryItem, SourceType
from memctl.task import Observation, TaskState

LONGMEMEVAL_S = Path(ru.PATH)
needs_s = pytest.mark.skipif(not LONGMEMEVAL_S.exists(), reason="LongMemEval-S is not present")
ROOT = Path(__file__).resolve().parents[1]


def test_reader_prompt_is_the_frozen_reader_prompt_without_the_note():
    item = MemoryItem("s1:0", "I graduated in Business Administration.", 6, 3, SourceType.USER,
                      metadata={"speaker": "user", "date": "2023/05/20 (Sat) 10:00"})
    question = Observation("q0000", "Question: (asked on 2023/05/30) What degree did I graduate with?",
                           SourceType.USER, requires_response=True)
    task = TaskState(1, "Answer the question.", question)
    agent = LLMAgent({"reasoning": False, "labels": "compact", "model": {"backend": "stub"}})
    built = agent.build_prompt(SimpleNamespace(active=[item]), question, task)
    assert built == ru.reader_prompt(task.goal, [compact_line(item)], question.content)
    assert built.endswith("\n\nAnswer:")
    assert ru.reader_prompt("g", [], "Q").count("(empty)") == 1
    text = ru.chat_text(built, "Business Administration")
    assert text.startswith("<|im_start|>system\nYou are Qwen") and text.endswith("assistant\nBusiness Administration")


def test_question_loss_matches_listsum_and_soft_targets():
    logits = torch.tensor([2.0, 0.0, -1.0], requires_grad=True)
    row = {"candidates": [{"evidence": 1}, {"evidence": 0}, {"evidence": 1}], "utility": [0.0, 3.0, 0.0]}
    listsum = (torch.logsumexp(logits, 0) - logits[[0, 2]]).sum()
    assert torch.isclose(ru.question_loss(logits, row, "evidence", 0.0), listsum)
    target = torch.softmax(torch.tensor([0.0, 3.0, 0.0]) / ru.TAU, 0)
    ce = torch.logsumexp(logits, 0) - (target * logits).sum()
    assert torch.isclose(ru.question_loss(logits, row, "utility", 1.0), ce)
    assert torch.isclose(ru.question_loss(logits, row, "blend", 0.25), 0.75 * listsum + 0.25 * ce)
    none = {"candidates": [{"evidence": 0}] * 3, "utility": [0.0, 0.0, 0.0]}
    assert ru.question_loss(logits, none, "evidence", 0.0) is None  # no evidence in the pool: skipped, as §19
    assert ru.question_loss(logits, none, "blend", 0.5) is not None  # the utility term still applies
    loss = ru.question_loss(logits, row, "utility", 1.0)
    loss.backward()
    assert logits.grad[1] < 0  # the useful candidate is pushed up


@needs_s
def test_fold_guard_parts_are_disjoint_from_the_fold_test_part():
    from memctl.envs.longmemeval import _index
    from memctl.splits import fold_indices

    index = _index(ru.PATH)
    for k in range(5):
        test = set(fold_indices(index, 5, k, "test"))
        train = ru.allowed_questions(k, "train")
        inner_train, inner_val = ru.allowed_questions(k, "inner_train"), ru.allowed_questions(k, "inner_val")
        assert not test & set(train) and not test & set(inner_train) and not test & set(inner_val)
        assert sorted(inner_train + inner_val) == train and not set(inner_train) & set(inner_val)
        assert abs(len(inner_train) / len(train) - 0.75) < 0.01


def _rows_files(tmp_path, fold):
    """500-line rows and utility files whose lines for anything outside fold `fold`'s training part are POISON."""
    allowed = set(ru.allowed_questions(fold, "train"))
    feature = [0.0] * 24
    rows, utility = tmp_path / "rows.jsonl.gz", tmp_path / "utility.jsonl.gz"
    with gzip.open(rows, "wt") as r, gzip.open(utility, "wt") as u:
        for i in range(500):
            if i not in allowed:
                r.write("POISON\n")
                u.write("POISON\n")
                continue
            cands = [{"id": f"t{j}", "created_at": j, "line": f"user: turn {j}", "tokens": 3,
                      "evidence": int(j == i % 4), "features": [x + (j == i % 4) for x in feature]} for j in range(6)]
            r.write(json.dumps({"index": i, "question_id": f"q{i}", "unanswerable": False, "gold": "x",
                                "goal": "g", "question": "Q", "global_features": [0.0] * 6,
                                "candidates": cands}) + "\n")
            u.write(json.dumps({"index": i, "utility": [2.0 * (j == i % 4) for j in range(6)]}) + "\n")
    return rows, utility


@needs_s
def test_training_rows_never_read_test_rows(tmp_path):
    rows_path, utility_path = _rows_files(tmp_path, 2)
    rows = ru.training_rows(rows_path, utility_path, 2, "train")  # would fail to parse a POISON line
    assert len(rows) == 400 and all("utility" in r for r in rows)
    inner = ru.training_rows(rows_path, utility_path, 2, "inner_val")
    assert {r["index"] for r in inner} <= {r["index"] for r in rows}
    with pytest.raises(json.JSONDecodeError):
        ru.training_rows(rows_path, utility_path, 3, "train")  # fold 3's training part includes fold 2's test rows


@needs_s
def test_heads_learn_their_signal_and_load_in_the_controller(tmp_path):
    from memctl.controllers.rl import RLController

    rows_path, utility_path = _rows_files(tmp_path, 0)
    rows = ru.training_rows(rows_path, utility_path, 0, "train")[:64]
    for signal_name, w in (("utility", 1.0), ("blend", 0.5), ("evidence", 0.0)):
        policy, loss = ru.train_head(rows, signal_name, w, epochs=30)
        again, loss_again = ru.train_head(rows, signal_name, w, epochs=30)
        assert loss == loss_again  # deterministic
        hits = sum(rows[n]["candidates"][ru.top_k(policy, rows[n], 1)[0]]["evidence"] for n in range(len(rows)))
        assert hits == len(rows)
        path = tmp_path / f"{signal_name}.pt"
        ru.save_head(policy, path, {"signal": signal_name})
        controller = RLController({"checkpoint": str(path), "floor_head": True, "keep_none": True}, 0)
        assert sum(p.numel() for p in controller.policy.parameters()) == 27526


@pytest.fixture()
def fake_server():
    ports = []
    procs = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
    for port in ports:
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "configs/sweeps/exp29/fake_vllm.py"), str(port)],
                                      cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": ""}))
    for port in ports:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                time.sleep(0.1)
    yield [f"http://127.0.0.1:{p}/v1" for p in ports]
    for proc in procs:
        proc.terminate()
        proc.wait()


def test_scorer_reads_the_gold_span_and_resumes_against_another_url(tmp_path, fake_server):
    user = ru.reader_prompt("g", ["user, 2023/05/20: I graduated in Business Administration."], "Question: degree?")
    scorer = ru.Scorer(fake_server[0], str(tmp_path / "cache"))
    first = scorer.logprob(user, "Business Administration")
    assert first["check"] and first["n"] == 2
    assert first["logprob"] == pytest.approx(ru.stub_logprob(user, "Business Administration", 2)["logprob"])
    empty = scorer.logprob(ru.reader_prompt("g", [], "Question: degree?"), "Business Administration")
    assert first["logprob"] > empty["logprob"]  # the turn holding the answer has positive utility
    other = ru.Scorer(fake_server[1], str(tmp_path / "cache"))
    assert other.logprob(user, "Business Administration") == first and other.calls == 0 and other.hits == 2


def _cell(tmp_path, name, folds):
    import yaml

    cell = tmp_path / name
    cell.mkdir()
    env = {"name": "longmemeval", **({"folds": folds} if folds else {})}
    (cell / "config.yaml").write_text(yaml.safe_dump({"env": env}))
    (cell / "episodes.jsonl").write_text("".join(json.dumps({"seed": s, "evidence_complete_rate": 1.0}) + "\n"
                                                 for s in range(3)))
    (cell / "metadata.json").write_text(json.dumps({"resumed": True}))
    return cell


@needs_s
def test_stub_metrics_refuse_test_fold_cells(tmp_path):
    from memctl.envs.longmemeval import _index
    from memctl.splits import fold_indices

    test_cell = _cell(tmp_path, "test", {"k": 5, "fold": 1, "part": "test"})
    with pytest.raises(PermissionError):
        ru.stub_metrics([test_cell])
    with pytest.raises(PermissionError):  # no folds at all: the whole file, test questions included
        ru.cell_questions(_cell(tmp_path, "nofolds", None))
    train_cell = _cell(tmp_path, "train", {"k": 5, "fold": 1, "part": "train"})
    fold, ids = ru.cell_questions(train_cell)
    index = _index(ru.PATH)
    test_ids = {index[i]["question_id"] for i in fold_indices(index, 5, 1, "test")}
    assert fold == 1 and len(ids) == 3 and not test_ids & set(ids)
    assert ru.stub_metrics([train_cell])[str(train_cell)]["distinct"] == 3
    stage = (ROOT / "configs/sweeps/exp29/stage.sh").read_text()
    assert "s#part: test#part: train#" in stage  # the rehearsal's reader cells play the training parts
