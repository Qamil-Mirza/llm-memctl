"""§35 arm construction, D1, refusal rule, resume logic and report statistics (memctl/analysis/answer_selection.py,
scripts/exp35_generate.py, scripts/exp35_report.py, scripts/exp35_note_check.py)."""
import json
from pathlib import Path

import pytest

from memctl.agents.llm import DEFAULT_INSTRUCTIONS, REASONING_INSTRUCTIONS
from memctl.agents.llm import final_answer as agent_final_answer
from memctl.analysis import answer_selection as sel
from memctl.analysis.jacobian import ablate_evidence
from memctl.jacobian_data import PromptRecord

GOAL = ("Below is your memory of a long conversation, followed by a question about it. Answer with a short phrase "
        "that states the fact asked for. If the memory does not contain the answer, reply: unknown")
LINES = ("user, 2023/05/01: I adopted a cat named Luna.", "assistant, 2023/05/01: Congratulations!",
         "user, 2023/05/09: I went to Leeds.")


def make_prompt(lines=LINES, goal=GOAL) -> str:
    return "\n".join([f"{goal}\n{REASONING_INSTRUCTIONS}", "", "## Memory", *lines, "", "## Current input",
                      "Question: (asked on 2023/06/01) What is my cat's name?", "", "Note:"])


def record(qid="q1", fold=0, output="Luna is the cat.\nAnswer: Luna", gold="Luna", correct=True, unknown=False,
           reader="qwen3b", qtype="single-session-user", evidence=(LINES[0],), prompt=None) -> PromptRecord:
    return PromptRecord(qid, fold, reader, prompt or make_prompt(), output, gold, qtype, True, LINES, tuple(evidence),
                        True, correct, unknown)


# --- exact strings ------------------------------------------------------------------------------------------------

def test_pre_registered_strings_are_verbatim():
    assert sel.B_INSTRUCTION == "Based on the note above, give the answer as a short phrase. Do not reply unknown."
    assert sel.F_REPLACEMENTS == (
        ("If the memory does not contain the answer, reply: unknown",
         "Always answer with your best guess, even if the memory does not contain the answer; never reply unknown"),
        ("Reply unknown only if the memory has nothing at all on the topic.",
         "Never reply unknown; if the memory has little on the topic, give your best guess anyway."),
    )
    assert sel.ARMS == ("N", "A", "B", "F", "Fcut", "Ncut")
    # both originals are in the frozen prompt pieces the §23 readers saw
    assert sel.F_REPLACEMENTS[0][0] in DEFAULT_INSTRUCTIONS and sel.F_REPLACEMENTS[1][0] in REASONING_INSTRUCTIONS


def test_forced_prompt_replaces_each_clause_exactly_once():
    prompt = make_prompt()
    forced = sel.forced_prompt(prompt)
    for old, new in sel.F_REPLACEMENTS:
        assert old not in forced and forced.count(new) == 1
    assert "unknown" not in forced.replace("never reply unknown", "").replace("Never reply unknown", "")
    # everything outside the two clauses is untouched
    rebuilt = forced
    for old, new in sel.F_REPLACEMENTS:
        rebuilt = rebuilt.replace(new, old)
    assert rebuilt == prompt


@pytest.mark.parametrize("prompt", [make_prompt(goal="Answer briefly."), make_prompt() + "\n" + sel.F_REPLACEMENTS[1][0]])
def test_forced_prompt_refuses_zero_or_two_occurrences(prompt):
    with pytest.raises(ValueError, match="exactly 1"):
        sel.forced_prompt(prompt)


# --- arm inputs ---------------------------------------------------------------------------------------------------

def test_frozen_note_is_byte_exact_and_suffixes_are_pre_registered():
    note = "  Facts:\tLuna (cat), 1 May.\r\nanswer: maybe\n\n"   # odd whitespace and an earlier lowercase marker kept
    rec = record(output=note + "Answer: Luna")
    a = sel.build_arm_input(rec, "A")
    b = sel.build_arm_input(rec, "B")
    assert a.note == note and b.note == note
    assert a.continuation == note + "Answer:"
    assert b.continuation == note + "\n" + sel.B_INSTRUCTION + "\n" + "Answer:"
    assert a.prompt == b.prompt == rec.prompt
    assert a.intact_prompt_sha256 == sel.sha256(rec.prompt) and a.max_new_tokens == 64


def test_note_is_text_before_the_last_marker():
    rec = record(output="Answer: first try\nthen more\nAnswer: Luna")
    assert sel.build_arm_input(rec, "A").note == "Answer: first try\nthen more\n"


def test_marker_less_output_has_no_frozen_note_arm():
    rec = record(output="unknown")
    assert sel.build_arm_input(rec, "A") is None and sel.build_arm_input(rec, "B") is None
    assert sel.build_arm_input(rec, "N").prompt == rec.prompt


def test_cut_arms_use_the_ablated_prompt():
    rec = record()
    cut = ablate_evidence(rec.prompt, rec.memory_lines, rec.evidence_lines)
    fcut = sel.build_arm_input(rec, "Fcut")
    ncut = sel.build_arm_input(rec, "Ncut")
    assert LINES[0] not in fcut.prompt and LINES[0] not in ncut.prompt
    assert ncut.prompt == cut and fcut.prompt == sel.forced_prompt(cut)
    assert fcut.intact_prompt_sha256 == ncut.intact_prompt_sha256 == sel.sha256(rec.prompt)
    assert sel.build_arm_input(rec, "F").prompt == sel.forced_prompt(rec.prompt)
    assert fcut.prompt_sha256 == sel.sha256(fcut.prompt) and fcut.continuation == ""


def test_cut_arm_raises_for_evidence_not_in_memory():
    with pytest.raises(ValueError):
        sel.build_arm_input(record(evidence=("not a memory line",)), "Ncut")


def test_full_output_for_frozen_and_regenerated_arms():
    rec = record(output="Note text.\nAnswer: unknown")
    b = sel.build_arm_input(rec, "B")
    row = sel.make_row(b, " Luna\n", "stub", "m", "raw")
    assert row.final_answer == "Luna" and row.marker and not row.refusal
    n = sel.build_arm_input(rec, "N")
    assert sel.make_row(n, "  The note.\nAnswer: Luna  ", "stub", "m", "raw").final_answer == "Luna"
    assert row.sampling["temperature"] == 0.0 and row.sampling["max_new_tokens"] == 64


# --- D1, answers and refusals -------------------------------------------------------------------------------------

@pytest.mark.parametrize("note,gold,expected", [
    ("The cat is called Luna.", "Luna", True),
    ("She has 4 hours left", "4 hours", True),
    ("hours: 4", "4 hours", True),          # every gold token
    ("Nothing on it.", "Luna", False),
    (None, "Luna", None),
    ("anything", "...", None),
])
def test_d1(note, gold, expected):
    assert sel.d1(note, gold) is expected


def test_gold_in_note_uses_the_cached_note_only():
    assert sel.gold_in_note(record(output="No idea.\nAnswer: Luna")) is False
    assert sel.gold_in_note(record(output="Luna!\nAnswer: unknown")) is True


@pytest.mark.parametrize("output", ["note\nAnswer: Luna", "x\nanswer: unknown\nmore", "Answer:", "no marker here",
                                    "a\nAnswer: b\nAnswer: c d\ne"])
def test_final_answer_copy_matches_the_agent(output):
    assert sel.final_answer(output) == agent_final_answer(output)


@pytest.mark.parametrize("output,answer,refusal", [
    ("note\nAnswer: unknown", "unknown", True),
    ("note\nAnswer: Unknown, nothing on it", "Unknown, nothing on it", True),
    ("note\nAnswer:", "unknown", True),                                 # empty phrase graded as unknown (LLMAgent)
    ("note\nAnswer: I don't know", "I don't know", False),              # REFUSALS list says yes; the §23 rule says no
    ("unknown, the memory has nothing", "unknown, the memory has nothing", True),   # marker-less: whole output
    ("The answer is unknown", "The answer is unknown", False),
])
def test_refusal_is_the_section_23_rule(output, answer, refusal):
    got, _ = sel.agent_answer(output)
    assert got == answer and sel.is_refusal_answer(got) is refusal


def test_outcome():
    assert sel.outcome(True, "unknown") == "correct"
    assert sel.outcome(False, "unknown") == "refusal"
    assert sel.outcome(False, "Leeds") == "wrong"


def test_judge_prompt_matches_the_judge_class():
    from memctl.judge import Judge
    judge = Judge({"backend": "stub", "style": "longmemeval"})
    for qtype in ("temporal-reasoning", "single-session-preference", "multi-session"):
        assert sel.judge_prompt("Q?", "G", "A", qtype) == judge.prompt("Q?", "G", "A", qtype)


def test_judge_cache_key_matches_tracked_llm(tmp_path):
    from memctl.llm import build_llm
    llm = build_llm({"backend": "stub", "name": "qwen2.5-7b-instruct", "cache_dir": str(tmp_path)})
    llm.generate("PROMPT", max_new_tokens=sel.JUDGE_MAX_NEW_TOKENS)
    (path,) = list(tmp_path.glob("*/*.json"))
    assert path.stem == sel.judge_cache_key("PROMPT", sel.judge_settings("stub") | {"name": "qwen2.5-7b-instruct"})


def test_stub_chat_template_matches_the_qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no cached tokenizer: {error}")
    from memctl.analysis.jacobian import wrap_chat
    assert wrap_chat(tok, "P\nQ") == sel.QWEN25_CHAT.format(prompt="P\nQ")


# --- resume -------------------------------------------------------------------------------------------------------

def test_pending_inputs_skips_done_and_refuses_stale_rows():
    rec = record()
    items = [sel.build_arm_input(rec, "N"), sel.build_arm_input(record(qid="q2"), "N")]
    done = sel.make_row(items[0], "Answer: Luna", "stub", "m", "raw").to_dict()
    assert sel.pending_inputs(items, [done], "stub") == [items[1]]
    with pytest.raises(sel.StaleRowError, match="engine"):
        sel.pending_inputs(items, [done], "vllm-0.8.5")
    with pytest.raises(sel.StaleRowError, match="different input"):
        sel.pending_inputs(items, [{**done, "input_sha256": "x"}], "stub")
    with pytest.raises(sel.StaleRowError, match="different input"):
        sel.pending_inputs(items, [{**done, "wording_sha256": "old"}], "stub")


def _synthetic(n=4):
    return [record(qid=f"q{i}", output=("Luna.\nAnswer: Luna" if i % 2 else "unknown")) for i in range(n)]


def test_generate_reader_resumes_and_survives_a_torn_line(tmp_path):
    from scripts import exp35_generate as gen
    recs = _synthetic()
    by_q = {r.question_id: r for r in recs}
    qids = sorted(by_q)
    engine = gen.StubEngine("m")
    engine.tokenizer = None
    first = gen.generate_reader("qwen3b", [0], list(sel.ARMS), engine, tmp_path, by_q, qids, log=lambda *_: None)
    assert first["N"] == {"rows": 4, "new": 4, "no_marker": 0, "errors": 0}
    assert first["A"]["rows"] == 2 and first["A"]["no_marker"] == 2
    again = gen.generate_reader("qwen3b", [0], list(sel.ARMS), engine, tmp_path, by_q, qids, log=lambda *_: None)
    assert all(c["new"] == 0 for c in again.values())
    path = gen.arm_file(tmp_path, "qwen3b", "N", 0)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][:30])        # one good row lost, one torn
    third = gen.generate_reader("qwen3b", [0], ["N"], engine, tmp_path, by_q, qids, log=lambda *_: None)
    assert third["N"]["new"] == 2 and third["N"]["rows"] == 4
    rows = gen.read_jsonl(path)
    assert sorted(r["question_id"] for r in rows) == qids and all(r["engine"] == "stub" for r in rows)


def test_rows_carry_engine_sampling_and_hashes(tmp_path):
    from scripts import exp35_generate as gen
    recs = _synthetic(2)
    by_q = {r.question_id: r for r in recs}
    engine = gen.StubEngine("m")
    engine.tokenizer = None
    gen.generate_reader("qwen3b", [0], ["F", "B"], engine, tmp_path, by_q, sorted(by_q), log=lambda *_: None)
    (f_row, _) = gen.read_jsonl(gen.arm_file(tmp_path, "qwen3b", "F", 0))
    assert f_row["prompt_sha256"] == sel.sha256(sel.forced_prompt(recs[0].prompt))
    assert f_row["engine"] == "stub" and f_row["sampling"]["seed"] == 0 and f_row["n_teacher_forced"] is None
    (b_row,) = gen.read_jsonl(gen.arm_file(tmp_path, "qwen3b", "B", 0))
    item = sel.build_arm_input(recs[1], "B")
    assert b_row["raw_input_sha256"] == sel.sha256(sel.QWEN25_CHAT.format(prompt=item.prompt) + item.continuation)


# --- report -------------------------------------------------------------------------------------------------------

def _write_run(root: Path, n_s: int, a_reproduces: bool = True, engines=("stub", "stub"), fold: int = 0):
    """3B: n_s cached refusals whose note holds the gold (S), 4 with no gold in the note (S'), 4 cached successes.
    A reproduces the cached refusal; B answers S correctly and S' wrongly. The 7B is cached-correct everywhere."""
    from scripts import exp35_generate as gen
    recs3, recs7 = [], []
    for i in range(n_s + 8):
        qid = f"q{i:02d}"
        if i < n_s:
            out, correct, unknown = "Luna is the cat.\nAnswer: unknown", False, True
        elif i < n_s + 4:
            out, correct, unknown = "Nothing.\nAnswer: unknown", False, True
        else:
            out, correct, unknown = "Luna.\nAnswer: Luna", True, False
        recs3.append(record(qid=qid, fold=fold, output=out, correct=correct, unknown=unknown))
        recs7.append(record(qid=qid, fold=fold, output="Luna.\nAnswer: Luna", correct=True, unknown=False, reader="qwen7b"))
    for reader, recs, engine in (("qwen3b", recs3, engines[0]), ("qwen7b", recs7, engines[1])):
        judge = []
        for arm in sel.ARMS:
            rows = []
            for i, rec in enumerate(recs):
                item = sel.build_arm_input(rec, arm)
                in_s = reader == "qwen3b" and i < n_s
                if arm == "A":
                    text = " unknown" if (rec.unknown and a_reproduces) else " Luna"
                elif arm == "B":
                    text = " Luna" if (in_s or rec.correct) else " Paris"
                else:
                    text = "x\nAnswer: Luna"
                row = sel.make_row(item, text, engine, "m", "raw").to_dict()
                rows.append(row)
                judge.append({"question_id": rec.question_id, "fold": fold, "reader": reader, "item": arm,
                              "verdict": row["final_answer"] == "Luna", "source": "stub"})
            gen.write_jsonl(gen.arm_file(root, reader, arm, fold), rows)
        for rec in recs:
            judge.append({"question_id": rec.question_id, "fold": fold, "reader": reader, "item": "D2",
                          "verdict": "Luna" in rec.output.split("Answer:")[0], "source": "stub"})
            judge.append({"question_id": rec.question_id, "fold": fold, "reader": reader, "item": "cached",
                          "verdict": bool(rec.correct), "source": "stub"})
        gen.write_jsonl(gen.judge_file(root, reader, fold), judge)
    by_reader = {"qwen3b": {r.question_id: r for r in recs3}, "qwen7b": {r.question_id: r for r in recs7}}
    return by_reader, sorted(by_reader["qwen3b"])


@pytest.fixture
def report(monkeypatch):
    from scripts import exp35_report as rep
    monkeypatch.setattr(rep, "natural_control", lambda folds, runs: {})
    return rep


def _build(rep, monkeypatch, tmp_path, **kw):
    by_reader, qids = _write_run(tmp_path / "run", **kw)
    monkeypatch.setattr(rep, "load_population", lambda folds, runs: (by_reader, qids, []))
    return rep.build({0}, tmp_path / "run", samples=500, own_root=tmp_path / "own")


def test_report_claim1_shown_with_gate_and_matched_gap(report, monkeypatch, tmp_path):
    text, numbers = _build(report, monkeypatch, tmp_path, n_s=16)
    n = numbers["readers"]["qwen3b"]
    assert n["gate_g"]["p"] == 1.0 and n["gate_g"]["pass"] and n["gate_g_on_S"]["n"] == 16
    assert n["claim1"]["n_S"] == 16 and n["claim1"]["mean"] == 1.0 and n["claim1"]["verdict"] == "SHOWN"
    assert n["claim2"]["mean"] == 1.0 and n["claim2"]["verdict"] == "SHOWN"
    b_a = n["claim3"]["B_minus_A"]
    # B fixes 16 of 24 rows; the 7B's cached accuracy is 1, the 3B's 4/24: matched gap 20/24
    assert b_a["mean"] == pytest.approx(16 / 24) and b_a["matched_gap"] == pytest.approx(20 / 24)
    assert b_a["gap_share"] == pytest.approx(16 / 20)
    assert n["cost_i"]["S_prime"]["p"] == 1.0
    assert n["beside_D2"]["verdict"].startswith("descriptive")
    assert "D2 decides" not in text and "0.397" not in text and "SHOWN" in text
    assert numbers["readers"]["qwen7b"]["claim3"]["B_minus_A"]["gap_share"] is None


def test_report_underpowered_below_15(report, monkeypatch, tmp_path):
    _, numbers = _build(report, monkeypatch, tmp_path, n_s=10)
    assert numbers["readers"]["qwen3b"]["claim1"]["verdict"].startswith("UNDERPOWERED")


def test_confirm_report_adds_the_pooled_estimate_when_underpowered(report, monkeypatch, tmp_path):
    by_reader, qids = _write_run(tmp_path / "run", n_s=10, fold=2)
    monkeypatch.setattr(report, "load_population", lambda folds, runs: (by_reader, qids, []))
    text, numbers = report.build({2}, tmp_path / "run", samples=200, own_root=tmp_path / "own", pooled=True)
    n = numbers["readers"]["qwen3b"]
    assert n["claim1"]["verdict"].startswith("UNDERPOWERED") and n["pooled"]["n_S"] == 10
    assert "MIXES development and confirmation" in text
    _, dev = report.build({2}, tmp_path / "run", samples=200, own_root=tmp_path / "own", pooled=False)
    assert "pooled" not in dev["readers"]["qwen3b"]


def test_report_gate_g_failure_makes_claims_not_interpretable(report, monkeypatch, tmp_path):
    _, numbers = _build(report, monkeypatch, tmp_path, n_s=16, a_reproduces=False)
    n = numbers["readers"]["qwen3b"]
    assert not n["gate_g"]["pass"]
    assert n["claim1"]["verdict"].startswith("not interpretable") and n["claim2"]["verdict"].startswith("not interpretable")


def test_report_refuses_rows_from_an_older_wording(report, monkeypatch, tmp_path):
    from scripts import exp35_generate as gen
    by_reader, qids = _write_run(tmp_path / "run", n_s=16)
    path = gen.arm_file(tmp_path / "run", "qwen3b", "B", 0)
    rows = gen.read_jsonl(path)
    rows[0]["wording_sha256"] = "old"
    gen.write_jsonl(path, rows)
    monkeypatch.setattr(report, "load_population", lambda folds, runs: (by_reader, qids, []))
    with pytest.raises(SystemExit, match="wording"):
        report.build({0}, tmp_path / "run", samples=100, own_root=tmp_path / "own")


def test_report_refuses_mixed_engines(report, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="mixes engines"):
        _build(report, monkeypatch, tmp_path, n_s=16, engines=("vllm-0.8.5", "hf-transformers-5.17.0"))


def test_report_token_check_excludes_mismatches(report, monkeypatch, tmp_path):
    from scripts import exp35_generate as gen
    by_reader, qids = _write_run(tmp_path / "run", n_s=16)
    path = gen.arm_file(tmp_path / "run", "qwen3b", "A", 0)
    rows = gen.read_jsonl(path)
    for r in rows:
        r["n_input_tokens"] = 100
    gen.write_jsonl(path, rows)
    own = tmp_path / "own" / "qwen3b"
    own.mkdir(parents=True)
    (own / "q00.json").write_text(json.dumps({"n_tokens": 102, "n_leading_ws": 1, "answer_marker": True}))   # 101: mismatch
    (own / "q01.json").write_text(json.dumps({"n_tokens": 101, "n_leading_ws": 1, "answer_marker": True}))   # 100: match
    monkeypatch.setattr(report, "load_population", lambda folds, runs: (by_reader, qids, []))
    _, numbers = report.build({0}, tmp_path / "run", samples=200, own_root=tmp_path / "own")
    tc = numbers["readers"]["qwen3b"]["token_check"]
    assert tc["mismatch"] == {"q00": (100, 101)} and len(tc["missing_own_read"]) == 22
    assert numbers["readers"]["qwen3b"]["claim1"]["n_S"] == 15


def test_report_and_note_check_refuse_confirmation_folds_without_confirm(capsys):
    from scripts import exp35_note_check, exp35_report
    assert exp35_report.main(["--folds", "0,1,2"]) == 2
    assert exp35_note_check.main(["--folds", "3"]) == 2
    assert "confirmation folds" in capsys.readouterr().out


def test_wilson_interval():
    from scripts.exp35_report import wilson
    low, high = wilson(7, 10)
    assert low == pytest.approx(0.3968, abs=1e-3) and high == pytest.approx(0.8922, abs=1e-3)
    assert wilson(0, 0) == (None, None)


def test_judge_cache_reuses_section_23_only_when_asked(tmp_path):
    from scripts.exp35_generate import JudgeCache
    old = tmp_path / "judge_exp23"
    key = sel.judge_cache_key("P", sel.judge_settings("openai"))
    (old / key[:2]).mkdir(parents=True)
    (old / key[:2] / f"{key}.json").write_text(json.dumps({"output": "Yes"}))
    off = JudgeCache(tmp_path / "own", sel.judge_settings("vllm-0.8.5"), None, sel.judge_settings("openai"))
    on = JudgeCache(tmp_path / "own", sel.judge_settings("vllm-0.8.5"), old, sel.judge_settings("openai"))
    assert off.get("P", allow_reuse=True) == (None, None)
    assert on.get("P", allow_reuse=True) == ("Yes", "exp23_cache") and on.get("P", allow_reuse=False) == (None, None)
    off.put("P", "No")
    assert on.get("P", allow_reuse=True) == ("No", "exp35_cache")
