"""§37 elenchus: numbering, turn builders, the E1 parser, the normaliser, scores, planted thesis, calibration and the
development gate (memctl/analysis/elenchus.py, scripts/exp37_*.py)."""
import hashlib

import pytest

from memctl.agents.llm import REASONING_INSTRUCTIONS
from memctl.analysis import elenchus as el
from memctl.analysis import verification as ver
from memctl.analysis.jacobian import ablate_evidence
from memctl.jacobian_data import PromptRecord

GOAL = ("Below is your memory of a long conversation, followed by a question about it. Answer with a short phrase "
        "that states the fact asked for. If the memory does not contain the answer, reply: unknown")
LINES = ("user, 2023/05/01 (Mon) 10:15: I adopted a cat named Luna.\nShe is 2 years old and sleeps a lot.",
         "assistant, 2023/05/01 (Mon) 10:16: Congratulations! Tip [3] of 5: feed her twice a day.",
         "user, 2023/05/09 (Tue) 09:00: I went to Leeds for 3 days.")
QUESTION = "Question: (asked on 2023/06/01 (Thu) 12:00) What is my cat's name?"


def make_prompt(lines=LINES) -> str:
    return "\n".join([f"{GOAL}\n{REASONING_INSTRUCTIONS}", "", "## Memory", *lines, "", "## Current input", QUESTION, "",
                      "Note:"])


def record(lines=LINES, evidence=(LINES[0],), gold="Luna", qid="q1"):
    return PromptRecord(qid, 0, "qwen3b", make_prompt(lines), "note\nAnswer: Luna", gold, "single-session-user", True,
                        lines, tuple(evidence), True, True, False)


# --- texts ------------------------------------------------------------------------------------------------------

def test_texts_are_byte_exact():
    assert el.E1_TEXT == ("Which memory lines is this answer based on? Reply with the line numbers only, for example "
                          "[2] [5]. If no line bears on the question, reply NONE.")
    assert el.E2_TEXT == ("Using only the lines above, what is the answer? Answer with a short phrase. If the lines above "
                          "do not determine the answer, reply: not determined.")
    assert el.E3_TEXT == ("Do Answer 1 and Answer 2 give the same answer to the question? Reply with one letter. A = same, "
                          "B = partly the same or unclear, C = different")
    assert el.E1_FALLBACK == ("List the numbers of the memory lines that support your answer, separated by spaces, or "
                              "write NONE.")
    assert el.e3_text(True).startswith("Are Answer 1 and Answer 2 the same answer? Reply with one letter.")
    assert el.E2_DIALOGUE.format(G="[2] [5]") == ("Based only on lines [2] [5], what do those lines say the answer is? "
                                                  "Short phrase, or: not determined.")
    assert el.MAX_TOKENS["E1"] == 24 and el.MAX_TOKENS["E2"] == 32
    shas = {el.wording_sha256(a, b) for a in (False, True) for b in (False, True)}
    assert len(shas) == 4


# --- numbering ----------------------------------------------------------------------------------------------------

def test_one_number_per_item_none_inside_an_item():
    rec = record()
    numbered = el.number_prompt(rec.prompt, rec.memory_lines)
    positions = el.number_positions(rec.prompt, rec.memory_lines)
    assert [numbered[p:p + 4] for p in positions] == ["[1] ", "[2] ", "[3] "]
    for k, (p, line) in enumerate(zip(positions, LINES), start=1):
        assert numbered[p + len(f"[{k}] "):p + len(f"[{k}] ") + len(line)] == line
    # the multi-line item gets one number only; its second text line is not numbered
    assert "\nShe is 2 years old" in numbered and "[2] She" not in numbered
    # the "[3] " inside item 2's own text is not a number we inserted: still exactly 3 inserted tags
    assert numbered.count("[3] ") == 2 and len(positions) == 3
    block = el.numbered_block(rec.prompt, rec.memory_lines)
    assert block.startswith("## Memory\n[1] user, 2023/05/01") and GOAL not in block and REASONING_INSTRUCTIONS not in block
    assert "## Memory\n" in block and "[0]" not in block


def test_cut_numbering_equals_numbering_after_ablate():
    rec = record(evidence=(LINES[0],))
    cut, rem = el.cut_view(rec.prompt, rec.memory_lines, rec.evidence_lines)
    assert cut == ablate_evidence(rec.prompt, rec.memory_lines, rec.evidence_lines) and rem == LINES[1:]
    block = el.numbered_block(cut, rem)
    assert block == "## Memory\n[1] " + LINES[1] + "\n[2] " + LINES[2]
    assert LINES[0] not in block


def test_number_prompt_refuses_a_line_not_in_the_prompt():
    with pytest.raises(ValueError):
        el.number_prompt(make_prompt(), ("not there",))


# --- turns ---------------------------------------------------------------------------------------------------------

def test_e1_layout():
    rec = record()
    text = el.e1_messages(rec.prompt, rec.memory_lines, "Luna")[0]["content"]
    assert text == (el.numbered_block(rec.prompt, rec.memory_lines) + "\n\n" + QUESTION + "\nYou answered: Luna\n" + el.E1_TEXT)
    assert el.e1_messages(rec.prompt, rec.memory_lines, "Luna", True)[0]["content"].endswith(el.E1_FALLBACK)


def test_blind_e2_never_contains_the_candidate():
    rec = record()
    msgs = el.e2_blind_messages(rec.prompt, rec.memory_lines, [3, 1])
    text = msgs[0]["content"]
    assert len(msgs) == 1 and "Mittens" not in text and "You answered" not in text
    assert text == f"[1] {LINES[0]}\n[3] {LINES[2]}\n\n{QUESTION}\n{el.E2_TEXT}"
    with pytest.raises(ValueError):
        el.e2_blind_messages(rec.prompt, rec.memory_lines, [])


def test_dialogue_e2_carries_e1_and_the_candidate():
    rec = record()
    e1 = el.e1_messages(rec.prompt, rec.memory_lines, "Mittens")
    msgs = el.e2_dialogue_messages(e1, "[1]", [1])
    assert msgs[0] == e1[0] and msgs[1] == {"role": "assistant", "content": "[1]"} and "Mittens" in msgs[0]["content"]
    assert msgs[2]["content"] == el.E2_DIALOGUE.format(G="[1]")


def test_e3_order_from_sha256_parity():
    for q in ("q1", "gpt4_93159ced", "0a995998", "abc_abs"):
        flip = int(hashlib.sha256(q.encode()).hexdigest()[0], 16) % 2 == 1
        assert el.order_flip(q) == flip
        text = el.e3_messages(make_prompt(), q, "CAND", "DERIV")[0]["content"]
        first, second = ("DERIV", "CAND") if flip else ("CAND", "DERIV")
        assert text == f"{QUESTION}\nAnswer 1: {first}\nAnswer 2: {second}\n{el.E3_TEXT}"


def test_vnum_is_frame_a_on_the_numbered_prompt():
    rec = record()
    text = el.vnum_messages(rec.prompt, rec.memory_lines, "Luna")[0]["content"]
    assert text.startswith(GOAL + "\n\n## Memory\n[1] ") and text.endswith(ver.INSTRUCTIONS["P2"])


# --- E1 parser -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply,G,g,parses", [
    ("NONE", [], 0, True), ("none.", [], 0, True), ("[2] [5]", [2, 5], 1, True), ("2, 5", [2, 5], 1, True),
    ("[9] [12]", [], 0, False), ("[3] [9]", [3], 1, True), ("", [], 0, False), ("Lines 5 and 2, and 5 again", [2, 5], 1, True),
    ("I think the answer is Luna", [], 0, False)])
def test_parse_e1(reply, G, g, parses):
    p = el.parse_e1(reply, 8)
    assert (p["G"], p["g"], p["parses"]) == (G, g, parses)


def test_parse_e1_out_of_range_dropped():
    assert el.parse_e1("[0] [8] [9]", 8)["dropped"] == [0, 9]


# --- normaliser ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d,c,m", [
    ("Answer: Luna.", "Luna", 1), ("A: luna", "“Luna”", 1), ("'Luna'", "Luna", 1),
    ("$1,200", "1200", 1), ("1,200 dollars", "$1200", 1), ("three", "3", 1), ("a dozen", "12", 1),
    ("twenty", "20", 1), ("third", "3", 0), ("5 goals and 2 assists", "5", 0), ("4", "5", 0), ("two", "2", 1),
    ("2.5 hours", "2.50", 1), ("The Nike store", "Nike", 1), ("Leeds", "Leeds Bradford", 1), ("Paris", "London", 0),
    ("Denver", "Emily lives in Denver, Colorado.", 1), ("10:30", "10:30 am", 0), ("10:30", "10:30.", 1),
    ("7 May 2023", "7", 0), ("7 May 2023", "7 may 2023", 1), ("May 7", "8", 0), ("Room 12 B", "Room 12", 1),
    ("the", "the", 0), ("", "x", 0),
    ("Dr. Smith", "Dr Smith", 1), ("well—known", "well-known", 1), ("a dog", "dog", 1), ("Annie", "an", 0)])
def test_same_answer(d, c, m):
    assert el.same_answer(d, c) == m


def test_normaliser_version_is_v2():
    assert el.NORMALISER_VERSION == "v2" and el.wording()["normaliser"] == "v2"


def test_normalise_steps():
    assert el.normalise("Answer: “The $1,200 Fee.”") == "1200 fee"
    assert el.normalise("A dozen eggs") == "12 eggs"
    assert el.normalise("  Two-thirds ") == "2 thirds"


@pytest.mark.parametrize("text,nd", [("not determined", True), ("Not determined.", True), ("It cannot be determined", True),
                                     ("Undetermined", True), ("unknown", True), ("not specified in the lines", True),
                                     ("Not mentioned", True), ("Luna", False), ("determined", False)])
def test_not_determined(text, nd):
    assert el.is_not_determined(text) == nd


def test_real_answer():
    assert el.real_answer("Luna") and not el.real_answer("not determined") and not el.real_answer("Unknown cat")
    assert not el.real_answer("") and not el.real_answer(None) and not el.real_answer("the")


# --- scores ------------------------------------------------------------------------------------------------------

def test_elenchus_scores():
    assert el.elenchus_scores(0, None, None, "Luna") == {"g": 0, "e": 0.0, "m": 0, "e_mech": 0}
    assert el.elenchus_scores(1, "not determined", 0.9, "Luna")["e"] == 0.0
    s = el.elenchus_scores(1, "Luna", 0.8, "luna.")
    assert s == {"g": 1, "e": 0.8, "m": 1, "e_mech": 1}
    assert el.elenchus_scores(1, "Max", 0.1, "Luna")["e_mech"] == 0


def test_e3_score_uses_v2_scorer():
    import math
    top = {"ĠA": math.log(0.6), "C": math.log(0.2), "x": math.log(0.1)}
    s = el.e3_score(top)
    assert s["a"] == pytest.approx(0.75) and s["label_mass"] == pytest.approx(0.8) and s["argmax"] == "A"


def test_evidence_hit():
    assert el.evidence_hit([1, 3], LINES, (LINES[0],)) == 1
    assert el.evidence_hit([2, 3], LINES, (LINES[0],)) == 0 and el.evidence_hit([], LINES, (LINES[0],)) == 0


# --- planted thesis ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gold", ["2", "25", "Luna", "Dr. Thompson", "Trader Joe's"])
def test_planted_never_equals_the_gold(gold):
    pool = ["Luna", "Max", "Bella", "Dr. Thompson", "Trader Joe's", "Leeds"]
    value = el.planted(gold, pool, "q1", make_prompt())
    assert value is not None and el.normalise(value) != el.normalise(gold) and not el.same_answer(value, gold)
    assert el.planted("the cat", pool, "q1", make_prompt()) is None


# --- calibration and gate ------------------------------------------------------------------------------------------

def test_calibration_pairs_and_pass():
    rows = [(f"q{i:02d}", "p", ["Luna", "luna.", "Max", "7"][i % 4]) for i in range(25)]
    pairs = el.calibration_pairs(rows)
    assert len(pairs) == 40 and {p["kind"] for p in pairs} == {"identical", "different"}
    for p in pairs:
        if p["kind"] == "identical":
            assert p["x"] == p["y"] and p["expect"] == "A"
        else:
            assert el.normalise(p["x"]) != el.normalise(p["y"]) and p["expect"] == "C"
    good = [{"kind": p["kind"], "expect": p["expect"], "argmax": p["expect"]} for p in pairs]
    assert el.calibration_pass(good)["pass"]
    for r in good[:6:2]:   # 3 identical wrong -> 17/20
        r["argmax"] = "B"
    assert not el.calibration_pass(good)["pass"]


def test_dev_gate_arithmetic():
    assert el.dev_gate(0.66, 0.559)["pass"] and el.dev_gate(0.66, 0.559)["bar_vs_s"] == pytest.approx(0.609)
    assert not el.dev_gate(0.64, 0.50)["pass"]          # below 0.65
    assert not el.dev_gate(0.66, 0.62)["pass"]          # below s + 0.05
    assert not el.dev_gate(None, 0.559)["pass"]


# --- scripts ------------------------------------------------------------------------------------------------------

def test_dev_script_refuses_other_folds(capsys):
    from scripts.exp37_dev import main
    assert main(["--folds", "0,3", "--backend", "stub"]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_report_refuses_confirmation_folds(capsys):
    from scripts.exp37_report import main
    assert main(["--folds", "2,3,4"]) == 2


def test_plan_parsing_and_triage_order():
    from scripts.exp37_elenchus import DEFAULT_PLAN, parse_plan
    plan = parse_plan(DEFAULT_PLAN["qwen3b"], "qwen3b", {"R1": 900, "R2": 600, "R3": 300})
    assert plan == [(["V", "CutB", "CutF", "Pnum"], 0), (["Pname"], 300), (["Vnum"], 600), (["Vdlg"], 900)]
    assert "V_own" not in str(DEFAULT_PLAN["qwen7b"])
    with pytest.raises(ValueError):
        parse_plan("CutF", "qwen7b", {})
