"""§36 verifier: prompt assembly, score extraction, counterfactual substitution, cut and length-control invariants,
resume rules and report statistics (memctl/analysis/verification.py, scripts/exp36_verify.py)."""
import math

import numpy as np
import pytest

from memctl.agents.llm import REASONING_INSTRUCTIONS
from memctl.analysis import verification as ver
from memctl.analysis.jacobian import ablate_evidence
from memctl.jacobian_data import PromptRecord

GOAL = ("Below is your memory of a long conversation, followed by a question about it. Answer with a short phrase "
        "that states the fact asked for. If the memory does not contain the answer, reply: unknown")
LINES = ("user, 2023/05/01 (Mon) 10:15: I adopted a cat named Luna and she is 2 years old.",
         "assistant, 2023/05/01 (Mon) 10:16: Congratulations on the new cat!",
         "user, 2023/05/09 (Tue) 09:00: I went to Leeds for 3 days.")
QUESTION = "Question: (asked on 2023/06/01 (Thu) 12:00) What is my cat's name?"


def make_prompt(lines=LINES) -> str:
    return "\n".join([f"{GOAL}\n{REASONING_INSTRUCTIONS}", "", "## Memory", *lines, "", "## Current input", QUESTION, "",
                      "Note:"])


def record(qid="q1", gold="Luna", evidence=(LINES[0],), lines=LINES, output="Luna, a cat.\nAnswer: Luna", fold=0):
    return PromptRecord(qid, fold, "qwen3b", make_prompt(lines), output, gold, "single-session-user", True, lines,
                        tuple(evidence), True, True, False)


# --- wording ------------------------------------------------------------------------------------------------------

def test_reasoning_copy_matches_agent_and_gold_kind_rule_matches_script():
    from scripts.exp34_gold_kinds import NUMBER_WORDS, OPENERS, kind
    assert ver.REASONING_INSTRUCTIONS == REASONING_INSTRUCTIONS
    assert ver.OPENERS == OPENERS and ver.NUMBER_WORDS == NUMBER_WORDS
    for token in ("2", "$30", "Luna", "The", "three", "yes", "x", "Dr.", ""):
        assert ver.token_kind(token) == kind(token)


def test_pre_registered_v_instruction_is_verbatim():
    assert ver.INSTRUCTIONS["V"] == (
        "Does the memory above support the proposed answer? Reply with one letter.\n"
        "A = the memory states or directly implies this answer\n"
        "B = the memory is partly relevant but does not settle it\n"
        "C = the memory does not support it, or contradicts it")
    assert ver.ANSWER_LITERAL == "Answer: "
    assert len({ver.INSTRUCTIONS[k] for k in ("V", "P2", "P3")}) == 3
    assert len(ver.wording_sha256()) == 64


# --- prompt assembly ----------------------------------------------------------------------------------------------

def test_verifier_prompt_layout_is_exact():
    text = ver.verifier_prompt(make_prompt(), "Luna")
    expected = "\n".join([GOAL, "", "## Memory", *LINES, "", QUESTION, "Proposed answer: Luna", ver.INSTRUCTIONS["V"]])
    assert text == expected
    assert REASONING_INSTRUCTIONS not in text and "## Current input" not in text and "Note:" not in text


def test_verifier_prompt_with_note_and_paraphrase():
    note = "  Facts: Luna (cat).\nShe is 2.\n\n"
    text = ver.verifier_prompt(make_prompt(), "Luna", "P2", note)
    assert "\n\nA previous reader wrote this note: Facts: Luna (cat).\nShe is 2.\n" + QUESTION + "\n" in text
    assert text.endswith(ver.INSTRUCTIONS["P2"])


def test_reasoning_must_occur_exactly_once():
    with pytest.raises(ValueError, match="exactly 1"):
        ver.verifier_prompt(make_prompt().replace(REASONING_INSTRUCTIONS, "x"), "Luna")
    with pytest.raises(ValueError, match="exactly 1"):
        ver.strip_reasoning(make_prompt() + "\n" + REASONING_INSTRUCTIONS)


def test_input_hashes_and_c4_input():
    rec = record()
    a = ver.label_input("V", rec, rec.prompt, "Luna", "qwen3b/B")
    b = ver.label_input("CutB", rec, ver.cut_prompt(rec.prompt, rec.memory_lines, rec.evidence_lines), "Luna", "qwen3b/B")
    assert a.continuation == "Answer: " and a.intact_prompt_sha256 == b.intact_prompt_sha256 == ver.sha256(rec.prompt)
    assert a.input_sha256 != b.input_sha256 and a.kind == "label"
    c4 = ver.c4_input(rec, "Luna", "qwen3b/B")
    assert c4.prompt == rec.prompt and c4.continuation == "Luna, a cat.\nAnswer:" and c4.suffixes == (" Luna", " unknown")
    assert c4.kind == "c4"
    assert ver.c4_input(record(output="no marker"), "Luna", "x") is None


def test_candidates():
    assert ver.candidate_of_row({"marker": True, "refusal": False, "final_answer": "Luna"}) == "Luna"
    assert ver.candidate_of_row({"marker": True, "refusal": True, "final_answer": "unknown"}) is None
    assert ver.candidate_of_row({"marker": False, "refusal": False, "final_answer": "long note"}) is None
    assert ver.candidate_of_row(None) is None
    assert ver.candidate_of_output("note\nAnswer: Harvard") == "Harvard"
    assert ver.candidate_of_output("note\nAnswer: unknown") is None
    assert ver.candidate_of_output("a note cut off") == "a note cut off"


# --- score extraction ---------------------------------------------------------------------------------------------

def test_label_probs_sums_spellings_and_scores():
    top = {"A": math.log(0.5), " A": math.log(0.1), "B": math.log(0.2), " C": math.log(0.1), "The": math.log(0.05)}
    r = ver.label_probs(top)
    assert r["p_A"] == pytest.approx(0.6) and r["p_B"] == pytest.approx(0.2) and r["p_C"] == pytest.approx(0.1)
    assert r["label_mass"] == pytest.approx(0.9)
    assert r["s"] == pytest.approx((0.6 + 0.5 * 0.2) / 0.9) and r["argmax"] == "A"


def test_label_probs_without_labels_and_ties():
    r = ver.label_probs({"The": -0.1})
    assert r["label_mass"] == 0 and r["s"] is None and r["argmax"] is None
    tie = ver.label_probs({"B": math.log(0.4), "C": math.log(0.4)})
    assert tie["argmax"] == "B" and tie["s"] == pytest.approx(0.25)


def test_reliability_rule():
    assert ver.reliability([0.9] * 9 + [0.4])["reliable"]          # 0.1 of rows low: not more than 0.1
    assert not ver.reliability([0.9] * 8 + [0.4, 0.3])["reliable"]  # 0.2
    assert not ver.reliability([])["reliable"]


def test_confidence_margin():
    assert ver.confidence_margin([-1.0, -0.5], [-0.25]) == pytest.approx(-1.25)


# --- cut and length control ---------------------------------------------------------------------------------------

def test_length_control_replaces_exact_span_and_cuts_back_to_the_cut_prompt():
    rec = record(evidence=(LINES[0], LINES[2]))
    pool = ["user, 2022/01/01: " + "x" * 60, "user, 2022/01/02: " + "y" * 58, "user, 2022/01/03: Luna is here " + "z" * 40,
            "user, 2022/01/04: " + "w" * 30]
    lc, lines, reps = ver.length_control(rec.prompt, rec.memory_lines, rec.evidence_lines, pool, "q1", rec.gold)
    assert len(reps) == 2 and LINES[0] not in lc and LINES[2] not in lc and LINES[1] in lc
    assert all("luna" not in r.lower() for r in reps)          # a line holding the gold is not "irrelevant"
    assert lines == (reps[0], LINES[1], reps[1])
    # every character outside the evidence spans is unchanged
    assert lc.replace(reps[0], LINES[0]).replace(reps[1], LINES[2]) == rec.prompt
    # cutting the replacements gives exactly the Cut-B prompt
    assert ablate_evidence(lc, lines, reps) == ver.cut_prompt(rec.prompt, rec.memory_lines, rec.evidence_lines)
    # closest length wins; deterministic
    assert ver.length_control(rec.prompt, rec.memory_lines, rec.evidence_lines, list(reversed(pool)), "q1", rec.gold)[0] == lc
    best = min((p for p in pool if "luna" not in p.lower()), key=lambda p: abs(len(p) - len(LINES[2])))
    assert reps[1] == best or abs(len(reps[1]) - len(LINES[2])) == abs(len(best) - len(LINES[2]))


def test_length_control_never_uses_lines_already_in_the_prompt():
    rec = record()
    lc, _, reps = ver.length_control(rec.prompt, rec.memory_lines, rec.evidence_lines, [LINES[1], "user: other"], "q", "")
    assert reps == ("user: other",)


def test_cut_prompt_removes_evidence_and_refuses_no_op():
    rec = record()
    cut = ver.cut_prompt(rec.prompt, rec.memory_lines, rec.evidence_lines)
    assert LINES[0] not in cut and LINES[1] in cut and cut.count("\n") == rec.prompt.count("\n") - 1
    with pytest.raises(ValueError):
        ver.cut_prompt(rec.prompt, rec.memory_lines, ("not a memory line",))


# --- counterfactual -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gold,kind", [("2", "number"), ("Luna", "name"), ("Dr. Thompson", "name"),
                                       ("Trader Joe's", "name"), ("The cat", None), ("two", None), ("Yes", None),
                                       ("$300", None), ("Luna is my cat", None), ("3 days", None)])
def test_gold_kind(gold, kind):
    assert ver.gold_kind(gold) == kind


def test_counterfactual_number_substitution_by_exact_span():
    rec = record(gold="2")
    cf = ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, [], "q1")
    assert cf["original"] == "2" and cf["substitute"] == "4" and cf["kind"] == "number"
    s, e = cf["start"], cf["end"]
    assert rec.prompt[s:e] == "2" and cf["prompt"] == rec.prompt[:s] + "4" + rec.prompt[e:]
    assert "she is 4 years old" in cf["prompt"] and "2023/05/01" in cf["prompt"]  # the date label is untouched
    assert cf["evidence_lines"] == (LINES[0].replace("is 2 years", "is 4 years"),)
    assert cf["memory_lines"][1:] == LINES[1:]
    # the counterfactual prompt is still a valid §23 prompt for the verifier
    assert "she is 4 years old" in ver.verifier_prompt(cf["prompt"], "4")


def test_counterfactual_name_substitution_is_deterministic_and_absent_from_prompt():
    rec = record()
    pool = ["Luna", "Leeds", "Max", "Bella", "the dog", "42"]
    cf = ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, pool, "q1")
    assert cf["substitute"] in ("Max", "Bella")             # not the gold, not in the prompt (Leeds), a name
    assert cf == ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, list(reversed(pool)), "q1")
    assert cf["prompt"].count(cf["substitute"]) == 1 and "named " + cf["substitute"] in cf["prompt"]


def test_counterfactual_requires_exactly_one_occurrence_in_evidence_content():
    twice = ("user, 2023/05/01 (Mon) 10:15: Luna is my cat; Luna is 2.",) + LINES[1:]
    assert ver.counterfactual(make_prompt(twice), twice, (twice[0],), "Luna", ["Max"], "q") is None
    rec = record(gold="Leeds")                              # only in a non-evidence line
    assert ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, ["Max"], "q") is None
    rec = record(gold="10")                                 # "10" occurs only in the date label 10:15
    assert ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, [], "q") is None
    rec = record(gold="3", evidence=(LINES[2],))            # a number inside a longer number does not count
    lines = LINES[:2] + ("user, 2023/05/09 (Tue) 09:00: I paid 3,500 and stayed 3 days.",)
    rec = record(gold="3", evidence=(lines[2],), lines=lines)
    cf = ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, [], "q")
    assert cf is not None and "3,500 and stayed 5 days" in cf["prompt"]


def test_counterfactual_counts_copies_outside_the_evidence():
    lines = LINES[:2] + ("user, 2023/05/09 (Tue) 09:00: Luna came to Leeds too.",)
    rec = record(lines=lines)
    cf = ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, ["Max"], "q")
    assert cf["outside"] == 1 and cf["prompt"].count("Luna") == 1


# --- resume -------------------------------------------------------------------------------------------------------

def _row(x, engine="stub"):
    return ver.make_row(x, "qwen3b", engine, "m", "raw", 10, 0.1, {"top": {"A": -0.1}})


def test_pending_inputs_resume_and_refusals():
    rec = record()
    a = ver.label_input("V", rec, rec.prompt, "Luna", "s")
    b = ver.label_input("V", record(qid="q2"), make_prompt(), "Luna", "s")
    assert ver.pending_inputs([a, b], [_row(a)], "stub") == [b]
    with pytest.raises(ver.StaleRowError, match="engine"):
        ver.pending_inputs([a], [_row(a, "vllm-0.8.5")], "stub")
    other = ver.label_input("V", rec, rec.prompt, "Lunar", "s")
    with pytest.raises(ver.StaleRowError, match="different input"):
        ver.pending_inputs([a], [_row(other)], "stub")


def test_cf_rows_are_keyed_by_variant():
    rec = record()
    xs = [ver.label_input("CF", rec, rec.prompt, c, "gold", v) for c, v in (("Luna", "orig@intact"), ("Max", "sub@intact"))]
    assert ver.pending_inputs(xs, [_row(xs[0])], "stub") == [xs[1]]


def test_make_row_c4_and_label():
    rec = record()
    c4 = ver.c4_input(rec, "Luna", "s")
    row = ver.make_row(c4, "qwen3b", "stub", "m", "raw", 20, 0.1, {"suffix_logprobs": [[-1.0, -0.5], [-2.0]], "n_prefix_tokens": 9})
    assert row["m"] == pytest.approx(0.5) and row["sampling"]["prompt_logprobs"] == 1 and row["item"] == "C4"
    lab = _row(ver.label_input("V", rec, rec.prompt, "Luna", "s"))
    assert lab["sampling"]["logprobs"] == 20 and lab["sampling"]["max_tokens"] == 1 and lab["argmax"] == "A"
    assert lab["wording_sha256"] == ver.wording_sha256()


# --- statistics and decisions -------------------------------------------------------------------------------------

def test_auroc_stat_matches_pairwise_and_clusters():
    boot = ver.Bootstrap(["a", "b", "c", "d"], samples=200, seed=1)
    st = ver.auroc_stat([("a", 0.9), ("b", 0.5)], [("c", 0.5), ("d", 0.1)], boot)
    assert st["estimate"] == pytest.approx((1 + 1 + 0.5 + 1) / 4)
    assert st["ci_low"] <= st["estimate"] <= st["ci_high"]
    # a question can contribute to both classes (A'): its rows move together
    st2 = ver.auroc_stat([("a", 0.9)], [("a", 0.1), ("b", 0.2)], boot)
    assert st2["estimate"] == 1.0


def test_ratio_difference_and_mean():
    boot = ver.Bootstrap(["a", "b", "c"], samples=100)
    d = ver.ratio_difference([("a", 1, 1, 0, 1), ("b", 0, 1, 0, 1), ("c", 0, 0, 1, 1)], boot)
    assert d["estimate"] == pytest.approx(0.5 - 1 / 3)
    m = ver.mean_stat([("a", 1.0), ("b", 0.0)], boot)
    assert m["estimate"] == 0.5 and m["n"] == 2


def test_average_precision():
    assert ver.average_precision([0.9, 0.8], [0.1]) == pytest.approx(1.0)
    assert ver.average_precision([0.1], [0.9]) == pytest.approx(0.5)
    assert ver.average_precision([], [0.5]) is None


def test_top_k_ties_by_question_id_and_none_lowest():
    assert ver.top_k({"b": 0.5, "a": 0.5, "c": None, "d": 0.1}, 2) == {"a", "b"}
    assert ver.top_k({"b": 0.5, "a": 0.5, "c": None}, 1) == {"a"}
    assert "c" not in ver.top_k({"c": None, "d": 0.0}, 1)


def test_fit_tau_lowest_threshold_under_cap():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.2}
    wrong = {"a": False, "b": False, "c": True, "d": True}
    # tau 0.2: risk 2/4; tau 0.7: 1/3 > 0.3; tau 0.8: 0/2
    assert ver.fit_tau(scores, wrong, 0.30) == 0.8
    assert ver.fit_tau(scores, wrong, 0.50) == 0.2
    assert ver.fit_tau({"a": 0.5}, {"a": True}, 0.3) is None
    assert ver.emit_at(scores, 0.8) == {"a", "b"} and ver.emit_at(scores, None) == set()


def test_fixed_sequence():
    neg = {"ci_high": -0.01}
    pos = {"ci_high": 0.02}
    assert ver.fixed_sequence("NOT SHOWN", neg, neg) == ("not tested (Claim 1 not SHOWN)", "not tested (Claim 1 not SHOWN)")
    assert ver.fixed_sequence("SHOWN", pos, neg) == ("NOT SHOWN", "not tested (Claim 2 not SHOWN)")
    assert ver.fixed_sequence("SHOWN", neg, pos) == ("SHOWN", "NOT SHOWN")
    assert ver.fixed_sequence("SHOWN", neg, neg) == ("SHOWN", "SHOWN")


def test_bootstrap_weights_keep_question_rows_together():
    boot = ver.Bootstrap(["a", "b"], samples=50)
    w = boot.weights(["a", "a", "b"])
    assert np.array_equal(w[:, 0], w[:, 1]) and np.allclose(w.sum(1) - w[:, 1], 2)


# --- report gate --------------------------------------------------------------------------------------------------

def test_report_refuses_confirmation_folds_without_confirm(capsys):
    from scripts.exp36_report import main
    assert main(["--folds", "2,3,4"]) == 2
    assert "confirmation folds" in capsys.readouterr().out


def test_plan_parsing():
    from scripts.exp36_verify import ITEMS_BY_READER, parse_plan
    assert parse_plan("V,CutB|Vn,LC@600", ITEMS_BY_READER["qwen3b"]) == [(["V", "CutB"], 0), (["Vn", "LC"], 600)]
    with pytest.raises(ValueError):
        parse_plan("V_own", ITEMS_BY_READER["qwen3b"])


class _FakeD35:
    """The slice of exp35_report.ReaderData the §36 report reads."""

    def __init__(self, rows):  # rows: qid -> (candidate or None, B correct, cached refusal)
        self.frozen = sorted(rows)
        self._rows = rows
        self.rows = {"B": {q: {"marker": True, "refusal": c is None, "final_answer": c or "unknown"}
                           for q, (c, _, _) in rows.items()}}

    def correct(self, arm, q):
        return self._rows[q][1]

    def cached_refusal(self, q):
        return self._rows[q][2]


def test_report_claim1_rows_and_forced_abstentions():
    from scripts.exp36_report import Frame, discrimination, matched, policy
    d = _FakeD35({"a": ("x", True, False), "b": ("y", False, False), "c": ("z", True, True),
                  "n1": (None, False, True), "n2": (None, False, True)})
    f = Frame(d)
    assert f.c0 == {"a", "b"} and not f.has["n1"]
    boot = ver.Bootstrap(f.rows, samples=50)
    s = {"a": 0.2, "b": 0.9, "c": 0.1}                       # a score that is worse than chance on candidate rows
    st = discrimination(f, s, boot)
    assert (st["n_pos"], st["n_neg"]) == (2, 1) and st["estimate"] == 0.0
    assert st["all_fixed_rows"]["n_neg"] == 3 and st["all_fixed_rows"]["estimate"] > 0.5   # the inflation
    chosen = matched(f, s)
    assert chosen == {"b", "a"}                               # C0's coverage (2), ranked by s
    p = policy(f, chosen, boot)
    assert p["coverage"]["estimate"] == pytest.approx(2 / 5)  # no-candidate rows stay in the denominator
    assert p["risk"]["estimate"] == pytest.approx(0.5)


def test_control_b_primary_is_numbers_without_other_copies():
    from scripts.exp36_report import cf_subsets
    extras = {"a": {"kind": "number", "outside": 0}, "b": {"kind": "number", "outside": 2},
              "c": {"kind": "name", "outside": 0}, "d": {"kind": "name", "outside": 1}}
    sub = cf_subsets(extras)
    assert sub["primary"] == ["a"] and sub["number_with_other_copies"] == ["b"]
    assert sub["name"] == ["c", "d"] and sub["all_rows"] == ["a", "b", "c", "d"]
