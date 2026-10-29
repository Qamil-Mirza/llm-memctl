"""§44 rev 1 §A swap gate: verbatim matching, surface types, placement exclusions, substitute rules (no model)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp44_swap_gate as G  # noqa: E402
from memctl.jacobian_data import PromptRecord  # noqa: E402

HEAD = "Use only the memory.\n\n## Memory\n"
TAIL = "\n\n## Current input\nQuestion: (asked on 2023/05/30) {q}\n\nNote:"


def rec(qid, lines, evidence, gold, q="Where was I born?", fold=0):
    prompt = HEAD + "\n".join(lines) + TAIL.format(q=q)
    return PromptRecord(qid, fold, "x", prompt, "", gold, "single-session-user", True, tuple(lines), tuple(evidence),
                        True, False, True)


class Tok:
    """Whitespace-piece tokenizer: ' word' is one token, digits split off a leading space (like Qwen)."""

    def __init__(self):
        self.vocab = {}

    def _pieces(self, text):
        out = []
        for m in re.finditer(r" ?[^ ]+", text):
            p = m.group(0)
            if p.startswith(" ") and p[1:2].isdigit():
                out += [" ", p[1:]]
            else:
                out.append(p)
        return out

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(p, len(self.vocab)) for p in self._pieces(text)]
        return type("E", (), {"input_ids": ids})()

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_verbatim_whole_tokens_and_normalisation():
    pat = G.gold_pattern("$25,000")
    assert pat.search("it cost $25,000 total") and pat.search("25 000")
    assert not pat.search("125,000") and not G.gold_pattern("5").search("25 apples")
    assert G.gold_pattern("Trader Joe's").search("went to TRADER JOE'S today")
    for text, gold in [("born in Leeds.", "leeds"), ("5 days", "5"), ("25", "5"), ("a-b c", "a b")]:
        assert bool(G.gold_pattern(gold).search(text)) == G.normalised_contains(text, gold)
    assert G.gold_pattern("!!") is None
    assert G.core("$140") == "140" and G.core("Mindful.org.") == "Mindful.org"


def test_surface_types():
    assert G.surface_type("June 3rd") == "date" and G.surface_type("6:00 pm") == "date"
    assert G.surface_type("$25,000") == "number" and G.surface_type("three months") == "number"
    assert G.surface_type("Paris") == "name" and G.surface_type("The Glass Menagerie") == "name"
    assert G.surface_type("Yes") == "other" and G.surface_type("in my bedroom") == "other"


def test_placement_exclusions():
    chk = G.Checker()
    lines = ["user, d1: hello", "user, d2: I was born in Leeds.", "user, d3: bye"]
    assert G.place_gold(rec("a", lines, [lines[1]], "Leeds"), chk)[0] is None
    assert G.place_gold(rec("a", lines, [lines[0]], "Leeds"), chk)[0] == "gold_zero_evidence"
    two = lines[:2] + ["user, d3: Leeds again"]
    assert G.place_gold(rec("a", two, two[1:], "Leeds"), chk)[0] == "gold_several_evidence"
    assert G.place_gold(rec("a", lines, [lines[1]], "Leeds", q="Was I born in Leeds?"), chk)[0] == "gold_in_question"
    other = lines[:2] + ["user, d3: Leeds is far"]
    assert G.place_gold(rec("a", other, [other[1]], "Leeds"), chk)[0] == "gold_elsewhere"
    assert chk.disagreements == 0


def test_swap_is_byte_identical_elsewhere():
    lines = ["user, d1: hello", "user, d2: I paid $25,000 for it.", "user, d3: bye"]
    r = rec("a", lines, [lines[1]], "$25,000", q="How much did I pay?")
    _, place = G.place_gold(r, G.Checker())
    prompt, old, new = G.swap_prompt(r, place, "$140")
    assert new == "user, d2: I paid $140 for it." and prompt == r.prompt.replace(old, new)


def test_substitute_rules_and_order():
    tok = Tok()
    lines = ["user, d1: hello Sarah", "user, d2: I was born in Leeds.", "user, d3: bye"]
    r = rec("a", lines, [lines[1]], "Leeds", fold=0)
    pool_recs = [r, rec("b", lines, [lines[1]], "Sarah", fold=1),          # occurs in the prompt: skipped
                 rec("c", lines, [lines[1]], "Leeds", fold=1),            # same normalised gold: skipped
                 rec("d", lines, [lines[1]], "three months", fold=2),     # other surface type: skipped
                 rec("e", lines, [lines[1]], "Paris", fold=3)]
    tinfo = {x.question_id: G.token_info(tok, x.gold) for x in pool_recs}
    types = {x.question_id: G.surface_type(x.gold) for x in pool_recs}
    _, place = G.place_gold(r, G.Checker())
    cand, why, prompt, _ = G.choose_substitute(r, place, pool_recs, tinfo, types, G.Checker())
    assert why is None and cand.question_id == "e" and "born in Paris." in prompt
    # a substitute with the same first token is refused: first_token_equal when it is the only one left
    same = [r, rec("f", lines, [lines[1]], "Leeds United", fold=1)]
    tinfo = {x.question_id: G.token_info(tok, x.gold) for x in same}
    types = {x.question_id: G.surface_type(x.gold) for x in same}
    assert G.choose_substitute(r, place, same, tinfo, types, G.Checker())[1] in ("first_token_equal", "no_substitute")
    # pool order is deterministic (seed 44) and independent of input order
    assert [x.question_id for x in G.pool_order(pool_recs)] == [x.question_id for x in G.pool_order(pool_recs[::-1])]


def test_gate_arithmetic():
    def t(n, types, ref, wrong):
        return {"eligible": {"confirmation": n}, "by_question_type": {"confirmation": types},
                "by_label": {"confirmation": {"refusal": ref, "wrong": wrong, "success": 0},
                             "all": {"refusal": ref, "wrong": wrong, "success": 0}}}
    assert G.gate(t(60, {"a": 20, "b": 15}, 10, 20))["verdict"] == "proceed"
    assert G.gate(t(60, {"a": 20, "b": 15}, 5, 10))["verdict"].startswith("pool")
    assert G.gate(t(49, {"a": 20, "b": 15}, 20, 20))["verdict"] == "stop"
    assert G.gate(t(60, {"a": 20, "b": 15}, 5, 9))["verdict"] == "stop"


def test_swap_inserts_the_substitute_surface_form():
    lines = ["user, d1: hello", "user, d2: it took two weeks, I think.", "user, d3: bye"]
    r = rec("a", lines, [lines[1]], "two weeks", q="How long did it take?")
    _, place = G.place_gold(r, G.Checker())
    assert G.swap_prompt(r, place, "$5,150.")[2] == "user, d2: it took $5,150, I think."
    lines = ["user, d1: hello", "user, d2: rates rose 20% this year.", "user, d3: bye"]
    r = rec("a", lines, [lines[1]], "20%", q="By how much?")
    _, place = G.place_gold(r, G.Checker())
    assert G.swap_prompt(r, place, "8")[2] == "user, d2: rates rose 8 this year."


# --- rev 2 rulings 2-3 ----------------------------------------------------------------------------------------------

def test_number_subtypes_and_type_key():
    assert [G.number_subtype(x) for x in ("$800", "20 dollars", "20%", "5 percent", "three months", "24 hours",
                                          "30-minute", "10", "five", "10 pounds", "2-3 eggs")] == \
        ["currency", "currency", "percent", "percent", "duration", "duration", "duration", "bare", "bare", "bare",
         "bare"]
    assert G.type_key("$800") == "number:currency" and G.type_key("$800", rev=1) == "number"
    assert G.type_key("Paris") == "name" and G.type_key("June 3rd") == "date"


def test_rev2_other_rules_and_cap():
    tok = Tok()
    lines = ["user, d1: hello", "user, d2: I keep it under my bed.", "user, d3: bye"]
    r = rec("a", lines, [lines[1]], "under my bed", q="Where is it?")
    _, place = G.place_gold(r, G.Checker())

    def other(qid, gold, qtype="single-session-user"):
        x = rec(qid, lines, [lines[1]], gold, fold=1)
        return x._replace(question_type=qtype) if hasattr(x, "_replace") else \
            __import__("dataclasses").replace(x, question_type=qtype)
    pool = [r, other("b", "in the garage", "knowledge-update"),      # another question type: skipped
            other("c", "In the garage"),                             # other capitalisation: skipped
            other("d", "in the old blue garage shed"),               # +-1 token fails (rev 1's +-2 would too)
            other("e", "in the attic")]
    tinfo = {x.question_id: G.token_info(tok, x.gold) for x in pool}
    types = {x.question_id: G.type_key(x.gold) for x in pool}
    cand, why, prompt, _ = G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), uses=G.Counter(), rev=2)
    assert why is None and cand.question_id == "e" and "keep it in the attic." in prompt
    # rev 1 ignores the question type and capitalisation: the first candidate is taken
    assert G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), rev=1)[0].question_id == "b"
    # the cap: "in the attic" already used CAP times -> skipped; the row has no other candidate -> no_substitute
    stats = G.Counter()
    uses = G.Counter({"in the attic": G.CAP})
    out = G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), uses=uses, rev=2, stats=stats)
    assert out[0] is None and out[1] == "no_substitute"
    assert stats["cap_skips"] == 1 and stats["no_substitute_cap_only"] == 1
    uses = G.Counter({"in the attic": G.CAP - 1})
    assert G.choose_substitute(r, place, pool, tinfo, types, G.Checker(), uses=uses, rev=2)[0].question_id == "e"


def test_rev2_per_row_draw_is_seeded_by_row_index():
    recs = [rec(f"q{i}", ["user, d1: x"], ["user, d1: x"], "x", fold=i % 5) for i in range(20)]
    recs = sorted(recs, key=lambda r: (r.fold, r.question_id))
    a, b = G.row_candidates(recs, 3), G.row_candidates(recs, 3)
    assert [x.question_id for x in a] == [x.question_id for x in b] and recs[3] not in a and len(a) == 19
    ref = [r for j, r in enumerate(recs) if j != 3]
    G.random.Random(44 + 3).shuffle(ref)
    assert [x.question_id for x in a] == [x.question_id for x in ref]
    assert [x.question_id for x in G.row_candidates(recs, 4)] != [x.question_id for x in a]


def test_gate_rev2_edge_case_proceeds_with_caveat():
    t = {"eligible": {"confirmation": 50}, "by_question_type": {"confirmation": {"a": 15, "b": 14}},
         "by_label": {"confirmation": {"refusal": 14, "wrong": 12, "success": 0},
                      "all": {"refusal": 19, "wrong": 20, "success": 0}}}
    assert G.gate(t, rev=1)["verdict"].startswith("stop")
    v = G.gate(t, rev=2)["verdict"]
    assert v.startswith("pool") and "type-diversity caveat" in v
