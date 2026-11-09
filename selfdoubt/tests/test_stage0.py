"""Stage 0 pipeline tests: prompt assembly vs the authors' format, cuts, wrappers, gold parsing, scoring, resume.

Needs the Qwen3 tokenizer and the benchmark parquet in the HF cache (run with HF_HUB_OFFLINE=1 after one download).
    HF_HUB_OFFLINE=1 .venv/bin/python -m pytest selfdoubt/tests -q
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import engines as E  # noqa: E402
from selfdoubt import scoring as S  # noqa: E402
from selfdoubt import stage0_report as R  # noqa: E402
from selfdoubt import stage0_run as RUN  # noqa: E402
from selfdoubt.vendor.interrupt_lrm import prompt_utils_qwen as PU  # noqa: E402

M8 = "Qwen/Qwen3-8B"
CLONE = Path("/home/qamil-mirza/Code/llm-memctl/external/interrupt-lrm")


@pytest.fixture(scope="module")
def tok():
    return E.load_tokenizer(M8)


@pytest.fixture(scope="module")
def items():
    return C.load_items()


# ---------------------------------------------------------------- vendored code is the authors', unchanged

def _funcs(src: str) -> dict:
    t = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in t.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.skipif(not CLONE.exists(), reason="authors' clone not present")
def test_vendored_files_match_clone():
    v = C.VENDOR
    for f in ("wo_prompt_guidance.json", "w_prompt_guidance.json", "user_turn.json"):
        assert (v / "prompts/intervene" / f).read_bytes() == (CLONE / "src/prompts/intervene" / f).read_bytes()
    theirs = _funcs((CLONE / "src/prompt_utils.py").read_text())
    for name, body in _funcs((v / "prompt_utils_qwen.py").read_text()).items():
        if name != "make_mistral_chat_completion_request":
            assert body == theirs[name], name
    theirs = _funcs((CLONE / "eval/math/utils.py").read_text())
    for name, body in _funcs((v / "math_utils.py").read_text()).items():
        assert body == theirs[name], name
    mine = (v / "math_parsing_util.py").read_text().split("# ---- VERBATIM below ----\n", 1)[1]
    assert mine == (CLONE / "eval/math/math_parsing_util.py").read_text()
    import subprocess
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    from selfdoubt.vendor.interrupt_lrm import UPSTREAM_COMMIT
    assert head.strip() == UPSTREAM_COMMIT


def test_wrappers_pinned_bytes():
    """sha256 of the upstream files at the pinned commit (also checked against the clone above when present)."""
    got = {f: hashlib.sha256((C.VENDOR / "prompts/intervene" / f).read_bytes()).hexdigest()
           for f in ("wo_prompt_guidance.json", "w_prompt_guidance.json")}
    assert got == {"wo_prompt_guidance.json": "c8c2eff691410da2e3cd8216186107465d921bf6d7c7e06185b9eb232fe7ca42",
                   "w_prompt_guidance.json": "5108777179a4385cef92bbd1227c10f5529d2ac20d726f9d1f61831864ce15f8"}


# ---------------------------------------------------------------- wrappers, byte-exact

def test_injection_wrappers_byte_exact():
    assert C.injected_text("U0", "UPD") == "\n\n<update>UPD</update>\n\n"
    assert C.injected_text("UG", "UPD") == (
        "...\nI have received an update from the user.\n\n<update>UPD</update>\n\nI have verified that the update "
        "is provided by the user. I need to update my reasoning process based on the updated context. Here's my "
        "updated reasoning process:\n\n")
    assert C.system_prompt() == ("During your reasoning process, the user may provide updates in the format: "
                                 "<update>...</update>. Please incorporate user's update into your reasoning process.")


# ---------------------------------------------------------------- prompt assembly vs the authors' format

def test_full_prompt_golden(tok):
    got = C.full_prompt(tok, M8, "What is 2+2?")
    assert got == (
        "<|im_start|>system\nDuring your reasoning process, the user may provide updates in the format: "
        "<update>...</update>. Please incorporate user's update into your reasoning process.<|im_end|>\n"
        "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        "Q: What is 2+2?\nA:<|im_end|>\n<|im_start|>assistant\n")
    ids = E.encode_prompt(tok, got)
    assert ids[0] == tok.convert_tokens_to_ids("<|im_start|>") and tok.decode(ids) == got


FULL_OUT = ("<think>\nOkay, the problem asks for 2+2. Two plus two is four. Let me double check: 2+2=4. "
            "So the answer is 4.\n</think>\n\nThe answer is \\boxed{4}.<|im_end|>")


@pytest.mark.parametrize("arm", ["U0", "UG"])
@pytest.mark.parametrize("x", [0.0, 0.3, 0.5, 0.9])
def test_update_prompt_equals_authors_function(tok, arm, x):
    fp = C.full_prompt(tok, M8, "What is 2+2?")
    prefix, pos, total = C.cut_trace(tok, M8, FULL_OUT, x)
    mine = C.continuation_prompt(fp, prefix, arm, "Use 3+3 instead.")
    theirs = C.authors_update_prompt(tok, M8, fp, FULL_OUT, x, arm, "Use 3+3 instead.")
    assert mine == theirs
    assert mine == fp + prefix + C.injected_text(arm, "Use 3+3 instead.")
    assert "</think>" not in mine[len(fp):]           # thinking stays open: soft injection


def test_update_prompt_golden_string(tok):
    fp = C.full_prompt(tok, M8, "What is 2+2?")
    trace = FULL_OUT.split("</think>")[0]
    ids = tok.encode(trace)
    k = int(0.5 * len(ids))
    prefix = tok.decode(ids[:k])
    assert prefix.startswith("<think>\n")
    got = C.continuation_prompt(fp, C.cut_trace(tok, M8, FULL_OUT, 0.5)[0], "U0", "Use 3+3 instead.")
    assert got == fp + prefix + "\n\n<update>Use 3+3 instead.</update>\n\n"
    n = C.continuation_prompt(fp, C.cut_trace(tok, M8, FULL_OUT, 0.5)[0], "N", "Use 3+3 instead.")
    assert n == fp + prefix


def test_released_cli_mode_drops_update_text(tok):
    """Documents an upstream bug: with --mode subsequent_interrupt_update (the only update mode the CLI accepts),
    format_subsequent_input_prompt checks for "subsequent_intervene" and never inserts ex["update"]."""
    fp = C.full_prompt(tok, M8, "What is 2+2?")
    released = C.authors_update_prompt(tok, M8, fp, FULL_OUT, 0.5, "U0", "Use 3+3 instead.",
                                       mode="subsequent_interrupt_update")
    assert released.endswith("\n\n<update></update>\n\n")


# ---------------------------------------------------------------- the cut

def test_cut_token_level(tok):
    trace = FULL_OUT.split("</think>")[0]
    n = len(tok.encode(trace))
    p0, pos0, tot = C.cut_trace(tok, M8, FULL_OUT, 0.0)
    assert (p0, pos0, tot) == ("<think>\n", 0, n)
    p1, pos1, _ = C.cut_trace(tok, M8, FULL_OUT, 1.0)
    assert p1 == trace and pos1 == n
    p5, pos5, _ = C.cut_trace(tok, M8, FULL_OUT, 0.5)
    assert pos5 == int(0.5 * n) and tok.encode(p5) == tok.encode(trace)[:pos5]
    assert trace.startswith(p5)


def test_cut_char_level_alternative(tok):
    trace = FULL_OUT.split("</think>")[0]
    p, pos, tot = C.cut_trace(tok, M8, FULL_OUT, 0.5, unit="char")
    assert tot == len(trace) and pos == int(0.5 * len(trace)) and p == trace[:pos]


def test_cut_without_close_tag_uses_whole_output(tok):
    out = "<think>\nstill thinking and thinking"           # FULL hit max_tokens inside the thinking
    p, pos, tot = C.cut_trace(tok, M8, out, 0.5)
    assert tot == len(tok.encode(out)) and out.startswith(p)


# ---------------------------------------------------------------- data, split, seeds

def test_split_deterministic_and_balanced(items, tmp_path):
    s1, s2 = C.make_split(items), C.make_split(items)
    assert s1 == s2
    for src in C.SOURCES:
        d, h = set(s1["dev"][src]), set(s1["heldout"][src])
        assert len(d) == len(h) == 250 and not d & h
    split, sha = C.ensure_split(items, tmp_path / "split.json")
    assert hashlib.sha256((tmp_path / "split.json").read_bytes()).hexdigest() == sha
    C.ensure_split(items, tmp_path / "split.json")    # second call verifies, does not rewrite
    (tmp_path / "split.json").write_text("{}")
    with pytest.raises(SystemExit):
        C.ensure_split(items, tmp_path / "split.json")


def test_stub_selection(items):
    split = C.make_split(items)
    sel = C.stub_items(items, split, 5)
    assert [r["source"] for r in sel] == ["gsm8k"] * 5 + ["math500"] * 5
    dev = {i for s in C.SOURCES for i in split["dev"][s]}
    assert all(int(r["id"]) in dev for r in sel)
    m = [r for r in items if r["source"] == "math500" and int(r["id"]) in dev]
    assert min(len(r["original_answer"]) for r in sel[5:]) >= sorted(len(r["original_answer"]) for r in m)[-5]


def test_seeds_distinct():
    seeds = {C.request_seed(M8, 3, a, x) for a in ("N", "U0", "UG") for x in C.CUTS}
    seeds |= {C.request_seed(M8, 3, "FULL", None), C.request_seed("Qwen/Qwen3-1.7B", 3, "U0", 0.3)}
    assert len(seeds) == 11
    assert C.request_seed(M8, 3, "U0", 0.3) == C.request_seed(M8, 3, "U0", 0.30000)


# ---------------------------------------------------------------- gold parsing (authors' code) on dataset rows

def test_gold_gsm8k(items):
    r = next(x for x in items if int(x["id"]) == 0)
    assert S.gold_answer("gsm8k", r["original_answer"]) == "109"
    assert S.is_correct("gsm8k", r["original_answer"], "...</think>\n\nSo \\boxed{109}.")
    assert not S.is_correct("gsm8k", r["original_answer"], "...</think>\n\nSo \\boxed{110}.")
    assert S.is_correct("gsm8k", r["original_answer"], "</think> The total is \\boxed{109} years")


def test_gold_math500(items):
    r = next(x for x in items if int(x["id"]) == 500)
    assert "boxed" not in r["original_answer"]                               # bare answer, not boxed
    assert S.is_correct("math500", r["original_answer"], "</think> \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}")
    assert S.is_correct("math500", r["original_answer"], "</think> \\boxed{(3,\\frac{\\pi}{2})}")
    assert not S.is_correct("math500", r["original_answer"], "</think> \\boxed{(5, -0.927)}")
    r = next(x for x in items if int(x["id"]) == 501)
    assert S.is_correct("math500", r["original_answer"], "\\boxed{p - q}")


# ---------------------------------------------------------------- scoring categories, drop rule, truncated

def test_categorize_rules():
    kw = dict(n_valid=True, n_is_gold=False, truncated=False, correct=False, stale=False)
    assert S.categorize(**kw) == "other"
    assert S.categorize(**{**kw, "correct": True}) == "new"
    assert S.categorize(**{**kw, "stale": True}) == "stale"
    assert S.categorize(**{**kw, "truncated": True, "correct": True}) == "truncated"
    assert S.categorize(**{**kw, "n_is_gold": True, "correct": True}) == "dropped_ref_is_gold"
    assert S.categorize(**{**kw, "n_valid": False}) == "dropped_ref_invalid"


def _rec(item, arm, cut, text, hit=False, src="gsm8k"):
    return {"key": C.rec_key(M8, item, arm, cut), "model": M8, "item_id": item, "source": src, "arm": arm,
            "cut": cut, "text": text, "hit_max": hit, "injected": "" if arm in ("FULL", "N") else
            C.injected_text(arm, "upd")}


def test_score_model_end_to_end(items):
    gold0 = next(x for x in items if int(x["id"]) == 0)                      # gold 109
    its = {0: dict(gold0, update="upd"), 1: dict(gold0, id=1, update="upd")}
    b = lambda v: f"stuff\n</think>\n\n\\boxed{{{v}}}"                       # noqa: E731
    recs = [_rec(0, "FULL", None, b(50)),
            _rec(0, "N", 0.3, b(50)), _rec(0, "U0", 0.3, b(109)), _rec(0, "UG", 0.3, b(50)),
            _rec(0, "N", 0.5, b(109)), _rec(0, "U0", 0.5, b(109)), _rec(0, "UG", 0.5, b(3)),
            _rec(0, "N", 0.9, b(51)), _rec(0, "U0", 0.9, "<think> no end", hit=True), _rec(0, "UG", 0.9, b(7)),
            _rec(0, "U0", 0.0, b(50)), _rec(0, "UG", 0.0, b(109))]
    rows = {(r["arm"], r["cut"]): r for r in R.score_model(recs, its)}
    assert rows[("U0", 0.3)]["category"] == "new"
    assert rows[("UG", 0.3)]["category"] == "stale"
    assert rows[("U0", 0.5)]["category"] == "dropped_ref_is_gold"           # N == gold: undistinguishable
    assert rows[("UG", 0.5)]["category"] == "dropped_ref_is_gold"
    assert rows[("U0", 0.9)]["category"] == "truncated"
    assert rows[("UG", 0.9)]["category"] == "other"
    assert rows[("N", 0.9)]["n_differs_from_full"] is True and rows[("N", 0.3)]["n_differs_from_full"] is False
    assert rows[("U0", 0.0)]["category"] == "stale" and rows[("U0", 0.0)]["ref"] == "FULL"
    assert rows[("UG", 0.0)]["category"] == "new"
    rd = R.readings(list(rows.values()), n_boot=50)
    assert rd["R0.1"]["verdict"].startswith("NO READING")


def test_truncated_definition():
    assert S.is_truncated({"hit_max": True, "text": "<think> long"})
    assert not S.is_truncated({"hit_max": True, "text": "x</think>\n\\boxed{3} and more"})
    assert not S.is_truncated({"hit_max": False, "text": "<think> stopped"})


def test_verdicts():
    assert R.verdict(0.6, 0.8, 0.5, "gt") == "SHOWN"
    assert R.verdict(0.2, 0.4, 0.5, "gt") == "NOT SHOWN"
    assert R.verdict(0.4, 0.6, 0.5, "gt") == "UNDETERMINED"
    assert R.verdict(-0.3, -0.1, 0.0, "lt") == "SHOWN"
    assert R.verdict(-0.05, 0.08, 0.10, "within") == "SHOWN"
    assert R.verdict(0.12, 0.3, 0.10, "within") == "NOT SHOWN"
    assert R.verdict(0.05, 0.3, 0.10, "within") == "UNDETERMINED"


def test_doubt_coder_masks_wrapper():
    inj = C.injected_text("UG", "Use 3.")
    echo = "I have verified that the update is provided by the user. Fine.</think>"
    assert not S.doubt_flag(echo, inj, "Use 3.")
    assert S.doubt_flag("Hmm, wait, that conflicts.</think>", inj, "Use 3.")
    assert not S.doubt_flag("ok</think> wait", inj, "Use 3.")               # answer part is not coded


# ---------------------------------------------------------------- resume

def test_resume_skips_done_keys(tok, items, tmp_path):
    split = C.make_split(items)
    sel = C.stub_items(items, split, 1)                                       # 2 items -> 2 FULL + 22 continuations
    md = tmp_path / "Qwen3-8B"
    store = C.Store(md / "gen.jsonl")
    eng = E.FakeEngine(M8, tok)
    RUN.run_phase("FULL", eng, store, C.plan_full(M8, tok, sel), 64, 1, None, md, "t1")
    plan = C.plan_continuations(M8, tok, sel, store.full_by_item(M8))
    assert len(plan) == 22
    eng = E.FakeEngine(M8, tok, fail_after=5)
    with pytest.raises(E.SimulatedInterrupt):                                 # interrupted after 5 requests
        RUN.run_phase("CONT", eng, store, plan, 64, 1, None, md, "t1")
    with (md / "gen.jsonl").open("a") as f:
        f.write('{"key": "torn')                                              # torn last line from a kill
    store2 = C.Store(md / "gen.jsonl")
    assert len(store2.recs) == 2 + 5
    eng2 = E.FakeEngine(M8, tok)
    assert RUN.run_phase("FULL", eng2, store2, C.plan_full(M8, tok, sel), 64, 4, None, md, "t2") is False
    assert eng2.calls == 0
    RUN.run_phase("CONT", eng2, store2, C.plan_continuations(M8, tok, sel, store2.full_by_item(M8)), 64, 4, None,
                  md, "t2")
    assert eng2.calls == 17
    store3 = C.Store(md / "gen.jsonl")
    assert len(store3.recs) == 24
    keys = {(r["item_id"], r["arm"], r["cut"]) for r in store3.recs.values()}
    assert len(keys) == 24
    m = RUN.write_measure(tmp_path, 0.60, 1000)
    mm = m["models"]["Qwen3-8B"]
    assert mm["n_items"] == 2 and mm["n_records"] == 24
    assert mm["phases"]["FULL"]["n_records"] == 2 and mm["phases"]["CONT"]["n_records"] == 22
    assert (tmp_path / "measure.json").exists() and m["projected_usd_all_models"] >= 0


def test_deadline_stops_between_chunks(tok, items, tmp_path):
    sel = C.stub_items(items, C.make_split(items), 1)
    store = C.Store(tmp_path / "gen.jsonl")
    eng = E.FakeEngine(M8, tok)
    assert RUN.run_phase("FULL", eng, store, C.plan_full(M8, tok, sel), 64, 1, 0.0001, tmp_path, "t") is True
    assert eng.calls == 0 and not store.recs


def test_deadline_mid_chunk_keeps_finished(tok, items, tmp_path):
    """The engine stops inside a chunk: finished requests are stored, unfinished ones only counted, rerun fills them."""
    import json as _json
    sel = C.stub_items(items, C.make_split(items), 1)                         # 2 items
    md = tmp_path / "Qwen3-8B"
    store = C.Store(md / "gen.jsonl")
    RUN.run_phase("FULL", E.FakeEngine(M8, tok), store, C.plan_full(M8, tok, sel), 64, 64, None, md, "t1")
    plan = C.plan_continuations(M8, tok, sel, store.full_by_item(M8))
    eng = E.FakeEngine(M8, tok, deadline_after=7)
    assert RUN.run_phase("CONT", eng, store, plan, 64, 64, 1e12, md, "t1") is True
    assert len(store.recs) == 2 + 7
    st = [_json.loads(l) for l in (md / "phases.jsonl").read_text().splitlines()][-1]
    assert st["n_unfinished"] == 15 and st["gen_tokens_partial"] == 45
    eng2 = E.FakeEngine(M8, tok)
    assert RUN.run_phase("CONT", eng2, C.Store(md / "gen.jsonl"), plan, 64, 64, None, md, "t2") is False
    assert eng2.calls == 15 and len(C.Store(md / "gen.jsonl").recs) == 24
    m = RUN.write_measure(tmp_path, 0.60, 1000, source_counts={"gsm8k": 500, "math500": 500})
    mm = m["models"]["Qwen3-8B"]
    assert mm["phases"]["CONT"]["n_unfinished_at_deadline_all_invocations"] == 15
    assert mm["phases"]["CONT"]["n_missing_now"] == 0                       # the rerun filled them
    assert mm["phases"]["CONT"]["gen_tokens_per_request_high"] >= mm["phases"]["CONT"]["gen_tokens_per_request_low"]
    assert m["projected_token_high_usd_all_models"] >= m["projected_token_low_usd_all_models"] > 0
    assert set(mm["projection_by_source"]) == {"gsm8k", "math500"}


def test_busy_rate():
    # 10 requests; first two windows have >= 5 running (100 tok in 10 s, then 80 in 10 s), the tail is excluded
    tr = [[0.0, 10, 0], [10.0, 8, 100], [20.0, 3, 180], [30.0, 1, 200]]
    assert RUN.busy_rate([tr], [10]) == 9.0
    assert RUN.busy_rate([None], [10]) is None


# ---------------------------------------------------------------- Stage 0 math normaliser (frozen 2026-11-09)

PM = r"\begin{pmatrix} -2 \\ -14 \\ -7 \end{pmatrix}"


def test_norm_row_separators_survive():
    """The released double strip turned this into '-2-14-7'; the normaliser keeps the entries."""
    assert S.vector_entries(PM) == ["-2", "-14", "-7"]
    assert S.pred_answer("math500", "</think> \\boxed{" + PM + "}") == ["-2", "-14", "-7"]
    assert S.is_correct("math500", PM, "</think>\n\n\\boxed{" + PM + "}")
    assert not S.is_correct_authors("math500", PM, "</think>\n\n\\boxed{" + PM + "}")   # the released miss


def test_norm_pmatrix_tuple_bmatrix_left_right():
    gold = r"\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}"
    for p in ["(-18,-49,96)", r"\left( -18, -49, 96 \right)", r"\begin{bmatrix} -18 \\ -49 \\ 96 \end{bmatrix}",
              r"\begin{pmatrix}-18\\-49\\96\end{pmatrix}"]:
        assert S.is_correct("math500", gold, "\\boxed{" + p + "}"), p
    assert not S.is_correct("math500", gold, "\\boxed{(-18,-49,95)}")
    assert not S.is_correct("math500", gold, "\\boxed{(-18,-49)}")                      # length must match
    assert not S.is_correct("math500", gold, "\\boxed{-18}")                             # scalar never = vector
    assert not S.same_answer("math500", ["-18", "-49", "96"], "-18")


def test_norm_fractions_inside_vectors():
    gold = r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}"
    assert S.is_correct("math500", gold, r"\boxed{\left(\frac{16}{49}, \frac{48}{49}, \frac{24}{49}\right)}")
    assert S.is_correct("math500", gold, r"\boxed{\begin{pmatrix} \dfrac{16}{49} \\ \frac{48}{49} \\ \tfrac{24}{49} \end{pmatrix}}")
    assert not S.is_correct("math500", gold, r"\boxed{\left(\frac{2}{3}, \frac{14}{15}, \frac{1}{3}\right)}")
    assert S.vector_entries(r"(\frac{1,2}{3}, 4)") == [r"\frac{1,2}{3}", "4"]          # commas inside braces


def test_norm_intervals_stay_out_of_tuple_path():
    for iv in ["(0,1]", "[0,1)", "[0,1]", r"(-\infty, 3]", "(1+2)", "(3)"]:
        assert S.vector_entries(iv) is None, iv
    assert S.is_correct("math500", "(0,1]", r"\boxed{(0,1]}")
    assert not S.is_correct("math500", "(0,1]", r"\boxed{(0,1)}")                       # interval != tuple
    assert not S.is_correct("math500", "[0,1]", r"\boxed{(0,1)}")


def test_norm_prefixed_tuple():
    assert S.is_correct("math500", r"(2,\infty)", r"\boxed{x \in (2, \infty)}")
    assert S.is_correct("math500", r"(2,\infty)", r"\boxed{(2, \infty)}")
    assert not S.is_correct("math500", r"(2,\infty)", r"\boxed{x \in [2, \infty)}")
    assert S.is_correct("math500", r"x \in [-2,7]", r"\boxed{[-2, 7]}")                  # dev gold 883 form
    assert S.is_correct("math500", r"[-2,7]", r"\boxed{x \in [-2,7]}")


def test_norm_matrix_shape():
    m = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
    assert S.vector_entries(m) == [["1", "2"], ["3", "4"]]
    assert S.is_correct("math500", m, r"\boxed{\begin{bmatrix}1&2\\3&4\end{bmatrix}}")
    assert not S.is_correct("math500", m, r"\boxed{(1,2,3,4)}")
    assert S.vector_entries(r"\begin{pmatrix} 1 & 2 \end{pmatrix}") == ["1", "2"]          # row vector flattens


def test_norm_scalars_single_strip():
    assert S.is_correct("math500", r"\frac{14}{3}", r"\boxed{\dfrac{14}{3}}")
    assert S.is_correct("math500", "p - q", r"\boxed{p - q}")
    assert S.same_answer("math500", S.pred_answer("math500", r"\boxed{\frac{1}{2}}"), "0.5")


def test_norm_dev_gold_roundtrip(items):
    """Every dev MATH-500 gold equals itself written as a model would (\\boxed{gold}). Dev split only."""
    split = C.make_split(items)
    by_id = {int(r["id"]): r for r in items}
    dev = split["dev"]["math500"]
    assert len(dev) == 250
    bad = []
    n_vec = 0
    for i in dev:
        g = by_id[i]["original_answer"]
        cg = S.gold_answer("math500", g)
        n_vec += isinstance(cg, list)
        text = "...</think>\n\nThe answer is $\\boxed{" + g + "}$."
        if not (S.valid_pred(cg) and S.pred_answer("math500", text) == cg and S.is_correct("math500", g, text)):
            bad.append((i, g))
    assert not bad, bad[:10]
    assert n_vec >= 4                                    # 4 matrix + tuple-like golds take the vector path


def test_shuffled_heldout_rule(items):
    """Stage 0 plan (2026-11-09): MATH-500 = first 100 held-out ids in seed-56 shuffle order; fixed, content-blind."""
    split = C.make_split(items)
    o = C.shuffled_heldout(items, split, "math500")
    assert len(o) == 250 and sorted(o) == split["heldout"]["math500"]
    first = C.shuffled_heldout(items, split, "math500", 100)
    assert first == o[:100] and len(set(first)) == 100
    assert first != sorted(first)                                            # shuffle order, not id order
    sel = C.select_items(items, split, "full", "heldout", 100, sources=("math500",))
    assert [int(r["id"]) for r in sel] == first
    assert not set(first) & set(split["dev"]["math500"])
    g = C.select_items(items, split, "full", "heldout", None, sources=("gsm8k",))
    assert sorted(int(r["id"]) for r in g) == split["heldout"]["gsm8k"]
