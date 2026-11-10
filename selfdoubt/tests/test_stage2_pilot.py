"""Stage 2 pilot build checks ($0): string/token helpers with the real Qwen3 tokenizer, and the patching invariants on
a tiny random Qwen3 (embedding-layer swap patch = the donor run; last-layer patch = the base run; KV-cache suffix
forwards = full forwards).

    .venv/bin/python -m pytest selfdoubt/tests/test_stage2_pilot.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import stage2_pilot as P  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tok():
    from selfdoubt.engines import load_tokenizer
    return load_tokenizer(P.MODEL)


def make_row(tok, upd="Use 12 apples and 7 pears instead of 30.", new="19", old="37", cls="stale"):
    fp = C.full_prompt(tok, P.MODEL, "Tom has 30 apples and 7 pears. How many fruits?")
    prefix = "<think>\nTom has 30 apples and 7 pears, so 30 + 7 = 37."
    base = C.continuation_prompt(fp, prefix, "U0", upd)
    x0 = C.continuation_prompt(fp, "<think>\n", "U0", upd)
    return {"row_id": f"1|0.50|{cls}", "item_id": 1, "cut": 0.5, "cls": cls, "new": new, "old": old, "update": upd,
            "base_prompt": base, "base_span": P.update_span(base, fp + prefix, "U0", upd),
            "x0_prompt": x0, "x0_span": P.update_span(x0, fp + "<think>\n", "U0", upd),
            "n_prompt": fp + prefix, "continuation": "Okay, with 12 apples and 7 pears the total is 12 + 7 = 19.",
            "prompt_sha256": C.sha256_text(base)}


def test_swap_digits_changes_every_digit_and_keeps_length(tok):
    s = "x <update>Use 12 apples, 7 pears and 1/5 at $3.50.</update> 99"
    span = (s.index("Use"), s.index("</update>"))
    d = P.swap_digits(s, span, "k")
    assert len(d) == len(s) and d[:span[0]] == s[:span[0]] and d[span[1]:] == s[span[1]:]
    for a, b in zip(s[span[0]:span[1]], d[span[0]:span[1]]):
        assert (a == b) == (not a.isdigit())
    assert d[span[0]:span[1]].split()[1][0] != "0"          # a multi-digit run never starts with 0
    assert P.swap_digits(s, span, "k") == d                 # deterministic


def test_readout_tokens(tok):
    f, a, b = P.readout_tokens(tok, "14", "18")
    assert tok.decode(f).endswith("{1") and tok.decode([a]) == "4" and tok.decode([b]) == "8"
    f, a, b = P.readout_tokens(tok, "14", "140")
    assert tok.decode([a]) == "}" and tok.decode([b]) == "0"
    f, a, b = P.readout_tokens(tok, "-3", "3")              # "{-" is one token: the split is at the brace
    assert tok.decode(f).endswith("is \\boxed") or tok.decode([a]) != tok.decode([b])
    with pytest.raises(AssertionError):
        P.readout_tokens(tok, "5", "5")


def test_answer_str():
    assert P.answer_str("1,450") == "1450" and P.answer_str("12.0") == "12" and P.answer_str("-3.5") == "-3.5"
    assert P.answer_str("x+1") is None and P.answer_str(None) is None and P.answer_str(["1"]) is None


def test_digit_positions_and_alignment(tok):
    row = make_row(tok)
    ids, offs = P.encode_with_offsets(tok, row["base_prompt"])
    D = P.digit_positions(tok, ids, offs, tuple(row["base_span"]))
    assert [tok.decode([ids[i]]) for i in D] == list("127" + "30")
    d_ids = tok(P.swap_digits(row["base_prompt"], tuple(row["base_span"]), "z"), add_special_tokens=False)["input_ids"]
    assert len(d_ids) == len(ids) and [i for i, (a, b) in enumerate(zip(ids, d_ids)) if a != b] == D


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).float()
    return P.Runner(P.MODEL, None, "cpu", "float32", model=m, tok=tok)


def test_cache_suffix_equals_full_forward(tiny, tok):
    row = make_row(tok)
    ids = tok(row["base_prompt"], add_special_tokens=False)["input_ids"]
    full, _ = tiny.forward(ids)
    p0 = len(ids) - 20
    cache = tiny.prefix_cache(ids[:p0])
    for _ in range(2):                                      # the cache is cropped back after each suffix forward
        part, _ = tiny.forward(ids[p0:], cache, p0)
        assert torch.allclose(full, part, atol=1e-4)


def test_patch_invariants(tiny, tok):
    row = make_row(tok)
    (r,) = [x for x in P.measure_row(tiny, row, offsets=(0,)) if x["offset"] == 0]
    assert "skip" not in r and r["layers"] == list(range(tiny.n_layers + 1))
    assert abs(r["patch_swap"][0] - r["m_donor"]) < 1e-3     # embedding-layer swap patch = the donor
    assert abs(r["patch_swap"][-1] - r["m_base"]) < 1e-3     # last-layer patch at digit positions cannot reach the readout
    assert abs(r["patch_x0"][0] - r["m_base"]) < 1e-3        # same tokens at the embedding: no change
    assert abs(r["lens_base"][-1] - r["m_base"]) < 1e-3      # lens at the last layer = the real logits
    assert len(r["lens_n"]) == tiny.n_layers + 1 and r["tok_new"] == "1" and r["tok_old"] == "3"


def test_run_resumes_and_report(tiny, tok, tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(make_row(tok, cls=c, new=n)) + "\n"
                            for c, n in (("stale", "19"), ("new", "19"))))
    rr = [json.loads(x) for x in rows.read_text().splitlines()]
    rr[1]["row_id"] = "2|0.50"; rr[1]["item_id"] = 2
    rows.write_text("".join(json.dumps(x) + "\n" for x in rr))
    monkeypatch.setattr(P, "Runner", lambda *a, **k: tiny)
    out = tmp_path / "res.jsonl"
    assert P.run(rows, out, P.MODEL, None, "cpu", "float32", (0, 8), None, 1)["rows_done_now"] == 1
    assert P.run(rows, out, P.MODEL, None, "cpu", "float32", (0, 8), None, None)["rows_done_now"] == 1
    assert P.run(rows, out, P.MODEL, None, "cpu", "float32", (0, 8), None, None)["rows_done_now"] == 0
    rep = P.report(out, n_boot=50)
    assert set(rep["by_offset"]) == {"0", "8"} and "swap_effect_stale_minus_new" in rep["by_offset"]["0"]


# ---------------------------------------------------------------- pod script, plan mode ($0)

import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

SD = WT / "selfdoubt"


@pytest.fixture()
def pod(tmp_path):
    wt = tmp_path / "wt"
    shutil.copytree(SD, wt / "selfdoubt", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (wt / "runs/selfdoubt/s2").mkdir(parents=True)
    shutil.copy(WT / "runs/selfdoubt/split.json", wt / "runs/selfdoubt/split.json")
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls.txt\nexit 0\n'); runner.chmod(0o755)
    return {"wt": wt, "runner": runner, "calls": tmp_path / "calls.txt"}


def run_pod(pod, rows, sha=None, run_s="1000"):
    p = pod["wt"] / "runs/selfdoubt/s2/rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = sha or hashlib.sha256(p.read_bytes()).hexdigest()
    e = dict(os.environ, ENGINE="plan", PYTHON=str(pod["runner"]), RUN_S=run_s, OUT="runs/selfdoubt/s2_out",
             ROWS="runs/selfdoubt/s2/rows.jsonl", ROWS_SHA=sha)
    return subprocess.run(["bash", str(pod["wt"] / "selfdoubt/stage2_pilot_pod.sh")], env=e, capture_output=True,
                          text=True, timeout=60)


def test_pod_plan_runs_and_reports(pod):
    dev = json.loads((WT / "runs/selfdoubt/split.json").read_text())["dev"]["gsm8k"]
    r = run_pod(pod, [{"item_id": dev[0], "row_id": "a"}, {"item_id": dev[1], "row_id": "b"}])
    assert r.returncode == 0, r.stdout + r.stderr
    run, rep = pod["calls"].read_text().splitlines()
    assert run.startswith("-u selfdoubt/stage2_pilot.py run --rows runs/selfdoubt/s2/rows.jsonl")
    assert "--snapshot PLAN17" in run and "--offsets 0,32,128" in run and "--deadline-s" in run
    assert rep == "-u selfdoubt/stage2_pilot.py report --res runs/selfdoubt/s2_out/res.jsonl"
    assert "rows ok: 2 dev GSM8K rows" in (pod["wt"] / "runs/selfdoubt/s2_out/pod.log").read_text()


def test_pod_refuses_heldout_rows_and_bad_sha(pod):
    split = json.loads((WT / "runs/selfdoubt/split.json").read_text())
    r = run_pod(pod, [{"item_id": split["heldout"]["gsm8k"][0], "row_id": "h"}])
    assert r.returncode == 3 and not pod["calls"].exists()
    r = run_pod(pod, [{"item_id": split["dev"]["gsm8k"][0], "row_id": "a"}], sha="0" * 64)
    assert r.returncode == 3 and not pod["calls"].exists()
