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
            "n_prompt": fp + prefix,
            "continuation": "Okay, with 12 apples and 7 pears the total is 12 + 7 = 19.\n</think>\n\n\\boxed{19}",
            "fact_new": "12", "fact_old": "30", "fact_skip": None,
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


def test_fact_pair_and_end_offset(tok):
    assert P.fact_pair("He has 30 apples and 7 pears.", "He has 12 apples and 7 pears.",
                       "Use 30 apples instead of 12.") == (("30", "12"), None)
    assert P.fact_pair("A 1,200 B 3", "A 1,500 B 3", "Make it 1,200.") == (("1200", "1500"), None)
    assert P.fact_pair("x 5", "x 5", "same")[0] is None
    assert P.fact_pair("x 5 y 6", "x 5", "u")[1].startswith("number counts")
    assert P.fact_pair("x 5", "x 4", "set it to 9")[1].startswith("first changed")
    c = "so 12 + 7 = 19.\n</think>\n\n\\boxed{19}"
    k = P.end_offset(tok, c)
    assert tok.decode(tok(c, add_special_tokens=False)["input_ids"][:k]).endswith("\n\n") and P.end_offset(tok, "no") is None


def test_patch_invariants(tiny, tok):
    row = make_row(tok)
    recs = P.measure_row(tiny, row, offsets=(0, "end"))
    assert not any("skip" in x or "skip_x0" in x for x in recs), recs
    (r,) = [x for x in recs if x["offset"] == 0 and x["target"] == "ans"]
    (f,) = [x for x in recs if x["offset"] == 0 and x["target"] == "fact"]
    (e,) = [x for x in recs if x["offset"] == "end" and x["target"] == "ans"]
    assert f["tok_new"] == "1" and f["tok_old"] == "3" and abs(f["patch_swap"][0] - f["m_donor"]) < 1e-3
    assert e["patch_swap"] == [] and e["k_tokens"] > 0
    assert r["base_logit_new"] - r["base_logit_old"] == pytest.approx(r["m_base"], abs=1e-3)
    assert r["lens_base_new"][-1] - r["lens_base_old"][-1] == pytest.approx(r["m_base"], abs=1e-3)
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
    assert {"ans|0", "ans|8", "fact|0", "fact|8"} <= set(rep["tables"])
    assert "swap_effect_stale_minus_new" in rep["tables"]["ans|0"] and rep["skips"] == {}


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
    assert "--snapshot PLAN17" in run and "--offsets 0,32,128,end" in run and "--deadline-s" in run
    assert rep == "-u selfdoubt/stage2_pilot.py report --res runs/selfdoubt/s2_out/res.jsonl --label dev"
    assert "rows ok: 2 dev GSM8K rows" in (pod["wt"] / "runs/selfdoubt/s2_out/pod.log").read_text()


def test_pod_refuses_heldout_rows_and_bad_sha(pod):
    split = json.loads((WT / "runs/selfdoubt/split.json").read_text())
    r = run_pod(pod, [{"item_id": split["heldout"]["gsm8k"][0], "row_id": "h"}])
    assert r.returncode == 3 and not pod["calls"].exists()
    r = run_pod(pod, [{"item_id": split["dev"]["gsm8k"][0], "row_id": "a"}], sha="0" * 64)
    assert r.returncode == 3 and not pod["calls"].exists()


def test_combined_pod_plan_wiring(pod):
    dev = json.loads((WT / "runs/selfdoubt/split.json").read_text())["dev"]["gsm8k"]
    p = pod["wt"] / "runs/selfdoubt/s2/rows.jsonl"
    p.write_text(json.dumps({"item_id": dev[0], "row_id": "a"}) + "\n")
    e = dict(os.environ, ENGINE="plan", PYTHON=str(pod["runner"]), RUN_S="9000", OUT="runs/selfdoubt/c",
             PILOT_S="4000", ROWS="runs/selfdoubt/s2/rows.jsonl", ROWS_SHA=hashlib.sha256(p.read_bytes()).hexdigest(),
             S1_STEPS="s1_1p7b_gsm8k_dev|17|gsm8k|dev|250|3300", S1_EXTRA="--cut 0.9 --payload plain,doc")
    r = subprocess.run(["bash", str(pod["wt"] / "selfdoubt/s1s2_pod.sh")], env=e, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = pod["calls"].read_text().splitlines()
    assert calls[0].startswith("-u selfdoubt/stage2_pilot.py run") and "--out runs/selfdoubt/c/s2/res.jsonl" in calls[0]
    assert calls[1].startswith("-u selfdoubt/stage2_pilot.py report")
    s1 = calls[2]
    assert s1.startswith("-u selfdoubt/stage1_run.py --engine vllm --mode full --split-name dev")
    assert "--out-root runs/selfdoubt/c/s1/dev" in s1 and s1.endswith("--cut 0.9 --payload plain,doc")
    assert "--allow-heldout" not in s1 and (pod["wt"] / "runs/selfdoubt/c/pod.done").exists()
    e["S1_STEPS"] = "s1_x|17|gsm8k|heldout|250|3300"
    assert subprocess.run(["bash", str(pod["wt"] / "selfdoubt/s1s2_pod.sh")], env=e, capture_output=True,
                          timeout=60).returncode == 3


def test_combined_plan_file():
    v = {}
    for line in (SD / "plans/s1s2_dev.env").read_text().splitlines():
        if line and not line.startswith("#"):
            k, x = line.split("=", 1)
            v[k] = x.strip('"')
    assert v["KEY_VAR"] == "RUNPOD_API_KEY_2" and v["POD_SCRIPT"] == "selfdoubt/s1s2_pod.sh"
    assert int(v["COPY_AT"]) == int(v["HARD_S"]) - 240
    ceiling = 200 + int(v["PILOT_S"]) + int(v["S1_STEPS"].split("|")[-1]) + 400
    assert int(v["HARD_S"]) >= 1.25 * ceiling - 1
    rows = WT / v["ROWS"]
    if rows.exists():
        assert hashlib.sha256(rows.read_bytes()).hexdigest() == v["ROWS_SHA"]


# ---------------------------------------------------------------- held-out replication + R2 (PROPOSAL c69f28e)

def test_l_half():
    assert P.l_half([-10, -10, -9, -6, -4, -1, 0]) == 4          # -4 / -10 = 0.4 < 0.5
    assert P.l_half([8, 8, 5, 3, 0]) == 3                         # sign of e[0] does not matter
    assert P.l_half([0, 0, 0]) is None and P.l_half([-2, -2, -2]) is None


def _r2_rows(n=45, shift=0, fact0=10.0, ans128=-8.0, items_per_class=None):
    """Synthetic usable rows: per class n rows at fact|0, fact|32, fact|128, ans|128."""
    import random
    rng = random.Random(1)
    rows = []
    for c in P.CLASSES:
        for j in range(n):
            iid = (0 if c == "stale" else 1000) + j
            noise = lambda: rng.gauss(0, 0.5)  # noqa: E731
            lh = 20 + (shift if c == "stale" else 0)
            curve = [-12.0 if l < lh else 0.0 for l in range(29)]
            base = {"item_id": iid, "cls": c, "layers": list(range(29))}
            m0 = fact0 + noise()
            rows.append({**base, "target": "fact", "offset": 0, "m_base": m0,
                         "patch_swap": [m0 + e for e in curve]})
            rows.append({**base, "target": "fact", "offset": 32, "m_base": (5 if c == "stale" else 9) + noise()})
            rows.append({**base, "target": "fact", "offset": 128, "m_base": (5 if c == "stale" else 9) + noise()})
            rows.append({**base, "target": "ans", "offset": 128, "m_base": (ans128 if c == "stale" else 1) + noise()})
    return rows


def test_r2_readings_synthetic():
    r = P.r2_readings(_r2_rows(), n_boot=200)
    assert r["R2.1"]["verdict"] == "SHOWN" and r["R2.2"]["verdict"] == "SHOWN" and r["R2.3"]["verdict"] == "SHOWN"
    assert r["R2.4"]["L_half"] == {"stale": 20, "new": 20} and r["R2.4"]["verdict"] == "SHOWN"
    assert r["map"]["R2.1_R2.3_R2.4_all_shown"]
    r = P.r2_readings(_r2_rows(shift=5, fact0=-10.0, ans128=3.0), n_boot=200)
    assert r["R2.1"]["verdict"] == "NOT SHOWN" and r["R2.3"]["verdict"] == "NOT SHOWN"
    assert r["R2.4"]["point"] == 5 and r["R2.4"]["verdict"] == "NOT SHOWN"
    r = P.r2_readings(_r2_rows(n=39), n_boot=50)                   # < 40 usable rows per class
    assert all(r[k]["verdict"] == "UNDETERMINED" for k in ("R2.1", "R2.2", "R2.3", "R2.4"))


def test_select_heldout_needs_confirmation(tmp_path):
    with pytest.raises(SystemExit, match="heldout-confirmed"):
        P.main(["select", "--split-name", "heldout", "--store", str(tmp_path / "x.jsonl"),
                "--out", str(tmp_path / "rows.jsonl")])


def test_pod_heldout_mode(pod):
    split = json.loads((WT / "runs/selfdoubt/split.json").read_text())
    rows = [{"item_id": split["heldout"]["gsm8k"][0], "row_id": "h"}]
    p = pod["wt"] / "runs/selfdoubt/s2/rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e = dict(os.environ, ENGINE="plan", PYTHON=str(pod["runner"]), RUN_S="1000", OUT="runs/selfdoubt/s2_out",
             ROWS="runs/selfdoubt/s2/rows.jsonl", ROWS_SHA=hashlib.sha256(p.read_bytes()).hexdigest(),
             SPLIT_NAME="heldout")
    sh = ["bash", str(pod["wt"] / "selfdoubt/stage2_pilot_pod.sh")]
    assert subprocess.run(sh, env=e, capture_output=True, timeout=60).returncode == 3     # not confirmed
    assert not pod["calls"].exists()
    e["HELDOUT_CONFIRMED"] = "1"
    r = subprocess.run(sh, env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert pod["calls"].read_text().splitlines()[1].endswith("--label heldout")
    p.write_text(json.dumps({"item_id": split["dev"]["gsm8k"][0], "row_id": "d"}) + "\n")   # dev row in heldout mode
    e["ROWS_SHA"] = hashlib.sha256(p.read_bytes()).hexdigest()
    pod["calls"].unlink()
    assert subprocess.run(sh, env=e, capture_output=True, timeout=60).returncode == 3 and not pod["calls"].exists()


def test_combined_v2_wiring(pod):
    split = json.loads((WT / "runs/selfdoubt/split.json").read_text())
    p = pod["wt"] / "runs/selfdoubt/s2/rows.jsonl"
    p.write_text(json.dumps({"item_id": split["heldout"]["gsm8k"][0], "row_id": "h"}) + "\n")
    e = dict(os.environ, ENGINE="plan", PYTHON=str(pod["runner"]), RUN_S="9000", OUT="runs/selfdoubt/c",
             PILOT_S="3300", ROWS="runs/selfdoubt/s2/rows.jsonl", ROWS_SHA=hashlib.sha256(p.read_bytes()).hexdigest(),
             S2_SPLIT="heldout", S2_HELDOUT_CONFIRMED="1",
             S1_STEPS="s1v2_1p7b_gsm8k_dev|17|gsm8k|dev|250|3900", S1_EXTRA="--cut 0.9 --payload plain --sys s1t,s0")
    sh = ["bash", str(pod["wt"] / "selfdoubt/s1s2_pod.sh")]
    r = subprocess.run(sh, env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = pod["calls"].read_text().splitlines()
    assert calls[1].endswith("--label heldout")
    assert "--split-name dev" in calls[2] and calls[2].endswith("--cut 0.9 --payload plain --sys s1t,s0")
    assert "--allow-heldout" not in calls[2]
    e["S1_EXTRA"] = "--cut 0.9 --payload plain --sys s1t,s0 --split-name heldout"
    assert subprocess.run(sh, env=e, capture_output=True, timeout=60).returncode == 3


def test_v2_plan_file():
    v = {}
    for line in (SD / "plans/s1s2_v2.env").read_text().splitlines():
        if line and not line.startswith("#"):
            k, x = line.split("=", 1)
            v[k] = x.strip('"')
    assert v["KEY_VAR"] == "RUNPOD_API_KEY_2" and v["EXPECT_EMAIL"] == "qamilmirza@gmail.com"
    assert v["S2_SPLIT"] == "heldout" and v["S2_HELDOUT_CONFIRMED"] == "1"
    assert v["S1_STEPS"].split("|")[3] == "dev" and "heldout" not in v["S1_EXTRA"]
    assert "--sys s1t,s0" in v["S1_EXTRA"] and "--payload plain" in v["S1_EXTRA"]
    assert int(v["COPY_AT"]) == int(v["HARD_S"]) - 240
    ceiling = 200 + int(v["PILOT_S"]) + int(v["S1_STEPS"].split("|")[-1]) + 400
    assert int(v["HARD_S"]) >= 1.25 * ceiling - 1
    for f in v["BUNDLE_EXTRA"].split():
        assert f.startswith("runs/selfdoubt/")
    rows = WT / v["ROWS"]
    if rows.exists():
        assert hashlib.sha256(rows.read_bytes()).hexdigest() == v["ROWS_SHA"]
        ho = set(json.loads((WT / "runs/selfdoubt/split.json").read_text())["heldout"]["gsm8k"])
        assert all(json.loads(x)["item_id"] in ho for x in rows.read_text().splitlines())
