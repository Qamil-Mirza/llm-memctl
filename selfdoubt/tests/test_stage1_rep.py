"""Stage 1 replication build (PROPOSAL 362b16c; Advisor go 2026-11-10, $0): plan files (one pod / two pods), bundled
FULL stores, plan-mode pod runs with the real plans, and the frozen readings (dry check, stratified pool, stop rule).

    runs/selfdoubt/_venv457/bin/python -m pytest selfdoubt/tests/test_stage1_rep.py -q
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SD = Path(__file__).resolve().parents[1]
WT = SD.parent
sys.path.insert(0, str(WT))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import stage1_rep as REP  # noqa: E402

PLANS = {k: SD / f"plans/s1rep_{k}.env" for k in ("one", "a", "b")}
STAGE0 = {"17": WT / "runs/selfdoubt/s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl",
          "8": WT / "runs/selfdoubt/s0c_pod2/heldout/full/Qwen3-8B/gen.jsonl"}
SETUP_S, PRICE = 200, 0.59
CAPS = {"one": 2.77, "a": 1.35, "b": 1.46}            # quoted hard caps at $0.59/h (disk on top)
SIZE = {("17", "math500"): 250, ("8", "gsm8k"): 250, ("8", "math500"): 100}


def plan(k: str) -> dict:
    out = {}
    for line in PLANS[k].read_text().splitlines():
        if line and not line.startswith("#"):
            a, v = line.split("=", 1)
            out[a] = v.strip('"')
    return out


def steps(p: dict) -> list[list[str]]:
    return [s.split("|") for s in p["STEPS"].split(";")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("k", ["one", "a", "b"])
def test_plan_files(k):
    p = plan(k)
    assert p["KEY_VAR"] == "RUNPOD_API_KEY_2" and p["EXPECT_EMAIL"] == "qamilmirza@gmail.com"
    assert p["OUT"] == p["OUTREL"] and p["POD_SCRIPT"] == "selfdoubt/stage0_full_pod.sh"
    assert p["RUNNER"] == "selfdoubt/stage1_run.py"
    assert p["EXTRA"].split() == ["--cut", "0.9", "--payload", "plain", "--sys", "s1", "--arms", "N,V0,VG,I1,I2b"]
    st = steps(p)
    assert all(s[3] == "heldout" and int(s[4]) == SIZE[(s[1], s[2])] for s in st)
    assert int(p["COPY_AT"]) == int(p["HARD_S"]) - 240
    assert int(p["HARD_S"]) >= 1.25 * (SETUP_S + sum(int(s[5]) for s in st)) - 1
    assert abs(int(p["HARD_S"]) * PRICE / 3600 - CAPS[k]) < 0.006
    sizes = [int(s[4]) for s in st]
    assert sizes[-1] == min(sizes)                    # the smallest cell last: a deadline hit costs it first
    fc = dict(x.split("|") for x in p["FULL_CHECK"].split(";"))
    assert sorted(fc) == sorted(p["BUNDLE_EXTRA"].split())
    tags = {s[1] for s in st}
    assert {("Qwen3-1.7B" if t == "17" else "Qwen3-8B") for t in tags} == {Path(x).parent.name for x in fc}
    assert ("ALLOW_SIBLING" in p) == (k == "b")       # B launches second, with A as its allowed sibling
    if k == "b":
        assert p["ALLOW_SIBLING"] == plan("a")["L"]


def test_two_pods_cover_the_one_pod_plan():
    one = {tuple(s[1:5]) for s in steps(plan("one"))}
    two = {tuple(s[1:5]) for s in steps(plan("a")) + steps(plan("b"))}
    assert one == two and len(one) == 3


@pytest.mark.parametrize("k", ["one", "a", "b"])
def test_bundled_stores_are_stage0_and_cover_every_item(k):
    p = plan(k)
    fc = dict(x.split("|") for x in p["FULL_CHECK"].split(";"))
    if not all((WT / x).exists() for x in fc) or not all(s.exists() for s in STAGE0.values()):
        pytest.skip("stores not on this machine")
    items = C.load_items()
    split, _ = C.ensure_split(items, WT / "runs/selfdoubt/split.json")
    for path, h in fc.items():
        t = "17" if "Qwen3-1.7B" in path else "8"
        assert sha(WT / path) == h == sha(STAGE0[t])
        model = "Qwen/Qwen3-1.7B" if t == "17" else "Qwen/Qwen3-8B"
        full = C.Store(WT / path).full_by_item(model)
        for s in steps(p):
            if s[1] == t:
                sel = C.select_items(items, split, "full", "heldout", int(s[4]), sources=(s[2],))
                assert len(sel) == int(s[4]) and {int(r["id"]) for r in sel} <= set(full)   # no FULL generated


@pytest.fixture()
def pod(tmp_path):
    wt = tmp_path / "wt"
    shutil.copytree(SD, wt / "selfdoubt", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (wt / "runs/selfdoubt").mkdir(parents=True)
    shutil.copy(WT / "runs/selfdoubt/split.json", wt / "runs/selfdoubt/split.json")
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls.txt\nexit 0\n'); runner.chmod(0o755)
    return {"wt": wt, "runner": runner, "calls": tmp_path / "calls.txt"}


def run_plan(pod, k):
    p = plan(k)
    run_s = int(p["HARD_S"]) - 40 - 240
    cmd = (f"set -a && . selfdoubt/plans/s1rep_{k}.env && set +a && ENGINE=plan PYTHON={pod['runner']} "
           f"RUN_S={run_s} bash selfdoubt/stage0_full_pod.sh")
    return subprocess.run(["bash", "-c", cmd], cwd=pod["wt"], env=dict(os.environ), capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("k", ["one", "a", "b"])
def test_plan_mode_with_the_real_plan(pod, k):
    p = plan(k)
    fc = dict(x.split("|") for x in p["FULL_CHECK"].split(";"))
    if not all((WT / x).exists() for x in fc):
        pytest.skip("bundled FULL stores not on this machine")
    for x in fc:
        (pod["wt"] / x).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(WT / x, pod["wt"] / x)
    r = run_plan(pod, k)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = pod["calls"].read_text().splitlines()
    assert len(calls) == len(steps(p))
    for call, s in zip(calls, steps(p)):              # in plan order
        model = "Qwen/Qwen3-1.7B" if s[1] == "17" else "Qwen/Qwen3-8B"
        assert call.startswith("-u selfdoubt/stage1_run.py --engine vllm --mode full --split-name heldout --allow-heldout")
        assert f"--out-root {p['OUT']}/heldout" in call and f"--model {model}" in call
        assert f"--source {s[2]} --limit {s[4]}" in call
        assert call.endswith("--cut 0.9 --payload plain --sys s1 --arms N,V0,VG,I1,I2b")
    log = (pod["wt"] / p["OUT"] / "pod.log").read_text()
    assert all(f"bundled store ok: {x}" in log for x in fc)


def test_plan_mode_refuses_a_tampered_store(pod):
    p = plan("one")
    fc = list(dict(x.split("|") for x in p["FULL_CHECK"].split(";")))
    if not all((WT / x).exists() for x in fc):
        pytest.skip("bundled FULL stores not on this machine")
    for x in fc:
        (pod["wt"] / x).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(WT / x, pod["wt"] / x)
    with open(pod["wt"] / fc[1], "a") as f:          # the second store (8B) changed by one line
        f.write('{"tampered": true}\n')
    r = run_plan(pod, "one")
    assert r.returncode == 3 and not pod["calls"].exists(), r.stdout + r.stderr


# ------------------------------------------------------------------ readings

def test_stratified_pool_keeps_stratum_sizes():
    # constant within each stratum: every stratified draw has the same mean, so the interval collapses on the point
    d = {"gsm8k": {i: 1.0 for i in range(30)}, "math500": {100 + i: 0.0 for i in range(10)}}
    r = REP.stratified(d, n_boot=200)
    assert r["n_items"] == 40 and r["n_by_source"] == {"gsm8k": 30, "math500": 10}
    assert r["point"] == 0.75 and r["ci"] == [0.75, 0.75]


def _cell(i1, i2b, n=40, src="gsm8k"):
    by = {i: {"I2b": i2b, "VG": 1, "V0": 1, "I1": i1} for i in range(n)}
    return {"model": "m", "source": src, "n_items_in_store": n, "arms_in_store": list(REP.ARMS),
            "n_items_not_dropped": n, "acceptance": {"V0": 1.0, "VG": 1.0, "I1": float(i1), "I2b": float(i2b)},
            "categories": {}, "r1b2_pairing": {"join": "item id", "s1_v0_items_kept": n, "stage0_ug_items_kept": n,
                                               "pairs": n},
            "paired_means": {},
            "diffs": {"R1.B1": {i: by[i]["I2b"] - 1 for i in by}, "R1.B2": {i: 0 for i in by},
                      "lift": {i: i2b - i1 for i in by}}}


def test_payload_failure_is_not_read_nor_pooled():
    out = REP.readings({("8B", "gsm8k"): _cell(1, 1), ("8B", "math500"): _cell(0, 0, src="math500")}, n_boot=50)
    bad = out["per_cell"]["8B|math500"]
    assert not bad["payload_sanity"]["pass"] and "R1.B1" not in bad and "payload failure" in bad["note"]
    pool = out["pooled"]["8B"]
    assert pool["sources_read"] == ["gsm8k"] and "not the frozen pool" in pool["note"]
    assert pool["R1.B1"]["n_items"] == 40 and pool["R1.B1"]["verdict"] == "SHOWN"


def test_cli_needs_confirmation():
    with pytest.raises(SystemExit):
        REP.main(["--layout", "two"])


@pytest.mark.skipif(not (REP.RUNS / REP.S1_GSM8K_17).exists(), reason="1.7B GSM8K S1 store absent")
def test_dry_check_reproduces_the_accepted_cell():
    """The 1.7B GSM8K cell (existing S1 store, c123a57) through the replication code: R1.B1 -0.038 [-0.068, -0.009]."""
    c = {("1.7B", "gsm8k"): REP.cell(REP.RUNS / REP.S1_GSM8K_17, REP.RUNS / REP.STAGE0[("1.7B", "gsm8k")],
                                     REP.MODELS["1.7B"], "gsm8k")}
    out = REP.readings(c)
    b1, b2 = out["per_cell"]["1.7B|gsm8k"]["R1.B1"], out["per_cell"]["1.7B|gsm8k"]["R1.B2"]
    assert [round(x, 3) for x in (b1["point"], *b1["ci"])] == [-0.038, -0.068, -0.009] and b1["verdict"] == "SHOWN"
    assert [round(x, 3) for x in (b2["point"], *b2["ci"])] == [-0.055, -0.094, -0.017] and b2["verdict"] == "SHOWN"
    assert b1["n_items"] == b2["n_items"] == 235
    assert out["per_cell"]["1.7B|gsm8k"]["payload_sanity"]["pass"]
