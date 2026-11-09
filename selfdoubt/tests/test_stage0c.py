"""Stage 0 completion build (2026-11-09): store seeding, STEPS pod script (plan mode, $0), the two pod plan files.

    .venv/bin/python -m pytest selfdoubt/tests/test_stage0c.py -q
"""
from __future__ import annotations

import json
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
from selfdoubt import stage0_seed_store as SEED  # noqa: E402

ROWS = [(a, c) for c in C.CUTS for a in ("N", "U0", "UG")] + [(a, C.X0) for a in ("U0", "UG")]


def rec(i, arm, cut, src="math500"):
    return json.dumps({"key": f"{i}|{arm}|{cut}", "item_id": i, "source": src, "arm": arm, "cut": cut,
                       "text": f"T{i}{arm}{cut}"})


def test_seed_store_drops_all_rows_of_incomplete_items(tmp_path):
    lines = []
    for i in (1, 2, 3):
        lines.append(rec(i, "FULL", None))
    lines += [rec(1, a, c) for a, c in ROWS]                    # complete
    lines += [rec(2, a, c) for a, c in ROWS[:6]]                # 6 of 11: incomplete
    lines += [rec(3, a, c) for a, c in ROWS[1:]]                # 10 of 11: incomplete
    lines += [rec(9, "FULL", None, "gsm8k"), rec(9, "N", 0.3, "gsm8k")]
    src = tmp_path / "gen.jsonl"; src.write_text("\n".join(lines) + "\n")
    dst = tmp_path / "out" / "gen.jsonl"
    out = SEED.seed(src, dst, "math500")
    assert out["incomplete"] == [2, 3] and out["n_dropped_rows"] == 16 and out["n_complete"] == 1
    kept = dst.read_text().splitlines()
    assert len(kept) == 3 + 11
    assert all(json.loads(x)["source"] == "math500" for x in kept)
    assert set(kept) <= set(lines)                               # byte-for-byte copies
    with pytest.raises(SystemExit):
        SEED.seed(src, dst, "math500")                           # never overwrites


def test_seed_store_expectations_abort(tmp_path):
    src = tmp_path / "gen.jsonl"
    src.write_text("\n".join([rec(1, "FULL", None)] + [rec(1, a, c) for a, c in ROWS[:3]]) + "\n")
    with pytest.raises(SystemExit):
        SEED.main(["--src", str(src), "--dst", str(tmp_path / "d.jsonl"), "--source", "math500", "--expect-drop", "5"])
    assert not (tmp_path / "d.jsonl").exists()


def test_seeded_store_resume_regenerates_exactly_the_dropped_rows(tmp_path):
    """The real pod-1 store (if built): every complete item has 11 rows, no incomplete item has any."""
    p = WT / "runs/selfdoubt/s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl"
    if not p.exists():
        pytest.skip("pod-1 store not seeded here")
    full, rows = set(), {}
    for x in p.read_text().splitlines():
        r = json.loads(x)
        assert r["source"] == "math500"
        if r["arm"] == "FULL":
            full.add(r["item_id"])
        else:
            rows.setdefault(r["item_id"], set()).add((r["arm"], r["cut"]))
    assert len(full) == 100 and len(rows) == 81 and all(v == set(ROWS) for v in rows.values())
    rd = WT / "runs/selfdoubt/full/heldout_1p7b_readings.json"
    if rd.exists():
        inc = {int(i) for i in json.loads(rd.read_text())["math500"]["incomplete"]}
        assert inc == full - set(rows)


@pytest.fixture()
def pod(tmp_path):
    wt = tmp_path / "wt"
    shutil.copytree(SD, wt / "selfdoubt", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (wt / "runs/selfdoubt").mkdir(parents=True)
    shutil.copy(WT / "runs/selfdoubt/split.json", wt / "runs/selfdoubt/split.json")
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls.txt\nexit 0\n'); runner.chmod(0o755)
    return {"wt": wt, "runner": runner, "calls": tmp_path / "calls.txt"}


def run_pod(pod, steps, run_s="1000", out="runs/selfdoubt/t"):
    e = dict(os.environ, ENGINE="plan", PYTHON=str(pod["runner"]), STEPS=steps, RUN_S=run_s, OUT=out)
    return subprocess.run(["bash", str(pod["wt"] / "selfdoubt/stage0_full_pod.sh")], env=e, capture_output=True,
                          text=True, timeout=60)


def test_pod_steps_args_and_ceiling_gate(pod):
    r = run_pod(pod, "a_dev|17|gsm8k|dev|250|100;b_big|8|math500|heldout|100|5000;c_ho|8|math500|heldout||100")
    assert r.returncode == 8, r.stdout + r.stderr                # b skipped -> rc 8, but c still ran
    calls = pod["calls"].read_text().splitlines()
    assert len(calls) == 2
    a, c = calls
    assert "--split-name dev" in a and "--allow-heldout" not in a and "--out-root runs/selfdoubt/t/dev" in a
    assert "--model Qwen/Qwen3-1.7B" in a and "--snapshot PLAN17" in a and "--limit 250" in a and "--source gsm8k" in a
    assert "--split-name heldout --allow-heldout" in c and "--out-root runs/selfdoubt/t/heldout" in c
    assert "--model Qwen/Qwen3-8B" in c and "--limit" not in c and "--snapshot PLAN8" in c
    log = (pod["wt"] / "runs/selfdoubt/t/pod.log").read_text()
    assert "skipping b_big" in log and "< its ceiling 5000 s" in log
    for sp in ("dev", "heldout"):
        assert (pod["wt"] / f"runs/selfdoubt/t/{sp}/split.json").read_bytes() == \
            (WT / "runs/selfdoubt/split.json").read_bytes()


@pytest.mark.parametrize("steps", ["", "x|17|gsm8k|dev|10", "x|9|gsm8k|dev|10|5", "x|17|aime|dev|10|5",
                                   "x|17|gsm8k|test|10|5", "x y|17|gsm8k|dev|10|5", "x|17|gsm8k|dev|ten|5"])
def test_pod_rejects_bad_steps(pod, steps):
    r = run_pod(pod, steps)
    assert r.returncode == 3 and not pod["calls"].exists()


def plan(name):
    out = {}
    for line in (SD / "plans" / name).read_text().splitlines():
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v.strip('"')
    return out


@pytest.mark.parametrize("name,setup_s", [("s0c_pod1.env", 400), ("s0c_pod2.env", 400)])
def test_plan_files(name, setup_s):
    p = plan(name)
    assert p["KEY_VAR"] == "RUNPOD_API_KEY_2" and p["EXPECT_EMAIL"] == "qamilmirza@gmail.com"
    assert p["OUT"] == p["OUTREL"] and p["POD_SCRIPT"] == "selfdoubt/stage0_full_pod.sh"
    assert int(p["COPY_AT"]) == int(p["HARD_S"]) - 240
    steps = [s.split("|") for s in p["STEPS"].split(";")]
    ceiling = setup_s + sum(int(s[5]) for s in steps)
    assert int(p["HARD_S"]) >= 1.25 * ceiling - 1, "hard stop must leave >= 25% over the ceiling"
    other = {"s0c_pod1.env": "s0c_pod2", "s0c_pod2.env": "s0c_pod1"}[name]
    assert p["ALLOW_SIBLING"] == f"runs/selfdoubt/{other}_launch"


def test_plan_pod1_holds_dev_step_first_and_the_seeded_store():
    p = plan("s0c_pod1.env")
    steps = [s.split("|") for s in p["STEPS"].split(";")]
    assert [s[1:5] for s in steps] == [["17", "gsm8k", "dev", "250"], ["17", "math500", "heldout", "250"]]
    assert p["BUNDLE_EXTRA"] == "runs/selfdoubt/s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl"


def test_plan_pod2_steps():
    steps = [s.split("|") for s in plan("s0c_pod2.env")["STEPS"].split(";")]
    assert [s[1:5] for s in steps] == [["8", "gsm8k", "heldout", "250"], ["8", "math500", "heldout", "100"]]
