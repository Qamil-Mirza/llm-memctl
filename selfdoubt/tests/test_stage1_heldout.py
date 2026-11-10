"""Stage 1 held-out build (Advisor go 2026-11-09, $0): plan file, plan-mode pod run with the real plan, the bundled
FULL store check, the frozen held-out readings and their dev dry check.

    runs/selfdoubt/_venv457/bin/python -m pytest selfdoubt/tests/test_stage1_heldout.py -q
"""
from __future__ import annotations

import hashlib
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

from selfdoubt import stage1_report as SR  # noqa: E402

PLAN = SD / "plans/s1_heldout.env"
STAGE0_HELDOUT = WT / "runs/selfdoubt/full/Qwen3-1.7B/gen.jsonl"          # the store Stage 0 scored (GSM8K held-out)
DEV_S1T = WT / "runs/selfdoubt/s1s2v2_pod/s1/dev/stage1_s1t/Qwen3-1.7B/gen.jsonl"
DEV_S1 = WT / "runs/selfdoubt/s1s2_pod/s1/dev/stage1/Qwen3-1.7B/gen.jsonl"
DEV_S0 = WT / "runs/selfdoubt/s0c_pod1/dev/full/Qwen3-1.7B/gen.jsonl"


def plan() -> dict:
    out = {}
    for line in PLAN.read_text().splitlines():
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v.strip('"')
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_plan_file():
    p = plan()
    assert p["KEY_VAR"] == "RUNPOD_API_KEY_2" and p["EXPECT_EMAIL"] == "qamilmirza@gmail.com"
    assert "ALLOW_SIBLING" not in p                                     # 0 pods on the account before create
    assert p["OUT"] == p["OUTREL"] and p["POD_SCRIPT"] == "selfdoubt/stage0_full_pod.sh"
    assert p["RUNNER"] == "selfdoubt/stage1_run.py"
    assert p["EXTRA"].split() == ["--cut", "0.9", "--payload", "plain", "--sys", "s1t,s1"]   # primary S1t first
    (step,) = [s.split("|") for s in p["STEPS"].split(";")]
    assert step[1:5] == ["17", "gsm8k", "heldout", "250"]
    assert int(p["COPY_AT"]) == int(p["HARD_S"]) - 240
    setup_s = 180                                                       # measured on wp1txc49i5wg8r: 37 + 66 + ~70
    assert int(p["HARD_S"]) >= 1.25 * (setup_s + int(step[5])) - 1
    assert int(p["HARD_S"]) * float(p["PRICE_H"]) / 3600 <= 0.85        # the quoted cap
    path, h = p["FULL_CHECK"].split("|")
    assert path == p["BUNDLE_EXTRA"] == f"{p['OUT']}/heldout/full/Qwen3-1.7B/gen.jsonl"


def test_bundled_full_store_is_the_store_stage0_scored():
    p = plan()
    path, h = p["FULL_CHECK"].split("|")
    if not (WT / path).exists() or not STAGE0_HELDOUT.exists():
        pytest.skip("stores not on this machine")
    assert sha(WT / path) == h == sha(STAGE0_HELDOUT)
    ho = set(json.loads((WT / "runs/selfdoubt/split.json").read_text())["heldout"]["gsm8k"])
    full = {r["item_id"] for r in map(json.loads, (WT / path).read_text().splitlines())
            if r["arm"] == "FULL" and r["source"] == "gsm8k"}
    assert full == ho and len(full) == 250                              # every item has its FULL: none generated


@pytest.fixture()
def pod(tmp_path):
    wt = tmp_path / "wt"
    shutil.copytree(SD, wt / "selfdoubt", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (wt / "runs/selfdoubt").mkdir(parents=True)
    shutil.copy(WT / "runs/selfdoubt/split.json", wt / "runs/selfdoubt/split.json")
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls.txt\nexit 0\n'); runner.chmod(0o755)
    return {"wt": wt, "runner": runner, "calls": tmp_path / "calls.txt"}


def run_plan(pod, full_check=None):
    """The real plan, sourced as the pod does (set -a), in plan mode; RUN_S as the launcher would set it."""
    p = plan()
    run_s = int(p["HARD_S"]) - 40 - 240
    fc = f"FULL_CHECK='{full_check}' " if full_check is not None else ""
    cmd = (f"set -a && . selfdoubt/plans/s1_heldout.env && set +a && {fc}ENGINE=plan PYTHON={pod['runner']} "
           f"RUN_S={run_s} bash selfdoubt/stage0_full_pod.sh")
    return subprocess.run(["bash", "-c", cmd], cwd=pod["wt"], env=dict(os.environ), capture_output=True, text=True,
                          timeout=120)


def test_plan_mode_with_the_real_plan(pod):
    p = plan()
    path, _ = p["FULL_CHECK"].split("|")
    if not (WT / path).exists():
        pytest.skip("bundled FULL store not on this machine")
    (pod["wt"] / path).parent.mkdir(parents=True)
    shutil.copy(WT / path, pod["wt"] / path)
    r = run_plan(pod)
    assert r.returncode == 0, r.stdout + r.stderr
    (call,) = pod["calls"].read_text().splitlines()
    assert call.startswith("-u selfdoubt/stage1_run.py --engine vllm --mode full --split-name heldout --allow-heldout")
    assert "--out-root runs/selfdoubt/s1ho_pod/heldout" in call and "--model Qwen/Qwen3-1.7B" in call
    assert "--source gsm8k --limit 250" in call and call.endswith("--cut 0.9 --payload plain --sys s1t,s1")
    log = (pod["wt"] / "runs/selfdoubt/s1ho_pod/pod.log").read_text()
    assert f"bundled store ok: {path}" in log


def test_plan_mode_refuses_a_different_full_store(pod):
    path = "runs/selfdoubt/s1ho_pod/heldout/full/Qwen3-1.7B/gen.jsonl"
    (pod["wt"] / path).parent.mkdir(parents=True)
    (pod["wt"] / path).write_text('{"tampered": true}\n')
    r = run_plan(pod)                                                   # the plan's sha, a different file
    assert r.returncode == 3 and not pod["calls"].exists(), r.stdout + r.stderr
    (pod["wt"] / path).unlink()
    r = run_plan(pod)                                                   # missing
    assert r.returncode == 3 and not pod["calls"].exists()
    r = run_plan(pod, full_check="")                                    # no check asked: runs
    assert r.returncode == 0 and pod["calls"].exists()


def test_heldout_cli_needs_confirmation(tmp_path):
    with pytest.raises(SystemExit):
        SR.main(["heldout", "--store-s1t", "x", "--store-s1", "y", "--stage0-store", "z"])


@pytest.mark.skipif(not (DEV_S1T.exists() and DEV_S1.exists() and DEV_S0.exists()), reason="dev stores absent")
def test_dry_check_on_dev_reproduces_the_dev_numbers(monkeypatch):
    """Dev stores through the held-out readings: S1t selectivity = the v2 dev reading; R1.B1 / R1.B2 = the dev numbers
    the rules were formed on (I2b 0.809 vs VG 0.817; V0 under S1 0.776 vs Stage 0 UG 0.780)."""
    out = SR.heldout(DEV_S1T, DEV_S1, DEV_S0, "dev", 2000)
    sel = out["primary_s1t"]["selectivity"]
    assert sel["verdict"] == "NOT SELECTIVE" and abs(sel["point"] - 0.5265) < 1e-3
    b = out["s1_r1b"]
    assert abs(b["R1.B1"]["point"] - (-0.0083)) < 1e-3 and b["R1.B1"]["verdict"] == "SHOWN"
    assert abs(b["R1.B2"]["point"] - (-0.0041)) < 1e-3 and b["R1.B2"]["verdict"] == "SHOWN"
    assert b["r1b2_pairing"]["pairs"] == b["R1.B2"]["n_items"] == 241 and not b["r1b2_pairing"]["below_min"]
    monkeypatch.setattr(SR, "MIN_B2_PAIRS", 300)                        # fewer pairs than the floor: reported
    b = SR.r1b(DEV_S1, DEV_S0, "dev", 200)
    assert b["r1b2_pairing"]["below_min"] and "241 R1.B2 pairs (< 300)" in b["r1b2_pairing"]["note"]
    assert b["R1.B2"]["n_items"] == 241                                 # the reading itself is unchanged
