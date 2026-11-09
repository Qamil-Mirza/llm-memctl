"""Second-account plumbing: KEY_VAR selection, pre-flight account check, one key per run (launcher, timer, waiter).

All RunPod calls go to a fake curl on PATH (no network, $0); ssh/scp are fakes that fail.
    .venv/bin/python -m pytest selfdoubt/tests/test_pod_key.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SD = Path(__file__).resolve().parents[1]
KEY1 = "rpa_TESTKEYONE_berkeley_000000000000"
KEY2 = "rpa_TESTKEYTWO_gmail_1111111111111111"

FAKE_CURL = r'''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
fake = os.environ["FAKE"]
open(os.path.join(fake, "argv.log"), "a").write(json.dumps(a) + "\n")
method, auth, url, i = "GET", None, None, 0
while i < len(a):
    x = a[i]
    if x == "-X": method = a[i + 1]; i += 2; continue
    if x == "-H" and a[i + 1].startswith("@"):
        auth = open(a[i + 1][1:]).read().strip(); i += 2; continue
    if x in ("-H", "-d"):
        if x == "-d": method = "POST" if method == "GET" else method
        i += 2; continue
    if x.startswith("http"): url = x
    i += 1
open(os.path.join(fake, "calls.jsonl"), "a").write(json.dumps({"method": method, "url": url, "auth": auth}) + "\n")
def out(name, default):
    p = os.path.join(fake, name)
    sys.stdout.write(open(p).read() if os.path.exists(p) else default)
if "graphql" in url: out("gql.json", "{}")
elif method == "DELETE": out("delete.json", "")
elif method == "POST": out("create.json", "{}")
elif url.endswith("/pods"): out("pods.json", "[]")
else: out("podget.json", "{}")
'''


@pytest.fixture()
def env(tmp_path):
    fake = tmp_path / "fake"; fake.mkdir()
    bindir = tmp_path / "bin"; bindir.mkdir()
    (bindir / "curl").write_text(FAKE_CURL)
    for tool in ("ssh", "scp"):
        (bindir / tool).write_text("#!/bin/sh\nexit 1\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    envf = tmp_path / "dotenv"
    envf.write_text(f"RUNPOD_API_KEY={KEY1}\nexport RUNPOD_API_KEY_2='{KEY2}'\n")
    wt = tmp_path / "wt"
    shutil.copytree(SD, wt / "selfdoubt", ignore=shutil.ignore_patterns("__pycache__", "tests", "vendor"))
    (wt / "runs" / "selfdoubt").mkdir(parents=True)
    for f in ("split.json", "split.sha256"):
        (wt / "runs" / "selfdoubt" / f).write_text("{}")
    e = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FAKE=str(fake), ENV_FILE=str(envf), WT=str(wt),
             MAIN=str(tmp_path / "main"))
    e.pop("KEY_VAR", None); e.pop("EXPECT_EMAIL", None)
    return {"fake": fake, "env": e, "wt": wt}


def gql(fake, email, bal):
    (fake / "gql.json").write_text(json.dumps({"data": {"myself": {"email": email, "clientBalance": bal}}}))


def calls(fake):
    p = fake / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def preflight(env, **kw):
    e = dict(env["env"], **kw)
    return subprocess.run(["bash", str(env["wt"] / "selfdoubt" / "pod_preflight.sh")], env=e, capture_output=True,
                          text=True)


def no_key_leak(env, *outs):
    argv = (env["fake"] / "argv.log").read_text() if (env["fake"] / "argv.log").exists() else ""
    for k in (KEY1, KEY2):
        assert k not in argv, "key on a curl command line"
        for o in outs:
            assert k not in o, "key printed"


def test_preflight_ok_uses_chosen_key(env):
    gql(env["fake"], "qamilmirza@gmail.com", 25.0)
    r = preflight(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com", CAP_USD="1.5")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "qamilmirza@gmail.com, balance $25.0000" in r.stdout and "pre-flight OK" in r.stdout
    cs = calls(env["fake"])
    assert len(cs) == 2 and all(c["auth"] == f"Authorization: Bearer {KEY2}" for c in cs)
    no_key_leak(env, r.stdout, r.stderr)


def test_default_key_var_is_berkeley(env):
    gql(env["fake"], "qamilmirza@berkeley.edu", 3.59)
    r = preflight(env, EXPECT_EMAIL="qamilmirza@berkeley.edu", CAP_USD="0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(c["auth"] == f"Authorization: Bearer {KEY1}" for c in calls(env["fake"]))


@pytest.mark.parametrize("email,bal,pods,msg", [
    ("qamilmirza@berkeley.edu", 25.0, [], "expected qamilmirza@gmail.com"),
    ("qamilmirza@gmail.com", 1.0, [], "below cap"),
    ("qamilmirza@gmail.com", 25.0, [{"id": "abc123", "name": "x", "desiredStatus": "RUNNING"}], "already on this account"),
])
def test_preflight_aborts(env, email, bal, pods, msg):
    gql(env["fake"], email, bal)
    (env["fake"] / "pods.json").write_text(json.dumps(pods))
    r = preflight(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com", CAP_USD="1.5")
    assert r.returncode != 0 and "ABORT" in r.stdout and msg in r.stdout
    if pods:
        assert "pod abc123" in r.stdout
    no_key_leak(env, r.stdout, r.stderr)


def test_preflight_aborts_on_bad_auth_reply(env):
    (env["fake"] / "gql.json").write_text('{"errors": [{"message": "unauthorized"}]}')
    r = preflight(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com")
    assert r.returncode != 0 and "account query failed" in r.stdout


@pytest.mark.parametrize("var", ["RUNPOD_API_KEY_3", "PATH", "RUNPOD_API_KEY_2; rm"])
def test_bad_or_missing_key_var(env, var):
    r = preflight(env, KEY_VAR=var, EXPECT_EMAIL="qamilmirza@gmail.com")
    assert r.returncode != 0 and "ABORT" in r.stderr
    assert calls(env["fake"]) == []


def test_missing_expect_email(env):
    gql(env["fake"], "qamilmirza@gmail.com", 25.0)
    r = preflight(env, KEY_VAR="RUNPOD_API_KEY_2")
    assert r.returncode != 0 and "EXPECT_EMAIL" in r.stderr


def launch(env, **kw):
    e = dict(env["env"], **{"L": "runs/selfdoubt/t_launch", "OUTREL": "runs/selfdoubt/t_out", "HARD_S": "60",
                            "NOSSH_S": "0", **kw})
    return subprocess.run(["bash", str(env["wt"] / "selfdoubt" / "pod_launch.sh")], env=e, capture_output=True,
                          text=True, timeout=60)


def test_launch_never_creates_when_preflight_fails(env):
    gql(env["fake"], "qamilmirza@berkeley.edu", 25.0)
    r = launch(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com")
    assert r.returncode == 2 and "pre-flight failed" in r.stdout
    assert not any(c["method"] == "POST" and "graphql" not in c["url"] for c in calls(env["fake"]))
    assert not (env["wt"] / "runs/selfdoubt/t_launch/auth.hdr").exists()
    no_key_leak(env, r.stdout, r.stderr)


def test_launch_cap_is_hard_s_times_price(env):
    gql(env["fake"], "qamilmirza@gmail.com", 0.50)
    r = launch(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com", HARD_S="3600")
    assert r.returncode == 2 and "cap $0.590" in r.stdout and "below cap" in r.stdout


def test_launch_and_wait_use_one_account(env):
    fake, wt = env["fake"], env["wt"]
    gql(fake, "qamilmirza@gmail.com", 25.0)
    (fake / "create.json").write_text(json.dumps({"id": "podX", "costPerHr": 0.59}))
    r = launch(env, KEY_VAR="RUNPOD_API_KEY_2", EXPECT_EMAIL="qamilmirza@gmail.com")
    assert r.returncode == 4 and "NO SSH MAPPING" in r.stdout, r.stdout + r.stderr   # deleted at once, as designed
    L = wt / "runs/selfdoubt/t_launch"
    pod = json.loads((L / "pod.json").read_text())
    assert pod["key_var"] == "RUNPOD_API_KEY_2" and pod["email"] == "qamilmirza@gmail.com"
    hdr = L / "auth.hdr"
    assert hdr.exists() and oct(hdr.stat().st_mode & 0o777) == "0o600"
    # the waiter, started with a different KEY_VAR in its environment, must still use the launcher's account
    (L / "ssh.txt").write_text("127.0.0.1 1\n")
    hdr.unlink()                                       # force the waiter to rebuild the header from pod.json's key_var
    n_before = len(calls(fake))
    w = subprocess.run(["bash", str(wt / "selfdoubt" / "pod_wait.sh")],
                       env=dict(env["env"], L="runs/selfdoubt/t_launch", COPY_AT="0", KEY_VAR="RUNPOD_API_KEY"),
                       capture_output=True, text=True, timeout=60)
    assert "WAIT DONE pods_after=0" in w.stdout, w.stdout + w.stderr
    cs = calls(fake)
    assert all(c["auth"] == f"Authorization: Bearer {KEY2}" for c in cs), [c["url"] for c in cs]
    assert any(c["method"] == "DELETE" for c in cs[:n_before]) and any(c["method"] == "DELETE" for c in cs[n_before:])
    assert not hdr.exists()
    assert "(RUNPOD_API_KEY_2)" in (L / "wait.log").read_text()
    no_key_leak(env, r.stdout, r.stderr, w.stdout, w.stderr)
    # killing the timer kills its whole group: no orphaned sleep from this test is left
    import glob, time
    time.sleep(0.5)
    left = []
    for p in glob.glob("/proc/[0-9]*/environ"):
        try:
            if f"FAKE={fake}".encode() in open(p, "rb").read():
                left.append(p)
        except OSError:
            pass
    assert left == [], left
