"""scripts/pod_watch.sh waits for the chain sentinel and never searches for a process name."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_watch.sh"
POD_SCRIPTS = [Path(__file__).resolve().parents[1] / "scripts" / n for n in ("exp34_pod.sh", "exp33_pod.sh")]


def run_watch(done: Path, log: Path | None, interval="1", max_polls="5", ssh="bash -c"):
    env = dict(os.environ, POD_SSH=ssh)
    return subprocess.run(["bash", str(SCRIPT), "h", "1", str(done), str(log or ""), interval, max_polls],
                          capture_output=True, text=True, env=env, timeout=60)


def test_exits_zero_when_sentinel_appears(tmp_path):
    done, log = tmp_path / "exp34_pod.done", tmp_path / "main.log"
    log.write_text("stage install: elapsed 29 s\n")
    threading.Timer(2.0, lambda: done.write_text("exit=0 at T\n")).start()
    result = run_watch(done, log)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHAIN_DONE" in result.stdout and "exit=0 at T" in result.stdout
    assert "stage install" in result.stdout


def test_times_out_without_sentinel(tmp_path):
    result = run_watch(tmp_path / "never.done", None, interval="0", max_polls="3")
    assert result.returncode == 4 and "WATCH_TIMEOUT" in result.stdout


def test_unreachable_pod_exits_3(tmp_path):
    result = run_watch(tmp_path / "x.done", None, interval="0", max_polls="10", ssh="false")
    assert result.returncode == 3 and "POD_UNREACHABLE" in result.stdout


def test_no_process_search_in_watcher():
    text = SCRIPT.read_text()
    assert "pgrep" not in text and "ps -" not in text


def test_pod_scripts_write_sentinel_on_exit():
    for script in POD_SCRIPTS:
        text = script.read_text()
        assert "DONE_FILE=runs/" in text and "trap 'rc=$?" in text, script.name
