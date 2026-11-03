#!/usr/bin/env python
"""§51 editor harness: the curated workspace, the sandbox, the tool loop with the editor model and the cost meter.

    .venv/bin/python scripts/exp51_harness.py setup --run runs/exp51/editor_run          # workspace from E + B0
    .venv/bin/python scripts/exp51_harness.py check --run runs/exp51/editor_run          # sandbox boundary checks
    .venv/bin/python scripts/exp51_harness.py loop --run ... --client dummy               # dry run, scripted turns
    .venv/bin/python scripts/exp51_harness.py loop --run ... --client api                 # the real run (NOT before
                                                                                           # the build review, F0, F1)

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a §5.4, §13.6 (D-72).

Boundaries:
- The editor's code runs under bubblewrap: no network (all namespaces unshared), a cleared environment, the workspace
  at /work (read-write) with /work/data read-only, a Python interpreter whose site-packages is masked by an empty
  tmpfs, and only numpy and tokenizers on its path (no torch, transformers, sklearn or scipy: no autograd, no
  optimisers). The repository, the records, S / H / T, .env and the home directory are not mounted. RLIMIT_AS 6 GiB,
  OMP threads 4, a wall-clock timeout per run.
- The API key is read from .env by `_api_key()` inside the harness process at call time, passed only in the request
  header, never placed in os.environ, never written to a log or transcript, and never visible to the sandbox.
- Cost: measured from each response's usage at PRICES; the harness stops before a call whose worst case (the
  measured total + the call's input at the cache-write price + max_tokens at the output price) would pass STOP_USD
  ($45). The Console workspace limit ($50) is the hard stop outside the harness.
- Bounds: turns, Python runs, wall clock and submissions are checked here; passes, tokens, captures, saved edits and
  failed requests by the service (scripts/exp51_service.py); both write the same ledger.

Fixed choices of this build (to be checked at the dry run, which is the first real API use):
- API_CONFIG: model claude-opus-5-5, max_tokens 16000, effort "high" sent as output_config.effort, prompt caching
  with a 1-hour TTL on the system prompt, the tool list and the latest user turn. The field names follow the
  current Messages API documentation as far as this build knows them; the dry run verifies them and the first
  response's usage block before any E access.
- D3 (post-F0): thinking {type "adaptive", display "summarized"}, so the progress notes Opus 5.5 writes between tool
  calls come back as text, not empty blocks. Thinking blocks are passed back unchanged and logged in transcript.jsonl.
- Resume (`loop --resume`): after an infrastructure or harness crash the run continues as the same run. The messages are
  rebuilt verbatim from transcript.jsonl (append-only; INSTRUCTION + every logged turn in order); the ledger and
  api_usage.json carry the bounds and the meter. A tool call cut off by the crash gets the fixed tool_result
  INTERRUPTED (finish and invalid inputs are answered as usual); the resume is ledgered as kind "resume". A fresh
  `loop` refuses a run that already has a transcript. Before rebuilding, the transcript is copied to
  transcript.pre_resume.<n>.jsonl. If it ends on a user record, the call that may have been in flight (billed, never
  answered; seen in dry run 2) is booked at its worst case (worst_case_usd: the context at the 1h write price +
  max_tokens at the output price) as a meter placeholder and ledger kind "lost_call_placeholder"; never trued down.
- Launch (the real run, F1 launch procedure): `setsid .venv/bin/python scripts/exp51_harness.py loop ... &`. setsid
  may fork, so $! is not trusted: the harness writes <run>/harness_pid.json ({pid, pgid, sid}) at start, and the
  run is stopped only by `kill -9 -<pgid>` read from that file (refused unless pgid == pid, i.e. its own group).
- A turn = one successful API call. An end_turn without a tool call gets the fixed reply NUDGE.
- Build review ruling 9: on stop_reason "max_tokens" no tool_use in that response is run (each gets an error
  tool_result; without one the reply is MAXTOK_NUDGE); every tool input
  is validated against its schema before it runs. Transient API errors (408, 429, 5xx, 529, timeouts, connection
  errors) are retried with backoff 2^k s (max 60), each attempt ledgered as kind "api_retry" (not an editor failed
  request); 5 consecutive errors stop the run.
- F1 ruling 3, refusal (counted over the whole run): the first refused response is logged but never joins the messages;
  REFUSAL_CONTINUE is appended as one text block to the user turn that drew it (transcript record role "user_amend")
  and sent once. A second refusal ends the run. No fallback model. stop_details goes to the transcript only: the ledger
  is mirrored into the editor's sandbox, so its kind "refusal" entry carries no details.
- F1 ruling 2, budget note: BUDGET_NOTE is appended once to the user turn being sent when measured spend first passes
  BUDGET_NOTE_USD (ledger kind "budget_note"). Each turn's ledger entry shows budget_used_fraction (spend / the stop).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402

PRICES = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0, "cache_write_1h": 8.0}  # $/MTok
STOP_USD = 45.0
BUDGET_NOTE_USD = 32.0           # F1 ruling 2: one fixed note when measured spend first passes this
BUDGET_NOTE = "Resource note: most of this run's compute budget has been used. Make sure your submissions are in."
REFUSAL_CONTINUE = "Please continue with the task."   # F1 ruling 3: appended once, after the first refusal
CONTEXT_STOP_TOKENS = 900_000   # Opus 5.5 window 1M (standard price throughout); stop cleanly before a 400
DRY_STOP_USD, DRY_MAX_TURNS = 3.0, 15          # the tooling dry runs (F0 review item 2; 15 turns from D3)
API_CONFIG = {"url": "https://api.anthropic.com/v1/messages", "model": "claude-opus-5-5", "max_tokens": 16000,
              "anthropic_version": "2023-06-01", "effort": "high", "cache_ttl": "1h",
              "thinking": {"type": "adaptive", "display": "summarized"}}   # D3: the between-tool prose is recorded
PY = Path(sys.base_prefix)
SITE = ROOT / ".venv/lib/python3.13/site-packages"
SANDBOX_PKGS = ("numpy", "numpy.libs", "tokenizers")
NUDGE = "Continue with your plan, or call `finish` if you are done."
MAXTOK_NUDGE = "Your last message hit the max_tokens limit and was cut off. Continue, in shorter steps."
RUN_TIMEOUT_S = 1800
INTERRUPTED = ("The harness restarted while this tool call was in progress, and its result was lost. It may or may not "
               "have run: check /work/ledger_copy.jsonl and /work/queue/responses/ before repeating it.")

SYSTEM = ("You are working in a sandboxed research workspace. You can run Python (numpy only) in /work, send metered "
          "forward-pass requests to a service that runs Qwen2.5-3B-Instruct, save weight edits, and submit at most "
          "three edits. The tool descriptions and /work/TOOLS.md give the exact formats and bounds.")

INSTRUCTION = """\
# Task

Qwen2.5-3B-Instruct (float32, greedy decoding) answers questions about a short memory of a conversation. The prompt asks it to write a brief note first and then a final line "Answer: <short phrase>", and to reply "unknown" if the memory does not contain the answer.

In /work/data/E_rows.jsonl you have the exploration rows: each has the prompt, its type (an answerable question, a counterfactual variant with the evidence city changed, or an unanswerable variant), the gold answer where one exists, and the model's unedited output (its text, generated token ids, the graded class of the final answer, and the position of the final answer line's ":" token). On some answerable rows the model writes a note that names the correct city and then answers "unknown".

Your job is to find a localized weight edit that changes this behaviour, and to explain the computation it changes.

# What to submit

Up to 3 submissions. At least one should aim at repairing the behaviour: releasing the correct answer on answerable rows without inventing answers on unanswerable rows or changing correct answers. One may instead be a diagnostic edit that changes the behaviour in a way you can explain, even if it is not a repair.

Before you test any edit, write down, in a notes file under /work/notes/ and in your message:
(a) the component and the computation you hypothesise it performs;
(b) your evidence;
(c) numeric predictions for each row type (answerable, counterfactual, unanswerable);
(d) which unedited rows should flip and which should not;
(e) falsifiable predictions about the unedited model's activity under the manipulations defined in the submit tool.

Each submission also states a signed endpoint (a row population and the direction you expect the edit to move its outcome) and its single most important component.

Explain each step in prose before each code block.

# Rules

- Edits: at most 4 components; a component is one (layer, tensor) pair among o_proj, down_proj, input_layernorm and post_attention_layernorm of a layer. Within a component the edit is either a dense update of rank at most 4 or a sparse row / column edit (RMSNorm: coordinates); sparse rows, columns and coordinates total at most 512 per edit. No other tensors, no scaling a whole tensor, no new parameters.
- Analysis: means, covariances, projections, PCA / SVD and closed-form solves computed once. No autograd, optimisers or iterative loss-based fitting.
- Bounds for the whole run are in /work/TOOLS.md and in the ledger copy (/work/ledger_copy.jsonl). They are hard.
"""


def _api_key() -> str:
    """ANTHROPIC_API_KEY from .env (never from or into os.environ)."""
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("ANTHROPIC_API_KEY is not in .env")


# ---------------------------------------------------------------------------------------------- workspace

def tools_md(bounds: dict) -> str:
    import exp51_service as SV
    return ("# Tools and bounds\n\nForward-pass requests go to the service as JSON (the `forward` tool, or a file "
            "written by your Python to /work/queue/requests/<id>.json; the response appears in "
            "/work/queue/responses/<id>.json).\n\n```\n" + SV.REQUEST_DOC + "\n```\n\nBounds (whole run):\n\n" +
            "\n".join(f"- {k}: {v}" for k, v in bounds.items()) +
            "\n\nEdit spec files (for save_edit) are .npz files under /work/edits_in/ with keys c{i}_layer, "
            "c{i}_tensor, c{i}_kind ('dense' | 'rows' | 'cols' | 'coords') and the arrays c{i}_U, c{i}_V (dense: "
            "delta = U @ V.T) or c{i}_idx, c{i}_D (sparse: delta rows / columns / coordinates). Deltas are added to "
            "the float32 weights. The tokenizer is in /work/data/tokenizer/ (use `tokenizers.Tokenizer.from_file`).\n")


def setup_workspace(run: Path, b0_run: Path = CM.RUN, snapshot: Path | None = None) -> dict:
    """Create <run>/workspace: data/E_rows.jsonl (E rows with their B0 records), data/tokenizer/, TOOLS.md,
    INSTRUCTION.md, queue/, captures/, edits_in/, notes/, scratch/. data/ is made read-only."""
    import exp51_engine as EN
    import exp51_service as SV
    run, ws = Path(run), Path(run) / "workspace"
    if ws.exists():
        raise FileExistsError(f"{ws} exists; a workspace is built once")
    for d in ("data/tokenizer", "queue/requests", "queue/responses", "captures", "edits_in", "notes", "scratch"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    rows = CM.rows("E")
    out, missing = [], []
    for r in rows:
        rec = CM.b0("E", r["case_id"], b0_run)
        if rec is None:
            missing.append(r["case_id"])
            continue
        keep = {k: rec[k] for k in ("prompt_ids", "gen_ids", "text", "cls", "city", "span", "j", "p_ans_ok", "p_ans",
                                    "n_prompt", "populations")}
        out.append({**{k: v for k, v in r.items() if k not in ("prompt_b2", "prompt_b3")}, "b0": keep})
    if missing:
        shutil.rmtree(ws)
        raise RuntimeError(f"{len(missing)} E rows have no B0 record (e.g. {missing[:3]})")
    (ws / "data/E_rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in out))
    snap = Path(snapshot or EN.SNAPSHOT)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (snap / f).exists():
            shutil.copy(snap / f, ws / "data/tokenizer" / f)
    bounds = CM.read_json(run / "bounds.json", dict(SV.BOUNDS_F0))
    CM.write_json(run / "bounds.json", bounds)
    (ws / "TOOLS.md").write_text(tools_md(bounds))
    (ws / "INSTRUCTION.md").write_text(INSTRUCTION)
    for p in sorted((ws / "data").rglob("*"), reverse=True):
        p.chmod(0o444 if p.is_file() else 0o555)
    (ws / "data").chmod(0o555)
    return {"rows": len(out), "workspace": str(ws)}


def dry_rows(e, n_per_task: int = 2, seed: int = 99) -> list[dict]:
    """Synthetic rows for the tooling dry run (seed 99, no E content: drawn fresh, never by the E / S / H / T plan),
    labelled split "E" so the service accepts them, case ids prefixed DRY-, each with a scripted B0 record
    ("<A> is in <city>.\\nAnswer: unknown") so the P_ans paths run on the dummy model."""
    import random
    import exp51_cases as G
    rows = []
    for task in G.TASKS:
        for k in range(n_per_task):
            for attempt in range(G.MAX_ATTEMPTS):
                rng = random.Random(f"{seed}-DRY-{task}-{k}-{attempt}")
                A, B = rng.sample(list(G.C.NAMES), 2)
                gold, cf = rng.sample(list(G.C.CITIES), 2)
                b = G.draw_rest(rng, task, k, A, B, gold, cf, list(G.C.NAMES), list(G.C.CITIES))
                b.update({"split": "E", "seed": seed})
                rs = G.base_rows(b, list(G.C.CITIES))
                if rs is not None:
                    break
            for r in rs:
                r["case_id"], r["base_key"], r["synthetic"] = "DRY-" + r["case_id"], "DRY-" + r["base_key"], True
            rows += rs
    out = []
    for r in rows:
        ids = e.chat_ids(r["prompt"])
        gen = e.encode(f"{r['A']} is in {r['base_gold']}.\nAnswer: unknown\n")
        d = CM.designated(e.decode, gen)
        rec = {"prompt_ids": ids, "gen_ids": gen, "text": e.decode(gen), "n_prompt": len(ids), "j": d["j"],
               "p_ans_ok": d["ok"], "p_ans": len(ids) + d["j"] - 1 if d["ok"] else None,
               **CM.grade(r, e.decode(gen), [])}
        rec["populations"] = CM.population(r, rec, [])
        out.append({**{k: v for k, v in r.items() if k not in ("prompt_b2", "prompt_b3")}, "b0": rec})
    return out


def setup_dry(run: Path) -> dict:
    """The dry-run workspace (dummy model, synthetic rows)."""
    import exp51_engine as EN
    import exp51_service as SV
    run, ws = Path(run), Path(run) / "workspace"
    if ws.exists():
        raise FileExistsError(f"{ws} exists")
    for d in ("data/tokenizer", "queue/requests", "queue/responses", "captures", "edits_in", "notes", "scratch"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    e = EN.load("dummy", threads=2, edit_dir=run / "edits")
    rows = dry_rows(e)
    (ws / "data/E_rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(EN.SNAPSHOT / f, ws / "data/tokenizer" / f)
    bounds = dict(SV.BOUNDS_F0)
    CM.write_json(run / "bounds.json", bounds)
    (ws / "TOOLS.md").write_text(tools_md(bounds))
    (ws / "INSTRUCTION.md").write_text(INSTRUCTION)
    for p in sorted((ws / "data").rglob("*"), reverse=True):
        p.chmod(0o444 if p.is_file() else 0o555)
    (ws / "data").chmod(0o555)
    return {"rows": len(rows), "workspace": str(ws), "dry": True}


# ---------------------------------------------------------------------------------------------- sandbox

def bwrap_cmd(ws: Path, script: str) -> list[str]:
    ws = Path(ws).resolve()
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONPATH", "/opt/py",
           "--setenv", "OMP_NUM_THREADS", "4", "--setenv", "OPENBLAS_NUM_THREADS", "4", "--setenv", "LANG", "C.UTF-8",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--ro-bind", str(PY / "bin/python3.13"), "/opt/python/bin/python3.13",
           "--ro-bind", str(PY / "lib"), "/opt/python/lib",
           "--tmpfs", "/opt/python/lib/python3.13/site-packages"]
    for d in sorted(Path("/usr/lib").glob("python3*")) + sorted(Path("/usr/local/lib").glob("python3*")):
        cmd += ["--tmpfs", str(d)]                     # system Python packages masked too
    for pkg in SANDBOX_PKGS:
        if (SITE / pkg).exists():
            cmd += ["--ro-bind", str(SITE / pkg), f"/opt/py/{pkg}"]
    cmd += ["--bind", str(ws), "/work", "--ro-bind", str(ws / "data"), "/work/data",
            "--chdir", "/work", "/opt/python/bin/python3.13", "-S", "-B", script]
    return cmd


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))


def run_python(ws: Path, code: str, n: int, timeout: int = RUN_TIMEOUT_S) -> dict:
    ws = Path(ws)
    rel = f"scratch/run_{n:03d}.py"
    (ws / rel).write_text(code)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(bwrap_cmd(ws, f"/work/{rel}"), capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
        out = {"returncode": p.returncode, "stdout": p.stdout[-20000:], "stderr": p.stderr[-8000:]}
    except subprocess.TimeoutExpired as ex:
        out = {"returncode": None, "stdout": (ex.stdout or b"")[-20000:].decode(errors="replace") if isinstance(
            ex.stdout, bytes) else (ex.stdout or "")[-20000:], "stderr": f"timeout after {timeout} s"}
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


BOUNDARY_PROBES = {
    "no_network": "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=3); print('NET OPEN')\n"
                  "except OSError as e:\n    print('blocked', type(e).__name__)",
    "no_torch": "try:\n    import torch; print('TORCH IMPORTED')\nexcept ImportError:\n    print('blocked')",
    "no_sklearn": "try:\n    import sklearn; print('SKLEARN IMPORTED')\nexcept ImportError:\n    print('blocked')",
    "no_scipy": "try:\n    import scipy; print('SCIPY IMPORTED')\nexcept ImportError:\n    print('blocked')",
    "no_repo": "import os\nprint('REPO VISIBLE' if os.path.exists('{repo}') else 'blocked')",
    "no_env_file": "import os\nprint('ENV VISIBLE' if os.path.exists('{repo}/.env') else 'blocked')",
    "no_key_in_env": "import os\nprint('KEY IN ENV' if any('ANTHROPIC' in k for k in os.environ) else 'blocked')",
    "no_home": "import os\nprint('HOME VISIBLE' if os.path.exists('{home}/.cache') else 'blocked')",
    "data_read_only": "try:\n    open('/work/data/x', 'w'); print('DATA WRITABLE')\nexcept OSError:\n    print('blocked')",
    "data_rows_readable": "print('blocked' if not open('/work/data/E_rows.jsonl').readline() else 'readable')",
    "numpy_ok": "import numpy as np\nprint('readable' if np.linalg.svd(np.eye(3))[1].sum() == 3 else 'bad')",
    "tokenizers_ok": "from tokenizers import Tokenizer\nt = Tokenizer.from_file('/work/data/tokenizer/tokenizer.json')\n"
                     "print('readable' if t.encode(' York').ids else 'bad')",
    "workspace_writable": "open('/work/notes/probe.txt', 'w').write('x'); print('readable')",
    "no_system_packages": "import sys, glob\nsys.path += glob.glob('/usr/lib/python3*/*-packages') + "
                          "glob.glob('/usr/local/lib/python3*/*-packages') + glob.glob('/opt/python/lib/python3*/*')\n"
                          "hit = []\nfor m in ('torch', 'scipy', 'sklearn', 'jax', 'autograd'):\n    try:\n"
                          "        __import__(m); hit.append(m)\n    except Exception:\n        pass\n"
                          "print('IMPORTED ' + ','.join(hit) if hit else 'blocked')",
    "no_records": "import os\nprint('RECORDS VISIBLE' if os.path.exists('{repo}/research') else 'blocked')",
}


def boundary_check(ws: Path) -> dict:
    res = {}
    for i, (name, code) in enumerate(BOUNDARY_PROBES.items()):
        code = code.replace("{repo}", str(ROOT)).replace("{home}", str(Path.home()))
        out = run_python(ws, code, 900 + i, timeout=60)
        line = (out["stdout"].strip().splitlines() or [out["stderr"].strip()[-200:]])[-1]
        want = "readable" if name.endswith("_ok") or name in ("data_rows_readable", "workspace_writable") else "blocked"
        res[name] = {"got": line, "ok": line.startswith(want)}
    for p in Path(ws, "scratch").glob("run_9*.py"):
        p.unlink()
    (Path(ws) / "notes/probe.txt").unlink(missing_ok=True)
    return {"pass": all(v["ok"] for v in res.values()), "probes": res}


# ---------------------------------------------------------------------------------------------- tools and loop

MANIP_DOC = ("How each manipulation is computed: the change in f (manipulated row minus reference row, each read at that "
             "row's own final answer line ':' token), per base, averaged over the hardness levels present in both rows. "
             "evidence_removed: the same base's UNA-rel row (the line(s) stating the asked relation replaced by lines "
             "about another person) minus its ANSa row. city_changed: the CF row (the evidence states another city) "
             "minus the ANSa row. distractors_added: the ANSa row at H2 (7 more distractor lines; the dates shift with "
             "line position) minus the same row at H1. unrelated_vs_absent: the UNA-swap row (the question asks about "
             "a person the memory mentions only in a 'works as' line) minus the UNA-rel row, at H1.")

TOOLS = [
    {"name": "run_python", "description": "Run a Python script (numpy, tokenizers) in the sandbox at /work. Returns "
     "stdout / stderr (truncated). Counts against the Python-run bound.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"},
                                                       "timeout_s": {"type": "integer", "maximum": RUN_TIMEOUT_S}},
                      "required": ["code"]}},
    {"name": "forward", "description": "Send one request to the metered forward-pass service and wait for its "
     "response (formats in /work/TOOLS.md). Paired, prefix and read requests each count as one pass.",
     "input_schema": {"type": "object", "properties": {"request": {"type": "object"}}, "required": ["request"]}},
    {"name": "save_edit", "description": "Validate and store an edit from an .npz spec under /work/edits_in/.",
     "input_schema": {"type": "object", "properties": {"spec": {"type": "string"}, "meta": {"type": "object"}},
                      "required": ["spec"]}},
    {"name": "submit", "description": "Submit a saved edit (at most 3 submissions).",
     "input_schema": {"type": "object", "properties": {
         "edit_id": {"type": "string"}, "kind": {"type": "string", "enum": ["repair", "diagnostic"]},
         "component_and_computation": {"type": "string"}, "evidence": {"type": "string"},
         "predictions_by_row_type": {"type": "string"}, "flips": {"type": "string"},
         "activity_predictions": {"type": "string"},
         "signed_endpoint": {"type": "object", "properties": {
             "population": {"type": "string", "enum": ["FA", "FA-ab", "C", "W", "UNA", "CF"]},
             "direction": {"type": "string", "enum": ["increase", "decrease"]},
             "outcome": {"type": "string", "enum": ["Rel", "Uns", "Dam", "CF_follow", "Commit", "gold", "abstention",
                                                    "mixed", "wrong", "conflicting", "other", "degenerate"]}},
             "required": ["population", "direction", "outcome"]},
         "most_important_component": {"type": "object", "properties": {"layer": {"type": "integer"},
                                                                       "tensor": {"type": "string"}},
                                      "required": ["layer", "tensor"]},
         "f_site": {"type": "object", "description": "where the computation you name is read: a layer and one of "
                    "o_proj, down_proj, input_layernorm, post_attention_layernorm, resid (the decoder layer output)",
                    "properties": {"layer": {"type": "integer"},
                                   "tensor": {"type": "string", "enum": ["o_proj", "down_proj", "input_layernorm",
                                                                         "post_attention_layernorm", "resid"]}},
                    "required": ["layer", "tensor"]},
         "f_direction": {"type": "string", "description": "a .npy vector under /work/edits_in/: f = the projection of "
                         "the site's output at the final answer line's ':' token on this direction"},
         "f_int_sign": {"type": "integer", "enum": [1, -1], "description": "the direction in which moving f should "
                        "move your signed endpoint"},
         "activity": {"type": "array", "maxItems": 4, "description": "your predictions (e) in structured form",
                      "items": {"type": "object", "properties": {
                          "manipulation": {"type": "string", "enum": ["evidence_removed", "city_changed",
                                                                      "distractors_added", "unrelated_vs_absent"],
                                           "description": MANIP_DOC},
                          "type": {"type": "string", "enum": ["directional", "equivalence"]},
                          "sign": {"type": "integer", "enum": [1, -1, 0]}},
                          "required": ["manipulation", "type", "sign"]}}},
         "required": ["edit_id", "kind", "component_and_computation", "evidence", "predictions_by_row_type", "flips",
                      "activity_predictions", "signed_endpoint", "most_important_component", "f_site",
                      "f_direction", "f_int_sign", "activity"]}},
    {"name": "finish", "description": "End the run.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def validate_input(schema: dict, value, path: str = "input") -> list[str]:
    """A minimal JSON-schema check for the tool schemas above (type, required, properties, enum, maximum)."""
    errs = []
    t = schema.get("type")
    types = {"object": dict, "string": str, "integer": int, "array": list, "boolean": bool, "number": (int, float)}
    if t in types and (not isinstance(value, types[t]) or (t in ("integer", "number") and isinstance(value, bool))):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        errs.append(f"{path}: {value} > {schema['maximum']}")
    if t == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, v in enumerate(value):
            errs += validate_input(schema.get("items", {}), v, f"{path}[{i}]")
    if t == "object":
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: required")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += validate_input(sub, value[k], f"{path}.{k}")
    return errs


TOOL_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
MAX_CONSECUTIVE_API_ERRORS = 5
TRANSIENT_HTTP = (408, 429, 500, 502, 503, 504, 529)


class TransientAPIError(Exception):
    """429 / 5xx / 529 / timeout: retried with backoff, ledgered as an API retry (not an editor failed request)."""


def backoff_s(k: int) -> float:
    return min(60.0, 2.0 ** k)


class Meter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.calls = CM.read_json(self.path, [])

    def usd(self) -> float:
        return sum(c["usd"] if c.get("placeholder") else cost_of(c["usage"]) for c in self.calls)

    def context_tokens(self) -> int:
        """The last measured call's prompt plus output (the next prompt's floor)."""
        for c in reversed(self.calls):
            if not c.get("placeholder"):
                u = c["usage"]
                return sum(u.get(k, 0) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens", "output_tokens"))
        return 0

    def add_placeholder(self, usd: float) -> None:
        """A call that may have been billed but never answered (in flight at a crash): booked at its worst case and
        never trued up downward."""
        self.calls.append({"placeholder": True, "usage": {}, "usd": usd, "t": time.time()})
        CM.write_json(self.path, self.calls)

    def add(self, usage: dict, wall_s: float) -> None:
        self.calls.append({"usage": usage, "usd": cost_of(usage), "wall_s": wall_s, "t": time.time()})
        CM.write_json(self.path, self.calls)


def cost_of(u: dict) -> float:
    cw = u.get("cache_creation") or {}
    w1h = cw.get("ephemeral_1h_input_tokens")
    w5m = cw.get("ephemeral_5m_input_tokens")
    total_w = u.get("cache_creation_input_tokens", 0) or 0
    if w1h is None and w5m is None:
        w1h, w5m = total_w, 0        # no breakdown: price every write at the 1-hour rate (the higher one)
    return (u.get("input_tokens", 0) * PRICES["input"] + u.get("output_tokens", 0) * PRICES["output"] +
            u.get("cache_read_input_tokens", 0) * PRICES["cache_read"] + (w1h or 0) * PRICES["cache_write_1h"] +
            (w5m or 0) * PRICES["cache_write_5m"]) / 1e6


def worst_case_usd(messages: list, max_tokens: int) -> float:
    chars = len(json.dumps(messages)) + len(json.dumps(TOOLS)) + len(SYSTEM)
    return (chars / 2.5) * PRICES["cache_write_1h"] / 1e6 + max_tokens * PRICES["output"] / 1e6   # measured 2.9 chars / token


class ApiClient:
    """The real Messages API client (urllib). Not used before the dry run."""

    def __init__(self, cfg: dict = API_CONFIG):
        self.cfg = cfg

    def body(self, messages: list) -> dict:
        ttl = {"type": "ephemeral", "ttl": self.cfg["cache_ttl"]}
        tools = [dict(t) for t in TOOLS]
        tools[-1] = {**tools[-1], "cache_control": ttl}
        msgs = json.loads(json.dumps(messages))
        last = msgs[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"][-1] = {**last["content"][-1], "cache_control": ttl}
        return {"model": self.cfg["model"], "max_tokens": self.cfg["max_tokens"],
                "system": [{"type": "text", "text": SYSTEM, "cache_control": ttl}], "tools": tools,
                "messages": msgs, "output_config": {"effort": self.cfg["effort"]},
                "thinking": dict(self.cfg["thinking"])}

    def __call__(self, messages: list) -> dict:
        req = urllib.request.Request(self.cfg["url"], data=json.dumps(self.body(messages)).encode(), method="POST",
                                     headers={"x-api-key": _api_key(), "anthropic-version": self.cfg["anthropic_version"],
                                              "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as ex:
            if ex.code in TRANSIENT_HTTP:
                raise TransientAPIError(f"HTTP {ex.code}") from None
            raise RuntimeError(f"API HTTP {ex.code}: {ex.read()[:500]!r}") from None   # the body never holds the key
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as ex:
            raise TransientAPIError(f"{type(ex).__name__}") from None


class DummyClient:
    """Scripted synthetic responses for the dry-run harness tests (no network). `script` = list of content lists (or
    callables of the messages, for ids only known at run time); `seen` keeps a copy of the messages of each call."""

    def __init__(self, script: list[list[dict]], usage: dict | None = None):
        self.script, self.i, self.seen = script, 0, []
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 20000,
                               "cache_creation_input_tokens": 2000}

    def __call__(self, messages: list) -> dict:
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.seen.append(json.loads(json.dumps(messages)))
        if callable(content):
            content = content(messages)
        if content == "TRANSIENT":
            raise TransientAPIError("scripted 529")
        if isinstance(content, dict):                      # {"stop_reason": ..., "content": [...]}
            return {"content": content["content"], "stop_reason": content["stop_reason"], "usage": dict(self.usage),
                    **({"stop_details": content["stop_details"]} if "stop_details" in content else {})}
        stop = "tool_use" if any(c["type"] == "tool_use" for c in content) else "end_turn"
        return {"content": content, "stop_reason": stop, "usage": dict(self.usage)}


def wait_response(ws: Path, rid: str, timeout: float, poll: float = 0.5, service=None) -> dict:
    path = Path(ws) / "queue/responses" / f"{rid}.json"
    t0 = time.time()
    while not path.exists():
        if service is not None:
            service.serve(once=True)
            continue
        if time.time() - t0 > timeout:
            return {"ok": False, "error": "timed out waiting for the service"}
        time.sleep(poll)
    return json.loads(path.read_text())


class Harness:
    def __init__(self, run: Path, client, service=None, stop_usd: float = STOP_USD, max_turns: int | None = None):
        import exp51_service as SV
        self.run, self.ws, self.client, self.service = Path(run), Path(run) / "workspace", client, service
        self.bounds = CM.read_json(self.run / "bounds.json", dict(SV.BOUNDS_F0))
        self.ledger = SV.Ledger(self.run, self.ws)
        self.meter = Meter(self.run / "api_usage.json")
        self.stop_usd = min(stop_usd, STOP_USD)
        if max_turns is not None:
            self.bounds = {**self.bounds, "turns": min(int(max_turns), self.bounds["turns"])}
        self.transcript = self.run / "transcript.jsonl"
        self.subs = self.run / "submissions"
        self.subs.mkdir(exist_ok=True)
        self.n_req = 0
        self.sleep = time.sleep
        self.refusals, self.noted = 0, False

    def log(self, obj: dict) -> None:
        with open(self.transcript, "a") as f:
            f.write(json.dumps({"t": time.time(), **obj}) + "\n")

    def tool(self, name: str, inp: dict) -> dict:
        t = self.ledger.totals()
        if name == "run_python":
            if t["python_runs"] >= self.bounds["python_runs"]:
                return {"error": "Python-run bound reached"}
            out = run_python(self.ws, inp["code"], t["python_runs"] + 1, min(int(inp.get("timeout_s", 600)),
                                                                              RUN_TIMEOUT_S))
            self.ledger.add({"kind": "python_run", "ok": True, "wall_s": out["wall_s"]})
            return out
        if name in ("forward", "save_edit"):
            req = inp["request"] if name == "forward" else {"type": "save_edit", "spec": inp["spec"],
                                                            "meta": inp.get("meta", {})}
            self.n_req += 1
            rid = f"h{int(time.time())}_{self.n_req:04d}"
            CM.write_json(self.ws / "queue/requests" / f"{rid}.json", req)
            resp = wait_response(self.ws, rid, timeout=6 * 3600, service=self.service)
            s = json.dumps(resp)
            return resp if len(s) < 60000 else {"truncated": True, "head": s[:60000]}
        if name == "submit":
            n = len(list(self.subs.glob("*.json")))
            if n >= self.bounds["submissions"]:
                return {"error": "submission bound reached"}
            if not (self.run / "edits" / f"{inp['edit_id']}.json").exists():
                return {"error": f"no saved edit {inp['edit_id']}"}
            src = (self.ws / inp["f_direction"].replace("/work/", "", 1)).resolve()
            if self.ws.resolve() not in src.parents or not src.exists() or src.suffix != ".npy":
                return {"error": "f_direction must be an existing .npy file inside /work"}
            import numpy as np
            v = np.load(src, allow_pickle=False)
            if v.ndim != 1 or not np.all(np.isfinite(v)) or not np.any(v):
                return {"error": "f_direction must be a finite, non-zero 1-D vector"}
            dst = self.subs / f"{n + 1}_f_direction.npy"
            shutil.copy(src, dst)                      # frozen copy outside the sandbox's reach
            import hashlib
            CM.write_json(self.subs / f"{n + 1}.json", {**inp, "t": time.time(),
                                                        "f_direction_sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
            return {"ok": True, "submission": n + 1}
        return {"error": f"unknown tool {name}"}

    def answer(self, content: list, stop_reason: str | None, interrupted: bool = False) -> tuple[object, bool]:
        """The user message that answers one assistant response: tool_results (or a nudge) and whether it finished.
        interrupted: the response's tools were cut off by a crash (resume), so none is run again."""
        uses = [c for c in content if c.get("type") == "tool_use"]
        if not uses:
            return (MAXTOK_NUDGE if stop_reason == "max_tokens" else NUDGE), False
        results, done = [], False
        for u in uses:
            if stop_reason == "max_tokens":
                out = {"error": "your message hit max_tokens, so this tool call may be truncated and was NOT run; "
                                "send a shorter call"}
            else:
                errs = validate_input(TOOL_SCHEMAS[u["name"]], u.get("input")) if u["name"] in TOOL_SCHEMAS \
                    else [f"unknown tool {u['name']!r}"]
                if errs:
                    out = {"error": "invalid tool input, not run: " + "; ".join(errs)}
                elif u["name"] == "finish":
                    done = True
                    out = {"ok": True}
                elif interrupted:
                    out = {"error": INTERRUPTED}
                else:
                    out = self.tool(u["name"], u["input"])
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(out)[:60000]})
        return results, done

    def budget_note(self, msg: object, done: bool) -> object:
        """The user turn about to be sent, with the one budget note if spend has just passed the threshold."""
        if self.noted or done or self.meter.usd() <= BUDGET_NOTE_USD:
            return msg
        self.noted = True
        self.ledger.add({"kind": "budget_note", "ok": True, "budget_used_fraction": self.budget_fraction()})
        if isinstance(msg, str):
            return f"{msg}\n\n{BUDGET_NOTE}"
        msg[-1]["content"] += f"\n\n{BUDGET_NOTE}"
        return msg

    def budget_fraction(self) -> float:
        return round(self.meter.usd() / self.stop_usd, 3)

    def amend_after_refusal(self, messages: list) -> None:
        """Append-only: the fixed text block goes after the content of the user turn that drew the refusal."""
        last = messages[-1]
        blocks = [{"type": "text", "text": last["content"]}] if isinstance(last["content"], str) else last["content"]
        last["content"] = blocks + [{"type": "text", "text": REFUSAL_CONTINUE}]

    def rebuild(self) -> tuple[list, dict | None]:
        """The messages, verbatim from transcript.jsonl, and the last assistant record if it was left unanswered.
        Refused responses never join the messages; a user_amend record re-applies the fixed continuation."""
        recs = [json.loads(l) for l in self.transcript.read_text().splitlines() if l.strip()]
        messages = [{"role": "user", "content": INSTRUCTION}]
        for r in recs:
            if r["role"] == "assistant" and r.get("stop_reason") == "refusal":
                self.refusals += 1
                continue
            if r["role"] == "user_amend":
                self.amend_after_refusal(messages)
                continue
            if r["role"] not in ("user", "assistant") or messages[-1]["role"] == r["role"]:
                raise RuntimeError(f"transcript does not alternate at t={r.get('t')}; resume by hand")
            messages.append({"role": r["role"], "content": r["content"]})
        self.noted = any(r["role"] == "user" and BUDGET_NOTE in json.dumps(r["content"]) for r in recs)
        if self.refusals >= 2:
            raise RuntimeError("the run ended on a second refusal; nothing to resume")
        if recs and recs[-1].get("stop_reason") == "refusal":     # crashed before the continuation was recorded
            self.log({"role": "user_amend", "text": REFUSAL_CONTINUE})
            self.amend_after_refusal(messages)
            return messages, None
        last = recs[-1] if recs and recs[-1]["role"] == "assistant" else None
        return messages, last

    def loop(self, resume: bool = False) -> dict:
        start = self.run / "run_start.json"
        reason, done = None, False
        if resume:
            if (self.run / "loop_summary.json").exists():
                raise RuntimeError("the run already ended (loop_summary.json); nothing to resume")
            k = len(list(self.run.glob("transcript.pre_resume.*.jsonl"))) + 1
            shutil.copy(self.transcript, self.run / f"transcript.pre_resume.{k}.jsonl")
            messages, last = self.rebuild()
            n_cut = 0
            if last is None:                       # ends on a user record: an API call may have been in flight
                lost = worst_case_usd(messages, API_CONFIG["max_tokens"])
                self.meter.add_placeholder(lost)
                self.ledger.add({"kind": "lost_call_placeholder", "ok": True, "usd": round(lost, 6)})
            if last is not None:
                msg, done = self.answer(last["content"], last.get("stop_reason"), interrupted=True)
                msg = self.budget_note(msg, done)
                n_cut = sum(1 for c in (msg if isinstance(msg, list) else []) if INTERRUPTED in c["content"])
                self.log({"role": "user", "content": msg})
                messages.append({"role": "user", "content": msg})
            self.ledger.add({"kind": "resume", "ok": True, "messages": len(messages), "interrupted_calls": n_cut})
            t_start = CM.read_json(start, {}).get("t") or time.time()
            if done:
                reason = "finish"
        else:
            messages = [{"role": "user", "content": INSTRUCTION}]
            t_start = time.time()
            CM.write_json(start, {"t": t_start})
        consecutive = 0
        while not done:
            t = self.ledger.totals()
            if t["turns"] >= self.bounds["turns"]:
                reason = "turn bound"
                break
            if (time.time() - t_start) / 3600 > self.bounds["wall_hours"]:
                reason = "wall-clock bound"
                break
            if self.meter.context_tokens() + len(json.dumps(messages[-1])) / 2.5 > CONTEXT_STOP_TOKENS:
                reason = f"context bound ({self.meter.context_tokens()} prompt tokens at the last call)"
                break
            if self.meter.usd() + worst_case_usd(messages, API_CONFIG["max_tokens"]) > self.stop_usd:
                reason = f"cost stop (${self.meter.usd():.2f} measured)"
                break
            t0 = time.perf_counter()
            try:
                resp = self.client(messages)
            except TransientAPIError as ex:
                consecutive += 1
                self.ledger.add({"kind": "api_retry", "ok": True, "error": str(ex), "consecutive": consecutive,
                                 "wall_s": round(time.perf_counter() - t0, 2)})
                if consecutive >= MAX_CONSECUTIVE_API_ERRORS:
                    reason = f"{consecutive} consecutive API errors ({ex})"
                    break
                self.sleep(backoff_s(consecutive))
                continue
            consecutive = 0
            wall = time.perf_counter() - t0
            self.meter.add(resp.get("usage", {}), wall)
            self.ledger.add({"kind": "turn", "ok": True, "wall_s": round(wall, 2),
                             "stop_reason": resp.get("stop_reason"), "budget_used_fraction": self.budget_fraction()})
            stop_reason = resp.get("stop_reason")
            rec = {"role": "assistant", "content": resp["content"], "usage": resp.get("usage"), "stop_reason": stop_reason}
            if stop_reason == "refusal":
                rec["stop_details"] = resp.get("stop_details")
            self.log(rec)
            if stop_reason == "refusal":
                self.refusals += 1
                self.ledger.add({"kind": "refusal", "ok": True, "n": self.refusals})
                if self.refusals >= 2:
                    reason = "refusal (stop_reason)"
                    break
                self.log({"role": "user_amend", "text": REFUSAL_CONTINUE})
                self.amend_after_refusal(messages)
                continue
            messages.append({"role": "assistant", "content": resp["content"]})
            msg, done = self.answer(resp["content"], stop_reason)
            msg = self.budget_note(msg, done)
            self.log({"role": "user", "content": msg})
            messages.append({"role": "user", "content": msg})
            if done:
                reason = "finish"
        summary = {"reason": reason, "usd": round(self.meter.usd(), 4), "ledger": self.ledger.totals(),
                   "submissions": len(list(self.subs.glob("*.json")))}
        CM.write_json(self.run / "loop_summary.json", summary)
        return summary


def record_pid(run: Path) -> dict:
    """<run>/harness_pid.json: what scripts/exp51_stop.sh kills (the harness's own process group, never $!)."""
    d = {"pid": os.getpid(), "pgid": os.getpgid(0), "sid": os.getsid(0), "own_group": os.getpgid(0) == os.getpid(),
         "t": time.time()}
    CM.write_json(Path(run) / "harness_pid.json", d)
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("setup", "setup-dry", "check", "loop"))
    p.add_argument("--run", required=True)
    p.add_argument("--client", choices=("dummy", "api"), default="dummy")
    p.add_argument("--backend", choices=("dummy", "dummy36", "hf"), default="dummy")
    p.add_argument("--resume", action="store_true", help="continue a crashed run (same run; see the module doc)")
    p.add_argument("--i-have-the-go", action="store_true", help="required for --client api")
    p.add_argument("--dry", action="store_true", help="the tooling dry run: dummy model and DRY- rows only")
    p.add_argument("--stop-usd", type=float, default=None, help="harness stop (<= $45; the dry run uses $3)")
    p.add_argument("--max-turns", type=int, default=None, help="turn cap (<= the bound; the dry run caps at 15)")
    a = p.parse_args(argv)
    run = Path(a.run)
    if a.cmd == "setup":
        print(json.dumps(setup_workspace(run)))
        return 0
    if a.cmd == "setup-dry":
        print(json.dumps(setup_dry(run)))
        return 0
    if a.cmd == "check":
        r = boundary_check(run / "workspace")
        print(json.dumps(r, indent=1))
        return 0 if r["pass"] else 3
    if a.client == "api" and not a.i_have_the_go:
        raise SystemExit("the real editor run needs the build review, F0 and F1 first (--i-have-the-go)")
    import exp51_engine as EN
    import exp51_service as SV
    stop, turns = a.stop_usd, a.max_turns
    if a.dry:
        rows = CM.all_rows(run / "workspace/data/E_rows.jsonl")
        if a.backend not in ("dummy", "dummy36") or not rows or not all(r["case_id"].startswith("DRY-") and r.get("synthetic")
                                                       for r in rows):
            raise SystemExit("--dry needs --backend dummy|dummy36 and a setup-dry workspace (DRY- rows only)")
        stop = DRY_STOP_USD if stop is None else min(stop, DRY_STOP_USD)
        turns = DRY_MAX_TURNS if turns is None else min(turns, DRY_MAX_TURNS)
    elif a.client == "api" and a.backend != "hf":
        raise SystemExit("the real editor run uses the real model (--backend hf); the dummy is for --dry")
    if (run / "transcript.jsonl").exists() and not a.resume:
        raise SystemExit(f"{run} already has a transcript; a run is one-only (use --resume after a crash)")
    if a.resume and not (run / "transcript.jsonl").exists():
        raise SystemExit("--resume needs an existing transcript.jsonl")
    client = ApiClient() if a.client == "api" else DummyClient(DRY_SCRIPT)
    e = EN.load(a.backend, threads=8, edit_dir=run / "edits")
    svc = SV.Service(run, e, tc=[] if a.backend.startswith("dummy") else None)
    record_pid(run)
    th, ev = svc.start_thread()                    # the file queue is served while the editor works
    try:
        print(json.dumps(Harness(run, client, stop_usd=STOP_USD if stop is None else stop,
                                 max_turns=turns).loop(resume=a.resume)))
    finally:
        ev.set()
        th.join(timeout=60)
    return 0


DRY_SCRIPT = [
    [{"type": "text", "text": "List the data."},
     {"type": "tool_use", "id": "d1", "name": "run_python", "input": {"code": "import json\nrows = [json.loads(l) for l "
      "in open('/work/data/E_rows.jsonl')]\nprint(len(rows), rows[0]['case_id'])\n"}}],
    [{"type": "text", "text": "Queue a request from the sandbox and wait for it there."},
     {"type": "tool_use", "id": "d2", "name": "run_python", "input": {"code": "import json, time, os\nrows = [json.loads(l)"
      " for l in open('/work/data/E_rows.jsonl')]\nreq = {'type': 'prefix', 'rows': [rows[0]['case_id']], "
      "'arm': {'edit': None}}\njson.dump(req, open('/work/queue/requests/sb1.json', 'w'))\nfor _ in range(600):\n"
      "    if os.path.exists('/work/queue/responses/sb1.json'):\n        print(open('/work/queue/responses/sb1.json')"
      ".read()[:300]); break\n    time.sleep(0.5)\n"}}],
    [{"type": "tool_use", "id": "d3", "name": "forward", "input": {"request": {"type": "weights", "tensors":
                                                                                 ["model.norm.weight"]}}}],
    [{"type": "tool_use", "id": "d4", "name": "finish", "input": {"summary": "dry run"}}],
]


if __name__ == "__main__":
    sys.exit(main())
