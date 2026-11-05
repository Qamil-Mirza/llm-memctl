#!/usr/bin/env python
"""§51 metered forward-pass service (CPU float32, file queue). The editor (through the harness or its sandboxed
Python) writes request JSON files into <ws>/queue/requests/; this process validates each against the search bounds,
runs it on E rows (or editor-built texts) only, writes <ws>/queue/responses/<id>.json (+ arrays under
<ws>/captures/), and appends one line per request to the ledger (outside the workspace; a read-only copy is mirrored
into it).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_service.py --run runs/exp51/editor_run [--backend dummy] [--once]

Request types (schema in REQUEST_DOC):
- paired:  {"rows": [E case ids], "arms": [arm, arm]}; total-effect generation (<= 160 tokens) of each row under both
           arms; <= 48 pairs (96 sequences). An arm is {"edit": null | edit_id, "dose": 1.0}.
- prefix:  {"rows": [...], "arm": arm, "hybrid": false}; controlled prefix from B0's stored ids (<= 16 tokens);
           <= 256 rows.
- read:    {"items": [{"row": id, "source": "prompt"|"b0_prefix"|"b0_full"} | {"text": user prompt,
           "continuation": generated text}], "positions": "last"|"p_ans"|[ints], "layers": [...],
           "components": [[layer, tensor], ...], "logits": false, "logit_ids": [...]|null, "arm": arm};
           teacher-forced; <= 512 texts; arrays to captures/<id>.npz.
- weights: {"tensors": [param names], "unembed_ids": [...]}: float32 arrays of editable tensors, the final norm or
           unembedding rows (no forward pass; not counted as a pass).
- save_edit: {"spec": "edits_in/<file>.npz", "meta": {...}}: validates the candidate family and stores the edit
           (<= 40 saved edits).

Bounds (§5.4, F0 maxima; the run's values are in <run>/bounds.json): passes <= 30 (paired, prefix and read each count
one), input tokens <= 2.0M, generated <= 0.4M, captures on disk <= 20 GB, saved edits <= 40, failed requests <= 10
(not passes; their tokens and time are in the ledger). A request that would exceed a bound is refused and counts as a
failed request.

Fixed choices of this build: an arm with edit null in a paired request reuses B0's stored generation (the same path;
the no-op gate checks it) and its tokens are ledgered as "cached"; generated-token bounds are checked against the
worst case (rows x 160 / 16) before running; the rows the service can see are exactly <ws>/data/E_rows.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402

BOUNDS_F0 = {"passes": 30, "paired_pairs": 64, "prefix_rows": 256, "read_texts": 512, "input_tokens": 2_000_000,
             "generated_tokens": 400_000, "capture_bytes": 20 * 10 ** 9, "python_runs": 100, "turns": 120,
             "wall_hours": 48.0, "saved_edits": 40, "failed_requests": 10, "submissions": 3}
BOUNDS_F1 = {**BOUNDS_F0, "paired_pairs": 48}   # F1 (§13.6): a largest paired pass stays inside the 1 h cache TTL
PASS_TYPES = ("paired", "prefix", "read")
REQUEST_DOC = __doc__


class Refused(Exception):
    pass


_LEDGER_LOCK = threading.Lock()


class Ledger:
    """Append-only JSONL; the file is the single source of truth (the harness and the service each hold one; the
    service may run in a thread of the harness process, so appends take a lock)."""

    def __init__(self, run: Path, ws: Path):
        self.path, self.mirror = Path(run) / "ledger.jsonl", Path(ws) / "ledger_copy.jsonl"

    @property
    def entries(self) -> list[dict]:
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []

    def totals(self) -> dict:
        t = {"passes": 0, "input_tokens": 0, "generated_tokens": 0, "cached_tokens": 0, "capture_bytes": 0,
             "saved_edits": 0, "failed_requests": 0, "wall_s": 0.0, "python_runs": 0, "turns": 0}
        for e in self.entries:
            t["wall_s"] += e.get("wall_s", 0.0)
            t["input_tokens"] += e.get("input_tokens", 0)
            t["generated_tokens"] += e.get("generated_tokens", 0)
            t["cached_tokens"] += e.get("cached_tokens", 0)
            t["capture_bytes"] += e.get("capture_bytes", 0)
            if e["kind"] == "python_run":
                t["python_runs"] += 1
            if e["kind"] == "turn":
                t["turns"] += 1
            if not e.get("ok", True):
                t["failed_requests"] += 1
            elif e["kind"] in PASS_TYPES:
                t["passes"] += 1
            elif e["kind"] == "save_edit":
                t["saved_edits"] += 1
        return t

    def mirror_text(self) -> str:
        """The editor's copy: the ledger, without the dollar figure of a lost-call placeholder (F1 ruling)."""
        out = []
        for line in self.path.read_text().splitlines(keepends=True):
            if '"lost_call_placeholder"' in line:
                e = json.loads(line)
                e.pop("usd", None)
                line = json.dumps(e) + "\n"
            out.append(line)
        return "".join(out)

    def add(self, entry: dict) -> None:
        entry = {"t": time.time(), **entry}
        with _LEDGER_LOCK:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            try:
                self.mirror.write_text(self.mirror_text())
            except OSError:
                pass


class Service:
    def __init__(self, run: Path, engine, tc: list[str] | None = None):
        self.run, self.e = Path(run), engine
        self.ws = self.run / "workspace"
        self.q_in, self.q_out = self.ws / "queue/requests", self.ws / "queue/responses"
        self.q_in.mkdir(parents=True, exist_ok=True)
        self.q_out.mkdir(parents=True, exist_ok=True)
        (self.ws / "captures").mkdir(exist_ok=True)
        self.bounds = CM.read_json(self.run / "bounds.json", dict(BOUNDS_F0))
        self.ledger = Ledger(self.run, self.ws)
        self.rows = {r["case_id"]: r for r in CM.all_rows(self.ws / "data/E_rows.jsonl")}
        if any(r["split"] != "E" for r in self.rows.values()):
            raise RuntimeError("the workspace holds non-E rows")
        self.b0 = {r["case_id"]: r["b0"] for r in self.rows.values()}
        self.tc = tc if tc is not None else CM.t_cities()
        self.edit_dir = self.run / "edits"
        self.e.edits.edit_dir = self.edit_dir

    # -------------------------------------------------------------- validation
    def _check(self, kind: str, worst_in: int, worst_gen: int) -> None:
        t, b = self.ledger.totals(), self.bounds
        if t["failed_requests"] >= b["failed_requests"]:
            raise Refused(f"failed-request bound reached ({b['failed_requests']}); the service is closed")
        if kind in PASS_TYPES and t["passes"] >= b["passes"]:
            raise Refused(f"pass bound reached ({b['passes']})")
        if t["input_tokens"] + worst_in > b["input_tokens"]:
            raise Refused(f"input-token bound: {t['input_tokens']} + {worst_in} > {b['input_tokens']}")
        if t["generated_tokens"] + worst_gen > b["generated_tokens"]:
            raise Refused(f"generated-token bound: {t['generated_tokens']} + {worst_gen} > {b['generated_tokens']}")
        if (time.time() - self.started()) / 3600 > b["wall_hours"]:
            raise Refused("wall-clock bound reached")

    def started(self) -> float:
        return self.ledger.entries[0]["t"] if self.ledger.entries else time.time()

    def _row(self, cid: str) -> dict:
        if cid not in self.rows:
            raise Refused(f"unknown row {cid!r} (only E rows exist here)")
        return self.rows[cid]

    def _edit(self, arm: dict | None) -> tuple | None:
        arm = arm or {}
        eid = arm.get("edit")
        if eid is None:
            return None
        if not (self.edit_dir / f"{eid}.json").exists():
            raise Refused(f"no saved edit {eid!r}")
        man = json.loads((self.edit_dir / f"{eid}.json").read_text())
        if man.get("family") != "candidate":
            raise Refused(f"{eid} is not an editor edit")
        dose = float(arm.get("dose", 1.0))
        if not (0 < abs(dose) <= 4):
            raise Refused("dose must satisfy 0 < |dose| <= 4")
        return (eid, dose)

    # -------------------------------------------------------------- passes
    def _used(self, **extra) -> dict:
        """The running usage of the current request (so a request that fails part-way still ledgers what it used)."""
        self.cur = {"input_tokens": 0, "generated_tokens": 0, **extra}
        return self.cur

    def paired(self, req: dict) -> dict:
        rows = req["rows"]
        if not (1 <= len(rows) <= self.bounds["paired_pairs"]) or len(req.get("arms", [])) != 2:
            raise Refused(f"paired: 1-{self.bounds['paired_pairs']} rows and exactly 2 arms")
        edits = [self._edit(a) for a in req["arms"]]
        rs = [self._row(c) for c in rows]
        n_in = sum(len(self.b0[r["case_id"]]["prompt_ids"]) for r in rs) * sum(x is not None for x in edits)
        self._check("paired", n_in, len(rs) * EN.MAX_NEW_TOTAL * sum(x is not None for x in edits))
        out, used = [], self._used(cached_tokens=0)
        for r in rs:
            item = {"row": r["case_id"]}
            for k, ed in zip(("a", "b"), edits):
                b0 = self.b0[r["case_id"]]
                if ed is None:
                    item[k] = {"text": b0["text"], "cls": b0["cls"], "cached": True}
                    used["cached_tokens"] += len(b0["gen_ids"])
                    continue
                g = self.e.generate(list(b0["prompt_ids"]), mode="total", edit=ed)
                gr = CM.grade(r, g.text, self.tc)
                item[k] = {"text": g.text, "cls": gr["cls"], "city": gr["city"], "n_gen": len(g.gen_ids)}
                used["input_tokens"] += len(b0["prompt_ids"])
                used["generated_tokens"] += len(g.gen_ids)
            out.append(item)
        return {"items": out, **used}

    def prefix(self, req: dict) -> dict:
        rows = req["rows"]
        if not (1 <= len(rows) <= self.bounds["prefix_rows"]):
            raise Refused(f"prefix: 1-{self.bounds['prefix_rows']} rows")
        ed = self._edit(req.get("arm"))
        rs = [self._row(c) for c in rows]
        bad = [r["case_id"] for r in rs if not self.b0[r["case_id"]].get("p_ans_ok")]
        if bad:
            raise Refused(f"rows without a P_ans (NM): {bad[:5]}")
        n_in = sum(len(CM.prefix_ids(self.b0[r["case_id"]])) for r in rs)
        self._check("prefix", n_in, len(rs) * EN.MAX_NEW_RELEASE)
        out, used = [], self._used()
        arm = EN.Arm(hybrid=bool(req.get("hybrid")))
        for r in rs:
            b0 = self.b0[r["case_id"]]
            ids = CM.prefix_ids(b0)
            g = self.e.generate(ids, arm, mode="prefix", edit=ed)
            text = self.e.decode(ids[b0["n_prompt"]:]) + g.text
            gr = CM.grade(r, text, self.tc)
            out.append({"row": r["case_id"], "answer": g.text, "cls": gr["cls"], "city": gr["city"]})
            used["input_tokens"] += len(ids)
            used["generated_tokens"] += len(g.gen_ids)
        return {"items": out, **used}

    def _read_ids(self, it: dict) -> tuple[list[int], dict]:
        if "row" in it:
            r = self._row(it["row"])
            b0 = self.b0[r["case_id"]]
            src = it.get("source", "b0_prefix")
            if src == "prompt":
                return list(b0["prompt_ids"]), {"p_ans": None}
            if src == "b0_full":
                return list(b0["prompt_ids"]) + list(b0["gen_ids"]), {"p_ans": b0.get("p_ans")}
            if not b0.get("p_ans_ok"):
                raise Refused(f"{r['case_id']} has no P_ans")
            return CM.prefix_ids(b0), {"p_ans": b0["p_ans"]}
        ids = self.e.chat_ids(it["text"]) + (self.e.encode(it["continuation"]) if it.get("continuation") else [])
        return ids, {"p_ans": None}

    def read(self, req: dict, rid: str) -> dict:
        items = req["items"]
        if not (1 <= len(items) <= self.bounds["read_texts"]):
            raise Refused(f"read: 1-{self.bounds['read_texts']} texts")
        ed = self._edit(req.get("arm"))
        layers = [int(l) for l in req.get("layers", [])]
        comps = [(int(l), t) for l, t in req.get("components", [])]
        for l, t in comps:
            if t not in EN.COMPONENT_MODULE or not 0 <= l < self.e.n_layers:
                raise Refused(f"bad component {(l, t)} (tensors {sorted(EN.COMPONENT_MODULE)}, layers 0-"
                              f"{self.e.n_layers - 1})")
        if any(not 0 <= l < self.e.n_layers for l in layers):
            raise Refused(f"bad layer (layers 0-{self.e.n_layers - 1})")
        prepared = [self._read_ids(it) for it in items]
        n_in = sum(len(i) for i, _ in prepared)
        per = len(layers) + len(comps)
        npos = [self._positions(req.get("positions", "last"), ids, info) for ids, info in prepared]
        bytes_est = sum(len(p) for p in npos) * (per * self.e.hidden * 4 + (len(req.get("logit_ids") or []) or
                                                                         (self.e.tok.vocab_size if req.get("logits") else 0)) * 4)
        t = self.ledger.totals()
        if t["capture_bytes"] + bytes_est > self.bounds["capture_bytes"]:
            raise Refused("capture-disk bound")
        self._check("read", n_in, 0)
        arrays, meta = {}, []
        used = self._used()
        for k, ((ids, info), pos) in enumerate(zip(prepared, npos)):
            used["input_tokens"] += len(ids)
            rd = self.e.read(ids, pos, layers, comps, pos if req.get("logits") else [], req.get("logit_ids"), edit=ed)
            for l in layers:
                arrays[f"i{k}_resid_L{l}"] = rd["resid"][l]
            for (l, tname) in comps:
                arrays[f"i{k}_{tname}_L{l}"] = np.stack([rd["comp"][(l, tname)][p] for p in pos])
            if rd["logits"] is not None:
                arrays[f"i{k}_logits"] = rd["logits"]
            meta.append({"item": k, "n_tokens": len(ids), "positions": pos, "p_ans": info["p_ans"]})
        path = self.ws / "captures" / f"{rid}.npz"
        np.savez(path, **arrays)
        return {"items": meta, "npz": str(path.relative_to(self.ws)), "input_tokens": n_in,
                "capture_bytes": path.stat().st_size}

    def _positions(self, spec, ids, info) -> list[int]:
        if spec == "last":
            return [len(ids) - 1]
        if spec == "p_ans":
            if info["p_ans"] is None:
                raise Refused("p_ans is only defined for B0-prefix / B0-full row items")
            return [info["p_ans"]]
        pos = [int(p) if int(p) >= 0 else len(ids) + int(p) for p in spec]
        if any(not 0 <= p < len(ids) for p in pos) or len(pos) > 64:
            raise Refused("positions out of range (or more than 64)")
        return pos

    def weights(self, req: dict, rid: str) -> dict:
        names = req.get("tensors", [])
        allowed = set(self.e.edits.shapes()) | {"model.norm.weight"}
        arrays = {}
        for n in names:
            if n not in allowed:
                if n not in dict(self.e.net.named_parameters()):
                    raise Refused(f"{n}: no such tensor in this model (layers 0-{self.e.n_layers - 1})")
                raise Refused(f"{n} is not readable (editable tensors and model.norm.weight only)")
            arrays[n.replace(".", "__")] = self.e.edits.params[n].detach().float().cpu().numpy()
        ids = [int(i) for i in req.get("unembed_ids", [])]
        if len(ids) > 4096:
            raise Refused("at most 4096 unembedding rows")
        if ids:
            arrays["unembed_rows"] = self.e.net.lm_head.weight[ids].detach().float().cpu().numpy()
            arrays["unembed_ids"] = np.asarray(ids)
        path = self.ws / "captures" / f"{rid}.npz"
        np.savez(path, **arrays)
        return {"npz": str(path.relative_to(self.ws)), "capture_bytes": path.stat().st_size}

    def save_edit(self, req: dict) -> dict:
        t = self.ledger.totals()
        if t["saved_edits"] >= self.bounds["saved_edits"]:
            raise Refused("saved-edit bound reached")
        spec = (self.ws / str(req["spec"]).removeprefix("/work/")).resolve()   # the sandbox's /work path too
        if self.ws.resolve() not in spec.parents:
            raise Refused("the spec must be inside the workspace")
        comps = spec_components(np.load(spec, allow_pickle=False))
        man = ED.save_edit(self.edit_dir, comps, self.e.edits.shapes(), "candidate",
                           originals=self.e.edits.original_np, meta=req.get("meta", {}),
                           model_fingerprint=self.e.fingerprint())
        self.e.edits.record_results(man["edit_id"])
        self.e.edits.orig.clear()
        return {"edit_id": man["edit_id"], "n_components": man["n_components"], "n_sparse": man["n_sparse"]}

    # -------------------------------------------------------------- loop
    def handle(self, path: Path) -> dict:
        rid = path.stem
        t0 = time.perf_counter()
        entry = {"id": rid, "kind": "unknown", "ok": False}
        self.cur = {}
        try:
            req = json.loads(path.read_text())
            entry["kind"] = kind = req.get("type")
            if self.ledger.totals()["failed_requests"] >= self.bounds["failed_requests"]:
                raise Refused(f"failed-request bound reached ({self.bounds['failed_requests']}); the service is closed")
            if kind == "paired":
                res = self.paired(req)
            elif kind == "prefix":
                res = self.prefix(req)
            elif kind == "read":
                res = self.read(req, rid)
            elif kind == "weights":
                res = self.weights(req, rid)
            elif kind == "save_edit":
                res = self.save_edit(req)
            else:
                raise Refused(f"unknown request type {kind!r}")
            resp = {"id": rid, "ok": True, **res}
            entry.update({"ok": True, **{k: res[k] for k in ("input_tokens", "generated_tokens", "cached_tokens",
                                                              "capture_bytes") if k in res}})
        except (Refused, ED.EditRefused, KeyError, ValueError, TypeError) as ex:
            resp = {"id": rid, "ok": False, "refused": True, "error": f"{type(ex).__name__}: {ex}"}
            entry["error"] = resp["error"]
        except Exception as ex:      # noqa: BLE001  (a crash is a failed request, ledgered)
            resp = {"id": rid, "ok": False, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]}
            entry["error"] = resp["error"]
            self.e.edits.revert()
        if not entry["ok"]:
            entry.update({k: v for k, v in self.cur.items() if v})       # tokens used before the failure
        entry["wall_s"] = round(time.perf_counter() - t0, 3)
        resp["wall_s"] = entry["wall_s"]
        resp["ledger"] = {k: v for k, v in self.ledger.totals().items()}
        self.ledger.add(entry)
        resp["ledger"] = self.ledger.totals()
        CM.write_json(self.q_out / f"{rid}.json", resp)
        path.rename(path.with_suffix(".done"))
        return resp

    def pending(self) -> list[Path]:
        return sorted(self.q_in.glob("*.json"))

    def serve(self, once: bool = False, poll: float = 0.5, stop_file: Path | None = None,
              stop_event: threading.Event | None = None) -> None:
        while True:
            for p in self.pending():
                self.handle(p)
            if once or (stop_file and stop_file.exists()) or (stop_event is not None and stop_event.is_set()):
                return
            time.sleep(poll)

    def start_thread(self) -> tuple[threading.Thread, threading.Event]:
        ev = threading.Event()
        th = threading.Thread(target=self.serve, kwargs={"stop_event": ev}, daemon=True)
        th.start()
        return th, ev


def spec_components(z) -> list[dict]:
    """Editor spec .npz -> components. Keys: c{i}_layer, c{i}_tensor (a 0-d unicode array), c{i}_kind and the arrays
    c{i}_U / c{i}_V (dense), c{i}_idx / c{i}_D (rows, cols, coords)."""
    idx = sorted({int(k.split("_")[0][1:]) for k in z.files})
    out = []
    for i in idx:
        c = {"layer": int(z[f"c{i}_layer"]), "tensor": str(z[f"c{i}_tensor"]), "kind": str(z[f"c{i}_kind"])}
        for k in ("U", "V", "D", "idx"):
            if f"c{i}_{k}" in z.files:
                c[k] = z[f"c{i}_{k}"]
        out.append(c)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--backend", default="hf", choices=("hf", "dummy"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--once", action="store_true")
    a = p.parse_args(argv)
    run = Path(a.run)
    e = EN.load(a.backend, a.threads, edit_dir=run / "edits")
    Service(run, e).serve(once=a.once, stop_file=run / "STOP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
