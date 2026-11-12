"""Timeout ledger (Advisor, 2026-11-09, after the _timed fix).

The scorer's rule is "a sympy-backed comparison that takes over 5 s counts as not equal". That rule depends on the
machine, so every scoring run reports which rows rest on a timeout, and rescores those rows with a 60 s timeout as a
SENSITIVITY check. The primary stays at 5 s.

A row rests on a timeout when any call its category depends on timed out:
  - the answer extraction or correctness check of its own record, or of its reference (the N record at the same cut
    for U arms, FULL at X = 0 and for N's differs-from-FULL flag);
  - the same-answer comparison between its answer and the reference's answer.
stage0_report.score_model is unchanged; the attribution is reconstructed from scoring.TIMEOUTS (function, args).
"""
from __future__ import annotations

import json
from collections import Counter

from selfdoubt import common as C
from selfdoubt import scoring as S
from selfdoubt import stage0_report as R

SENS_TIMEOUT_S = 60


def _k(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _rkey(r) -> tuple:
    return (r["item_id"], r["arm"], r["cut"])


def _attribute(recs: list[dict], rows: list[dict], hits: list) -> set:
    texts = {a[-1] for f, a in hits if f in ("canon_pred", "_correct")}
    pairs = {(_k(a[1]), _k(a[2])) for f, a in hits if f == "_same"}
    full = {r["item_id"]: r for r in recs if r["arm"] == "FULL"}
    nrec = {(r["item_id"], r["cut"]): r for r in recs if r["arm"] == "N"}
    mine = {_rkey(r): r for r in recs if r["arm"] != "FULL"}
    full_pred = {r["item_id"]: r["ref_pred"] for r in rows if r["arm"] != "N" and r["cut"] == C.X0}
    out = set()
    for row in rows:
        k = _rkey(row)
        own = mine[k]
        if row["arm"] == "N":
            ref, ref_pred = full.get(row["item_id"]), full_pred.get(row["item_id"])
        else:
            ref = full.get(row["item_id"]) if row["cut"] == C.X0 else nrec.get((row["item_id"], row["cut"]))
            ref_pred = row["ref_pred"]
        if (own["text"] in texts or (ref is not None and ref["text"] in texts)
                or (_k(row["pred"]), _k(ref_pred)) in pairs):
            out.add(k)
    return out


def score_with_ledger(recs: list[dict], items: dict, sens_timeout: int = SENS_TIMEOUT_S):
    """Score at the primary timeout; return (rows, rows_sensitivity, ledger). rows_sensitivity = rows with every
    timeout-resting row replaced by its rescore at sens_timeout (the same list object as rows when there is none)."""
    assert S.TIMEOUT_S == 5, S.TIMEOUT_S
    S.TIMEOUTS.clear()
    rows = R.score_model(recs, items)
    hits = list(S.TIMEOUTS)
    hit_rows = _attribute(recs, rows, hits)
    by_fn = Counter(f"{a[0]}|{f}" for f, a in hits)             # a[0] is the source for all three timed functions
    src = {r["item_id"]: r["source"] for r in recs}
    by_cell = Counter(f"{src[k[0]]}|{k[1]}@{k[2]}" for k in hit_rows)
    ledger = {"timeout_s": S.TIMEOUT_S, "n_calls_timed_out": len(hits), "calls_by_source_fn": dict(sorted(by_fn.items())),
              "n_rows_resting_on_timeout": len(hit_rows), "rows_by_source_arm_cut": dict(sorted(by_cell.items())),
              "sensitivity_timeout_s": sens_timeout}
    if not hit_rows:
        ledger.update({"n_calls_timed_out_sensitivity": 0, "category_moves_sensitivity": []})
        return rows, rows, ledger
    ids = {k[0] for k in hit_rows}
    S.TIMEOUT_S = sens_timeout
    S.TIMEOUTS.clear()
    try:
        sub = {_rkey(r): r for r in R.score_model([r for r in recs if r["item_id"] in ids], items)}
    finally:
        hits60 = list(S.TIMEOUTS)
        S.TIMEOUT_S = 5
        S.TIMEOUTS.clear()
    sens = [sub[_rkey(r)] if _rkey(r) in hit_rows else r for r in rows]
    moves = sorted([list(k), r["category"], sub[k]["category"]] for k, r in ((_rkey(r), r) for r in rows)
                   if k in hit_rows and sub[k]["category"] != r["category"])
    ledger.update({"n_calls_timed_out_sensitivity": len(hits60), "category_moves_sensitivity": moves})
    return rows, sens, ledger
