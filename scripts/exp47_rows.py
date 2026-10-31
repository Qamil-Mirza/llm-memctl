#!/usr/bin/env python
"""§47: the bridge rows (data only; the Qwen tokenizer, no model). For each of the 70 §44 rev 8 Qwen 3B rows, the
§44 forced readout ids (clean / swap: chat + forced `Answer:` + whitespace + shared prefix) are rebuilt with
exp44_swap's own functions and checked against the S files' stamps; a row is aligned when clean and swap have equal
length. Writes runs/exp47_bridge/rows47.jsonl (aligned rows only) + rows47_counts.json.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp47_rows.py

Spec: research/proposals/2026-10-31-exp47-bridge.md rev 0 (be2b9e5) + rev 1 (a9b70ff). Per row: the chat ids (the
generation input; the forced ids must start with them), the forced tail, the differing positions (S1: exactly those,
all occurrences), multi_occurrence (the gate's gold pattern matches more than once in the evidence line, each match replaced), P1 = the last forced id, the
§44 readout ids (g_id, s_id, refusal ids), Δ_clean / Δ_swap from the S file (the asserted reproduction target),
follows["1"] (the entry filter), the §23 label and the §44 class. Exit 3 on any mismatch with the §44 files.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

S_DIR = Path("runs/exp44_swap/qwen3b_rev8/S")
OUT = Path("runs/exp47_bridge")
SMOKE = ("ec81a493", "7401057b", "5809eb10")     # rev 1 ruling 6


def main() -> int:
    import exp44_swap as X
    import exp44_swap_gate as G
    import transformers
    from memctl.analysis import patching as pt
    from memctl.analysis import swap as sw
    from memctl.analysis.jacobian import wrap_chat

    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    records = {r.question_id: r for r in X.load_population("qwen3b")}
    order, eligible_sha = X.load_eligible("qwen3b", gate_rev=8)
    refusal_ids = X.MODELS["qwen3b"]["refusal"].refusal_ids
    rows, problems, all_rows = [], [], []
    for e in order:
        q = e["question_id"]
        s44 = json.loads((S_DIR / f"{q}.json").read_text())
        rec = records[q]
        prompt, old, new = X.rebuild_swap(rec, e)
        x = sw.build_swap_inputs(tok, rec, prompt, old, new, e["substitute"], refusal_ids, None)
        if sw.sha_ids(x.clean_ids) != s44["stamp"]["clean_ids_sha256"] or sw.sha_ids(x.swap_ids) != s44["stamp"]["swap_ids_sha256"]:
            problems.append(f"{q}: forced ids sha differs from the S file")
            continue
        chat = {k: tok(wrap_chat(tok, p), add_special_tokens=False).input_ids for k, p in (("clean", rec.prompt), ("swap", prompt))}
        for k, ids in (("clean", x.clean_ids), ("swap", x.swap_ids)):
            if list(ids[:len(chat[k])]) != chat[k]:
                problems.append(f"{q}: the {k} forced ids do not start with the chat ids")
        aligned = len(x.clean_ids) == len(x.swap_ids) and len(chat["clean"]) == len(chat["swap"])
        all_rows.append({"question_id": q, "label": s44["label"], "aligned": aligned})
        if not aligned:
            continue
        diff = [i for i, (a, b) in enumerate(zip(chat["clean"], chat["swap"])) if a != b]
        tail_c, tail_s = list(x.clean_ids[len(chat["clean"]):]), list(x.swap_ids[len(chat["swap"]):])
        if tail_c != tail_s:
            problems.append(f"{q}: forced tails differ")
        if [i for i, (a, b) in enumerate(zip(x.clean_ids, x.swap_ids)) if a != b] != diff:
            problems.append(f"{q}: forced-sequence diff differs from the chat diff")
        n_occ = len(list(G.gold_pattern(rec.gold).finditer(old)))      # the gate's own match: each one is replaced
        if n_occ < 1:
            problems.append(f"{q}: the gate's gold pattern matches nothing in the evidence line")
        multi = n_occ > 1
        rows.append({"question_id": q, "fold": rec.fold, "label": s44["label"], "cls1": s44["cls"]["1"],
                     "follows1": bool(s44["follows"]["1"]), "entered": bool(s44["follows"]["1"]),
                     "question_type": s44["question_type"], "surface_type": s44["surface_type"],
                     "gold": rec.gold, "substitute": e["substitute"], "multi_occurrence": multi,
                     "n_occurrences": n_occ,
                     "chat_clean": chat["clean"], "chat_swap": chat["swap"], "forced_tail": tail_c,
                     "diff_positions": diff, "n_diff": len(diff), "n_chat": len(chat["clean"]),
                     "p1": len(x.clean_ids) - 1, "g_id": x.g_id, "s_id": x.s_id, "refusal_ids": list(x.refusal_ids),
                     "delta44_clean": s44["clean"]["delta"], "delta44_swap": s44["swap"]["delta"],
                     "top1_44_swap": s44["swap"]["top1"], "T_fwd44": s44.get("T_fwd"),
                     "forced_clean_sha256": s44["stamp"]["clean_ids_sha256"],
                     "forced_swap_sha256": s44["stamp"]["swap_ids_sha256"], "smoke": q in SMOKE})
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows47.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    ent = [r for r in rows if r["entered"]]
    counts = {"source_rows": len(all_rows), "aligned": len(rows),
              "aligned_by_label": dict(Counter(r["label"] for r in rows)),
              "entered_by_label": dict(Counter(r["label"] for r in ent)),
              "entered_success": sum(r["label"] == "success" for r in ent),
              "entered_failure": sum(r["label"] != "success" for r in ent),
              "span_n_diff": dict(sorted(Counter(r["n_diff"] for r in rows).items())),
              "multi_occurrence": sorted(r["question_id"] for r in rows if r["multi_occurrence"]),
              "smoke": {q: next(r["n_chat"] for r in rows if r["question_id"] == q) for q in SMOKE},
              "n_chat_min_median_max": [min(r["n_chat"] for r in rows), sorted(r["n_chat"] for r in rows)[len(rows) // 2],
                                        max(r["n_chat"] for r in rows)],
              "eligible_sha256": eligible_sha, "rows47_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "rows47_counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps({k: v for k, v in counts.items() if k != "eligible_sha256"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
