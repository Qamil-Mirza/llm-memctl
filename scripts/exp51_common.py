#!/usr/bin/env python
"""§51 shared pieces: paths, the case list, the B0 store (stored token ids, designated marker, P_ans), populations,
grading of a generation, and the frozen comparator ingredients (B1 token set, B4 vector, B2 / B3 prefixes).

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a (D-72).

Fixed choices of this build:
- P_ans: the designated marker is the LAST "\\nAnswer:" in B0's generation; j = the smallest k with
  len(decode(gen[:k])) >= the marker's end; P_ans = n_prompt + j − 1 (absolute). If decode(gen[:j]) does not end with
  "\\nAnswer:" (the ':' merged into a longer token) the row has no usable P_ans and is NM for P_ans analyses
  (counted; §49 handled merged tokens the same way).
- Sealed splits (H, Tcity; T-format dropped before F0): `rows(split)` refuses them unless `allow_sealed=True`, which only the guarded
  confirmatory stage passes (after the F2 commit check).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import grade51 as GR  # noqa: E402
from memctl.analysis import why50 as W  # noqa: E402

CASES = ROOT / "runs/exp51/cases51.jsonl"
RUN = ROOT / "runs/exp51/run"
SEALED = ("H", "Tcity")                     # T-format dropped before F0 (build review ruling 2)
OPEN_SPLITS = ("E", "S")
B1_GRID = (-1.0, -2.0, -4.0, -8.0, -16.0, float("-inf"))
B1_SIGNED_GRID = (-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0)
B4_GRID = (0.5, 1.0, 2.0, 4.0)
B4_FILE = ROOT / "runs/exp38_dev/qwen3b/directions.npz"
B5_DOSES = (0.25, 0.5, 0.75, 1.0)
B6_SEEDS = (1, 2, 3, 4, 5)
CAND_DOSES = (0.5, 1.0, 1.5, 2.0)
B2_TAIL = "\nDo not reply unknown.\nAnswer:"


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def all_rows(path: Path = CASES) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def rows(split: str, path: Path = CASES, allow_sealed: bool = False) -> list[dict]:
    if split in SEALED and not allow_sealed:
        raise PermissionError(f"{split} is sealed until the F2 freeze")
    return [r for r in all_rows(path) if r["split"] == split]


def cities_for(row: dict, t_cities: list[str]) -> list[str]:
    """Recognised cities: both pools (§13.5)."""
    return list(GR.MAIN_CITIES) + list(t_cities)


def t_cities(counts: Path = ROOT / "runs/exp51/cases51_counts.json") -> list[str]:
    return json.loads(Path(counts).read_text())["design"]["t_cities"]


def context_of(row: dict) -> str:
    """Memory plus question, for the quoted-mention rule."""
    return row["prompt"].split("## Memory")[1]


def grade(row: dict, text: str, tc: list[str]) -> dict:
    g = GR.classify(text, row["gold"], cities_for(row, tc), context_of(row))
    return {"cls": g["cls"], "city": g["city"], "span": g["span"]}


# ---------------------------------------------------------------------------------------------- B0 store

def designated(e_decode, gen: list[int]) -> dict:
    """The last marker's token index j (gen[:j] ends with the marker) or why not."""
    text = e_decode(gen)
    i = text.rfind(GR.MARKER)
    if i < 0:
        return {"j": None, "ok": False, "why": "no marker"}
    end = i + len(GR.MARKER)
    for k in range(1, len(gen) + 1):
        head = e_decode(gen[:k])
        if len(head) >= end:
            if head.endswith(GR.MARKER) and len(head) == end:
                return {"j": k, "ok": True, "why": None}
            return {"j": k, "ok": False, "why": f"merged token {head[-12:]!r}"}
    return {"j": None, "ok": False, "why": "marker not reached"}


def b0_path(split: str, case_id: str, run: Path = RUN) -> Path:
    return Path(run) / "b0" / split / f"{case_id}.json"


def b0(split: str, case_id: str, run: Path = RUN) -> dict | None:
    return read_json(b0_path(split, case_id, run))


def prefix_ids(rec: dict) -> list[int]:
    """B0's stored ids through the designated marker's ':' (prompt + gen[:j])."""
    return list(rec["prompt_ids"]) + list(rec["gen_ids"][:rec["j"]])


def population(row: dict, rec: dict, tc: list[str]) -> list[str]:
    """§4.1 labels of one row from its B0 record."""
    out = []
    if not rec.get("p_ans_ok"):
        out.append("NM")
    cls = rec["cls"]
    if row["set"] == "ANS":
        if cls == "abstention":
            out.append("FA")
            if W.answer_bearing(GR.note_text(rec["text"]), row["gold"], row["task"], cities_for(row, tc)):
                out.append("FA-ab")
        elif cls == "gold":
            out.append("C")
        elif cls in ("wrong", "conflicting", "other", "mixed"):
            out.append("W")
    elif row["set"] in ("UNA-rel", "UNA-swap"):
        out.append("UNA")
    elif row["set"] == "CF":
        out.append("CF")
    return out


# ---------------------------------------------------------------------------------------------- comparators

def b1_token_set(tok) -> list[int]:
    """Every vocabulary id whose decoded, lower-cased, stripped text begins with 'unknown' (frozen at F0)."""
    out = []
    for i in range(len(tok)):
        s = tok.decode([i]).lower().strip()
        if s.startswith("unknown"):
            out.append(i)
    return out


def b4_vector(alpha: float, path: Path = B4_FILE) -> np.ndarray:
    """§38 v_29 (vnorm units): alpha * v."""
    v = np.load(path)["L29"].astype(np.float64)
    return (alpha * v).astype(np.float32)


def b4_sha(path: Path = B4_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def b2_prefix(e, row: dict, rec: dict) -> list[int]:
    """Controlled-prefix B2: B0's stored prompt ids and note ids, then "\\nDo not reply unknown.\\nAnswer:"; P_ans = the
    new marker's ':' (role alignment, §44). If B0's last note token also carries the start of the marker (a merged
    ".\\n"), the note keeps the ids before it and the leftover note characters are encoded with the tail (counted
    in `b2_retokenised`)."""
    ids, left = note_ids(e, rec)
    return list(rec["prompt_ids"]) + ids + e.encode(left + B2_TAIL)


def note_ids(e, rec: dict) -> tuple[list[int], str]:
    """B0's generated ids before the designated marker (the longest gen[:m] whose text is a prefix of the note) and
    the note characters left over (non-empty only when a token straddles the note / marker boundary)."""
    gen, j = rec["gen_ids"], rec["j"]
    target = e.decode(gen[:j])[: -len(GR.MARKER)]
    m = j
    while m > 0 and not target.startswith(e.decode(gen[:m])):
        m -= 1
    return list(gen[:m]), target[len(e.decode(gen[:m])):]


def b3_prefix(e, row: dict, rec: dict) -> list[int]:
    """Controlled-prefix B3: the modified instruction's prompt ids, then B0's stored generated ids through its marker
    (note and marker unchanged, no re-tokenisation)."""
    return e.chat_ids(row["prompt_b3"]) + list(rec["gen_ids"][:rec["j"]])
