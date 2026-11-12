"""Stage 0 shared pieces: data, split, models, seeds, prompt assembly (the authors' protocol), JSONL store.

Prompt assembly reproduces dynamic-lm/interrupt-lrm (vendored in selfdoubt/vendor/interrupt_lrm):
  FULL   = apply_chat_template([system: wrapper.system_prompt, user: "Please reason step by step, and put your final
           answer within \\boxed{}.\\nQ: " + revised_problem + "\\nA:"], add_generation_prompt, enable_thinking=True)
  cut    = reasoning trace (FULL output up to "</think>"), tokenised, first int(X * n_tokens) ids, decoded, and
           "<think>\\n" re-added if lost (authors' truncate_reasoning_trace; their cut is in TOKENS, not characters)
  U0/UG  = FULL prompt + cut + reasoning_prefix + update + reasoning_suffix   (assistant-role injection, no </think>)
  N      = FULL prompt + cut                                                   (ours: the same prefix, no update)
  X = 0  = the same formula at X = 0: the cut is empty, so the trace is "<think>\\n" + wrapper.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from selfdoubt.vendor.interrupt_lrm import UPSTREAM_COMMIT
from selfdoubt.vendor.interrupt_lrm import prompt_utils_qwen as PU

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / "vendor" / "interrupt_lrm"
RUNS = ROOT / "runs" / "selfdoubt"

DATASET = "dynamic-lm/update-interrupt-benchmark"
DATASET_REVISION = "6ac4ea4baadeccafbb452c1649c90e24ffac4cfc"   # HF API sha, 2026-11-08
DATASET_FILE = "data_math.parquet"                             # config "Math", split "test" (README configs)
SOURCES = ("gsm8k", "math500")
SPLIT_SEED = 56

# Pinned with HfApi().model_info(...).sha on 2026-11-08. All three share one chat template (sha256 a55ee1b1...).
MODELS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",   # local CPU smoke only
}

WRAPPERS = {"U0": "wo_prompt_guidance.json", "UG": "w_prompt_guidance.json"}
CUTS = (0.3, 0.5, 0.9)
X0 = 0.0
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)


def model_tag(model: str) -> str:
    return model.split("/")[-1]


# ---------------------------------------------------------------- data and split

def load_items() -> list[dict]:
    """All 1,060 Math rows at the pinned dataset revision (parquet read directly; `datasets` is not needed)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
    rows = pq.read_table(p).to_pylist()
    assert len(rows) == 1060, len(rows)
    return rows


def make_split(items: list[dict], seed: int = SPLIT_SEED) -> dict:
    """50/50 dev/held-out per source (gsm8k, math500), ids sorted then shuffled with random.Random(seed)."""
    out = {"seed": seed, "dataset": DATASET, "dataset_revision": DATASET_REVISION, "dev": {}, "heldout": {}}
    for src in SOURCES:
        ids = sorted(int(r["id"]) for r in items if r["source"] == src)
        rng = random.Random(f"{seed}:{src}")
        rng.shuffle(ids)
        half = len(ids) // 2
        out["dev"][src] = sorted(ids[:half])
        out["heldout"][src] = sorted(ids[half:])
    return out


def split_bytes(split: dict) -> bytes:
    return (json.dumps(split, indent=1, sort_keys=True) + "\n").encode()


def ensure_split(items: list[dict], path: Path | None = None) -> tuple[dict, str]:
    """Write runs/selfdoubt/split.json (+ .sha256) once; afterwards verify it is unchanged. Returns (split, sha256)."""
    path = path or RUNS / "split.json"
    split = make_split(items)
    b = split_bytes(split)
    sha = hashlib.sha256(b).hexdigest()
    if path.exists():
        old = path.read_bytes()
        if old != b:
            raise SystemExit(f"{path} differs from the recomputed split (seed {SPLIT_SEED}); refusing to continue")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b)
        path.with_suffix(".sha256").write_text(f"{sha}  split.json\n")
    return split, sha


def stub_items(items: list[dict], split: dict, n_per_source: int = 5) -> list[dict]:
    """Stub: dev only; first n GSM8K dev ids; the n MATH-500 dev items with the longest original_answer (tie: id)."""
    by_id = {int(r["id"]): r for r in items}
    g = [by_id[i] for i in split["dev"]["gsm8k"][:n_per_source]]
    m = sorted((by_id[i] for i in split["dev"]["math500"]), key=lambda r: (-len(r["original_answer"]), int(r["id"])))
    return g + m[:n_per_source]


def shuffled_order(items: list[dict], split: dict, src: str, which: str, seed: int = SPLIT_SEED) -> list[int]:
    """The ids of split[which][src] in the order the seed-56 shuffle (make_split) produced them. split.json stores each
    half sorted; this recovers the shuffle order, which is content-blind. Used to take "the first N" of a half."""
    ids = sorted(int(r["id"]) for r in items if r["source"] == src)
    rng = random.Random(f"{seed}:{src}")
    rng.shuffle(ids)
    half = len(ids) // 2
    order = ids[:half] if which == "dev" else ids[half:]
    assert sorted(order) == list(split[which][src]), "split.json does not match the seed-56 shuffle"
    return order


def shuffled_heldout(items: list[dict], split: dict, src: str, n: int | None = None) -> list[int]:
    """First n held-out ids of a source in seed-56 shuffle order (PROPOSAL, Stage 0 plan of 2026-11-09)."""
    o = shuffled_order(items, split, src, "heldout")
    return o[:n] if n else o


def select_items(items: list[dict], split: dict, mode: str, which: str = "dev", limit: int | None = None,
                 sources=SOURCES) -> list[dict]:
    """stub / smoke: fixed dev items. full: per source, the ids of the chosen half in seed-56 shuffle order, the first
    `limit` of each source (all if None)."""
    if mode == "stub":
        return stub_items(items, split, 5)
    if mode == "smoke":
        s = stub_items(items, split, 1)
        return s[:limit] if limit else s
    by_id = {int(r["id"]): r for r in items}
    sel = []
    for src in sources:
        o = shuffled_order(items, split, src, which)
        sel += [by_id[i] for i in (o[:limit] if limit else o)]
    return sel


# ---------------------------------------------------------------- seeds and keys

def request_seed(model: str, item_id: int, arm: str, cut) -> int:
    h = hashlib.sha256(f"selfdoubt-stage0|{model}|{int(item_id)}|{arm}|{cut_str(cut)}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def cut_str(cut) -> str:
    return "-" if cut is None else f"{float(cut):.2f}"


def rec_key(model: str, item_id: int, arm: str, cut) -> str:
    return f"{model}|{int(item_id)}|{arm}|{cut_str(cut)}"


# ---------------------------------------------------------------- prompt assembly

def load_wrapper(arm: str) -> dict:
    return json.loads((VENDOR / "prompts" / "intervene" / WRAPPERS[arm]).read_text())


def system_prompt() -> str:
    a, b = load_wrapper("U0")["system_prompt"], load_wrapper("UG")["system_prompt"]
    assert a == b, "wo/w guidance system prompts differ; one FULL trace cannot serve both"
    return a


def full_prompt(tokenizer, model: str, problem: str) -> str:
    """The authors' format_math_initial_input_prompt, Qwen3 branch, with the update-aware system prompt."""
    msgs = PU.get_math_messages_qwen({"revised_problem": problem}, {"system_prompt": system_prompt()},
                                     "initial", "revised_problem")
    return PU.prompt_formatting_model(model, tokenizer, msgs)


def reasoning_trace(model: str, full_output: str) -> str:
    return PU.extract_reasoning_trace(full_output, model)


def total_reasoning_length(tokenizer, model: str, full_output: str) -> int:
    """As in the authors' inference_loop: token count of the reasoning trace of the initial output."""
    return len(tokenizer.encode(reasoning_trace(model, full_output)))


def cut_trace(tokenizer, model: str, full_output: str, x: float, unit: str = "token") -> tuple[str, int, int]:
    """Prefix of the reasoning trace at fraction x. Returns (prefix, cut position, total length) in `unit`s.

    unit="token" is the authors' rule (int(x * total_reasoning_length) tokens, decoded, "<think>\\n" re-added).
    unit="char" is the character-level alternative (not the authors'), kept only for a sensitivity check.
    """
    tr = reasoning_trace(model, full_output)
    if unit == "token":
        total = len(tokenizer.encode(tr))
        pos = int(x * total)
        return PU.truncate_reasoning_trace(model, tr, tokenizer, pos), pos, total
    if unit == "char":
        total = len(tr)
        pos = int(x * total)
        return PU.thinking_token_check(model, tr[:pos]), pos, total
    raise ValueError(unit)


def injected_text(arm: str, update: str) -> str:
    """The text inserted after the prefix, exactly as the authors' assistant-role path builds it."""
    w = load_wrapper(arm)
    upd = w.get("update_prefix", "") + update + w.get("update_suffix", "")
    return w.get("reasoning_prefix", "") + upd + w.get("reasoning_suffix", "")


def continuation_prompt(fprompt: str, prefix: str, arm: str, update: str) -> str:
    if arm == "N":
        return fprompt + prefix
    return fprompt + prefix + injected_text(arm, update)


def authors_update_prompt(tokenizer, model: str, fprompt: str, full_output: str, x: float, arm: str, update: str,
                          mode: str = "subsequent_intervene") -> str:
    """Call the authors' own format_subsequent_input_prompt (vendored verbatim). mode="subsequent_intervene" is the
    branch that injects the update text; the released CLI's mode "subsequent_interrupt_update" does not, because
    the mode names do not match (shown in selfdoubt/tests/test_stage0.py)."""
    args = SimpleNamespace(mode=mode, model_name=model, interrupt_pos=x, interrupt_role="assistant", task="math")
    ex = {"update": update, "formatted_input_prompt": [fprompt], "output": [full_output],
          "metadata": [{"total_reasoning_length": total_reasoning_length(tokenizer, model, full_output)}]}
    return PU.format_subsequent_input_prompt(args, ex, load_wrapper(arm), tokenizer)


# ---------------------------------------------------------------- request plan

@dataclass
class Request:
    key: str
    model: str
    item_id: int
    source: str
    arm: str           # FULL, N, U0, UG
    cut: float | None  # None for FULL
    prompt: str
    seed: int
    meta: dict


def plan_full(model: str, tokenizer, items: list[dict]) -> list[Request]:
    out = []
    for it in items:
        iid = int(it["id"])
        out.append(Request(rec_key(model, iid, "FULL", None), model, iid, it["source"], "FULL", None,
                           full_prompt(tokenizer, model, it["revised_problem"]), request_seed(model, iid, "FULL", None),
                           {}))
    return out


def plan_continuations(model: str, tokenizer, items: list[dict], full_recs: dict[int, dict],
                       cuts=CUTS, cut_unit: str = "token") -> list[Request]:
    out = []
    for it in items:
        iid = int(it["id"])
        fr = full_recs.get(iid)
        if fr is None:
            continue
        fp = full_prompt(tokenizer, model, it["revised_problem"])
        arms = [(x, a) for x in cuts for a in ("N", "U0", "UG")] + [(X0, "U0"), (X0, "UG")]
        for x, arm in arms:
            prefix, pos, total = cut_trace(tokenizer, model, fr["text"], x, cut_unit)
            prompt = continuation_prompt(fp, prefix, arm, it["update"])
            meta = {"cut_unit": cut_unit, "cut_pos": pos, "trace_len": total, "prefix_chars": len(prefix),
                    "injected": "" if arm == "N" else injected_text(arm, it["update"]),
                    "full_prompt_chars": len(fp)}
            out.append(Request(rec_key(model, iid, arm, x), model, iid, it["source"], arm, x, prompt,
                               request_seed(model, iid, arm, x), meta))
    return out


# ---------------------------------------------------------------- JSONL store (resumable)

class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.recs: dict[str, dict] = {}
        if path.exists():
            raw = path.read_bytes()
            if raw and not raw.endswith(b"\n"):          # torn last line from a kill mid-write: cut it off so the
                with path.open("r+b") as f:              # next append does not glue onto it
                    f.truncate(raw.rfind(b"\n") + 1)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # a torn last line from a kill mid-write: ignore, it is regenerated
                    continue
                self.recs[r["key"]] = r

    def done(self, key: str) -> bool:
        return key in self.recs

    def append(self, recs: list[dict]) -> None:
        with self.path.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        for r in recs:
            self.recs[r["key"]] = r

    def full_by_item(self, model: str) -> dict[int, dict]:
        return {r["item_id"]: r for r in self.recs.values() if r["model"] == model and r["arm"] == "FULL"}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def provenance() -> dict:
    return {"upstream": "https://github.com/dynamic-lm/interrupt-lrm", "upstream_commit": UPSTREAM_COMMIT,
            "dataset": DATASET, "dataset_revision": DATASET_REVISION, "split_seed": SPLIT_SEED}
