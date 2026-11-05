"""§45: descriptive J-lens maps of answer computation on controlled tasks (Qwen 3B).

Spec: research/proposals/2026-10-30-exp45-jlens-maps.md (rev 0, 78d5ad1), D-49. One greedy generation (<= 16 new
tokens), then one forward over prompt + generated ids with every block's residual recorded; at every recorded layer
and every chosen position, for each lens (the §34 56-prompt J-lens, the refit J-lens, the logit lens): the top-10
token ids and the rank (1-based, strictly greater, as §34) and log-probability of every pinned token. The final layer
is the model's own logits for every lens (as `jlens_readout.rank_table`).

Pinned tokens: the first content token of every entity, in two surface forms, "sp" (" " + text, the in-sentence form)
and "bare" (no leading space, the form the assistant turn starts with). Leading determiners are dropped ("The Song"
-> "Song") and leading whitespace-only tokens skipped (Qwen splits " 4" into " " and "4"), the §44 lesson. ` unknown`
and `unknown` are always pinned. Ids must be distinct across entities (two entities sharing a first content token
would make a readout ambiguous); the two forms of one entity may coincide (digits) and are then one pinned id.

Outcome (the §23 normalised match, memctl.metrics): the answer is the text after the last `Answer:` marker, or the
whole output when there is none (as memctl.agents.llm.LLMAgent); gold = score_answer(...)["correct"] against the gold,
competitor = the same test against a competitor, abstain = memctl.metrics.is_refusal with neither hit, other = the
rest (both gold and a competitor, or a refusal beside a hit, is "other" with ambiguous=True).

Readability (proposal §4 as refined by rev 1, build review 2d7122f): a pinned entity is readable on a prompt if its
" X" (space-form) id is in the lens top-10 at a counted position, or its bare "X" id is in the top-10 at the position
that predicts the first generated token (the last prompt token: the readout at position p is the distribution over
token p+1), at any layer ("before the final layer" for the gold criterion (a)). Counted positions: the last prompt
token and the answer span, i.e. every position from the one that predicts the first answer token to the last
generated token (F1: all generated tokens; F2 and stage 3: the tokens after "Answer:", none when the marker is
absent). First readability = the earliest layer with a hit at a counted position (the earliest such position on it).
Beside: "last prompt token only", "any generated token" (the same form rule, every generated position) and "either
form anywhere" (both forms at every position from the last prompt token on).

``torch`` and ``jlens`` are imported only inside the functions that run a model.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from memctl.agents.llm import final_answer
from memctl.analysis.jacobian import _ids
from memctl.metrics import contains_answer, is_refusal, score_answer, token_f1

DETERMINERS = ("the", "a", "an")
FORMS = ("sp", "bare")
OUTCOMES = ("gold", "competitor", "other", "abstain")
K = 10
MAX_NEW = 16                 # F1 (rev 1 ruling 2)
MAX_NEW_NOTE = 160           # F2 and stage 3: up to 160 new tokens, stopping at the end of the "Answer:" line
ANSWER_MARKER = "answer:"
MAX_SEQ_LEN = 4096
LENS_NAMES = ("j56", "jrefit", "logit")      # array order; a run without the refit lens has ("j56", "logit")


# ---------------------------------------------------------------------------------------------- pinned tokens

def strip_determiner(text: str) -> str:
    """Drop one leading determiner word when more words follow ("The Song of Achilles" -> "Song of Achilles")."""
    words = text.strip().split(None, 1)
    if len(words) == 2 and words[0].lower() in DETERMINERS:
        return words[1]
    return text.strip()


def first_content_token(tokenizer, text: str, form: str = "sp") -> tuple[int, str]:
    """(id, decoded text) of the first content token of `text` in `form` ("sp": after a leading space, "bare": none):
    a leading determiner word is dropped first, then whitespace-only tokens are skipped."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}")
    body = strip_determiner(text)
    if not body:
        raise ValueError(f"no content in {text!r}")
    ids = _ids(tokenizer, (" " if form == "sp" else "") + body)
    for token in ids:
        piece = tokenizer.decode([token])
        if piece.strip():
            return int(token), piece
    raise ValueError(f"no content token in {text!r}")


def pin_tokens(tokenizer, entities: Sequence[tuple[str, str]], unknown: bool = True,
               fixed: dict[tuple[str, str], int] | None = None, strict: bool = True,
               dropped: list | None = None) -> list[dict]:
    """Pinned ids for (role, surface) pairs, both forms each, plus ` unknown` / `unknown`.

    Returns one entry per distinct id: {"id", "text", "surface", "forms", "roles"}. A surface named under several
    roles (gold and entity) is one entry with every role. `fixed` maps (surface, form) to an id chosen elsewhere (the
    §44 gate's read ids). Two different surfaces sharing an id: ValueError when `strict` (the case list), otherwise
    the later (surface, form) is left out and appended to `dropped` (stage 3: free-text golds may collide)."""
    pairs = list(entities) + ([("unknown", "unknown")] if unknown else [])
    by_id: dict[int, dict] = {}
    for role, surface in pairs:
        for form in FORMS:
            if fixed and (surface, form) in fixed:
                token = int(fixed[(surface, form)])
                piece = tokenizer.decode([token])
            else:
                token, piece = first_content_token(tokenizer, surface, form)
            entry = by_id.get(token)
            if entry is None:
                by_id[token] = {"id": token, "text": piece, "surface": surface, "forms": [form], "roles": [role]}
                continue
            if entry["surface"] != surface:
                if not strict:
                    if dropped is not None:
                        dropped.append({"surface": surface, "form": form, "id": token, "kept": entry["surface"]})
                    continue
                raise ValueError(f"pinned token {piece!r} ({token}) is the first content token of both "
                                 f"{entry['surface']!r} and {surface!r}")
            if form not in entry["forms"]:
                entry["forms"].append(form)
            if role not in entry["roles"]:
                entry["roles"].append(role)
    return list(by_id.values())


def entity_columns(pinned: list[dict], surface: str) -> list[int]:
    """Indices into the pinned list of every id of one surface (its sp and bare forms)."""
    return [i for i, p in enumerate(pinned) if p["surface"] == surface]


def role_surfaces(pinned: list[dict], role: str) -> list[str]:
    seen = []
    for p in pinned:
        if role in p["roles"] and p["surface"] not in seen:
            seen.append(p["surface"])
    return seen


# ---------------------------------------------------------------------------------------------- outcome

def answer_text(output: str) -> tuple[str, bool]:
    """(the graded text, whether an `Answer:` marker was present)."""
    parsed = final_answer(output or "")
    return (output or "", False) if parsed is None else (parsed, True)


def _hit(answer: str, target: str) -> bool:
    return bool(target) and (contains_answer(answer, target) or token_f1(answer, target) >= 0.5)


def classify(output: str, gold: str, competitors: Sequence[str] = ()) -> dict:
    """The §23 normalised match on one output: {"outcome", "ambiguous", "answer", "answer_marker", "hits"}."""
    answer, marked = answer_text(output)
    gold_hit = bool(score_answer(answer, gold)["correct"]) if answer.strip() else False
    comp_hits = [c for c in competitors if c != gold and _hit(answer, c)]
    refusal = is_refusal(answer)
    if gold_hit and not comp_hits and not refusal:
        outcome = "gold"
    elif comp_hits and not gold_hit and not refusal:
        outcome = "competitor"
    elif refusal and not gold_hit and not comp_hits:
        outcome = "abstain"
    else:
        outcome = "other"
    ambiguous = (gold_hit and bool(comp_hits)) or (refusal and (gold_hit or bool(comp_hits)))
    return {"outcome": outcome, "ambiguous": bool(ambiguous), "answer": answer, "answer_marker": marked,
            "hits": {"gold": gold_hit, "competitors": comp_hits, "refusal": refusal}}


# ---------------------------------------------------------------------------------------------- readability

def readability(ranks: np.ndarray, columns: Sequence[int], layers: Sequence[int], from_pos: int, *, k: int = K,
                exclude_final: bool = False) -> dict:
    """One readability call. `ranks` [L, P, T] (one lens); `columns` the pinned indices of one entity (its forms);
    positions from index `from_pos` (the last prompt token) to the end; layers all, or all but the last.

    Returns {"readable", "first_layer", "first_depth" (layer / n_layers, n_layers = layers[-1] + 1), "first_pos"
    (index into the stored positions, the earliest position at the first layer), "best_rank"}."""
    if not columns:
        return {"readable": None, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": None}
    sub = ranks[:, from_pos:, :][:, :, list(columns)].min(axis=-1)        # [L, P']
    layers = list(layers)
    n_layers = int(layers[-1]) + 1
    if exclude_final:
        sub, layers = sub[:-1], layers[:-1]
    hit = sub <= k
    best = int(sub.min()) if sub.size else None
    if not hit.any():
        return {"readable": False, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": best}
    li = int(np.argmax(hit.any(axis=1)))
    pi = int(np.argmax(hit[li]))
    layer = int(layers[li])
    return {"readable": True, "first_layer": layer, "first_depth": layer / n_layers if n_layers else None,
            "first_pos": from_pos + pi, "best_rank": best}


def form_columns(pinned: list[dict], surface: str) -> tuple[list[int], list[int]]:
    """(space-form columns, bare-form columns) of one surface; an id that is both (digits) is in both lists."""
    cols = entity_columns(pinned, surface)
    return [i for i in cols if "sp" in pinned[i]["forms"]], [i for i in cols if "bare" in pinned[i]["forms"]]


def readability_rule(ranks: np.ndarray, sp_cols: Sequence[int], bare_cols: Sequence[int], layers: Sequence[int],
                     counted: Sequence[int], bare_at: Sequence[int], *, k: int = K, exclude_final: bool = False) -> dict:
    """The rev 1 call: the space-form ids at the `counted` positions, the bare-form ids at the `bare_at` positions
    (indices into the stored positions). Same return as `readability`."""
    if not sp_cols and not bare_cols:
        return {"readable": None, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": None}
    layers = list(layers)
    n_layers = int(layers[-1]) + 1
    sub = ranks[:-1] if exclude_final else ranks
    use = layers[:-1] if exclude_final else layers
    P = ranks.shape[1]
    best = np.full((sub.shape[0], P), np.iinfo(np.int64).max, dtype=np.int64)
    if sp_cols and len(counted):
        idx = list(counted)
        best[:, idx] = np.minimum(best[:, idx], sub[:, idx, :][:, :, list(sp_cols)].min(-1))
    if bare_cols and len(bare_at):
        idx = list(bare_at)
        best[:, idx] = np.minimum(best[:, idx], sub[:, idx, :][:, :, list(bare_cols)].min(-1))
    considered = sorted(set(counted if sp_cols else []) | set(bare_at if bare_cols else []))
    if not considered:
        return {"readable": False, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": None}
    view = best[:, considered]
    hit = view <= k
    b = int(view.min())
    if not hit.any():
        return {"readable": False, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": b}
    li = int(np.argmax(hit.any(axis=1)))
    pi = considered[int(np.argmax(hit[li]))]
    layer = int(use[li])
    return {"readable": True, "first_layer": layer, "first_depth": layer / n_layers, "first_pos": pi, "best_rank": b}


ECHO_RULES = ("A", "B")


def echo_counted(counted: Sequence[int], input_at: Sequence[int], target_ids, rule: str, lp: int) -> list[int]:
    """The counted positions left for one target under a post hoc echo rule (§45 results review, 1a6e1e5).

    `input_at[p]` is the token fed in at stored position p (the readout at p is the distribution over the next token);
    `target_ids` the target's pinned ids (both forms); `lp` the last prompt token's stored index. Stored positions
    from `lp` on are contiguous sequence positions (`map_prompt` keeps every one).
    Rule "A" (input-token exclusion): drop p when the token fed in at p is the target.
    Rule "B" (production positions): keep only the positions up to and including the one that produces the target's
    first occurrence after the prompt, i.e. drop p once the target is among the tokens fed in at lp+1..p.
    The last prompt token is never dropped by B (nothing generated yet) and, its input being a template token, not
    by A in practice; the bare form at the last prompt token is handled by the caller, unchanged."""
    ids = {int(t) for t in target_ids}
    if rule == "A":
        return [p for p in counted if int(input_at[p]) not in ids]
    if rule == "B":
        first = next((q for q in range(lp + 1, len(input_at)) if int(input_at[q]) in ids), None)
        return [p for p in counted if first is None or p < first]
    raise ValueError(f"rule must be one of {ECHO_RULES}, not {rule!r}")


def combine_calls(results: Sequence[dict]) -> dict:
    """Several per-surface readability calls as one entity-set call: readable if any is; the earliest layer wins (the
    earliest position on it); best rank the minimum."""
    results = [c for c in results if c["readable"] is not None]
    if not results:
        return {"readable": None, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": None}
    ranks = [c["best_rank"] for c in results if c["best_rank"] is not None]
    best = min(ranks) if ranks else None
    hits = [c for c in results if c["readable"]]
    if not hits:
        return {"readable": False, "first_layer": None, "first_depth": None, "first_pos": None, "best_rank": best}
    h = min(hits, key=lambda c: (c["first_layer"], c["first_pos"]))
    return {**h, "best_rank": best}


def gold_trajectory(ranks: np.ndarray, columns: Sequence[int], from_pos: int) -> np.ndarray:
    """Per layer, the best rank of an entity over the positions from `from_pos` on (the stage 2 comparison line)."""
    return ranks[:, from_pos:, :][:, :, list(columns)].min(axis=(1, 2))


def departure_layer(fail_traj: np.ndarray, pair_traj: np.ndarray, layers: Sequence[int], k: int = K) -> int | None:
    """The first layer at which the paired success has the gold in the top-k and the failure does not (descriptive)."""
    for layer, f, s in zip(layers, fail_traj, pair_traj):
        if s <= k < f:
            return int(layer)
    return None


# ---------------------------------------------------------------------------------------------- statistics

def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None, float | None]:
    """(share, lower, upper) Wilson score interval; (None, None, None) for n = 0."""
    if n <= 0:
        return None, None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def criterion_verdict(successes: int, n: int, share_min: float = 0.70, lower_min: float = 0.55) -> str:
    """§5: share >= 0.70 with the Wilson lower bound >= 0.55 -> "recovers"; else "not recovered"; n = 0 -> "no data"."""
    p, lo, _ = wilson(successes, n)
    if p is None:
        return "no data"
    return "recovers" if (p >= share_min and lo >= lower_min) else "not recovered"


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation with average ranks for ties; None when either side is constant or n < 3."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def quartiles(values: Sequence[float]) -> list[float] | None:
    vals = [v for v in values if v is not None]
    return [float(q) for q in np.quantile(vals, [0.25, 0.5, 0.75])] if vals else None


# ---------------------------------------------------------------------------------------------- the model pass

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class MapInputs:
    """One prompt to map. `chat_text` is the full chat (template + user message + generation prompt); `positions`
    None = every position from the first prompt token to the last generated token, else the prompt positions to keep
    (the last prompt token and every generated position are always added)."""
    key: str
    chat_text: str
    pinned_ids: list[int]
    positions: list[int] | None = None
    max_new: int = MAX_NEW
    answer_span: str = "all"           # "all" (F1: every generated token) or "after_marker" (F2, stage 3)


def lens_to(lens, device) -> None:
    lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}


def lens_mismatch(lens, lens_model) -> str | None:
    widths = {tuple(J.shape) for J in lens.jacobians.values()}
    if widths != {(lens_model.d_model, lens_model.d_model)}:
        return f"Jacobian shapes {sorted(widths)} do not match the model's d_model {lens_model.d_model}"
    if not lens.source_layers or max(lens.source_layers) >= lens_model.n_layers:
        return f"source layer {max(lens.source_layers, default=None)} is not below the model's n_layers {lens_model.n_layers}"
    return None


def answer_line_done(text: str) -> bool:
    """True once the text holds an "Answer:" marker followed by a non-empty answer and the end of that line."""
    marker = text.lower().rfind(ANSWER_MARKER)
    if marker < 0:
        return False
    lines = text[marker + len(ANSWER_MARKER):].split("\n")
    return any(line.strip() for line in lines[:-1])          # a non-empty answer line that has been terminated


def generate_greedy(hf_model, input_ids: list[int], max_new: int, eos_ids: Sequence[int], tokenizer=None,
                    stop_after_answer: bool = False) -> list[int]:
    """Greedy continuation ids (the EOS, if produced, is kept as the last id). `stop_after_answer`: stop once the
    generated text completes its "Answer:" line (the newline token is kept)."""
    import torch
    import transformers

    device = next(hf_model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    n0 = len(input_ids)
    criteria = None
    if stop_after_answer:
        class AnswerLine(transformers.StoppingCriteria):
            def __call__(self, seq, scores, **kwargs):
                done = answer_line_done(tokenizer.decode(seq[0, n0:].tolist(), skip_special_tokens=True))
                return torch.full((seq.shape[0],), done, dtype=torch.bool, device=seq.device)
        criteria = transformers.StoppingCriteriaList([AnswerLine()])
    with torch.no_grad():
        out = hf_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                                temperature=None, top_p=None, top_k=None, eos_token_id=list(eos_ids),
                                pad_token_id=list(eos_ids)[0], stopping_criteria=criteria)
    return [int(t) for t in out[0, n0:].tolist()]


def answer_start_token(tokenizer, gen_ids: Sequence[int]) -> int | None:
    """Index of the first generated token that starts after the last "Answer:" marker (None: no marker, or nothing
    after it). Char offsets come from decoding growing prefixes."""
    text = tokenizer.decode(list(gen_ids), skip_special_tokens=True)
    marker = text.lower().rfind(ANSWER_MARKER)
    if marker < 0:
        return None
    end = marker + len(ANSWER_MARKER)
    for j in range(len(gen_ids)):
        if len(tokenizer.decode(list(gen_ids[:j]), skip_special_tokens=True)) >= end:
            return j
    return None


def torch_ranks(logits, token_ids: Sequence[int], chunk: int = 4):
    """1 + #(logits strictly greater than each pinned token's), per position: [P, T] (torch). Refuses non-finite."""
    import torch

    if not torch.isfinite(logits).all():
        raise ValueError(f"non-finite logits ({int((~torch.isfinite(logits)).sum())} entries); ranks would be meaningless")
    idx = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
    chosen = logits[:, idx]                                               # [P, T]
    out = []
    for s in range(0, len(token_ids), chunk):
        c = chosen[:, s:s + chunk]
        out.append((logits[:, None, :] > c[:, :, None]).sum(-1))
    return torch.cat(out, dim=1) + 1


def readout_layer(logits, pinned_ids: Sequence[int], k: int = K):
    """(ranks [P, T] int32, log-probs [P, T] float32, top ids [P, k] int32, top log-probs [P, k] float32) as numpy."""
    import torch

    logits = logits.float()
    ranks = torch_ranks(logits, pinned_ids)
    logp = torch.log_softmax(logits, dim=-1)
    idx = torch.as_tensor(list(pinned_ids), dtype=torch.long, device=logits.device)
    top_lp, top_ids = torch.topk(logp, k=min(k, logp.shape[-1]), dim=-1)
    return (ranks.to(torch.int32).cpu().numpy(), logp[:, idx].cpu().numpy(),
            top_ids.to(torch.int32).cpu().numpy(), top_lp.cpu().numpy())


def map_prompt(hf_model, lens_model, tokenizer, x: MapInputs, lenses: dict, *, k: int = K, eos_ids=None,
               break_bos: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    """Generate, forward, read every lens at every layer and the chosen positions.

    `lenses`: name -> JacobianLens, or None for the logit lens; names in LENS_NAMES order. Returns (payload, arrays):
    arrays ranks / logp [n_lens, L, P, T], top_ids / top_logp [n_lens, L, P, k], layers [L], positions [P],
    input_ids [n]. `break_bos` (D-30 negative test only) prepends an <|endoftext|>, which the smoke's BOS check refuses."""
    import torch
    from jlens.hooks import ActivationRecorder

    from memctl.analysis.jlens_readout import encode_chat_ids

    started = time.perf_counter()
    prompt_ids = encode_chat_ids(tokenizer, x.chat_text, MAX_SEQ_LEN)
    if len(prompt_ids) >= MAX_SEQ_LEN:
        raise ValueError(f"{x.key}: chat text reaches {MAX_SEQ_LEN} tokens; truncated")
    if break_bos:
        prompt_ids = [int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))] + prompt_ids
    if eos_ids is None:
        eos_ids = [t for t in {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id} if t is not None]
    t_gen = time.perf_counter()
    note = x.answer_span == "after_marker"
    gen_ids = (generate_greedy(hf_model, prompt_ids, x.max_new, eos_ids, tokenizer, stop_after_answer=note)
               if x.max_new > 0 else [])
    gen_s = time.perf_counter() - t_gen
    full = prompt_ids + gen_ids
    last_prompt = len(prompt_ids) - 1
    if x.positions is None:
        positions = list(range(len(full)))
    else:
        positions = sorted({p for p in x.positions if 0 <= p < last_prompt} | set(range(last_prompt, len(full))))
    final = lens_model.n_layers - 1
    source = sorted(set().union(*[set(l.source_layers) for l in lenses.values() if l is not None]) or set())
    layers = sorted(set(range(lens_model.n_layers)) if not source else (set(source) | {final}))
    names = [n for n in LENS_NAMES if n in lenses]
    for lens in lenses.values():
        if lens is not None and set(lens.source_layers) != set(source):
            raise ValueError(f"{x.key}: the J-lenses cover different source layers")
    ids_t = torch.tensor([full], dtype=torch.long, device=lens_model.input_device)
    pos_t = torch.as_tensor(positions, dtype=torch.long)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(ids_t)
        residuals = {l: recorder.activations[l][0][pos_t.to(recorder.activations[l].device)].float() for l in layers}
    T, P, L = len(x.pinned_ids), len(positions), len(layers)
    ranks = np.zeros((len(names), L, P, T), dtype=np.int32)
    logp = np.zeros((len(names), L, P, T), dtype=np.float32)
    top_ids = np.zeros((len(names), L, P, k), dtype=np.int32)
    top_lp = np.zeros((len(names), L, P, k), dtype=np.float32)
    with torch.no_grad():
        final_read = readout_layer(lens_model.unembed(residuals[final]), x.pinned_ids, k)
        for li, layer in enumerate(layers):
            for ni, name in enumerate(names):
                if layer == final:
                    r = final_read
                else:
                    lens = lenses[name]
                    resid = residuals[layer] if lens is None else lens.transport(residuals[layer], layer)
                    r = readout_layer(lens_model.unembed(resid), x.pinned_ids, k)
                ranks[ni, li], logp[ni, li], top_ids[ni, li], top_lp[ni, li] = r
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    lp_index = positions.index(last_prompt)
    # counted positions (indices into `positions`): the last prompt token and the answer span, i.e. from the position
    # predicting the first answer token (generated token j0 is predicted at sequence position last_prompt + j0) on
    if note:
        j0 = answer_start_token(tokenizer, gen_ids)
        span = [] if j0 is None else list(range(lp_index + j0, len(positions)))
    else:
        j0 = 0
        span = list(range(lp_index, len(positions)))
    counted = sorted({lp_index} | set(span))
    bos_like = {t for t in (getattr(tokenizer, "bos_token_id", None), tokenizer.convert_tokens_to_ids("<|endoftext|>"))
                if t is not None}
    payload = {"key": x.key, "n_prompt_tokens": len(prompt_ids), "n_generated": len(gen_ids), "n_tokens": len(full),
               "generated_ids": gen_ids, "generated_text": gen_text, "last_prompt_index": lp_index,
               "answer_span_rule": x.answer_span, "answer_start_token": j0, "answer_span_found": j0 is not None,
               "counted_positions": counted, "max_new": x.max_new,
               "lens_names": names, "n_layers": lens_model.n_layers, "layers": layers, "k": k,
               "n_bos": sum(1 for t in prompt_ids if t in bos_like),
               "first_id": prompt_ids[0], "chat_sha256": sha256_text(x.chat_text),
               "position_tokens": [tokenizer.decode([full[p]]) for p in positions],
               "gen_s": gen_s, "elapsed_s": time.perf_counter() - started}
    arrays = {"ranks": ranks, "logp": logp.astype(np.float16), "top_ids": top_ids, "top_logp": top_lp.astype(np.float16),
              "layers": np.asarray(layers, dtype=np.int32), "positions": np.asarray(positions, dtype=np.int32),
              "input_ids": np.asarray(full, dtype=np.int32)}
    return payload, arrays


def final_top1_at_last_prompt(arrays: dict[str, np.ndarray], payload: dict) -> int:
    """The final-layer top-1 id at the last prompt token (= the first greedy token), identical across lenses."""
    return int(arrays["top_ids"][0, -1, payload["last_prompt_index"], 0])
