"""§38: a content-free "answer, don't refuse" direction on the 3B, with an evidence-cut plausibility control.

Pre-registration: research/proposals/2026-10-22-exp38-refusal-direction.md (rev 1, bb758c3) with amendment 1 (c367d28).
Pure logic plus the hooked engines; torch is imported lazily so the analysis package stays importable without it.

- Populations (from the §35 cache, `scripts/exp35_report.ReaderData`, marked rows after the §35 token check):
  R = arm A refuses (A row 'refusal', the §23 rule); pairs = R rows where B answers (B row 'refusal' false), the only
  rows the direction is estimated from; unseen = R minus pairs (B refuses too); C = A judged correct; damage = the
  first `DAMAGE_N` C rows by question_id (amendment 1.8).
- Prompts: A and B are §35's (`answer_selection.build_arm_input`): ids(chat text) + ids(note + suffix). The cut prompt
  is §35's N-cut prompt (`ablate_evidence`) + the N-cut arm's own note (`split_note` of the cached N-cut 'generated',
  byte-exact) + "Answer:"; no marker -> no note (amendment 1.4), flagged `cut_no_marker`.
- Position: `last` is the final prompt token, the ':' of "Answer:" (amendment 1.7). The prefix KV cache covers every
  prompt token BEFORE it; the ':' token is recomputed under each setting (amendment 1.9), so an add at layer L also
  changes its keys and values above L. Generation is greedy, stops at EOS or at a newline (the §35 phrase rule).
- Residual = the output of decoder layer L (`model.model.layers[L]`), at the current position; the final layer's
  output is before the final RMSNorm.
- Direction: v_L = mean over the reproduced pairs of resid_B(L, last) - resid_A(L, last). Steer adds alpha * v_L
  (unit 'vnorm', amendment 1.1; 'residnorm' = alpha * mean||resid_A(L, last)|| * v_hat is kept as an option, unused)
  at layer L at the ':' position and at every generated token. Random: a seeded Gaussian direction rescaled to the
  steer vector's norm. Reverse: minus the steer vector.
- Output classes (amendment 1.2): 'refusal' (the §23 rule on the phrase), 'degenerate' (empty after strip, no
  alphanumeric character, or a consecutive run of >= `DEGENERATE_REPEAT` identical token ids), else
  'answer'; ans = class == 'answer'.
- §39b (amendment 2 of the §39 proposal): `HookedEngine.run_forced` feeds a given token as the first generated token
  on the unpushed cache (':' run unpushed, no hook at any step, checked); `continuation_first_id` is the tokenisation
  of a forced text after "Answer:"; `ForcedArm` / `make_forced_row` stamp the rows.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from memctl.analysis import answer_selection as sel

# --- pre-registered numbers -----------------------------------------------------------------------------------
N_LAYERS = 36                       # Qwen2.5-3B-Instruct
HIDDEN = 2048
SWEEP_DEPTHS = (0.5, 0.6, 0.7, 0.8, 0.9)
SMALL_DEPTHS = (0.6, 0.8, 0.9)      # §5.6 fallback when the timing projects over 5 h
SHALLOW_DEPTH = 0.2                 # sanity S, descriptive (amendment 1.6)
ALPHAS = (1.0, 2.0, 4.0)            # amendment 1.1, unit ||v_L||
SMALL_ALPHAS = (4.0,)
UNITS = ("vnorm", "residnorm")
DEFAULT_UNIT = "vnorm"
MAX_NEW_TOKENS_LAPTOP = 16
MAX_NEW_TOKENS_POD = 64
REPRO_GATE = 0.8
SANITY_FINAL_GATE = 0.9
RANDOM_SHARE = 0.2                  # ans(Random) <= 0.2 * ans(Steer)
ANS_GATE = 0.5                      # ans(Steer) on the 28 dev R rows
DEGENERATE_MAX = 0.2                # a setting above this degenerate rate is not eligible (amendment 1.2)
DEGENERATE_REPEAT = 8
DAMAGE_N = 8
RANDOM_SEED = 38
TIMING_N = 5
PROJECTION_LIMIT_H = 5.0
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "greedy": True, "stop": "eos or newline"}
# Soft refusals (answer-class phrases that hedge): the pattern frozen in §41 rev 1 (877f86d), the §39 review's.
SOFT_REFUSAL_PATTERN = (r"not (specified|mentioned|provided|determined|explicitly|contain|named|available|stated|given)"
                        r"|does not contain|cannot be determined|no (information|mention)")
CLASSIFIER_VERSION = "v2: refusal = §23 rule on the phrase; degenerate = empty / no alnum / a run of >= 8 identical ids"


def depth_layer(depth: float, n_layers: int = N_LAYERS) -> int:
    """0.5 -> 18, 0.6 -> 22, 0.7 -> 25, 0.8 -> 29, 0.9 -> 32, 0.2 -> 7 for 36 layers."""
    return int(round(depth * n_layers))


def final_layer(n_layers: int = N_LAYERS) -> int:
    return n_layers - 1


def wording(max_new_tokens: int = MAX_NEW_TOKENS_LAPTOP) -> dict:
    return {"answer_marker": sel.ANSWER_MARKER, "b_instruction": sel.B_INSTRUCTION, "exp35_wording": sel.wording_sha256(),
            "cut": "ablate_evidence + N-cut note (split_note of generated, byte-exact; no marker -> no note) + Answer:",
            "position": "':' of Answer:, prefix cache excludes it", "residual": "decoder layer output, pre final norm",
            "sampling": SAMPLING, "max_new_tokens": max_new_tokens, "classifier": CLASSIFIER_VERSION}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def wording_sha256(max_new_tokens: int = MAX_NEW_TOKENS_LAPTOP) -> str:
    return sha256(json.dumps(wording(max_new_tokens), sort_keys=True))


# --- populations ----------------------------------------------------------------------------------------------

@dataclass
class Population:
    R: list[str]
    pairs: list[str]
    unseen: list[str]
    C: list[str]
    damage: list[str]
    b_outcome: dict[str, str]          # R rows: cached B 'correct' / 'wrong' / 'refusal'


def populations(data, damage_n: int = DAMAGE_N) -> Population:
    """`data`: a scripts/exp35_report.ReaderData for the 3B. Marked rows after the token check (`data.frozen`)."""
    frozen = sorted(q for q in data.frozen if q in data.rows["A"])
    R = [q for q in frozen if data.rows["A"][q]["refusal"]]
    pairs = [q for q in R if q in data.rows["B"] and not data.rows["B"][q]["refusal"]]
    C = [q for q in frozen if data.correct("A", q)]
    b_outcome = {}
    for q in R:
        row = data.rows["B"].get(q)
        b_outcome[q] = ("refusal" if row is None or row["refusal"] else
                        "correct" if data.correct("B", q) else "wrong")
    return Population(R, pairs, [q for q in R if q not in pairs], C, sorted(C)[:damage_n], b_outcome)


# --- prompts --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SteerInput:
    kind: str                  # 'A', 'B' or 'cut'
    question_id: str
    fold: int
    prompt: str                # user message, chat-formatted by the engine
    continuation: str          # appended after the chat text's ids
    input_sha256: str          # sha256 of json [prompt, continuation]
    intact_prompt_sha256: str
    cut_no_marker: bool = False


def _make(kind: str, record, prompt: str, continuation: str, no_marker: bool = False) -> SteerInput:
    return SteerInput(kind, record.question_id, record.fold, prompt, continuation,
                      sha256(json.dumps([prompt, continuation])), sha256(record.prompt), no_marker)


def arm_input(record, arm: str) -> SteerInput:
    """§35 arm A or B for one PromptRecord (raises ValueError if the cached output has no marker)."""
    item = sel.build_arm_input(record, arm)
    if item is None:
        raise ValueError(f"{record.question_id}: no Answer: marker, not a frozen-note row")
    return _make(arm, record, item.prompt, item.continuation)


def cut_input(record, ncut_generated: str) -> SteerInput:
    """The Steer-cut / A-cut prompt: §34c's cut view + the N-cut arm's own note + 'Answer:'."""
    prompt = sel.build_arm_input(record, "Ncut").prompt
    note, _ = sel.split_note(ncut_generated)
    return _make("cut", record, prompt, (note or "") + sel.ANSWER_MARKER, note is None)


# --- directions and vectors -----------------------------------------------------------------------------------

def estimate_direction(resid_a: dict[str, np.ndarray], resid_b: dict[str, np.ndarray], pairs: list[str],
                       layer: int) -> np.ndarray:
    """mean over pairs of resid_B[layer] - resid_A[layer]; resid arrays are (n_layers, hidden) at the ':' position."""
    if not pairs:
        raise ValueError("no pairs to estimate a direction from")
    diffs = [resid_b[q][layer].astype(np.float64) - resid_a[q][layer].astype(np.float64) for q in pairs]
    return np.mean(diffs, axis=0).astype(np.float32)


def mean_resid_norm(resid: dict[str, np.ndarray], qids: list[str], layer: int) -> float:
    return float(np.mean([np.linalg.norm(resid[q][layer].astype(np.float64)) for q in qids]))


def direction_stats(v: np.ndarray, resid_a: dict[str, np.ndarray], pairs: list[str], layer: int) -> dict:
    vn = float(np.linalg.norm(v.astype(np.float64)))
    rn = mean_resid_norm(resid_a, pairs, layer)
    return {"layer": layer, "n_pairs": len(pairs), "v_norm": vn, "mean_resid_A_norm": rn,
            "v_over_resid": vn / rn if rn else None, "vector_sha256": vector_sha(v)}


def scaled_vector(v: np.ndarray, alpha: float, unit: str = DEFAULT_UNIT, mean_resid: float | None = None) -> np.ndarray:
    """'vnorm': alpha * v (alpha in units of ||v||); 'residnorm': alpha * mean||resid_A|| * v / ||v||."""
    if unit == "vnorm":
        return (alpha * v.astype(np.float64)).astype(np.float32)
    if unit == "residnorm":
        if mean_resid is None:
            raise ValueError("unit 'residnorm' needs the mean residual norm")
        n = np.linalg.norm(v.astype(np.float64))
        return (alpha * mean_resid * v.astype(np.float64) / n).astype(np.float32)
    raise ValueError(f"unknown unit {unit!r} (known: {UNITS})")


def random_vector(like: np.ndarray, layer: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """A seeded Gaussian direction (one per layer, the same for every alpha and row) with the norm of `like`."""
    rng = np.random.default_rng([seed, layer])
    g = rng.standard_normal(like.shape[-1])
    return (g / np.linalg.norm(g) * np.linalg.norm(like.astype(np.float64))).astype(np.float32)


def vector_sha(v: np.ndarray | None) -> str | None:
    return None if v is None else hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()


# --- settings -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    mode: str                  # 'none', 'steer', 'random', 'reverse', 'zero', 'overwrite_B', 'overwrite_self'
    layer: int | None = None
    alpha: float | None = None
    unit: str | None = None
    steps: tuple | None = None  # §39: the add acts only at these steps (0 = the ':' token); None = every step (§38)

    @property
    def key(self) -> str:
        if self.mode == "none":
            return "none"
        if self.alpha is None:
            base = f"{self.mode}:L{self.layer}"
        else:
            base = f"{self.mode}:L{self.layer}:a{self.alpha:g}:{self.unit}"
        return base if self.steps is None else f"{base}:{steps_label(self.steps)}"

    def to_dict(self) -> dict:
        out = {"mode": self.mode, "layer": self.layer, "alpha": self.alpha, "unit": self.unit, "key": self.key}
        if self.steps is not None:      # §38 rows (every step) keep their exact dict
            out["steps"] = list(self.steps)
        return out


def steps_label(steps) -> str:
    """(0,) -> 's0'; (0, 1, 2) -> 's0-2'; a non-contiguous set -> 's0,3'. Every step (None) has no label."""
    s = sorted(set(int(x) for x in steps))
    if not s:
        raise ValueError("an empty step set acts nowhere; use Setting('none')")
    if len(s) == 1:
        return f"s{s[0]}"
    if s == list(range(s[0], s[-1] + 1)):
        return f"s{s[0]}-{s[-1]}"
    return "s" + ",".join(str(x) for x in s)


def parse_steps(label: str) -> tuple:
    """Inverse of `steps_label`."""
    if not label.startswith("s"):
        raise ValueError(f"not a step label: {label!r}")
    body = label[1:]
    if "," in body:
        return tuple(sorted(int(x) for x in body.split(",")))
    if "-" in body:
        a, b = body.split("-")
        return tuple(range(int(a), int(b) + 1))
    return (int(body),)


def steps_upto(k: int) -> tuple:
    """Steer-k of §39: steps 0..k (k = 0 is Steer-0, the ':' position only)."""
    return tuple(range(k + 1))


def sweep_settings(layers, alphas, unit: str = DEFAULT_UNIT) -> list[Setting]:
    return [Setting("steer", L, float(a), unit) for L in layers for a in alphas]


# --- hooks ----------------------------------------------------------------------------------------------------

@dataclass
class HookSpec:
    layer: int
    mode: str                         # 'add', 'overwrite' or 'capture'
    vector: np.ndarray | None = None
    steps: frozenset | None = None    # None: every step (the ':' token is step 0); else only these steps


@dataclass
class HookState:
    step: int = 0
    captured: dict = field(default_factory=dict)    # layer -> np.ndarray (step 0 only)
    calls: int = 0


def add_spec(layer: int, vector, steps=None) -> HookSpec:
    """Add at the current position at the ':' token and every generated token (steps None), or only at `steps`."""
    return HookSpec(layer, "add", np.asarray(vector, dtype=np.float32), None if steps is None else frozenset(steps))


def overwrite_spec(layer: int, vector) -> HookSpec:
    """Overwrite the ':' position's residual (step 0 only); generated tokens are left alone."""
    return HookSpec(layer, "overwrite", np.asarray(vector, dtype=np.float32), frozenset({0}))


def capture_spec(layer: int) -> HookSpec:
    return HookSpec(layer, "capture", None, frozenset({0}))


@contextmanager
def hooked(layers, specs: list[HookSpec], state: HookState):
    """Register forward hooks on `layers[spec.layer]`; they act on the last position of the layer's output (the
    current token: every forward under them has sequence length 1). Always removed, also on an exception."""
    import torch

    by_layer: dict[int, list[HookSpec]] = {}
    for spec in specs:
        if spec.mode not in ("add", "overwrite", "capture"):
            raise ValueError(f"unknown hook mode {spec.mode!r}")
        by_layer.setdefault(spec.layer, []).append(spec)
    handles = []

    def make(layer_specs):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            active = [s for s in layer_specs if s.steps is None or state.step in s.steps]
            if not active:
                return None
            state.calls += 1
            new = h
            for s in active:
                if s.mode == "capture":
                    continue
                if new is h:
                    new = h.clone()
                vec = torch.as_tensor(s.vector, dtype=h.dtype, device=h.device)
                if s.mode == "add":
                    new[:, -1, :] = new[:, -1, :] + vec
                else:
                    new[:, -1, :] = vec
            for s in active:
                if s.mode == "capture":
                    state.captured[s.layer] = new[0, -1].detach().float().cpu().numpy().copy()
            if new is h:
                return None
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return fn

    try:
        for layer, layer_specs in by_layer.items():
            handles.append(layers[layer].register_forward_hook(make(layer_specs)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def specs_for(setting: Setting, vectors: dict, own_state: np.ndarray | None = None,
              b_state: np.ndarray | None = None, hidden: int = HIDDEN) -> list[HookSpec]:
    """Hook specs for a setting. `vectors[(layer, alpha, unit)]` is the steer vector; random / reverse derive from it."""
    if setting.mode == "none":
        return []
    vec = hook_vector(setting, vectors, own_state, b_state, hidden)
    if setting.mode in ("overwrite_B", "overwrite_self"):
        return [overwrite_spec(setting.layer, vec)]
    return [add_spec(setting.layer, vec, setting.steps)]


def hook_vector(setting: Setting, vectors: dict, own_state: np.ndarray | None = None, b_state: np.ndarray | None = None,
                hidden: int = HIDDEN) -> np.ndarray | None:
    """The vector a setting writes (what its row's vector_sha256 stamps); None for the unsteered setting. For the
    overwrite settings `own_state` / `b_state` are the (n_layers, hidden) residuals at the ':' position."""
    m = setting.mode
    if m == "none":
        return None
    if m == "zero":
        return np.zeros(hidden, np.float32)
    if m == "overwrite_B":
        return np.asarray(b_state[setting.layer], np.float32)
    if m == "overwrite_self":
        return np.asarray(own_state[setting.layer], np.float32)
    return setting_vector(setting, vectors)


def setting_vector(setting: Setting, vectors: dict) -> np.ndarray | None:
    if setting.mode not in ("steer", "random", "reverse"):
        return None
    v = vectors[(setting.layer, setting.alpha, setting.unit)]
    if setting.mode == "random":
        return random_vector(v, setting.layer)
    if setting.mode == "reverse":
        return -v
    return v


# --- classification -------------------------------------------------------------------------------------------

def phrase_of(generated: str) -> str:
    """The generated phrase: the first line, stripped (generation already stops at a newline)."""
    return generated.split("\n", 1)[0].strip()


def is_degenerate(phrase: str, token_ids: list[int] | None = None) -> bool:
    p = (phrase or "").strip()
    if not p or not re.search(r"[A-Za-z0-9]", p):
        return True
    run, prev = 0, None
    for t in token_ids or []:
        run = run + 1 if t == prev else 1
        prev = t
        if run >= DEGENERATE_REPEAT:
            return True
    return False


def classify(phrase: str, token_ids: list[int] | None = None) -> str:
    if sel.is_refusal_answer(phrase):
        return "refusal"
    if is_degenerate(phrase, token_ids):
        return "degenerate"
    return "answer"


def provisional_gold(phrase: str, gold: str, cls: str) -> bool:
    """Laptop gold, PROVISIONAL (never a claim): the §35 D1 string rule on the generated phrase, answers only."""
    return cls == "answer" and sel.d1(phrase, gold) is True


# --- reproduction ---------------------------------------------------------------------------------------------

def reproduces(row: dict, cached: dict, gold: str) -> dict:
    """A regenerated A or B row (R rows and pairs) against the §35 cached row: the refusal flag (§23 rule on
    continuation + generated, as the cache) must match (build review ruling 1). The D1 verdict match is reported as a
    column, not required."""
    flag = bool(row["refusal"]) == bool(cached["refusal"])
    verdict = True
    if flag and not cached["refusal"]:
        verdict = (sel.d1(row["final_answer"], gold) is True) == (sel.d1(cached["final_answer"], gold) is True)
    return {"flag_match": flag, "verdict_match": verdict, "ok": flag,
            "same_answer": row["final_answer"] == cached["final_answer"]}


def reproduces_c_judge(row: dict, cached: dict, gold: str, verdict: bool | None) -> dict:
    """A C row on the pod (A judged correct in §35): A must answer on the new engine and the NEW judge (this run's
    engine) must call it correct. The pod's analogue of `reproduces_damage`, with the judge in place of D1 (the
    Executor's choice at the pod build, for the Advisor to rule). `verdict` None (not judged) -> not reproduced."""
    out = reproduces(row, cached, gold)
    out["judge_new"] = verdict
    out["ok"] = (not row["refusal"]) and verdict is True
    return out


def reproduces_damage(row: dict, cached: dict, gold: str) -> dict:
    """A damage row (A judged correct in §35): on the new engine A must answer and pass D1 (build review ruling 1)."""
    out = reproduces(row, cached, gold)
    out["d1_new"] = (not row["refusal"]) and sel.d1(row["final_answer"], gold) is True
    out["ok"] = out["d1_new"]
    return out


# --- the pod: settings, the judge's reply ---------------------------------------------------------------------

SECONDARY = (29, 2.0, "vnorm")      # amendment 2: L29 alpha 2 (vnorm), disclosed secondary setting
SECONDARY_LABEL = "secondary (chosen after dev D1 gold; disclosed)"
SECONDARY_ONLY = "shown at the secondary setting only; two settings were tested"
JUDGE_AGREEMENT_BAR = 0.90          # dev review addendum, condition 2
JUDGE_CAVEAT = "judge engine differs from §35"


def setting_from_key(key: str) -> Setting:
    """'steer:L25:a4:vnorm' -> Setting('steer', 25, 4.0, 'vnorm'); a §39 key adds the steps ('...:vnorm:s0-2')."""
    parts = key.split(":")
    mode, layer, alpha, unit = parts[:4]
    steps = parse_steps(parts[4]) if len(parts) == 5 else None
    if len(parts) not in (4, 5):
        raise ValueError(f"not a steer setting key: {key!r}")
    return Setting(mode, int(layer[1:]), float(alpha[1:]), unit, steps)


def map_layer(layer: int, n_layers: int) -> int:
    """A layer of the 36-layer 3B on an engine with `n_layers` (identity on the 3B; the stub has fewer layers)."""
    return layer if n_layers == N_LAYERS else int(round(layer * n_layers / N_LAYERS))


def parse_verdict(reply: str | None) -> bool | None:
    """The judge's reply as a verdict: True for 'yes...', False for 'no...', None when it is neither (the §36 run 1
    failure mode: the model answered instead of grading). `answer_selection.verdict_of` is the §35 rule (anything not
    'yes' is False); the smoke and the report use this one to count replies that are not verdicts."""
    r = (reply or "").strip().lower()
    if r.startswith("yes"):
        return True
    if r.startswith("no"):
        return False
    return None


# --- sanity and the design gate -------------------------------------------------------------------------------

def sanity_summary(final_first_match: list[bool], zero_exact: list[bool], self_exact: list[bool],
                   shallow_first_match: list[bool]) -> dict:
    def rate(x):
        return sum(x) / len(x) if x else None
    out = {"final_overwrite_rate": rate(final_first_match), "n_final": len(final_first_match),
           "zero_exact_all": bool(zero_exact) and all(zero_exact), "n_zero": len(zero_exact),
           "self_exact_all": bool(self_exact) and all(self_exact), "n_self": len(self_exact),
           "shallow_overwrite_rate": rate(shallow_first_match), "n_shallow": len(shallow_first_match),
           "shallow_note": "descriptive only (amendment 1.6)"}
    out["final_pass"] = out["final_overwrite_rate"] is not None and out["final_overwrite_rate"] >= SANITY_FINAL_GATE
    out["pass"] = out["final_pass"] and out["zero_exact_all"] and out["self_exact_all"]
    return out


def setting_order(steer: dict[str, dict]) -> list[str]:
    """Settings by highest ans(Steer); ties to the shallower layer, then the smaller alpha."""
    return sorted(steer, key=lambda k: (-steer[k]["ans"], steer[k]["layer"], steer[k]["alpha"]))


def design_gate(steer: dict[str, dict], random_ans: dict[str, float | None], sanity_pass: bool,
                repro_rate: float | None) -> dict:
    """`steer[key]` = {layer, alpha, ans, degenerate, n} on the dev R rows; `random_ans[key]` = ans(Random) at that
    setting (None or missing: not run yet). Walks the settings in `setting_order`: a setting is eligible if its
    degenerate rate is <= 0.2 and ans(Random) <= 0.2 * ans(Steer); the first eligible one is chosen. A candidate with
    ans(Steer) < 0.5 cannot pass the gate, so no Random run is asked for it. status: 'need_random' (run Random at
    `need`), 'chosen' or 'none'."""
    log, need, chosen = [], None, None
    for key in setting_order(steer):
        s = steer[key]
        entry = {"setting": key, "ans": s["ans"], "degenerate": s["degenerate"], "n": s.get("n")}
        if s["degenerate"] > DEGENERATE_MAX:
            entry["eligible"], entry["why"] = False, f"degenerate {s['degenerate']:.3f} > {DEGENERATE_MAX}"
            log.append(entry)
            continue
        if s["ans"] < ANS_GATE:
            entry["eligible"], entry["why"] = False, f"ans(Steer) {s['ans']:.3f} < {ANS_GATE}: cannot pass the gate"
            log.append(entry)
            continue
        r = random_ans.get(key)
        if r is None:
            need = key
            entry["why"] = "Random not run yet"
            log.append(entry)
            break
        bound = RANDOM_SHARE * s["ans"]
        entry["random_ans"], entry["random_bound"] = r, bound
        if r <= bound:
            entry["eligible"], entry["why"] = True, f"ans(Random) {r:.3f} <= {RANDOM_SHARE} x {s['ans']:.3f} = {bound:.3f}"
            log.append(entry)
            chosen = key
            break
        entry["eligible"], entry["why"] = False, f"ans(Random) {r:.3f} > {RANDOM_SHARE} x {s['ans']:.3f} = {bound:.3f}"
        log.append(entry)
    repro_ok = repro_rate is not None and repro_rate >= REPRO_GATE
    out = {"order": log, "need": need, "chosen": chosen, "sanity_pass": sanity_pass, "repro_rate": repro_rate,
           "repro_pass": repro_ok,
           "status": "need_random" if need else "chosen" if chosen else "none"}
    if chosen:
        out["chosen_setting"] = {"layer": steer[chosen]["layer"], "alpha": steer[chosen]["alpha"],
                                 "ans": steer[chosen]["ans"], "degenerate": steer[chosen]["degenerate"],
                                 "random_ans": random_ans[chosen]}
    out["pass"] = bool(sanity_pass and repro_ok and chosen and steer[chosen]["ans"] >= ANS_GATE)
    if need:
        out["pass"] = None
    return out


# --- engines --------------------------------------------------------------------------------------------------

@dataclass
class Prepared:
    ids: list[int]
    last_id: int
    cache: object             # KV cache over ids[:-1] (everything before the ':' token)
    n_prefix: int
    prefill_s: float


@dataclass
class Generation:
    generated: str
    gen_ids: list[int]
    first_token: int | None
    captured: dict            # layer -> np.ndarray at the ':' position
    first_step_s: float
    decode_s: float
    copy_s: float
    forced: int | None = None          # §39b: the token fed as the first generated token (None: not forced)
    natural_first: int | None = None   # §39b: the argmax at ':' on the unpushed cache, recorded and not used

    @property
    def n_new(self) -> int:
        return len(self.gen_ids)


class HookedEngine:
    """Greedy generation from a cached prefix, one token at a time, with forward hooks on `self.layers`.

    Subclasses provide `name`, `model`, `layers`, `n_layers`, `hidden`, `stop_ids`, `chat_text`, `encode`, `decode`,
    `_prefill(ids) -> cache` and `_step(token, cache) -> logits (1-D tensor)`; `_step` updates the cache in place."""

    def input_ids(self, item: SteerInput) -> list[int]:
        return self.encode(self.chat_text(item.prompt)) + (self.encode(item.continuation) if item.continuation else [])

    def prepare(self, item: SteerInput) -> Prepared:
        ids = self.input_ids(item)
        if len(ids) < 2:
            raise ValueError("prompt too short to split off the last token")
        t0 = time.perf_counter()
        cache = self._prefill(ids[:-1])
        return Prepared(ids, ids[-1], cache, len(ids) - 1, time.perf_counter() - t0)

    def last_token_text(self, prepared: Prepared) -> str:
        return self.decode([prepared.last_id])

    def run(self, prepared: Prepared, specs: list[HookSpec], max_new_tokens: int,
            capture: tuple[int, ...] = ()) -> Generation:
        import torch

        t0 = time.perf_counter()
        cache = copy.deepcopy(prepared.cache)
        t1 = time.perf_counter()
        state = HookState()
        all_specs = list(specs) + [capture_spec(L) for L in capture]
        gen: list[int] = []
        with torch.no_grad(), hooked(self.layers, all_specs, state):
            logits = self._step(prepared.last_id, cache)
            t2 = time.perf_counter()
            first = None
            while True:
                nxt = int(logits.argmax())
                if first is None:
                    first = nxt
                if nxt in self.stop_ids:
                    break
                gen.append(nxt)
                if "\n" in self.decode(gen) or len(gen) >= max_new_tokens:
                    break
                state.step += 1
                logits = self._step(nxt, cache)
        t3 = time.perf_counter()
        return Generation(self.decode(gen), gen, first, dict(state.captured), t2 - t1, t3 - t2, t1 - t0)

    def assert_no_hooks(self) -> None:
        """§39b: refuse to run a forced generation while any forward hook is registered on a decoder layer (or
        globally), so 'no hook at any step' is checked, not assumed."""
        from torch.nn.modules import module as tmod

        stray = [i for i, layer in enumerate(self.layers) if layer._forward_hooks or layer._forward_pre_hooks]
        if stray or tmod._global_forward_hooks or tmod._global_forward_pre_hooks:
            raise RuntimeError(f"forward hooks registered (layers {stray}); a forced generation runs with none")

    def run_forced(self, prepared: Prepared, forced: int, max_new_tokens: int) -> Generation:
        """§39b, the first-token-only clean-cache path. From the unpushed prefix cache (every token before ':') the
        ':' token is run unpushed (its argmax is recorded as `natural_first` and discarded); `forced` is then the
        first generated token (gen_ids[0]) and greedy decoding continues from it with no hook at any step. Stops as
        `run`: EOS (never recorded), a newline in the decoded text (the newline token is kept, as in `run`), or
        `max_new_tokens` generated tokens counting the forced one. A forced token whose text contains a newline
        therefore ends the generation at once (gen_ids = [forced]). A stop id cannot be forced (ValueError)."""
        import torch

        forced = int(forced)
        if forced in self.stop_ids:
            raise ValueError(f"cannot force a stop id ({forced}); it is never a generated token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1 (the forced token counts)")
        self.assert_no_hooks()
        t0 = time.perf_counter()
        cache = copy.deepcopy(prepared.cache)
        t1 = time.perf_counter()
        with torch.no_grad():
            natural = int(self._step(prepared.last_id, cache).argmax())
            t2 = time.perf_counter()
            gen = [forced]
            while "\n" not in self.decode(gen) and len(gen) < max_new_tokens:
                nxt = int(self._step(gen[-1], cache).argmax())
                if nxt in self.stop_ids:
                    break
                gen.append(nxt)
        t3 = time.perf_counter()
        return Generation(self.decode(gen), gen, forced, {}, t2 - t1, t3 - t2, t1 - t0, forced, natural)


class HFEngine(HookedEngine):
    """transformers; fp32 on CPU (the laptop), bf16 on CUDA (the pod)."""

    def __init__(self, model: str, threads: int = 8) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch, self.model = torch, model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.name = f"hf-transformers-{transformers.__version__}-{self.device}-{str(dtype).split('.')[-1]}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.net = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.layers = self.net.model.layers
        self.n_layers, self.hidden = len(self.layers), self.net.config.hidden_size
        self.stop_ids = {self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id}

    def chat_text(self, prompt: str) -> str:
        from memctl.analysis.jacobian import wrap_chat
        return wrap_chat(self.tokenizer, prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _prefill(self, ids: list[int]):
        torch = self.torch
        with torch.no_grad():
            out = self.net(input_ids=torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1)
        return out.past_key_values

    def _step(self, token: int, cache):
        torch = self.torch
        out = self.net(input_ids=torch.tensor([[token]], device=self.device), past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
        return out.logits[0, -1]


class _StubCache:
    def __init__(self, total, count: int, length: int, forced: list[int] | None = None) -> None:
        self.total, self.count, self.length = total, count, length
        self.forced, self.pos = forced, 0


class StubEngine(HookedEngine):
    """A tiny deterministic torch model with the same hook surface (`layers[L]` returning the residual) and a byte
    tokenizer, for tests and the CPU stub run. No download, no Qwen weights. The 'cache' is the running sum of the
    prefix's embeddings plus its length, so `prepare` really excludes the last token (`cache.length`)."""

    EOS = 256

    def __init__(self, model: str = "stub", n_layers: int = 20, hidden: int = 16, seed: int = 0,
                 tuple_output: bool = False) -> None:
        import torch
        from torch import nn

        self.torch, self.model, self.name = torch, model, "stub"
        self.n_layers, self.hidden = n_layers, hidden
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(257, hidden, generator=g)
        self.head = torch.randn(hidden, 257, generator=g)
        self.head_bias = torch.full((257,), -4.0)
        for ch in " abcdefghijklmnopqrstuvwxyz":
            self.head_bias[ord(ch)] = 0.0
        self.head_bias[self.EOS] = -1.0

        class Layer(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.w = torch.randn(hidden, hidden, generator=g) / hidden ** 0.5
                inner.u = torch.randn(hidden, hidden, generator=g) / hidden ** 0.5

            def forward(inner, h, ctx):
                out = h + torch.tanh(h @ inner.w + ctx @ inner.u)
                return (out,) if tuple_output else out

        self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
        self.stop_ids = {self.EOS}
        self.oracle: dict[str, str] = {}     # sha256(chat text + continuation) -> text the stub is biased to emit

    def set_oracle(self, texts: dict[str, str]) -> None:
        """Bias the stub (by `ORACLE_BIAS` on the logits, so hooks still act on the residual) toward a canned
        continuation per input: lets a CPU run reproduce the cached flags and exercise every later stage."""
        self.oracle = dict(texts)

    ORACLE_BIAS = 50.0

    def prepare(self, item: SteerInput) -> Prepared:
        prepared = super().prepare(item)
        text = self.oracle.get(sha256(self.chat_text(item.prompt) + item.continuation))
        if text is not None:
            prepared.cache.forced = self.encode(text)
        return prepared

    def chat_text(self, prompt: str) -> str:
        return sel.QWEN25_CHAT.format(prompt=prompt)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

    def _prefill(self, ids: list[int]) -> _StubCache:
        return _StubCache(self.emb[ids].sum(0), len(ids), len(ids))

    def _step(self, token: int, cache: _StubCache):
        e = self.emb[token]
        ctx = (cache.total / max(cache.count, 1))[None, None, :]
        cache.total, cache.count, cache.length = cache.total + e, cache.count + 1, cache.length + 1
        h = e[None, None, :]
        for layer in self.layers:
            out = layer(h, ctx)
            h = out[0] if isinstance(out, tuple) else out
        last = h[0, -1]
        logits = 2.0 * (last / (last.norm() + 1e-6)) @ self.head + self.head_bias
        if cache.forced is not None:
            target = cache.forced[cache.pos] if cache.pos < len(cache.forced) else self.EOS
            logits = logits.clone()
            logits[target] += self.ORACLE_BIAS
        cache.pos += 1
        return logits


def continuation_first_id(engine, continuation: str, text: str) -> int:
    """§39b: the first token id of `text` as the tokenizer emits it after `continuation` (the prompt's own ids end
    with ids(continuation)): ids(continuation + text) minus the ids(continuation) prefix. Raises if appending `text`
    re-tokenises the continuation (the prefix is not stable) or adds no token."""
    a = engine.encode(continuation)
    b = engine.encode(continuation + text)
    if b[:len(a)] != a or len(b) <= len(a):
        raise ValueError(f"appending {text!r} changes the tokens of the continuation; no first id")
    return b[len(a)]


@dataclass(frozen=True)
class ForcedArm:
    """§39b: a forced-first-token arm (no hook; the forced id is per row and stamped on the row)."""
    name: str                  # 'unknown_first', 'steer_first', 'steer_first_cut', 'gold_first'

    @property
    def key(self) -> str:
        return f"forced:{self.name}"

    def to_dict(self) -> dict:
        return {"mode": "forced", "layer": None, "alpha": None, "unit": None, "key": self.key, "arm": self.name}


def build_engine(backend: str, model: str, threads: int = 8):
    if backend == "stub":
        return StubEngine(model)
    if backend == "hf":
        return HFEngine(model, threads)
    raise ValueError(f"unknown backend {backend!r} (known: hf, stub)")


# --- rows -----------------------------------------------------------------------------------------------------

def make_row(stage: str, pop: str, item: SteerInput, setting: Setting, gen: Generation, engine, prepared: Prepared,
             vector: np.ndarray | None, max_new_tokens: int, gold: str, reader: str = "qwen3b") -> dict:
    """One generation row, sha-stamped (input, wording, engine, setting and vector)."""
    answer, marker = sel.agent_answer(item.continuation + gen.generated)
    phrase = phrase_of(gen.generated)
    cls = classify(phrase, gen.gen_ids)
    return {"stage": stage, "pop": pop, "kind": item.kind, "question_id": item.question_id, "fold": item.fold,
            "reader": reader, "engine": engine.name, "model": engine.model, "wording_sha256": wording_sha256(max_new_tokens),
            "input_sha256": item.input_sha256, "intact_prompt_sha256": item.intact_prompt_sha256,
            "cut_no_marker": item.cut_no_marker, "setting": setting.to_dict(), "vector_sha256": vector_sha(vector),
            "vector_norm": None if vector is None else float(np.linalg.norm(vector.astype(np.float64))),
            "n_input_tokens": len(prepared.ids), "n_prefix_tokens": prepared.n_prefix,
            "last_token": engine.last_token_text(prepared), "max_new_tokens": max_new_tokens,
            "generated": gen.generated, "gen_ids": gen.gen_ids, "first_token": gen.first_token, "phrase": phrase,
            "final_answer": answer, "marker": marker, "refusal": sel.is_refusal_answer(answer), "cls": cls,
            "ans": cls == "answer", "gold_d1_provisional": provisional_gold(phrase, gold, cls),
            "timing": {"prefill_s": prepared.prefill_s, "copy_s": gen.copy_s, "first_step_s": gen.first_step_s,
                       "decode_s": gen.decode_s, "n_new": gen.n_new}}


def make_forced_row(stage: str, pop: str, item: SteerInput, arm: ForcedArm, gen: Generation, engine, prepared: Prepared,
                    max_new_tokens: int, gold: str, source: dict, reader: str = "qwen3b") -> dict:
    """§39b: `make_row` for a forced generation (vector None: no hook), plus the forced id, its text, the unpushed
    ':' argmax and where the forced id came from."""
    row = make_row(stage, pop, item, arm, gen, engine, prepared, None, max_new_tokens, gold, reader)
    row.update({"forced_token": gen.forced, "forced_text": engine.decode([gen.forced]),
                "natural_first_token": gen.natural_first, "forced_source": source})
    return row


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["pop"], row["kind"], row["setting"]["key"], row["question_id"])


class StaleRowError(RuntimeError):
    """A row on disk came from a different input, wording, engine or vector than this run would use."""


def check_row(row: dict, item: SteerInput, engine_name: str, wording_sha: str, vector: np.ndarray | None) -> None:
    where = f"{row['pop']}/{row['kind']}/{row['setting']['key']}/{row['question_id']}"
    if row["input_sha256"] != item.input_sha256 or row["wording_sha256"] != wording_sha:
        raise StaleRowError(f"{where}: different input or wording on disk; move the old file aside")
    if row["engine"] != engine_name:
        raise StaleRowError(f"{where}: engine {row['engine']!r} on disk, this run {engine_name!r}")
    if row["vector_sha256"] != vector_sha(vector):
        raise StaleRowError(f"{where}: the vector on disk differs from this run's (direction changed?)")


# --- rates ----------------------------------------------------------------------------------------------------

def rates(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "ans": None, "degenerate": None, "refusal": None, "gold_d1": None}
    return {"n": n, "ans": sum(r["cls"] == "answer" for r in rows) / n,
            "degenerate": sum(r["cls"] == "degenerate" for r in rows) / n,
            "refusal": sum(r["cls"] == "refusal" for r in rows) / n,
            "gold_d1": sum(bool(r["gold_d1_provisional"]) for r in rows) / n}
