"""§43: one config per model family, beside the Qwen defaults (rev 1 ruling B3; runs/exp43_stage0/note.md §7).

Nothing here changes a §38-§42 path: `steering.HFEngine(model)` without a family, `jacobian_data.load_records` and the
lens scripts without `--family` keep their Qwen constants. A FamilyConfig is passed explicitly where a new family is
run (`HFEngine(model_id, family=cfg)`, `jacobian_data.load_family_records(cfg, fold)`, `jlens_read.py --family`).

Facts are from §43 stage 0 (f007fe2, `runs/exp43_stage0/tok.json`, `gen*.json`):
- Llama 3.2's template writes "Today Date: <date>" into a system block; vLLM rendered the §24 pod's date, taken as
  09 Oct 2026, so every Llama run pins `date_string` (the laptop and the pod use the same string).
- Gemma 3 has no system role: a system text, if one is ever set, goes into the user turn as "system\\n\\nuser" (what the
  template itself does). §23/§24 send no system message, so nothing is folded.
- Stop ids: the turn end and the end-of-text token per family (HFEngine's Qwen default `{<|im_end|>, eos}` misses
  Gemma's `<end_of_turn>` and Phi's `<|end|>`).
- Refusal ids: the first ids of " unknown" and " Unknown" after "Answer:" (Gemma and Phi refuse with ' Unknown' too);
  `check_refusal_ids` re-derives them from the tokenizer with `steering.continuation_first_id` at load.
- BOS: Llama's and Gemma's chat text carries its BOS as text, so `encode(add_special_tokens=False)` gives exactly one;
  Phi and Qwen have none. `lens_force_bos` is jlens's fit-time BOS for raw (non-chat) text: on for Llama and Gemma,
  off for Phi (its tokenizer's "BOS" is <|endoftext|>, which its chat prompts never start with; the fit text then
  starts the way the read prompts do), on (a no-op: no BOS token) for Qwen as §34 fitted.
Pure: no torch, no transformers at import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REFUSAL_PROBE = "Some note.\nAnswer:"          # the continuation the refusal ids are derived after (stage 0 tok.py)
REFUSAL_TEXTS = (" unknown", " Unknown")
LAPTOP_ENV = {"MALLOC_ARENA_MAX": "2", "HF_ENABLE_PARALLEL_LOADING": "false"}
LLAMA_DATE = "09 Oct 2026"


@dataclass(frozen=True)
class FamilyConfig:
    key: str                         # reader key of the §24 runs: runs/{exp_tag}_{key}_f{fold}
    model_id: str                    # HF id (the tokenizer always comes from here)
    weights: str | None              # local fp32 copy for the laptop (runs/exp43_stage0/to_fp32.py), else None
    partner_key: str                 # the §34 pair's large model
    partner_model_id: str
    exp_tag: str                     # "exp24" (or "exp23" for Qwen)
    verdicts: str                    # judge labels of the §24 run
    verdict_arm: str                 # "fixed8"
    layer_path: str                  # decoder layers, from the top-level model
    n_layers: int
    hidden: int
    template_kwargs: dict = field(default_factory=dict)
    system_text: str | None = None   # None: no system message (§23/§24)
    system_role: bool = True         # False (Gemma): a system text is folded into the user turn
    stop_tokens: tuple[str, ...] = ()
    refusal_ids: tuple[int, ...] = ()        # first ids of REFUSAL_TEXTS after "Answer:", in that order
    bos_in_chat: bool = False        # the chat text starts with the BOS token as text
    bos_token: str | None = None     # the tokenizer's BOS piece when bos_in_chat
    lens_force_bos: bool = True      # jlens.from_hf(force_bos=...) for raw-text fits
    laptop_env: dict = field(default_factory=dict)
    sweep_depths: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)

    @property
    def sweep_layers(self) -> tuple[int, ...]:
        """round(depth * n_layers): Llama 3.2 3B (28) -> 17, 20, 22, 25."""
        return tuple(int(round(d * self.n_layers)) for d in self.sweep_depths)

    @property
    def run_root(self) -> str:
        return f"{self.exp_tag}_{self.key}"

    def run_dir(self, fold: int, runs_dir: str = "runs") -> str:
        return f"{runs_dir}/{self.exp_tag}_{self.key}_f{fold}"

    @property
    def cache_dir(self) -> str:
        return f"cache/generations_{self.exp_tag}_{self.key}"

    def stamp(self) -> dict:
        """What a row's wording hash covers for a family (the template and its pinned arguments)."""
        return {"key": self.key, "model_id": self.model_id, "template_kwargs": dict(sorted(self.template_kwargs.items())),
                "system_text": self.system_text, "system_role": self.system_role, "stop_tokens": list(self.stop_tokens),
                "refusal_ids": list(self.refusal_ids)}


LLAMA3B = FamilyConfig(
    key="llama3b", model_id="meta-llama/Llama-3.2-3B-Instruct", weights=None,
    partner_key="llama8b", partner_model_id="meta-llama/Llama-3.1-8B-Instruct",
    exp_tag="exp24", verdicts="runs/exp24_verdicts.json", verdict_arm="fixed8",
    layer_path="model.layers", n_layers=28, hidden=3072,
    template_kwargs={"date_string": LLAMA_DATE},
    stop_tokens=("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"), refusal_ids=(9987, 22435),
    bos_in_chat=True, bos_token="<|begin_of_text|>", lens_force_bos=True, laptop_env=LAPTOP_ENV)

GEMMA4B = FamilyConfig(
    key="gemma4b", model_id="google/gemma-3-4b-it", weights="cache/exp43_fp32/gemma-3-4b-it",
    partner_key="gemma12b", partner_model_id="google/gemma-3-12b-it",
    exp_tag="exp24", verdicts="runs/exp24_verdicts.json", verdict_arm="fixed8",
    layer_path="model.language_model.layers", n_layers=34, hidden=2560,
    system_role=False,
    stop_tokens=("<end_of_turn>", "<eos>"), refusal_ids=(11908, 60355),
    bos_in_chat=True, bos_token="<bos>", lens_force_bos=True, laptop_env=LAPTOP_ENV)

PHI4MINI = FamilyConfig(
    key="phi4mini", model_id="microsoft/Phi-4-mini-instruct", weights="cache/exp43_fp32/Phi-4-mini-instruct",
    partner_key="phi4", partner_model_id="microsoft/phi-4",
    exp_tag="exp24", verdicts="runs/exp24_verdicts.json", verdict_arm="fixed8",
    layer_path="model.layers", n_layers=32, hidden=3072,
    stop_tokens=("<|end|>", "<|endoftext|>"), refusal_ids=(17927, 41855),
    bos_in_chat=False, bos_token=None, lens_force_bos=False, laptop_env=LAPTOP_ENV)

# The Qwen reference, for comparison and tests only: the §38-§42 code paths never read it (their constants stay put).
QWEN3B = FamilyConfig(
    key="qwen3b", model_id="Qwen/Qwen2.5-3B-Instruct", weights=None,
    partner_key="qwen7b", partner_model_id="Qwen/Qwen2.5-7B-Instruct",
    exp_tag="exp23", verdicts="runs/exp23_verdicts.json", verdict_arm="fixed8",
    layer_path="model.layers", n_layers=36, hidden=2048,
    stop_tokens=("<|im_end|>",), refusal_ids=(9788, 21693),
    bos_in_chat=False, bos_token=None, lens_force_bos=True, laptop_env={})

FAMILIES = {c.key: c for c in (LLAMA3B, GEMMA4B, PHI4MINI, QWEN3B)}


def get(key: str) -> FamilyConfig:
    if key not in FAMILIES:
        raise KeyError(f"unknown family {key!r} (known: {sorted(FAMILIES)})")
    return FAMILIES[key]


# --- tokenizer-side helpers (a tokenizer object is passed in; nothing is loaded here) ----------------------------

def messages(cfg: FamilyConfig, prompt: str) -> list[dict]:
    if cfg.system_text is None:
        return [{"role": "user", "content": prompt}]
    if not cfg.system_role:                      # Gemma: the system text goes into the user turn
        return [{"role": "user", "content": cfg.system_text + "\n\n" + prompt}]
    return [{"role": "system", "content": cfg.system_text}, {"role": "user", "content": prompt}]


def chat_text(tokenizer, cfg: FamilyConfig, prompt: str) -> str:
    """The chat text for one user message with the generation prompt, the family's template arguments pinned."""
    return tokenizer.apply_chat_template(messages(cfg, prompt), tokenize=False, add_generation_prompt=True,
                                         **cfg.template_kwargs)


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def stop_ids(tokenizer, cfg: FamilyConfig, generation_eos=None) -> set[int]:
    """The config's stop tokens plus the tokenizer's eos; every stop token must exist, and the generation config's
    eos ids (if given) must be among them."""
    out = set()
    for piece in cfg.stop_tokens:
        i = tokenizer.convert_tokens_to_ids(piece)      # (Phi's unk id IS <|endoftext|>: compare the piece back)
        if not isinstance(i, int) or tokenizer.convert_ids_to_tokens(i) != piece:
            raise ValueError(f"{cfg.key}: stop token {piece!r} is not in the vocabulary")
        out.add(i)
    if tokenizer.eos_token_id is not None:
        out.add(tokenizer.eos_token_id)
    if generation_eos is not None:
        gen = set(generation_eos if isinstance(generation_eos, (list, tuple, set)) else [generation_eos])
        if not gen <= out:
            raise ValueError(f"{cfg.key}: generation_config eos {sorted(gen)} not all in the stop ids {sorted(out)}")
    return out


def check_bos(tokenizer, cfg: FamilyConfig, ids: list[int]) -> None:
    """Exactly one BOS at the start for a family whose chat text carries it, none otherwise."""
    bos = tokenizer.bos_token_id
    n = ids.count(bos) if bos is not None else 0
    if cfg.bos_in_chat:
        if n != 1 or ids[0] != bos:
            raise ValueError(f"{cfg.key}: {n} BOS in the encoded chat (first id {ids[0]}); want exactly one, first")
    elif n:
        raise ValueError(f"{cfg.key}: the encoded chat holds {n} BOS-token id(s) ({bos}); want none")


def check_refusal_ids(encode_fn, cfg: FamilyConfig) -> tuple[int, ...]:
    """Re-derive the refusal ids with steering.continuation_first_id after REFUSAL_PROBE and assert they equal the
    config (so a tokenizer revision cannot change them silently). `encode_fn(text) -> ids` without special tokens."""
    from memctl.analysis.steering import continuation_first_id

    class _E:
        encode = staticmethod(encode_fn)
    got = tuple(continuation_first_id(_E, REFUSAL_PROBE, t) for t in REFUSAL_TEXTS)
    if got != cfg.refusal_ids:
        raise ValueError(f"{cfg.key}: refusal ids {got} from the tokenizer, config says {cfg.refusal_ids}")
    return got
