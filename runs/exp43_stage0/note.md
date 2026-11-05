# §43 stage 0: feasibility note (laptop, CPU, $0)

2026-10-26, Executor. Spec: `research/proposals/2026-10-26-exp43-cross-family.md` rev 0 + amendment 1 (21e4e35, D-40).
Feasibility only: no layer or α was chosen, nothing on folds 2–4 was read except label-presence counts, nothing on
LoCoMo. No pod, no RunPod call, no spend. Numbers: `runs/exp43_stage0/stage0.json`; scripts beside it.
Laptop wall time: about 55 min (11:26–12:21 PDT), including about 5 min of downloads and two fp32 conversions.

## 1. The short version

- **All six models are reachable on HF with the `.env` token** (Llama 3.2 3B, Gemma 3 4B and Phi-4-mini, plus the large
  partners Llama 3.1 8B, Gemma 3 12B and Phi-4). The Llama 403 from §24 is gone. Llama and Gemma are `gated: manual`
  and the token has been granted; Phi is not gated.
- **All three small models load in fp32 on the laptop under the 22 GB guard and reproduce the §24 vLLM-bf16 cache**:
  A arm 5/5, 5/5, 4/5 (Llama, Gemma, Phi), N arm 5/5 first token and marker for all three. D-22 bar (≥ 4/5): **PASS ×3**
  (Phi only just: 4/5).
- **Part A has little to open in every family.** Among marked rows with all evidence in view (the §35/§38 population),
  refusals are 10.4% for Llama (13 of 125), 1.7% for Gemma (2 of 121) and 3.4% for Phi (3 of 89). Qwen 3B's figure on the
  same rows is 30% (30 of 100). Every family is under the ~15% line: **"little to open" flagged for all three.** On the
  100 Qwen dev rows (all present, same prompts, byte-identical), Llama refuses on 9, Gemma on 2 and Phi on 3. Of Qwen's
  30 refusals, Llama gets 16 right, Gemma 19 and Phi 11.
- Llama's high §24 "unknown share" (0.249 under fixed8) comes from rows where the evidence is **not** all in view (26% of
  all marked rows refuse, against 10% with all evidence in view). That is the right behaviour, not a refusal disposition.
- **Part B (lens) population is the same 330 questions for every family**: 126 on dev (61 + 65), 204 on confirmation
  (66 + 68 + 70), the same per-fold counts as Qwen's §34. All are present at both sizes with byte-identical prompts and a
  judge label on both. The fixed8 controller does not depend on the reader, so the memory view is identical in every
  family.
- **The jlens read pipeline does not run on any new family as committed.** `scripts/jlens_read.py` (and
  `jlens_eval.py`) check `lens_model.encode(chat)` against the readout's ids. jlens's `force_bos=True` plus
  `add_special_tokens` defaulting on add a second BOS for Llama and Gemma, and a spurious `<|endoftext|>` for Phi. So
  every read raises and the script would exit 8 on the pod. Qwen never hit this because it has no BOS token. With the
  guard encoding as the readout does (`add_special_tokens=False`), the unchanged `read_record` runs end to end on
  Llama 3B and Gemma 4B (5/5 rows each, plumbing only).
- **J-lens fit time is the binding cost in Part B.** Scaled from §34's fits, the small models take about 51–73 min each.
  The large ones take about 145 min (Llama 8B), about 200 min (Gemma 12B) and about 320 min (Phi-4). That puts both fits
  over a 3 h stop for every family, and Phi-4's large fit alone is over it. Reads plus downloads for both sizes come to
  about 8–13 min per family.

## 2. Per-family table (folds 0–1)

| | Llama-3.2-3B-Instruct | gemma-3-4b-it | Phi-4-mini-instruct | Qwen2.5-3B (ref) |
|---|---|---|---|---|
| HF access (small / large partner) | yes / yes (8B) | yes / yes (12B) | yes / yes (phi-4 14B) | — |
| class, layer path | LlamaForCausalLM, `model.layers` | Gemma3ForConditionalGeneration, `model.language_model.layers` | Phi3ForCausalLM, `model.layers` | Qwen2, `model.layers` |
| layers / hidden / params | 28 / 3072 / 3.21 B | 34 / 2560 / 4.30 B (text 3.88 B + vision 0.42 B) | 32 / 3072 / 3.84 B | 36 / 2048 / 3.09 B |
| fp32 load route | direct (`MALLOC_ARENA_MAX=2`, `HF_ENABLE_PARALLEL_LOADING=false`) | local fp32 copy (`to_fp32.py`), zero-copy mmap | local fp32 copy | direct |
| peak RSS / VmPeak (GB) | 18.6 / 20.6 | 15.9 / 19.6 | 15.6 / 17.5 | — |
| prefill at ~1,100 tokens (ms/token) | 12.9 (1,099 tok) | 15.8 (1,101) | 18.1 (1,093) | 13.1 (§38) |
| decode (s/token, 16 steps) | 0.77 | 0.93 | 0.90 | 0.69 (§38) |
| template fit (single user turn, as §24) | yes; injects a system block with "Today Date: <today>" | yes; no system role (a system message is folded into the user turn) | yes; no default system | default Qwen system |
| `' unknown'` after `Answer:` | 9987 `Ġunknown` | 11908 `▁unknown` | 17927 `Ġunknown` | 9788 |
| `' Unknown'` / `'unknown'` / `'Unknown'` | 22435 / 16476 / 14109 | 60355 / 20774 / 41387 | 41855 / 33936 / 24560 | 21693 / 16088 / 13790 |
| cached refusal-start ids (in-view pop.; all rows f0–1) | {9987: 13}; {9987: 51} | {11908: 2}; {11908: 17, 60355: 1} | {17927: 3}; {17927: 15, 41855: 3} | {9788} |
| `Answer:` ends in `:` as last prompt token | 125/125 | 121/121 | 89/89 | yes |
| n records / with verdict | 200 / 188 | 200 / 188 | 200 / 188 | — |
| P(all evidence in view) | 0.665 | 0.665 | 0.665 | — |
| **marker rate**: all rows / all-in-view / §35 pop. | 0.980 / 0.992 / 0.992 (125/126) | 0.965 / 0.962 / 0.960 (121/126) | 0.715 / 0.714 / 0.706 (89/126) | 1.00 (100/100 dev) |
| **refusal among marked, §23/§35 rule** (cached correct False ∧ unknown True): all / in-view / pop. | 0.219 / 0.098 / **0.104 (13/125)** | 0.078 / 0.016 / **0.017 (2/121)** | 0.098 / 0.032 / **0.034 (3/89)** | 0.300 (30/100) |
| **refusal among marked, `is_refusal_answer`**: all / in-view / pop. | 0.260 / 0.129 / 0.104 | 0.093 / 0.016 / 0.017 | 0.126 / 0.063 / 0.034 | 0.300 |
| **judged correct** (§24 judge): all / in-view / pop. | 0.436 / 0.563 / 0.563 (71/126) | 0.537 / 0.659 / 0.659 (83/126) | 0.383 / 0.468 / 0.468 (59/126) | 0.390 (39/100) |
| **D1 provisional** (gold in note, marked): all / in-view / pop. | 0.281 / 0.326 / 0.344 | 0.326 / 0.383 / 0.405 | 0.427 / 0.505 / 0.539 | 0.370 |
| D1 among the rule's refusals (pop.) | 2/13 | 0/2 | 0/3 | 7/30 |
| Qwen dev rows (100) present / same prompt bytes / marked | 100 / 100 / 100 | 100 / 100 / 99 | 100 / 100 / 70 | — |
| on those 100: refusal (marked) / judged correct / D1 | 9/100 / 56/100 / 35/100 | 2/99 / 65/100 / 39/99 | 3/70 / 46/100 / 38/70 | 30 / 39 / 37 |
| Qwen's 30 refusals: family refuses too / family correct | 5 / 16 | 1 / 19 | 3 / 11 | — |
| **A-arm regen, 5 dev rows** (first id / refusal flag / phrase) | 5/5 / 5/5 / 5/5 | 5/5 / 5/5 / 5/5 | 4/5 / 4/5 / 4/5 | — |
| **N-arm regen** (first id / marker / refusal / byte-exact) | 5/5 / 5/5 / 5/5 / 2/5 | 5/5 / 5/5 / 5/5 / 4/5 | 5/5 / 5/5 / 4/5 / 1/5 | — |
| D-22 (≥ 4/5) A / N | PASS / PASS | PASS / PASS | PASS (4/5) / PASS | — |
| **little to open (< 15% refusals, in-view marked)** | **yes (10.4%)** | **yes (1.7%)** | **yes (3.4%)** | no |

Definitions. "All rows" = every fixed8 record on folds 0–1 (200). "In-view" = `all_in_view` True (133, unanswerable
included). "Pop." = the §35 rule for one reader: answerable, all evidence in view, at least one evidence line, judge
label present (126). The Qwen column is the §38 dev population: `scripts/exp35_report.load` `frozen`, 100 marked rows
on folds 0–1. The two refusal definitions agree on the population. They differ off it only on unanswerable (`_abs`)
rows, which the cached-verdict rule scores as correct. 12 of 200 records per family have no §24 verdict.

Regeneration rows: Qwen dev rows marked in the family, nearest to 1,100 A-arm tokens, the family's own in-view
refusals first (Llama 3 refusals + 2 answers; Gemma 2 + 3; Phi 3 + 2). Greedy decoding with fp32 HF transformers 5.17.0
on CPU (8 threads), against §24's vLLM bf16 cache. The A arm is §35's chat text + cached note + `Answer:`, ids by
concatenation, stopping at a newline or EOS (16 tokens max). The N arm is the full §23 prompt, 256 tokens. The one
miss is Phi `fca70973`, a preference/advice question: the cache says `unknown` and fp32 writes advice ("Check upcoming
events at Disneyland…"). Byte-exact N outputs are descriptive (2/5, 4/5, 1/5). The answers and flags match even where
the text drifts.

Load time is not comparable across columns. Llama's 3 s was a warm page cache. Gemma and Phi load as a lazy mmap of
the fp32 copy (0.3–0.5 s), so they pay their reads in the first forward.

## 3. Things that differ from Qwen (surprises first)

1. **Refusals with the evidence in view are rare outside Qwen.** That is the headline for Part A (table above). Under
   the rev-0 rule, Gemma (2 rows) and Phi (3 rows) have nothing to estimate a direction from. Llama has 13 in-view
   refusals on dev, 9 of them on Qwen's rows, against Qwen's 28 reproduced R rows in §38. Stage 1's ANS gate ("Steer
   answers on ≥ 0.5 of R") would be read on about 13 rows for Llama.
2. **The jlens BOS guard fails on all three families** (see §1). I confirmed it on the laptop for Llama and Gemma under
   both `force_bos=True` and `force_bos=False`, since the HF default `add_special_tokens=True` adds the BOS anyway
   (`lensplumb.py`). With a guard that encodes as the readout does, all 5/5 rows read for both families. `jlens_eval.py`'s
   `logits_at` has the same guard and would fail the same way. For Phi, `force_bos` also puts `<|endoftext|>` at the
   start of the WikiText fit prompts, while Phi's chat prompts carry no BOS. Whether Phi is fitted with or without it is
   a decision for rev 1.
3. **Llama's template writes today's date into the prompt** ("Cutting Knowledge Date: December 2023 / Today Date: 26 Oct
   2026"). vLLM rendered the pod's date in §24, which I take to be 09 Oct 2026, the §24 run date. It is not recoverable
   from the cache, which stores only the user message. I regenerated with `date_string="09 Oct 2026"`. With today's
   default date the 5 A rows give the same 5/5, but the input ids differ, so any Llama run should pin `date_string`
   (one fixed string for the laptop and the pod).
4. **Stop ids.** `HFEngine.stop_ids = {<|im_end|>, tokenizer.eos}` misses Gemma's `<end_of_turn>` (106; tokenizer eos
   is `<eos>`) and Phi's `<|end|>` (200020; eos is `<|endoftext|>`). Generation would run past the turn end, and decode
   with `skip_special_tokens` hides it. For Llama, eos is `<|eot_id|>`, so it works by luck. Use
   `generation_config.eos_token_id` plus the turn-end token.
5. **fp32 on the laptop.** A direct fp32 `from_pretrained` of Gemma 4B and Phi-4-mini hits the 22 GB address-space cap:
   the bf16 checkpoint stays mmapped while the fp32 copy is built (Gemma VmSize 10.2 GB after a bf16 load, + 17.2 GB
   fp32). A per-parameter upcast after a bf16 load fails for the same reason. What works is a one-time streamed fp32 copy
   of the checkpoint, the same tensors upcast exactly, 1 GB shards, in `cache/exp43_fp32/` (gitignored, 31 GB, delete
   after stage 1). The fp32 load is then a zero-copy mmap (VmPeak 19.6 / 17.5 GB). Llama loads directly, but only with
   `MALLOC_ARENA_MAX=2 HF_ENABLE_PARALLEL_LOADING=false`. Without them it failed at 22 GB.
6. **Gemma's config has no top-level `hidden_size`** (`Gemma3Config`, multimodal), so `HFEngine`'s
   `net.config.hidden_size` raises. Use `config.get_text_config()`. Its layers sit under `model.language_model.layers`
   (the vision tower's encoder layers also match a `*.layers` search, so pick the text path explicitly).
7. **Digit tokenisation differs, which changes what the P1 target is.** Like Qwen, Gemma splits numbers into single
   digits after a bare `▁`. Llama and Phi keep up to 3 digits per token after a bare `Ġ` (checked: " 18 days 2023" →
   `Ġ 18 Ġdays Ġ 202 3` for both; Gemma and Qwen give `1 8` and `2 0 2 3`). The P1 whitespace teacher-forcing works unchanged
   (`n_leading_ws` = 1 for numbers, 0 for words, in all three), but "the gold's first content token" is a different
   unit across families. That is a descriptive caveat for cross-family emergence depths.
8. **Phi-4-mini drops the marker on 29% of rows** (0.706 on the population), against ~1% for Llama and ~4% for Gemma.
   Its population shrinks to 89 marked rows.
9. Decoder layers return a bare `Tensor` in transformers 5.17 for all three (and for Qwen). The `steering.hooked`
   tuple/tensor branch already handles both.

## 4. Template details (exact)

- Llama 3.2: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: {date}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`.
  With a system message, its text goes after the date lines in the same system block.
- Gemma 3: `<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`. A system message does **not**
  error: the template folds it into the first user turn as `{system}\n\n{prompt}`. §24 sent no system message (the
  `openai` backend sends one user message, `memctl/llm.py`), so nothing needs folding for the §23 prompt. If a family
  config ever adds a system text, the Gemma rule is "system + `\n\n` + user, in the user turn", which is what the
  template does.
- Phi-4-mini: `<|user|>{prompt}<|end|><|assistant|>` (system: `<|system|>{s}<|end|>` first; no default).
- The §23 prompt is the user message, and none of the three injects a default system text except Llama's date block.
  §24's prompts are a single user turn rendered by vLLM's chat template. The cached `prompt` strings are that user
  message, the same bytes as Qwen's.
- The A continuation (`frozen_continuation`: note + `Answer:`) is appended by id concatenation directly after the
  assistant-turn header (`...assistant<|end_header_id|>\n\n`, `<start_of_turn>model\n`, `<|assistant|>`). This matches
  how vLLM continued in §24: the N-arm first tokens match the cache 5/5 in every family.
- BOS: Llama's and Gemma's chat text carries its BOS as text, so `encode(add_special_tokens=False)` (as `HFEngine` and
  `jacobian._ids` do) gives exactly one BOS. Phi has none. This is correct as committed, and wrong only in the jlens
  guard.

## 5. Part B (amendment 1) inputs

**Population (§34 rule: answerable, all evidence in view, ≥ 1 evidence line, both sizes, same prompt sha256, judge
label on both).** Counts per fold 0/1/2/3/4: **61 / 65 / 66 / 68 / 70 = 330** for Llama (3B + 8B), Gemma (4B + 12B) and
Phi (mini + 14B), the same per-fold counts as Qwen 3B/7B under §34's rule. 100% are paired by prompt hash and 100% carry
both judge labels. Folds 2–4 were counted only; no outcome was read.

**Large partners (access only; not downloaded or loaded):** Llama-3.1-8B-Instruct (32 layers, d 4096, 16.1 GB),
gemma-3-12b-it (48 layers, d 3840, 24.4 GB, same Gemma3ForConditionalGeneration layout), phi-4 (Phi3ForCausalLM,
40 layers, d 5120, 29.3 GB). jlens's `_find_layout` on a meta-device instance resolves all of them: `model` for Llama
and Phi3 (Phi3 has `norm`, so the "Phi" `final_layernorm` layout is not used), `model.language_model` for Gemma 3. None
has a final-logit softcap in its config. **The external `jacobian-lens` (581d398) supports all six architectures as far
as layout goes.** I checked Llama 3B and Gemma 4B end to end through `read_record` on the laptop.

**Laptop logit-lens read, 5 dev rows** (`lensplumb.py`; identity transport, so only the logit rows mean anything;
plumbing only, no claim). Llama 3B: 5/5 read, 28 layers, about 24 s/row on CPU fp32. Gemma 4B: 5/5 read, 34 layers,
about 37 s/row. P1 = P0 + 3 when the gold starts with a digit (the whitespace token teacher-forced), + 2 for words, as
§34. Tracked `' unknown'` ids are equal to the in-context ids above.

**Pod-time estimates (A40, bf16; J-lens 56 WikiText paragraphs, max_seq_len 128, dim_batch 32 as §34).** The fit time
is scaled by a power law in d²·L fitted through §34's two fits (Qwen 3B 1,931 s, 7B 5,419 s; exponent 1.19). Reads
assume 330 prompts at §34's 0.51–0.55 s/question, scaled up by parameter count above 7.6 B. Downloads assume
~150 MB/s. These are rough, ±50%.

| model | fit 56 (min) | read 330 (min) | download (min) |
|---|---|---|---|
| Llama 3.2 3B | ~63 | ~3 | ~1 |
| Llama 3.1 8B | ~145 | ~3 | ~2 |
| Gemma 3 4B | ~51 | ~3 | ~1 |
| Gemma 3 12B | ~202 | ~5 | ~3 |
| Phi-4-mini | ~73 | ~3 | ~1 |
| Phi-4 14B | ~322 | ~6 | ~3 |

Per family: logit reads + downloads for both sizes are about 8 / 11 / 13 min (Llama / Gemma / Phi). Both J-lens fits
are about 208 / 253 / 396 min. Within a 3 h stop, the small-model fit fits in every family. The large fit fits only if
it starts right after the small one, and only for Llama (63 + 145 = 208 min, still over 180). It does not fit for Gemma
12B or Phi-4. Unverified: Phi-4 (29 GB bf16) plus the fit graph on a 48 GB A40, and Gemma 12B's 262k-vocab unembed at
48 layers.

## 6. Qwen-specific code (what would break or silently differ)

`memctl/analysis/steering.py`
- `N_LAYERS = 36`, `HIDDEN = 2048` (defaults of `depth_layer`, `final_layer`, `specs_for(..., hidden=HIDDEN)`).
  `map_layer` rescales a 36-layer index to another depth, which is meaningful only for the stub. A new family needs
  its own layers, not mapped ones.
- `HFEngine.__init__`: `self.net.model.layers` (wrong for Gemma 3: `model.language_model.layers`);
  `self.net.config.hidden_size` (raises for Gemma 3); `stop_ids = {convert_tokens_to_ids("<|im_end|>"), eos}` (misses
  Gemma's `<end_of_turn>` and Phi's `<|end|>`); fp32 `AutoModelForCausalLM.from_pretrained(model_id)` on CPU (over the
  laptop guard for Gemma 4B and Phi-4-mini; needs a local fp32 copy path).
- `chat_text` → `jacobian.wrap_chat`: generic, but needs a pinned `date_string` for Llama.
- `StubEngine.chat_text` uses `QWEN25_CHAT` (stub only; `answer_selection.QWEN25_CHAT` is the Qwen template string).
- Population: `populations(data)` takes a `scripts/exp35_report.ReaderData`, i.e. §35's vLLM A/B/Ncut rows and the §34b
  token check (`runs/exp34_jlens_own`). No such rows exist for the new families, so stage 1 has to generate A, B (and
  N-cut for the cut control) itself, or take R from the §24 cache plus a reproduction pass, as here.

`memctl/analysis/exp40.py` (and `scripts/exp40_pod_run.py`, `exp38_pod_run.py`)
- `LAYER = 29`, `ALPHA = 2`, the v_29 / steer-vector SHA-256s, `runs/exp38_dev/qwen3b/directions.npz`.
- `DEFAULT_REFUSAL_IDS = (9788,)`, `REFUSAL_ID42 = 9788`, `RULE42_TEXT` ("id 9788"); `exp40_pod_run.py:243` uses the
  rule's refusal id; `tests/test_exp41.py` and `test_exp42.py` assert 9788.
- §40/§42 paths: `runs/exp40_pod/qwen3b/{view,apass,selection}.json`, `READER = "qwen3b"` (also `exp38_dev.py`,
  `exp39_dev.py`, `exp39b_dev.py`, `exp41_dev.py`, `--source runs/exp38_dev/qwen3b`).
- `scripts/exp39b_dev.py`: the unknown-first id is computed with `continuation_first_id`, which is generic. Only its
  docstring names 9788.

Data loaders
- `memctl/jacobian_data.load_records`: the run dir `runs/exp23_{reader}_f{fold}`, the cache
  `cache/generations_exp23_{reader}` and `runs/exp23_verdicts.json` are hard-coded. §24 needs `exp24_*`, and its
  verdicts file carries a `fold` key and arms `fixed8` / `fixed16` / `fifo_top5_t3000`. `data.py: load_records24` here
  is the adapted copy.
- `scripts/exp35_generate.py`: `READERS = ("qwen3b", "qwen7b")`, `MODELS` (Qwen only), `JUDGE_MODEL` (Qwen 7B, to keep).

jlens pipeline (`memctl/analysis/jlens_readout.py`, `memctl/analysis/jacobian.py`, `scripts/jlens_fit.py`,
`jlens_read.py`, `jlens_eval.py`, `exp34_report.py`, `exp34_pod.sh`)
- **Tokenizer / marker:** `readout_for` builds ids by concatenation (`ids(chat) + ids("Answer:") + leading`), which is
  generic and correct. The `read_record` guard (`lens_model.encode(chat)`) is the BOS bug in §3.2. `jlens_eval.logits_at`
  has the same guard.
- **P1 construction:** `_margin_length` / `leading` teacher-force the whitespace-only gold tokens. This is generic
  (`▁` / `Ġ` behave as Qwen's `Ġ`), but the first content token's granularity differs (§3.7).
- **`' unknown'`:** `unknown_id` = `ids(" unknown")[0]` computed per tokenizer, so nothing is hard-coded in the lens
  pipeline. It equals the in-context id for all three.
- **Layers and hooks:** `jlens.from_hf` resolves the layout per family (§5). `n_layers` and `d_model` come from
  `get_text_config()`, final layer = `n_layers - 1`, ranks are taken over the lens layers plus the model's final
  logits. Nothing is Qwen-specific beyond `MODEL_IDS` (Qwen only in `jlens_fit.py`, `jlens_read.py`, `jlens_eval.py`),
  `load_records`' exp23 paths, `exp34_pod.sh`'s preflight (exp23 inputs, `qwen3b` / `qwen7b`, `transformers==5.17.0`),
  and `exp34_report.py`'s `--small/--large` defaults.
- **BOS in the fit:** `force_bos=True` is right for raw WikiText on Llama and Gemma. For Phi it adds `<|endoftext|>`,
  which is a choice to make.

## 7. Proposed per-family config (design note only, nothing refactored)

```python
@dataclass(frozen=True)
class FamilyConfig:
    key: str                     # "llama3b", "gemma4b", "phi4mini", "qwen3b"
    model_id: str                # HF id (tokenizer always from here)
    weights: str | None          # local fp32 copy for the laptop (to_fp32.py), else None
    partner_key: str             # §34 pair: "llama8b", "gemma12b", "phi4", "qwen7b"
    partner_model_id: str
    exp_tag: str                 # "exp24" / "exp23": runs/{tag}_{key}_f{fold}, cache/generations_{tag}_{key}
    verdicts: str                # runs/exp24_verdicts.json (+ arm "fixed8", fold key)
    layer_path: str              # "model.layers" | "model.language_model.layers"
    n_layers: int; hidden: int   # from get_text_config(), checked at load
    template_kwargs: dict        # {"date_string": "09 Oct 2026"} for Llama, {} otherwise
    system_text: None            # §23 has none; Gemma rule if ever set: fold "system\n\nuser" into the user turn
    stop_tokens: tuple[str, ...] # ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>") / ("<end_of_turn>", "<eos>") / ("<|end|>", "<|endoftext|>") / ("<|im_end|>", "<|endoftext|>")
    refusal_ids: tuple[int, ...] # first ids of " unknown" / " Unknown": (9987, 22435) / (11908, 60355) / (17927, 41855) / (9788, 21693)
    bos_in_chat: bool            # True Llama/Gemma (BOS is text in the template), False Phi/Qwen
    lens_force_bos: bool         # fit-time BOS for jlens (Phi: decide)
    laptop_env: dict             # {"MALLOC_ARENA_MAX": "2", "HF_ENABLE_PARALLEL_LOADING": "false"}
    sweep_depths: tuple = (0.6, 0.7, 0.8, 0.9)   # stage 1 grid, layers = round(depth * n_layers), chosen in stage 1
```

The refusal ids should be checked at load as `continuation_first_id(engine, note + "Answer:", " unknown")` and
asserted equal to the config, so a tokenizer revision cannot change them silently. The directions, the chosen layer,
α and the §41/§42 rule constants stay per-family artefacts with their own SHAs, never config defaults.

## 8. Files

- `runs/exp43_stage0/stage0.json`: everything below, merged.
- `access.json` / `access_large.json`: HF access. `tok.json`: templates, ids, last-token checks. `data.json`: rates.
  `pop34.json`: Part B counts. `refusal_overlap.json`: overlap with Qwen's dev refusals.
- `{llama3b,gemma4b,phi4mini}/gen*.json`: load, timing, regeneration rows. `{llama3b,gemma4b}/lensplumb.json`: lens
  plumbing.
- Scripts: `access.py`, `access_large.py`, `download.py`, `data.py`, `tok.py`, `gen.py`, `to_fp32.py`, `lensplumb.py`,
  `pop34.py`, `assemble.py` (all run with `.venv/bin/python`; model steps under `GUARD_VM_GB=22
  runs/_pipelines/guard.sh` with the `laptop_env` above).
- fp32 copies: `cache/exp43_fp32/{gemma-3-4b-it,Phi-4-mini-instruct}` (31 GB, gitignored; rebuild with `to_fp32.py`).
