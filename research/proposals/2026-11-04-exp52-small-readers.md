# §52 (proposed): do readers smaller than 3B abstain with the evidence in view? A feasibility screen, then a queue

**Status: proposed, rev 0 (Advisor, 2026-11-04). Not built, not run, nothing spent.** Decision D-74 (the user's order).
The section number is provisional until the Executor opens the section in EXPERIMENTS.md.

## Why

§43 stage 0 measured in-view abstention at one size, about 3–4B:

| Model | In-view "unknown" among marked dev rows |
|---|---|
| Qwen 2.5 3B | 0.300 (30/100) |
| Llama 3.2 3B | 0.104 (13/125) |
| gemma-3-4b | 0.017 (2/121) |
| Phi-4-mini | 0.034 (3/89) |

So the habit is more common in Qwen in that setting, not exclusive to it. The record does not yet show whether it
follows the family or the size. The user's question is whether readers below 3B also say "unknown" when the evidence
is in view. If they do, each such model is queued for the four regimes already run on Qwen 3B: activation steering
(§38–§40), the lens readout (§34, §43 Part B), activation patching (§44, §47), and weight editing (§51).

## Readers (instruction-tuned, all below 3B)

| Family | Readers | Notes |
|---|---|---|
| Qwen 2.5 | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct | completes a size ladder with 3B (0.300) and 7B |
| Llama 3.2 | Llama-3.2-1B-Instruct | gated; the Llama licence is already accepted (§43) |
| Gemma 3 | gemma-3-1b-it (and gemma-3-270m-it as an optional floor) | gated, accepted; no system role (handled in `memctl/analysis/families.py`) |
| SmolLM2 | SmolLM2-1.7B-Instruct | a new family, ungated; 8k context is enough for `fixed8` |

Phi has no instruction-tuned reader below Phi-4-mini, so it is left out. Qwen3 (0.6B, 1.7B) is left out of rev 0
because its thinking mode changes the output format; it could be added as its own arm later.

## Stage 0: the screen (laptop, $0)

**Same setting as §43 stage 0, so the numbers compare.**
- Population: the 126 dev rows (folds 0–1, the §35 population). Folds 2–4 stay sealed.
- Prompt: the frozen reader prompt (7b5fc30), the `fixed8` arm, with the same contexts as §24 (the controller does
  not depend on the reader). One prompt for all readers; no tuning for any model.
- Greedy decoding, each model's own chat template, fp32 on the laptop CPU. Each model's family config lives in
  `memctl/analysis/families.py`, and the Qwen defaults stay byte-identical.
- New: these models have no §24 cache, so stage 0 generates their outputs once and saves them as the cache that every
  later stage reuses.

**Measured per model** (definitions as in §43):
1. **Marker rate:** the share of outputs with the `Answer:` marker.
2. **In-view abstention rate:** among marked rows with all the evidence in view, the share whose answer starts with
   "unknown".
3. **Reading floor:** the share of the 126 rows whose answer contains the gold answer (string containment, a free
   proxy). The Qwen 7B judge labels can be added later on a pod, with their own quote, and are descriptive only.
4. **Descriptive only:**
   - the "unknown" rate on rows where the evidence is *not* all in view (does the habit respond to the evidence?);
   - the share of in-view abstentions whose own note already contains the gold answer (the §49 "note, then refuse"
     signature);
   - the §24 refusal-phrase count for refusals in other words.

**Gate, fixed before any output is seen.** A model is queued only if all three hold:
- marker rate ≥ 0.70. Below that, it is reported as "does not follow the format; not measurable here";
- reading floor ≥ 0.15. Below that, an "unknown" can't be read as an evidence-use failure, because the model barely
  reads;
- in-view abstention rate ≥ 0.15, with at least 10 such rows (the §43 line; Llama's 13 rows were enough for its
  stage 1).

A model that fails the gate is a result, not a dead end. It is reported, and it says where the habit is absent.

**Timing.** Measure one full row per model before quoting a time (the measure-before-ETA rule). The only guide so far:
Llama 3.2 3B ran at 14.6 ms/token prefill and 0.80 s/token decode on this laptop, so models of 1.5B or smaller should
be faster.

**When.** After the §51 §8.4.2 replay finishes (expected Friday morning, 2026-11-06). The replay uses the laptop CPU
and is on §51's critical path. Stage 0 must not slow it.

## The queue (only for models that pass the gate)

Each item below is its own pre-registration (rev 0, reviewed before it is built), with its own quote if it needs a
pod. **This proposal approves none of them**, and none starts before the stage-0 report is reviewed.

| Order | Regime | Recipe reused | Where it would run |
|---|---|---|---|
| Q-A | Activation steering | §38/§39 direction, gated as in §43 Llama stage 1 (estimation pairs from rows with the evidence not all in view; gate on the in-view abstentions, which the estimation never sees; the same four bars) | laptop |
| Q-B | Lens readout | §34 and §43 Part B: logit lens first, J-lens only if its fit gate passes | laptop for logit; J-lens may need a pod |
| Q-C | Activation patching | §44 clean→cut and swap, then §47 interchange if §44 opens | laptop where feasible; otherwise a quoted pod |
| Q-D | Weight editing | the §51 pipeline (editor, frozen splits, sealed confirmation) | only after §51 reports on S/H/T and only if an edit is confirmed; editor API spend is quoted separately |

At most two readers go to Q-A to Q-D. If more pass, the order is the highest in-view abstention rate first, then the
family not yet covered.

## Rules that apply

- **No held-out peeking:** stage 0 and every queued stage use dev folds only until a confirmation pre-registration
  says otherwise.
- **D-45 stopping rule:** a failed gate or stage is documented, not rescued.
- **Spending:** every pod or API spend needs a quote and the user's own approval line, in the Executor pane for a pod,
  inside the $36 cap (D-73), never two pods at once. §51's pods S and H+T take priority.

## Confounds, stated in advance

- **One prompt for all readers.** The prompt was tuned on Qwen 7B, and the §43 record already notes that the 3B's rate
  is partly a prompt-format effect. A difference between models may be a difference in how they handle this prompt.
  A prompt-variation arm is a separate proposal.
- **Format failures grow as models shrink.** Marker rate is reported first for this reason, and the gate excludes
  models that don't follow the format.
- **String containment is a rough proxy** for correct answers. It is used only for the reading floor, not for a claim.
- **Instruction-tuned only.** Whether instruction tuning teaches the habit (base versus instruct) is a separate
  question.
