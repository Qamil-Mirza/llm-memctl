# Jacobian lens on small readers: why a 3B ignores evidence it was given (design, 2026-10-12)

Status: design approved in chat on 2026-10-12. Experiment number: **§33** (the next free section of
EXPERIMENTS.md). Branch `jacobian-lens`. The pre-registration text for EXPERIMENTS.md is drafted from
this spec in `research/proposals/` before any confirmation-fold number exists.

## 1. Question

Experiment 23 showed that with the same eight retrieved turns in the prompt, a Qwen 2.5 3B reader
answers correctly on 0.397 of the questions whose evidence is fully in view, against 0.667 for the 7B
and 0.773 for the 14B. The 3B's failures are about half refusals ("unknown") and half wrong answers.
The hypothesis is that the 3B's output depends less on the evidence tokens than the 7B's does, on the
identical prompt. The **Jacobian lens** measures that directly: the gradient of a fixed scalar target
with respect to the input token embeddings, read region by region.

This is stage 1 of a two-stage plan (option 3 of the brainstorm). Stage 1 is the gate: an
input-embedding Jacobian on cached prompts. Stage 2, a layer-wise residual Jacobian that localises
where evidence information stops reaching the answer position, runs only if the gate passes and gets
its own pre-registration.

## 2. Data (nothing new is generated)

- **Prompts.** The exact prompts the readers saw in §23 under the `fixed8` arm are in the generation
  cache (`cache/generations_exp23_qwen3b/**.json` and `..._qwen7b`, key `prompt`, with the cached
  `output`). The 3B and 7B received byte-identical prompts (the controller and its picks do not depend
  on the reader). About 1,049 tokens each. 500 questions, of which 470 are answerable.
- **Evidence spans.** `runs/exp23_<reader>_f<k>/fixed8__fraction0.05/failures.jsonl` and
  `episodes.jsonl` give, per question, the gold evidence turns (`needle` text and carrier ids) and
  whether all evidence was in view (`evidence_complete_rate == 1.0`). For correct questions the
  evidence is read from the environment (`memctl/envs/longmemeval.py`, the `has_answer` turns), the same source the failure labels use.
- **Verdicts.** `runs/exp23_verdicts.json`: per (reader, arm, question) `correct`, `unknown`, `answer`,
  `gold`, `type`. One Qwen 7B judge (STATE.md caveat 3 travels with it).
- **Population.** Answerable questions whose evidence is fully in view under `fixed8`: about 330.
  Within it (3B / 7B outcomes on the same prompt): both right about 116; 7B right and 3B wrong 104;
  both wrong about 95; 3B right and 7B wrong about 15. Folds 0 and 1 hold about 130 of the 330.

## 3. Models

Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct (the §23 readers), Hugging Face transformers, bf16,
eval mode, no dropout, gradients enabled only with respect to the input embeddings. The chat template
is applied exactly as the vLLM server did (user message = cached prompt, generation prompt added, the
tokenizer's default system message included). A **regeneration sanity check**: 20 prompts per model
regenerated greedily with transformers must match the cached vLLM outputs on at least 15 (the §29b
noise floor allows a few differences); a worse match stops the run.

Qwen2.5-0.5B-Instruct is the local development model. It runs on this machine's CPU, exercises the
whole pipeline on real prompts, and its numbers are descriptive only.

## 4. Prompt segmentation

Each prompt is split into regions by string matching on the cached prompt, then mapped to token spans
with the fast tokenizer's offset mapping:

| region | definition |
|---|---|
| `template` | chat-template tokens (system message, role markers); excluded from every share |
| `instruction` | from the start of the user message to the line before `## Memory` |
| `evidence` | memory lines whose content contains a gold `needle` (all of them, for multi-turn evidence) |
| `distractor` | every other memory line |
| `question` | from `## Current input` to the end of the user message (`Note:` included) |

Every non-template prompt token belongs to exactly one region. A prompt whose evidence span cannot be
located is logged and dropped from the population, with its count reported.

## 5. Targets (all teacher-forced, identical for both models)

The continuation is appended as the assistant's output, after the generation prompt.

| id | continuation | scalar target |
|---|---|---|
| `gold` (primary) | `Answer: <gold answer>` | sum of log-probabilities of the gold answer tokens |
| `margin` (secondary) | `Answer:` + the gold answer through its first content token | log-probability of the gold answer through its first content token (leading whitespace-only tokens included, since the Qwen tokenizer splits digits off a bare space on 194 of 470 golds) minus the log-probability of ` unknown` at the first answer position. For a gold whose first token is content this equals the logit margin at that position exactly. The first gold token text is stored per question |
| `own` (descriptive) | the model's own cached output | sum of log-probabilities of the output tokens |

`gold` and `margin` share one forward pass; `own` needs a second. The gradient of each target is taken
with respect to the prompt's input embeddings only (continuation tokens are excluded from all shares).

## 6. Measures (per prompt, per model, per target)

Let `g_i` be the L2 norm of the gradient at prompt token `i` (the development folds also compute the
gradient-times-input variant `|grad_i · x_i|`; one of the two is fixed on the development folds and used
for confirmation).

- **Region share** `S_R = Σ_{i∈R} g_i / Σ_i g_i` for each region, over non-template tokens.
- **Evidence lift** `L = S_evidence / (|evidence| / N)`: share divided by the evidence's share of tokens.
  1 means "no more than its length".
- **Top-k evidence fraction**: the fraction of the `k` largest-`g_i` tokens that lie in `evidence`.
  `k = 50` unless the development folds pick otherwise.
- Stored per question: all region shares, the lift, the top-k fraction, the target values, the token
  counts per region, and the per-token `g_i` vector (compressed) for later plots.

## 7. Ablation check (development folds only, both models, `gold` target)

Remove the evidence memory lines from the prompt and recompute the `gold` target. The ablation effect
is `Δ = target(full) − target(ablated)`. The lens is **usable** if Spearman's correlation between
`S_evidence` and `Δ` across questions is at least 0.3 with a bootstrap interval excluding 0, for both
models. If it is not, the Jacobian does not track real dependence on this data and the gate fails on
method grounds; that result is reported as such.

## 8. Pre-registered claims (confirmation folds 2, 3, 4; reported on all folds as well)

Population: answerable questions with all evidence in view under `fixed8`. Paired bootstrap over
questions, 10,000 resamples, 95% intervals.

- **Claim 1 (the hypothesis).** Mean `S_evidence(7B) − S_evidence(3B)` on the `gold` target is
  positive with the interval excluding 0. Same test on `margin` as the secondary.
- **Claim 2 (within the 3B).** Among in-view questions, `S_evidence(3B)` is lower on questions the 3B
  gets wrong than on those it gets right: AUC above 0.5 with the interval excluding 0.5, and the
  difference of means reported. The same statistic for the 7B is descriptive.
- **Descriptive, no claim.** Where the 3B's gradient mass goes instead (instruction, question,
  distractor shares); the `own` target on refusals, that is what "unknown" depends on; the four
  3B-vs-7B outcome cells; per question type.

**Gate.** Stage 2 is proposed only if the ablation check passes and Claim 1 holds. Otherwise §33 is
closed with the result written up, and the Jacobian is not used for the pivot.

## 9. Split and leak rules

Folds 0 and 1 are development: every method choice (gradient-mass variant, `k`, the ablation bar, any
filter) is fixed there and written into the pre-registration. The paid run computes all five folds in one
pass (raw per-token gradients for both mass variants, so every method choice can be made later), but no
table, figure or number is produced from folds 2 to 4 until the pre-registration is committed with a hash;
the report tool refuses those folds without an explicit flag. The reviewer signs the pre-registration before the paid
run. The §23 verdicts are existing data and are read for both halves; no new generations are made.

## 10. Compute, cost and controls

- **Local (free):** development and tests on CPU with the 0.5B; the 3B also fits this machine's
  23 GB of RAM for a timed stub if needed.
- **Paid run:** one RunPod A40 (about $0.49/h), transformers, batch size 1, both models downloaded on
  the pod. Estimate from the development stub before launch; expected about 1 to 1.5 hours including
  downloads, about $1, hard stop $2. The user approves the quoted price before launch. Pre-launch
  checklist (user rule): full-size stub run, peak memory, cost from measured per-call time, resume
  test on a fresh process. Never leave the pod billing while waiting on a question.
- **Resume:** one output file per (model, question, target); existing files are skipped.

## 11. Code

| path | purpose |
|---|---|
| `memctl/analysis/jacobian.py` | segmentation, chat-template wrapping, targets, gradient pass, measures; no I/O beyond reading the cache and traces |
| `scripts/jacobian_lens.py` | CLI: `--reader`, `--folds`, `--targets`, `--ablate`, `--out`, `--limit`, `--device`; writes `runs/exp33_jacobian/<reader>/<question_id>.<target>.json` and `.npz` |
| `runs/_pipelines/exp33_report.py` | the statistics of §8 and §7, and `runs/exp33_report.md` |
| `tests/test_jacobian.py` | segmentation on real cached prompts, region coverage, target values against a direct log-softmax, gradient shares summing to 1, ablation plumbing; a tiny random Qwen2 config for speed |

The code lives in `memctl/analysis/` and must respect `tests/test_import_boundaries.py`. Results are
written to the main tree's `runs/` (symlinked from any worktree). The report copy goes to
`docs/research/exp33_jacobian_report.md`, figures to `docs/research/figures/`.

## 12. What stage 2 would be (not designed here)

If the gate passes: for each layer, the Jacobian of the final answer position's residual stream with
respect to the evidence-token residual streams, to find the depth at which evidence influence drops
off in the 3B versus the 7B. It gets its own spec and pre-registration.

## 13. Out of scope

Training or fine-tuning any reader; new reader generations; other model families (Granite 2B is a
possible later addition); the layer-wise analysis; any change to the controller.
