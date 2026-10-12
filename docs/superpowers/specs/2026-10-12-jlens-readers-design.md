# J-lens on small readers: where in depth does the evidence become sayable? (design, 2026-10-12)

Status: design agreed in chat on 2026-10-12 (J-lens primary; the §33 input-gradient run is the secondary
measurement). Experiment number: **§34**. Branch `jacobian-lens` (shared with §33). Pre-registration drafted
in `research/proposals/` before any confirmation-fold number exists.

## 1. Question

Experiment 23 showed that with identical retrieved turns in the prompt, a Qwen 2.5 3B reader answers
correctly on 0.397 of the questions whose evidence is fully in view, against 0.667 for the 7B. The 3B's
failures are about half refusals ("unknown") and half wrong answers. §33 asks how much the output depends
on the evidence tokens. §34 asks the complementary question: **does the gold answer ever become sayable
inside the 3B, and if so, where is it lost?**

The instrument is Anthropic's Jacobian lens (J-lens; Gurnee, Sofroniew et al., "Verbalizable
Representations Form a Global Workspace in Language Models", arXiv:2607.15495; reference code
`anthropics/jacobian-lens`, pinned at `581d398`, vendored under `external/jacobian-lens/`, Apache 2.0).
It transports a residual-stream vector at layer `l` into the final-layer basis with a corpus-averaged
Jacobian `J_l = E[∂h_final/∂h_l]` and decodes it with the model's own unembedding, so it reads which
tokens an activation is disposed to make the model say, including tokens it never emits, and stays
readable in early layers where the logit lens is noise.

For each question and each reader we read the lens at the answer position at every layer and track the
rank of the gold answer's first content token and of `unknown`. Each question then falls into one of
three outcomes:

- **A, never present:** the gold token never enters the top-k at any layer. The model did not extract
  the answer from the memory.
- **B, present then lost:** the gold token enters the top-k at some layer but is outside it at the final
  layer. The model read the answer and something downstream overrode it.
- **C, present to the end:** the gold token is in the top-k at the final layer.

The 3B-vs-7B comparison on identical prompts, and the A/B/C split of the 3B's own failures, is the result.

## 2. Data (nothing new is generated)

Identical to §33 §2: the cached `fixed8` prompts of §23, byte-identical for the 3B and 7B, with the gold
evidence lines marked and the §23 verdicts attached, through `memctl/jacobian_data.py`
(`load_records`). Population for the claims: answerable questions with all evidence in view (about 330).

## 3. Models and lenses

Readers: Qwen2.5-3B-Instruct (36 layers, d_model 2048) and Qwen2.5-7B-Instruct (28 layers, d_model 3584),
bf16, Hugging Face transformers, wrapped with `jlens.from_hf`.

**Lens fitting.** One lens per reader, fitted by `jlens.fit` on the same corpus with the same settings:
paragraphs of at least 150 words from the WikiText-103 test split
(`data/jlens_corpus/wikitext-103-raw-v1-test.parquet`, 591 paragraphs available; generic web-like text as
the paper and README prescribe), 128 tokens each, `skip_first=16`, every layer below the final one as a
source layer, target = final layer. Target `N = 128` prompts, minimum `N = 32`; fitting is checkpointed
every prompt and **time-boxed per reader** (see §9), so the lens is usable at whatever `N` the box allows.
The 7B is fitted first; the 3B is then fitted with `N` capped at the 7B's achieved count (same seed, so the
same prompt prefix), so both lenses share the corpus, the settings and `N`.
The lens file records `n_prompts`. No Qwen2.5-3B lens exists on the Hub; two community lenses exist for
Qwen2.5-7B-Instruct (`andyx10/jacobian-lens-qwen2.5-7b-instruct`, `anicka/jlens-qwen2.5-7b-instruct`,
loaded with `weights_only=True`). One of them is a **descriptive cross-check** of our 7B lens (top-10
overlap at the answer position on the development folds) and the fallback if the 7B fit hits its time box
below the minimum.

**Lens validity gate (development folds; must pass before the claims are read).** The fitted J-lens must
beat the vanilla logit lens (`use_jacobian=False`) on the same activations: lower mean log-rank of the gold
first content token over the middle third of layers at the answer position, paired by question, interval
excluding 0, for each reader. As a second check, on the repository's own `data/evaluations/*.json` prompt
sets the J-lens must rank the expected token higher than the logit lens does at mid layers. If the gate
fails for a reader, its claims are reported with the logit lens and flagged.

## 4. Readout positions

One forward per question per reader on the chat-wrapped prompt (same `wrap_chat` as §33) with the fixed
continuation `Answer:` appended. From that forward:

| id | position | role |
|---|---|---|
| `P1` (primary) | the last token of `Answer:` plus the gold answer's leading whitespace-only tokens (teacher-forced by id concatenation; usually none, or one bare space before a digit), so the position predicts the gold's first content token | the decision position; matches §33's `margin` definition |
| `P1_raw` (descriptive) | the token right after `Answer:` | the same position without the whitespace fix; for a content-first gold it equals `P1` |
| `P0` (secondary) | the last token of the generation prompt, before `Answer:` | the workspace at the end of the prompt, before any answer-format cue |
| `E` (descriptive) | the last token of each gold evidence line | is the answer sayable where the evidence sits? |
| `Q` (descriptive) | the last token of the question line | is it sayable once the question is read? |

Tracked tokens at every position and layer: the gold answer's first content token (as in §33 §5, leading
whitespace-only tokens skipped), ` unknown` and `unknown`, plus the top-10 token ids and their lens logits
per layer, and the model's final-layer logits at the same positions. Stored per question as one `.npz`
(ranks `[n_layers, n_tracked]` per position, top-10 ids and logits) and one `.json` of the measures.

## 5. Measures (per question, reader, position)

With `k` the top-k threshold (default 10; fixed on the development folds):

- **best rank** = min over layers of the gold token's rank (1 = top).
- **emergence depth** = the first layer at which the gold token's rank ≤ k, as a fraction of depth
  (`layer / n_layers`), so 36- and 28-layer models compare; `None` if never.
- **final rank** = the gold token's rank at the final layer (the model's own next-token ranking).
- **outcome** ∈ {A, B, C} as defined in §1, at `P1`.
- **unknown trajectory** = the rank of ` unknown` per layer, and the first layer at which it reaches the
  top-1 (`None` if never).

## 6. Pre-registered claims (confirmation folds 2, 3, 4; reported on all folds too)

Population: answerable questions with all evidence in view under `fixed8`, present for both readers.
Paired bootstrap over questions, 10,000 resamples, 95% intervals.

- **Claim 1 (size).** At `P1`, the 7B's best log-rank of the gold token is lower than the 3B's, paired by
  question, interval excluding 0. Secondary: the 3B's share of outcome A exceeds the 7B's (paired
  difference of proportions, interval excluding 0).
- **Claim 2 (within the 3B).** Among in-view questions, the 3B's best log-rank is worse on the questions
  it gets wrong than on those it gets right: AUC above 0.5 with the interval excluding 0.5.
- **The decomposition (the interpretive result, no directional claim).** Among the 3B's in-view
  failures, the shares of A, B and C with bootstrap intervals, split by refusal versus wrong answer; the
  same for the 7B. The reading: mostly A means the 3B does not extract the answer; mostly B means it
  extracts and overrides, which points at the refusal rule and the late layers rather than at reading.
- **Descriptive.** Emergence-depth distributions per reader; `E` and `Q` positions; the `unknown`
  trajectory on refusals; the cross-check against the community 7B lens; per question type; the §33
  evidence share against the §34 outcome on the same questions (the two instruments on the same data).

**Gate for stage 2.** The lens validity gate passes for both readers and Claim 1 holds. Stage 2 would
intervene (for example, add the lens-transported gold direction or suppress the `unknown` direction at
the emergence layer) to test whether the B outcomes are causal; it gets its own spec.

## 7. Prediction, recorded before any data (for the write-up)

The 3B's refusals are mostly outcome B: the gold token rises into the top-k in the middle third and
` unknown` overtakes it in the last third. The 7B shows C on most of the same questions. The 3B's wrong
answers (not refusals) look more like A. Emergence, when it happens, is at a similar fractional depth in
both readers.

## 8. Split and leak rules

As §33 §9: folds 0 and 1 are development (k, the "middle third" layer window, the corpus size, the lens
gate thresholds are fixed there and written into the pre-registration); the pod run reads all five folds
but the report tool refuses folds 2–4 without `--confirm` until the pre-registration is committed with a
hash. The reviewer signs the pre-registration before the paid run.

## 9. Compute, cost and controls

- **Fitting cost.** `jlens.fit` costs one forward and `d_model / dim_batch` backward passes per prompt;
  the parameters are frozen, so a backward pass costs about one forward, and the total per prompt is about
  `d_model × forward(128 tokens)`, independent of `dim_batch`. Measured here (0.5B, CPU, float32,
  `dim_batch` 8): 1.44 s per backward pass over 8 × 128 tokens, 161 s for one prompt with two source
  layers. A community fit of the same recipe for Qwen2.5-7B-Instruct (100 WikiText prompts, 128 tokens,
  `skip_first` 16, `dim_batch` 64, all 27 source layers) took 18–22 s per prompt on one H100, 34 minutes
  in all (`andyx10/jacobian-lens-qwen2.5-7b-instruct` model card). On an A40 expect roughly 3–4× that:
  about 1–2 hours for the 7B at `N = 100` and about 30 minutes for the 3B, so about $1.50 for both
  lenses at $0.59/h, with the H100 (about $2.7/h, under an hour for both) as the alternative if A40
  stock is short. Fitting the 7B is the cost driver; the 3B is about a quarter of it.
- **Time box.** The pod script fits each reader under `--max-minutes` with checkpoints every prompt;
  the lens is saved at whatever `N ≥ 32` the box allows, and the run reports `N`. The box and the hard stop
  go into the user's approval quote. No question is asked while the pod bills.
- **Reads.** One forward per question per reader (500 × 2) plus per-layer unembeds at four positions:
  minutes on an A40.
- **Secondary.** The §33 gradient run (`scripts/exp33_pod.sh`) runs on the same pod after the reads.
- **Local development.** The 0.5B on CPU: a lens fitted on a handful of prompts for plumbing, or the
  community `MildHotSauce/jlens-qwen2.5-0.5b-instruct` lens (`weights_only=True`), descriptive only.
- **Budget.** $20 programme cap, about $13.75 spent. The quote states the time box so the total stays
  under the cap; the user approves before launch; pre-launch checklist as §33.

## 10. Code

| path | purpose |
|---|---|
| `memctl/analysis/jlens_readout.py` | positions from the chat-wrapped prompt and the memory lines; tracked-token ids; ranks, emergence depth, outcome A/B/C from lens logits; logit-lens baseline. Imports `jlens` and `torch` lazily |
| `scripts/jlens_fit.py` | corpus loading (parquet → paragraphs → prompts), `jlens.fit` with checkpoint, `--max-minutes`, `--n-prompts`, `--reader`; writes `runs/exp34_jlens/lenses/<reader>.pt` and a `.json` sidecar (n_prompts, settings, wall time) |
| `scripts/jlens_read.py` | per-question readout over `load_records`, resumable per question, `--lens` path, `--logit-lens` baseline switch, writes `runs/exp34_jlens/<reader>/<qid>.{npz,json}`, `_timing.json`, `_failed.json` |
| `scripts/exp34_report.py` | lens gate, claims, decomposition, descriptives, fold guard with `--confirm` |
| `scripts/exp34_pod.sh` | preflight of inputs, fit both lenses under the time box, read both readers, then `scripts/exp33_pod.sh` |
| `tests/test_jlens.py` | positions and tracked ids on real cached prompts; rank/outcome arithmetic on synthetic logits; a fit-and-read round trip on the tiny random Qwen2 model; fold guard and resume |

`external/jacobian-lens` is pinned in `research/EXPERIMENT_INDEX.md` like the other external code. The
package is installed editable into `.venv` (`pip install -e external/jacobian-lens`); the pod installs the
same commit.

## 11. Out of scope

Interventions (stage 2); other model families; fitting lenses on conversation text; any change to the
readers or the controller; re-running the §23 readers.
