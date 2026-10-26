# §43 proposal, rev 0: does the refusal-direction result hold in other model families? (2026-10-26)

Author: Advisor. Ordered by the user on 2026-10-26: "so far we've been running these experiment on one model
family, let's run it on at least 2 more model families". Status: DRAFT, waiting on the Executor's feasibility note
(stage 0) before rev 1.

## Question

§38–§42 are one family (Qwen2.5-3B-Instruct, judge Qwen2.5-7B). The claims they make about a small reader are:
(i) its refusals with the evidence in view are a late, content-free disposition that one direction opens (§38, §40);
(ii) the answer that comes out is read from the evidence (gold with the evidence − gold with the evidence cut);
(iii) opening costs some already-correct answers and answers on unanswerable questions (§40, §42). §43 asks whether
(i)–(iii) hold in two more families at the same size class, with the Qwen pipeline, prompt and judge unchanged.
A free lens readout (the §34 emergence depth, logit lens) rides along for the user's advisor thread.

## Candidate families (from §24, LongMemEval `fixed8`, the only cross-family numbers in the record)

| family | small reader | unknown share (FIFO → fixed8) | P(correct given answered) | notes |
|---|---|---|---|---|
| Llama 3.2 | Llama-3.2-3B-Instruct | 0.430 → 0.249 | 0.466 → 0.569 | refuses most; gated on HF (403 in §24, re-check) |
| Gemma 3 | gemma-3-4b-it | 0.185 → 0.087 | 0.439 → 0.552 | few refusals; no system role in its template; bf16 only |
| Phi-4 | Phi-4-mini-instruct | 0.187 → 0.083 | 0.312 → 0.420 | few refusals, degenerate outputs (loops) in §24 |

Recommendation: **Llama 3.2 3B first** (the closest analogue: a refuser), **Gemma 3 4B second** (the family where the
controller bridged a size step in §24); Phi-4-mini only if one of those is infeasible. A family that refuses on fewer
than about 15% of marked dev rows with the evidence in view has too few rows to steer, and that is itself reported
(it says the Qwen disposition is not universal).

## Stages

**Stage 0, feasibility (Executor, laptop, free, no design).** For each candidate: HF access with the `.env` token
(report yes/no, never the token); load in fp32 on the laptop (RAM, seconds per token at the §38 prompt length;
the laptop has 23 GB RAM and a 4 GB GPU, so 4B fp32 is the ceiling); layer count; chat template fit for the §38
prompt (Gemma has no system role: fold the system text into the user turn and say so); the token(s) the model
emits for "unknown" after `Answer:` and whether the `Answer:` marker survives (marker rate); refusal rate and
judged-correct rate on folds 0–1 of LongMemEval with the evidence in view, D1 string gold provisional. Nothing on
folds 2–4; nothing on LoCoMo.

**Stage 1, design on development data (laptop, free; §38 amendment 2 recipe per family).** Direction v_L = mean
resid(B) − resid(A) at the ':' of `Answer:` over the dev pairs (A = the model's own refusal, B = the answer under the
§35 line), at each candidate layer. Candidate layers: relative depths {0.6, 0.7, 0.8, 0.9} of the family's depth
(Qwen's 29 of 36 is 0.81); α in norm units {1, 2, 4}. Design gate on folds 0–1 only, read on the dev R rows:
Steer answers on ≥ 0.5 of R, gold(Steer) − gold(Steer-cut) ≥ 0.15, Random-direction answers ≤ 0.2 × Steer's,
no word salad (degenerate ≤ 0.1). One setting per family is frozen by commit before stage 2. Gate FAIL closes the
family as "no content-free refusal direction found at this budget", a result, not a retry.

**Stage 1b, lens readout (laptop, free, descriptive).** Logit-lens emergence depth of the gold first token on the dev
C-class rows for the small model of each family (§34 recipe, logit lens only; the J-lens fit is not repeated),
reported beside Qwen 3B's median 0.889 and 7B's 0.464. No claim; it feeds the advisor discussion.

**Stage 2, confirmation (one pod per family, quoted separately, user's line in the Executor pane).** The §40/§42
pipeline on fresh LoCoMo rows (a new seed, §40's and §42's ids excluded), setting fixed from stage 1, judge
Qwen2.5-7B bf16 as before, R ≤ 120, C ≤ 40, 40 adversarial, max new tokens 64, D-22 smoke (batched-vs-single
agreement) and D-30 bundle test as §40.
- Claim 1 (per family): gold(Steer-all) − gold(Steer-all-cut) on R, paired bootstrap LB > 0. Descriptive beside:
  gold(Steer-all) against the 0.30 line; Steer-all − B.
- Control X: ans(Random) ≤ 0.2 × ans(Steer-all).
- Control C: gold(Steer-all) − 1 on C; harmful if UB < −0.10 (state ties as ties).
- Control U: adversarial answer rate, A / B / Steer-all; a cost.
- Cross-family reading (descriptive, no threshold): the number of families (of Qwen + the two) in which Claim 1 is
  SHOWN and Control X passes.
- Not run: §41/§42's selective policy (closed negative on Qwen; no reason to expect it elsewhere); §39/§39b (free
  dev-only; may be added to stage 1 at no cost if the Executor has time).

## Readings fixed in advance

- Both new families SHOWN: the content-free refusal direction is a property of small instruction-tuned readers, not
  of Qwen; the thesis sentence "the policy half of the small-reader gap is a late disposition" generalises.
- One SHOWN, one FAIL or NOT SHOWN: family-dependent; report which, and whether the failing family had refusals to
  open at all (stage 0 rate).
- Neither: Qwen-specific; the §38–§42 chain stays a single-family result and the thesis says so.

My prediction: Llama 3B passes the gate and shows Claim 1; Gemma 4B has too few refusals on dev to pass the gate
(≈ 9% unknown in §24) and closes at stage 0 or 1 as "little to open".

## Budget

Stages 0, 1 and 1b: $0, laptop time (a 3–4B fp32 dev run took about an hour on the laptop for Qwen; allow 2–3 h
per family). Stage 2: one A40 secure pod per family at the then-current price (last $0.59/h), hard stop 60 min,
at most about $0.59 each, about $0.15–0.30 expected each (a 4B model is slower than Qwen 3B). Programme spend is
about $17.94 of the $20 cap: two stage-2 pods fit only if both run clean (≤ $18.53 expected, ≤ $19.12 at the hard
stops). A third family, a failed attempt, or a 7B-partner lens run would need the cap raised; the user decides.

## Amendment 1 (2026-10-26, same day): Part B, the lens readout, and the raised cap

The user, after being told that §34 was measured on Qwen only: "Let's expand then, do those experiments for more
families. I've extended the budget limit from 20 to 30 too." Part A above (the refusal direction) is unchanged.
The aim is now **three new families if stage 0 allows** (the user's "at least 2"): Llama 3.2 / 3.1, Gemma 3, Phi-4,
each at its §24 pair of sizes.

### Part B: where does the small reader lose the answer, per family (§34 recipe)

**Population.** §24's cached `fixed8` prompts and outputs for both sizes of the family (`cache/generations_exp24_*`,
`runs/exp24_{family}_f{0..4}`, verdicts in `runs/exp24_verdicts.json`): answerable questions with all evidence in
view, present for both sizes with a judge label, paired by prompt hash, as §34. Development folds 0–1 for the gate
and any choice; confirmation folds 2–4 read once, after the choices are committed.

**Instrument.** The logit lens `unembed(h_l)` at P1 (the token before the gold's first content token after
`Answer:`, whitespace teacher-forced, `Answer:` appended directly as in §34) **decides** the claims; it is
pre-registered as the deciding row because the Qwen 3B's J-lens failed its gate and the logit rows decided §34, and
because a J-lens fit costs about 90 minutes of pod time per model. A J-lens with 56 WikiText paragraphs is fitted
for each model **only as time within the pod's hard stop allows**, small model first, and shown beside; it takes a
verdict role only if it passes the §34 gate (a) and (b) on folds 0–1. k = 10 for A / B / C.

**Claims per family (confirmation folds 2–4; same tests as §34a).**
- Claim B1 (size): log10 best rank of the gold at P1, small minus large, paired by question, lower bound > 0.
  Secondary: share of outcome A, small minus large, lower bound > 0.
- Claim B2 (within the small model): AUC of the gold's P1 rank for wrong against right, lower bound > 0.5.
- Descriptive, no threshold: the A / B / C decomposition of the small model's in-view failures, split refusal /
  answered wrong; the median emergence depth (first layer at which the gold enters the top-10, as a fraction of
  depth) for both sizes, beside Qwen's 0.889 (3B) and 0.464 (7B).
- Cross-family reading (descriptive): in how many of the four families (Qwen + new) the small model's failures are
  mostly C (share > 0.5) with an A floor, and in how many the small model's emergence depth is later than the
  large model's.

**Readings fixed in advance.** C-majority with later emergence in every family: the "answer present, decision late"
picture is a property of small readers under this prompt, not of Qwen. Mixed: family-dependent, report which, and
check whether the families that differ are the ones that refuse least (§24's unknown shares). Qwen alone: a Qwen
fact; the advisor briefing's §34 paragraph is reworded to say so.

My prediction: Llama 3B reproduces the Qwen picture (C-majority, late emergence, refusals); Gemma 4B and Phi-4-mini
show C-majority too but with wrong answers rather than refusals, and emergence is still late relative to their
larger partners.

### Budget under the $30 cap

Spend to date about $17.94. Per family: Part A pod ≤ $0.59 (about $0.15–0.30 expected); Part B pod with a 3 h hard
stop ≤ $1.77 (reads for both sizes take minutes; the J-lens fits take the rest and are cut at the stop; about
$1.00–1.50 expected). Three families: ≤ $7.08 at the stops, about $3.50–5.50 expected; spend ≤ $25.02 at the stops.
Every pod is quoted separately and needs the user's own line in the Executor pane. Stage 0 and stage 1 stay free.
