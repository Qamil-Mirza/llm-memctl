# Advisory review: the proposed J-lens mapping study (Codex handoff, 2026-10-30)

**Status: review only. No proposal, no build, no spend. The user decides whether to amend D-45; §44 closes first.**

## 1. What the handoff asks

Return to the J-lens and produce inspectable layer-by-token maps (prompt and generated response) on a small controlled
set of direct-extraction, relational-composition and temporal-updating tasks, on Qwen 3B with the existing fitted lens,
matched substitutions / question changes / order changes, solved cases first, then matched failures, all cases kept.
Only after readable intermediates exist on solved cases, formulate one targeted causal test of a specific intermediate.

## 2. Fact check against the record

| handoff statement | record | verdict |
|---|---|---|
| Qwen 3B failed the aggregate lens gate but passed the gold-rank component | §34 gate (a) PASS +1.219 (+0.907, +1.531); gate (b) FAIL (library eval sets pass@10 0.014 vs 0.039) | correct |
| Llama 3B and Phi-4-mini passed their gates | §43 Part B: Llama PASS, Phi PASS, Gemma FAIL | correct |
| original Qwen fits used 56 prompts | §34: 7B time box closed at 56; 3B fitted on the same 56 | correct |
| exploration restricted to forced "Answer:" and a middle-third band | §34 read at the forced P1 (own position in §34b on marked outputs); gate band = middle third | correct |
| the arithmetic evaluation omitted synonym expansion | the lens library's evaluation sets were run as shipped; not checked by this pane | unverified, plausible |
| "we have not established that smaller models lack abstraction capacity" | nothing in STATE claims that; F11 says the small readers hold the gold late, not that they lack it | correct |

## 3. What is sound in the direction

1. **Three distinct things** — answer accuracy, lens recovery of an intermediate, intervention success — are kept
   apart. §44 is the lesson: single-position patching measured where an answer is *usable*, and the swap faults were
   readout faults (type-incompatible substitutes, digit tokens, determiners), not model facts.
2. **Patching should test an identified computation.** §44 rev 0 patched a position, not a computation. A map
   first, then one causal test of one transition, is the right order and is how the J-space paper proceeds.
3. **Expected intermediates are hypotheses.** Our lens caveats already on record (article-first tokens, digit-by-digit
   numbers in Qwen's tokenizer, depth carrying token frequency, 56-prompt fits) are exactly the checks the handoff
   asks for before reading an absent signal as absent reasoning.
4. **Keep every case,** including ambiguous and negative ones. This is the no-selection rule in exploratory form.

## 4. Concerns to settle before anything is built

1. **D-45.** The user's own rule: no new probes, scaling or routing; if §44 does not establish the distinction,
   document it. A descriptive mapping study is not an intervention, but it is a new line. It needs the user's explicit
   amendment (recorded as a decision), not a reading of the rule by the Advisor or by Codex.
2. **Bridge to the thesis question.** Three-sentence toy prompts are the right place to learn what the lens shows when
   the computation is known, but the 3B will solve nearly all of them; the thesis failure is "evidence in view, long
   context, wrong or abstain". The bridge already exists: the 70 §44 rev 8 rows have clean and swap prompts, §23 labels,
   per-layer transfer curves and the forced-position reads. Stage 3 of any mapping study should be those rows, so the
   maps meet the failures the thesis is about.
3. **Fit stability.** 56 prompts is thin. A refit on more prompts is a pod (spend: ask first). Within $0 the honest
   move is the existing lens beside the logit lens on every map, with disagreement reported, as the handoff says.
4. **A readability criterion fixed in advance.** "Readable intermediate" must be defined before the maps are made
   (for example: the intermediate entity's first content token in the lens top-k at some layer of a fixed band at the
   position where it is needed, on ≥ a stated share of solved cases), otherwise the deliverable is pictures chosen
   after the fact. Decide it on a dev half of the case list.
5. **Tokenization.** Qwen splits numbers into digits ("132" → "1", "3", "2") and names often start with " The";
   pinned trajectories must track the first *content* token and say so, or the maps repeat the §44 swap fault.
6. **Generated tokens.** Maps over the model's own response extend §34b (own position, marked outputs). Unmarked and
   refusing outputs need a stated position rule.

## 5. Recommendation

- **Now:** nothing. §44 closes within hours (7B GPU confirmation, rev 8 S); record changes are pre-approved.
- **If the user amends D-45:** the Advisor writes a §45 proposal, exploratory and descriptive: a fixed case list
  (about 30 base cases × 3 task types × matched variants) written before any run; stage 1 solved cases, stage 2 matched
  failures, stage 3 the §44 rows; deliverable = reproducible maps with top J-lens readouts, pinned trajectories, the
  outcome, and the logit lens beside; the readability criterion fixed on the dev half; **no causal test in the
  proposal** until one intermediate or transition is named from the maps; laptop only, Qwen 3B fp32, $0; the Executor
  builds after §44's text is final. No claim about erased thoughts; "descriptive maps" in the title.
- **If the user keeps D-45 as is:** §44 is documented as the result and the programme pauses for the thesis write-up.
