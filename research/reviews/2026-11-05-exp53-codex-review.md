# Codex review: §53 rev 0 — head controls

2026-11-05. Reviewed `research/proposals/2026-11-05-exp53-head-controls.md` at 6bca4d8 against the research-gap audit and §51, including its binding §13 amendments. Review only: no experiments, code changes, proposal edits, spend or commits. This file is the requested review artifact.

**Recommendation:** revise before building. Layer-matched head ablations are useful controls, but the proposed rank is not automatically a calibrated significance test. Arm B can describe attention and causal influence; its current rules cannot identify option copying versus evidence weighing. Arm C needs a stronger baseline check and narrower conclusions. These are specification problems, not reasons to abandon the follow-up.

## 1. BLOCKING — State the exact opening event and preserve the tested dose

**Location:** opening rule; A-target/A-amp.

§51 has different H outcomes for repair, comparative trade-off, pathway location and account support. “Confirms submission 1 under §51's own confirmatory rule” does not say which outcome opens §53. Define an explicit Boolean rule using named §51 states, without modifying those states. A repair confirmation and a mechanism-path confirmation support different introductory claims; neither entails the other.

Also pin the actual evaluated effective edit, including its dose. If §51 selects a scaled dose, a byte-identical kernel is not necessarily the same effective edit as completely zeroing the columns. State whether the opening event must concern the exact full-ablation candidate, or whether §53 studies a separately declared full-ablation dose. Pin checkpoint/tokenizer, zero-based layer/head indexing, head-to-column slices and the amplification manifest. The ×2 diagnostic is a separate perturbation, not an algebraic inverse of zeroing.

## 2. BLOCKING — The 1/21 rank is a descriptive benchmark unless an exchangeability null is justified

**Location:** C1/C2.

Twenty unique, uniformly sampled layer-matched pairs are a reasonable economical comparison for “how exceptional is this selected pair among these alternatives?” They do not by themselves produce an exact randomization test. The target was selected through an editor search on E; it is not a uniform random draw from the population supplying the controls. Fresh N prevents selection on N outcomes, but does not make head identities exchangeable under the broad null “general disruption.” Heads may have stable, heterogeneous effects that carry from E to N without the proposed mechanism being correct.

Under a separately justified exchangeability model, the conservative Monte Carlo score would be `(1 + number of control scores >= target score) / 21`; ties count against the target. Without that model, call this an **empirical rank among 21 evaluated pairs**, not a calibrated p-value. The cheapest defensible revision is a descriptive specificity benchmark, with paired content-group uncertainty for the target-minus-control-summary contrast. A confirmatory population contrast against a specified random-head distribution would require its own estimand and sampling/inference justification; a plus-one formula alone does not provide that.

Freeze whether pairs are sampled without replacement, the admissible head set, and whether excluding “the targets” excludes both individual heads or just the target pair. Excluding both individual heads is sensible for a disjoint-pair comparison but cannot establish that no pair sharing one target works.

The 0.30 threshold is a **30-percentage-point operational margin**, not a significance threshold or a literature-established effect size. It is acceptable if justified as the minimum difference of interest. Report its estimate and uncertainty; distinguish a point estimate exceeding 0.30 from evidence that the population difference exceeds 0.30. Do not tune it after N results.

If C1 and C2 are only used as a conjunction, an intersection–union formulation does not need a two-test correction internally, assuming valid elementary tests. If each is separately advertised as a confirmatory discovery, declare multiplicity handling. With only 20 random controls, a rank-based minimum of 1/21 cannot pass a two-hypothesis Holm first threshold of 0.025. Increasing the control count alone would not fix the exchangeability problem.

## 3. NON-BLOCKING — Layer matching is useful, but “specific” needs a narrow definition

**Location:** Arm A; specificity reading; stated confounds.

Same-layer pairs match head count and position, but not weight norm, activation norm, output-logit alignment or functional perturbation size. A target pair may simply exert a larger generic answer-suppression effect. §51's B6 controls random directions on the same entries; it does not remove this between-head confound. Report per-head output/perturbation norms on the fixed population, preferably also on N-dev, as diagnostics. Do not select norm-matched controls after inspecting held effects.

C2 is well posed as **an operational behavioral contrast** if the amplification dose is fixed, the denominator is all sampled B0-correct rows, and the endpoint is the frozen grader's abstention class. Also report wrong, mixed, degenerate and missing-marker outcomes: amplification causing general damage is not the same as selectively creating abstention. The full damage rate on C should accompany A-target and A-single too. Evaluating ablated A-rand on C would strengthen a repair-selectivity comparison, but is optional if claims remain about release rank rather than selective repair.

The two-single-head arms help assess whether one head dominates. They do not establish interaction unless a contrast and endpoint are specified. “Outperformed the sampled disjoint layer-matched pairs at these doses” is supportable; “the unique abstention circuit,” “necessary,” and “specific to evidence weighing” are not. Joint amplification changes the downstream input to L33 as well as its gain, so it does not independently identify both heads' roles.

## 4. BLOCKING — A new seed is not a disjoint split; freeze N's content and sample construction

**Location:** Data and Rules.

Use §51 §13.2's effective-key collision closure for N, both against prior editor/analyst data and between N-dev/N-held. A different RNG seed can reproduce content from finite name/city pools. Keep all linked ANS/CF variants and twins in one content group. Old prompt hashes, key collisions and partial name/city overlap need an explicit audit. If fresh content groups cannot be supplied, disclose the overlap and narrow the claim; do not silently call seed 54 independent content.

Freeze the generator size or a bounded generation rule, task quotas, UNA-rel/UNA-swap allocation, sampling with/without replacement, and a shortfall rule. “150 per class, stratified by task” is not enough: it could mean equal task allocation or proportional allocation. State how many independent bases/content groups remain, not just how many rows. Related rows cannot supply independent uncertainty. If pairs for Arm B require rows outside the 450, specify their eligible pool and additional workload in advance.

There is a sequencing contradiction: sample IDs and generated-note spans depend on N-held B0 outputs, but the manifest must precede **any** N-held output. Use two freezes:

1. Before any N-held inference: dataset, split/overlap rules, automatic B0 classification, quota/shortfall rules, pair-selection algorithm, metrics and all arms.
2. After a scripted B0 pass: realized sample IDs, denominators and spans, frozen before edited outputs.

No analyst choices may depend on those B0 outcomes. This is valid baseline-conditioned sampling if specified as such; it is not an untouched population sample. N-dev stays limited to technical checks/timing unless a new permission for method development is declared prospectively.

Finally, Rules says LongMemEval folds 2–4 “are read once,” contradicting their sealed status in Data. Replace that clause with **never accessed by §53**.

## 5. BLOCKING — Separate controlled-prefix and total-effect estimands throughout Arm A

**Location:** Arm A mode; C1–C3; sample definitions.

Specify, for every arm and population, whether it receives controlled-prefix generation, total-effect generation, or both. Under §51 the controlled prefix is B0's stored token IDs through P_ans and the prefix KV is recomputed under the edited weights. It is not a head-output patch only at P_ans. Record the missing-marker exclusion and sample shortfall rule, and do not obtain P_ans by decoding/re-tokenizing.

C1/C2 can test the controlled-prefix response. If C3 measures unsupported answers under total-effect generation, report it as a **different estimand**. It cannot be combined with controlled-prefix release into a trade-off or selective-repair claim. Likewise, a 150/150/150 class-conditioned sample does not reproduce §51's base-weighted utility or natural class prevalence. Keep fixed class denominators and report FA-ab versus other FA descriptively, because the frozen note may already contain the gold.

## 6. BLOCKING — Arm B's reading rules are both underspecified and non-identifying

**Location:** fixed predictions; option-removed analysis; audit's corresponding mechanism claim.

“Higher,” “separates more,” and “little” have no frozen metrics, tolerances or uncertainty rules. Choosing among two heads and three evidence/note groups leaves several implicit maxima. Both proposed readings can hold simultaneously; neither can be established from these diagnostics alone. An evidence-sensitive head may copy the option after integrating context, and a generic output-suppression head can attend to evidence tokens. Whole-head patching demonstrates influence through that output, not which attended source supplied the relevant computation.

Cheapest revision: retain numerical diagnostics, remove the forced **option-copying/evidence-weighing** verdict, and state “consistent with this prediction; alternatives unresolved.” Include insufficient support/ambiguous outcomes explicitly if retaining any prediction table. A deterministic threshold table would prevent post-hoc naming, but would not solve identifiability.

The B3 survivor analysis has a further problem. Original-prompt FA that **still abstain** after option removal are a selected, potentially harder subgroup; a small edit effect there is not evidence that the original head copied the option. If B3 leaves no abstentions, the statistic is undefined, not zero.

An economical improvement is a predeclared 2×2 comparison on the **same original-B0 FA population**: original versus option-removed prompt, each with no edit versus target edit. Report all four rates and the edit-by-prompt interaction. Role-aligned controlled prefixes must retain the same stored note content where feasible; total-effect results answer a different question. This tests prompt dependence, still not literal copying, because removal changes both instructions and available tokens. To identify a source more directly, add a separately specified source-restricted value/output intervention or matched semantic-versus-lexical manipulation. That is an optional stronger study, not something to improvise after N.

## 7. BLOCKING — Make token groups and the DLA calculation executable definitions

**Location:** B1/B2 measurements.

Define disjoint primary token groups. At present g is nested in n; report g as a declared subcategory, or make n mean note-minus-g. Mark evidence-bearing relation/city tokens separately from distractor memory if claiming evidence weighing: mass on the whole memory does not identify relevant evidence. For UNA, there is no correct answer city; “gold-city tokens” and a gold logit must be NA or explicitly labelled as the withheld/base reference city, never treated as available supporting evidence.

The rules can be frozen before generation, but the model's note spans cannot literally be known then. Freeze token-ID span extraction, repeated mentions, missing note/marker handling, answer-marker assignment, instruction-option tokenization and multiple occurrences. Report group token counts and attention mass per token alongside total mass, since groups differ greatly in length. Fix task/hardness weighting; otherwise FA/C differences can reflect their task or note-length composition. Note that P_ans attention observes a model already conditioned on its own note.

For a residual-space head contribution `c[l,h] = W_O[:, head_slice] z[l,h]`, define the frozen-RMSNorm attribution explicitly, for example

`DLA[l,h] = (u_unknown - u_gold)^T diag(gamma_final) c[l,h] / sqrt(mean(r_final_B0^2) + eps)`.

Here `r_final_B0` is the clean pre-final-norm residual at the same P_ans, and token IDs are the actual first continuation tokens under the fixed prefix. Pin any embedding/logit conventions. If the capture is pre-o_proj, first apply the correct head slice; do not project a head-dimension vector directly as though it were residual space. This is a frozen-normalization **direct readout**, not the derivative of RMSNorm and not the total causal effect after downstream computation. Report the unknown and gold contributions separately: their difference can rise by suppressing the city without copying unknown. First-token ambiguity/shared city prefixes should be counted and explicitly bounded in the interpretation.

## 8. BLOCKING — Freeze donor matching, hooks and controls for the causal patch

**Location:** B4; patching as Arm B's “causal check.”

“Same base, one variant answered, one abstained” leaves unresolved whether the donor must be correct, share the same gold, and have the same evidence relations and answerability. Exclude ANS↔UNA and changed-gold CF pairs from this particular interpretation. Hardness, distractors and generated-note differences remain possible confounds even within a base. Freeze eligibility, matching/tie rules, pair direction, maximum reuse and treatment of multiple pairs per content group. If no pairs qualify, report no support; do not search elsewhere.

Specify whether each head is patched alone, jointly, or both; capture the post-o_proj head contribution before residual addition; replace only that contribution at the recipient's P_ans. Keep the recipient prefix/cache and all other components recipient-derived. For a joint patch, say that both values come from an unpatched donor run; the L33 replacement then overwrites its natural response to the L32 intervention. Pin whether later generated positions are left unpatched. Include self/no-op replay and, if calling the result mechanism-discriminating, a donor control matched for answer status or a source-specific intervention.

Predeclare generated endpoints in both directions, and report continuous margins only as secondary. Even a successful patch supports “this head output causally influences the answer under this intervention,” not “this head weighs evidence rather than copies an option.” Activation-patching results depend materially on corruption, matching and metric choices [2].

## 9. BLOCKING — Arm C's baseline, grader and interpretation do not support its current conclusion

**Location:** Arm C.

A first-token reproduction check on five rows with one mismatch allowed is inadequate for a comparison scored on full answers, especially when the historical cache and new run may differ in precision/backend/prompt. Prefer rerunning B0 and the edit under the same pinned configuration on the 126 dev rows, with full-output or graded-endpoint no-op replay checks. If cached outputs are retained, freeze a full-endpoint replay gate and a mismatch fallback before looking at edit outcomes. A mismatch must not be counted as an edit effect.

Gold-string containment can credit quotations, incorrect relations, negations and “I don't know whether [gold]” responses. Label it an exact containment metric, not answer correctness. Freeze an answer-assertion check or blinded manual adjudication of changed outputs before making claims about correct release or low damage. A later 7B judge would be a separate exploratory assessment, not a way to choose the favorable verdict. Unanswerable questions require a grading rule that does not rely on gold-string containment. Also pin how “in view” is established; mere presence of an answer string need not supply sufficient evidence.

These are previously inspected dev folds, so the result is descriptive transfer to this population, not independent external confirmation. A positive result supports an effect on those natural-question rows. A weak or null result means **transfer was not demonstrated under this prompt/context configuration**; it does not establish a template-specific mechanism. Low support, retrieval insufficiency and a different failure cause remain alternatives. Define “low damage” or report the rate with uncertainty without a binary adjective.

## 10. NON-BLOCKING — The compute estimate is a useful subtotal; one cache-sharing assertion is false

**Location:** Compute.

The listed Arm A cells do total 13,500 generations: 450 target + 900 singles + 6,000 layer-matched ablations + 3,000 late ablations + 3,150 amplifications. At 9 seconds each the arithmetic is 33.75 hours. This excludes N's B0 screening, extra total-effect generations, attention captures, B3, patching, replay and LongMemEval. If every listed ablation on UNA also gets total-effect generation, that alone adds 4,950 generations at a different per-row cost. Inventory those separately before quoting.

“Layers below 32 are unchanged by every arm A edit” is false for A-late pairs in L30/L31. Reuse is bounded by each arm's earliest modified layer. It is also token-history-specific: teacher-forced shared-prefix activations can be reused where exact, but autoregressive branches cannot share activations for divergent generated token histories. Keep the N-dev parity check and pin which outputs/caches it checks. Dropping A-late first is a cheaper design if the primary question is specifically the L32/L33 matched comparison; make that choice before held inference, not after seeing random controls.

## 11. NON-BLOCKING — Narrow the audit's claims before reusing them in the thesis or technical report

**Location:** companion research-gap audit, not a request to modify §51.

The audit explicitly says the editor's E observations are not §51 results; keep that distinction. Several other sentences exceed the available evidence:

- “No push tried is selective” drops the documented template-limited gated result. Use the narrower §51 wording about the evaluated ungated pushes and report the gated exception.
- “The answer is present inside the model” needs a readout-qualified definition, particularly given the evidence-cut survival result. Neither a probe read nor a generated gold mention alone establishes evidence-derived knowledge.
- “Read it and won't say it” is a hypothesis, not established comprehension or intentional suppression.
- A fixed weight edit can avoid an added inference operation, but that alone does not make it better than prompts/steering; efficacy and costs still require comparison.
- A successful agent-guided edit in one setting does not establish method transfer to other failures.
- “Heads attend mainly to unknown, therefore copying” and “analogous heads elsewhere, therefore general” are not warranted mechanistic conclusions.
- The claim that prior work lacks component-level localization needs the promised fresh literature audit, including the already-discussed Commit-Abstain Circuit. This review does not certify novelty.

## 12. NON-BLOCKING — A smaller coherent follow-up is available

My suggested minimum, to be agreed and frozen before any N inference:

1. An explicit §51 opening rule; new content-disjoint N with a two-stage B0/sample freeze.
2. Target, single heads, 20 unique disjoint layer-matched pairs, and their fixed-dose amplification comparison; report empirical ranks and paired content-group uncertainty with the stated margins. Keep amplification and release as distinct endpoints.
3. Attention and correctly defined DLA as descriptive diagnostics; exact head-output patching with self-replay, and no categorical option-copying/evidence-weighing verdict.
4. If prompt dependence is important, the small same-population 2×2 B3 comparison instead of relying on B3 survivors.
5. Optional LongMemEval dev transfer with a paired fresh baseline and assertion-aware grading; otherwise defer it. A-late is also optional.

This tests whether the pair is unusually effective among specified alternatives, whether its output influences answering, and how strongly the effect depends on the instruction. It does not by itself identify an evidence-weighing mechanism. That narrower contribution would still be useful and reportable.

## Sources checked for methodological boundaries

1. Phipson and Smyth, *Permutation P-values Should Never Be Zero*: [primary manuscript](https://arxiv.org/abs/1603.05766). The plus-one correction concerns valid random-permutation procedures; it does not create an exchangeability argument for a selected head pair.
2. Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*: [primary manuscript](https://arxiv.org/abs/2309.16042). Supports specifying patching/corruption and metrics rather than treating all patch designs as interchangeable.
3. Elhage et al., *A Mathematical Framework for Transformer Circuits*: [primary report](https://transformer-circuits.pub/2021/framework/index.html). Background for distinguishing head routing, value/output content and residual/logit contributions; the Qwen frozen-RMSNorm formula above is an explicit model-specific specification, not a claim that this report studied Qwen.

No source here establishes novelty of §53. No files in `writing/` were accessed. Only this requested review file was created.
