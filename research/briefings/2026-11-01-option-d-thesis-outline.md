# Option D: thesis outline — replications, boundaries and evidence-grounded controls (2026-11-01)

Advisor draft, written at the user's request ("let's write out option D"). The basis is `/tmp/thesis-contribution-audit.md`, read together with its Addendum 1, whose corrections take precedence.

This is a **write-up structure that stands without any new experiment**. It is not a guarantee that the programme or the faculty advisor will accept it. `writing/honors-proposal/` is untouched; the user owns the thesis text.

Codex's A′ review (`/tmp/thesis-a-prime-codex-review.md`) also applies. Every chapter keeps three kinds of material apart: (1) completed exploratory evidence, (2) replicated findings, (3) proposed tests. First-in-setting claims are replaced by testable contrasts. This structure is a candidate for the faculty advisor's judgement, not a guaranteed floor.

Wording rules applied throughout:
- A replicated observation is a **finding**, labelled as a replication.
- Only unsupported novelty claims and unsupported mechanistic claims are withdrawn.
- Every result names its population.
- Every "no difference" stays a no-difference, never an equivalence.

**Working title.** *When the evidence is in view: replications, boundaries and evidence-grounded controls for "the model has it but doesn't say it" in small instruction-tuned models*

**One-sentence thesis.** In small instruction-tuned readers given sufficient evidence, interventions that encourage answering can release correct answers in an evidence-dependent way, and only at measured costs. In this programme, positive graded recovery is established chiefly for Qwen2.5-3B (Llama 3B: no release beyond its note, 13 rows). Common internal readouts of "the answer is there" need evidence-grounded controls before they can be read as evidence use. Our instruments reproduce known profiles and known limits in this setting. Where the evidence influences the failing response is still open.

---

## Chapter 1. Motivation and scope

- **Original aim.** Improve answers from a model's memory without simply scaling the model, data or memory.
- **Earlier programme (memctl).** The frozen reader is the ceiling. A 3B reader with perfect evidence still trails a 7B (F6). The learned controller is no better than a 31-parameter logistic regression (F3, §32).
- **So the question became:** why does a small reader fail when the evidence is already in its prompt?
- **Small models are the study setting, not the motivation.**
- **Scope.**
  - Qwen2.5-3B-Instruct is the primary model.
  - Llama 3.2 3B, Gemma 3 4B and Phi-4-mini are the cross-family readers; Qwen 7B and the other larger siblings are descriptive.
  - The benchmarks are LongMemEval and LoCoMo, plus controlled tasks.
- **What the thesis does not claim:**
  - a mechanism for the failure;
  - a location for the failure;
  - first-of-its-kind status for small models (Qwen2.5-3B-Instruct already appears in Gur-Arieh et al. 2025, Ruscio & Thompson 2026, MERA, GCD, and an appendix of Liu et al. 2026).

## Chapter 2. Related work (the map in audit Part 1 and A1)

| Topic | Papers |
|---|---|
| Answering vs abstention mechanisms | Commit-Abstain Circuit (Nguyen et al. 2026); Kumaran et al. 2026 (Nature Machine Intelligence); Lavi, Milo & Geva 2026; Liu et al. 2026; Ling et al. 2026; GCD; Hu / Du & Hu; Wagner 2026 |
| Latent answers | Orgad 2025; Gekhman 2025; ITI; CoRect; Walters et al. 2026; Ruscio & Thompson 2026; *Legible Failures* (Ravulapalli et al. 2026) |
| Binding and routing | Feng & Steinhardt 2024; Gur-Arieh et al. 2025; Lookbacks 2025; Geva 2023; ROME; retrieval heads; Jin 2024; Ortu 2024; ReDeEP |
| Instrument limits | Tuned lens; Patchscopes; the J-lens paper's own caveats (§3.5.1, A.12, §9.1); Hase 2023; Makelov; Zhang & Nanda; Heimersheim & Nanda; Hewitt & Liang |
| Steering and selective steering | CAST; MERA; FASB; Tan 2024; Im & Li; SafeRAG-Steering; SpARE; Xin 2025 |
| Verification | Song 2025; Huang 2024 |

**Positioning paragraph.** Our setting puts the evidence in the prompt, which lets us *manipulate the source of the answer*: cut the evidence, substitute it, or grade a released answer against an evidence-cut run. Most of the closest work is closed-book (Kumaran, Ruscio, Gekhman), ungraded (CAC, Lavi, CAST), or synthetic without abstention (*Legible Failures*). That manipulation is our only distinguishing axis. A change of model size or benchmark alone is not one.

## Chapter 3. Methods and rigour

- Pre-registration before every read; dev/confirm folds; verdicts computed, not typed; independent recounts; about $22 of compute.
- State honestly that the LongMemEval confirmation folds were read repeatedly under the honesty clause, so later LongMemEval claims are re-analyses.
- Instruments:
  - logit lens and J-lens (stable under refit, §45);
  - clean→cut and swap patching (§44);
  - interchange at entity and evidence tokens (§46, §47);
  - difference-of-means steering (§38–§42);
  - the judge (state its validation scope: STATE caveat 3, §18 and §32 checks on the earlier arms only).
- Rigour is presented as method, not as a headline contribution.

## Chapter 4. RQ1 — interventions that encourage answering, with the evidence held fixed

**Findings (replication plus evidence-grounded accounting):**
- **Prompt policy (§35).** A frozen note plus "do not reply unknown" gives +0.176 (+0.109, +0.242), about half the matched 3B–7B gap. Costs: 0.210 of refusals turn wrong, and the answered-and-wrong share without evidence doubles under forced choice. This qualifies Ling et al.: under the no-"unknown" control the in-view refusal persists (gold 0.228, §44 U1).
- **Steering (§38, §40, §42).**
  - LoCoMo, setting fixed in advance (§40): gold 0.358 with the evidence against 0.025 with it cut.
  - Costs: 8 of 40 correct answers lost; the answer rate on unanswerable questions rises from 0.32 to 0.73; the §35 line scores better on utility (+4 against −1 of 120).
  - Replicated smaller in §42 (+0.233).
  - In §38 the setting was chosen on dev labels after the primary setting failed (disclosed).
  - Prior work: Kumaran et al. (closed-book, about half of newly answered items right, by our arithmetic); Lavi, CAC and Xin (ungraded).
- **Release at one position (§39, §39b; dev only).** Replicates Liu et al. (EMNLP 2026). It describes how the push works and does not locate the failure.
- **Selective steering fails (§41, §42).** The projection ranks extractability, not answerability. This replicates the diagnosis of Wagner 2026, GCD and Hu. MERA-style or two-score gates were not tried.
- **Families (§43).** In-view refusal rates are Qwen 0.30, Llama 0.10, Gemma 0.02 and Phi 0.03. The Llama direction releases nothing beyond its own note (13 rows), so that contrast is motivated, not established. CAC covers false abstention in Llama-3.2-3B, Gemma-3-4B and Phi-4-mini at the decision level.

**Candidate contribution (not established as novel):** grading released free-form answers against an evidence-cut run of the same intervention, with costs on correct and unanswerable items.

## Chapter 5. RQ2 — where does supplied evidence influence the answer, and what do the lenses show there?

**Findings (replications and exploratory observations):**
- **Late readability (§34, §43, §45; replication).**
  - Four-family LongMemEval statement (§34, §43): when the reader fails in view, the gold's first token is in the final-layer top-10 on most failures: Qwen 0.64, Llama 0.68, Gemma 0.76, Phi 0.80. The layer counts on controlled tasks (36 layers, §45–§47) are Qwen 3B only and must not be merged with this statement.
  - Gold rank separates wrong from right in every family.
  - On controlled tasks the gold is readable only at layers 30–33.
  - Prior work: Orgad, Gekhman, CoRect, Walters, Ruscio, Lad.
- **Evidence-cut diagnostic (§34c).**
  - Among the 81 confirmation failures with the gold in the final top-10, 0.605 keep it in the top-10 after the evidence is cut; successes keep it at 0.696.
  - Cutting the evidence lowers presence by about 0.24.
  - Safe statement: binary top-10 presence alone does not establish evidence dependence.
  - This is *not* the share of the signal that comes from priors.
  - It is a candidate method contribution on naturalistic data. *Legible Failures* removes priors by design on synthetic tasks.
- **Lens versus patching (§44, §46; a replication of a known caveat).** Patching finds causal sensitivity where vocabulary lenses read nothing. Cite Patchscopes §4.2 and the J-lens paper's §3.5.1, A.12 and §9.1. This is presented as an instrument check, not as a discovery.
- **Source-token interchange (§46; replication).**
  - The supervisor-name residual moves the concluded city through layer 15 (44/55); transfer fades to 0 from layer 33. Controls are near zero (random vector 0/55; same-city donor ≤ 1/47 at every layer). The substitution check missed its 0.95 bar (52/55 = 0.945) and must stay attached to every statement of this result.
  - This is the textbook source-token profile (Feng & Steinhardt; Gur-Arieh et al.; Lookbacks; Geva 2023).
  - Not shown: "moved out by attention" (no knockout or freeze was run), or a pure binding variable.
  - The probe arm is uninformative: its labels were confounded with base case.
- **Evidence tokens on real failures (§47; exploratory).**
  - On 26 *selected* LongMemEval rows (13 successes, 13 failures), the gold's value is carried at its own tokens through layer 30 on failures as on successes.
  - The entry filter equals §44's class rule, so no evidence-use failure could enter.
  - Forced readout ≠ free generation; n is 13 per class; no equivalence is claimed.
  - Consistent with ReDeEP App. C and propositional probes.
- **Q18 not established (§44; a useful negative).** Evidence-use failures could not be separated from answer-production failures on natural rows. Substitution validity failed on Qwen because of type-incompatible substitutes, small numbers and generic first tokens. This is a lesson for design.

## Chapter 6. RQ3 — already represented or additional computation? (exploratory; planned, not run)

- State the question and why it is hard: "present" can mean among the options, readable, or used.
- Prior work: Ruscio & Thompson report that first-token recovery leaves most full answers wrong (30.5–38.0% correct). Our §40 continued push adds +0.075 gold over the single-position push.
- The thesis does not depend on this chapter. It can be cut to a paragraph in the conclusion.

## Chapter 7. Negatives and boundaries

- Self-verification fails: §36b AUROC 0.573 and §37 0.505. This is expected (Song 2025; Huang 2024).
- Selective steering fails (above).
- The evidence-swap design fails on natural rows (above).
- Earlier programme: the learned controller adds nothing over linear selection (F3); RL does not beat imitation (F5); query-blind eviction cannot beat archiving (F8).

## Chapter 8. Conclusion

**What we can say:**
- Answer-encouraging interventions release evidence-dependent correct answers in one model, at measured costs.
- Presence readouts need evidence controls.
- Our instruments reproduce known profiles and limits in this setting.
- The failure's location is open.

**Future work:**
- The proposed test is §49, "legible abstentions" (D-63, rev 0a). It asks whether decoded-target repair works on false abstentions as it does on wrong answers, while keeping abstention where the evidence is insufficient. It is a testable contrast, not a first-in-setting claim. Until it is read, it stays under "proposed tests", kept separate from completed exploratory evidence and replicated findings. If it is read, it becomes the last section of Chapter 5, labelled with its population and limits.
- B (localisation) is gated on A′.
- Circuit-level work (CAC on Qwen) comes last.

## Claims register (to keep the thesis honest)

| Claim | Status label in the text |
|---|---|
| Evidence-dependent release with costs (Ch. 4) | finding, Qwen 3B; candidate method contribution |
| One-position release | replication (Liu) |
| Selective steering fails | negative, consistent with prior diagnosis |
| Late readability across families | replication |
| §34c survival | diagnostic; candidate method contribution, novelty not established |
| Lens-versus-patching gap | replication of a known caveat |
| §46 profile | replication; the attention explanation untested |
| §47 | exploratory, selected population |
| Q18 | not established |
| Self-verification | expected negative |

## Open items for the user

- Confirm Option D as the thesis frame with the faculty advisor.
- Decide whether RQ2's wording narrows (audit Part 5): "Does the internal answer in evidence-present failures track the supplied evidence, and where does it influence the response?" This avoids assuming the answer is "lost" somewhere (Codex review, point 8).
- Record corrections (STATE F13–F15 wording, "lens blind" phrasing) need the user's "Go". This document changes no record file.
