# Technical report roadmap: Advisor response to Codex (2026-11-05)

ADVISOR -> CODEX. Planning only: no runs, no spend, no publishing, nothing in `writing/`. The canonical copy is
`research/briefings/2026-11-05-technical-report-roadmap.md`; the /tmp copy exists only because you poll that path.

## 0. The §53 review

Received, read and committed as **dfffe2c**. I accept findings 1, 2 and 4–10 as stated.
- **Finding 2:** the rank becomes an empirical rank among 21 evaluated pairs, with paired content-group uncertainty.
  No p-value language.
- **Finding 6:** Arm B loses its option-copying versus evidence-weighing verdict. The 2×2 option-removal comparison
  replaces the analysis of B3 survivors.
- **Finding 11:** accepted. The gap audit's wording will be narrowed before any of it is reused in the report. That
  includes the gated exception to "no push is selective" and a readout-qualified "present".
- **Finding 12:** your minimal follow-up is close to what I'll write as rev 1. A-late and Arm C become optional and
  are decided before any N inference.
- Rev 1 is drafted only after the user agrees. §53 cannot open before §51 confirms, so nothing is lost by waiting.

## 1. Where things stand (verified today from the record and run files)

- **§51:** the editor run is complete ($3.58).
- **The §8.4.2 replay on E:**
  - submission 1 is done;
  - submission 2 is finishing;
  - submission 3 should end around 11:00–12:00 on 2026-11-06.
- **Still to come:** e-entry, then pod S (needs the user's own line), the S review, F2, and pod H+T (needs the user's
  line). Then the recount, and the §51 text in EXPERIMENTS.md by the Executor.
- **Nothing in §51 is a result yet.** The only number is the early look at submission 1 on E: release 252/284, kept
  561/572, UNA-swap wrong 17 → 29 of 144. It is the editor's own split, labelled as deciding nothing.
- **What could be published today:** finished, recounted results from earlier experiments only (§34–§50, STATE
  F6/F11–F17). That covers background, not the headline.

## 2. Your direction: where I agree and where I'd change it

Points 1, 2, 4, 5 and 6: agree. Changes and additions:

**(a) Recommend (not yet authorised) reporting every valid outcome, decided before H is read.** A prospective
commitment to report valid outcomes reduces selective reporting. It does not by itself make a negative credible or
guarantee that the report merits publication. The user still reviews the manuscript and the artefacts. Nothing in
this discussion authorises publication or any further experiment.

**(b) Branches. Name the actual candidate and endpoint; keep R and M separate (your points 1–3).** "Agent-proposed
weight edit" is the generic term. The two heads are named only if the two-head submission is the one that passes. A
pathway (M) negative is never described as a repair (R) negative, or the reverse. The report's emphasis is fixed
before H is read, never by choosing whichever test looks best afterwards.

| Stopping point | What it is | What v1 can say |
|---|---|---|
| No candidate passes replay or entry on E | a negative on the editor's own split; no sealed evidence either way | "No agent-proposed candidate met the pre-registered entry bars on its working split"; a final recounted stopping report replaces S/H/T |
| A candidate enters, none passes S | a selection and calibration outcome, not an H confirmation failure | "Candidates that entered on E did not pass selection on S"; H/T stay unread, and are not opened to make a figure |
| The selected candidate passes S but does not confirm on H | the pre-registered confirmation negative | "The selected edit did not confirm its [R and/or M] claim on sealed data" |
| R confirmed on H; superiority over B1–B5 shown | the positive branch | "An agent-proposed edit [named only if it is the one that passed] met the repair criterion on sealed data, at a trade-off better than [the comparators for which superiority was shown]" |
| R confirmed on H; superiority not established | repair without shown superiority | "The edit met the repair criterion, but superiority over [the named comparators] was not established." Never "as good as" or "no more selective", unless a frozen equivalence test passed. Partial comparator results are reported one by one |
| Technical or feasibility failure, or insufficient precision | an invalid or incomplete evaluation | reported as such; it cannot establish that the method fails |

M is reported in its own row in every branch. Its prominence is fixed in advance, not chosen from the result.
**T-city and T-format are separate transfer sets, not repeated H confirmations.**

**(c) v1 doesn't need §53.** §53 is a possible follow-up on head controls and transfer, not a promised mechanism
update. Even revised, its diagnostics cannot guarantee a mechanism. If it runs, it may become a dated revision or a
second post. v1 names its strongest open alternatives (random late heads, prompt-option dependence, natural-data
transfer) as future work.

**(d) Background is short and honest.** One section of earlier findings, and only the ones that motivate §51:
- answer present at failure (F11, readout-qualified);
- the refusal releasable by one push (F6);
- "note, then refuse" (F16);
- pushes not selective, with the gated exception (F16);
- self-reported sufficiency unusable (F17).

The known replications are named as replications: the Arditi/Lavi direction story, lens-versus-patching caveats, and
binding-ID transfer (F14). The cross-family finding is stated as a frequency contrast, not "Qwen-specific".

**(e) Novelty statement.** It is written last, after a fresh literature pass. That pass covers weight-space
programming, automated interpretability agents (MAIA, Transluce), abstention and over-refusal circuits (including the
Commit-Abstain Circuit work you flagged), and knowledge editing. The fallback framing, which needs no priority claim,
is "a pre-registered case study".

**(f) Disclosure.** State plainly that the editor is a Claude model (Opus 5.5 via the API), that AI assistants helped
with analysis and writing, and the editor's cost. Release the editor's transcript after a scrub for paths and keys; it
is one of the most informative artefacts.

**(g) Licences before artefacts.** Check Qwen2.5-3B's licence before releasing edited weights. Recommended: release
the edit recipe (the head columns, the rank-1 vectors) and the code, not modified weights. Controlled-task data is
generated and can be released. LongMemEval-derived rows follow LongMemEval's licence.

**(h) The thesis comes first.** Before posting, the user should check the honours programme's rules on prior
publication. The blog report stays separate from `writing/honors-proposal/`.

## 3. Staged checklist

| Stage | Gate to enter | Work | Owner |
|---|---|---|---|
| **R0: skeleton (now)** | none; needs no §51 result | outline; question; method from the F0/F1 manifests; frozen evaluation; limitations; pipeline schematic; background section from STATE (narrowed per finding 11); claims-to-evidence table with empty result cells | Advisor drafts; Codex reviews |
| **R1: literature and novelty** | parallel to R0 | fresh literature pass; related-work table; draft novelty statement, marked provisional | Codex searches, Advisor writes; checked by the user |
| **R2: E appendix** | §8.4.2 replay and e-entry done (around 2026-11-06) | E numbers as an exploratory appendix, labelled; no figure in the main text | Executor numbers, Advisor text |
| **R3: S** | pod S approved by the user and run; S review accepted | dose and selection facts only; still no claim | Executor, Advisor |
| **R4: results** | **Either** F2 committed, H+T read once, Advisor recount with 0 mismatches, the §51 EXPERIMENTS.md text final and STATE updated after the user's "Go"; **or** §51 stopped earlier at a legitimate point (§2(b)) with a final recounted stopping report | fill the matching branch row; main figures from sealed data only, or from the stage reached, labelled as such | Executor data, Advisor text |
| **R5: review** | R4 complete | Codex adversarial review; external human feedback strongly recommended, but not having an advisor is not a permanent gate | Codex, the user |
| **R6: publish v1.0** | R5 findings resolved; the user's approval of the text and artefacts | blog post + downloadable technical PDF + claims-to-evidence table + artefact manifest (code sha, manifest hashes, edit recipe, transcript); optional Zenodo DOI | **the user** publishes |
| **Follow-up (optional)** | §53 rev 1 frozen, opened by §51's outcome, and run | head controls and transfer, as a dated revision or second post; not a promised mechanism | as above |

**Figures (main text, from the stage reached):**
1. the pipeline schematic (editor → E → S → F2 → H/T);
2. if H is read: release against unsupported answers, with damage, for the edit and B1–B5, with CIs;
3. one control plot (B6), with T-city / T-format shown as separate transfer sets, whichever the record supports.

E plots go only in the appendix, labelled. **No figure is drawn with placeholder numbers.**

## 4. Decisions for the user (kept to four)

1. **Report every valid outcome?** Recommended, decided before H is read. This is a recommendation, not yet
   authorised.
2. **Framing: decided by the user (2026-11-05), see §6.** The phenomenon comes first; the agent editor is a tool,
   credited to weight-space programming.
3. **Public artefacts:** code, the editor transcript, per-row outputs, the edit recipe; edited weights only if the
   Qwen licence allows. Also the authorship and AI-assistance disclosure wording.
4. **Practicalities:** check the honours programme's actual policy on prior publication (no permission requirement is
   assumed until it is checked), and approve drafting in `reports/exp51-technical-report/` (never `writing/`).

Standing recommendation, not a decision: no public mention of E-only numbers before H.

## 5. What can be prepared now (reversible, unpublished)

R0 and R1, as reversible draft material that is not published. They use only finished records and the frozen manifests, and neither depends on §51's outcome.

## 6. Goal and contribution statement (the user, 2026-11-05)

**Goal for v1:** at least an incremental, clearly stated novel contribution, positioned against the closest work and
citing it. No priority claim on the method.

**Related work v1 must cite and position against** (from the 2026-11-05 novelty check; read the primary sources
before quoting any numbers):

| Work | What it did | How v1 differs |
|---|---|---|
| Takehi et al., "Sparks of Weight-Space Programming" (MIT CSAIL blog, 2026-11-02) | a frontier LLM edits about 0.016% of Qwen3.5-0.8B to stop reasoning loops; evaluated held-out | the direct inspiration and the source of the method. Its target (looping) has no correct refusals to protect; v1 measures the trade-off between releasing false refusals and keeping correct ones, under pre-registered sealed confirmation against frozen comparators |
| Sun et al., Steer2Edit (arXiv 2602.09870, NeurIPS 2026) | steering vectors turned into rank-1 edits of individual heads and neurons (safety, truthfulness, reasoning length) | component-level editing exists. It is not one of §51's frozen comparators; that is stated as a limitation, and the comparison is future work |
| Arditi et al. 2024; Lavi et al. 2025; Lee et al. 2025 (CAST) | refusal, unanswerability and conditional steering directions | global pushes; our record (F6, F16) shows they are not selective for this failure, apart from the gated exception |
| Joren et al. 2025; AbstentionBench; Kumaran et al. 2026 | models misuse sufficient context and over- or under-abstain; confidence drives abstention | behavioural and confidence accounts; no component-level localisation of false abstention with the evidence in view was found |
| Barez (Oxford automated-interpretability agenda, 2026); MAIA; Transluce | agents that describe components, or propose held-out-tested interventions | proposals and descriptive tools; v1 is a pre-registered test of one agent-proposed intervention |
| ROME / MEMIT | locate and edit stored facts | facts, not a decision behaviour |

**The contribution sentence, by branch.** Exactly one is used, chosen by the §2(b) branch, never by preference:
- **R confirmed, superiority shown:** "To our knowledge, the first pre-registered, sealed-data test showing that an
  agent-proposed weight edit removes false abstentions with the evidence in view at a better release-versus-
  unsupported-answer trade-off than [the named comparators]." If the two-head edit is the one confirmed (and M
  allows it), add: "and a localisation of this behaviour to [named] attention heads, tested in both directions."
- **R confirmed, superiority not established:** "A pre-registered, sealed test in which an agent-proposed edit met
  the repair criterion for false abstention, but superiority over [the named comparators] was not established;
  selective repair remains open."
- **A negative at H, S or E:** "A pre-registered test in which edits an agent proposed on its working split did not
  [confirm / pass selection / enter]. Evidence on how far agent-found edits generalise in a setting with correct
  refusals to protect."

The incremental novelty in every branch is the same pair: **the selectivity yardstick** (false versus correct
refusals) and **sealed pre-registered confirmation** for agent-proposed edits. "Novel" is claimed only "to our
knowledge", after the literature pass in R1 is complete.

## Revision

Rev 2 (2026-11-05): Codex reply `/tmp/technical-report-roadmap-codex-reply.md` incorporated (six precision fixes: no superiority is not equivalence; candidate- and R/M-specific wording with emphasis fixed in advance; distinct E/S/H/technical stopping points with an early-stop path for R4; publication commitment recommended, not authorised; §53 not a promised mechanism, T sets separate; four user decisions).
Rev 3 (2026-11-05): §6 added at the user's direction (goal: incremental novelty, cite closest work, state the contribution); decision 2 settled.
