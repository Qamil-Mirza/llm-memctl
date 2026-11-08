# Editorial review: §51 technical report, v0.2 → v0.3

Reviewed 2026-11-07. Baseline: v0.2 as committed in `e39ad04` (the draft PDF
`exp51-technical-report-draft-2026-11-07.pdf` is in the same commit). Revision: v0.3, this commit.

Sources checked: `runs/exp51/gpu_podHT/h_report.json` (sealed H report, sha 51bbbd2e…), `runs/exp51/gpu/design.json`
(F2 design), `runs/exp51/editor_run/{submissions,loop_summary.json}`, `runs/exp51/run/b0` and `runs/exp51/run/arms/E`,
`EXPERIMENTS.md` §51, `research/STATE.md` (F13, F17, F18 and the cross-family rows), the H and S reviews in
`research/reviews/`, the proposal `research/proposals/2026-11-02-exp51-agent-weight-editing.md`, and the analysis code
(`memctl/analysis/{mech51,stats51,edit51}.py`, `scripts/exp51_{common,stage4}.py`). Bibliography entries were checked
against arXiv records, ACL Anthology, NeurIPS/ICLR/PMLR proceedings and the authors' pages.

No experimental data, frozen protocol, edit recipe, test output or historical artefact was modified. No model was run.
The only new computation was descriptive: marginal bootstrap intervals for Figure 2, computed from stored H outputs with
the frozen code (`figures/tradeoff_data.py`), and component p-values for the O2 test against B1 and B4, recomputed to
confirm which component drove the result. Every point estimate the new script produces is asserted equal to
`h_report.json`.

## 1. Executive assessment

The v0.2 draft had sound science and careful qualifications, but a reader new to the project would struggle to find the
argument. The main result was clear in the abstract. Several things were not: why the failure is interesting, what
exactly the agent changed, why the B1 comparison failed, and which edit the mechanistic section was about. The draft
also leaned on earlier write-ups for motivation and on internal shorthand (O1/O2, M1–M3, FA/C/UNA) before defining it.

Three factual errors were found and corrected (section 4 below). None changes a confirmatory verdict. One claim, the
"first pre-registered test", was narrower than its wording and has been reworded. The bibliography had placeholder
authors and titles in most entries, and one wrong title; all entries are now verified, with three venue caveats left as
comments.

v0.3 keeps every verified number, test, split, freeze decision and limitation, and reorganises the report along the
line failure → why it matters → why simple fixes fail → question → design → evaluation → findings → interpretation →
limits.

## 2. Major writing and structural problems in v0.2

| # | Problem | Where (v0.2) | Fix in v0.3 |
|---|---|---|---|
| 1 | The failure was described abstractly; no example. | `01-question.tex` §1 | Verbatim example from stored outputs (Fig. 1, case `E-DE01_ANSa_H0`), labelled as exploration-split illustration. |
| 2 | The "question and main finding" section mixed motivation, earlier results from other experiments, cross-model rates and the main finding. Readers could not tell which numbers belonged to this study. | `01-question.tex` | Split into Introduction (§1) and "Earlier observations on this model" (§2.1), which is labelled as context, not results. |
| 3 | Not self-contained: relied on two blog posts for the motivation. | `01-question.tex` | Links moved to a footnote; §2.1 states the needed facts. |
| 4 | Abbreviations (FA, C, UNA, CF, Rel, Uns, Dam, Deg, O1/O2, M1–M3, B0–B6) introduced in passing or used before definition. | `03-setup.tex`, `05-evaluation.tex` | Notation table (Table 1); prose uses descriptive names, tables keep the symbols. O1/O2 explained in plain words with superiority / non-inferiority / equivalence distinguished. |
| 5 | Base-weighted vs pooled rates mentioned in one sentence, with no example. | `03-setup.tex` | New paragraph with the damage example (11/593 = 0.019 pooled vs 0.030 base-weighted) and why Uns coincides. |
| 6 | The selected edit's form was not given; "rank-1" without saying that the dense update changes ~4.19M entries. | `04-proposals.tex` | Formula, read/write explanation, and "low-rank, not sparse" note. |
| 7 | Figure 1 was a four-cell table that floated two pages from its text and did not show data access. | `03-setup.tex` | TikZ pipeline with a data-access row under each stage (Fig. 2). |
| 8 | Planned trade-off figure was a comment. | `06-results.tex` | Fig. 3 drawn from stored outputs, with marginal intervals labelled descriptive. |
| 9 | The B1 result said "did not survive Holm" without saying which part failed. | `06-results.tex` | States that the Uns reduction is clear and that the O2 recovery non-inferiority component (p 0.015 → Holm 0.092) was the bottleneck (verified by recomputation). |
| 10 | §7 mixed the two edits; easy to read pathway results as being about the layer-35 repair. | `07-what-edits-explain.tex` | Three-level table (repair / pathway / account) with the edit tested at each level; explicit statement that nothing in §7 tests the layer-35 explanation. |
| 11 | No discussion section; no conclusion. | — | §8 Discussion (repair, B1, edit-specific mechanism, behaviour vs explanation, scope, dose grid, one run, controls) and §10 Conclusion. |
| 12 | Pending follow-up shown as a results section with an orange placeholder. | `07b-followup.tex` | Moved into the Discussion as "§8.1 Pre-registered follow-up (in progress; no results reported)"; design text kept. |
| 13 | Key-findings box repeated the abstract. | `keyfindings.tex` | Replaced by a claim / evidence / status table (shown, inconclusive, not established, descriptive). |
| 14 | Related-work table (v0.2 Table 7) dense, ungrouped, with some inaccurate descriptions. | `D-related-work.tex` | Grouped longtable with repeated headers and "continued" footer; descriptions corrected against abstracts. |
| 15 | References printed before appendices with "VERIFY" text visible; layout gaps on pages 4 and 8. | `refs.bib`, `main.tex` | Bibliography verified; caveats are BibTeX comments. Float parameters and placement fixed. |

## 3. Scientific-communication concerns

- **Which tests are confirmatory.** v0.2 said this in several places. v0.3 has one table (Table 4) listing every
  analysis as confirmatory, descriptive or exploratory.
- **Point-estimate thresholds.** Damage ≤ 0.05 and CF-follow ≥ 0.90 apply to point estimates. v0.3 says what passing
  them does and does not show (§5, stage 4) and repeats it in the limitations.
- **Inconclusive vs equivalent.** The B1 wording is kept as "inconclusive: neither superiority nor equivalence". The
  Discussion explains why the comparison matters (B1 is a pure output-level change).
- **Repair vs explanation.** The layer-35 edit's own "answer available" account was never tested in confirmation.
  v0.2 implied this; v0.3 states it in §7 and §8.
- **The note.** v0.3 keeps the v0.2 caution that the note is a prompted summary, not a faithful trace, and does not
  describe the model as "knowing" the answer.
- **Generality.** Every claim is scoped to one model, synthetic city templates, one instruction format and one agent
  run.

## 4. Errors, ambiguities and overstatements found

| # | v0.2 text | Problem | Evidence | v0.3 |
|---|---|---|---|---|
| E1 | "over about 16.4 hours" (agent run) | 58,978 s is the ledger's summed pass time, which the record says is not a wall-clock time. | `EXPERIMENTS.md` §51 ("about 34,950 s from start to the last submission… The ledger figure is not a wall-clock time"); `loop_summary.json` | "about 9.7 hours from start to the last submission". |
| E2 | "Two activity predictions held" (M3) | `h_report.json` gives p = 1e-4 for all four activity tests: evidence_removed (directional) and city_changed, distractors_added, unrelated_vs_absent (TOST within 2·SD margins). The H review lists three while saying "two". | `h_report.json` → `M.M3.activity` | "All four passed their frozen tests", plus the distractor caveat (estimate +1.03 [0.52, 1.55] excludes zero and exceeds the agent's own stated bound of 0.5). **Author should confirm** (blocker B6). |
| E3 | "Allowed edits were rank-≤4 updates or sparse column edits" | The proposal allows sparse row *or* column edits, ≤ 512 per candidate. | Proposal, "Components and the edit family" | Corrected. |
| E4 | "To our knowledge, this is the first pre-registered test…" | True only of the narrow combination; based on a targeted search. Also did not say what "pre-registered" means here. | Literature check 2026-11-07 (bib verification notes) | "We are not aware of an earlier…; this rests on a targeted search, not a systematic review." New paragraph: pre-registration = time-ordered, hashed commits, not an external registry. |
| E5 | Related-work descriptions | Kumaran et al. title wrong; GRAB-RAG has four context conditions and tests prompt-based policies (not "no intervention"); Residual Paving trains new modules ("no weight change" true only of the backbone); AbstentionBench is about under-abstention; commit–abstain circuit reduces false abstention with a trained policy (closest work, was under-described); LLM auditors are single-prompt runs that also repair; Dr. No also tests a text model. | Primary sources | All corrected in §2.2 and Appendix D. |
| E6 | B6 "five size-matched random edits" | Ambiguous: B6 perturbs the same weight entries as the edit, not random locations. This matters for what M1 shows. | Proposal, comparator table | Stated precisely in §7 and Appendix C. |
| E7 | "Edits 1 and 3 change about 0.52M weights; edit 2 changes about 4.19M" | Correct, but "rank-1" readers may assume a tiny change. | 2·128·2048 = 524,288; 2048² = 4,194,304 | Kept, with the low-rank note. |

Unchanged on purpose: every number in the abstract, key findings, Table 5 (v0.2 Table 3), T-city, costs, follow-up
design and limitations. All were re-checked against `h_report.json`, `EXPERIMENTS.md` §51 and `STATE.md`.

## 5. Bibliography

All 23 entries now have full titles and author lists from primary sources; 19 were corrected. Remaining caveats (kept
as BibTeX comments, listed in the draft blockers box):

- `kumaran2026confidence`: the *Nature Machine Intelligence* version is reported only by secondary sources, so it is
  cited as arXiv 2603.22161.
- `commitabstain2026`: NeurIPS 2026 is taken from the arXiv comment; proceedings page not checked.
- `setiawan2026grabrag`: arXiv comment says "Committed to AACL-IJCNLP 2026"; cited as arXiv.
- The "no earlier head-to-head comparison of a weight edit with logit bias" claim is kept as "we found no…"; it was not
  searched systematically.

## 6. Reproducibility audit

Specified in the report: model and revision hash, fp32, greedy decoding, token limits, the verbatim prompt and the B2/B3
variants (new Appendix C.1), case generation seeds and partition rule, case-file hash, grading rules, utility, edit
manifests and dose scaling, selection rules, F0/F1/F2 commits, inference rules, stopping rules, costs and hardware.

Material gaps (actions for the author):

1. **Nothing is public yet.** Code, per-row outputs, the transcript and the commit history are held privately. The
   pre-registration claim can be checked only once the commit history is public. *Action:* publish the repository (or
   a frozen archive with DOI) including `scripts/exp51_*`, `memctl/analysis/{edit51,mech51,stats51}.py`, the case
   file, per-row outputs and the redacted transcript.
2. **Edited weights.** Release depends on the Qwen licence. *Action:* check the licence; if allowed, release the three
   manifests (they are small: factors, not dense matrices).
3. **Agent prompt and tool specification.** The report describes the sandbox but does not reproduce the agent's system
   prompt or tool schemas. *Action:* add them to the artefact release, or an appendix.
4. **Earlier-experiment context (§2.1)** cites results from other experiments without a public source. *Action:* either
   link the corresponding records once public or label them as unpublished.

## 7. Recommended fixes, ranked

Done in v0.3 (1–10), remaining (11–15):

1. Correct E1 and E2 (factual).
2. Narrow the novelty claim and define pre-registration (E4).
3. Explain the B1 result by component, without implying equivalence.
4. Separate repair, pathway and account evidence, and name the edit tested at each level.
5. Add a concrete example and the trade-off figure.
6. Explain base-weighting, utility, O1/O2 and the evidential status of every analysis.
7. Verify and correct the bibliography and related-work descriptions.
8. Add Discussion and Conclusion; move the pending follow-up out of the results.
9. Redraw Figure 1 to show data access; fix float placement and the first page.
10. Add the prompt, edit-application details and a reproducibility checklist.
11. **Author:** confirm the M3 activity count (E2).
12. **Author:** approve the AI-assistance disclosure wording.
13. **Author:** publish artefacts and add the URL/DOI.
14. **Author:** decide whether to keep the in-progress follow-up section in the release or move it to future work.
15. **Author:** decide whether to keep the two informal web links.

## 8. Unresolved blockers requiring author input

B1 public repository URL / DOI · B2 edited-weight licence check and redacted transcript · B3 disclosure wording ·
B4 follow-up section (no results; do not fill until recounted and reviewed) · B5 three bibliography venue caveats ·
B6 M3 activity-prediction count · B7 informal links. The same list prints at the end of the main text when `\drafttrue`.

## 9. Final quality test (skeptical first-time reader, main text only)

| Question | Answered in v0.3 |
|---|---|
| 1. What failure is studied? | §1 and Fig. 1: "unknown" after a note that quotes the answer; 232/261 H false abstentions are answer-bearing. |
| 2. Why study it? | §1 "Why the failure is hard to fix": the recovery–unsupported trade-off; earlier fixes raised unsupported answering to 0.188 and 0.451. |
| 3. What did the agent propose? | §4, Table 3 and the formula for the selected edit. |
| 4. How was it evaluated independently? | §5, Fig. 2: E → S → F2 freeze → H and T-city read once; Table 4 marks what is confirmatory. |
| 5. What improved? | §6.1: utility +0.117 [0.096, 0.138], recovery 0.816, unsupported 0.035 → 0.042, damage 0.030. |
| 6. Which comparisons were inconclusive? | §6.2: B1 inconclusive (with reason), B3 not established. |
| 7. Evidence for a causal pathway? | §7.1: M1 and M2 at layer 33 for the two-head edit; sufficient, not necessary. |
| 8. Unsupported mechanistic claims? | §7.2 (agent's account not established, not refuted); §8 (layer-35 explanation untested). |
| 9. Main limitations? | §9 list; §8 dose grid, one run, one model. |
| 10. Contribution beyond prior work? | §2.3 and the Contribution paragraph of §1: the combined evaluation design and its result, not agent weight editing itself. |

## 10. Build

`./build.sh /path/to/tectonic` (Tectonic 0.17.0 was used). The build has no errors, no undefined references or
citations, and no overfull boxes. `python3 flatten.py` regenerates `exp51-technical-report-single.tex`.
Figure 3 is regenerated by `.venv/bin/python figures/tradeoff_data.py` then
`uv run --no-project --with matplotlib python figures/tradeoff_plot.py` (from the repository root).
