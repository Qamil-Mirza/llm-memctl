# Literature positioning for §33–§44 (2026-10-29)

Pass run by a background search agent on the Advisor's brief; the two closest neighbours (Lavi et al. 2025, Walters
et al. 2026) re-verified by the Advisor from the arXiv abstracts. Items marked (unverified) rest on the agent's
reading of secondary sources or memory and must be checked before citation.

## 1. The abstention direction with evidence in context (§38–§42)

- Arditi et al. 2024, *Refusal in LMs Is Mediated by a Single Direction*, NeurIPS 2024, arXiv:2406.11717. Safety
  refusal; difference-in-means direction; ablation removes refusal in 13 chat models. Correctness of un-refused
  outputs not measured.
- Ferrando et al. 2025, *Do I Know This Entity?*, ICLR 2025, arXiv:2411.14257. SAE known/unknown-entity latents
  causally regulate refusal, closed-book; answers after intervention graded only qualitatively.
- **Lavi, Milo & Geva 2025, *Detecting (Un)answerability in LLMs with Linear Directions*, arXiv:2509.22449
  (verified).** Extractive QA, passage + question; a direction that detects unanswerability and, added or ablated,
  controls abstention; generalises across datasets. The abstract does not name models and does not report the
  correctness of answers after steering. The closest neighbour to §38–§42.
- Slobodkin et al. 2023, *The Curious Case of Hallucinatory (Un)answerability*, EMNLP 2023 (arXiv id unverified):
  linear probes separate answerable from unanswerable in extractive QA.
- Luo et al. 2026, YOPO, arXiv:2608.14465 (unverified): a sufficiency direction in frozen Qwen 2.5 gates abstention.

**What ours adds, stated cautiously.** The opened answers are graded (about half correct with the evidence in view,
a fifth without, §38; replicated on LoCoMo, §40/§42); one position suffices and the first token carries it (§39,
§39b); and in Qwen 2.5 3B the projection on the direction ranks extractability, not answerability (fires as readily on
unanswerable refusals, §42). The last point sits against Lavi et al.'s reading that such a direction detects
unanswerability; the settings differ (their direction is built from unanswerable vs answerable contrasts, ours from
refusing vs answering with the evidence present), so the right statement is "a refusal direction is not an
answerability direction", not a contradiction of their result.

## 2. Answer present but not output (§34, §43)

- Orgad et al. 2024, *LLMs Know More Than They Show*, ICLR 2025, arXiv:2410.02707. Truthfulness signal at the
  exact-answer tokens; models may encode the right answer and generate a wrong one.
- Gekhman et al. 2025, *Inside-Out: Hidden Factual Knowledge*, COLM 2025, arXiv:2503.15299. Closed-book; internal
  ranking exceeds expressed knowledge by about 40% relative.
- **Walters, Chen & Toni 2026, *Retrieved but Not Read*, arXiv:2610.21731 (verified).** Search-augmented LLMs with
  up-to-date evidence available: a layer-wise gap between evidence available and evidence used; retrieved evidence
  moves internal predictions while the final answer stays outdated. (The agent reports logit- and J-lens readouts and a
  no-page baseline in the body; the abstract names neither, so cite for the gap, verify the lens detail before citing
  it.) No patching.
- Ma, Zhang & Nie 2026, CoRect, arXiv:2602.08221 (unverified): "parametric suppression" of context tokens by deep
  FFN layers in RAG.
- Villuri et al. 2026, arXiv:2609.04582 (unverified): answer-surface features can mimic probing results; a
  non-monotonic size result.
- Jin et al. 2024, arXiv:2402.18154: late-layer memory heads vs context heads.

**What ours adds.** Four instruction-tuned families at 3B–14B on long-term-memory QA with sealed folds; and the
evidence-ablation control (§34c): about 60% of top-10 presence survives removing the evidence. The agent found no
prior paper reporting that share; it is the main caveat on the "knows but doesn't say" family and should be stated as
such.

## 3. Lens disagreement and causal adjudication (§34, §44)

- nostalgebraist 2020, logit lens (LessWrong). Belrose et al. 2023, tuned lens, arXiv:2303.08112 (logit lens biased;
  fails on GPT-Neo and BLOOM). Din et al. 2023, *Jump to Conclusions*, arXiv:2303.09435 (fitted maps read earlier).
- Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace*, arXiv:2607.15495 (J-lens; unverified
  by fetch). Zhang & Nanda 2024, *Best Practices of Activation Patching*, ICLR 2024, arXiv:2309.16042 (metric choice
  moves where effects appear: our §44 reports log p and rank). Meng et al. 2022 causal tracing; Ghandeharioun et
  al. 2024 Patchscopes; Geva et al. 2023 *Dissecting Recall*, arXiv:2304.14767; Wu et al. 2024 retrieval heads,
  arXiv:2404.15574.
- **Gupta et al. 2025, *How Do LLMs Use Their Depth?*, arXiv:2510.18871 (unverified):** tuned-lens transition
  layer matched by patching in GPT2-XL / Pythia-6.9B / Llama-2-7B but not Llama-3-8B. The only lens-vs-patching depth
  comparison found. Lad et al. 2024, *Stages of Inference*, arXiv:2406.19384: a logit-lens phase transition across
  families, no size trend.

**What ours adds.** A 0.46 vs 0.82 disagreement between two lenses on the same 7B, adjudicated by evidence-cut
single-position patching on in-context answers (§44, running). Report §44 with Zhang & Nanda's metric caution.

## 4. Depth vs size across families (§43)

Nothing found stating that larger models hold the answer at an earlier relative depth; Villuri et al. report a
non-monotonic size case. Our Qwen-only size effect and its absence in Llama and Gemma should be framed as an
observation, not a law.

## 5. Self-verification (§36b, §37)

Huang et al. 2023, arXiv:2310.01798 (no intrinsic self-correction without oracle feedback). **Song et al. 2024,
*Mind the Gap*, ICLR 2025, arXiv:2412.02674:** the generation-verification gap is non-positive for small models and
grows with verifier size. Saad-Falcon et al. 2025, Weaver, arXiv:2506.18203 (weak verifiers need ensembling). Ours is
a context-grounded instance: evidence in view does not rescue a 3B's self-check (AUROC 0.57 / 0.50); the 7B checker
reaches 0.69, as Song et al. would predict.

## 6. Pre-registration in interpretability (method)

NeurIPS 2020/2021 pre-registration workshops (PMLR 148); Shi et al. 2024, *Hypothesis Testing the Circuit
Hypothesis*, arXiv:2410.13032; Sharkey et al. 2025, *Open Problems in Mech Interp*, arXiv:2501.16496. No sealed-fold
protocol for lens or steering claims found; ours is an early example and may be stated as a contribution of method,
citing the above as precedent.

## Actions

- Cite Lavi et al. and Walters et al. in EXPERIMENTS §38 and §43 (Executor, on the Advisor's word) and in the thesis
  outline. Verify the (unverified) items before any citation.
- The contributions statement (`2026-10-29-contributions.md`) is updated to point here.
