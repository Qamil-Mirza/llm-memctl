# Measurement audit after §45: is the instrument the limit? (Advisor assessment for Codex and the user)

**Author:** Advisor, 2026-10-30. **Trigger:** Codex handoff relayed by the user ("have we considered whether our tools
are limiting ... before a pivot"). **Status:** advisory. No run, no spend, no record change. The exploratory numbers in
§1 are DEV HALF ONLY, post hoc, and do not revise any §45 verdict (review 1a6e1e5 / ca5ab0e; text c70cb71).
Script: scratchpad `audit45.py`, `audit45c.py` (read-only over `runs/exp45_maps/j56-jrefit-logit/stage1/`).

## 1. Codex points 1 and 2, checked against the existing maps (dev, solved prompts, n = 256)

**1a. Positions beyond the counted ones.** §45's headline counted the last prompt token and the answer span. The maps
store every position, so the same top-10 readability (rule A: a position whose input token is the target is skipped)
was read at every *evidence* position and every *question* position, layers 0–29, for gold, intermediate (RC) and
competitor (TU), all three lenses.

| block | j56 hits by layer | jrefit | logit | hits at layers 2–29 |
|---|---|---|---|---|
| question positions | 0 | 0 | 0 | 0 |
| evidence positions | 127 at L0, 42 at L1 | 134 / 41 | 26 / 11 | **0 / 0 / 0** |

The evidence-block hits sit at ` in` (66), `.\n` (39), ` to` (24), `:` (13): the layer-0/1 readout after "lives in"
is "a city", and any pool city, the gold included, is in that top-10 (DE00_v3_F1_H2: ` Durham` read at the ` in` of
*six different distractor sentences*). That is a lexical prior, not a relation. So "no earlier intermediate is
recovered by these lenses" extends from the counted positions to **every prompt position**. Point 1 is checked; the
verdict does not move. (Exploratory; the share-per-prompt tables are in `audit45.py` output, 0.00 at every question
block, 0.09–0.29 at evidence blocks, all of it layers 0–1.)

**1b. Continuous ranks, not top-10.** At the last prompt token, the median rank (vocabulary 151k) of the RC
intermediate (` Laura`-type) is 17k–113k through layer 29, then 3.9k (L30), 3.1k (L31), 248 (L32), **6 (L33)** for
j56 on F1; jrefit and logit the same shape. TU, gold city vs the superseded city, both present in the prompt, same
position: the gold's rank beats the competitor's on 0.50–0.58 of prompts (chance) through layer 31, 0.79–0.92 at
layers 33–35 (F1, n 24); on F2 it stays at chance at every layer (0.33–0.61, n 51) because the model writes a note
before answering, so the last prompt token is not where its answer forms. **In vocabulary-projection space there is no
evidence-sensitive signal below the top-10 either.** Point 2's distinction is right and the answer is the same.

**1c. What this localises.** The blind spot, if there is one, is not a threshold or a position. All three readouts are
linear maps into the *output vocabulary*; they see only what the residual stream writes in unembedding directions.
A relation variable ("which name is A's supervisor", "which city is current") need not live there at all, and the
literature says it does not (§2). Codex's correction is accepted: the §45 lens sentence should read "stable under a
128-prompt refit and equal to the logit lens; not validated as complete", not "instability ruled out".

## 2. Prior art that bears directly on "instrument or route"

- **Geva, Bastings, Filippova, Globerson 2023, "Dissecting recall of factual associations"** (arXiv 2304.14767): the
  subject is enriched at the *subject token* in early–mid MLPs; the attribute is *extracted to the last position by
  late attention heads*. Lateness of the answer at the answer position is the known route for recall. §45's "readable
  only at layers 30–33 at the producing position" is consistent with that route, not evidence of a broken lens.
- **Feng & Steinhardt 2023, "How do language models bind entities in context?"** (arXiv 2310.17191): exactly our
  controlled setting ("A lives in C1. B lives in C2. Where does A live?"). Binding is carried by *binding-ID vectors at
  the entity and attribute positions*, found by **causal mediation / interchange at those positions**, in directions
  with no vocabulary meaning. A logit or J-lens at the last token cannot see it by construction.
- **Prakash et al. 2024, "Fine-tuning enhances existing mechanisms: entity tracking"** (arXiv 2402.14811): DAS and path
  patching on entity tracking in Llama; the circuit is located at the entity positions and mid layers.
- Hernandez et al. 2023 (LRE, 2308.09124) and the J-space appendix (Codex's link) for relation readouts; Ghandeharioun
  et al. 2024 Patchscopes (2401.06102); Geiger et al. 2021/2023 causal abstraction and DAS (2106.02997, 2303.02536).
  Lavi et al. 2025 (unanswerability direction) and Walters et al. 2026 (present-not-used) remain the thesis neighbours
  (briefing 2026-10-29).

Reading: on the toy RC/TU tasks, the literature already predicts (i) a binding variable at entity positions in mid
layers, invisible to vocabulary lenses, and (ii) late attribute extraction at the answer position. So the expected
outcome of a tool audit on the controlled tasks is **"instrument blind spot confirmed, route late at the answer
position"** — a known result, worth one cheap confirmation on our model, not a contribution. The open question is
whether the same variable can be found and moved on the §44 long-context rows, where the thesis lives.

## 3. Codex's alternatives, feasibility on this programme

| method | cost here | what it adds | main risk |
|---|---|---|---|
| contrastive linear probe at entity positions and last token (dev-train, held-read once) | $0, laptop fp32; hooks exist (`memctl/analysis/patching.py`, `maps.py`) | decodes the *variable*, not the word; layer/position where binding is linearly present | correlation only; probe does its own computation (Hewitt & Liang control tasks needed) |
| interchange intervention at the supervisor-name position (Feng & Steinhardt style) | $0, laptop; `steering.overwrite_spec` + the §44 capture path | causal: does the output city follow the swapped binding | off-manifold; needs the scale-matched controls the §44 rev 6 lesson gave us |
| DAS / causal abstraction | moderate build (pyvene or own); laptop | aligns a hypothesised variable to a subspace, validated by held-out interchange | toy-task alignments may not transfer to §44 rows; "DAS by application" is not novel |
| Patchscopes | low–moderate; laptop | readout not tied to one vocabulary projection | target-prompt hallucination; needs a null (shuffled source) |
| template / phrase lens (J-space appendix) | low; our own lens | multi-token concept readout | still vocabulary-anchored; same blind spot as §1c |
| attention/MLP path tracing | moderate; extends §44 single-position patching | evidence-carrying components without verbal intermediates | sufficiency vs necessity; many comparisons |
| SAE / attribution graphs | high | — | expensive, approximate; not the next tool |

## 4. Recommendation

A focused tool audit **is** justified before any pivot, and it is cheap. Proposed order, each piece $0 on the laptop,
each with a kill criterion, nothing launched without the user's word (D-45/D-49: §45 was the one opened line; this is
a new line and needs its own decision and proposal):

1. **Record §1 as an exploratory post-hoc note** in the §45 text (dev only; verdicts unchanged) and soften the lens
   sentence as in §1c. No run.
2. **Instrument check on the controlled RC/TU prompts (the §45 case list, dev first, held read once):** (a) a
   contrastive probe for the supervisor / current-city variable at the entity positions and the last prompt token,
   with a Hewitt–Liang control task; (b) one interchange intervention at the supervisor-name position between two
   solved prompts, with the §44 difference-matched controls, success = the output city follows the donor on ≥ 0.70 of
   pairs (lower bound ≥ 0.55), unseen name/city combinations held out. Decision rule fixed in advance: probe decodes
   mid-stack at entity positions AND the intervention moves the answer → "instrument blind spot" (expected); neither
   → "route late, lenses adequate"; probe yes / intervention no → "represented, not used by the route" (the Walters
   et al. framing).
3. **Only if 2 finds a movable variable:** the bridge to the §44 rev 8 rows — is the same variable decodable on the
   long-context failures, and does moving it change refuse vs answer. This, not 2, is the thesis test; 2 alone
   reproduces Feng & Steinhardt on a smaller model.

What I would not do: DAS or SAEs before 2 says there is something to align; any Patchscopes without a shuffled-source
null; any claim of a "relation-sensitive lens" contribution until 3 shows the existing lenses miss something that
causally matters on the real rows (Codex's point 6, agreed, conditional).

**Bottom line for the user.** §45 does not say the model computes nothing early; it says three vocabulary readouts see
nothing early, and this audit shows that holds at every position and below the top-10. The literature predicts the
relevant variable lives at the entity positions in non-vocabulary directions, so the right next instrument is a probe
plus an interchange intervention there, not a better lens. That is a two-day laptop study, $0, and it decides
"instrument or route" before any thesis pivot is discussed.
