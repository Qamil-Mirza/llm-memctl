# Research gap check: gap, importance, decisive evidence (Advisor, 2026-11-05)

The thesis question, with §51 as the current lead, checked against three questions the user set: what is the gap,
why closing it matters, and what evidence would convincingly answer it. The §51 numbers below come from the editor's
claims and the Executor's early look at submission 1 on E (`runs/exp51/run/early_look_sub1_NOT_8.4.2.json`). **E is
the split the editor saw. Nothing here is a §51 result**, which waits for the §8.4.2 replay, e-entry, S, F2 and H/T.

## 1. What is the gap?

**Established outside this project:**
- **Models fail to use sufficient context:** they answer wrong or abstain even when the context is enough (Joren et
  al. 2025, arXiv:2411.06037).
- **Refusal and abstention can be steered with single directions:**
  - safety refusal (Arditi et al. 2024, arXiv:2406.11717);
  - unanswerability (Lavi et al. 2025);
  - conditional, input-dependent steering (CAST, Lee et al. 2025, arXiv:2409.05907).
- **Confidence causally drives abstention** (Kumaran et al. 2026, arXiv:2603.22161).
- **Fact storage can be located and edited in specific weights** (ROME/MEMIT), and **interpretability agents** can
  describe what components do (MAIA; Transluce's tools).

**Established in this project** (STATE.md):
- **F11:** the answer is usually present inside the model at failure, in four families.
- **F6:** many failures are a refusal that one instruction line or one steering push releases.
- **F16:** on controlled tasks, most false "unknown"s are "note, then refuse".
- **No push tried is selective** (F6, F16).
- **F17:** self-reported sufficiency is not a usable measure.
- **F12–F14:** lenses are blind mid-stack, and patching located where the answer is usable, not which part decides
  to refuse.

**Open:**
1. **Which components produce a false "unknown"** when the evidence is in view. Prior work gives control directions,
   not component-level localisation of this failure.
2. **A selective fix:** one that releases false "unknown"s while keeping correct abstentions. Nothing tested, here or
   in the cited work, does this well.
3. **Whether agent-proposed weight edits survive pre-registered, sealed confirmation.** Interpretability agents mostly
   describe components. Agent-proposed edits confirmed on sealed splits appear less studied. **No fresh literature
   search has been done on this point; do one before claiming novelty.**

**Why existing approaches fall short:**
- Directions are global pushes, so they move answerable and unanswerable rows alike.
- Vocabulary lenses are blind mid-stack.
- Patching shows where information flows, not where the decision is made.

## 2. Why does closing it matter?

- **Science:**
  - It separates "can't read the evidence" from "read it and won't say it", at component level.
  - It tests whether over-abstention is a small local circuit or spread across the network.
- **Capability:**
  - A fixed weight edit has no run-time cost. If it is selective, it beats prompts and steering.
  - A confirmed agent-guided editing method would carry over to other failures.
- **Practice:**
  - Small models are deployed cheaply and on-device in retrieval and memory assistants. A false "I don't know" with
    the answer retrieved wastes the pipeline.
  - Abstention is also a safety behaviour, so the fix must be selective.
- **Who benefits:** builders of retrieval and memory systems, people deploying small models, interpretability
  researchers, and builders of automated-research agents.
- **Limit:**
  - This is one 3B model, and the 7B gains only +0.02 from the §35 line.
  - The scale question is open: §52 below 3B, and a check above 3B is not yet planned.

## 3. What evidence would convincingly answer it?

**Hypothesis.** In Qwen 2.5 3B, false abstention with sufficient evidence is driven substantially by two late
attention heads, L32h2 and L33h10:
- removing their output releases the answer;
- amplifying it produces abstention;
- and this is more selective than generic "answer more" pushes.

**Alternatives and tests:**

| Alternative | Test | Status |
|---|---|---|
| The edit fits E only | sealed S, then H and T-city / T-format | §51, pre-registered |
| It is another "answer more" knob | B1–B5 on the same rows, U(λ) trade-off | §51, pre-registered |
| Any change to those weights would do | B6, a matched random edit on the same entries | §51, pre-registered |
| Removing **any** two late heads would do | random other heads at matched layers | **not planned; §53 draft** |
| It works only on the templates | natural LongMemEval rows | **not planned; §53 draft** |
| The heads copy the prompt's "unknown" option | attention by token class; head-output patching; the effect with the option removed | **not planned; §53 draft** |

**Supporting outcomes:**
- §51 confirmation on H;
- a better U trade-off than B1–B5;
- the amplification direction holding out of sample;
- the target pair beating random late-head pairs;
- partial transfer to LongMemEval.

**Contradicting outcomes:**
- S or H failure (overfitting);
- a trade-off no better than the comparators (another knob);
- random late-head pairs doing as well (general disruption).

**What would change the conclusions:**
- **The heads attend mainly to the "unknown" option text.** Then the mechanism is prompt-option copying, not an
  evidence judgement, which ties back to the prompt-format caveat.
- **Template-only effects.** Then the mechanism is specific to the templates.
- **Analogous heads in other families** (§52 / §43 models). Then it is general.

## Upshot

- **§51's frozen design already covers** the two largest alternatives (overfitting; "just another knob").
- **Two cheap controls are missing:** random other heads, and what the heads attend to.
- **They go in a separate follow-up**, `research/proposals/2026-11-05-exp53-head-controls.md`, which opens only after
  §51 confirms. Adding them to §51 now would break its freeze.
