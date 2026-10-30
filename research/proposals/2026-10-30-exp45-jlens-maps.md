# §45 proposal (rev 0): descriptive J-lens maps of answer computation on controlled tasks, Qwen 3B

**Author:** Advisor, 2026-10-30. **Order:** the user, Advisor pane: "amend D-45, draft §45 after the 7B lands" (D-49),
then "Can you prepare 45 to be run on a pod in parallel with 44" (the draft is brought forward; the build runs beside
the §44 pod work; the §45 pod runs after the §44 pod has terminated — never two pods at once).
**Status:** pre-registered, exploratory and descriptive. No causal test is in this proposal. **Source:** the Codex
handoff of 2026-10-30 and the Advisor's review `research/briefings/2026-10-30-jlens-mapping-direction.md`.

## 1. Question

Why does a model abstain or answer wrongly with sufficient evidence in context? §34–§44 read one token at one forced
position. §45 asks what the lenses show about *how the answer computation develops across layers and tokens* on tasks
whose intermediate steps are known by construction: direct extraction (no intermediate), relational composition (one
named intermediate), temporal updating (one superseded competitor). Readable intermediates on solved cases come
first; matched failures second; the §44 rows third. The output is maps, not claims about erased thoughts.

Three things are kept apart throughout: the model's answer accuracy, the lens's recovery of an expected
intermediate, and (later, not here) causal intervention success. Expected intermediates are hypotheses about the
computation; a missing readout is "not recovered by this lens", never "not computed".

## 2. Instrument, fixed

- **Model.** Qwen2.5-3B-Instruct, bf16 on the pod (the §34/§43 pod reads were bf16; the laptop dev reads fp32).
- **Lenses.** (i) The §34 J-lens, 56 prompts, sha `ddbdb32d…de21b`, unchanged, for comparability with §34/§43.
  (ii) **A second J-lens refitted on more WikiText prompts** (target 128, minimum 96, same corpus, seed 1, same
  settings as §34, time box 75 min on the A40; §34 measured 34.5 s per prompt for the 3B). (iii) The logit lens.
  Every map carries all three; the refit addresses the "fit stability" question directly instead of by argument.
  If the refit closes below 96 prompts it is reported as such and (i) and (iii) carry the maps.
- **Readout.** At every layer 0…L and at every position from the first prompt token to the last generated token:
  the lens top-10 tokens, and the rank and log-probability of each pinned token. Pinned tokens per case: the first
  content token (after any leading whitespace or determiner, the §44 lesson: Qwen splits numbers into digits and
  many names start with " The") of every entity in the case, of the gold, of every competitor, and ` unknown`.
  Ranks 1-based, strictly-greater ties, as §34. No forced "Answer:" position and no fixed layer band: the model's
  own chat template, its own generated response (greedy, ≤ 16 new tokens), and the maps run over all of it.
- **Deliverable per case.** One `.npz` (ranks and top-10 ids, layers × positions × lenses) and one `.json`
  (prompt, tokens, generated text, outcome, pinned ids, lens hashes); one rendered map file per case (layers ×
  positions heat map of the pinned tokens' ranks, with the top-10 strip at the final prompt token and each generated
  token) — plain HTML, no server. A summary report per stage.

## 3. Cases, fixed by template, pool and seed (the instantiated list is committed before any model run)

Three task types × 12 base cases = 36 base cases. Each base case has four **variants**: v0 base; v1 factual
substitution (one entity replaced by another from the pool); v2 question change (ask for a different slot of the same
facts); v3 order change (the fact sentences in a different order). Each variant appears in two **formats**: F1 plain
chat (the facts, a blank line, the question) and F2 the §23 memory-note wrapper used in every earlier section (the
facts as memory lines, the same question), so the format's effect is in the maps. Each (variant, format) appears at
three **hardness** levels: H0 bare; H1 plus 3 distractor facts about other pool people (other names, other cities,
other jobs), shuffled; H2 plus 10. Total 36 × 4 × 2 × 3 = **864 prompts**. The pools (12 first names, 12 UK cities,
12 jobs, 12 months) and the templates below are the design; `scripts/exp45_cases.py --seed 45` instantiates them
and writes `runs/exp45_maps/cases.jsonl`; the Advisor reviews that file (counts, no duplicate gold, every
competitor distinct from the gold, every pinned token distinct in its first content token) before the run.

- **Direct extraction (DE).** "{A} lives in {C1}." Q "Where does {A} live?" Gold C1. v1: C1 → C2. v2: "Who lives in
  {C1}?" gold A. v3: a second fact "{B} works as a {J}." placed before / after.
- **Relational composition (RC).** "{A}'s supervisor is {B}. {B} works in {C1}." Q "In which city does {A}'s supervisor
  work?" Gold C1; **intermediate B**. v1: B → B′ (with "{B′} works in {C1}"). v2: "Who is {A}'s supervisor?" gold B
  (one hop; the control for whether B is produced when asked directly). v3: the two sentences swapped.
- **Temporal updating (TU).** "{A} lived in {C1}. Later, {A} moved to {C2}." Q "Where does {A} live now?" Gold C2;
  **competitor C1**. v1: C1 ↔ C2 swapped (gold becomes the other city). v2: "Where did {A} live before moving?" gold
  C1. v3: the update stated first ("{A} now lives in {C2}. Before that, {A} lived in {C1}.").

Dev / held split for every criterion below: base cases with an even index are the dev half, odd the held half.
The criteria are fixed here; the dev half is read first and may adjust *presentation* only (colour scales, which
strips to show), never the criteria or the case list.

## 4. Stages and what each reports

- **Stage 1, solved cases.** Outcome per prompt (gold / competitor / other / abstain, by the §23 normalised match).
  **Readability criterion (fixed):** an expected intermediate is *readable* on a prompt if its first content token
  is in the lens top-10 at the last prompt token or at any generated token, at any layer; the layer and position of
  first readability are recorded. Per task type and lens: the share of solved prompts on which (a) the gold is
  readable before the final layer, (b) the intermediate B (RC) is readable, (c) the competitor C1 (TU) is readable,
  with Wilson intervals; the layer of first readability as quartiles over depth. **Fit stability:** agreement of the
  readability call between the 56-prompt lens and the refit lens per (prompt, pinned token), and the rank
  correlation of the two lenses' pinned-token trajectories; the logit lens beside.
- **Stage 2, matched failures.** Every prompt the model gets wrong or abstains on, paired with the solved prompt of
  the same base case and variant at the lowest hardness; the same tables on the failures; the per-pair map
  difference (where the gold's trajectory departs). All failures kept, including ambiguous ones (reported as such).
  If fewer than 10 failures occur at H2 the stage is reported as "too few failures on these tasks" and nothing is
  added to induce more.
- **Stage 3, the §44 rows.** The 70 rev 8 rows of Qwen 3B (`runs/exp44_swapgate/rev8/eligible_qwen3b.jsonl`),
  clean and swap prompts, with the model's own generation: pinned tokens gold, substitute, ` unknown`; maps over the
  evidence line, the question, the last prompt token and the generated tokens; the §23 label, the §44 class and d_T
  beside. Descriptive; it shows what the lenses see on the failures the thesis is about.

## 5. What settles what (fixed now)

- Stage 1 readability (a)–(c) ≥ 0.70 on solved prompts with the lower bound ≥ 0.55, for a lens: "this lens recovers
  that intermediate on this task type". Below: "not recovered by this lens on this task type" — nothing about the
  model. The two J-lenses disagreeing on ≥ 0.20 of calls: "the 56-prompt lens is not stable", stated in F11's lens
  caveat.
- Stage 2 is descriptive. The only pre-registered sentence it can earn: "on the matched failures the gold is / is
  not readable before the final layer on ≥ 0.70 of them (lens named)". No "erased", "suppressed" or "overridden".
- Stage 3 is descriptive. No criterion.
- **No causal test.** If a specific intermediate or transition is readable and a specific failure departs from it at
  a specific layer and position, that is the hypothesis for a later proposal (§46), written after these maps exist.

## 6. Pod plan and cost (one A40, $0.59/h, after the §44 pod has terminated; never two pods)

Install → download 3B bf16 → stub check → **smoke**: 3 cases through the full map path with both lenses (lens
shape check, finite ranks, BOS count, the dev final-layer top-1 equal to a laptop fp32 read on the same 3 cases) →
**stage A refit** (75-min box) → **stage B maps** (864 prompts, ~1–2 s each ≈ 20–30 min; then the §44 rows, 140 long
prompts ≈ 10 min) → copy back → terminate → list-pods empty. Rendering and reports on the laptop.
**Quote: about $1.20 (≈ 2 h); hard stop $2.50 (≈ 4.2 h).** HARD_STOP_S 14400, SELF_STOP_S 15000, a laptop DELETE
after. If the refit box is cut by the deadline the maps still run with lenses (i) and (iii). Programme after this
≈ $22.3 of $30. The user's approval line is needed in both panes before the pod is created.
Laptop fallback: everything except the refit runs on the laptop in fp32 (~4–6 h), $0.

## 7. Build (the Executor, beside the §44 pod work; nothing runs before the Advisor's build review)

`scripts/exp45_cases.py` (templates, pools, seed 45, dev/held parity, writes `cases.jsonl` + counts),
`memctl/analysis/maps.py` (one forward with hidden states + greedy generation; lens readouts at every layer and
position for the three lenses; pinned-token ranks; outcome by the §23 match), `scripts/exp45_maps.py` (resumable per
prompt, `--stage 1|3`, device auto, outputs under `runs/exp45_maps/<lens-set>/`), `scripts/exp45_render.py` (HTML
maps), `scripts/exp45_report.py` (stage tables, Wilson intervals, lens agreement), `scripts/exp45_pod.sh` (the §43/§44
pod pattern: preflight, install, GPU check, watchdog, downloads, smoke, refit box, maps, pod.done, stage_times).
Tests: templates instantiate without duplicate golds; first-content-token rule on digits and determiners; readability
call on a synthetic rank table; the §23 match on the four outcome classes; stub run end to end (D-30 fresh venv).

## 8. Caveats fixed in advance

Toy prompts are not long-context memory; stage 3 is the bridge. The 56-prompt lens failed §34's library gate; the
refit may or may not pass it (the §34 eval runs again on the refit lens, reported, not a gate here). bf16 on the pod
vs fp32 on the laptop: the smoke pins the final-layer top-1 on 3 cases. Readability is a top-10 statement about a
linear readout, not evidence of use. Greedy decoding only. One model, one family; nothing here generalises.
