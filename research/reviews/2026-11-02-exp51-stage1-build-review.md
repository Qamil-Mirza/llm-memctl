# §51 stage-1 build review (Advisor, 2026-11-02)

Build under review: commit 70c5e07, plus the Executor's interim report of 2026-11-02 (B5 fit still running).
Contract: proposal rev 1a (`research/proposals/2026-11-02-exp51-agent-weight-editing.md`, 15ec3ee), §13 binding.
Nothing was spent, and no API call was made. No H or T output exists.

**Verdict: ACCEPT WITH REQUIRED CHANGES.**
- F0 waits on: items 1, 2, 4, 9 and 10 below; the B5 result; and the user's decision on compute (item 8).
- The dry run (stage 2) waits on item 9.

## What I checked independently

- **B1 token set.** I re-ran the §5.2 rule on the Qwen2.5-3B tokenizer: every token whose decoded, lower-cased,
  stripped form begins "unknown". It gives exactly the Executor's 6 ids. Sub-word spellings ("unk", " unknow") are
  outside the rule as written, so this is correct as specified, and the gap is disclosed.
- **T-city pool.** It is disjoint from the 12-city pool in `exp45_cases.CITIES`. The aliases cover only gold and pool
  cities, plus Londres for London.
- **Inventory arithmetic.** The stage-4 rows sum to 320.0 CPU h with the shortcut excluded. The shortcut replaces
  79.0 h (B1 online 47.4 + B4 31.6) with 29.2 h.
- **Harness API body** (`exp51_harness.ApiClient`), checked against the current Claude API reference:
  - `output_config.effort = "high"` is correct for Opus 5.5;
  - no `thinking` field is sent, which is correct, since thinking is always on;
  - assistant content, thinking blocks included, is appended verbatim, which preserves the thinking binding;
  - cache breakpoints are on system, tools and the last message (3 of 4);
  - the 1-hour TTL syntax is correct;
  - prices are $4 / $20 per MTok, $0.20 for cache reads and $8 for 1-hour writes, and match the reference;
  - the worst-case pre-call stop is conservative.

## Rulings

1. **No-op gate: ACCEPT, with a rerun.** Before F0, rerun it with at least 8 RC and 8 TU rows, once their B0 exists.
   The same zero-mismatch criterion applies.

2. **Feasibility (r = 0.958): DROP T-FORMAT** (pending the user's acknowledgement).
   - §10 names the transfer sets as secondary and first to drop.
   - Dropping T-format restores E 48, S 32 and H 48 per task. It also removes 40.6 CPU h.
   - This is recorded as a pre-F0 amendment, not as a uniform reduction under §13.2.
   - T-city stays at 24 per task: it uses its own pool, so it does not compete for keys.
   - Regenerate `cases51.jsonl` and re-hash. Nothing in H or T has been read, so this is not a redraw after outcomes.
   - The T-format wording itself was acceptable: a paraphrase that keeps "reply: unknown".

3. **B1 grids: ACCEPT.**
   - The R grid is as specified.
   - The M3b signed grid ±{0.5, 1, 2, 3, 4, 6, 8, 16} meets "8 values each side".
   - The B4 grid of α {0.5, 1, 2, 4} × v_29, with the §38 file sha, is accepted.

4. **Grader additions: ACCEPT, with fixtures.** The added cues are:
   - "moved from / formerly / previously / used to" (negation);
   - "not sure / uncertain" and "not for her/him/them" (abstention);
   - "is mentioned / mentions / whether" (doubt).

   Each one needs its own fixture before F0. Add at least:
   - "The memory mentions York. York." → York;
   - "She used to live in Leeds; now York." → York;
   - "Not sure." → abstention.

   On the 14 §50 dev rows with no strict marker:
   - they are NM, reported separately, as §4.1 says;
   - list them in the build notes.

5. **Strict marker and NM handling: ACCEPT.** The 3/1,440 inline " Answer:" cases are disclosed.

6. **Base-weighted rates: ACCEPT.** They are declared at F0. This is consistent with the base-level bootstrap and the
   U weights.

7. **B5: ACCEPT, with a recorded deviation.**
   - The all-36-layer `o_proj` + `down_proj` set (Arditi's set) is a rule frozen a priori, so it satisfies "frozen
     from E".
   - The direction layer = argmax of in-sample E AUROC is accepted.
   - **Deviation:** §5.2 says "B0-answered"; the build uses gold-answered rows. Record it in the F0 manifest as a
     deviation, with the reason: wrong answers would mix error with commitment. Do not refit.
   - B5 is exempt from the candidate limits as a comparator: correct.

8. **Compute. The user decides under §12.3; my recommendation follows.**
   - **(c) B0 prefix plus triggered continuation for B1/B4 online: APPROVED** as estimand-preserving.
     - Condition: a parity test on 24 mixed E rows. Run B1 at −4 and B4 at α 2, both as full online generation and as
       the shortcut. Every row must match token for token; any mismatch blocks the shortcut.
     - Saving: about 50 h on S, and proportionally on H.
   - **(d) Batched decoding: measure only, at $0.**
     - On the laptop, on E rows, at batch 8 and 16, report rows/h and agreement with batch-1 (token-level, and the
       graded class).
     - No confirmatory use until the user rules. If it is used, every arm on a split uses the same backend and the
       same fixed batch layout.
   - **GPU quote** (no pod): an inventory-based quote for S and H after (c) and the T-format drop:
     - an A40 at batch 1 vs batched, fp32;
     - including the §10 CPU↔pod replay check, the hard stop and the D-22/D-30 steps.

     Remaining envelope: about $5.8 of the $30 cap.
   - **(b) Grid cuts: hold.** If needed, the order is:
     1. R-candidate doses to {0.5, 1, 2};
     2. B5 to 3 doses;
     3. secondary analyses.

     ANS+UNA-only for the R arms would drop CF-follow from the R report. Disclose it if chosen.

9. **Harness: MUST FIX before the dry run.**
   - **a. `stop_reason`.**
     - On `max_tokens`, never execute a truncated `tool_use`. Return an error `tool_result` asking for a shorter call.
     - On `refusal`, log it and stop the run, with the reason recorded.
     - Validate each tool input against its schema before running it.
   - **b. Transient API errors** (429, 5xx, 529, timeouts):
     - retry with backoff; the attempt is ledgered as an API retry, not as an editor failed request;
     - stop after 5 consecutive failures;
     - today, an exception ends the run, and that run counts.
   - **c. Optional:** stream responses, given `max_tokens` 16,000 and the 900 s timeout.
   - The dry run must also confirm two things:
     - that `usage` reports the 1-hour cache-write breakdown;
     - that the meter matches the Console's figure within a few cents.

10. **Initial instruction: ACCEPT.**
    - It names no sufficiency, confidence, B/C, steering or result numbers.
    - The sentence describing note-then-"unknown" defines the target behaviour, which is visible in E. It is the task
      definition, not a mechanism claim.
    - Freeze it verbatim at F0.

11. **Accepted as built, frozen at F0:**
    - Effective key and TU evidence city. The old city is not in the key; old-city overlaps are disclosed as partial
      overlap.
    - Contiguous key pre-allocation.
    - Redraw until a UNA-swap entity exists.
    - The T-city pool.
    - The B2 prefix and straddling-token handling. Add a unit test for the straddle case.
    - The B3 prefix.
    - Candidate doses δ × {0.5, 1, 1.5, 2}. The proposal named no candidate grid, so this fills that gap.
    - The edit arithmetic: float64 ΔW, rounded once, sha-checked on every apply, with whole-tensor scaling refused at
      |cos| > 0.999.
    - The engine's online trigger and hybrid definition.
    - M2 hooks refused after P_ans, which matches §13.3.
    - B6 matching, at seeds 1–5.
    - Service caching of unedited arms from B0.
    - Sandbox probes.
    - Dropping template id from the nuisance model, because it encodes the answerability relation.

## Order of work

1. Finish B5, then run B0 on RC/TU.
2. Rerun the mixed no-op gate.
3. Regenerate the cases without T-format.
4. Add the grader fixtures and the harness fixes.
5. Run the shortcut parity test and the batching measurement.
6. Prepare the GPU quote.
7. Advisor re-check of the deltas.
8. User decision on compute.
9. F0 freeze, Advisor-committed.
10. Dry run within the editor cap.
11. F1.

Throughout: no API call before F0, no pod, and no H or T outputs.

## Addendum: user decision (2026-11-02)

The user's words, in the Advisor pane: "yes drop T-format". Ruling 2 is final. T-format is removed before F0, and it is recorded as a pre-F0 amendment in the F0 manifest.

## Re-check of the build-review deltas (Advisor, 2026-11-02)

Commits a93ca57, 172442c, 2689dc7. Recounted from the raw files in `runs/exp51/`:

- `cases51.jsonl`: sha 5eeb7c40…a8501, 5,928 rows. Old list kept as `superseded/cases51_r0958.jsonl` (6,396 rows).
- No-op gate: 24/24 mixed rows (DE 9, RC 9, TU 6) plus 9/9 TU top-up, every check true. RC 9 and TU 15 meet ruling 1. **PASS.**
- Shortcut parity: 48/48 token-identical (DE 18, RC 18, TU 12 arm-rows), no row without a marker, trigger step equal to B0's on every row. **PASS**; (c) is approved for confirmatory use.
- Batching on CPU: b8 and b16 are each 24/24 token- and class-identical to batch-1. The gain is small (153 → 202 rows/h).
- B0 directories hold E only. No H or T output exists.
- GPU quote: the items sum to 10.62 h, ×1.3 = 13.8 h run + fixed = 14.1 h. The arithmetic checks.
  - The file is written at $0.35/h (community). Record the secure $0.59/h figures ($8.34, stop $9.59) in it too.

**Verdict on the deltas: ACCEPT.** Rulings 1–7 and 9 are met. Ruling 8 waits on the measurement and on the user.

### Rule fixed before any GPU measurement (so the result can't steer it)

- Every confirmatory arm on a split runs on one backend with one fixed batch layout, B0 included. No split mixes CPU and GPU arms.
- GPU batched vs CPU batch-1 on the 64 E rows (the §10 replay preview):
  - graded-class agreement ≥ 62/64 → the batched GPU route is usable; token mismatches are disclosed;
  - below that → the batched route is not used for confirmatory work.
- Measured unit rates go into the quote as they come out. If the measured total at the live price exceeds the money left, nothing runs until the user decides (cut, cap raise, or a cheaper host).

### B5 fit re-check (Advisor, 2026-11-03)

Recounted from `runs/exp51/run/b0/E/*.json` and `b5_fit.json`:
- 864 E ANS rows with B0; 857 with P_ans, 7 NM.
- Classes per task match the Executor's report. The fit uses 284 abstained (192 + 69 + 23) and 572 gold-answered (91 + 219 + 262) rows.
- ℓ* = 34 is the argmax of in-sample AUROC (0.9397; L33 0.9394). Chosen by the frozen rule, no refit.
- The direction sha a8282ec8… reproduces with `edit51.sha_array` (it hashes dtype + shape + bytes, so it differs from the plain file sha).
- 4 comparator edits at β 0.25–1.0. No B5 arm has run on any split.
- E support (bases with ≥ 1 false abstention, out of 48): DE 44, RC 35, TU 15. TU is below the 20-base minimum, as §3 anticipated.

**ACCEPT.** Deviation D1 is recorded in the F0 draft.

### Stage-4 pod structure ruling (Advisor, 2026-11-03)

- **Two pods (Executor option 1): ACCEPT.** F2 needs the Advisor's S review and a git commit, so under D-73's own clause
  S and H/T run as separate pods.
  - Pod S + replay: $5.81, stop $6.68.
  - Pod H + T-city: $3.92, stop $4.51.
  - The worst case at both stops is about $24.21 + $11.19 = $35.40, under the $36 cap.
  - Each pod is quoted and approved on its own, with the user's line in the Executor pane.
- **Isolation.** The S bundle carries no H or T-city rows. The H+T pod runs `require_f2` (the git-verified guard) before
  any sealed row is read.
- **Option 2 (wait in pod): rejected.** It bills while idle, and a review longer than the wait forces a second pod anyway.
- **§8.4.2 harness replay of submissions on E: laptop CPU at batch 1, $0.** This keeps E on one backend. GPU replay on E
  is not used.
- **Still to come:** the stage-4 chain, which needs a build review before its D-22/D-30 tests. Capacity retries follow
  the timing-pod pattern: the approved card and price only, and a create refuses if any pod exists.

### Stage-4 chain build review (Advisor, 2026-11-03; commit 4f24fb7)

I read `scripts/exp51_stage4.py` (selection, follow-ups, replay, h-run), the bundle and pod scripts, and the
`stats51` / `mech51` rule functions. I re-ran the tests: 35 passed.

**Checked and correct.**
- **Dose rules:** max U1, with ties going to the setting nearest no-op; B5 by β.
- **R entry:** lower bound > 0, and ≥ 20 FA bases in DE or RC.
- **R candidate:** highest U1, then fewer changed params, then index.
- **M dose:** largest oriented effect with Deg ≤ 0.10.
- **M eligibility and ranking:** oriented ≥ 0.05, bound beyond 0, support ≥ 20; ranked by oriented / max(SE, 0.01);
  ties to components, then params, then index (§13.1).
- **Primary component:** the revert with the largest oriented removal; ties to the editor's named component, then
  index.
- **M2 hooks** at P_ans only (§13.3).
- **Equivalence margins:** 2 × SD of per-base mean Δf.
- **Predictor:** both classes required overall; encoding at task × hardness for small cells (§13.4c).
- **Isolation:**
  - the S bundle filters to E + S and asserts no H/T-city rows;
  - the H+T bundle refuses without the guard;
  - the pod H+T guard exits 4 (PIPESTATUS checked);
  - `h_run` itself calls `require_design` before `CM.rows("H", allow_sealed=True)`;
  - the D-22 smoke on pod H+T uses S rows only.
- **Replay paths:** they match the laptop `e-replay` naming (`EDIT_<id>_d1`).

**Required fix (before F0).** §13.4(b) is hard-coded to FA/UNA. The f-intervention grid, the B1 signed grid and the H
Commit interaction all use FA and UNA only.
- If the M candidate's declared population is not FA (e.g. C), §13.4(b) says the population pair is declared at F2,
  with UNA as the default second population.
- Run the f-int and signed-B1 grids on the declared endpoint population ∪ the second population. Write the pair into
  `f2_candidate.json`. Read it from F2 in `h_report`.
- Add a test with a C-population endpoint.

**Rulings on the Executor's questions.**
- **(a) Structured submit fields** (f_site, f_direction .npy with sha, f_int_sign, ≤ 4 activity entries from the enum,
  enum endpoint): ACCEPT.
  - The submit tool's schema and every field description are frozen verbatim at F0 and hashed, because the editor reads
    them as part of its instructions.
  - They are held to the instruction's rule: no mention of sufficiency, confidence, B/C, steering or results.
  - The instruction text stays unchanged.
- **(b) Manipulations:** ACCEPT as defined. Confirm in the docstring that ANS-a H1 → H2 changes only the distractor
  lines, with the same base, rep and evidence.
- **(c) f-int doses** = f_int_sign × {0.5, 1, 2, 4} × SD of f over S ANS rows with a P_ans: ACCEPT.
  - **Replay gate:** class agreement ≥ 62/64 scaled (≥ 194/200) in all four comparisons: ACCEPT.
  - Under §10, every mismatch is listed and investigated by the Advisor before S results are used, even when the gate
    passes.
- **(d) "One edit" = submission 1 at dose 1:** ACCEPT.

**Verdict: ACCEPT WITH ONE REQUIRED FIX** (the §13.4(b) population pair). D-22 and D-30 for each pod follow once real
submissions exist.

### Fix check and ruling (b) correction (Advisor, 2026-11-03; commit f806c9a)

- **§13.4(b) pair fix: ACCEPT.** `interaction_pair` gives [declared, UNA], or [UNA, FA] when UNA is declared. Both grids
  run on the pair rows, the pair goes into F2, and `h_report` reads it. A C-endpoint test is included.
- **distractors_added:** keep as built. No generator change.
  - H1 → H2 adds 7 distractor lines. The dates shift with line position: the question date in 144/144 pairs, the
    evidence date in 129/144.
  - Name it as what it is, "more distractor lines, with the date shifts that follow from position", in the docstring,
    the F0 manifest, and any result wording.
  - A distractor-only contrast would be a new row type added only to purify one optional activity test. That is not
    worth a pre-F0 generator change.
- **Required (before F0).** The submit tool shows the editor only the enum names. Add a neutral definition of each
  manipulation to the `activity.manipulation` field description, as the pairs are actually computed:
  - evidence_removed = UNA-rel − ANS-a;
  - city_changed = CF − ANS-a;
  - distractors_added = ANS-a H2 − H1, with the date shifts;
  - unrelated_vs_absent = UNA-swap − UNA-rel at H1;
  - each per base, averaged over the hardness levels present.

  The editor's predictions must refer to defined contrasts. The text is frozen with the tool schema at F0.

- **310f4db: ACCEPT.** The manipulation definitions are in the submit tool (`MANIP_DOC`), frozen with tools_sha256. The
  word check is clean.
  - The UNA-rel wording is corrected to "replaced", not "removed": the relation's lines are replaced with lines about
    another person, keeping the same line count.
  - No stage-4 items remain open before F0.

## F0 freeze review (Advisor, 2026-11-03; draft sha a7cff9ae…, HEAD d0e5864)

Recounted and checked from the raw files:
- 24/24 `sources_sha256` match the working tree.
- The instruction and system text equal `exp51_harness` verbatim (instruction sha eccf4e48).
- `tools_verbatim` equals `TOOLS`. tools_sha256 f5436c0c = sha of `canonical(TOOLS)`.
- The word check on the instruction, system and tools is clean.
- Cases sha 5eeb7c40.
- 38/38 grader fixtures ok.
- E B0: 1,872/1,872 records. The class counts per set match `e_b0.by_set_class`.
- B5 block as accepted.

**Verdict: ACCEPT, after two pre-freeze items.**
1. **Instruction (e).** Change "under manipulations you specify" to "under the manipulations defined in the submit
   tool". Only those four are testable at M3a, so the editor should not be invited to make untestable ones. This is a
   text change before F0; re-hash.
2. **Dry-run spend controls.** The harness stop is $45, which is the whole editor budget. The API dry run needs:
   - its own `--stop-usd` (proposed $3) and a turn cap (proposed 10);
   - the dummy model and synthetic DRY- rows only;
   - an estimate from the per-call worst case.

   The user approves the dry run's price and stop before the API key is used.

**Disclosure to add.** B0 CF-follow on E is 282/432 = 0.65. The §8.1 repair criterion (CF-follow ≥ 0.90) is therefore a
high bar: a repair has to raise CF-follow, not only keep it. The criterion is pre-registered and unchanged.

**Commit.** The F0 commit (`git add -f` of the final draft, by explicit path) is made by the Advisor once the user says
to commit.
