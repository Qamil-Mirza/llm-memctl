# Fact check for the Medium article: two failure examples and the four equations (2026-10-30)

Advisor note for the user and Codex. Read-only: nothing was re-run; every number below was read from the files named.
Row files: `runs/exp23_verdicts.json`, `cache/generations_exp23_<reader>/`, `runs/exp35/qwen3b/<arm>/fold<f>.jsonl`,
`runs/exp34_jlens*/qwen3b/<qid>.json`. The loader is `memctl/jacobian_data.py:load_records` (same as §34/§35).

## 1. Two failure examples (verbatim, publication-suitable)

Both are LongMemEval-S questions from §23's `fixed8__fraction0.05` arm (the controller shows the reader 8 memory lines;
the needed evidence is in view). Reader: Qwen2.5-3B-Instruct, greedy, the §23 note-then-answer prompt. The prompt's
instruction, verbatim from the cached prompt: *"Answer with a short phrase that states the fact asked for. If the memory
does not contain the answer, reply: unknown ... First write a brief note of at most 80 words ... Reply unknown only if
the memory has nothing at all on the topic. End with one line of the form 'Answer: <short phrase>'."* Labels are the §23
judge (Qwen2.5-7B-Instruct, official LongMemEval per-type prompts). Both rows sit on confirmation folds of experiments
that are complete and were read once (§34, §35), so publishing them leaks nothing.

### Example A: the answer is in the model's own note, and it still refuses

| field | value |
|---|---|
| question id | `1e043500` (fold 4, type single-session-user) |
| question | (asked on 2023/05/30 (Tue) 19:19) What is the name of the playlist I created on Spotify? |
| gold | Summer Vibes |
| the evidence line in view (one of 8 memory lines) | user, 2023/05/21 (Sun) 13:13: What's a good music streaming service that can help me discover new music and artists, especially in the ambient and lo-fi genres? Also, by the way, I've been listening to this one playlist on Spotify that I created, called Summer Vibes, and it's got all these chill tracks that are perfect for relaxing or working out. |
| Qwen2.5-3B output (cached §23 run, verbatim) | `user mentioned creating a playlist called "Summer Vibes" on Spotify.` ⏎ `Answer: unknown` |
| §23 verdict | wrong, refusal (`unknown: true`) |
| Qwen2.5-7B output on the identical prompt | `Fact: The user mentioned creating a playlist on Spotify called "Summer Vibes".` ⏎ ⏎ `Answer: Summer Vibes` (correct; the 14B, Granite 2B and Granite 8B are also correct on this row) |
| §35 arm A: the 3B's own note frozen + `Answer:` | ` unknown` (judged wrong) |
| §35 arm B: same note + "Based on the note above, give the answer as a short phrase. Do not reply unknown." + `Answer:` | ` Summer Vibes` (judged correct) |
| §35 arm F: prompt with the two "unknown" clauses rewritten, regenerated | note identical to the cached one, then `Answer: Summer Vibes` (correct) |
| §34 lens, forced answer position (logit lens, k = 10) | the gold token ` Summer` reaches rank 1 at some layer and is rank 2 at the final layer; ` unknown` is rank 1 at the final layer (J-lens agrees: 1 / 2 / 1). At the model's own answer position (after its note, §34b): gold final rank 3, ` unknown` rank 1 |

Artifacts: `cache/generations_exp23_qwen3b/44/44d8f46d2d0e43dbce73e6cd5468ddf3601c6193250ddf68948a2f9a9194a8d1.json`
(prompt + output), `cache/generations_exp23_qwen7b/90/9068ef359d076b2e01c789e0082c3c4ca3d851f0132bf9b2df100020c3b383c8.json`,
`runs/exp23_verdicts.json` (reader qwen3b, arm fixed8, fold 4), `runs/exp23_qwen3b_f4/fixed8__fraction0.05/`,
`runs/exp35/qwen3b/{A,B,N,F}/fold4.jsonl`, `runs/exp35/qwen3b/judge/fold4.jsonl` (items N, A, B, F, Fcut, Ncut, D2, cached),
`runs/exp34_jlens/qwen3b/1e043500.json`, `runs/exp34_jlens_own/qwen3b/1e043500.json`, `runs/exp34_jlens/qwen7b/1e043500.json`.

How representative it is (§35, folds 2–4, 165 marked 3B rows): 62 cached refusals; the gold string is in the model's own
note (criterion D1) on 22 of them (0.35; 0.108 of the 204-question population); under arm B, 14 of those 22 answer
correctly, 6 still refuse, 2 answer wrong. This row is one of the 14. The other 13 recovered rows: 577d4d32 ("7 pm"),
6f9b354f, ed4ddc30, gpt4_c27434e8, a82c026e, gpt4_76048e76, gpt4_d31cdae3, 0ea62687, c19f7a0b, dccbc061, gpt4_483dd43c,
gpt4_6ed717ea, gpt4_b5700ca0.

### Example B: the evidence is one line, the number is misread

| field | value |
|---|---|
| question id | `0e5e2d1a` (fold 4, type single-session-assistant) |
| question | (asked on 2023/05/30 (Tue) 18:13) I wanted to follow up on our previous conversation about binaural beats for anxiety and depression. Can you remind me how many subjects were in the study published in the journal Music and Medicine that found significant reductions in symptoms of depression, anxiety, and stress? |
| gold | 38 subjects |
| the evidence line in view (an assistant turn; the relevant sentence) | assistant, 2023/05/26 (Fri) 05:42: ... 1. In a study published in the journal Alternative Therapies in Health and Medicine, 15 subjects with anxiety and depression listened to binaural beats daily for four weeks. ... 3. Another study published in the journal Music and Medicine involved 38 subjects who listened to binaural beats for 30 minutes daily for three weeks. The study found significant reductions in symptoms of depression, anxiety, and stress. ... |
| Qwen2.5-3B output (cached §23 run, verbatim) | `18 subjects participated in the study published in the journal Music and Medicine that found significant reductions in symptoms of depression, anxiety, and stress using binaural beats.` (no note, no `Answer:` line; the whole output was graded) |
| §23 verdict | wrong, not a refusal |
| Qwen2.5-7B output on the identical prompt | `The study published in the journal Music and Medicine involved 38 subjects.` ⏎ ⏎ `Answer: 38` (correct) |
| §35 arm F (unknown clauses rewritten, regenerated) | `38` (correct) |
| §35 F-cut (same forced prompt, every evidence line removed) | `15 subjects` (wrong; with nothing in view the model produces the other study's count) |
| §34 lens, forced answer position | the gold's first token `3` is rank 1 at the final layer for the 3B as well as the 7B; the cached output never went through that position (it skipped the note and the marker) |

Artifacts: `cache/generations_exp23_qwen3b/34/34025ed3f56bba24d60bf0e3bb01678f34619b7d6bc76f73bacbf2f4a5e7988a.json`,
`cache/generations_exp23_qwen7b/cc/cc869e6aadb8aa5196907f71cb0cd1f2601644bc79c2f536cf0ce885afa90756.json`,
`runs/exp23_verdicts.json` (fold 4), `runs/exp23_qwen3b_f4/fixed8__fraction0.05/`, `runs/exp35/qwen3b/{N,F,Fcut,Ncut}/fold4.jsonl`,
`runs/exp35/qwen3b/judge/fold4.jsonl`, `runs/exp34_jlens/qwen3b/0e5e2d1a.json`, `runs/exp34_jlens_ablated/qwen3b/0e5e2d1a.json`.

Caveats to print with it: the output has no `Answer:` line, so it is outside §35's frozen-note arms (the 39 marker-less
3B rows); the misread is one digit (18 for 38); the forced-choice arm fixes this row, which says the format, not the
reading, was the obstacle here, so do not present it as "the model cannot read the number".

### Alternate for B, if a composition failure reads better: `e56a43b9` (fold 3, multi-session)

Question: "How much discount will I get on my next purchase at FreshMart?" Gold `$5`. Two evidence lines in view: "I just
reached 500 points, the minimum points required for a discount at FreshMart" (2023/05/24) and "every 100 points translate
to a $1 discount on my next purchase" (2023/05/27). 3B: `user mentioned reaching 500 points, which translates to a $1
discount on the next purchase at FreshMart. Since the user is asking about the next purchase, they would receive a $1
discount.` ⏎ `Answer: $1` (wrong). 7B: `Fact: The user has 500 points at FreshMart, and every 100 points translate to a $1
discount.` ⏎ ⏎ `Answer: $5 discount` (correct). Both facts are in the 3B's note; the step 500 / 100 × $1 is not done.
Caveat: §35's F-cut arm answered `$5 discount` with every evidence line removed (judged correct), so the gold is
guessable on this row; the in-view failure is real, the control is awkward. Files:
`cache/generations_exp23_qwen3b/3a/3a3268c429cba119b341b897af4cb2376c1efcc70ac6d9f53e1d8e1c2ff8fe14.json`,
`cache/generations_exp23_qwen7b/a9/a918dd0b130cb0d763b05e1ecc0223e17119fba7bb100f43d2ebc8ae4c192074.json`,
`runs/exp35/qwen3b/{N,A,B,F,Fcut}/fold3.jsonl`, `runs/exp34_jlens/qwen3b/e56a43b9.json`.

## 2. The equations as implemented

### 2.1 Delete-only ILP (`memctl/hindsight/exact.py:36–99`, `solve_delete_only`)

Variables: for every needed item i and its k-th need time, x_{i,k} ∈ {0,1} = "still in ACTIVE at that need time, kept
continuously since arrival"; for every query q, y_q ∈ {0,1} = "all of q's evidence present".

    maximise   Σ_q y_q − (1 / (10 · Σ_i s_i·|needs(i)|)) · Σ_{i,k} s_i · x_{i,k}
    subject to x_{i,k+1} ≤ x_{i,k}                                   (once evicted, never back)       line 56
               y_q ≤ Σ_{i ∈ carriers(req)} x_{i,pos(i,q)}  for every requirement of q               line 68
               Σ_{i present at τ} s_i · x_{i,pos(i,τ)} ≤ B  at every arrival and need step τ          lines 72–83

s_i = the item's tokens, B = the budget. Solver CBC via PuLP, 60 s limit, at most 50,000 variables; on any non-optimal
status the oracle falls back to the approximate rule and records `oracle_method: approx` (`controllers/oracle.py:53–62`).
Exact under three conditions stated in the file: a deleted item never returns, items are not rewritten, the reader is
right whenever the evidence is in ACTIVE. The research note's §4.4 statement matches the code term for term.
Used in the `oracle_exact` rows of Experiments 1, 1b–1d, 2, 3, 4 (eval), 6, 7 (`configs/sweeps/*.yaml`). The
`oracle_approx` controller is Belady's furthest-next-use rule plus just-in-time retrieval; it is not optimal with
unequal item sizes (Exp 1c measures the gap, up to 0.12). Do not call the approximate oracle "optimal".

### 2.2 Archive-aware regret expert (`memctl/rl/expert.py:21–22` docstring, `83–99` code)

For every ACTIVE item i at step t, with needed = 1[next_need(i,t) ≠ NEVER]:

    regret(i, EVICT)   = delete_cost   + needed
    regret(i, ARCHIVE) = archive_cost  + retrieval_risk · needed

The expert accepts every (item, operation) pair of least regret (ties within a regret level ordered by next need,
furthest first); retrieval label for a shortlisted archive item j: 1[next_need(j,t) = t]. For cost-sensitive imitation it
also returns cost(i, op) = regret + 0.05 · place(i), place from 0 (never or furthest needed) to 1 (needed soonest).
Settings actually used: D1 defaults (delete_cost 0, archive_cost 0, retrieval_risk 0), so EVICT costs 1 if needed again
else 0 and ARCHIVE costs 0, exactly as EXPERIMENTS.md §D1 says; D5 sets archive_cost = p ∈ {0, 0.01, 0.05, 0.2}
(`configs/rl/d5/*.yaml`); the LongMemEval floor head (Exp 15) sets retrieval_risk 0.36 (`configs/rl/lme_cv/*.yaml`);
delete_cost was added for 11d. This is a closed-form one-step regret with the oracle assumed to play on, not a replay.
The `oracle` expert (lines 68–81) is the same rule with the tie broken towards EVICT for never-needed items; that
tie-break is the 4b failure.

### 2.3 Imitation loss (`memctl/rl/algorithms.py:28–75`, `imitation_loss`)

Follow the expert's removal sequence while the decision is over budget: at each pick, with the policy's masked softmax
over all (item, operation) pairs, loss += −log Σ_{(i,op) ∈ accepted} p(i,op), where accepted = every available pair of
lowest expert rank; the first accepted pair is then removed and its savings subtracted. Retrieval head: binary
cross-entropy on the shortlist labels by default (`bce`); `listsum` and `listwise` variants (softmax mass on the needed
items) were used for the §15/§16 floor head. Cost-sensitive variant (`cost_loss`, lines 78–113): at each pick
loss += Σ_pairs p(i,op) · cost(i,op), the policy's expected regret (AggreVaTe-style). DAgger: iteration 0 expert-driven,
later iterations learner-driven with expert labels on every visited state, aggregated (cap 40,000). The research note's
§4.6 matches; it does not mention the listsum/listwise or cost variants, which were added later.

### 2.4 RL reward and returns (`memctl/rewards.py`, `memctl/rl/train.py`, `memctl/rl/algorithms.py`)

    R_t = Σ_k w_k · φ_k(ctx_t),  defaults w = {task_reward 1.0, forced_fallback −0.1, invalid_action −0.1}   rewards.py:19
    G_t = Σ_{k ≥ t} γ^{k−t} R_k,  γ = 0.995 (or 0 in the gamma0 runs)                                      train.py:70–81

Every term is logged every step whatever its weight. PPO (algorithms.py:163–210): clipped ratio (clip 0.2), learned
value baseline (coefficient 0.5), entropy bonus 0.01, advantages standardised over the batch. GRPO (train.py:87–134,
algorithms.py:213–260): 8 samples per episode, advantage = (episode return − group mean) / group std by default
(`advantage: std`; `mean`, `batch`, `best` exist), clipped objective, no value network, optional imitation term.
Two corrections to the research note (`docs/research/HINDSIGHT_MEMORY_RESEARCH_NOTE.md:139`): the 14 D5 configs do set
`archive_write: −p` in the reward, so "no RL config overrides `reward`" is no longer true; but every D5 learner was
imitation (`bc` or `cost`), so no policy was ever optimised on that priced reward; the price entered through the expert
and the priced score is computed afterwards by `memctl/analysis/priced.py`. The later GRPO experiments use their own
rewards: §16 (`rl/select_grpo.py:10–11, 206–213`) reward = judge verdict + aux_weight × token-F1, mean-only advantage;
§22 (`rl/note_grpo.py:183–203`) per-session credit variants.

### 2.5 Continuation regret: implemented, unit-tested, never run (`memctl/hindsight/regret.py:83–101`)

    regret(i, t, a) = S(π to t, oracle from t) − S(π to t, a forced at t, oracle from t)

A `SwitchController` hands over to the approximate oracle at step t; two full episodes per decision. The only caller is
`tests/test_hindsight.py:78`; `runs/`, `configs/` and `scripts/` contain no reference to it. D1's regret expert is the
closed form of 2.2, not this replay, although the research note's D1 plan (lines 827 and 869) said it would use
`continuation_regret`. The article should say "a closed-form hindsight regret label (one if the item is needed again,
else zero; archiving free)", and may mention replay regret only as implemented-and-untested. The note's line 211
("Unit-tested; not used at scale") is accurate.

## 3. Statements that would be too strong, and stale lines in the research note

- Research note line ~232, "neither was tried" (cost-aware expert, priced deletion): stale. Both were tried on 2026-10-05
  (D1 regret expert, D5 priced archive). The note is dated 2026-09-30 and last edited 2026-10-10 (f3c85a7).
- D1's "imitation now beats every fixed rule" holds at 2–10 % budgets on the synthetic recall task with a free archive and
  a scripted reader. On the LLM tasks the learned head is non-inferior, not better, than a 31-parameter logistic regression
  (§32) and no controller lifts the 3B to the 7B (§23). Keep the two settings apart.
- §35: say "in about a third of the 3B's refusals the gold string is already in its own note (22 of 62, folds 2–4), and
  forbidding 'unknown' with the note frozen recovers 14 of those 22". Do not say the 3B "always knows the answer". The
  accuracy lift is +0.176 (0.109, 0.242) on 165 marked rows, about half the matched gap to the 7B (0.53, CI 0.33–0.73),
  at the cost of a fifth of all marked refusals turning into wrong answers (0.21) and a doubled answered-and-wrong share
  when there is no evidence at all. The pre-registered prediction (+0.05 to +0.08) was wrong by two to three times.
- §34: "present" means the gold token is in the lens top-10 at some layer at the forced answer position, in about two
  thirds of the 3B's failures. About 0.64 of those stay "present" with every evidence line cut (§34c), so the word
  "present" must not become "the model has read the answer from the evidence". §44 then tried to establish the
  evidence-use vs answer-production split by patching and an evidence swap and did not (STATE F12, Q18 NOT ESTABLISHED).
- §38–§42 steering: the result is at the disclosed secondary setting only (layer 29, α 2; two settings tested); it
  transfers to LoCoMo (+0.333, replicates +0.233); the selective policy of §42 fails; the projection signal ranks
  extractability, not answerability. §43: four families, scorecard in STATE F11.
- Labels: one judge (Qwen2.5-7B-Instruct) that understates accuracy by about 0.1 on the §23 estimate; paired comparisons
  are unaffected. Two flagged question types (knowledge-update, single-session-preference) use different judge prompts;
  avoid examples from them (`71315a70`, `cf22b7bf` are knowledge-update).
- Example rows: both are from folds that were sealed during design and read once at confirmation; the experiments are
  closed, so publishing them is fine, but say they are illustrative rows, chosen by the author, from their classes.
- The two examples are Qwen2.5-3B-Instruct under one prompt family on one benchmark (LongMemEval-S); §24 and §43 extend
  parts of the picture to other families, but these rows do not.

## 4. Addendum: Codex's two development-fold rows, verified (2026-10-30, later)

- `b320f3f8` (fold 0, single-session-user, gold "a blue Snaggletooth"): cached 3B output `user mentioned buying a rare blue
  Snaggletooth action figure from a thrift store a few weeks ago.` ⏎ `Answer: unknown`; §23 verdict wrong/refusal; D1 true
  (gold in the note); 7B correct ("A rare blue Snaggletooth action figure."); §35 arm A ` unknown`, arm B ` Snaggletooth
  action figure` judged correct, arm F correct, F-cut and N-cut refuse. Cache file as Codex named it
  (`cache/generations_exp23_qwen3b/5e/5eff6547….json`). Clean own-note refusal; fine to publish.
- `4bc144e2` (fold 1, multi-session, gold "$65"): cached 3B output `user mentioned spending $15 on a car wash on February
  3rd and $50 on a parking ticket on January 5th.` ⏎ `Answer: unknown`; 7B `Answer: $65` correct; §35 arm B ` $65` judged
  correct. **Caveat: D1 is false on this row** (the gold string "$65" is not in the note, only the two amounts), so in §35's
  terms it is an S′ row, one of the "note holds the ingredients, the last step happens at the answer" recoveries described
  as exploratory in §35a, not one of the 22 S rows that Claim 1 is computed on. Also arm F (the forced prompt regenerated)
  answered `$55`, wrong, and F-cut answered `$15` with no evidence. Describe it as "the note names both amounts and the sum
  is done only when 'unknown' is forbidden", not as "the gold was in the note".
- Both rows are development-fold rows (folds 0–1), which were open throughout; no leak concern.
