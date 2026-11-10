# §56 proposal: which MLP neurons carry false "unknown" at the answer position? (Qwen 2.5 3B)

**Status: proposed, rev 0 (Advisor, 2026-11-10). Not built, not run, nothing spent.** Stages 0–3 are laptop-only and
$0, and use development data only. Stage 4 (sealed confirmation) needs a separate approval. The section number is
provisional until the Executor opens it in EXPERIMENTS.md. The decision number D-83 is proposed.

## 1. The question in one paragraph

Every fix for false abstention so far (prompt, §38 push, §51 rank-1 edit, removing the option) behaves like a dial.
§53 found that a plain logit bias against "unknown" matches the best weight edit. None of these fixes cuts false
"unknown"s without also cutting true ones. Goodfire's "Neurons as an Interpretable Basis" (Tigges et al., 2026-11-10)
argues that MLP neurons are already sparse, mostly single-purpose units. This study asks three questions about
Qwen 2.5 3B at the answer position:
1. Is the push toward "unknown" carried by a small set of MLP neurons?
2. Are they the same neurons for false and true abstentions (a shared output channel), or partly different?
3. Does switching that set off release false abstentions more selectively than the logit bias does?

## 2. What we already know

| Source | Fact | What it implies here |
|---|---|---|
| §34 / F11 | On 64% of the 3B's failures the gold is in the final-layer top 10, and "unknown" wins in the last ~10% of layers | Late layers are where to look |
| §39 / §39b | One push at the ":" token (P_ans) is enough | A single position carries the decision |
| §51 / F18 | Pathway located in attention: L32h2 + L33h10 → L35 attention output. Sufficient, not necessary | MLP neurons have never been examined. The L35 MLP gave back part of the edit's effect (compensation) |
| §53 / F19 | Logit bias B1 (−8) ties both edits (U1 0.406 vs 0.403 / 0.414) | The bar a "selective" fix must beat |
| §54 / F20 | 89% of false abstentions follow a note holding the gold; twin flips are mostly note-driven | Part of the cause is the note text, which no neuron-level fix will reach. This study looks only at the answer step |
| §55 / F22 | Removing the option: FA 0.311 → 0, Uns 0.030 → 0.653 | The trade-off a fix is judged on |
| Outside | Abstention inflation (arXiv 2507.16199): the answer survives mid-layer and late layers override it. Commit-Abstain Circuit (2609.32964) on other families. Confidence-regulation neurons (Stolfo et al., 2024) act through the final norm, not the unembedding | Novelty must be checked before any claim. Direct logit attribution alone misses norm-mediated neurons (§8) |

## 3. Definitions

- **Model, prompt, decoding:** Qwen 2.5 3B Instruct, fp32, laptop CPU, the frozen reader prompt P0 (7b5fc30), greedy
  decoding. P0 is kept for comparability with §34–§55.
- **P_ans:** the final ":" of the "Answer:" marker (§50, §51, §54). Each row's own P0 note is forced, followed by the
  marker, so one forward pass reaches P_ans.
- **Neuron:** one SwiGLU hidden unit, a_i = silu(gate_i(x)) · up_i(x) at P_ans, with write vector w_i = column i of
  `down_proj`. There are 36 layers × 11,008 = 396,288 neurons. Both signs count: a negative a_i writes −w_i.
- **Decision score D:** logit("unknown" first token) − logit(t*) at P_ans, where t* is the top non-"unknown" token
  of the unmodified model on that row. The same rule applies to every class, so no gold is needed. For ANS rows,
  agreement of t* with the gold's first token is reported.
- **Direct push of a neuron (DLA):** a_i · ⟨w_i ⊙ g / σ, u_unk − u_t*⟩, where σ is the row's final-RMSNorm scale at
  P_ans, g is the norm weight, and u are unembedding rows (tied embeddings). Positive means it pushes "unknown".
- **Classes** (from the model's own P0 generation, §51 grader, head + tail):
  - **FA:** answerable, answered "unknown";
  - **C:** answerable, answered gold;
  - **TA:** unanswerable (UNA-rel, UNA-swap), answered "unknown";
  - **UA:** unanswerable, answered. UA is descriptive only, since it is expected to be about 3% of rows.

## 4. Data

- **W, controlled (§54 split, seed 55, development).** 150 bases, 1,950 rows. CF rows are not used.
  - **Dev-A** = even `base_index`, 75 bases: ANS from §54 B0, UNA from §55 P0 (both already generated).
  - **Dev-B** = odd `base_index`, 75 bases: ANS from §54 B0. UNA under P0 is new free generation (§51 engine, same
    settings), about 300 rows.
  - Expected counts: about 144 FA and about 290 TA per half (F20: 288 FA over both halves; F22: Uns 0.030).
- **No held-out split is read.** Seed 56 stays unused. Stage 4 would use a new sealed split, seed 57, generated only
  after the stage-3 freeze.

## 5. Stages

### Stage 0: freeze ($0)

The Executor commits the following before any neuron is measured:
- the row lists and class labels for dev-A and dev-B, with shas;
- the t* rule;
- the DLA code, tested on two rows: the summed neuron DLA of each layer must equal that layer's MLP DLA to 1e-4.

The Executor also times one real row: one forward pass storing layers 20–35, and one ablated generation. The schedule
below is replaced by the measured times.

### Stage 1: rank the neurons (dev-A only)

1. One forward pass per dev-A row (FA, C, TA, UA). Store a_i and DLA_i at P_ans for all layers. If storage forces a
   restriction, keep layers 20–35 and record the restriction.
2. **Concentration (R1).** On FA rows, take each neuron's mean DLA and sum the positive means; that sum is the total
   positive push. K50 is the smallest number of top-ranked neurons whose summed mean DLA reaches half of it.
3. **Candidate set K.** The top min(K50, 50) neurons by mean FA DLA, frozen by commit at the end of stage 1.
4. **Channel type (R2).** For set K, the ratio of its summed mean DLA on TA rows to that on FA rows.
5. Descriptive outputs:
   - the same quantities on C rows;
   - the layer histogram of K;
   - overlap of K's write vectors with the §51 L35 write vector and the §38 direction (cosine);
   - K50 computed on TA rows.

### Stage 2: causal test and selectivity (dev-B only)

1. **Ablation.** At P_ans only, set every neuron in K to its dev-A mean activation on C rows. Then generate the
   answer greedily and grade it with the §51 grader. The ablation applies only at P_ans.
2. **Random controls.** 20 random neuron sets matched to K in size and per-layer counts, ablated the same way.
3. **Logit bias B1.** A constant bias −b on the "unknown" first token at P_ans. Calibrate b on dev-A so that its FA
   release equals the K ablation's dev-A release. b is frozen before any dev-B read.
4. Measures on dev-B:
   - **Rel:** FA rows answered gold;
   - **Uns:** TA rows now given an answer;
   - **Dam:** C rows now wrong;
   - **U(λ = 1)**, as in §51.

### Stage 3: one step upstream, and what K responds to (dev only, descriptive)

This stage runs only if R3 is SHOWN.
- **Upstream.** For the top 10 neurons of K, take the direct contribution of each upstream component at P_ans to the
  neuron's gate and up pre-activations: the 16 heads per layer (L32h2 and L33h10 always named) and earlier MLP outputs.
- **Descriptions** (Goodfire-style, small). For each of the top 10, the Advisor writes a one-line description from:
  - the 20 highest- and 20 lowest-activating dev rows;
  - 200 LongMemEval dev prompts at P_ans.

  A synthesis check follows. The Advisor writes 10 inputs predicted to activate the neuron and 10 matched controls
  (same template, one changed element). Pass if the prediction beats its control on ≥ 8/10 pairs.
- Descriptions and checks are written before the Executor runs the inputs. The Advisor sees only activations, never
  generations.

### Stage 4: sealed confirmation (separate approval)

R1–R4 are re-read on a new seed-57 split with K, the mean-ablation values and b frozen. Laptop if it fits; otherwise
one pod quoted in advance.

## 6. Readings, fixed now

| Reading | Statistic | Verdict |
|---|---|---|
| **R1 concentration** (stage 1) | K50 on dev-A FA rows | **CONCENTRATED** if K50 ≤ 50; **DISTRIBUTED** if K50 > 500; **PARTLY CONCENTRATED** otherwise. Bootstrap over rows (clustered by base, 2,000 draws, seed 56) gives an interval for K50; the verdict uses the point, and the interval is reported |
| **R2 channel type** (stage 1) | ratio = Σ_K mean DLA on TA ÷ Σ_K mean DLA on FA | **SHARED CHANNEL** if the 95% interval is wholly ≥ 0.8; **FA-LEANING** if wholly ≤ 0.5; **MIXED** otherwise |
| **R3 causal** (stage 2, dev-B) | Rel of the K ablation vs the 20 random sets | **SHOWN** if K ranks 1 of 21 and Rel(K) ≥ 0.20; **NOT SHOWN** otherwise |
| **R4 selectivity vs the dial** (stage 2, dev-B) | Uns(K) − Uns(B1) at matched dev-A release; 95% interval clustered by base | **BETTER THAN THE DIAL** if wholly < 0; **WORSE** if wholly > 0; **NO DIFFERENCE SHOWN** otherwise. Read only if R3 is SHOWN. Rel(K) − Rel(B1) on dev-B and Dam are reported beside it, because dev-A calibration may not transfer |

## 7. Stopping rules (decided now)

| Where | Result | What happens |
|---|---|---|
| Stage 0 | The DLA sum check fails, or one row takes more than 10 min | Fix or report. No stage 1 until it passes |
| Stage 1 | R1 = DISTRIBUTED | Report K50, its interval, and the layer profile: "at neuron resolution the push is distributed." Stop (D-45) |
| Stage 2 | R3 = NOT SHOWN | Report "the direct-push ranking does not find causal units." Stop. K is not re-chosen on dev-B |
| Stage 2 | R3 SHOWN, R4 read | Run stage 3 regardless of R4. Then propose stage 4 to the user |
| Stage 4 | Fails | Report under D-45. The dev result stays exploratory |

## 8. What each outcome would mean

- **Concentrated + shared channel + R3 shown + no difference from the dial.** This is the most likely outcome. We
  have found the neurons that write "unknown", and they are the dial. The cause sits upstream, and stage 3 says what
  drives them. It is still a step beyond F18/F19: the dial gets a physical location in the MLPs, beside the attention
  pathway.
- **Concentrated + FA-leaning + better than the dial.** The first selective fix in the programme. Stage 4 becomes the
  priority.
- **Distributed.** Goodfire's picture does not carry over to this small dense model for this behaviour. That is a
  clean negative, and it supports the threshold account: no small set of units decides.
- **R3 not shown.** The units that write "unknown" directly are not where the decision is made. Neurons that act
  through the norm (Stolfo et al.) or indirect paths would be next; that needs a new proposal.

## 9. Threats and how the design handles them

| Threat | Handling |
|---|---|
| **DLA misses neurons acting through the final norm or later layers** (Stolfo et al. 2024) | R3 tests the set causally, with downstream recomputation. As a descriptive check, the 20 top neurons by \|mean-ablation effect on D\| are measured on 30 dev-A rows and compared with K |
| **Mean-ablation to C values is off-distribution** | It uses natural class means, not zeros. Zero-ablation is reported as a sensitivity check |
| **The same rows choose and test K** | Dev-A chooses; dev-B tests. b is calibrated on dev-A |
| **Small dense models are more polysemantic** than Goodfire's 35B MoE | R1 measures this directly and has a DISTRIBUTED exit |
| **Compensation by L35 MLP or attention** (§51) | Downstream components recompute in R3. The direct-only effect is reported beside it |
| **The note is part of the cause** (F20) | Claims are limited to the answer step under the model's own note, stated in every reading |
| **Synthetic templates only** | The LongMemEval dev prompts in stage 3 are descriptive. No claim beyond W until stage 4 or a natural-data replication |
| **Novelty** | A short literature check on neuron-level abstention (Commit-Abstain Circuit, abstention inflation, confidence neurons) before any claim is written |
| **Laptop memory** (§53 swapped at 15–19 GB of 23 GB) | Store activations for layers 20–35 by default (176k floats per row), streamed to disk. Time one row first |

## 10. Cost and roles

- **Stages 0–3:** $0, laptop, development data only. Built and run by the Executor; the Advisor reviews at each
  gate. Stage 3 descriptions are written by the Advisor in this session, with no API spend.
- **Stage 4:** a separate decision. Laptop if it fits, otherwise a quoted pod. About $1.66 remains under the Berkeley
  $36 cap; the gmail account is reserved for the self-doubt project.
- **No change to STATE or DECISIONS** until results are reviewed and the user says "Go".

## 11. Rough schedule (to be replaced by measured times)

| Step | Estimate |
|---|---|
| Stage 0 | Same day |
| Dev-B UNA generation (~300 rows) | ~1.5 h (§54 B0 rate ≈ 18 s/row) |
| Stage 1 (~750 dev-A rows, one forward pass each) | Hours; timed first |
| Stage 2 (21 sets + B1 on ~900 dev-B rows, one short generation each) | Possibly over a day; if over 24 h, use 10 random sets and record it before running |
| Stage 3 | Hours |

## 12. References

- Tigges, C., Kowal, M., McGrath, T., Lewis, O. and Cammarata, N. (2026). Neurons as an interpretable basis. Goodfire
  Research.
- Stolfo, A., Wu, B., Gurnee, W., Belinkov, Y., Song, X., Sachan, M. and Nanda, N. (2024). Confidence regulation
  neurons in language models. NeurIPS.
- Geva, M., Caciularu, A., Wang, K. and Goldberg, Y. (2022). Transformer feed-forward layers build predictions by
  promoting concepts in the vocabulary space. EMNLP.
- Gurnee, W. et al. (2024). Universal neurons in GPT2 language models. TMLR.
- Ferrando, J., Obeso, O., Rajamanoharan, S. and Nanda, N. (2025). Do I know this entity? ICLR.
- Abstention inflation, arXiv:2507.16199; Commit-Abstain Circuit, arXiv:2609.32964 (as cited in §55 and the
  2026-11-01 audit).

## 13. Decision requested

Approve stages 0–3 ($0, development data only), approve with changes, or decline. Stage 4 would come back as a
separate decision.
