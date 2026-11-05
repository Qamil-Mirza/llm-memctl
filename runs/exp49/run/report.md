# §49 legible abstentions: held read (computed)

Design sha256 `01911c8688845baa154ea2829961f4bcf4487b63e7c3a3f540e63be4892fdd22`; α 0.5, ℓ* P_pre 35, P_ans 33, F1 35, suff 24; τ_c 0.9833313539420416, τ_s 0.7361310920112776.

Held false abstentions 165, genuine 144.

Centroids: {"pooled_fallback_at_P_ans": "no-op: F1 has no Answer: site, so the B3 pooled F1 + F2 fallback adds no trial at P_ans", "unestimable_classes_P_ans": [], "unestimable_classes_F1": [], "label_counts_by_part": {"fit": {"Brighton": 10, "Cambridge": 17, "Cardiff": 7, "Chester": 13, "Derby": 15, "Durham": 11, "Edinburgh": 12, "Leeds": 22, "Liverpool": 18, "Manchester": 19, "Norwich": 6, "York": 18}, "select": {"Brighton": 22, "Cambridge": 15, "Cardiff": 32, "Chester": 11, "Derby": 10, "Durham": 17, "Edinburgh": 15, "Leeds": 13, "Liverpool": 10, "Manchester": 9, "Norwich": 11, "York": 3}, "tune": {"Brighton": 10, "Cambridge": 10, "Cardiff": 3, "Chester": 18, "Derby": 17, "Durham": 14, "Edinburgh": 15, "Leeds": 7, "Liverpool": 14, "Manchester": 14, "Norwich": 25, "York": 21}}}.

No-op replays (held): 20/20. Exclusions (no P_ans): {"dev|ANS|degenerate|no Answer: marker": 4, "held|ANS|degenerate|no Answer: marker": 2}.

## Q-P (primary)
- gold R1 − R3 on false abstentions: +0.836 [+0.774, +0.894] (n 165, bases 90)
- unsupported R1 − R3 on genuine abstentions: +1.000 [+1.000, +1.000] (n 144, bases 144)
- reading: direction-specific recovery (R1 against random) on false abstentions; unsupported-answer change on genuine abstentions, descriptive: +1.000 [+1.000, +1.000]

## Q-2
- gold_R1_minus_R2_false: -0.024 [-0.069, +0.013] (n 165, bases 90)
- gold_R1_minus_R4_false: -0.097 [-0.149, -0.052] (n 165, bases 90)
- gold_R2_minus_R3_false: +0.861 [+0.806, +0.911] (n 165, bases 90)
- gold_R4_minus_R3_false: +0.933 [+0.890, +0.970] (n 165, bases 90)
- unsupported_R1_minus_R2_genuine: +0.812 [+0.750, +0.875] (n 144, bases 144)
- unsupported_R1_minus_R4_genuine: +0.549 [+0.465, +0.632] (n 144, bases 144)
- unsupported_R2_minus_R3_genuine: +0.188 [+0.125, +0.250] (n 144, bases 144)
- unsupported_R4_minus_R3_genuine: +0.451 [+0.368, +0.535] (n 144, bases 144)
- R4_damage_on_correct: +0.011 [+0.000, +0.026] (n 265, bases 122)
- Dgold_gold_false: +1.000 [+1.000, +1.000] (n 165, bases 90)

## Q-3 tracking (H1; own denominators; R1 and R2 not compared)
- R1: bases 51, released 51, CF city 50, ANS gold 48; tracking among released +0.980 [+0.941, +1.000] (n 51, bases 51)
- R2 (one-position): bases 51, released 51, CF city 50, ANS gold 50; tracking among released +0.980 [+0.941, +1.000] (n 51, bases 51)

## Q-4 gates (secondary; H1 abstentions; templates only)
- λ 1: U never +0.000, always_R1 -0.470, G_content +0.125, G_suff +0.235
  - G_suff_minus_never: +0.235 [+0.179, +0.283] (n 200, bases 144)
  - G_content_minus_never: +0.125 [+0.080, +0.169] (n 200, bases 144)
  - G_suff_minus_always_R1: +0.705 [+0.662, +0.751] (n 200, bases 144)
  - G_content_minus_always_R1: +0.595 [+0.524, +0.672] (n 200, bases 144)
  - always_R1_minus_never: -0.470 [-0.560, -0.390] (n 200, bases 144)
- λ 2: U never +0.000, always_R1 -1.205, G_content +0.110, G_suff +0.215
  - G_suff_minus_never: +0.215 [+0.144, +0.275] (n 200, bases 144)
  - G_content_minus_never: +0.110 [+0.056, +0.162] (n 200, bases 144)
  - G_suff_minus_always_R1: +1.420 [+1.340, +1.511] (n 200, bases 144)
  - G_content_minus_always_R1: +1.315 [+1.202, +1.437] (n 200, bases 144)
  - always_R1_minus_never: -1.205 [-1.340, -1.085] (n 200, bases 144)

## Q-5 F1 replication (labelled)
- n_f1_wrong: 46
- n_f1_gold: 26
- n_f1_wrong_degenerate: 24
- note: F1-wrong = not gold (B3: no-city trials stay in, unrepaired); the city-wrong-only population (LF's firing population) is reported beside it
- LF_minus_random_gold_on_city_wrong: +0.091 [-0.136, +0.318] (n 22, bases 22)
- n_f1_city_wrong: 22
- fired_wrong: 18
- fired_gold: 25
- LF_minus_random_gold_on_wrong: +0.043 [-0.065, +0.152] (n 46, bases 46)
- LF_gold_on_wrong: +0.152 [+0.065, +0.261] (n 46, bases 46)
- kept_correct_on_gold: +0.962 [+0.885, +1.000] (n 26, bases 26)

## Dev α components (tune H1)
- objective tied at every α -> the smaller α (tie rule, B4)
- α 0.5: {"n_false": 19, "gold_false": 0.8947368421052632, "abstain_false": 0.0, "n_genuine": 47, "unsupported_genuine": 1.0, "abstain_genuine": 0.0}
- α 1.0: {"n_false": 19, "gold_false": 0.8947368421052632, "abstain_false": 0.0, "n_genuine": 47, "unsupported_genuine": 1.0, "abstain_genuine": 0.0}
- α 2.0: {"n_false": 19, "gold_false": 0.8947368421052632, "abstain_false": 0.0, "n_genuine": 47, "unsupported_genuine": 1.0, "abstain_genuine": 0.0}

## Q-6 (descriptive)
- contrast not interpretable: calibration did not reproduce
- difference of repair gains: +0.793 [+0.671, +0.911] (n ?, bases ?)

## M1 decoding (diagnostics)
- P_pre|F2-abstain: acc +0.127 [+0.071, +0.193] (n 165, bases 90); − candidate -0.293 [-0.351, -0.227] (n 165, bases 90)
- P_pre|F2-gold: acc +0.079 [+0.047, +0.116] (n 265, bases 122); − candidate -0.351 [-0.393, -0.309] (n 265, bases 122)
- P_pre|F1-wrong: acc +0.136 [+0.000, +0.273] (n 22, bases 22); − candidate -0.154 [-0.286, +0.002] (n 22, bases 22)
- P_pre|F1-gold: acc +0.038 [+0.000, +0.115] (n 26, bases 26); − candidate -0.290 [-0.349, -0.212] (n 26, bases 26)
- P_pre_F1|F1-wrong: acc +0.136 [+0.000, +0.273] (n 22, bases 22); − candidate -0.154 [-0.286, +0.002] (n 22, bases 22)
- P_pre_F1|F1-gold: acc +0.038 [+0.000, +0.115] (n 26, bases 26); − candidate -0.290 [-0.349, -0.212] (n 26, bases 26)
- P_ans|F2-abstain: acc +0.903 [+0.851, +0.948] (n 165, bases 90); − candidate +0.483 [+0.420, +0.544] (n 165, bases 90)
- P_ans|F2-abstain|note_gold=True: acc +0.961 [+0.923, +0.993] (n 152, bases 84); − candidate +0.539 [+0.486, +0.593] (n 152, bases 84)
- P_ans|F2-abstain|note_gold=False: acc +0.231 [+0.000, +0.462] (n 13, bases 13); − candidate -0.170 [-0.534, +0.203] (n 13, bases 13)
- P_ans|F2-gold: acc +0.992 [+0.981, +1.000] (n 265, bases 122); − candidate +0.562 [+0.527, +0.594] (n 265, bases 122)
- P_ans|F2-gold|note_gold=True: acc +0.992 [+0.981, +1.000] (n 265, bases 122); − candidate +0.562 [+0.527, +0.594] (n 265, bases 122)

## Per-arm rates
```
{
 "false|R1": {
  "gold": {
   "n": 165,
   "n_bases": 90,
   "est": 0.9030303030303031,
   "lo": 0.8509246188594015,
   "hi": 0.9483870967741935
  },
  "abstain": 0.0,
  "competitor": 0.0,
  "wrong": 0.09696969696969697,
  "degenerate": 0.0,
  "status": {
   "ok": 165
  },
  "n": 165,
  "unsupported": {
   "n": 165,
   "n_bases": 90,
   "est": 0.09696969696969697,
   "lo": 0.05161290322580645,
   "hi": 0.14907538114059843
  },
  "degenerate_rate": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "false|R2": {
  "gold": {
   "n": 165,
   "n_bases": 90,
   "est": 0.9272727272727272,
   "lo": 0.8867924528301887,
   "hi": 0.963855421686747
  },
  "abstain": 0.05454545454545454,
  "competitor": 0.006060606060606061,
  "wrong": 0.006060606060606061,
  "degenerate": 0.006060606060606061,
  "status": {
   "ok": 165
  },
  "n": 165,
  "unsupported": {
   "n": 165,
   "n_bases": 90,
   "est": 0.012121212121212121,
   "lo": 0.0,
   "hi": 0.03125
  },
  "degenerate_rate": {
   "n": 165,
   "n_bases": 90,
   "est": 0.006060606060606061,
   "lo": 0.0,
   "hi": 0.0189873417721519
  }
 },
 "false|R3": {
  "gold": {
   "n": 165,
   "n_bases": 90,
   "est": 0.06666666666666667,
   "lo": 0.029940119760479042,
   "hi": 0.10975609756097561
  },
  "abstain": 0.9333333333333333,
  "competitor": 0.0,
  "wrong": 0.0,
  "degenerate": 0.0,
  "status": {
   "ok": 165
  },
  "n": 165,
  "unsupported": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "degenerate_rate": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "false|R4": {
  "gold": {
   "n": 165,
   "n_bases": 90,
   "est": 1.0,
   "lo": 1.0,
   "hi": 1.0
  },
  "abstain": 0.0,
  "competitor": 0.0,
  "wrong": 0.0,
  "degenerate": 0.0,
  "status": {
   "ok": 165
  },
  "n": 165,
  "unsupported": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "degenerate_rate": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "false|Dgold": {
  "gold": {
   "n": 165,
   "n_bases": 90,
   "est": 1.0,
   "lo": 1.0,
   "hi": 1.0
  },
  "abstain": 0.0,
  "competitor": 0.0,
  "wrong": 0.0,
  "degenerate": 0.0,
  "status": {
   "ok": 165
  },
  "n": 165,
  "unsupported": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "degenerate_rate": {
   "n": 165,
   "n_bases": 90,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "genuine|R1": {
  "gold": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "abstain": 0.0,
  "competitor": 0.0,
  "wrong": 1.0,
  "degenerate": 0.0,
  "status": {
   "ok": 144
  },
  "n": 144,
  "unsupported": {
   "n": 144,
   "n_bases": 144,
   "est": 1.0,
   "lo": 1.0,
   "hi": 1.0
  },
  "degenerate_rate": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "genuine|R2": {
  "gold": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "abstain": 0.6527777777777778,
  "competitor": 0.0,
  "wrong": 0.1875,
  "degenerate": 0.1597222222222222,
  "status": {
   "ok": 144
  },
  "n": 144,
  "unsupported": {
   "n": 144,
   "n_bases": 144,
   "est": 0.1875,
   "lo": 0.125,
   "hi": 0.25
  },
  "degenerate_rate": {
   "n": 144,
   "n_bases": 144,
   "est": 0.1597222222222222,
   "lo": 0.10416666666666667,
   "hi": 0.2222222222222222
  }
 },
 "genuine|R3": {
  "gold": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "abstain": 1.0,
  "competitor": 0.0,
  "wrong": 0.0,
  "degenerate": 0.0,
  "status": {
   "ok": 144
  },
  "n": 144,
  "unsupported": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "degenerate_rate": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  }
 },
 "genuine|R4": {
  "gold": {
   "n": 144,
   "n_bases": 144,
   "est": 0.0,
   "lo": 0.0,
   "hi": 0.0
  },
  "abstain": 0.04861111111111111,
  "competitor": 0.0,
  "wrong": 0.4513888888888889,
  "degenerate": 0.5,
  "status": {
   "ok": 144
  },
  "n": 144,
  "unsupported": {
   "n": 144,
   "n_bases": 144,
   "est": 0.4513888888888889,
   "lo": 0.3680555555555556,
   "hi": 0.5347222222222222
  },
  "degenerate_rate": {
   "n": 144,
   "n_bases": 144,
   "est": 0.5,
   "lo": 0.4166666666666667,
   "hi": 0.5833333333333334
  }
 }
}
```
